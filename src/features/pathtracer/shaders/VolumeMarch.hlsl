// air-shader-target: cs_6_6
// The real-time volumes' ray march (docs/architecture.md §8.13): after the trace has written this
// frame's samples and the distance to their first hit (PathTracerVolumeShadows.hlsl), every pixel
// marches its camera ray through the volumes in front of that hit (VolumeMarch.hlsli), lays their light
// over the surface's and adds the result to the accumulation.
// A thread group covers 8 x 8 pixels (its threads in Morton order, so that 4 consecutive lanes are a
// 2 x 2 block and 16 a 4 x 4 one): it finds the volumes some of its rays cross, of which its rays march
// the first kMarchMaxVolumes, and keeps them in groupshared memory. With kMarchAdaptive a wave whose
// pixels are deep in smoke, or close enough that a voxel covers several of them, lights fewer steps
// (MarchRayShared).

#include "air/renderer/shader/SceneShared.h"
#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<VolumeMarchConstants> g : register(b0);

SamplerState gLinearClamp : register(s0);

// --- Hooks ---------------------------------------------------------------------------------------

uint VolumeIndirectionLoad(VolumeBrickLevel l, uint index)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.indirection)];
    return buffer[index];
}

uint VolumeDistanceLoad(VolumeBrickLevel l, uint index)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.distance)];
    return buffer[index];
}

uint VolumeBrickWordLoad(VolumeBrickLevel l, uint index)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.bricks)];
    return buffer[index];
}

uint VolumeRangeLoad(VolumeBrickLevel l, uint brick)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.ranges)];
    return buffer[brick];
}

VolumeMarchData LoadMarchVolume(uint index)
{
    StructuredBuffer<VolumeMarchData> volumes = ResourceDescriptorHeap[g.volumes];
    return volumes[index];
}

VolumeBrickLevel LoadVolumeLevel(uint index)
{
    StructuredBuffer<VolumeBrickLevel> levels = ResourceDescriptorHeap[g.levels];
    return levels[index];
}

// The tile's volumes and their level g.level, read once by the thread group (Main): a step reads
// these instead of the tables.
groupshared VolumeMarchData gTileVolume[kMarchMaxVolumes];
groupshared VolumeBrickLevel gTileLevel[kMarchMaxVolumes];

VolumeMarchData LoadRayVolume(uint volume, uint slot)
{
    return gTileVolume[slot];
}

VolumeBrickLevel LoadRayLevel(uint volume, uint slot, uint level)
{
    return gTileLevel[slot];
}

float VolumeAoSample(VolumeMarchData v, float3 uvw)
{
    Texture3D<float> ao = ResourceDescriptorHeap[NonUniformResourceIndex(v.ao)];
    return ao.SampleLevel(gLinearClamp, uvw, 0.0f);
}

float4 ProbeFieldSample(float3 uvw)
{
    Texture3D<float4> field = ResourceDescriptorHeap[g.probes];
    return field.SampleLevel(gLinearClamp, uvw, 0.0f);
}

LightTriangle LoadLight(uint index)
{
    StructuredBuffer<LightTriangle> lights = ResourceDescriptorHeap[g.lights];
    return lights[index];
}

uint LoadMarchLight(uint i)
{
    StructuredBuffer<uint> lights = ResourceDescriptorHeap[g.marchLights];
    return lights[i];
}

float3 SkyTableLoad(uint texel)
{
    return float3(0.0f, 0.0f, 0.0f); // the sky's radiance is not read here
}

float4 AerialInscatterSample(SkyData sky, float3 uvw)
{
    Texture3D<float4> table = ResourceDescriptorHeap[sky.aerialInscatter];
    return table.SampleLevel(gLinearClamp, uvw, 0.0f);
}

float4 AerialTransmittanceSample(SkyData sky, float3 uvw)
{
    Texture3D<float4> table = ResourceDescriptorHeap[sky.aerialTransmittance];
    return table.SampleLevel(gLinearClamp, uvw, 0.0f);
}

#include "VolumeShadowHooks.hlsli"

LightGroup LoadLightGroup(uint group)
{
    StructuredBuffer<LightGroup> groups = ResourceDescriptorHeap[gVolumeShadows.lightGroups];
    return groups[group];
}

#include "Camera.hlsli"
#include "Sampling.hlsli"
#include "air/renderer/shader/SkyLookup.hlsli"
#include "air/renderer/shader/VolumeBricks.hlsli"
#include "VolumeProbes.hlsli"
#include "VolumeMarch.hlsli"

// --- Adaptive sampling groups --------------------------------------------------------------------

// MarchRay with adaptive sampling groups (Mueller, SIGGRAPH 2026): lighting a step costs far more than
// reading its density, so where the pixels of a 2 x 2 (4 x 4) block would light their steps alike they
// light them together: for a block of 4 (16) steps every pixel lights one of them, at its own ray's
// depth for that step, and each pixel's step takes the light of the block's pixel that lit it (which
// pixel lights which step turns with a hash of the block, its first step and `seed`, so that a step's
// light comes from every side alike: else the front of the smoke, which shows most, would always take
// its light from one side, a bias wherever the light changes across the image). A wave
// shares once its rays' transmittance has fallen below shareBelow2 (shareBelow4), where the light of a
// step matters less (Mueller's criterion), and wherever a voxel at the steps' depth covers
// shareFootprint2 (shareFootprint4) pixels or more, where the block's pixels lie within about a voxel
// of each other, closer than the light changes (volumes close to the camera, whose pixels are many).
// The wave marches one grid of steps, from the nearest entry of its rays at the smallest of their
// steps, so that a block's steps lie at the same depths on all its rays, and skips emptiness where all
// its rays are empty. Densities, depths and transmittances stay every pixel's own. All lanes run every
// iteration (finished ones still light their samples); `share` is the group of the wave's last block.
MarchResult MarchRayShared(MarchSetup s, MarchRayVolumes rv, float3 o, float3 d, float tSurface, float jitter,
                           float pixelAngle, uint seed, AIR_OUT(uint) share)
{
    MarchResult result = NoMarch();
    float tStart = 1e30f;
    float tEnd = 0.0f;
    float step = 1.0f;
    if (rv.count > 0u)
    {
        MarchSpan(s, rv, tSurface, tStart, tEnd, step);
    }
    bool live = rv.count > 0u && tStart < tEnd;
    bool ambient = (s.flags & kMarchNoIndirect) == 0u;
    float origin = WaveActiveMin(live ? tStart : 1e30f);
    float stride = WaveActiveMin(live ? step : 1e30f);
    float voxel = stride / s.stepScale;
    uint lane = WaveGetLaneIndex();
    float weightedDepth = 0.0f;
    float weight = 0.0f;
    share = 1u;
    uint dark = 1u; // the group the rays' transmittance allows; it only grows
    uint i = 0u;
    uint blockStart = 0u;
    uint blockEnd = 0u;
    uint turn = 0u;       // lane q of the block lights its step (q + turn) mod share
    bool pending = false; // the block's light is still to be sampled
    float3 light = float3(0.0f, 0.0f, 0.0f); // this lane's sample of its block, per unit of scattering
    uint lit = 0u;
    for (uint n = 0u; n < kMarchMaxSteps && WaveActiveAnyTrue(live); ++n)
    {
        float c0 = origin + float(i) * stride;
        if (i >= blockEnd)
        {
            // The group of the block of steps from i.
            float transmittance = WaveActiveMax(live ? result.transmittance : 0.0f);
            if (dark == 1u && transmittance < g.shareBelow2)
            {
                dark = 4u;
            }
            if (dark == 4u && transmittance < g.shareBelow4)
            {
                dark = 16u;
            }
            float pixels = voxel / max(c0 * pixelAngle, 1e-20f);
            share = max(dark, pixels >= g.shareFootprint4 ? 16u : (pixels >= g.shareFootprint2 ? 4u : 1u));
            blockStart = i;
            blockEnd = i + share;
            pending = share > 1u;
        }

        // This lane's part of step i (where its volumes are still ahead, it is empty up to them).
        float a = max(c0, tStart);
        float b = min(c0 + stride, tEnd);
        bool here = live && a < b;
        float t = a + jitter * (b - a);
        float3 x = o + d * t;
        MarchPoint p = NoPoint(live && !here && c0 < tStart ? tStart : 1e30f);
        if (here)
        {
            p = MarchPointAt(s, rv, x, d, t, tEnd, share == 1u && ambient);
        }
        bool scatters = here && p.sigmaT > 0.0f;

        // The block's light, once a step of it scatters: every pixel lights step i + q at its own ray's depth.
        if (pending && WaveActiveAnyTrue(scatters))
        {
            pending = false;
            blockStart = i;
            blockEnd = i + share;
            uint base = lane & ~(share - 1u);
            turn = Pcg(seed + Pcg(base + Pcg(i))) & (share - 1u);
            uint q = (lane + turn) & (share - 1u);
            float qa = max(origin + float(i + q) * stride, tStart);
            float qb = min(origin + float(i + q + 1u) * stride, tEnd);
            lit = 0u;
            if (rv.count > 0u && qa < qb)
            {
                float tq = qa + jitter * (qb - qa);
                bool valid;
                light = MarchLightAt(s, rv, o + d * tq, d, tq, ambient, valid);
                lit = valid ? 1u : 0u;
            }
        }
        float3 borrowed = float3(0.0f, 0.0f, 0.0f);
        uint borrowedLit = 0u;
        if (share > 1u && !pending)
        {
            uint from = (lane & ~(share - 1u)) + ((i - blockStart - turn) & (share - 1u));
            borrowed = WaveReadLaneAt(light, from);
            borrowedLit = WaveReadLaneAt(lit, from);
        }

        if (scatters)
        {
            float scattering = p.sigmaS.x + p.sigmaS.y + p.sigmaS.z + p.sigmaS.w;
            float3 scattered;
            if (share == 1u)
            {
                scattered = MarchDirect(s, rv, p, x, d) + p.around;
            }
            else if (borrowedLit != 0u)
            {
                scattered = borrowed * scattering;
            }
            else
            {
                bool valid;
                scattered = MarchLightAt(s, rv, x, d, t, ambient, valid) * scattering;
            }
            MarchIntegrate(result, weightedDepth, weight, scattered, p.sigmaT, b - a, t);
        }

        // Where no ray of the wave holds a volume, on to the step where the nearest emptiness ends (the
        // block's samples lie at their steps still).
        uint next = i + 1u;
        if (!WaveActiveAnyTrue(here && p.occupied))
        {
            float reach = WaveActiveMin(live ? p.skipTo : 1e30f);
            if (reach < 1e29f)
            {
                next = max(next, uint(max(floor((reach - origin) / stride), 0.0f)));
            }
        }
        i = next;
        live = live && origin + float(i) * stride < tEnd && result.transmittance >= kMarchOpaque;
    }
    result.depth = weight > 0.0f ? weightedDepth / weight : tStart;
    return result;
}

// --- Pass ----------------------------------------------------------------------------------------

// A denoiser that reprojects its history (DLSS Ray Reconstruction) follows the guides. Where the volumes
// hide most of the surface, theirs describe the volumes instead: depth, motion and normal together
// (at the depth their light comes from, facing the camera), the albedos and roughness blended by their
// opacity, as its integration guide asks for transparent layers. Else smoke over the sky would move
// with the sky and leave trails behind when the camera moves.
void DescribeVolumesInGuides(FrameData f, uint2 pixel, float3 o, float3 d, float2 size, MarchResult r, float albedo)
{
    float opacity = 1.0f - r.transmittance;
    RWTexture2D<float4> diffuseAlbedo = ResourceDescriptorHeap[g.diffuseAlbedo];
    RWTexture2D<float4> specularAlbedo = ResourceDescriptorHeap[g.specularAlbedo];
    RWTexture2D<float4> normalRoughness = ResourceDescriptorHeap[g.normalRoughness];
    diffuseAlbedo[pixel] = lerp(diffuseAlbedo[pixel], float4(albedo, albedo, albedo, 1.0f), opacity);
    specularAlbedo[pixel] = lerp(specularAlbedo[pixel], float4(0.0f, 0.0f, 0.0f, 1.0f), opacity);
    float4 normal = normalRoughness[pixel];
    normal.w = lerp(normal.w, 1.0f, opacity);
    if (opacity >= 0.5f)
    {
        RWTexture2D<float> viewDepth = ResourceDescriptorHeap[g.viewDepth];
        RWTexture2D<float2> motionVectors = ResourceDescriptorHeap[g.motionVectors];
        normal.xyz = -d;
        viewDepth[pixel] = r.depth * dot(d, Xyz(f.camera.forward));
        float2 at = float2(pixel) + f.jitter + 0.5f;
        motionVectors[pixel] = CameraProject(f.previousCamera, o + d * r.depth, 1.0f, size) - at;
    }
    normalRoughness[pixel] = normal;
}

groupshared uint gTileVolumes;

[numthreads(kMarchTile * kMarchTile, 1, 1)]
void Main(uint3 groupId : SV_GroupID, uint index : SV_GroupIndex)
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData f = frames[g.frameSlot];
    gVolumeShadows = f.volumeShadows;
    uint lx = (index & 1u) | ((index >> 1u) & 2u) | ((index >> 2u) & 4u);
    uint ly = ((index >> 1u) & 1u) | ((index >> 2u) & 2u) | ((index >> 3u) & 4u);
    // The tiles move by another offset every frame (an R2 sequence over 8 x 8 pixels; the pass has a tile
    // more per row and column), so that the blocks of pixels that share their lighting (MarchRayShared)
    // lie elsewhere every frame: accumulated, or under TAA, a pixel's light is that of the pixels around
    // it rather than of a fixed block's.
    int2 shift = int2(int((f.frameIndex * 3242174889u) >> 29u), int((f.frameIndex * 2447445413u) >> 29u));
    int2 at = int2(groupId.xy * kMarchTile + uint2(lx, ly)) - shift;
    bool inside = at.x >= 0 && at.y >= 0 && uint(at.x) < f.width && uint(at.y) < f.height;
    uint2 pixel = uint2(max(at, int2(0, 0)));
    float2 size = float2(f.width, f.height);
    float3 o = Xyz(f.camera.position);
    float3 d = CameraRayDirection(f.camera, float2(pixel) + f.jitter + 0.5f, size);
    Texture2D<float4> surface = ResourceDescriptorHeap[g.surface];
    float4 frame = inside ? surface[pixel] : float4(0.0f, 0.0f, 0.0f, 0.0f);
    float tSurface = frame.w;

    // The volumes the tile's rays cross in front of their surfaces.
    uint mask = 0u;
    for (uint v = 0u; v < f.volumeCount && inside; ++v)
    {
        float t0;
        float t1;
        if (MarchVolumeSegment(LoadMarchVolume(v), o, d, tSurface, t0, t1))
        {
            mask |= 1u << v;
        }
    }
    if (index == 0u)
    {
        gTileVolumes = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    InterlockedOr(gTileVolumes, mask);
    GroupMemoryBarrierWithGroupSync();
    uint tileIndices = 0u;
    uint tileCount = 0u;
    for (uint bits = gTileVolumes; bits != 0u && tileCount < kMarchMaxVolumes; bits &= bits - 1u)
    {
        tileIndices |= firstbitlow(bits) << (8u * tileCount);
        ++tileCount;
    }
    if (index < tileCount)
    {
        VolumeMarchData v = LoadMarchVolume((tileIndices >> (8u * index)) & 0xFFu);
        gTileVolume[index] = v;
        gTileLevel[index] = LoadVolumeLevel(v.levels + g.level);
    }
    GroupMemoryBarrierWithGroupSync();

    // This ray's volumes (none outside the image: the lanes there still take part in the waves').
    MarchRayVolumes rv;
    rv.count = 0u;
    rv.indices = 0u;
    rv.slots = 0u;
    rv.tIn = float4(0.0f, 0.0f, 0.0f, 0.0f);
    rv.tOut = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint k = 0u; k < tileCount; ++k)
    {
        uint vIndex = (tileIndices >> (8u * k)) & 0xFFu;
        VolumeMarchData v = LoadMarchVolume(vIndex);
        float t0;
        float t1;
        if (!inside || !MarchVolumeSegment(v, o, d, tSurface, t0, t1))
        {
            continue;
        }
        uint slot = rv.count;
        rv.indices |= vIndex << (8u * slot);
        rv.slots |= k << (8u * slot);
        rv.tIn = SetLane4(rv.tIn, slot, t0);
        rv.tOut = SetLane4(rv.tOut, slot, t1);
        ++rv.count;
    }

    MarchSetup s;
    s.sky = f.sky;
    s.sunMap = gVolumeShadows.sunMap;
    s.lightCount = g.marchLightCount;
    s.probeVolumes = g.probeVolumes;
    s.level = g.level;
    s.stepScale = g.stepScale;
    s.flags = g.flags;
    StructuredBuffer<uint> noise = ResourceDescriptorHeap[g.blueNoise];
    uint rank = noise[(pixel.y % 64u) * 64u + pixel.x % 64u];
    s.lightSample = MarchLightSample(rank, f.frameIndex);
    MarchChooseMaps(s);
    float jitter = MarchJitter(rank, f.frameIndex);
    uint share = 1u;
    MarchResult r;
    if ((g.flags & kMarchAdaptive) != 0u)
    {
        uint seed = Pcg(f.frameIndex + Pcg(groupId.x + Pcg(groupId.y)));
        r = MarchRayShared(s, rv, o, d, tSurface, jitter, 2.0f * f.camera.position.w / size.y, seed, share);
    }
    else
    {
        r = MarchRay(s, rv, o, d, tSurface, jitter);
    }
    if (!inside)
    {
        return;
    }
    bool lit = g.view == kDebugNone || g.view == kDebugDirect || g.view == kDebugIndirect;
    if ((g.flags & kMarchGuides) != 0u && lit && rv.count > 0u)
    {
        DescribeVolumesInGuides(f, pixel, o, d, size, r, gTileVolume[rv.slots & 0xFFu].albedo);
    }

    // Over the surface: the volumes' light and what they let through, the air in front of them taken
    // out of the surface's light at the depth their light comes from and laid over both (the fusion
    // of two media by one depth, as Mueller fuses fog).
    float samples = float(g.samplesPerPixel);
    float3 sum = float3(frame.x, frame.y, frame.z);
    float3 value;
    if (g.view == kDebugVolumeSteps)
    {
        value = float(r.steps) / f.debugScale * samples;
    }
    else if (g.view == kDebugVolumeSegments)
    {
        value = float(rv.count) / float(kMarchMaxVolumes) * samples;
    }
    else if (g.view == kDebugVolumeRate)
    {
        float3 rate = share == 16u ? float3(1.0f, 0.0f, 0.0f) : (share == 4u ? float3(0.0f, 1.0f, 0.0f)
                                                                               : float3(0.0f, 0.0f, 1.0f));
        value = rv.count > 0u ? rate * samples : sum;
    }
    else if (!lit)
    {
        value = sum; // the surfaces' own views
    }
    else
    {
        float3 front = r.radiance;
        if ((g.flags & kMarchAir) != 0u && r.transmittance < 1.0f)
        {
            Aerial air = AerialPerspective(f.sky, d, r.depth);
            front = air.inscatter * (1.0f - r.transmittance) + air.transmittance * r.radiance;
        }
        if ((g.flags & kMarchLayer) != 0u)
        {
            RWTexture2D<float4> layer = ResourceDescriptorHeap[g.layer];
            RWTexture2D<float> layerDepth = ResourceDescriptorHeap[g.layerDepth];
            layer[pixel] = float4(front, r.transmittance);
            layerDepth[pixel] = r.transmittance < 1.0f ? r.depth : 0.0f;
            return; // VolumeMarch.Temporal lays it over the surfaces
        }
        value = front * samples + sum * r.transmittance;
    }
    if ((g.flags & kMarchSurfaceTransmittance) != 0u)
    {
        RWTexture2D<float> surfaceTransmittance = ResourceDescriptorHeap[g.surfaceTransmittance];
        surfaceTransmittance[pixel] = r.transmittance;
    }
    RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
    float4 previous = (g.flags & kMarchReset) != 0u ? float4(0.0f, 0.0f, 0.0f, 0.0f) : accumulation[pixel];
    accumulation[pixel] = previous + float4(value, samples);
}
