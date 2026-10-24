// air-shader-target: cs_6_6
// The real-time volumes' ray march (docs/architecture.md §8.13): after the trace has written this
// frame's samples and the distance to their first hit (PathTracerVolumeShadows.hlsl), every pixel
// marches its camera ray through the volumes in front of that hit (VolumeMarch.hlsli), lays their light
// over the surface's and adds the result to the accumulation.
// A thread group covers 8 x 8 pixels (its threads in Morton order, so that 4 consecutive lanes are a
// 2 x 2 block and 16 a 4 x 4 one): it finds the volumes some of its rays cross, of which its rays march
// the first kMarchMaxVolumes. With kMarchAdaptive a wave whose pixels are all deep in smoke lights
// fewer steps (MarchRayShared).

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

uint VolumeOccupancyLoad(VolumeBrickLevel l, uint index)
{
    StructuredBuffer<uint> buffer = ResourceDescriptorHeap[NonUniformResourceIndex(l.occupancy)];
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
// reading its density, and once the light from behind has mostly gone, the light of a step matters
// less. When the transmittance of every pixel of the wave falls below shareBelow2, the pixels of each
// 2 x 2 block (below shareBelow4: 4 x 4) share their lighting: for a block of 4 (16) steps, every pixel
// lights one of them, at its own ray's depth for that step, and each pixel's step takes the light of
// the block's pixel that lit it, mixed with the next one's by the pixel's blue noise (the depth jitter
// carried over to the shared samples). Densities, depths and transmittances stay every pixel's own.
// All lanes run every iteration (the shared samples of finished ones are still read); `share` is the
// group the wave ended with.
MarchResult MarchRayShared(MarchSetup s, MarchRayVolumes rv, float3 o, float3 d, float tSurface, float jitter,
                           AIR_OUT(uint) share)
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
    float weightedDepth = 0.0f;
    float weight = 0.0f;
    uint lane = WaveGetLaneIndex();
    share = 1u;
    uint i = 0u;
    float3 last = float3(0.0f, 0.0f, 0.0f);
    bool hasLast = false;
    for (uint n = 0u; n < kMarchMaxSteps && WaveActiveAnyTrue(live); ++n)
    {
        // Pixels without volumes, or out of them again, keep their transmittance high.
        float waveTransmittance = WaveActiveMax(result.transmittance);
        if (share == 1u && waveTransmittance < g.shareBelow2)
        {
            share = 4u;
        }
        if (share == 4u && waveTransmittance < g.shareBelow4)
        {
            share = 16u;
        }
        if (share == 1u)
        {
            if (live)
            {
                live = MarchStep(s, rv, o, d, tStart, tEnd, step, jitter, ambient, i, result, weightedDepth, weight);
            }
            continue;
        }
        // This lane's sample of the block's light: at its own step i + q, per unit of scattering.
        uint base = lane & ~(share - 1u);
        uint q = lane - base;
        float3 own = float3(0.0f, 0.0f, 0.0f);
        uint valid = 0u;
        if (rv.count > 0u)
        {
            float tq = tStart + (float(i + q) + jitter) * step;
            float3 xq = o + d * tq;
            MarchPoint pq = MarchPointAt(s, rv, xq, d, tq, 1e30f, ambient);
            float scattering = pq.sigmaS.x + pq.sigmaS.y + pq.sigmaS.z + pq.sigmaS.w;
            if (scattering > 0.0f)
            {
                own = (MarchDirect(s, rv, pq, xq, d) + pq.around) / scattering;
                valid = 1u;
            }
        }
        for (uint j = 0u; j < share; ++j)
        {
            float3 a = WaveReadLaneAt(own, base + j);
            float3 b = WaveReadLaneAt(own, base + (j + 1u) % share);
            uint validA = WaveReadLaneAt(valid, base + j);
            uint validB = WaveReadLaneAt(valid, base + (j + 1u) % share);
            if (!live)
            {
                continue;
            }
            float t0 = tStart + float(i + j) * step;
            if (t0 >= tEnd)
            {
                live = false;
                continue;
            }
            float segment = min(step, tEnd - t0);
            float t = t0 + jitter * segment;
            float3 x = o + d * t;
            MarchPoint p = MarchPointAt(s, rv, x, d, t, tEnd, false);
            if (p.sigmaT <= 0.0f)
            {
                continue;
            }
            float3 light = last;
            if (validA != 0u && validB != 0u)
            {
                light = lerp(a, b, jitter);
            }
            else if (validA != 0u)
            {
                light = a;
            }
            else if (validB != 0u)
            {
                light = b;
            }
            else if (!hasLast)
            {
                MarchPoint lit = MarchPointAt(s, rv, x, d, t, tEnd, ambient);
                light = (MarchDirect(s, rv, lit, x, d) + lit.around) /
                        max(lit.sigmaS.x + lit.sigmaS.y + lit.sigmaS.z + lit.sigmaS.w, 1e-20f);
            }
            last = light;
            hasLast = true;
            float scattering = p.sigmaS.x + p.sigmaS.y + p.sigmaS.z + p.sigmaS.w;
            MarchIntegrate(result, weightedDepth, weight, light * scattering, p.sigmaT, segment, t);
            live = result.transmittance >= kMarchOpaque;
        }
        i += share;
    }
    result.depth = weight > 0.0f ? weightedDepth / weight : tStart;
    return result;
}

// --- Pass ----------------------------------------------------------------------------------------

groupshared uint gTileVolumes;

[numthreads(kMarchTile * kMarchTile, 1, 1)]
void Main(uint3 groupId : SV_GroupID, uint index : SV_GroupIndex)
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData f = frames[g.frameSlot];
    gVolumeShadows = f.volumeShadows;
    uint lx = (index & 1u) | ((index >> 1u) & 2u) | ((index >> 2u) & 4u);
    uint ly = ((index >> 1u) & 1u) | ((index >> 2u) & 2u) | ((index >> 3u) & 4u);
    uint2 tile = groupId.xy * kMarchTile;
    uint2 pixel = tile + uint2(lx, ly);
    bool inside = pixel.x < f.width && pixel.y < f.height;
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

    // This ray's volumes (none outside the image: the lanes there still take part in the waves').
    MarchRayVolumes rv;
    rv.count = 0u;
    rv.indices = 0u;
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
    float jitter = MarchJitter(rank, f.frameIndex);
    uint share = 1u;
    MarchResult r;
    if ((g.flags & kMarchAdaptive) != 0u)
    {
        r = MarchRayShared(s, rv, o, d, tSurface, jitter, share);
    }
    else
    {
        r = MarchRay(s, rv, o, d, tSurface, jitter);
    }
    if (!inside)
    {
        return;
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
    else if (g.view != kDebugNone && g.view != kDebugDirect && g.view != kDebugIndirect)
    {
        value = sum; // the surfaces' own views
    }
    else if ((g.flags & kMarchAir) != 0u && r.transmittance < 1.0f)
    {
        Aerial air = AerialPerspective(f.sky, d, r.depth);
        value = (air.inscatter * (1.0f - r.transmittance) + air.transmittance * r.radiance) * samples +
                sum * r.transmittance;
    }
    else
    {
        value = r.radiance * samples + sum * r.transmittance;
    }
    RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
    float4 previous = (g.flags & kMarchReset) != 0u ? float4(0.0f, 0.0f, 0.0f, 0.0f) : accumulation[pixel];
    accumulation[pixel] = previous + float4(value, samples);
}
