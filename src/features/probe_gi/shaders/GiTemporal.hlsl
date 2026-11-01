// Probe GI: the pixels' light over frames (ProbeGiShared.h). Each pixel follows its surface back into last
// frame's result (motion vectors, bilinear over the four texels there whose depth and normal match the
// surface's), and blends this frame's in: 1 / n over the first frames of a history, then at least
// GiFrame::temporalBlend where the pixel moves a pixel or more, GiFrame::staticBlend where it stays (the probes
// sit at another pixel of their tiles every frame, and the light a pixel interpolates between them changes with
// them: a longer history averages more of their places). A pixel without history starts over. Where the blend is
// 2% or more (a young history, a moving pixel) this frame's irradiance is first averaged over the pixel's surface
// around it, on a disk of 24 pixels, wider for a larger blend: a moving camera's histories are short (what comes
// into view, the blend that keeps lag down), and the probes' error, spatially correlated over tiles, came through
// as blotches (sponza moving sideways 3 cm a frame at 8% a frame, against a still camera's: error between 16 x 16
// blocks 3.5% -> 2.3%, grain within them 0.93% -> 0.50%; the pass 0.3 ms at 1280 x 720). The disk let the moving
// blend come down to 3%: a surface's change from one frame to the next (motion compensated) 0.27% -> 0.11%, its
// 90th percentile 1.9% -> 1.0%, in the darkest quarter 1.5% -> 0.7%.
//   GiConstants: input0 view depth, input1 normal + roughness, input2 motion vectors, input3 this frame's
//   irradiance, input4 its specular, input5 last frame's irradiance (a: frames of history), input6 last
//   frame's specular, input7 last frame's surfaces (rgb normal, a view depth); output irradiance, rays its
//   specular, counters its surfaces (RWTexture2D<float4> each).

#include "GiCommon.hlsli"
#include "ReflectionCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kSkyDepth = 65000.0f;
static const float kMaxHistory = 128.0f;
static const float kUnfittedBlend = 0.25f;
static const float kTracedSpecularBlend = 0.25f;
static const uint kSpatialSamples = 12u;
// The disk's radius in pixels up to blend kSpatialBlend (a moving pixel's), beyond it growing as the square root of
// the blend (the noise a blend lets through), at most kSpatialMaxRadius; none below kSpatialLeastBlend (a still
// pixel's long history).
static const float kSpatialRadius = 24.0f;
static const float kSpatialBlend = 0.08f;
static const float kSpatialMaxRadius = 32.0f;
static const float kSpatialLeastBlend = 0.02f;

// This frame's irradiance averaged over the pixel's surface around it: kSpatialSamples pixels on a disk of `radius`
// pixels, turned at random every frame, weighed by how close each lies to the pixel's plane and how alike its normal is.
float3 SpatialIrradiance(GiFrame f, int2 pixel, float3 p, float3 n, float3 centre, float radius, inout uint seed)
{
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.input1];
    Texture2D<float4> irradianceNow = ResourceDescriptorHeap[g.input3];
    float footprint = PixelFootprint(f, depth.Load(int3(pixel, 0)));
    float3 sum = centre;
    float weight = 1.0f;
    float turn = Rand(seed) * 6.2831853f;
    for (uint k = 0u; k < kSpatialSamples; ++k)
    {
        float angle = turn + 2.3999632f * float(k);
        float2 offset = float2(cos(angle), sin(angle)) * radius * sqrt((float(k) + 0.5f) / float(kSpatialSamples));
        int2 q = pixel + int2(round(offset));
        if (any(q < 0) || q.x >= int(f.width) || q.y >= int(f.height))
        {
            continue;
        }
        float zq = depth.Load(int3(q, 0));
        if (!(zq < kSkyDepth) || zq <= 0.0f)
        {
            continue;
        }
        float3 view;
        float3 offsetWorld = PixelPosition(f, uint2(q), zq, view) - p;
        float tolerance = 0.05f * length(offsetWorld) + 2.0f * footprint;
        float3 m = normalRoughness.Load(int3(q, 0)).xyz;
        float similar = dot(m, m) > 0.0f ? saturate(dot(n, normalize(m))) : 0.0f;
        float s2 = similar * similar;
        float w = saturate(1.0f - abs(dot(offsetWorld, n)) / tolerance) * (s2 * s2 * s2 * s2);
        if (w > 0.0f)
        {
            sum += irradianceNow.Load(int3(q, 0)).xyz * w;
            weight += w;
        }
    }
    return sum / weight;
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.width || id.y >= f.height)
    {
        return;
    }
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.input1];
    Texture2D<float4> irradianceNow = ResourceDescriptorHeap[g.input3];
    Texture2D<float4> specularNow = ResourceDescriptorHeap[g.input4];
    RWTexture2D<float4> irradianceOut = ResourceDescriptorHeap[g.output];
    RWTexture2D<float4> specularOut = ResourceDescriptorHeap[g.rays];
    RWTexture2D<float4> surfaceOut = ResourceDescriptorHeap[g.counters];
    float z = depth.Load(int3(id.xy, 0));
    float4 e = irradianceNow.Load(int3(id.xy, 0));
    float4 s = specularNow.Load(int3(id.xy, 0));
    float4 nr = normalRoughness.Load(int3(id.xy, 0));
    float3 n = nr.xyz;
    if (!(z < kSkyDepth) || z <= 0.0f)
    {
        irradianceOut[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        specularOut[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        surfaceOut[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    n = dot(n, n) > 0.0f ? normalize(n) : float3(0.0f, 1.0f, 0.0f);
    float3 history = float3(0.0f, 0.0f, 0.0f);
    float3 historySpecular = float3(0.0f, 0.0f, 0.0f);
    float frames = 0.0f;
    float leastBlend = f.temporalBlend;
    if ((f.flags & kGiHistory) != 0u && g.input5 != kInvalidIndex)
    {
        Texture2D<float2> motion = ResourceDescriptorHeap[g.input2];
        leastBlend = lerp(f.staticBlend, f.temporalBlend, saturate(length(motion.Load(int3(id.xy, 0)))));
        Texture2D<float4> irradiancePast = ResourceDescriptorHeap[g.input5];
        Texture2D<float4> specularPast = ResourceDescriptorHeap[g.input6];
        Texture2D<float4> surfacePast = ResourceDescriptorHeap[g.input7];
        float2 size = float2(f.width, f.height);
        float3 view;
        float3 p = PixelPosition(f, id.xy, z, view);
        float expected = CameraProject(f.previousCamera, p, size).z;
        // Where this pixel's sample was last frame, in last frame's texel grid (its samples at the texel
        // centres plus its jitter).
        float2 previous = float2(id.xy) + f.jitter + motion.Load(int3(id.xy, 0)) - f.previousJitter;        float2 base = floor(previous);
        float2 w1 = previous - base;
        // A surface seen at a grazing angle changes its depth quickly from one pixel to the next.
        float facing = lerp(0.25f, 1.0f, saturate(dot(n, -view)));
        float tolerance = (0.02f * expected + 2.0f * PixelFootprint(f, z)) / facing;
        float weight = 0.0f;
        float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 sumSpecular = float3(0.0f, 0.0f, 0.0f);
        for (uint k = 0u; k < 4u; ++k)
        {
            int2 corner = int2(k & 1u, k >> 1);
            int2 texel = int2(base) + corner;
            if (any(texel < 0) || texel.x >= int(f.width) || texel.y >= int(f.height))
            {
                continue;
            }
            float4 surface = surfacePast.Load(int3(texel, 0));
            if (surface.w <= 0.0f || abs(surface.w - expected) > tolerance || dot(surface.xyz, n) < 0.8f)
            {
                continue;
            }
            float2 wc = select(corner != 0, w1, 1.0f - w1);
            float w = wc.x * wc.y;
            sum += irradiancePast.Load(int3(texel, 0)) * w;
            sumSpecular += specularPast.Load(int3(texel, 0)).xyz * w;
            weight += w;
        }
        if (weight > 0.05f)
        {
            history = sum.xyz / weight;
            historySpecular = sumSpecular / weight;
            frames = sum.w / weight;
        }
    }
    frames = min(frames + 1.0f, kMaxHistory);
    float blend = max(1.0f / frames, leastBlend);
    if (e.w < 0.5f && frames > 1.0f)
    {
        // No probe fitted this pixel (the radiance cache's coarse light stands in): its history is the better
        // estimate, and this frame's moves it less.
        blend *= kUnfittedBlend;
    }
    uint seed = Pcg(id.y * f.width + id.x) ^ Pcg(f.frameIndex);
    float dither = Rand(seed);
    float3 now = e.xyz;
    if (blend >= kSpatialLeastBlend)
    {
        // A short or fast history lets this frame's noise through: average it over the surface around first.
        float3 view;
        float3 p = PixelPosition(f, id.xy, z, view);
        float radius = min(kSpatialRadius * sqrt(max(blend, kSpatialBlend) / kSpatialBlend), kSpatialMaxRadius);
        now = SpatialIrradiance(f, int2(id.xy), p, n, e.xyz, radius, seed);
    }
    float3 irradiance = lerp(history, now, blend);
    // Traced reflections move with what they reflect, not with the surface: a shorter history (ReflectionCommon.hlsli).
    float3 specular = lerp(historySpecular, s.xyz, max(blend, ReflectionWeight(nr.w) * kTracedSpecularBlend));
    irradianceOut[id.xy] = float4(RoundToHalf(irradiance, dither), frames);
    specularOut[id.xy] = float4(RoundToHalf(specular, dither), 1.0f);
    surfaceOut[id.xy] = float4(n, z);
}
