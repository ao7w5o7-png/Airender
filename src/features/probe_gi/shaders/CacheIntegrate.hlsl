// Probe GI radiance cache (ProbeGiShared.h): per updated probe (a group, a thread per octahedral texel),
// blends this frame's rays into the probe's radiance and distance moments and projects its radiance onto
// spherical harmonics, convolved with the clamped cosine (Ramamoorthi and Hanrahan 2001) into irradiance.
// A probe most of whose rays hit the backs of surfaces is inside a mesh: invalid. A texel that was invalid
// (a new probe) takes this frame's rays whole.
// Every level in one dispatch, as CacheTrace.hlsl.
//   GiConstants: probeCount (the rays' stride), rays, blend.

#include "VoxelTrace.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const uint kTexels = kCacheTexels * kCacheTexels;
static const float kInsideFraction = 0.25f;

groupshared float gsSh[kTexels][28]; // 27 SH terms and the texel's solid angle
groupshared uint gsBackfaces;

float SphericalTriangleArea(float3 a, float3 b, float3 c)
{
    return 2.0f * atan2(abs(dot(a, cross(b, c))), 1.0f + dot(a, b) + dot(b, c) + dot(c, a));
}

[numthreads(kCacheTexels, kCacheTexels, 1)]
void Main(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    uint level = group.y;
    if (level >= f.cascadeCount || f.cascades[level].valid == 0u)
    {
        return;
    }
    CascadeData c = f.cascades[level];
    uint count = c.cacheProbes == 0u ? kCacheProbes : c.cacheProbes;
    if (group.x >= count)
    {
        return;
    }
    uint first = c.cacheProbes == 0u ? 0u : (f.frameIndex * c.cacheProbes) % kCacheProbes;
    uint index = (first + group.x) % kCacheProbes;
    StructuredBuffer<float4> rays = ResourceDescriptorHeap[g.rays];
    float4 ray = rays[(level * g.probeCount + group.x) * kTexels + lane];
    if (lane == 0u)
    {
        gsBackfaces = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (ray.w < 0.0f)
    {
        InterlockedAdd(gsBackfaces, 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    float valid = float(gsBackfaces) < kInsideFraction * float(kTexels) ? 1.0f : 0.0f;

    RWTexture2D<float4> atlas = ResourceDescriptorHeap[c.cacheRadianceWritable];
    RWTexture2D<float2> moments = ResourceDescriptorHeap[c.cacheDistanceWritable];
    uint2 texel = CacheAtlasTexel(index, thread.xy);
    float4 old = atlas[texel];
    float blend = old.w > 0.0f ? g.blend : 1.0f;
    float3 radiance = lerp(old.xyz / kHalfLightScale, ray.xyz, blend);
    float distance = min(abs(ray.w), CacheSpacing(c) * float(kCacheResolution));
    float2 m = lerp(moments[texel], float2(distance, distance * distance), blend);
    atlas[texel] = float4(radiance * kHalfLightScale, valid);
    moments[texel] = m;

    // The texel's solid angle (two spherical triangles), normalized over the probe below.
    float3 d00 = CacheDirection(float2(thread.xy) / float(kCacheTexels));
    float3 d10 = CacheDirection(float2(thread.x + 1u, thread.y) / float(kCacheTexels));
    float3 d01 = CacheDirection(float2(thread.x, thread.y + 1u) / float(kCacheTexels));
    float3 d11 = CacheDirection(float2(thread.x + 1u, thread.y + 1u) / float(kCacheTexels));
    float area = SphericalTriangleArea(d00, d10, d11) + SphericalTriangleArea(d00, d11, d01);
    float3 direction = CacheDirection((float2(thread.xy) + 0.5f) / float(kCacheTexels));
    float b[9];
    ShBasis(direction, b);
    [unroll] for (uint i = 0u; i < 9u; ++i)
    {
        gsSh[lane][i * 3u + 0u] = radiance.x * b[i] * area;
        gsSh[lane][i * 3u + 1u] = radiance.y * b[i] * area;
        gsSh[lane][i * 3u + 2u] = radiance.z * b[i] * area;
    }
    gsSh[lane][27] = area;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = kTexels / 2u; stride > 0u; stride >>= 1)
    {
        if (lane < stride)
        {
            [unroll] for (uint k = 0u; k < 28u; ++k)
            {
                gsSh[lane][k] += gsSh[lane + stride][k];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        float scale = 4.0f * kPi / max(gsSh[0][27], 1e-6f);
        // Clamped cosine convolution per band: pi, 2 pi / 3, pi / 4.
        float bands[9] = {kPi, 2.0943951f, 2.0943951f, 2.0943951f, 0.7853982f, 0.7853982f, 0.7853982f, 0.7853982f, 0.7853982f};
        float3 coefficients[9];
        [unroll] for (uint i = 0u; i < 9u; ++i)
        {
            coefficients[i] = float3(gsSh[0][i * 3u], gsSh[0][i * 3u + 1u], gsSh[0][i * 3u + 2u]) * (scale * bands[i]);
        }
        RWStructuredBuffer<ProbeSh> probes = ResourceDescriptorHeap[c.cacheIrradianceWritable];
        probes[index] = ShPack(coefficients, valid);
    }
}
