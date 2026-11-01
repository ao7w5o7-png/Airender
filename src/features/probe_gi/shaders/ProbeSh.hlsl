// Probe GI screen probes (ProbeGiShared.h): every probe's rays as spherical harmonics up to band 2 of the
// irradiance, by quadrature over the cells of its hemi-octahedral grid (each cell's light times its solid
// angle). The hemisphere below the probe's surface takes the light of the one above mirrored in it, as the
// path tracer's diffuse lobe does for shading normals that lean away from the surface (docs §8.2): pixels
// whose normals lean away from the probe's see a little of it. One group per probe.
//   GiConstants: input3 probes, rays (Texture2D<float4>), output (RWStructuredBuffer<ProbeSh>).

#include "VoxelTrace.hlsli"
#include "ProbeRays.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const uint kCells = kScreenProbeTexels * kScreenProbeTexels;

groupshared float gsSh[kCells][31]; // 27 SH terms, the cell's solid angle, the irradiance at the normal


float SphericalTriangleArea(float3 a, float3 b, float3 c)
{
    return 2.0f * atan2(abs(dot(a, cross(b, c))), 1.0f + dot(a, b) + dot(b, c) + dot(c, a));
}

float3 CellCorner(uint2 cell)
{
    return HemiOctDecode(float2(cell) / float(kScreenProbeTexels) * 2.0f - 1.0f);
}

float CellArea(uint2 cell)
{
    return SphericalTriangleArea(CellCorner(cell), CellCorner(cell + uint2(1, 0)), CellCorner(cell + uint2(1, 1))) +
           SphericalTriangleArea(CellCorner(cell), CellCorner(cell + uint2(1, 1)), CellCorner(cell + uint2(0, 1)));
}

[numthreads(kScreenProbeTexels, kScreenProbeTexels, 1)]
void Main(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    uint index = group.y * f.probesX + group.x;
    StructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.input3];
    ScreenProbe probe = probes[index];
    Texture2D<float4> rays = ResourceDescriptorHeap[g.rays];
    float4 ray = rays.Load(int3(group.xy * kScreenProbeTexels + thread.xy, 0));
    ray.xyz /= kHalfLightScale;
    float3 n = UnpackNormal(probe.normal);
    float3 direction = ProbeRayDirection(MakeBasis(n), thread.xy, ProbeJitter(f, index).cell); // the trace's

    float3 mirrored = direction - 2.0f * dot(direction, n) * n;
    float area = CellArea(thread.xy);
    float b[9];
    float bm[9];
    ShBasis(direction, b);
    ShBasis(mirrored, bm);
    [unroll] for (uint i = 0u; i < 9u; ++i)
    {
        float w = (b[i] + bm[i]) * area;
        gsSh[lane][i * 3u + 0u] = ray.x * w;
        gsSh[lane][i * 3u + 1u] = ray.y * w;
        gsSh[lane][i * 3u + 2u] = ray.z * w;
    }
    gsSh[lane][27] = area;
    float cosine = max(dot(direction, n), 0.0f) * area;
    gsSh[lane][28] = ray.x * cosine;
    gsSh[lane][29] = ray.y * cosine;
    gsSh[lane][30] = ray.z * cosine;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = kCells / 2u; stride > 0u; stride >>= 1)
    {
        if (lane < stride)
        {
            [unroll] for (uint k = 0u; k < 31u; ++k)
            {
                gsSh[lane][k] += gsSh[lane + stride][k];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        float scale = 2.0f * kPi / max(gsSh[0][27], 1e-6f); // the cells cover the hemisphere exactly
        float bands[9] = {kPi, 2.0943951f, 2.0943951f, 2.0943951f, 0.7853982f, 0.7853982f, 0.7853982f, 0.7853982f, 0.7853982f};
        float3 coefficients[9];
        [unroll] for (uint i = 0u; i < 9u; ++i)
        {
            coefficients[i] = float3(gsSh[0][i * 3u], gsSh[0][i * 3u + 1u], gsSh[0][i * 3u + 2u]) * (scale * bands[i]);
        }
        RWStructuredBuffer<ProbeSh> output = ResourceDescriptorHeap[g.output];
        ProbeSh result = ShPack(coefficients, probe.depth > 0.0f ? 1.0f : 0.0f);
        result.normalIrradiance = float4(float3(gsSh[0][28], gsSh[0][29], gsSh[0][30]) * scale, 0.0f);
        output[index] = result;
    }
}
