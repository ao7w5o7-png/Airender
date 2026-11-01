// Probe GI screen probes (ProbeGiShared.h): each probe's harmonics averaged with its neighbours' in a
// (2 r + 1)^2 window of probes that lie on its surface (close to its plane) and face its way. One thread per
// probe.
//   GiConstants: input3 probes, input4 harmonics (StructuredBuffer<ProbeSh>), input5 radius, output
//   (RWStructuredBuffer<ProbeSh>).

#include "VoxelTrace.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

float PlaneWeight(ScreenProbe centre, float3 n, ScreenProbe other)
{
    float3 offset = other.position - centre.position;
    float distance = length(offset);
    float plane = abs(dot(offset, n));
    // Within a few per cent of the distance (a surface seen at an angle) and a few pixels' width.
    float tolerance = 0.05f * distance + 2.0f * centre.footprint;
    return saturate(1.0f - plane / tolerance);
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.probesX || id.y >= f.probesY + f.adaptiveRows)
    {
        return;
    }
    StructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.input3];
    StructuredBuffer<ProbeSh> input = ResourceDescriptorHeap[g.input4];
    RWStructuredBuffer<ProbeSh> output = ResourceDescriptorHeap[g.output];
    uint index = id.y * f.probesX + id.x;
    ScreenProbe centre = probes[index];
    if (centre.depth <= 0.0f || id.y >= f.probesY) // adaptive probes have no neighbours on the grid
    {
        output[index] = input[index];
        return;
    }
    float3 n = UnpackNormal(centre.normal);
    int radius = int(g.input5);
    float4 sum[8];
    [unroll] for (uint k = 0u; k < 8u; ++k)
    {
        sum[k] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float weight = 0.0f;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            int2 q = int2(id.xy) + int2(x, y);
            if (any(q < 0) || q.x >= int(f.probesX) || q.y >= int(f.probesY))
            {
                continue;
            }
            uint other = uint(q.y) * f.probesX + uint(q.x);
            ScreenProbe p = probes[other];
            if (p.depth <= 0.0f)
            {
                continue;
            }
            float3 m = UnpackNormal(p.normal);
            float w = PlaneWeight(centre, n, p) * pow(saturate(dot(n, m)), 8.0f);
            if (x != 0 || y != 0)
            {
                w *= 0.5f;
            }
            if (w <= 0.0f)
            {
                continue;
            }
            ProbeSh sh = input[other];
            sum[0] += sh.c0 * w;
            sum[1] += sh.c1 * w;
            sum[2] += sh.c2 * w;
            sum[3] += sh.c3 * w;
            sum[4] += sh.c4 * w;
            sum[5] += sh.c5 * w;
            sum[6] += float4(sh.c6, 0.0f) * w;
            sum[7] += sh.normalIrradiance * w;
            weight += w;
        }
    }
    ProbeSh result;
    float inv = 1.0f / max(weight, 1e-6f);
    result.c0 = sum[0] * inv;
    result.c1 = sum[1] * inv;
    result.c2 = sum[2] * inv;
    result.c3 = sum[3] * inv;
    result.c4 = sum[4] * inv;
    result.c5 = sum[5] * inv;
    result.c6 = sum[6].xyz * inv;
    result.valid = 1.0f;
    result.normalIrradiance = sum[7] * inv;
    output[index] = result;
}
