// Probe GI screen probes (ProbeGiShared.h): every probe's rays averaged with the rays of the probes around it, per
// direction, before they become harmonics. A neighbour lends the ray of its cell that points this ray's way, when
// it lies on this probe's surface and facing its way, weighed by how far the point that ray found lies from this
// ray's direction as seen from here (kMaxAngle: a ray that hit a wall next to the neighbour hits nothing from here
// when the wall is off to its side), its distance clamped to this ray's (a near occluder seen from here stays one
// for the neighbour's light: contact shadows survive). Where the harmonics of neighbouring probes were averaged,
// light from a direction that one probe saw occluded and the next did not became light from every direction.
// One group per probe, one thread per ray.
//   GiConstants: input3 probes, input4 the rays (Texture2D<float4>: rgb light times kHalfLightScale, a distance),
//   input5 radius (probes), rays the filtered rays (RWTexture2D<float4>).

#include "VoxelTrace.hlsli"
#include "ProbeRays.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kMaxAngle = 0.1745329f; // 10 degrees
static const float kNeighbourWeight = 0.5f;

// Hemi-octahedral map: the hemisphere z >= 0 onto the square [-1, 1]^2 (ProbeRays.hlsli's HemiOctDecode).
float2 HemiOctEncode(float3 v)
{
    v /= abs(v.x) + abs(v.y) + abs(v.z);
    return float2(v.x + v.y, v.x - v.y);
}

float PlaneWeight(ScreenProbe centre, float3 n, ScreenProbe other)
{
    float3 offset = other.position - centre.position;
    float plane = abs(dot(offset, n));
    float tolerance = 0.05f * length(offset) + 2.0f * centre.footprint;
    return saturate(1.0f - plane / tolerance);
}

[numthreads(kScreenProbeTexels, kScreenProbeTexels, 1)]
void Main(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    uint index = group.y * f.probesX + group.x;
    StructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.input3];
    Texture2D<float4> rays = ResourceDescriptorHeap[g.input4];
    RWTexture2D<float4> filtered = ResourceDescriptorHeap[g.rays];
    uint2 texel = group.xy * kScreenProbeTexels + thread.xy;
    float4 own = rays.Load(int3(texel, 0));
    ScreenProbe centre = probes[index];
    int radius = int(g.input5);
    if (centre.depth <= 0.0f || radius == 0 || group.y >= f.probesY) // adaptive probes have no neighbours on the grid
    {
        filtered[texel] = own;
        return;
    }
    float3 n = UnpackNormal(centre.normal);
    float3 direction = ProbeRayDirection(MakeBasis(n), thread.xy, ProbeJitter(f, index).cell);
    float3 sum = own.xyz;
    float weight = 1.0f;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            int2 q = int2(group.xy) + int2(x, y);
            if ((x == 0 && y == 0) || any(q < 0) || q.x >= int(f.probesX) || q.y >= int(f.probesY))
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
            float w = PlaneWeight(centre, n, p) * pow(saturate(dot(n, m)), 8.0f) * kNeighbourWeight;
            if (w <= 0.0f)
            {
                continue;
            }
            // The neighbour's cell that points this way, and where its ray went.
            Basis basis = MakeBasis(m);
            float3 local = ToLocal(basis, direction);
            if (local.z <= 0.0f)
            {
                continue;
            }
            uint2 cell = min(uint2((HemiOctEncode(local) * 0.5f + 0.5f) * float(kScreenProbeTexels)), kScreenProbeTexels - 1u);
            float4 ray = rays.Load(int3(uint2(q) * kScreenProbeTexels + cell, 0));
            float3 lent = ProbeRayDirection(basis, cell, ProbeJitter(f, other).cell);
            float3 hit = p.position + lent * min(ray.w, own.w) - centre.position;
            float error = acos(clamp(dot(hit, direction) * rsqrt(max(dot(hit, hit), 1e-12f)), -1.0f, 1.0f));
            w *= saturate(1.0f - error / kMaxAngle);
            sum += ray.xyz * w;
            weight += w;
        }
    }
    filtered[texel] = float4(sum / weight, own.w);
}
