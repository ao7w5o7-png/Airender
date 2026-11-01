// Probe GI screen probes (ProbeGiShared.h): the rays of every probe (ProbeRays.hlsli), one group per probe and one
// thread per ray, each traced as GiTrace.hlsli: the depth buffer, then the voxels. (Tracing the voxels in a second
// pass, over a list of the rays the depth buffer did not finish, took twice as long: a wave of the list held the
// rays of two or three probes, from different origins in different directions, where a wave here holds 32
// neighbouring rays of one probe.)
//   GiConstants: input0..2 and input4 as GiTrace.hlsli, input3 probes (StructuredBuffer<ScreenProbe>), rays
//   (RWTexture2D<float4>: rgb the light, a the distance; probe p's cell (x, y) at p's tile of the atlas).

#include "GiCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

#include "GiTrace.hlsli"
#include "ProbeRays.hlsli"

[numthreads(kScreenProbeTexels, kScreenProbeTexels, 1)]
void Main(uint3 thread : SV_GroupThreadID, uint3 group : SV_GroupID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    uint index = group.y * f.probesX + group.x;
    StructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.input3];
    ScreenProbe probe = probes[index];
    RWTexture2D<float4> rays = ResourceDescriptorHeap[g.rays];
    uint2 texel = group.xy * kScreenProbeTexels + thread.xy;
    if (probe.depth <= 0.0f)
    {
        rays[texel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    gSkyTable = f.skyTable;
    float3 n = UnpackNormal(probe.normal);
    ProbeRayJitter jitter = ProbeJitter(f, index);
    float3 direction = ProbeRayDirection(MakeBasis(n), thread.xy, jitter.cell);
    uint2 pixel = uint2(probe.pixel & 0xFFFFu, probe.pixel >> 16);
    GiRay ray;
    float tStart;
    if (!TraceGiScreen(f, probe.position, n, direction, pixel, probe.footprint, jitter.march, ray, tStart))
    {
        ray = TraceGiVoxels(f, probe.position, n, direction, tStart);
    }
    rays[texel] = float4(ray.radiance * kHalfLightScale, min(ray.distance, 65000.0f));
}
