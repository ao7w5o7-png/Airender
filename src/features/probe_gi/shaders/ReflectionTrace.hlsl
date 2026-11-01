// Probe GI: a reflection ray per 2 x 2 pixels (ReflectionCommon.hlsli), from the pixel of the four whose turn it is,
// when its surface is smooth enough: a direction drawn from its GGX lobe (visible normals), traced as the probes'
// rays (GiTrace.hlsli: the depth buffer, then the voxels, then the sky). One thread per half resolution texel.
//   GiConstants: input0..2 and input4 as GiTrace.hlsli, input3 normal + roughness (Texture2D<float4>), rays the
//   rays' light (RWTexture2D<float4>: rgb times kHalfLightScale, a the distance, -1: none), output their directions
//   (RWTexture2D<uint>, PackNormal).

#include "GiCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

#include "GiTrace.hlsli"
#include "ReflectionCommon.hlsli"

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    uint2 half = uint2((f.width + 1u) / 2u, (f.height + 1u) / 2u);
    if (any(id.xy >= half))
    {
        return;
    }
    RWTexture2D<float4> light = ResourceDescriptorHeap[g.rays];
    RWTexture2D<uint> directions = ResourceDescriptorHeap[g.output];
    uint2 pixel = min(ReflectionPixel(id.xy, f.frameIndex), uint2(f.width - 1u, f.height - 1u));
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.input3];
    float z = depth.Load(int3(pixel, 0));
    float4 nr = normalRoughness.Load(int3(pixel, 0));
    if (!(z < kSkyDepth) || z <= 0.0f || ReflectionWeight(nr.w) <= 0.0f || dot(nr.xyz, nr.xyz) <= 0.0f)
    {
        light[id.xy] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        directions[id.xy] = 0u;
        return;
    }
    gSkyTable = f.skyTable;
    float3 view;
    float3 p = PixelPosition(f, pixel, z, view);
    float3 n = normalize(nr.xyz);
    Basis basis = MakeBasis(n);
    float3 wo = ToLocal(basis, -view);
    wo.z = max(wo.z, 1e-4f);
    uint seed = Pcg(pixel.x + pixel.y * f.width ^ Pcg(f.frameIndex * 3u + 1u));
    float3 wi = nr.w < kMirrorRoughness ? float3(-wo.x, -wo.y, wo.z)
                                        : reflect(-wo, SampleGgxVndf(wo, ReflectionAlpha(nr.w), Rand2(seed)));
    if (wi.z <= 0.0f)
    {
        wi = float3(-wo.x, -wo.y, wo.z);
    }
    float3 direction = ToWorld(basis, wi);
    GiRay ray;
    float tStart;
    if (!TraceGiScreen(f, p, n, direction, pixel, PixelFootprint(f, z), Rand(seed), ray, tStart))
    {
        ray = TraceGiVoxels(f, p, n, direction, tStart);
    }
    light[id.xy] = float4(ray.radiance * kHalfLightScale, min(ray.distance, 65000.0f));
    directions[id.xy] = PackNormal(direction);
}
