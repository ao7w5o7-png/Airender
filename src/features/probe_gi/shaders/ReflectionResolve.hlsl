// Probe GI: every smooth pixel's reflected light from the reflection rays around it (ReflectionTrace.hlsl), at full
// resolution: the 3 x 3 half resolution rays around, each a sample of this pixel's lobe towards the point it hit
// (its light as seen from here), weighed by this pixel's lobe density that way over the density the ray was drawn
// with (Stachowiak 2015) and by how well its pixel's surface fits this one's (plane, normal). Mirror-like pixels
// reuse rays without the density ratio, which is all but zero off the mirror direction.
//   GiConstants: input0 view depth, input1 normal + roughness, input2 the rays' light (Texture2D<float4>), input3
//   their directions (Texture2D<uint>), output the reflected light (RWTexture2D<float4>: rgb times kHalfLightScale,
//   a 1 where some ray fitted).

#include "GiCommon.hlsli"
#include "ReflectionCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kSkyDepth = 65000.0f;
static const float kFarHit = 60000.0f; // rays that went this far saw the sky: a direction, no point

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.width || id.y >= f.height)
    {
        return;
    }
    RWTexture2D<float4> output = ResourceDescriptorHeap[g.output];
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.input1];
    float z = depth.Load(int3(id.xy, 0));
    float4 nr = normalRoughness.Load(int3(id.xy, 0));
    if (!(z < kSkyDepth) || z <= 0.0f || ReflectionWeight(nr.w) <= 0.0f || dot(nr.xyz, nr.xyz) <= 0.0f)
    {
        output[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    Texture2D<float4> light = ResourceDescriptorHeap[g.input2];
    Texture2D<uint> directions = ResourceDescriptorHeap[g.input3];
    float3 view;
    float3 p = PixelPosition(f, id.xy, z, view);
    float3 n = normalize(nr.xyz);
    Basis basis = MakeBasis(n);
    float3 wo = ToLocal(basis, -view);
    wo.z = max(wo.z, 1e-4f);
    float alpha = ReflectionAlpha(nr.w);
    bool mirror = nr.w < kMirrorRoughness;
    float footprint = PixelFootprint(f, z);
    int2 half = int2((f.width + 1u) / 2u, (f.height + 1u) / 2u);
    int2 centre = int2(id.xy / 2u);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int2 q = centre + int2(dx, dy);
            if (any(q < 0) || any(q >= half))
            {
                continue;
            }
            float4 s = light.Load(int3(q, 0));
            if (s.w < 0.0f)
            {
                continue;
            }
            uint2 pixel = min(ReflectionPixel(uint2(q), f.frameIndex), uint2(f.width - 1u, f.height - 1u));
            float zq = depth.Load(int3(pixel, 0));
            float4 nrq = normalRoughness.Load(int3(pixel, 0));
            float3 viewQ;
            float3 pq = PixelPosition(f, pixel, zq, viewQ);
            float3 nq = normalize(nrq.xyz);
            float plane = abs(dot(pq - p, n));
            float w = saturate(1.0f - plane / (0.02f * z + 2.0f * footprint)) * pow(saturate(dot(n, nq)), 16.0f);
            if (w <= 0.0f)
            {
                continue;
            }
            float3 ray = UnpackNormal(directions.Load(int3(q, 0)));
            float3 toHit = s.w >= kFarHit ? ray : pq + ray * s.w - p;
            float3 l = ToLocal(basis, normalize(toHit));
            if (l.z <= 0.0f)
            {
                continue;
            }
            if (!mirror)
            {
                // This pixel's lobe density towards the hit over the density the ray was drawn with at its pixel.
                Basis basisQ = MakeBasis(nq);
                float3 woQ = ToLocal(basisQ, -viewQ);
                woQ.z = max(woQ.z, 1e-4f);
                float drawn = GgxVndfReflectionPdf(woQ, ToLocal(basisQ, ray), ReflectionAlpha(nrq.w));
                w *= GgxVndfReflectionPdf(wo, l, alpha) / max(drawn, 1e-6f);
            }
            sum += s.xyz * w;
            weight += w;
        }
    }
    output[id.xy] = weight > 0.0f ? float4(sum / weight, 1.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
}
