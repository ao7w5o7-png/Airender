// Accumulation buffer -> frame output: average and exposure (radiance views), or the display
// encoding of a debug view. Tonemapping is the separate `tonemap` feature.

#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<ResolveConstants> g : register(b0);

float3 SrgbToLinear(float3 c) {
    c = saturate(c);
    return select(c <= 0.04045f, c / 12.92f, pow((c + 0.055f) / 1.055f, 2.4f));
}

// Turbo colormap (Mikhailov 2019), polynomial fit. x in [0, 1], returns sRGB-encoded color.
float3 Turbo(float x) {
    x = saturate(x);
    const float4 kr = float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
    const float4 kg = float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
    const float4 kb = float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
    const float2 kr2 = float2(-152.94239396f, 59.28637943f);
    const float2 kg2 = float2(4.27729857f, 2.82956604f);
    const float2 kb2 = float2(-89.90310912f, 27.34824973f);
    float4 v4 = float4(1.0f, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;
    return float3(dot(v4, kr) + dot(v2, kr2), dot(v4, kg) + dot(v2, kg2), dot(v4, kb) + dot(v2, kb2));
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID) {
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData f = frames[g.frameSlot];
    if (id.x >= f.width || id.y >= f.height) {
        return;
    }
    Texture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
    RWTexture2D<float4> output = ResourceDescriptorHeap[g.output];

    float4 sum = accumulation[id.xy];
    float3 value = sum.a > 0.0f ? sum.rgb / sum.a : 0.0f;
    switch (f.debugView) {
    case kDebugNone:
    case kDebugDirect:
    case kDebugIndirect: value *= f.environment.w; break; // exposure: radiance -> scene-linear
    case kDebugBaseColor: break;                           // already linear reflectance
    case kDebugIntersections:
    case kDebugBounces: value = SrgbToLinear(Turbo(value.r)); break;
    default: value = SrgbToLinear(value); break; // normals and scalars are shown as raw values
    }
    output[id.xy] = float4(value, 1.0f);
}
