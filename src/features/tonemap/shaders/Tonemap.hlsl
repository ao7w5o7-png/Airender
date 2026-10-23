// Scene-linear exposed color -> display. Runs in place on the frame output.

#include "TonemapShared.h"

[[vk::push_constant]] ConstantBuffer<TonemapConstants> g : register(b0);

// ACES fitted RRT + ODT (Stephen Hill), input and output linear sRGB.
float3 Aces(float3 color) {
    const float3x3 input = float3x3(0.59719f, 0.35458f, 0.04823f, 0.07600f, 0.90834f, 0.01566f, 0.02840f, 0.13383f,
                                    0.83777f);
    const float3x3 output = float3x3(1.60475f, -0.53108f, -0.07367f, -0.10208f, 1.10813f, -0.00605f, -0.00327f,
                                     -0.07276f, 1.07602f);
    color = mul(input, color);
    float3 a = color * (color + 0.0245786f) - 0.000090537f;
    float3 b = color * (0.983729f * color + 0.4329510f) + 0.238081f;
    return saturate(mul(output, a / b));
}

// Khronos PBR Neutral tone mapper.
float3 Neutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) {
        return color;
    }
    const float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g2 = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return lerp(color, newPeak * float3(1.0f, 1.0f, 1.0f), g2);
}

float3 LinearToSrgb(float3 c) {
    c = saturate(c);
    return select(c <= 0.0031308f, c * 12.92f, 1.055f * pow(c, 1.0f / 2.4f) - 0.055f);
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= g.width || id.y >= g.height) {
        return;
    }
    RWTexture2D<float4> image = ResourceDescriptorHeap[g.image];
    float4 value = image[id.xy];
    float3 color = value.rgb;

    if ((g.flags & kFlagHighlightInvalid) != 0u) {
        if (any(isnan(color)) || any(isinf(color))) {
            image[id.xy] = float4(1.0f, 0.0f, 1.0f, 1.0f);
            return;
        }
        if (any(color < 0.0f)) {
            image[id.xy] = float4(0.0f, 1.0f, 1.0f, 1.0f);
            return;
        }
    }
    if ((g.flags & kFlagDisplayInput) == 0u) {
        switch (g.op) {
        case kOperatorAces: color = Aces(color); break;
        case kOperatorNeutral: color = Neutral(color); break;
        default: color = color / g.whitePoint; break;
        }
    }
    color = saturate(color);
    if ((g.flags & kFlagEncodeSrgb) != 0u) {
        color = LinearToSrgb(color);
    }
    image[id.xy] = float4(color, value.a);
}
