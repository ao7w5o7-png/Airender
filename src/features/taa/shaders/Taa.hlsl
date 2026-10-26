// Temporal anti-aliasing and upsampling: each output pixel gathers the frame's 3 x 3 render samples
// around it (they lie at pixel + 0.5 + jitter) with a Gaussian of their distance, follows its motion
// back to the previous result (Catmull-Rom), clips that history to the neighbourhood's colour box in
// YCoCg (mean +- clip standard deviations) and blends the two. Its input is denoised: weighing samples by
// 1 / (1 + luminance) against fireflies (Karis 2014) took 5% of a small bright emitter's light.

#include "air/renderer/shader/SceneShared.h"
#include "TaaShared.h"

[[vk::push_constant]] ConstantBuffer<TaaConstants> g : register(b0);

SamplerState gLinearClamp : register(s0);

float Luma(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 ToYCoCg(float3 c)
{
    return float3(0.25f * c.r + 0.5f * c.g + 0.25f * c.b, 0.5f * c.r - 0.5f * c.b, -0.25f * c.r + 0.5f * c.g - 0.25f * c.b);
}

float3 FromYCoCg(float3 c)
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Catmull-Rom filtering of the history from five bilinear taps (the corners dropped).
float3 SampleCatmullRom(Texture2D<float4> t, float2 uv, float2 size)
{
    float2 position = uv * size;
    float2 center = floor(position - 0.5f) + 0.5f;
    float2 f = position - center;
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);
    float2 w12 = w1 + w2;
    float2 tc0 = (center - 1.0f) / size;
    float2 tc3 = (center + 2.0f) / size;
    float2 tc12 = (center + w2 / w12) / size;
    float3 c = t.SampleLevel(gLinearClamp, float2(tc12.x, tc0.y), 0.0f).rgb * (w12.x * w0.y) +
               t.SampleLevel(gLinearClamp, float2(tc0.x, tc12.y), 0.0f).rgb * (w0.x * w12.y) +
               t.SampleLevel(gLinearClamp, float2(tc12.x, tc12.y), 0.0f).rgb * (w12.x * w12.y) +
               t.SampleLevel(gLinearClamp, float2(tc3.x, tc12.y), 0.0f).rgb * (w3.x * w12.y) +
               t.SampleLevel(gLinearClamp, float2(tc12.x, tc3.y), 0.0f).rgb * (w12.x * w3.y);
    float weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(c / weight, 0.0f);
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    Texture2D<float4> color = ResourceDescriptorHeap[g.color];
    Texture2D<float2> motion = ResourceDescriptorHeap[g.motion];
    RWTexture2D<float4> result = ResourceDescriptorHeap[g.result];
    float2 renderSize = float2(g.renderWidth, g.renderHeight);
    float2 uv = (float2(id.xy) + 0.5f) / float2(g.width, g.height);
    float2 at = uv * renderSize; // in render pixels
    int2 center = int2(floor(at));
    int2 last = int2(renderSize) - 1;
    float2 jitter = float2(g.jitterX, g.jitterY);

    float3 sum = 0.0f;
    float weights = 0.0f;
    float3 m1 = 0.0f;
    float3 m2 = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int2 q = clamp(center + int2(dx, dy), int2(0, 0), last);
            float3 c = max(color[uint2(q)].rgb, 0.0f);
            float2 d = at - (float2(q) + 0.5f + jitter);
            float w = exp(-2.29f * dot(d, d));
            sum += c * w;
            weights += w;
            float3 y = ToYCoCg(c);
            m1 += y;
            m2 += y * y;
        }
    }
    float3 current = weights > 0.0f ? sum / weights : max(color[uint2(clamp(center, int2(0, 0), last))].rgb, 0.0f);
    float3 output = current;
    float2 previous = uv + motion[uint2(clamp(center, int2(0, 0), last))] / renderSize;
    if (g.history != kInvalidIndex && all(previous >= 0.0f) && all(previous <= 1.0f))
    {
        Texture2D<float4> history = ResourceDescriptorHeap[g.history];
        float3 past = SampleCatmullRom(history, previous, float2(g.width, g.height));
        float3 mean = m1 / 9.0f;
        float3 deviation = sqrt(max(m2 / 9.0f - mean * mean, 0.0f));
        past = FromYCoCg(clamp(ToYCoCg(past), mean - g.clip * deviation, mean + g.clip * deviation));
        output = lerp(max(past, 0.0f), current, g.blend);
    }
    result[id.xy] = float4(output, 1.0f);
}
