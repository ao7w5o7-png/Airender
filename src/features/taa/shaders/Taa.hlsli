// Temporal anti-aliasing and upsampling, shared by its two passes (docs/architecture.md §8.14).
//
// Each output pixel gathers the frame's 3 x 3 render samples around it (they lie at pixel + 0.5 +
// jitter) with a Gaussian of their distance, follows its motion back to the previous result
// (Catmull-Rom) and blends the two. The history is clipped against a box in YCoCg so that it follows
// changes of the image; where the box comes from decides whether the mean survives:
//
// - Around the frame's own samples (mean +- clip deviations of the 3 x 3), the box is drawn from noise
//   whose distribution is skewed (most samples are dark, a few bright): the neighbourhood's mean is
//   then below the true mean most of the time and its deviation small, so the history (which holds
//   the true mean) is clipped down in most frames, and the image is darker than the frames it averages
//   (5% for the sphere field's sunlight in real time).
// - So TaaFast.hlsl first keeps a fast history (a few frames, clipped only far out), and Taa.hlsl clips
//   the result's history to the fast history's 3 x 3 neighbourhood, widened by the frames' own temporal
//   deviation (the result keeps the mean of their squared luminance): averaged over frames, the box's
//   centre is no longer skewed, and a pixel whose frames are noisy gets the room its noise needs.
// Its input is denoised: weighing samples by 1 / (1 + luminance) against fireflies (Karis 2014) took
// 5% of a small bright emitter's light.

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

// Catmull-Rom filtering of a history from five bilinear taps (the corners dropped).
float4 SampleCatmullRomUnclamped(Texture2D<float4> t, float2 uv, float2 size)
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
    float4 c = t.SampleLevel(gLinearClamp, float2(tc12.x, tc0.y), 0.0f) * (w12.x * w0.y) +
               t.SampleLevel(gLinearClamp, float2(tc0.x, tc12.y), 0.0f) * (w0.x * w12.y) +
               t.SampleLevel(gLinearClamp, float2(tc12.x, tc12.y), 0.0f) * (w12.x * w12.y) +
               t.SampleLevel(gLinearClamp, float2(tc3.x, tc12.y), 0.0f) * (w3.x * w12.y) +
               t.SampleLevel(gLinearClamp, float2(tc12.x, tc3.y), 0.0f) * (w12.x * w3.y);
    float weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(c / weight, 0.0f);
}

// The same kept within the 2 x 2 texels around uv. Resampled every frame, a history's negative lobes
// ring around its bright texels; kept within them, the sphere field's real-time frame kept 1.3% more of
// its light.
float4 SampleCatmullRom(Texture2D<float4> t, float2 uv, float2 size)
{
    int2 corner = int2(floor(uv * size - 0.5f));
    int2 last = int2(size) - 1;
    float4 t00 = t[uint2(clamp(corner, int2(0, 0), last))];
    float4 t10 = t[uint2(clamp(corner + int2(1, 0), int2(0, 0), last))];
    float4 t01 = t[uint2(clamp(corner + int2(0, 1), int2(0, 0), last))];
    float4 t11 = t[uint2(clamp(corner + int2(1, 1), int2(0, 0), last))];
    return clamp(SampleCatmullRomUnclamped(t, uv, size), min(min(t00, t10), min(t01, t11)),
                 max(max(t00, t10), max(t01, t11)));
}

// The frame at an output pixel: its 3 x 3 render samples weighed by a Gaussian of their distance, and
// the first two moments of the samples in YCoCg.
struct Neighbourhood
{
    float3 current;
    float3 mean;
    float3 deviation;
    float2 previous; // where the pixel was in the last frame (uv)
    bool onScreen;
    float moving; // 0: still, 1: moved half a render pixel or more since the last frame
};

Neighbourhood Gather(uint2 id)
{
    Texture2D<float4> color = ResourceDescriptorHeap[g.color];
    Texture2D<float2> motion = ResourceDescriptorHeap[g.motion];
    float2 renderSize = float2(g.renderWidth, g.renderHeight);
    float2 uv = (float2(id) + 0.5f) / float2(g.width, g.height);
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
    Neighbourhood n;
    n.current = weights > 0.0f ? sum / weights : max(color[uint2(clamp(center, int2(0, 0), last))].rgb, 0.0f);
    n.mean = m1 / 9.0f;
    n.deviation = sqrt(max(m2 / 9.0f - n.mean * n.mean, 0.0f));
    float2 moved = motion[uint2(clamp(center, int2(0, 0), last))];
    n.previous = uv + moved / renderSize;
    n.onScreen = all(n.previous >= 0.0f) && all(n.previous <= 1.0f);
    n.moving = max(saturate(2.0f * length(moved)), g.cameraMoved);
    return n;
}

float3 Clip(float3 c, float3 mean, float3 halfWidth)
{
    return FromYCoCg(clamp(ToYCoCg(c), mean - halfWidth, mean + halfWidth));
}

// Rounds to a half-precision value at random, in proportion to the distance to its two neighbours, so
// that storing it is exact and the expected value unchanged. Converting to half on store may truncate
// (towards zero, as this GPU does): an average over frames that rounds its every update down settles
// half a step times the number of frames it averages below its inputs (0.2% for the result's 1 / 0.1).
float4 StochasticHalf(float4 v, uint2 pixel)
{
    uint h = (pixel.x * 0x8da6b343u) ^ (pixel.y * 0xd8163841u) ^ (g.frame * 0xcb1ab31fu);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    float4 u = float4(h & 0xFFu, (h >> 8) & 0xFFu, (h >> 16) & 0xFFu, h >> 24) * (1.0f / 256.0f) + (0.5f / 256.0f);
    v = clamp(v, 0.0f, 65504.0f);
    // The spacing of halves around v: 2^(exponent - 10), at least that of the subnormals (2^-24).
    float4 step = max(asfloat(asuint(v) & 0x7F800000u) * (1.0f / 1024.0f), 5.9604645e-8f);
    return min(floor(v / step + u) * step, 65504.0f);
}
