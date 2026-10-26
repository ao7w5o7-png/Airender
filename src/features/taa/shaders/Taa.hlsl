// TAA's result (Taa.hlsli): the frame blended into its reprojected last result by `blend`, that history
// clipped to the box of this frame's fast history around the pixel (TaaFast.hlsl), widened by how
// much the fast history's mean strays from the true one given the frames' temporal deviation.

#include "Taa.hlsli"

// A still pixel's box is this much wider than `clip` deviations: the fast history's deviation is small
// next to the frames' (it averages them), and a tight box around it still took 1.5% of the sphere
// field's light (0.9% at 1.6x).
static const float kStillClip = 1.6f;

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    Texture2D<float4> fast = ResourceDescriptorHeap[g.fastResult];
    RWTexture2D<float4> result = ResourceDescriptorHeap[g.result];
    Neighbourhood n = Gather(id.xy);
    float3 output = n.current;
    if (g.history != kInvalidIndex && n.onScreen)
    {
        float3 m1 = 0.0f;
        float3 m2 = 0.0f;
        int2 last = int2(g.width, g.height) - 1;
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                float3 y = ToYCoCg(fast[uint2(clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), last))].rgb);
                m1 += y;
                m2 += y * y;
            }
        }
        float3 mean = m1 / 9.0f;
        float3 spatial = max(m2 / 9.0f - mean * mean, 0.0f);
        Texture2D<float4> history = ResourceDescriptorHeap[g.history];
        float3 past = SampleCatmullRom(history, n.previous, float2(g.width, g.height)).rgb;
        // The frames' variance (of luminance, for every channel) around the result's mean, as the
        // fast history's average of fastBlend leaves it: blend / (2 - blend) of it.
        float l = Luma(past);
        float temporal = max(fast[id.xy].a - l * l, 0.0f) * g.fastBlend / (2.0f - g.fastBlend);
        // Where the image moves, the fast history lags behind it for a few frames and would leave a
        // trail: the box then comes from this frame's own samples (and gives up some of the mean
        // while it moves), fully from half a pixel of motion on.
        float3 halfWidth = lerp(kStillClip * g.clip * sqrt(spatial + temporal), g.clip * n.deviation, n.moving);
        mean = lerp(mean, n.mean, n.moving);
        past = Clip(past, mean, halfWidth);
        output = lerp(max(past, 0.0f), n.current, g.blend);
    }
    result[id.xy] = float4(StochasticHalf(float4(output, 1.0f), id.xy).rgb, 1.0f);
}
