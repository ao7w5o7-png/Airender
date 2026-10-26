// TAA's fast history (Taa.hlsli): the frame blended into its reprojected last value by fastBlend. A
// still pixel's history is clipped only to a wide box around the frame's own samples, widened by the
// pixel's temporal deviation, which leaves the skewed noise of a still image alone; a moving one's as
// tightly as the result's would be, so that what moved away leaves no trail for the result to follow
// once the camera stops. Its alpha keeps the mean of the frames' squared luminance over the result's
// longer window (blend), for their temporal deviation; it starts over where the colour was clipped.

#include "Taa.hlsli"

static const float kFastClip = 3.0f; // deviations of a still pixel's samples

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    RWTexture2D<float4> fast = ResourceDescriptorHeap[g.fastResult];
    Neighbourhood n = Gather(id.xy);
    float luma = Luma(n.current);
    float4 output = float4(n.current, luma * luma);
    if (g.fastHistory != kInvalidIndex && n.onScreen)
    {
        Texture2D<float4> history = ResourceDescriptorHeap[g.fastHistory];
        float4 past = SampleCatmullRom(history, n.previous, float2(g.width, g.height));
        // The box takes the pixel's temporal variance too: a pixel that sees a bright sample now and
        // then has a quiet neighbourhood in most frames, which would clip that sample's light away.
        float l = Luma(past.rgb);
        float temporal = max(past.a - l * l, 0.0f);
        float3 still = kFastClip * sqrt(n.deviation * n.deviation + temporal) + 1e-4f * n.mean.x;
        float3 clipped = Clip(past.rgb, n.mean, lerp(still, g.clip * n.deviation, n.moving));
        bool kept = all(clipped == past.rgb);
        output.rgb = lerp(max(clipped, 0.0f), n.current, g.fastBlend);
        output.a = kept ? lerp(past.a, luma * luma, g.blend) : luma * luma;
    }
    fast[id.xy] = StochasticHalf(output, id.xy);
}
