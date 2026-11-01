// Probe GI screen probes over frames (ProbeGiShared.h): each probe's harmonics blended with those of last frame's
// probe where its surface was (its position projected with last frame's camera: that tile's probe if it lay on
// this probe's plane and faced its way, else the closest such probe of the 3 x 3 tiles there). Over the first
// frames of a history 1 / n, then at least GiFrame::probeBlend: a probe's 64 rays a frame become hundreds, and
// the probe-sized blotches its noise made stop flickering. A probe without such a probe starts over, and so do the
// adaptive probes (ProbeAdapt.hlsl). After the rays' spatial filter (ProbeRayFilter.hlsl); the harmonics' own filter,
// when on (filterRadius), then averages accumulated probes. One thread per probe.
//   GiConstants: input3 probes, input4 this frame's harmonics, input5 last frame's probes, input6 last
//   frame's accumulated harmonics (kInvalidIndex: no history; normalIrradiance.w: frames of history), output
//   (RWStructuredBuffer<ProbeSh>).

#include "VoxelTrace.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kMaxProbeHistory = 32.0f;

ProbeSh Scale(ProbeSh sh, float w)
{
    sh.c0 *= w;
    sh.c1 *= w;
    sh.c2 *= w;
    sh.c3 *= w;
    sh.c4 *= w;
    sh.c5 *= w;
    sh.c6 *= w;
    sh.normalIrradiance *= w;
    return sh;
}

ProbeSh Add(ProbeSh a, ProbeSh b)
{
    a.c0 += b.c0;
    a.c1 += b.c1;
    a.c2 += b.c2;
    a.c3 += b.c3;
    a.c4 += b.c4;
    a.c5 += b.c5;
    a.c6 += b.c6;
    a.normalIrradiance += b.normalIrradiance;
    return a;
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.probesX || id.y >= f.probesY + f.adaptiveRows)
    {
        return;
    }
    StructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.input3];
    StructuredBuffer<ProbeSh> input = ResourceDescriptorHeap[g.input4];
    RWStructuredBuffer<ProbeSh> output = ResourceDescriptorHeap[g.output];
    uint index = id.y * f.probesX + id.x;
    ScreenProbe centre = probes[index];
    ProbeSh now = input[index];
    now.normalIrradiance.w = 0.0f;
    // Adaptive probes (the rows after the tiles') start over every frame: they move with what the tiles miss.
    if (centre.depth <= 0.0f || (f.flags & kGiHistory) == 0u || g.input6 == kInvalidIndex || id.y >= f.probesY)
    {
        now.normalIrradiance.w = 1.0f;
        output[index] = now;
        return;
    }
    StructuredBuffer<ScreenProbe> pastProbes = ResourceDescriptorHeap[g.input5];
    StructuredBuffer<ProbeSh> past = ResourceDescriptorHeap[g.input6];
    float3 n = UnpackNormal(centre.normal);
    float2 size = float2(f.width, f.height);
    float2 pixel = CameraProject(f.previousCamera, centre.position, size).xy;
    int2 tile = int2(floor(pixel / float(f.probeTile)));
    // The tile's own probe of last frame when it lay on this surface (the tiles keep their histories apart:
    // blending a tile's neighbours in, as their positions jitter, would mix their light into it a little
    // differently every frame, noise wherever the light changes across tiles), else the closest fitting
    // probe around (after a move, at an edge).
    int best = -1;
    float bestScore = 0.0f;
    for (int k = 0; k < 9; ++k)
    {
        int2 q = tile + int2(k % 3 - 1, k / 3 - 1);
        if (any(q < 0) || q.x >= int(f.probesX) || q.y >= int(f.probesY))
        {
            continue;
        }
        uint other = uint(q.y) * f.probesX + uint(q.x);
        ScreenProbe p = pastProbes[other];
        if (p.depth <= 0.0f)
        {
            continue;
        }
        float3 offset = p.position - centre.position;
        float plane = abs(dot(offset, n));
        float tolerance = 0.05f * length(offset) + 2.0f * centre.footprint;
        float fit = saturate(1.0f - plane / tolerance) * saturate((dot(n, UnpackNormal(p.normal)) - 0.9f) * 10.0f);
        if (fit <= 0.25f)
        {
            continue;
        }
        // Distance on the screen, in tiles (in metres a surface seen at a grazing angle would put its own
        // neighbours many tiles away): last frame's probe at its pixel, this one where it was then.
        float2 pastPixel = float2(p.pixel & 0xFFFFu, p.pixel >> 16) + 0.5f + f.previousJitter;
        float d = length(pastPixel - pixel) / float(f.probeTile);
        float score = (k == 4 ? 2.0f : 1.0f) / (1.0f + d);
        if (score > bestScore)
        {
            bestScore = score;
            best = int(other);
        }
    }
    ProbeSh history = (ProbeSh)0;
    float frames = 0.0f;
    if (best >= 0)
    {
        // Its irradiance at its own normal, moved to this probe's normal as the gather does for a pixel's.
        history = past[uint(best)];
        float3 m = UnpackNormal(pastProbes[uint(best)].normal);
        history.normalIrradiance.xyz =
            max(history.normalIrradiance.xyz + ShEvaluate(history, n) - ShEvaluate(history, m), 0.0f);
        frames = history.normalIrradiance.w;
    }
    frames = min(frames + 1.0f, kMaxProbeHistory);
    float blend = max(1.0f / frames, f.probeBlend);
    ProbeSh result = Add(Scale(history, 1.0f - blend), Scale(now, blend));
    if (frames <= 1.0f)
    {
        result = now;
    }
    result.valid = now.valid;
    result.normalIrradiance.w = frames;
    output[index] = result;
}
