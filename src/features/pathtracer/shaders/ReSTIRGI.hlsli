// ReSTIR GI: the indirect light at each pixel's first surface (every path with two or more bounces)
// resampled across pixels and frames (Ouyang et al. 2021), in the path-space form of ReSTIR PT (Lin
// et al. 2022) with the reconnection shift at the second vertex x2. PathTracer.Trace leaves that
// light out (kFlagReSTIRGI) and writes each pixel's initial reservoir: one of the two candidate
// paths its first sample found at x2 (NEE there, or the BSDF continuation with all it gathered),
// chosen by RIS (Integrator.hlsli, GiPath).
//
// A sample moves to another pixel by connecting that pixel's first surface x1 to the same x2 and
// keeping the rest of the path, which in area measure at x2 times solid angle at x2 is the
// identity. Its value there is evaluated in full: x1's BSDF towards x2, the geometry term, x2's BSDF
// between the new incoming and the kept outgoing direction, and the MIS weight between NEE and BSDF
// sampling at x2, whose BSDF pdf depends on the incoming direction. Only the radiance arriving at x2
// is reused as it is, and it does not depend on x1, so no Lambertian assumption is made and the reuse
// stays unbiased.
//   Temporal  generalized balance heuristic between the initial reservoir and last frame's at the
//             reprojected pixel.
//   Spatial   defensive pairwise MIS with neighbours; shades every candidate by its resampling
//             weight and keeps the chosen one as next frame's history.
// Visibility works as in ReSTIR DI: a reservoir only holds samples whose x2 its own surface sees
// (the initial one by construction), reused samples get this pixel's visibility in the target, and
// with kReSTIRUnbiased the MIS weights trace the other surfaces' visibility.
// Reconnecting is only worth it between rough vertices: through a near-mirror x1 or x2 the value of a
// moved path swings between nothing and a spike, which reuse turns into persistent fireflies. So the
// shift is defined only when x1 on both sides and x2 are at least kGiMinRoughness rough (Lin et al.
// 2026 keep such a threshold at the vertex before the reconnection too). Pixels with a smoother first
// surface keep their own sample, neighbours are chosen among rough pixels, and a sample with a
// smooth x2 stays where it was found, its MIS weight counting no other domain. The criteria depend on
// surfaces, not on random choices, so the reuse stays unbiased.
#ifndef AIR_PT_RESTIR_GI_HLSLI
#define AIR_PT_RESTIR_GI_HLSLI

#include "ReSTIRCommon.hlsli"

AIR_CONST float kGiMinRoughness = 0.25f; // perceptual roughness at x1 and x2 for the shift to exist

struct GiSample
{
    uint instance; // x2; kInvalidIndex: none
    uint primitive;
    float2 barycentrics;
    float3 direction;
    uint strategy;
    float3 radianceMis;
    float3 radianceFixed;
    float lightPdf;
    float W;
    float M;
};

GiSample UnpackGi(GiReservoir p)
{
    GiSample s;
    s.instance = p.instance;
    s.primitive = p.primitive;
    s.barycentrics = p.barycentrics;
    s.direction = p.direction;
    s.strategy = p.strategyAndConfidence & 1u;
    s.radianceMis = p.radianceMis;
    s.radianceFixed = p.radianceFixed;
    s.lightPdf = p.lightPdf;
    s.W = p.instance != kInvalidIndex ? p.weight : 0.0f;
    s.M = float((p.strategyAndConfidence >> 8) & 0xFFu);
    return s;
}

GiReservoir PackGi(GiSample s)
{
    GiReservoir p;
    p.instance = s.W > 0.0f ? s.instance : kInvalidIndex;
    p.primitive = s.primitive;
    p.barycentrics = s.barycentrics;
    p.direction = s.direction;
    p.lightPdf = s.lightPdf;
    p.radianceMis = s.radianceMis;
    p.weight = s.W;
    p.radianceFixed = s.radianceFixed;
    p.strategyAndConfidence = (s.strategy & 1u) | (min(uint(s.M + 0.5f), 255u) << 8);
    return p;
}

// x2 as the sample's path left it: its surface, normals on the side its outgoing direction leaves.
struct SecondaryVertex
{
    float3 position;
    float3 ng;
    float3 ns;
    BsdfMaterial material;
};

SecondaryVertex LoadSecondaryVertex(GiSample x, IntegratorParams params)
{
    HitInfo hit;
    hit.hit = true;
    hit.t = 0.0f;
    hit.instance = x.instance;
    hit.primitive = x.primitive;
    hit.barycentrics = x.barycentrics;
    // LoadSurface orients the normals against a ray; the path's side is the one it leaves from.
    Surface s = LoadSurface(hit, float3(0.0f, 0.0f, 0.0f), params, kSurfaceCoarse);
    if (dot(s.geometricNormal, x.direction) < 0.0f)
    {
        s.geometricNormal = -s.geometricNormal;
        s.shadingNormal = -s.shadingNormal;
    }
    SecondaryVertex v;
    v.position = s.position;
    v.ng = s.geometricNormal;
    v.ns = s.shadingNormal;
    v.material = s.material;
    return v;
}

// A sample's integrand at receiver r without the visibility between them (area measure at x2,
// solid angle at x2), and the segment r -> x2.
struct GiPoint
{
    float3 contribution;
    float3 diffuse; // the part of it the receiver's diffuse base reflects
    float3 wi; // from r towards x2
    float dist;
};

GiPoint EvaluateGi(Receiver r, SecondaryVertex v, GiSample x)
{
    GiPoint p;
    p.contribution = float3(0.0f, 0.0f, 0.0f);
    p.diffuse = float3(0.0f, 0.0f, 0.0f);
    p.wi = float3(0.0f, 0.0f, 1.0f);
    p.dist = 0.0f;
    float3 toX2 = v.position - r.origin;
    float dist2 = dot(toX2, toX2);
    if (!(dist2 > 0.0f))
    {
        return p;
    }
    p.dist = sqrt(dist2);
    p.wi = toX2 / p.dist;
    float3 wo2 = -p.wi; // at x2, towards r
    float cos2 = dot(wo2, v.ng);
    if (dot(p.wi, r.ng) <= 0.0f || !(cos2 > 0.0f))
    {
        return p; // x2 below r's horizon, or r on the other side of x2 than the path
    }
    BsdfEval e1 = EvaluateBsdf(r.bsdf, ToLocal(r.frame, p.wi));
    // x2's shading frame as TracePath builds it for this incoming direction.
    float3 ns2 = dot(v.ns, wo2) > 0.0f ? v.ns : v.ng;
    Frame frame2 = MakeFrame(ns2);
    BsdfEval e2 = EvaluateBsdf(v.material, ToLocal(frame2, wo2), ToLocal(frame2, x.direction));
    float mis = 1.0f;
    if (x.strategy == 0u)
    {
        mis = PowerHeuristic(x.lightPdf, e2.pdf);
    }
    else if (x.lightPdf > 0.0f)
    {
        mis = PowerHeuristic(e2.pdf, x.lightPdf);
    }
    float3 arriving = (cos2 / dist2) * e2.fCos * (x.radianceMis * mis + x.radianceFixed);
    p.contribution = e1.fCos * arriving;
    p.diffuse = e1.fCosDiffuse * arriving;
    return p;
}

// Whether the shift can move a sample to other pixels: x2 must be rough enough.
bool Shifts(SecondaryVertex v)
{
    return v.material.roughness >= kGiMinRoughness;
}

bool RoughPrimaryHit(PrimaryHit h)
{
    return DecodeRoughness(h.normalAndRoughness) >= kGiMinRoughness;
}

// Whether r sees x2 (the shadow ray stops just off x2's surface).
bool VisibleGi(Receiver r, SecondaryVertex v)
{
    float3 target = OffsetRayOrigin(v.position, v.ng);
    float3 toTarget = target - r.origin;
    float dist = length(toTarget);
    return dist > 0.0f && TraceVisible(r.origin, toTarget / dist, dist);
}

// p̂ of another surface's reservoirs at x: zero where that surface does not see x2 when
// traceVisibility, else assumed visible.
float DomainTargetGi(Receiver r, SecondaryVertex v, GiSample x, bool traceVisibility)
{
    float target = Luminance(EvaluateGi(r, v, x).contribution);
    if (traceVisibility && target > 0.0f && !VisibleGi(r, v))
    {
        target = 0.0f;
    }
    return target;
}

// Weighted reservoir sampling over inputs identified by an index (samples are too large to copy
// along): keeps one with probability w / sum(w).
struct GiChoice
{
    float targetPdf; // p̂ at this pixel of the chosen input
    float weightSum;
    uint source; // reservoir index the chosen input came from
    bool chosen;
};

GiChoice MakeGiChoice()
{
    GiChoice c;
    c.targetPdf = 0.0f;
    c.weightSum = 0.0f;
    c.source = 0u;
    c.chosen = false;
    return c;
}

void StreamGi(inout GiChoice c, float targetPdf, float w, uint source, inout uint seed)
{
    if (!(w > 0.0f)) // also rejects NaN
    {
        return;
    }
    c.weightSum += w;
    if (Rand(seed) * c.weightSum < w)
    {
        c.targetPdf = targetPdf;
        c.source = source;
        c.chosen = true;
    }
}

// The chosen input with W = sum(w) / p̂.
GiSample FinishGi(GiChoice c, GiSample chosen, float confidence)
{
    GiSample r = chosen;
    r.W = c.chosen && c.targetPdf > 0.0f ? c.weightSum / c.targetPdf : 0.0f;
    r.M = confidence;
    return r;
}

// --- Passes --------------------------------------------------------------------------------------

void GiTemporalPass(uint2 pixel)
{
    FrameData f = LoadFrame();
    if (pixel.x >= f.width || pixel.y >= f.height)
    {
        return;
    }
    uint index = pixel.y * f.width + pixel.x;
    StructuredBuffer<GiReservoir> initial = ResourceDescriptorHeap[g.input];
    RWStructuredBuffer<GiReservoir> output = ResourceDescriptorHeap[g.output];
    GiSample canonical = UnpackGi(initial[index]);
    PrimaryHit hit = LoadPrimaryHit(g.primaryHits, index);
    if (hit.instance == kInvalidIndex || (g.flags & kReSTIRTemporal) == 0u)
    {
        output[index] = PackGi(canonical);
        return;
    }
    IntegratorParams params = MakeParams(f);
    Receiver r = LoadReceiver(hit, f.camera.position.xyz, params);
    // The pixel last frame's camera saw this point in, if its surface there is this one.
    float2 size = float2(float(f.width), float(f.height));
    float3 previousEye = f.previousCamera.position.xyz;
    float2 previousPixel = floor(CameraProject(f.previousCamera, r.position, 1.0f, size));
    if (!all(previousPixel >= 0.0f) || !all(previousPixel < size))
    {
        output[index] = PackGi(canonical);
        return;
    }
    uint previousIndex = uint(previousPixel.y) * f.width + uint(previousPixel.x);
    PrimaryHit previousHit = LoadPrimaryHit(g.previousPrimaryHits, previousIndex);
    if (!SimilarSurface(hit, previousHit, length(r.position - previousEye)) || !RoughPrimaryHit(hit) ||
        !RoughPrimaryHit(previousHit))
    {
        output[index] = PackGi(canonical);
        return;
    }
    Receiver rt = LoadReceiver(previousHit, previousEye, params);
    StructuredBuffer<GiReservoir> previous = ResourceDescriptorHeap[g.previous];
    GiSample t = UnpackGi(previous[previousIndex]);
    t.M = min(t.M, g.maxConfidence);
    bool unbiased = (g.flags & kReSTIRUnbiased) != 0u;
    uint seed = StreamSeed(pixel, f.frameIndex, 5u);
    GiChoice c = MakeGiChoice();
    // Generalized balance heuristic between this pixel's domain and last frame's. The initial
    // sample's x2 was found by a ray from this surface, so it is visible.
    if (canonical.W > 0.0f)
    {
        SecondaryVertex v = LoadSecondaryVertex(canonical, params);
        float pc = Luminance(EvaluateGi(r, v, canonical).contribution);
        float pt = Shifts(v) ? DomainTargetGi(rt, v, canonical, unbiased) : 0.0f;
        float m = canonical.M * pc / (canonical.M * pc + t.M * pt);
        StreamGi(c, pc, m * pc * canonical.W, 0u, seed);
    }
    SecondaryVertex vt = (SecondaryVertex)0;
    if (t.W > 0.0f)
    {
        vt = LoadSecondaryVertex(t, params);
    }
    if (t.W > 0.0f && Shifts(vt))
    {
        SecondaryVertex v = vt;
        float pc = Luminance(EvaluateGi(r, v, t).contribution);
        float pt = Luminance(EvaluateGi(rt, v, t).contribution); // its surface sees it: W would be 0 otherwise
        float m = t.M * pt / (canonical.M * pc + t.M * pt);
        StreamGi(c, pc, m * pc * t.W, 1u, seed);
    }
    GiSample chosen = canonical; // (HLSL has no conditional operator on structs)
    if (c.source == 1u)
    {
        chosen = t;
    }
    GiSample result = FinishGi(c, chosen, canonical.M + t.M);
    // Last frame's sample needs a ray from here (visibility reuse).
    if (result.W > 0.0f && c.source == 1u && !VisibleGi(r, LoadSecondaryVertex(result, params)))
    {
        result.W = 0.0f;
    }
    output[index] = PackGi(result);
}

void GiSpatialPass(uint2 pixel)
{
    FrameData f = LoadFrame();
    if (pixel.x >= f.width || pixel.y >= f.height)
    {
        return;
    }
    uint index = pixel.y * f.width + pixel.x;
    StructuredBuffer<GiReservoir> temporal = ResourceDescriptorHeap[g.input];
    RWStructuredBuffer<GiReservoir> history = ResourceDescriptorHeap[g.output];
    GiSample canonical = UnpackGi(temporal[index]);
    PrimaryHit hit = LoadPrimaryHit(g.primaryHits, index);
    if (hit.instance == kInvalidIndex)
    {
        history[index] = PackGi(canonical);
        return;
    }
    IntegratorParams params = MakeParams(f);
    float3 eye = f.camera.position.xyz;
    Receiver r = LoadReceiver(hit, eye, params);
    bool unbiased = (g.flags & kReSTIRUnbiased) != 0u;

    // Neighbours and the sum of their confidences (the MIS weights need it before any of them).
    uint draws = min(canonical.M <= 1.0f ? g.disocclusionSamples : g.spatialSamples, kReSTIRMaxSpatialSamples);
    if (!RoughPrimaryHit(hit))
    {
        draws = 0u; // a smooth first surface keeps its own sample
    }
    uint firstDraw = StreamSeed(pixel, f.frameIndex, 6u);
    uint draw = firstDraw;
    float neighbourConfidence = 0.0f;
    for (uint i = 0u; i < draws; ++i)
    {
        uint neighbour;
        if (NextNeighbour(draw, pixel, hit, f, kGiMinRoughness, neighbour))
        {
            neighbourConfidence += UnpackGi(temporal[neighbour]).M;
        }
    }

    // Defensive pairwise MIS with this pixel's visibility in the target, and shading by every
    // candidate's resampling weight, as in ReSTIR DI's spatial pass.
    float cc = canonical.M;
    float total = cc + neighbourConfidence;
    SecondaryVertex vc = (SecondaryVertex)0;
    GiPoint canonicalPoint = (GiPoint)0;
    float pc = 0.0f;
    if (canonical.W > 0.0f) // its x2 is visible from here: zeroed otherwise
    {
        vc = LoadSecondaryVertex(canonical, params);
        canonicalPoint = EvaluateGi(r, vc, canonical);
        pc = Luminance(canonicalPoint.contribution);
    }
    bool canonicalShifts = Shifts(vc);
    float canonicalMis = cc / total;
    float3 indirect = float3(0.0f, 0.0f, 0.0f);
    float3 indirectDiffuse = float3(0.0f, 0.0f, 0.0f);
    uint seed = StreamSeed(pixel, f.frameIndex, 7u);
    GiChoice c = MakeGiChoice();
    draw = firstDraw;
    for (uint j = 0u; j < draws; ++j)
    {
        uint neighbour;
        if (!NextNeighbour(draw, pixel, hit, f, kGiMinRoughness, neighbour))
        {
            continue;
        }
        Receiver rn = LoadReceiver(LoadPrimaryHit(g.primaryHits, neighbour), eye, params);
        GiSample n = UnpackGi(temporal[neighbour]);
        if (pc > 0.0f)
        {
            float pn = canonicalShifts ? DomainTargetGi(rn, vc, canonical, unbiased) : 0.0f;
            canonicalMis += (n.M / total) * cc * pc / (neighbourConfidence * pn + cc * pc);
        }
        if (n.W > 0.0f)
        {
            SecondaryVertex v = LoadSecondaryVertex(n, params);
            if (!Shifts(v))
            {
                continue;
            }
            GiPoint here = EvaluateGi(r, v, n);
            float pq = Luminance(here.contribution);
            if (pq > 0.0f && VisibleGi(r, v))
            {
                float pn = Luminance(EvaluateGi(rn, v, n).contribution); // its surface sees x2: W would be 0 otherwise
                float m = (neighbourConfidence / total) * n.M * pn / (neighbourConfidence * pn + cc * pq);
                StreamGi(c, pq, m * pq * n.W, neighbour, seed);
                indirect += here.contribution * (m * n.W);
                indirectDiffuse += here.diffuse * (m * n.W);
            }
        }
    }
    if (pc > 0.0f)
    {
        StreamGi(c, pc, canonicalMis * pc * canonical.W, index, seed);
        indirect += canonicalPoint.contribution * (canonicalMis * canonical.W);
        indirectDiffuse += canonicalPoint.diffuse * (canonicalMis * canonical.W);
    }
    GiSample chosen = canonical;
    if (c.chosen && c.source != index)
    {
        chosen = UnpackGi(temporal[c.source]);
    }
    GiSample result = FinishGi(c, chosen, total);
    if ((g.flags & kReSTIRShade) != 0u && any(indirect > 0.0f) && all(isfinite(indirect)))
    {
        RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
        float3 air = CameraAirTransmittance(f, r.position);
        accumulation[pixel] += float4(indirect * air * float(f.samplesPerPixel), 0.0f);
        if ((g.flags & kReSTIRSignals) != 0u)
        {
            RWTexture2D<float4> diffuseSignal = ResourceDescriptorHeap[g.diffuseSignal];
            RWTexture2D<float4> specularSignal = ResourceDescriptorHeap[g.specularSignal];
            diffuseSignal[pixel] += float4(indirectDiffuse * air * f.exposure, 0.0f);
            specularSignal[pixel] += float4((indirect - indirectDiffuse) * air * f.exposure, 0.0f);
        }
    }
    history[index] = PackGi(result);
}

#endif
