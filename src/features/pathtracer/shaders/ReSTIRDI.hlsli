// ReSTIR DI: the direct light at each pixel's first surface, resampled from light samples found by
// the pixel, its neighbours and the previous frame (Bitterli et al. 2020, in the generalized RIS
// formulation of Lin et al. 2022 and the ReSTIR course notes, Wyman et al. 2023). Three compute
// passes run after PathTracer.Trace, which leaves that direct light out (kFlagReSTIR) and records
// each pixel's first hit (PrimaryHit):
//   Initial   RIS over lightCandidates points on lights chosen by power and bsdfCandidates BSDF rays
//             that reach an emitter, weighted by the balance heuristic between the two strategies.
//   Temporal  generalized balance heuristic between that reservoir and last frame's final one at
//             the reprojected pixel, whose confidence is capped at maxConfidence.
//   Spatial   defensive pairwise MIS with spatialSamples neighbours (disocclusionSamples where the
//             temporal pass found no history); shades every candidate by its resampling weight
//             into the accumulation and keeps the chosen one as next frame's history.
// A sample is a point on an emissive triangle, the same point in every pixel's domain (shift
// Jacobian 1). The target function is the luminance of the unshadowed contribution. Each pass traces
// the shadow ray of the sample it chose when that sample is new to the pixel, and zeroes W when it
// is occluded (visibility reuse): a reservoir only holds light its own surface sees. With
// kReSTIRUnbiased the MIS weights respect that: another surface's reservoir gets no weight for a
// sample that surface does not see, at one shadow ray from that surface per reused reservoir.
// Without it they assume the other surface sees the sample: fewer rays, darker penumbrae.
// Neighbours are chosen from their geometry (normal, distance) only, never from their samples,
// which keeps the reuse unbiased.
#ifndef AIR_PT_RESTIR_DI_HLSLI
#define AIR_PT_RESTIR_DI_HLSLI

#include "ReSTIRCommon.hlsli"

// --- Light samples -------------------------------------------------------------------------------

struct LightSample
{
    uint light; // light-list entry
    float2 barycentrics;
};

// The unshadowed light a point on an emissive triangle sends through the receiver's BSDF, per unit
// light area (f * Le * cosLight / distance^2), and what the MIS weights and the shadow ray need.
struct LightPoint
{
    float3 contribution;
    float3 wi; // towards the point
    float dist;
    float cosLight;
    float bsdfPdf; // solid angle pdf of SampleBsdf choosing wi
};

LightPoint EvaluateLightPoint(Receiver r, LightTriangle light, float2 b)
{
    LightPoint lp;
    lp.contribution = float3(0.0f, 0.0f, 0.0f);
    lp.wi = float3(0.0f, 0.0f, 1.0f);
    lp.dist = 0.0f;
    lp.cosLight = 0.0f;
    lp.bsdfPdf = 0.0f;
    float3 toLight = light.p0 + light.e1 * b.x + light.e2 * b.y - r.origin;
    float dist2 = dot(toLight, toLight);
    if (!(dist2 > 0.0f))
    {
        return lp;
    }
    lp.dist = sqrt(dist2);
    lp.wi = toLight / lp.dist;
    lp.cosLight = dot(-lp.wi, normalize(cross(light.e1, light.e2)));
    if (!(lp.cosLight > 0.0f) || dot(lp.wi, r.ng) <= 0.0f)
    {
        return lp;
    }
    BsdfEval e = EvaluateBsdf(r.bsdf, ToLocal(r.frame, lp.wi));
    lp.bsdfPdf = e.pdf;
    lp.contribution = e.fCos * light.radiance * (lp.cosLight / dist2);
    return lp;
}

LightPoint EvaluateSample(Receiver r, LightSample y)
{
    return EvaluateLightPoint(r, LoadLight(y.light), y.barycentrics);
}

bool Visible(Receiver r, LightPoint lp)
{
    return TraceVisible(r.origin, lp.wi, lp.dist * (1.0f - 1e-4f));
}

// The target function p̂: luminance of the unshadowed contribution.
float TargetPdf(Receiver r, LightSample y)
{
    return Luminance(EvaluateSample(r, y).contribution);
}

// p̂ of another surface's reservoirs at y: zero where that surface does not see y (its reservoirs
// never hold such samples) when traceVisibility, else assumed visible.
float DomainTargetPdf(Receiver r, LightSample y, bool traceVisibility)
{
    LightPoint lp = EvaluateSample(r, y);
    float target = Luminance(lp.contribution);
    if (traceVisibility && target > 0.0f && !Visible(r, lp))
    {
        target = 0.0f;
    }
    return target;
}

// Density per unit light area of the initial candidates at a light point: lightCandidates draws by
// power (uniform on the triangle) and bsdfCandidates BSDF rays. With the balance heuristic between
// the two, a candidate's resampling weight is p̂ / this.
float CandidatePdf(LightTriangle light, LightPoint lp, float lightCandidates, float bsdfCandidates)
{
    float pLight = light.area > 0.0f ? light.probability / light.area : 0.0f;
    float pBsdf = lp.dist > 0.0f ? lp.bsdfPdf * lp.cosLight / (lp.dist * lp.dist) : 0.0f;
    return lightCandidates * pLight + bsdfCandidates * pBsdf;
}

// --- Reservoirs ----------------------------------------------------------------------------------

struct ReservoirState
{
    LightSample y;
    float W; // unbiased contribution weight; 0: none
    float M; // confidence
};

ReservoirState MakeReservoir(float confidence)
{
    ReservoirState r;
    r.y.light = 0u;
    r.y.barycentrics = float2(0.0f, 0.0f);
    r.W = 0.0f;
    r.M = confidence;
    return r;
}

ReservoirState Unpack(Reservoir p)
{
    ReservoirState r;
    r.y.light = p.lightAndConfidence & kReSTIRMaxLight;
    r.y.barycentrics = p.barycentrics;
    r.W = p.weight;
    r.M = float(p.lightAndConfidence >> 24);
    return r;
}

Reservoir Pack(ReservoirState r)
{
    Reservoir p;
    p.lightAndConfidence = (r.y.light & kReSTIRMaxLight) | (min(uint(r.M + 0.5f), 255u) << 24);
    p.barycentrics = r.y.barycentrics;
    p.weight = r.W;
    return p;
}

// Weighted reservoir sampling: streams candidates with resampling weights w, keeps one with
// probability w / sum(w).
struct Resampler
{
    LightSample y;
    float targetPdf; // p̂ of y at this pixel
    float weightSum;
    uint source; // the input y came from (0: this pixel's own)
};

Resampler MakeResampler()
{
    Resampler s;
    s.y.light = 0u;
    s.y.barycentrics = float2(0.0f, 0.0f);
    s.targetPdf = 0.0f;
    s.weightSum = 0.0f;
    s.source = 0u;
    return s;
}

void Stream(inout Resampler s, LightSample y, float targetPdf, float w, uint source, inout uint seed)
{
    if (!(w > 0.0f)) // also rejects NaN
    {
        return;
    }
    s.weightSum += w;
    if (Rand(seed) * s.weightSum < w)
    {
        s.y = y;
        s.targetPdf = targetPdf;
        s.source = source;
    }
}

// The chosen sample with W = sum(w) / p̂(y).
ReservoirState Finish(Resampler s, float confidence)
{
    ReservoirState r = MakeReservoir(confidence);
    if (s.targetPdf > 0.0f)
    {
        r.y = s.y;
        r.W = s.weightSum / s.targetPdf;
    }
    return r;
}

// --- Passes --------------------------------------------------------------------------------------

void InitialPass(uint2 pixel)
{
    FrameData f = LoadFrame();
    if (pixel.x >= f.width || pixel.y >= f.height)
    {
        return;
    }
    uint index = pixel.y * f.width + pixel.x;
    IntegratorParams params = MakeParams(f);
    Receiver r = LoadReceiver(LoadPrimaryHit(g.primaryHits, index), f.camera.position.xyz, params);
    ReservoirState result = MakeReservoir(1.0f);
    if (r.valid)
    {
        uint seed = StreamSeed(pixel, f.frameIndex, 1u);
        Resampler s = MakeResampler();
        float lightCandidates = float(g.lightCandidates);
        float bsdfCandidates = float(g.bsdfCandidates);
        for (uint i = 0u; i < g.lightCandidates; ++i)
        {
            float uSelect = Rand(seed);
            float uAlias = Rand(seed);
            float2 uPoint = Rand2(seed);
            LightSample y;
            y.light = SampleLightIndex(uSelect, uAlias, params.lightCount);
            y.barycentrics = SampleTriangle(uPoint);
            LightTriangle light = LoadLight(y.light);
            LightPoint lp = EvaluateLightPoint(r, light, y.barycentrics);
            float target = Luminance(lp.contribution);
            Stream(s, y, target, target / CandidatePdf(light, lp, lightCandidates, bsdfCandidates), 0u, seed);
        }
        for (uint j = 0u; j < g.bsdfCandidates; ++j)
        {
            float uLobe = Rand(seed);
            float2 u = Rand2(seed);
            BsdfSample bs = SampleBsdf(r.bsdf, uLobe, u);
            float3 wi = ToWorld(r.frame, bs.wi);
            if (!bs.valid || dot(wi, r.ng) <= 0.0f)
            {
                continue;
            }
            // A BSDF candidate is the emitter point the ray reaches first, if any.
            HitInfo hit = TraceClosest(r.origin, wi, 1u);
            uint firstLight = hit.hit ? LoadInstanceLight(hit.instance) : kInvalidIndex;
            if (firstLight == kInvalidIndex)
            {
                continue;
            }
            LightSample y;
            y.light = firstLight + hit.primitive;
            y.barycentrics = hit.barycentrics;
            LightTriangle light = LoadLight(y.light);
            LightPoint lp = EvaluateLightPoint(r, light, y.barycentrics);
            float target = Luminance(lp.contribution);
            Stream(s, y, target, target / CandidatePdf(light, lp, lightCandidates, bsdfCandidates), 1u, seed);
        }
        result = Finish(s, 1.0f);
        if (result.W > 0.0f && !Visible(r, EvaluateSample(r, result.y)))
        {
            result.W = 0.0f; // visibility reuse
        }
    }
    RWStructuredBuffer<Reservoir> output = ResourceDescriptorHeap[g.output];
    output[index] = Pack(result);
}

void TemporalPass(uint2 pixel)
{
    FrameData f = LoadFrame();
    if (pixel.x >= f.width || pixel.y >= f.height)
    {
        return;
    }
    uint index = pixel.y * f.width + pixel.x;
    StructuredBuffer<Reservoir> initial = ResourceDescriptorHeap[g.input];
    ReservoirState canonical = Unpack(initial[index]);
    ReservoirState result = canonical;
    PrimaryHit hit = LoadPrimaryHit(g.primaryHits, index);
    if (hit.instance != kInvalidIndex && (g.flags & kReSTIRTemporal) != 0u)
    {
        IntegratorParams params = MakeParams(f);
        Receiver r = LoadReceiver(hit, f.camera.position.xyz, params);
        // The pixel last frame's camera saw this point in, if its surface there is this one.
        float2 size = float2(float(f.width), float(f.height));
        float3 previousEye = f.previousCamera.position.xyz;
        float2 previousPixel = floor(CameraProject(f.previousCamera, r.position, 1.0f, size));
        if (all(previousPixel >= 0.0f) && all(previousPixel < size))
        {
            uint previousIndex = uint(previousPixel.y) * f.width + uint(previousPixel.x);
            PrimaryHit previousHit = LoadPrimaryHit(g.previousPrimaryHits, previousIndex);
            if (SimilarSurface(hit, previousHit, length(r.position - previousEye)))
            {
                Receiver rt = LoadReceiver(previousHit, previousEye, params);
                StructuredBuffer<Reservoir> previous = ResourceDescriptorHeap[g.previous];
                ReservoirState t = Unpack(previous[previousIndex]);
                t.M = min(t.M, g.maxConfidence);
                bool unbiased = (g.flags & kReSTIRUnbiased) != 0u;
                uint seed = StreamSeed(pixel, f.frameIndex, 2u);
                Resampler s = MakeResampler();
                // Generalized balance heuristic between this pixel's domain and last frame's.
                if (canonical.W > 0.0f)
                {
                    float pc = TargetPdf(r, canonical.y);
                    float pt = DomainTargetPdf(rt, canonical.y, unbiased);
                    float m = canonical.M * pc / (canonical.M * pc + t.M * pt);
                    Stream(s, canonical.y, pc, m * pc * canonical.W, 0u, seed);
                }
                if (t.W > 0.0f)
                {
                    float pc = TargetPdf(r, t.y);
                    float pt = TargetPdf(rt, t.y); // its surface sees it: W would be 0 otherwise
                    float m = t.M * pt / (canonical.M * pc + t.M * pt);
                    Stream(s, t.y, pc, m * pc * t.W, 1u, seed);
                }
                result = Finish(s, canonical.M + t.M);
                // The initial sample's visibility is known; last frame's needs a ray from here.
                if (result.W > 0.0f && s.source != 0u && !Visible(r, EvaluateSample(r, result.y)))
                {
                    result.W = 0.0f;
                }
            }
        }
    }
    RWStructuredBuffer<Reservoir> output = ResourceDescriptorHeap[g.output];
    output[index] = Pack(result);
}

void SpatialPass(uint2 pixel)
{
    FrameData f = LoadFrame();
    if (pixel.x >= f.width || pixel.y >= f.height)
    {
        return;
    }
    uint index = pixel.y * f.width + pixel.x;
    StructuredBuffer<Reservoir> temporal = ResourceDescriptorHeap[g.input];
    RWStructuredBuffer<Reservoir> history = ResourceDescriptorHeap[g.output];
    ReservoirState canonical = Unpack(temporal[index]);
    PrimaryHit hit = LoadPrimaryHit(g.primaryHits, index);
    if (hit.instance == kInvalidIndex)
    {
        history[index] = Pack(canonical);
        return;
    }
    IntegratorParams params = MakeParams(f);
    float3 eye = f.camera.position.xyz;
    Receiver r = LoadReceiver(hit, eye, params);
    bool unbiased = (g.flags & kReSTIRUnbiased) != 0u;

    // Neighbours and the sum of their confidences (the MIS weights need it before any of them).
    uint draws = min(canonical.M <= 1.0f ? g.disocclusionSamples : g.spatialSamples, kReSTIRMaxSpatialSamples);
    uint firstDraw = StreamSeed(pixel, f.frameIndex, 3u);
    uint draw = firstDraw;
    float neighbourConfidence = 0.0f;
    for (uint i = 0u; i < draws; ++i)
    {
        uint neighbour;
        if (NextNeighbour(draw, pixel, hit, f, 0.0f, neighbour))
        {
            neighbourConfidence += Unpack(temporal[neighbour]).M;
        }
    }

    // Defensive pairwise MIS (Wyman et al. 2023, eq. 7.8): each neighbour plays a balance heuristic
    // game against this pixel, weighted by its share of the confidence; this pixel keeps at least
    // its own share. The target here includes this pixel's visibility: one ray per reused sample,
    // which makes the chosen sample visible (no shading ray) and lets the shading average every
    // candidate by its resampling weight instead of using the chosen one alone (the expectation of
    // f(Y) W over the choice; Lin et al. 2026), which removes most of the colour noise of one
    // light per pixel.
    float cc = canonical.M;
    float total = cc + neighbourConfidence;
    LightPoint canonicalPoint = EvaluateSample(r, canonical.y);
    float pc = canonical.W > 0.0f ? Luminance(canonicalPoint.contribution) : 0.0f; // visible: zeroed otherwise
    float canonicalMis = cc / total;
    float3 direct = float3(0.0f, 0.0f, 0.0f);
    uint seed = StreamSeed(pixel, f.frameIndex, 4u);
    Resampler s = MakeResampler();
    draw = firstDraw;
    for (uint j = 0u; j < draws; ++j)
    {
        uint neighbour;
        if (!NextNeighbour(draw, pixel, hit, f, 0.0f, neighbour))
        {
            continue;
        }
        Receiver rn = LoadReceiver(LoadPrimaryHit(g.primaryHits, neighbour), eye, params);
        ReservoirState n = Unpack(temporal[neighbour]);
        if (pc > 0.0f)
        {
            float pn = DomainTargetPdf(rn, canonical.y, unbiased);
            canonicalMis += (n.M / total) * cc * pc / (neighbourConfidence * pn + cc * pc);
        }
        if (n.W > 0.0f)
        {
            LightPoint lp = EvaluateSample(r, n.y);
            float pq = Luminance(lp.contribution);
            if (pq > 0.0f && Visible(r, lp))
            {
                float pn = TargetPdf(rn, n.y); // its surface sees it: W would be 0 otherwise
                float m = (neighbourConfidence / total) * n.M * pn / (neighbourConfidence * pn + cc * pq);
                Stream(s, n.y, pq, m * pq * n.W, j + 1u, seed);
                direct += lp.contribution * (m * n.W);
            }
        }
    }
    if (pc > 0.0f)
    {
        Stream(s, canonical.y, pc, canonicalMis * pc * canonical.W, 0u, seed);
        direct += canonicalPoint.contribution * (canonicalMis * canonical.W);
    }
    ReservoirState result = Finish(s, total);
    if ((g.flags & kReSTIRShade) != 0u && any(direct > 0.0f) && all(isfinite(direct)))
    {
        RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
        accumulation[pixel] += float4(direct * CameraAirTransmittance(f, r.position) * float(f.samplesPerPixel), 0.0f);
    }
    history[index] = Pack(result);
}

#endif
