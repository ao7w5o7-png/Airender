// Subsurface scattering (docs/architecture.md §8.20). Shared by HLSL and C++; included by
// Integrator.hlsli with AIR_PT_SUBSURFACE after TraceClosest, OffsetRayOrigin and LoadSurface.
//
// A material with MaterialData::subsurfaceRadius > 0 is a dielectric whose diffuse light does not
// leave where it entered (Bsdf.hlsli): of the light the surface does not reflect specularly, 1 -
// Es(wo) enters, spreads beneath it and leaves at another point x_o through the exit lobe. Beneath,
// the material is a homogeneous medium whose single-scattering albedo is chosen so that a half-space
// of it reflects baseColor in all (the inversion table in the BSDF LUT, of this medium's walk), and
// whose extinction makes the light spread about subsurfaceRadius (Chiang et al. 2016: sigma_t = 1 /
// (radius s(A) (1 - g)), s = 1.9 - A + 3.5 (A - 0.8)^2, with the anisotropy's similarity scaling). With a radius that goes to 0 the material becomes the
// plain dielectric again, exactly. Where the light leaves is found by
//   kSubsurfaceRandomWalk   a random walk in the medium inside the mesh itself (rays against the scene
//                           find where it leaves): unbiased for any shape, the reference
//   kSubsurfaceApproximate  a profile R(r) over the distance from x_i, Burley's normalized diffusion
//                           (Christensen and Burley 2015), the exit found with a probe ray through a
//                           disc around x_i (on the plane of the normal or of one of the tangents;
//                           the profile is exact for a flat half-space only, and only exits that face
//                           the way x_i does count: light through thin parts is the walk's)
//   kSubsurfaceNeural       the same probe with the profile a small MLP has learnt from the half-space
//                           walk: Burley's profile times exp(z(log(r / d), A, g))
// Colour channels walk together: one channel, chosen at random, draws every distance of the walk, and
// each channel's light is weighed by the balance heuristic over the three channels' pdfs of the whole
// walk (one-sample MIS, as pbrt-v4 does with its wavelengths): the weight stays below 3 times the
// product of the channel's albedos however the channels' radii differ. (Weighing each segment on its
// own bounded the segments but not their product: in the white furnace 1% of the light went missing
// in the rare enormous weights of long walks.)
#ifndef AIR_PT_SUBSURFACE_HLSLI
#define AIR_PT_SUBSURFACE_HLSLI

#include "SubsurfaceProfile.hlsli"

// Scattering events of a walk, at most (Russian roulette ends most long before): a medium of albedo
// 0.99 scatters about a hundred times before the light leaves it, one without absorption (a white
// furnace) thousands of times in a ball ten mean free paths across (512 lost 0.4% there).
AIR_CONST uint kSubsurfaceWalkEvents = 4096u;
// The probe's disc reaches this many times the widest channel's radius (Burley's profile beyond it:
// under 0.2% of the light).
AIR_CONST float kSubsurfaceProbeReach = 20.0f;
// Probe hits on the entering mesh, at most, among which one is chosen, and rays a probe may trace
// (it passes other meshes on the way: a budget of all hits lost the light where the chord ran through
// the neighbouring balls of `subsurface` first, a dark cross on each).
AIR_CONST uint kSubsurfaceProbeHits = 4u;
AIR_CONST uint kSubsurfaceProbeRays = 12u;

// Where the light that entered leaves: the hit of the ray that found the exit (from `origin` along
// `direction`), and what the light carries there per unit that entered.
struct SubsurfaceExit
{
    bool valid;
    HitInfo hit;
    float3 origin;
    float3 direction;
    float3 weight;
};

SubsurfaceExit NoSubsurfaceExit()
{
    SubsurfaceExit e;
    e.valid = false;
    e.hit.hit = false;
    e.hit.t = 0.0f;
    e.hit.instance = 0u;
    e.hit.primitive = 0u;
    e.hit.barycentrics = float2(0.0f, 0.0f);
    e.origin = float3(0.0f, 0.0f, 0.0f);
    e.direction = float3(0.0f, 0.0f, 1.0f);
    e.weight = float3(0.0f, 0.0f, 0.0f);
    return e;
}

float3 SubsurfaceHgDirection(float3 direction, float g, float2 u)
{
    float cosTheta = SampleHgCosine(g, u.x);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = kTwoPi * u.y;
    Frame f = MakeFrame(direction);
    return ToWorld(f, float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));
}

// --- The random walk ------------------------------------------------------------------------------

// From x_i (`p`, outward geometric normal `n`) into the medium along a cosine around -n, until a ray
// of the walk reaches a surface before its next collision: there the light leaves.
SubsurfaceExit SubsurfaceWalk(SubsurfaceMedium m, float3 p, float3 n, AIR_INOUT(uint) seed)
{
    SubsurfaceExit e = NoSubsurfaceExit();
    Frame f = MakeFrame(-n);
    float3 d = ToWorld(f, SampleCosineHemisphere(Rand2(seed)));
    float3 o = OffsetRayOrigin(p, -n);
    // The walk's value per channel over the sampling channel's pdf (beta), and each channel's pdf over
    // it (ratio, 1 for the sampling channel): the channel's light is beta / mean(ratio).
    uint c = min(uint(Rand(seed) * 3.0f), 2u);
    float sigmaC = SubsurfaceComponent(m.sigmaT, c);
    float3 beta = float3(1.0f, 1.0f, 1.0f);
    float3 ratio = float3(1.0f, 1.0f, 1.0f);
    for (uint event = 0u; event < kSubsurfaceWalkEvents; ++event)
    {
        float t = -log(1.0f - Rand(seed)) / sigmaC;
        HitInfo hit = TraceClosest(o, d, 1u);
        if (!hit.hit)
        {
            return e; // an open mesh: the light is lost
        }
        if (t >= hit.t)
        {
            // Past the boundary: every channel's chance is its transmittance.
            float3 through = exp(-m.sigmaT * hit.t);
            float throughC = exp(-sigmaC * hit.t);
            beta *= through / throughC;
            ratio *= through / throughC;
            e.valid = true;
            e.hit = hit;
            e.origin = o;
            e.direction = d;
            e.weight = beta / SubsurfaceMean(ratio);
            return e;
        }
        float3 collide = m.sigmaT * exp(-m.sigmaT * t);
        float collideC = sigmaC * exp(-sigmaC * t);
        beta *= m.albedo * collide / collideC;
        ratio *= collide / collideC;
        // Rescaled together (their quotient is what counts) before either leaves float's range.
        float largest = max(MaxComponent(ratio), MaxComponent(beta));
        if (largest > 1e10f)
        {
            beta /= largest;
            ratio /= largest;
        }
        o = o + d * t;
        d = SubsurfaceHgDirection(d, m.g, Rand2(seed));
        if (event >= 8u)
        {
            float survive = min(MaxComponent(beta) / SubsurfaceMean(ratio), 0.95f);
            if (Rand(seed) >= survive)
            {
                return e;
            }
            beta /= survive;
        }
    }
    return e; // too long a walk: its light is lost
}

// --- The profile ------------------------------------------------------------------------------------

// The probe's axes: the normal half of the time, either tangent a quarter.
float SubsurfaceAxisProbability(uint axis)
{
    return axis == 0u ? 0.5f : 0.25f;
}

float3 SubsurfaceAxis(Frame f, uint axis)
{
    return axis == 0u ? f.n : (axis == 1u ? f.t : f.b);
}

// Whether the probe's hit faces the way the entry does (its triangle winds outwards there).
bool SubsurfaceFacesAlong(HitInfo hit, float3 o, float3 d, float3 n)
{
    InstanceData instance = LoadInstance(hit.instance);
    float3 p0 = LoadVertex(instance, LoadIndex(instance, hit.primitive * 3u)).position;
    float3 p1 = LoadVertex(instance, LoadIndex(instance, hit.primitive * 3u + 1u)).position;
    float3 p2 = LoadVertex(instance, LoadIndex(instance, hit.primitive * 3u + 2u)).position;
    float3 w0 = TransformRow3(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2, p0, 1.0f);
    float3 w1 = TransformRow3(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2, p1, 1.0f);
    float3 w2 = TransformRow3(instance.objectToWorld0, instance.objectToWorld1, instance.objectToWorld2, p2, 1.0f);
    return dot(cross(w1 - w0, w2 - w0), n) > 0.0f;
}

// A point where the light leaves, found by a probe ray through a disc around x_i (`p`, outward
// geometric normal `n`) on the entering mesh (`instance`), weighed by the profile over the pdf of
// finding it (every axis and channel that could have).
SubsurfaceExit SubsurfaceProbe(SubsurfaceMedium m, float3 p, float3 n, uint instance, uint model, AIR_INOUT(uint) seed)
{
    SubsurfaceExit e = NoSubsurfaceExit();
    float uAxis = Rand(seed);
    uint axis = uAxis < 0.5f ? 0u : (uAxis < 0.75f ? 1u : 2u);
    uint c = min(uint(Rand(seed) * 3.0f), 2u);
    float r = SampleBurleyRadius(SubsurfaceComponent(m.radius, c), Rand(seed));
    float reach = kSubsurfaceProbeReach * MaxComponent(m.radius);
    if (r >= reach)
    {
        return e;
    }
    Frame f = MakeFrame(n);
    float3 a = SubsurfaceAxis(f, axis);
    Frame disc = MakeFrame(a);
    float phi = kTwoPi * Rand(seed);
    float halfChord = sqrt(reach * reach - r * r);
    float3 o = p + (disc.t * cos(phi) + disc.b * sin(phi)) * r + a * halfChord;
    float3 d = -a;
    // One of the hits on the entering mesh along the chord, uniformly.
    uint count = 0u;
    float travelled = 0.0f;
    for (uint k = 0u; k < kSubsurfaceProbeRays && count < kSubsurfaceProbeHits; ++k)
    {
        HitInfo hit = TraceClosest(o, d, 1u);
        if (!hit.hit || travelled + hit.t > 2.0f * halfChord)
        {
            break;
        }
        if (hit.instance == instance && SubsurfaceFacesAlong(hit, o, d, n))
        {
            ++count;
            if (Rand(seed) * float(count) < 1.0f)
            {
                e.hit = hit;
                e.origin = o;
                e.direction = d;
            }
        }
        float step = hit.t + 1e-4f * (1.0f + hit.t);
        travelled += step;
        o = o + d * step;
    }
    e.valid = count > 0u;
    e.weight = float3(float(count), float(count), float(count)); // completed by SubsurfaceProbeWeight
    return e;
}

// The probe's weight at the exit x_o (outward normal no): the profile at |x_o - x_i| times the hits
// it chose among over the pdf of finding x_o, per unit area: every axis and channel.
float3 SubsurfaceProbeWeight(SubsurfaceMedium m, float3 p, float3 n, float3 xo, float3 no, float count, uint model)
{
    Frame f = MakeFrame(n);
    float3 v = xo - p;
    float pdf = 0.0f;
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        float3 a = SubsurfaceAxis(f, axis);
        float along = dot(v, a);
        float rAxis = sqrt(max(dot(v, v) - along * along, 0.0f));
        float radial = (BurleyRadialPdf(rAxis, m.radius.x) + BurleyRadialPdf(rAxis, m.radius.y) +
                        BurleyRadialPdf(rAxis, m.radius.z)) *
                       (1.0f / 3.0f);
        pdf += SubsurfaceAxisProbability(axis) * radial / (kTwoPi * max(rAxis, 1e-9f)) * abs(dot(no, a));
    }
    if (pdf <= 0.0f)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return SubsurfaceProfile(m, length(v), model) * (count / pdf);
}

#endif
