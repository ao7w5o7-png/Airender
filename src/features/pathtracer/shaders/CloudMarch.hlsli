// The real-time clouds' ray march (docs/architecture.md §8.19), shared by their passes. In the manner
// of Schneider's Nubis (the clouds of Horizon Zero Dawn and Forbidden West, SIGGRAPH 2015-2023): the
// density is marched in steps rather than tracked, empty space is skipped whole (here the majorant
// cells of CloudLookup.hlsli, which bound the density as Nubis' signed distance field bounds its
// voxels), the sun's light at a step comes from a short march towards it whose steps double, the light
// that scattered more than once from a diffusion term matched to the path traced clouds, and the sky's
// light from an ambient term that grows with height in the layer. A frame marches a sixteenth of the
// render's pixels and the rest is reprojected (CloudReconstruct.hlsl). The cirrus above the layer is
// a thin sheet: its single scattering of the sun in closed form and the sky's light it scatters.
//
// The including shader includes SceneShared.h, PathTracerShared.h and CloudMarchShared.h and declares
// `g` (CloudMarchConstants) before this file.

SamplerState gLinearClamp : register(s0);
SamplerState gLinearWrap : register(s2);

float3 SkyTableLoad(uint texel)
{
    StructuredBuffer<float> table = ResourceDescriptorHeap[g.skyTable];
    return float3(table[texel * 3u], table[texel * 3u + 1u], table[texel * 3u + 2u]);
}

float4 AerialInscatterSample(SkyData sky, float3 uvw)
{
    Texture3D<float4> table = ResourceDescriptorHeap[sky.aerialInscatter];
    return table.SampleLevel(gLinearClamp, uvw, 0.0f);
}

float4 AerialTransmittanceSample(SkyData sky, float3 uvw)
{
    Texture3D<float4> table = ResourceDescriptorHeap[sky.aerialTransmittance];
    return table.SampleLevel(gLinearClamp, uvw, 0.0f);
}

float CloudBaseSample(CloudData c, float3 uvw)
{
    Texture3D<float> base = ResourceDescriptorHeap[c.baseTexture];
    return base.SampleLevel(gLinearWrap, uvw, 0.0f);
}

float CloudDetailSample(CloudData c, float3 uvw)
{
    Texture3D<float> detail = ResourceDescriptorHeap[c.detailTexture];
    return detail.SampleLevel(gLinearWrap, uvw, 0.0f);
}

float CloudMajorantLoad(CloudData c, uint cell)
{
    StructuredBuffer<float> majorants = ResourceDescriptorHeap[c.majorants];
    return majorants[cell];
}

#include "Camera.hlsli"
#include "Sampling.hlsli"
#include "air/renderer/shader/CloudLookup.hlsli"
#include "air/renderer/shader/SkyLookup.hlsli"

FrameData LoadFrame()
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    return frames[g.frameSlot];
}

// The sun's light at a step: its single scattering (the clouds' phase function, the optical depth
// towards the sun) and the light that scattered more than once, which the clouds diffuse: in thick
// clouds (optical depths of tens) it falls off as the two-stream solution of a conservative slab,
// 1 / (1 + 3/4 (1 - g) tau), not exponentially (octaves of single scattering, Wrenninge et al. 2013, came
// out 50% dark inside cumulus). Isotropic; its scale and slope matched to the path traced clouds.
static const float kCloudMsScale = 4.4f;
static const float kCloudMsSlope = 0.3f;
// Where the clouds thin out (their eroded fringe) little light has scattered more than once: the
// diffused light scales with the density to this power.
static const float kCloudMsEdge = 0.5f;
// The sky's light at a height h (0 at the base, 1 at the tops) of the layer: kCloudAmbientBottom +
// (1 - kCloudAmbientBottom) h of the light the sky sends from above, and the reverse share from below.
static const float kCloudAmbientBottom = 0.3f;
static const float kCloudAmbientScale = 0.4f;

// The optical depth from p towards the sun in `samples` steps that grow by 1.6 from 20 m (8 reach
// 1.4 km), each sampled at `jitter` (0..1) along it. Samples at fixed places stamped the outlines of
// the clouds they reached into the light as bands; jittered, the frames average them out.
float CloudSunDepth(CloudData c, float3 p, float3 sun, uint samples, float jitter)
{
    float depth = 0.0f;
    float start = 0.0f;
    float segment = 20.0f;
    for (uint i = 0u; i < samples; ++i)
    {
        float3 q = p + sun * (start + jitter * segment);
        if (CloudAltitude(q) > c.top)
        {
            break;
        }
        depth += CloudDensity(c, q) * segment;
        start += segment;
        segment *= 1.6f;
    }
    return depth * c.extinction;
}

// The sky's mean radiance over the upper (rgb of .up) and lower hemispheres (.down), from 8 directions
// each.
struct CloudAmbient
{
    float3 up;
    float3 down;
};

CloudAmbient SkyAmbient(SkyData sky)
{
    CloudAmbient a;
    a.up = float3(0.0f, 0.0f, 0.0f);
    a.down = float3(0.0f, 0.0f, 0.0f);
    for (uint i = 0u; i < 8u; ++i)
    {
        float phi = (float(i) + 0.5f) * (kTwoPi / 8.0f);
        float y = i % 2u == 0u ? 0.3f : 0.8f;
        float r = sqrt(1.0f - y * y);
        float3 d = float3(r * cos(phi), y, r * sin(phi));
        a.up += SkyRadiance(sky, d);
        a.down += SkyRadiance(sky, float3(d.x, -d.y, d.z));
    }
    a.up *= 1.0f / 8.0f;
    a.down *= 1.0f / 8.0f;
    return a;
}

struct CloudRay
{
    float3 light;        // what the clouds add along the ray (nits, through the air in front)
    float transmittance; // what they leave of what lies behind
    float depth;         // where their light comes from (its weighted mean distance); without clouds
                         // the middle of the layer
};

// The sky's light the cirrus scatters towards the viewer, of what it takes from the sky above (thin
// and forward scattering: mostly the light from above, carried on downwards).
static const float kCirrusAmbientScale = 1.0f;
// Light scattered on forwards inside the sheet (twice or more), which single scattering leaves out: the
// sun's term grows by this times the optical depth along the ray.
static const float kCirrusForward = 0.6f;

// The cirrus sheet's light along o + t d (nits, before the air in front) and what it lets through:
// the sun's single scattering through a thin slab of optical depth tau seen at a cosine muV and lit
// at muS, phase p (1 / muV) exp(-tau / muV) (1 - exp(-tau (1 / muS - 1 / muV))) / (1 / muS - 1 / muV),
// and the sky's light from above scattered as the sheet dims it.
float4 MarchCirrus(FrameData f, float3 o, float3 d, float tMax, CloudAmbient ambient, AIR_OUT(float) tCirrus)
{
    CloudData c = f.clouds;
    float depth;
    if (!CirrusCrossing(c, o, d, 1.0f, tMax, tCirrus, depth))
    {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float3 p = o + d * tCirrus;
    float3 up = normalize(float3(p.x / kCloudPlanetRadius, 1.0f, p.z / kCloudPlanetRadius));
    float invMuV = 1.0f / max(abs(dot(d, up)), kCirrusMinCosine);
    float tau = depth / invMuV; // vertical
    float transmittance = exp(-depth);
    float3 light = kCirrusAmbientScale * ambient.up * (1.0f - transmittance);
    if (HasSun(f.sky))
    {
        float3 sun = f.sky.sunDirection;
        float invMuS = 1.0f / max(dot(sun, up), kCirrusMinCosine);
        float k = invMuS - invMuV;
        float slab = abs(k * tau) > 1e-4f ? (1.0f - exp(-tau * k)) / k : tau;
        light += f.sky.sunRadiance * SunSolidAngle(f.sky) *
                 (CloudPhase(c, dot(d, sun)) * invMuV * transmittance * slab * (1.0f + kCirrusForward * depth));
    }
    return float4(light, transmittance);
}

// The layer along the march `m` (begun along o + t d): its light and transmittance into `ray`, the
// weights of its distances into weightSum and depthSum.
void MarchLayer(FrameData f, float3 o, float3 d, float offset, float lightOffset, uint maxSteps, uint lightSamples,
                CloudAmbient ambient, AIR_INOUT(CloudMarch) m, AIR_INOUT(CloudRay) ray, AIR_INOUT(float) weightSum,
                AIR_INOUT(float) depthSum)
{
    CloudData c = f.clouds;
    bool sun = HasSun(f.sky);
    float3 sunDirection = f.sky.sunDirection;
    float3 sunLight = sun ? f.sky.sunRadiance * SunSolidAngle(f.sky) : float3(0.0f, 0.0f, 0.0f);
    // The cirrus above dims the sun (taken where the ray enters the layer).
    sunLight *= CirrusTransmittance(c, o + d * m.t, sunDirection, 1e30f);
    float cloudPhase = CloudPhase(c, dot(d, sunDirection));
    uint steps = 0u;
    float t = m.t;
    float step = 30.0f;
    bool started = false;
    float t0;
    float t1;
    float majorant;
    while (steps < maxSteps && ray.transmittance > 0.003f && NextCloudSegment(c, m, t0, t1, majorant))
    {
        if (majorant <= 0.0f)
        {
            continue; // empty: skipped whole
        }
        // Steps of 30 m near the camera, longer far away (a pixel covers more there).
        step = clamp(t0 * 0.004f, 30.0f, 250.0f);
        t = max(t, t0);
        if (!started)
        {
            t = t0 + offset * step;
            started = true;
        }
        while (t < t1 && steps < maxSteps && ray.transmittance > 0.003f)
        {
            ++steps;
            float3 p = o + d * t;
            float density = CloudDensity(c, p);
            if (density > 0.0f)
            {
                float extinction = c.extinction * density;
                float scattering = extinction * c.albedo;
                float3 inscattered = float3(0.0f, 0.0f, 0.0f);
                if (sun)
                {
                    float jitter = frac(lightOffset + float(steps) * 0.618034f);
                    float depth = CloudSunDepth(c, p, sunDirection, lightSamples, jitter);
                    float multiple = kCloudMsScale * pow(density, kCloudMsEdge) /
                                     (4.0f * kPi * (1.0f + kCloudMsSlope * depth));
                    inscattered = sunLight * (cloudPhase * exp(-depth) + multiple);
                }
                float h = saturate((CloudAltitude(p) - c.bottom) / (c.top - c.bottom));
                inscattered += kCloudAmbientScale * (ambient.up * lerp(kCloudAmbientBottom, 1.0f, h) +
                                                     ambient.down * lerp(1.0f, kCloudAmbientBottom, h));
                // Integrated over the step with the step's own extinction (Hillaire 2016): no light is
                // created however long the step.
                float stepTransmittance = exp(-extinction * step);
                float3 light = inscattered * (scattering * (1.0f - stepTransmittance) / extinction);
                float weight = ray.transmittance * (1.0f - stepTransmittance);
                ray.light += ray.transmittance * light;
                weightSum += weight;
                depthSum += weight * t;
                ray.transmittance *= stepTransmittance;
            }
            t += step;
        }
    }
}

// The clouds along o + t d up to tMax, the cirrus beyond them included. `offset` and `lightOffset` in
// [0, 1) shift the steps along the ray and towards the sun (a frame's jitter); each step along the ray
// shifts its samples towards the sun on by the golden ratio, so that one ray already averages over
// their places.
CloudRay MarchClouds(FrameData f, float3 o, float3 d, float tMax, float offset, float lightOffset, uint maxSteps,
                     uint lightSamples)
{
    CloudData c = f.clouds;
    CloudRay ray;
    ray.light = float3(0.0f, 0.0f, 0.0f);
    ray.transmittance = 1.0f;
    ray.depth = min(CloudCrossing(MakeCloudCurve(o, d), 0.5f * (c.bottom + c.top), 0.0f), kCloudMaxDistance);
    CloudAmbient ambient = SkyAmbient(f.sky);
    float weightSum = 0.0f;
    float depthSum = 0.0f;
    CloudMarch m;
    if (BeginCloudMarch(c, o, d, tMax, m))
    {
        MarchLayer(f, o, d, offset, lightOffset, maxSteps, lightSamples, ambient, m, ray, weightSum, depthSum);
    }
    if (weightSum > 1e-4f)
    {
        ray.depth = depthSum / weightSum;
        // The air between the camera and the clouds, as a path tracer's walk sees it (EscapeRadiance):
        // it adds its own light where the clouds hide the sky and dims theirs.
        Aerial air = AerialPerspective(f.sky, d, ray.depth);
        ray.light = air.inscatter * (1.0f - ray.transmittance) + air.transmittance * ray.light;
    }
    // The cirrus behind them, through its own air.
    float tCirrus;
    float4 cirrus = MarchCirrus(f, o, d, tMax, ambient, tCirrus);
    if (cirrus.w < 1.0f)
    {
        Aerial air = AerialPerspective(f.sky, d, tCirrus);
        float3 light = air.inscatter * (1.0f - cirrus.w) + air.transmittance * float3(cirrus.x, cirrus.y, cirrus.z);
        float weight = ray.transmittance * (1.0f - cirrus.w);
        ray.light += ray.transmittance * light;
        ray.depth = weightSum + weight > 1e-4f ? (depthSum + weight * tCirrus) / (weightSum + weight) : ray.depth;
        ray.transmittance *= cirrus.w;
    }
    return ray;
}
