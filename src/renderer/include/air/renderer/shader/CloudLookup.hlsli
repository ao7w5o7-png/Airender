// Clouds (CloudShared.h): their density, free-flight sampling and transmittance through them, and
// their phase function. Shared by C++ and HLSL; the including code includes Sampling.hlsli (Rand)
// and defines, before including this file:
//   float CloudBaseSample(CloudData c, float3 uvw)    the base texture, trilinear, wrapping (0..1)
//   float CloudDetailSample(CloudData c, float3 uvw)  the detail texture, likewise
//   float CloudMajorantLoad(CloudData c, uint cell)   extinction per metre bounding the cell
//
// The layer follows the planet's curvature: its heights are altitudes above a sphere of radius
// kCloudPlanetRadius under the scene's origin, a(p) = p.y + (p.x^2 + p.z^2) / 2R (exact to a metre
// within kCloudMaxDistance). Far away the layer sinks towards the horizon, as clouds 100 km off do
// (785 m lower), instead of a flat sheet whose edge-on view fills the horizon with clouds. In
// (x, altitude, z) the grid stays flat and a ray is a parabola.
//
// Tracking walks the majorant cells along the ray (a 3D DDA over the periodic grid, which skips
// empty cells whole: its x and z boundaries are crossed where the line crosses them, its altitude
// boundaries where the parabola does) and samples tentative collisions in each at the rate of its
// majorant: delta tracking finds the first real collision, ratio tracking estimates transmittance.
// Both are unbiased because no density exceeds its cell's majorant.
//
// Above the layer a sheet of cirrus (CloudData::cirrusDepth > 0) is crossed once by a ray: it scatters
// with probability 1 - exp(-its optical depth along the ray), with the clouds' phase function and
// albedo, and dims the transmittance by exp(-depth). Its optical depth is procedural (gradient noise
// in fibres stretched along the wind and bent by broad swirls, gathered into patches).
#ifndef AIR_RENDERER_CLOUD_LOOKUP_HLSLI
#define AIR_RENDERER_CLOUD_LOOKUP_HLSLI

#include "air/renderer/shader/CloudShared.h"

AIR_CONST float kCloudPi = 3.14159265358979f;
AIR_CONST float kCloudMaxDistance = 100000.0f; // metres along a ray
AIR_CONST float kCloudPlanetRadius = 6371000.0f;

// A point's altitude, the height the layer's grid is indexed by.
float CloudAltitude(float3 p)
{
    return p.y + (p.x * p.x + p.z * p.z) * (0.5f / kCloudPlanetRadius);
}

// The altitude along o + t d: a t^2 + b t + c.
struct CloudCurve
{
    float a;
    float b;
    float c;
};

CloudCurve MakeCloudCurve(float3 o, float3 d)
{
    CloudCurve q;
    q.a = (d.x * d.x + d.z * d.z) * (0.5f / kCloudPlanetRadius);
    q.b = d.y + (o.x * d.x + o.z * d.z) / kCloudPlanetRadius;
    q.c = CloudAltitude(o);
    return q;
}

// The first t > tFrom where the altitude along the curve is `level`; 3e38 if none.
float CloudCrossing(CloudCurve q, float level, float tFrom)
{
    float c = q.c - level;
    if (q.a < 1e-14f)
    {
        float t = abs(q.b) > 1e-20f ? -c / q.b : -1.0f;
        return t > tFrom ? t : 3.0e38f;
    }
    float discriminant = q.b * q.b - 4.0f * q.a * c;
    if (discriminant < 0.0f)
    {
        return 3.0e38f;
    }
    // The numerically stable pair of roots.
    float s = sqrt(discriminant);
    float h = -0.5f * (q.b + (q.b >= 0.0f ? s : -s));
    float r0 = h / q.a;
    float r1 = h != 0.0f ? c / h : r0;
    float lo = min(r0, r1);
    float hi = max(r0, r1);
    return lo > tFrom ? lo : (hi > tFrom ? hi : 3.0e38f);
}

uint CloudWrap(int i, uint n)
{
    int m = i % int(n);
    return uint(m < 0 ? m + int(n) : m);
}

// The detail's flow at a point (of the drifting base, metres): a swirl that churns the clouds' fringe in
// place while the bodies drift. Divergence free across the wind (the curl of two products of sines,
// cells of about 300 to 600 m) and rising (0.2 to 0.6 of it), about unit length; in-place motion
// without stretching comes from two phases of it half a period apart, cross-faded (a flow map). Its
// sines run whole numbers of times over the tile (64, 83, 141 and 122), so the clouds still repeat.
float3 CloudFlowField(CloudData c, float3 q)
{
    float cycle = 2.0f * kCloudPi / c.tileSize;
    float a = 64.0f * cycle;
    float b = 83.0f * cycle;
    float a2 = 122.0f * cycle;
    float b2 = 141.0f * cycle;
    float s1x = sin(q.x * a + 1.3f);
    float c1x = cos(q.x * a + 1.3f);
    float s1z = sin(q.z * b + 0.7f);
    float c1z = cos(q.z * b + 0.7f);
    float s2x = sin(q.x * b2 + 2.1f);
    float c2x = cos(q.x * b2 + 2.1f);
    float s2z = sin(q.z * a2 + 0.4f);
    float c2z = cos(q.z * a2 + 0.4f);
    // psi = s1x s1z + 0.5 s2x s2z; the swirl is (d psi / dz, -d psi / dx), scaled to about 1.
    float vx = s1x * c1z * b + 0.5f * s2x * c2z * a2;
    float vz = -(c1x * s1z * a + 0.5f * c2x * s2z * b2);
    float rise = 0.4f + 0.2f * s1x * s2z;
    float scale = 1.0f / (b + 0.5f * a2); // the swirl's largest speed
    return float3(vx * scale, rise, vz * scale);
}

// The detail at a point of the detail's own drift, flowing: two phases of the flow half a period
// apart, each restarting where the other has all the weight.
float CloudDetailFlow(CloudData c, float3 q, float3 d)
{
    float3 uvw = float3(d.x / c.detailPeriod, d.z / c.detailPeriod, d.y / c.detailPeriod);
    if (c.flowAmount <= 0.0f)
    {
        return CloudDetailSample(c, float3(frac(uvw.x), frac(uvw.y), frac(uvw.z)));
    }
    float3 f = CloudFlowField(c, q) * (c.flowAmount / c.detailPeriod);
    float3 flow = float3(f.x, f.z, f.y); // to the detail's (x, z, altitude)
    float phase0 = frac(c.flowPhase);
    float phase1 = frac(c.flowPhase + 0.5f);
    float3 p0 = uvw - flow * phase0;
    float3 p1 = uvw - flow * phase1;
    float d0 = CloudDetailSample(c, float3(frac(p0.x), frac(p0.y), frac(p0.z)));
    float d1 = CloudDetailSample(c, float3(frac(p1.x), frac(p1.y), frac(p1.z)));
    return lerp(d0, d1, abs(1.0f - 2.0f * phase0));
}

// Density (0..1) at a point: the base texture, its fringe eroded by the detail texture, sharpened,
// each where the wind has carried it (CloudData's offsets). The erosion never raises the density and
// the sharpening is monotonic, so the majorants of the sharpened base texture bound it.
float CloudDensity(CloudData c, float3 p)
{
    float altitude = CloudAltitude(p);
    float h = (altitude - c.bottom) / (c.top - c.bottom);
    if (h < 0.0f || h > 1.0f)
    {
        return 0.0f;
    }
    float3 uvw = float3(frac((p.x - c.baseOffsetX) / c.tileSize), frac((p.z - c.baseOffsetZ) / c.tileSize),
                        (h * float(c.gridY) + 1.0f) / float(c.gridY + 2u));
    float base = CloudBaseSample(c, uvw);
    if (base <= 0.0f || c.detailStrength <= 0.0f)
    {
        return saturate(base * c.sharpness);
    }
    float detail = CloudDetailFlow(c, float3(p.x - c.baseOffsetX, altitude, p.z - c.baseOffsetZ),
                                   float3(p.x - c.detailOffsetX, altitude - c.detailOffsetY, p.z - c.detailOffsetZ));
    float erosion = detail * c.detailStrength * (1.0f - base);
    return saturate((base - erosion) / (1.0f - erosion) * c.sharpness);
}

// --- Cirrus --------------------------------------------------------------------------------------------

AIR_CONST float kCirrusMinCosine = 0.05f; // a grazing ray crosses the sheet as at 3 degrees

// Gradient noise in the plane (about -0.7..0.7), from an integer hash of the lattice.
float2 CirrusGradient(int x, int y)
{
    uint h = uint(x) * 0x8da6b343u ^ uint(y) * 0xd8163841u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    float a = float(h & 0xffffu) * (2.0f * kCloudPi / 65536.0f);
    return float2(cos(a), sin(a));
}

float CirrusNoise(float2 p)
{
    float ix = floor(p.x);
    float iy = floor(p.y);
    float2 f = float2(p.x - ix, p.y - iy);
    float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    int x = int(ix);
    int y = int(iy);
    float a = dot(CirrusGradient(x, y), f);
    float b = dot(CirrusGradient(x + 1, y), f - float2(1.0f, 0.0f));
    float c = dot(CirrusGradient(x, y + 1), f - float2(0.0f, 1.0f));
    float d = dot(CirrusGradient(x + 1, y + 1), f - float2(1.0f, 1.0f));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// A ridge of gradient noise: 1 on its zero line, falling off to either side (thin filaments, where
// the noise itself makes bands).
float CirrusRidge(float2 p)
{
    float r = 1.0f - abs(CirrusNoise(p)) * 2.2f;
    r = saturate(r);
    return r * r * r;
}

float2 CirrusRotate(float2 v, float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return float2(c * v.x - s * v.y, s * v.x + c * v.y);
}

// The sheet's vertical optical depth over (x, z). Cirrus is wisps of ice falling from where it forms
// and drawn out by the wind's shear: filaments a few hundred metres across and kilometres long, curling
// into hooks, tapering, in patches of a few to tens of kilometres with clear sky between, a faint veil
// among them. Here: ridges of gradient noise (thin lines, not the bands of plain noise) stretched 6:1
// along the wind, each of three octaves turned a little against the others, over coordinates warped
// by swirls of 18 and 6 km (the curls); a taper along them; a soft mask of patches over the fraction
// of the sky asked for (its noise's quantile, CirrusThreshold).
float CirrusOpticalDepth(CloudData c, float x, float z)
{
    float2 q = float2(x - c.cirrusOffsetX, z - c.cirrusOffsetZ);
    float speed = sqrt(c.windX * c.windX + c.windZ * c.windZ);
    float2 w = speed > 1e-3f ? float2(c.windX, c.windZ) / speed : float2(1.0f, 0.0f);
    // Patches, before the warp: where cirrus is at all.
    float patches = CirrusNoise(q * (1.0f / 15000.0f) + float2(31.3f, 11.1f)) +
                    0.5f * CirrusNoise(q * (1.0f / 6000.0f) + float2(7.3f, 23.9f));
    float k = saturate((0.5f + 0.7f * patches - c.cirrusThreshold) / 0.45f);
    float mask = k * k * (3.0f - 2.0f * k);
    if (mask <= 0.0f)
    {
        return 0.0f;
    }
    // The curls: a broad warp and a finer one.
    q += float2(CirrusNoise(q * (1.0f / 18000.0f) + float2(3.1f, 0.7f)),
                CirrusNoise(q * (1.0f / 18000.0f) + float2(17.7f, 5.3f))) * 6000.0f;
    q += float2(CirrusNoise(q * (1.0f / 6000.0f) + float2(9.1f, 4.4f)),
                CirrusNoise(q * (1.0f / 6000.0f) + float2(2.6f, 8.8f))) * 1200.0f;
    float2 along = float2(dot(q, w), dot(q, float2(-w.y, w.x)));
    float strands = 0.0f;
    float amplitude = 1.0f;
    float scale = 1.0f / 2400.0f;
    for (uint octave = 0u; octave < 3u; ++octave)
    {
        float2 r = CirrusRotate(along, 0.25f * (float(octave) - 1.0f)) * scale;
        float ridge = CirrusRidge(float2(r.x / 6.0f, r.y) + float2(5.1f, 1.7f) * float(octave + 1u));
        // Tapering along the strand: its strength drifts in and out along its length.
        float taper = saturate(0.6f + CirrusNoise(float2(r.x / 3.0f, r.y * 0.25f) + float2(13.3f, 6.1f)) * 1.4f);
        strands += amplitude * ridge * taper;
        amplitude *= 0.55f;
        scale *= 2.1f;
    }
    float veil = 0.18f * mask;
    return c.cirrusDepth * mask * saturate(veil + strands * 0.8f);
}

// Where o + t d crosses the sheet after tFrom and before tMax, and its optical depth along the ray
// there; false where the ray does not cross it (or it is clear).
bool CirrusCrossing(CloudData c, float3 o, float3 d, float tFrom, float tMax, AIR_OUT(float) t, AIR_OUT(float) depth)
{
    t = 3.0e38f;
    depth = 0.0f;
    if (c.enabled == 0u || c.cirrusDepth <= 0.0f)
    {
        return false;
    }
    t = CloudCrossing(MakeCloudCurve(o, d), c.cirrusAltitude, tFrom);
    if (t > min(tMax, kCloudMaxDistance))
    {
        return false;
    }
    float3 p = o + d * t;
    float3 up = normalize(float3(p.x / kCloudPlanetRadius, 1.0f, p.z / kCloudPlanetRadius));
    depth = CirrusOpticalDepth(c, p.x, p.z) / max(abs(dot(d, up)), kCirrusMinCosine);
    return depth > 0.0f;
}

// What the sheet lets through along o + t d over [0, tMax].
float CirrusTransmittance(CloudData c, float3 o, float3 d, float tMax)
{
    float t;
    float depth;
    return CirrusCrossing(c, o, d, 1.0f, tMax, t, depth) ? exp(-depth) : 1.0f;
}

// --- Majorant cells along a ray --------------------------------------------------------------------

struct CloudMarch
{
    float t;    // start of the next segment
    float tEnd; // where the ray leaves the layer (or stops)
    int cellX;  // current cell, not wrapped (y: of altitude)
    int cellY;
    int cellZ;
    int stepX;
    int stepY; // the altitude boundary at nextY: +1 the upper, -1 the lower
    int stepZ;
    float nextX; // ray distance to the next cell boundary on each axis
    float nextY;
    float nextZ;
    float deltaX; // ray distance between two boundaries along x and z
    float deltaZ;
    float cellHeight;
    CloudCurve curve;
};

// The next altitude boundary of cell y after tFrom, and which one it is.
void CloudNextAltitudeBoundary(CloudData c, AIR_INOUT(CloudMarch) m, float tFrom)
{
    float lower = c.bottom + float(m.cellY) * m.cellHeight;
    float up = CloudCrossing(m.curve, lower + m.cellHeight, tFrom);
    float down = CloudCrossing(m.curve, lower, tFrom);
    m.stepY = up <= down ? 1 : -1;
    m.nextY = min(up, down);
}

// The DDA along one axis from the point at distance `start`: its cell, the step, the distance to
// the cell's far boundary and between boundaries.
void CloudAxis(float origin, float direction, float cellSize, float start, AIR_OUT(int) cell, AIR_OUT(int) step,
               AIR_OUT(float) next, AIR_OUT(float) delta)
{
    cell = int(floor((origin + direction * start) / cellSize));
    if (direction > 0.0f)
    {
        step = 1;
        next = (float(cell + 1) * cellSize - origin) / direction;
        delta = cellSize / direction;
    }
    else if (direction < 0.0f)
    {
        step = -1;
        next = (float(cell) * cellSize - origin) / direction;
        delta = -cellSize / direction;
    }
    else
    {
        step = 0;
        next = 3.0e38f;
        delta = 3.0e38f;
    }
}

// Starts the march along o + t d over [0, tMax]; false when that misses the layer.
bool BeginCloudMarch(CloudData c, float3 o, float3 d, float tMax, AIR_OUT(CloudMarch) m)
{
    m.curve = MakeCloudCurve(o, d);
    float altitude = m.curve.c;
    // The first stretch of the ray in the layer (the altitude is convex along it: it may dip into the
    // layer from above and leave it below, but rises for good once it rises).
    float t0 = 0.0f;
    if (altitude < c.bottom)
    {
        t0 = CloudCrossing(m.curve, c.bottom, 0.0f);
    }
    else if (altitude > c.top)
    {
        t0 = CloudCrossing(m.curve, c.top, 0.0f);
    }
    float after = t0 + max(1e-3f, t0 * 1e-6f);
    float t1 = min(min(CloudCrossing(m.curve, c.bottom, after), CloudCrossing(m.curve, c.top, after)),
                   min(tMax, kCloudMaxDistance));
    m.t = t0;
    m.tEnd = t1;
    m.cellHeight = (c.top - c.bottom) / float(c.cellsY);
    m.cellX = 0;
    m.cellY = 0;
    m.cellZ = 0;
    m.stepX = 0;
    m.stepY = 0;
    m.stepZ = 0;
    m.nextX = 0.0f;
    m.nextY = 0.0f;
    m.nextZ = 0.0f;
    m.deltaX = 0.0f;
    m.deltaZ = 0.0f;
    if (c.enabled == 0u || !(t0 < t1))
    {
        m.tEnd = m.t;
        return false;
    }
    // Cells are anchored at (0, bottom, 0) of the drifting base; the altitude's cell at the middle of the
    // first step.
    CloudAxis(o.x - c.baseOffsetX, d.x, c.tileSize / float(c.cellsX), t0, m.cellX, m.stepX, m.nextX, m.deltaX);
    CloudAxis(o.z - c.baseOffsetZ, d.z, c.tileSize / float(c.cellsZ), t0, m.cellZ, m.stepZ, m.nextZ, m.deltaZ);
    float start = CloudAltitude(o + d * min(after, 0.5f * (t0 + t1)));
    m.cellY = min(max(int(floor((start - c.bottom) / m.cellHeight)), 0), int(c.cellsY) - 1);
    CloudNextAltitudeBoundary(c, m, after);
    return true;
}

// The next segment [t0, t1] of the march and the majorant over it (extinction per metre); false
// when the ray has left the layer.
bool NextCloudSegment(CloudData c, AIR_INOUT(CloudMarch) m, AIR_OUT(float) t0, AIR_OUT(float) t1,
                      AIR_OUT(float) majorant)
{
    t0 = m.t;
    t1 = m.t;
    majorant = 0.0f;
    if (m.t >= m.tEnd)
    {
        return false;
    }
    t1 = max(min(min(m.nextX, m.nextY), min(m.nextZ, m.tEnd)), m.t);
    // Rounding can put the DDA one cell off at a boundary; the majorants also bound the voxels
    // around their cell, which covers that.
    uint cy = uint(min(max(m.cellY, 0), int(c.cellsY) - 1));
    majorant = CloudMajorantLoad(c, (cy * c.cellsZ + CloudWrap(m.cellZ, c.cellsZ)) * c.cellsX + CloudWrap(m.cellX, c.cellsX));
    if (m.nextX <= m.nextY && m.nextX <= m.nextZ)
    {
        m.cellX += m.stepX;
        m.nextX += m.deltaX;
    }
    else if (m.nextY <= m.nextZ)
    {
        float crossed = m.nextY;
        m.cellY += m.stepY;
        CloudNextAltitudeBoundary(c, m, crossed + max(1e-3f, crossed * 1e-6f));
    }
    else
    {
        m.cellZ += m.stepZ;
        m.nextZ += m.deltaZ;
    }
    m.t = t1;
    return true;
}

// The first real collision with the clouds along o + t d within [0, tMax], by delta tracking, then
// with the cirrus above them (a ray meets the sheet after the layer: it starts below or in the layer,
// or at the sheet itself heading away from it).
bool CloudCollision(CloudData c, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed, AIR_OUT(float) tHit)
{
    tHit = tMax;
    CloudMarch m;
    if (BeginCloudMarch(c, o, d, tMax, m))
    {
        float t0;
        float t1;
        float majorant;
        while (NextCloudSegment(c, m, t0, t1, majorant))
        {
            if (majorant <= 0.0f)
            {
                continue;
            }
            float t = t0;
            for (;;)
            {
                t -= log(1.0f - Rand(seed)) / majorant;
                if (t >= t1)
                {
                    break;
                }
                if (Rand(seed) * majorant < c.extinction * CloudDensity(c, o + d * t))
                {
                    tHit = t;
                    return true;
                }
            }
        }
    }
    float t;
    float depth;
    if (CirrusCrossing(c, o, d, 1.0f, tMax, t, depth) && Rand(seed) >= exp(-depth))
    {
        tHit = t;
        return true;
    }
    return false;
}

// Transmittance through the clouds along o + t d over [0, tMax], by ratio tracking (and through the
// cirrus, exactly); Russian roulette ends it once it falls below 0.1 (unbiased).
float CloudTransmittance(CloudData c, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed)
{
    float transmittance = CirrusTransmittance(c, o, d, tMax);
    CloudMarch m;
    if (!BeginCloudMarch(c, o, d, tMax, m))
    {
        return transmittance;
    }
    float t0;
    float t1;
    float majorant;
    while (NextCloudSegment(c, m, t0, t1, majorant))
    {
        if (majorant <= 0.0f)
        {
            continue;
        }
        float t = t0;
        for (;;)
        {
            t -= log(1.0f - Rand(seed)) / majorant;
            if (t >= t1)
            {
                break;
            }
            transmittance *= max(1.0f - c.extinction * CloudDensity(c, o + d * t) / majorant, 0.0f);
            if (transmittance < 0.1f)
            {
                if (Rand(seed) >= transmittance * 10.0f)
                {
                    return 0.0f;
                }
                transmittance = 0.1f;
            }
        }
    }
    return transmittance;
}

// --- Phase function ---------------------------------------------------------------------------------
// Henyey-Greenstein for the diffraction peak blended with Draine's function for the rest, fitted to
// Mie scattering by water droplets (Jendersie and d'Eon 2023). u: cosine of the scattering angle
// between the directions light travels in before and after.

float HgPhase(float g, float u)
{
    float d = 1.0f + g * g - 2.0f * g * u;
    return (1.0f - g * g) / (4.0f * kCloudPi * d * sqrt(d));
}

float DrainePhase(float g, float alpha, float u)
{
    return HgPhase(g, u) * (1.0f + alpha * u * u) / (1.0f + alpha * (1.0f + 2.0f * g * g) / 3.0f);
}

float CloudPhase(CloudData c, float u)
{
    return (1.0f - c.wDraine) * HgPhase(c.gHg, u) + c.wDraine * DrainePhase(c.gDraine, c.alphaDraine, u);
}

float SampleHgCosine(float g, float xi)
{
    if (abs(g) < 1e-3f)
    {
        return 2.0f * xi - 1.0f;
    }
    float t = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi);
    return clamp((1.0f + g * g - t * t) / (2.0f * g), -1.0f, 1.0f);
}

// Exact inversion of the CDF of Draine's function (Jendersie and d'Eon 2023, supplemental).
float SampleDraineCosine(float g, float alpha, float xi)
{
    float g2 = g * g;
    float g4 = g2 * g2;
    float t0 = alpha - alpha * g2;
    float t1 = alpha * g4 - alpha;
    float t2 = -3.0f * (4.0f * (g4 - g2) + t1 * (1.0f + g2));
    float t3 = g * (2.0f * xi - 1.0f);
    float t4 = 3.0f * g2 * (1.0f + t3) + alpha * (2.0f + g2 * (1.0f + (1.0f + 2.0f * g2) * t3));
    float t5 = t0 * (t1 * t2 + t4 * t4) + t1 * t1 * t1;
    float t6 = t0 * 4.0f * (g4 - g2);
    float t7 = pow(t5 + sqrt(max(t5 * t5 - t6 * t6 * t6, 0.0f)), 1.0f / 3.0f);
    float t8 = 2.0f * (t1 + t6 / t7 + t7) / t0;
    float t9 = sqrt(max(6.0f * (1.0f + g2) + t8, 0.0f));
    float s = sqrt(max(6.0f * (1.0f + g2) - t8 + 8.0f * t4 / (t0 * t9), 0.0f)) - t9;
    return clamp(g / 2.0f + (1.0f / (2.0f * g) - s * s / (8.0f * g)), -1.0f, 1.0f);
}

// A direction scattered from `direction` (both the way light travels), distributed as CloudPhase.
float3 SampleCloudPhase(CloudData c, float3 direction, float3 u)
{
    float cosTheta = u.x < c.wDraine ? SampleDraineCosine(c.gDraine, c.alphaDraine, u.y) : SampleHgCosine(c.gHg, u.y);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * kCloudPi * u.z;
    // Orthonormal basis around the direction (Duff et al. 2017).
    float3 w = direction;
    float s = w.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + w.z);
    float b = w.x * w.y * a;
    float3 t = float3(1.0f + s * w.x * w.x * a, s * b, -s * w.x);
    float3 bt = float3(b, s + w.y * w.y * a, -w.y);
    return t * (sinTheta * cos(phi)) + bt * (sinTheta * sin(phi)) + w * cosTheta;
}

#endif
