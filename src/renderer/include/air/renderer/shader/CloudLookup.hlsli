// Clouds (CloudShared.h): their density, free-flight sampling and transmittance through them, and
// their phase function. Shared by C++ and HLSL; the including code includes Sampling.hlsli (Rand)
// and defines, before including this file:
//   float CloudBaseSample(CloudData c, float3 uvw)    the base texture, trilinear, wrapping (0..1)
//   float CloudDetailSample(CloudData c, float3 uvw)  the detail texture, likewise
//   float CloudMajorantLoad(CloudData c, uint cell)   extinction per metre bounding the cell
//
// Tracking walks the majorant cells along the ray (a 3D DDA over the periodic grid, which skips
// empty cells whole) and samples tentative collisions in each at the rate of its majorant: delta
// tracking finds the first real collision, ratio tracking estimates transmittance. Both are
// unbiased because no density exceeds its cell's majorant.
#ifndef AIR_RENDERER_CLOUD_LOOKUP_HLSLI
#define AIR_RENDERER_CLOUD_LOOKUP_HLSLI

#include "air/renderer/shader/CloudShared.h"

AIR_CONST float kCloudPi = 3.14159265358979f;
AIR_CONST float kCloudMaxDistance = 100000.0f; // metres along a ray; the layer is flat, not curved

uint CloudWrap(int i, uint n)
{
    int m = i % int(n);
    return uint(m < 0 ? m + int(n) : m);
}

// Density (0..1) at a point: the base texture, its fringe eroded by the detail texture, sharpened.
// The erosion never raises the density and the sharpening is monotonic, so the majorants of the
// sharpened base texture bound it.
float CloudDensity(CloudData c, float3 p)
{
    float h = (p.y - c.bottom) / (c.top - c.bottom);
    if (h < 0.0f || h > 1.0f)
    {
        return 0.0f;
    }
    float3 uvw = float3(frac(p.x / c.tileSize), frac(p.z / c.tileSize),
                        (h * float(c.gridY) + 1.0f) / float(c.gridY + 2u));
    float base = CloudBaseSample(c, uvw);
    if (base <= 0.0f || c.detailStrength <= 0.0f)
    {
        return saturate(base * c.sharpness);
    }
    float detail = CloudDetailSample(
        c, float3(frac(p.x / c.detailPeriod), frac(p.z / c.detailPeriod), frac(p.y / c.detailPeriod)));
    float erosion = detail * c.detailStrength * (1.0f - base);
    return saturate((base - erosion) / (1.0f - erosion) * c.sharpness);
}

// --- Majorant cells along a ray --------------------------------------------------------------------

struct CloudMarch
{
    float t;    // start of the next segment
    float tEnd; // where the ray leaves the layer (or stops)
    int cellX;  // current cell, not wrapped
    int cellY;
    int cellZ;
    int stepX;
    int stepY;
    int stepZ;
    float nextX; // ray distance to the next cell boundary on each axis
    float nextY;
    float nextZ;
    float deltaX; // ray distance between two boundaries on each axis
    float deltaY;
    float deltaZ;
};

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
    float t0 = 0.0f;
    float t1 = min(tMax, kCloudMaxDistance);
    if (abs(d.y) < 1e-9f)
    {
        if (o.y < c.bottom || o.y > c.top)
        {
            t1 = 0.0f;
        }
    }
    else
    {
        float ta = (c.bottom - o.y) / d.y;
        float tb = (c.top - o.y) / d.y;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    m.t = t0;
    m.tEnd = t1;
    if (c.enabled == 0u || t0 >= t1)
    {
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
        m.deltaY = 0.0f;
        m.deltaZ = 0.0f;
        return false;
    }
    // Cells are anchored at (0, bottom, 0).
    CloudAxis(o.x, d.x, c.tileSize / float(c.cellsX), t0, m.cellX, m.stepX, m.nextX, m.deltaX);
    CloudAxis(o.y - c.bottom, d.y, (c.top - c.bottom) / float(c.cellsY), t0, m.cellY, m.stepY, m.nextY, m.deltaY);
    CloudAxis(o.z, d.z, c.tileSize / float(c.cellsZ), t0, m.cellZ, m.stepZ, m.nextZ, m.deltaZ);
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
        m.cellY += m.stepY;
        m.nextY += m.deltaY;
    }
    else
    {
        m.cellZ += m.stepZ;
        m.nextZ += m.deltaZ;
    }
    m.t = t1;
    return true;
}

// The first real collision with the clouds along o + t d within [0, tMax], by delta tracking.
bool CloudCollision(CloudData c, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed, AIR_OUT(float) tHit)
{
    tHit = tMax;
    CloudMarch m;
    if (!BeginCloudMarch(c, o, d, tMax, m))
    {
        return false;
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
            if (Rand(seed) * majorant < c.extinction * CloudDensity(c, o + d * t))
            {
                tHit = t;
                return true;
            }
        }
    }
    return false;
}

// Transmittance through the clouds along o + t d over [0, tMax], by ratio tracking; Russian
// roulette ends it once it falls below 0.1 (unbiased).
float CloudTransmittance(CloudData c, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed)
{
    CloudMarch m;
    if (!BeginCloudMarch(c, o, d, tMax, m))
    {
        return 1.0f;
    }
    float transmittance = 1.0f;
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
