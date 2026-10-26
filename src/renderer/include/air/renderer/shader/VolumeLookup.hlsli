// The scene's volume (VolumeShared.h): its density, free-flight sampling and transmittance through
// it, and its phase function. Shared by C++ and HLSL; the including code includes Sampling.hlsli
// (Rand) and CloudLookup.hlsli (the DDA axis, Henyey-Greenstein), and defines, before including this
// file:
//   float VolumeDensitySample(VolumeData v, float3 uvw)   the density texture, trilinear, clamped (0..1)
//   float VolumeMajorantLoad(VolumeData v, uint cell)     extinction per metre bounding the cell
//
// Like the clouds' (CloudLookup.hlsli): a 3D DDA walks the majorant cells the ray crosses inside the
// box, skipping empty ones whole; delta tracking finds the first real collision, ratio tracking
// estimates transmittance, both unbiased because no density exceeds its cell's majorant.
#ifndef AIR_RENDERER_VOLUME_LOOKUP_HLSLI
#define AIR_RENDERER_VOLUME_LOOKUP_HLSLI

#include "air/renderer/shader/VolumeShared.h"

// Density (0..1) at a point.
float VolumeDensity(VolumeData v, float3 p)
{
    float3 q = (p - v.origin) / v.voxelSize; // voxels from the box's corner
    float3 uvw = float3((q.x + 1.0f) / float(v.sizeX + 2u), (q.y + 1.0f) / float(v.sizeY + 2u),
                        (q.z + 1.0f) / float(v.sizeZ + 2u));
    return VolumeDensitySample(v, uvw);
}

struct VolumeMarch
{
    float t;    // start of the next segment
    float tEnd; // where the ray leaves the box (or stops)
    int cellX;  // current cell
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

// The ray's interval inside [lo, hi] along one axis, intersected with [t0, t1].
void VolumeSlab(float origin, float direction, float lo, float hi, AIR_INOUT(float) t0, AIR_INOUT(float) t1)
{
    if (abs(direction) < 1e-12f)
    {
        if (origin < lo || origin > hi)
        {
            t1 = -1.0f; // parallel and outside
        }
        return;
    }
    float ta = (lo - origin) / direction;
    float tb = (hi - origin) / direction;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
}

// Starts the march along o + t d over [0, tMax]; false when that misses the volume.
bool BeginVolumeMarch(VolumeData v, float3 o, float3 d, float tMax, AIR_OUT(VolumeMarch) m)
{
    m.t = 0.0f;
    m.tEnd = 0.0f;
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
    if (v.enabled == 0u)
    {
        return false;
    }
    // The padded box: the grid's corner is one voxel out on every side.
    float3 lo = v.origin - float3(v.voxelSize, v.voxelSize, v.voxelSize);
    float3 extent = float3(float(v.sizeX + 2u), float(v.sizeY + 2u), float(v.sizeZ + 2u)) * v.voxelSize;
    float t0 = 0.0f;
    float t1 = tMax;
    VolumeSlab(o.x, d.x, lo.x, lo.x + extent.x, t0, t1);
    VolumeSlab(o.y, d.y, lo.y, lo.y + extent.y, t0, t1);
    VolumeSlab(o.z, d.z, lo.z, lo.z + extent.z, t0, t1);
    m.t = t0;
    m.tEnd = t1;
    if (t0 >= t1)
    {
        return false;
    }
    float cellSize = float(kVolumeCell) * v.voxelSize;
    CloudAxis(o.x - lo.x, d.x, cellSize, t0, m.cellX, m.stepX, m.nextX, m.deltaX);
    CloudAxis(o.y - lo.y, d.y, cellSize, t0, m.cellY, m.stepY, m.nextY, m.deltaY);
    CloudAxis(o.z - lo.z, d.z, cellSize, t0, m.cellZ, m.stepZ, m.nextZ, m.deltaZ);
    return true;
}

// The next segment [t0, t1] of the march and the majorant over it (extinction per metre); false
// when the ray has left the volume.
bool NextVolumeSegment(VolumeData v, AIR_INOUT(VolumeMarch) m, AIR_OUT(float) t0, AIR_OUT(float) t1,
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
    // Rounding can put the DDA one cell off at a boundary (or out of the grid at the box's faces);
    // the majorants also bound the voxels around their cell, which covers that.
    uint cx = uint(min(max(m.cellX, 0), int(v.cellsX) - 1));
    uint cy = uint(min(max(m.cellY, 0), int(v.cellsY) - 1));
    uint cz = uint(min(max(m.cellZ, 0), int(v.cellsZ) - 1));
    majorant = VolumeMajorantLoad(v, (cz * v.cellsY + cy) * v.cellsX + cx);
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

// The first real collision with the volume along o + t d within [0, tMax], by delta tracking.
bool VolumeCollision(VolumeData v, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed, AIR_OUT(float) tHit)
{
    tHit = tMax;
    VolumeMarch m;
    if (!BeginVolumeMarch(v, o, d, tMax, m))
    {
        return false;
    }
    float t0;
    float t1;
    float majorant;
    while (NextVolumeSegment(v, m, t0, t1, majorant))
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
            if (Rand(seed) * majorant < v.extinction * VolumeDensity(v, o + d * t))
            {
                tHit = t;
                return true;
            }
        }
    }
    return false;
}

// Transmittance through the volume along o + t d over [0, tMax], by ratio tracking; Russian
// roulette ends it once it falls below 0.1 (unbiased).
float VolumeTransmittance(VolumeData v, float3 o, float3 d, float tMax, AIR_INOUT(uint) seed)
{
    VolumeMarch m;
    if (!BeginVolumeMarch(v, o, d, tMax, m))
    {
        return 1.0f;
    }
    float transmittance = 1.0f;
    float t0;
    float t1;
    float majorant;
    while (NextVolumeSegment(v, m, t0, t1, majorant))
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
            transmittance *= max(1.0f - v.extinction * VolumeDensity(v, o + d * t) / majorant, 0.0f);
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

// Henyey-Greenstein; u is the cosine between the directions light travels in before and after.
float VolumePhase(VolumeData v, float u)
{
    return HgPhase(v.anisotropy, u);
}

// A direction scattered from `direction` (the way light travels), distributed as VolumePhase.
float3 SampleVolumePhase(VolumeData v, float3 direction, float2 u)
{
    float cosTheta = SampleHgCosine(v.anisotropy, u.x);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * kCloudPi * u.y;
    float3 w = direction;
    float s = w.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + w.z);
    float b = w.x * w.y * a;
    float3 t = float3(1.0f + s * w.x * w.x * a, s * b, -s * w.x);
    float3 bt = float3(b, s + w.y * w.y * a, -w.y);
    return t * (sinTheta * cos(phi)) + bt * (sinTheta * sin(phi)) + w * cosTheta;
}

#endif
