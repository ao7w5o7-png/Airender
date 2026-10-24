// Lookups in a volume's bricks (VolumeBricksShared.h), shared by C++ and HLSL: the density at a
// point, and whether the bricks around a point are empty (and how far that emptiness reaches) so a
// ray march can skip it. The including code defines, before including this file:
//   uint VolumeIndirectionLoad(VolumeBrickLevel l, uint index), VolumeOccupancyLoad(...),
//   uint VolumeBrickWordLoad(VolumeBrickLevel l, uint index), VolumeRangeLoad(VolumeBrickLevel l, uint brick)
// (StructuredBuffer reads on the GPU, the tables of renderer::VolumeBricks on the CPU).
#ifndef AIR_RENDERER_VOLUME_BRICKS_HLSLI
#define AIR_RENDERER_VOLUME_BRICKS_HLSLI

#include "air/renderer/shader/VolumeBricksShared.h"

// Voxel v (0..124) of a brick as its 4-bit code.
float BrickCode(VolumeBrickLevel l, uint brick, uint voxel)
{
    uint word = VolumeBrickWordLoad(l, brick * kBrickWords + (voxel >> 3u));
    return float((word >> ((voxel & 7u) * 4u)) & 15u);
}

// The density at world point p: trilinear over the level's voxels, zero outside them.
float BrickDensity(VolumeBrickLevel l, float3 p)
{
    float qx = (p.x - l.corner.x) / l.voxelSize - 0.5f; // voxel centres at whole numbers
    float qy = (p.y - l.corner.y) / l.voxelSize - 0.5f;
    float qz = (p.z - l.corner.z) / l.voxelSize - 0.5f;
    float fx = floor(qx);
    float fy = floor(qy);
    float fz = floor(qz);
    // The padded grid's outer voxels are zero: lookups that reach no further in read zero.
    if (fx < 0.0f || fy < 0.0f || fz < 0.0f || fx >= float(l.voxelsX - 1u) || fy >= float(l.voxelsY - 1u) ||
        fz >= float(l.voxelsZ - 1u))
    {
        return 0.0f;
    }
    uint ix = uint(fx);
    uint iy = uint(fy);
    uint iz = uint(fz);
    uint brick = VolumeIndirectionLoad(l, ((iz >> 2u) * l.bricksY + (iy >> 2u)) * l.bricksX + (ix >> 2u));
    if (brick == kNoBrick)
    {
        return 0.0f;
    }
    uint row = kBrickStored;
    uint slice = kBrickStored * kBrickStored;
    uint v = (ix & 3u) + row * (iy & 3u) + slice * (iz & 3u);
    float tx = qx - fx;
    float ty = qy - fy;
    float tz = qz - fz;
    float c00 = lerp(BrickCode(l, brick, v), BrickCode(l, brick, v + 1u), tx);
    float c10 = lerp(BrickCode(l, brick, v + row), BrickCode(l, brick, v + row + 1u), tx);
    float c01 = lerp(BrickCode(l, brick, v + slice), BrickCode(l, brick, v + slice + 1u), tx);
    float c11 = lerp(BrickCode(l, brick, v + slice + row), BrickCode(l, brick, v + slice + row + 1u), tx);
    float code = lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
    uint range = VolumeRangeLoad(l, brick);
    float lowest = f16tof32(range & 0xFFFFu);
    float highest = f16tof32(range >> 16u);
    return lowest + (highest - lowest) * (code * (1.0f / 15.0f));
}

// Whether a lookup at p may read a density above zero. When it cannot, emptyMin and emptyMax bound
// the region around p whose lookups cannot either (the lookups of p's brick, or of the 4 x 4 x 2
// bricks of its occupancy word when they are all empty), so a march can jump to where its ray leaves
// it. Outside the voxels the region is p itself.
bool BrickOccupied(VolumeBrickLevel l, float3 p, AIR_OUT(float3) emptyMin, AIR_OUT(float3) emptyMax)
{
    emptyMin = p;
    emptyMax = p;
    float fx = floor((p.x - l.corner.x) / l.voxelSize - 0.5f);
    float fy = floor((p.y - l.corner.y) / l.voxelSize - 0.5f);
    float fz = floor((p.z - l.corner.z) / l.voxelSize - 0.5f);
    if (fx < 0.0f || fy < 0.0f || fz < 0.0f || fx >= float(l.voxelsX - 1u) || fy >= float(l.voxelsY - 1u) ||
        fz >= float(l.voxelsZ - 1u))
    {
        return false;
    }
    uint bx = uint(fx) >> 2u;
    uint by = uint(fy) >> 2u;
    uint bz = uint(fz) >> 2u;
    uint wx = bx / kOccupancyX;
    uint wy = by / kOccupancyY;
    uint wz = bz / kOccupancyZ;
    uint word = VolumeOccupancyLoad(l, (wz * l.wordsY + wy) * l.wordsX + wx);
    uint bit = (bx % kOccupancyX) + kOccupancyX * (by % kOccupancyY) + kOccupancyX * kOccupancyY * (bz % kOccupancyZ);
    if (((word >> bit) & 1u) != 0u)
    {
        return true;
    }
    // Lookups whose lowest voxel lies in brick b sit in [4 b + 0.5, 4 b + 4.5) voxels from the corner.
    float lo0;
    float lo1;
    float lo2;
    float size0;
    float size1;
    float size2;
    if (word == 0u)
    {
        lo0 = float(wx * kOccupancyX * kBrickVoxels);
        lo1 = float(wy * kOccupancyY * kBrickVoxels);
        lo2 = float(wz * kOccupancyZ * kBrickVoxels);
        size0 = float(kOccupancyX * kBrickVoxels);
        size1 = float(kOccupancyY * kBrickVoxels);
        size2 = float(kOccupancyZ * kBrickVoxels);
    }
    else
    {
        lo0 = float(bx * kBrickVoxels);
        lo1 = float(by * kBrickVoxels);
        lo2 = float(bz * kBrickVoxels);
        size0 = float(kBrickVoxels);
        size1 = size0;
        size2 = size0;
    }
    emptyMin = l.corner + float3(lo0 + 0.5f, lo1 + 0.5f, lo2 + 0.5f) * l.voxelSize;
    emptyMax = emptyMin + float3(size0, size1, size2) * l.voxelSize;
    return false;
}

// Where a ray from o (inside the box [lo, hi]) along d leaves it.
float BoxExit(float3 o, float3 d, float3 lo, float3 hi)
{
    float t = 1e30f;
    if (abs(d.x) > 1e-12f)
    {
        t = min(t, ((d.x > 0.0f ? hi.x : lo.x) - o.x) / d.x);
    }
    if (abs(d.y) > 1e-12f)
    {
        t = min(t, ((d.y > 0.0f ? hi.y : lo.y) - o.y) / d.y);
    }
    if (abs(d.z) > 1e-12f)
    {
        t = min(t, ((d.z > 0.0f ? hi.z : lo.z) - o.z) / d.z);
    }
    return max(t, 0.0f);
}

// The part [t0, t1] of the ray o + t d (t in [tMin, tMax]) inside the box [lo, hi]; false when none.
bool BoxSegment(float3 o, float3 d, float3 lo, float3 hi, float tMin, float tMax, AIR_OUT(float) t0, AIR_OUT(float) t1)
{
    t0 = tMin;
    t1 = tMax;
    if (abs(d.x) > 1e-12f)
    {
        float ta = (lo.x - o.x) / d.x;
        float tb = (hi.x - o.x) / d.x;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    else if (o.x < lo.x || o.x > hi.x)
    {
        return false;
    }
    if (abs(d.y) > 1e-12f)
    {
        float ta = (lo.y - o.y) / d.y;
        float tb = (hi.y - o.y) / d.y;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    else if (o.y < lo.y || o.y > hi.y)
    {
        return false;
    }
    if (abs(d.z) > 1e-12f)
    {
        float ta = (lo.z - o.z) / d.z;
        float tb = (hi.z - o.z) / d.z;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    else if (o.z < lo.z || o.z > hi.z)
    {
        return false;
    }
    return t0 < t1;
}

#endif
