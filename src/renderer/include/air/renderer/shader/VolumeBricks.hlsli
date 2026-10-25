// Lookups in a volume's bricks (VolumeBricksShared.h), shared by C++ and HLSL: the density at a
// point, and whether the bricks around a point are empty (and how far that emptiness reaches) so a
// ray march can skip it. The including code defines, before including this file:
//   uint VolumeIndirectionLoad(VolumeBrickLevel l, uint index), VolumeDistanceLoad(...),
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

// Where a lookup at a point reads a level: the voxel of the lowest of the 8 it reads and the weights
// of the next ones.
struct BrickCell
{
    uint ix;
    uint iy;
    uint iz;
    float tx;
    float ty;
    float tz;
};

// The cell of world point p; false where the lookup reaches outside the voxels (the padded grid's
// outer voxels are zero: lookups that reach no further in read zero).
bool BrickLocate(VolumeBrickLevel l, float3 p, AIR_OUT(BrickCell) c)
{
    float qx = (p.x - l.corner.x) / l.voxelSize - 0.5f; // voxel centres at whole numbers
    float qy = (p.y - l.corner.y) / l.voxelSize - 0.5f;
    float qz = (p.z - l.corner.z) / l.voxelSize - 0.5f;
    float fx = floor(qx);
    float fy = floor(qy);
    float fz = floor(qz);
    bool inside = fx >= 0.0f && fy >= 0.0f && fz >= 0.0f && fx < float(l.voxelsX - 1u) && fy < float(l.voxelsY - 1u) &&
                  fz < float(l.voxelsZ - 1u);
    c.ix = inside ? uint(fx) : 0u;
    c.iy = inside ? uint(fy) : 0u;
    c.iz = inside ? uint(fz) : 0u;
    c.tx = qx - fx;
    c.ty = qy - fy;
    c.tz = qz - fz;
    return inside;
}

uint BrickPosition(VolumeBrickLevel l, BrickCell c)
{
    return ((c.iz >> 2u) * l.bricksY + (c.iy >> 2u)) * l.bricksX + (c.ix >> 2u);
}

// The density of allocated brick `brick` at cell c: trilinear over its codes, between its range.
float BrickRead(VolumeBrickLevel l, uint brick, BrickCell c)
{
    uint row = kBrickStored;
    uint slice = kBrickStored * kBrickStored;
    uint v = (c.ix & 3u) + row * (c.iy & 3u) + slice * (c.iz & 3u);
    float c00 = lerp(BrickCode(l, brick, v), BrickCode(l, brick, v + 1u), c.tx);
    float c10 = lerp(BrickCode(l, brick, v + row), BrickCode(l, brick, v + row + 1u), c.tx);
    float c01 = lerp(BrickCode(l, brick, v + slice), BrickCode(l, brick, v + slice + 1u), c.tx);
    float c11 = lerp(BrickCode(l, brick, v + slice + row), BrickCode(l, brick, v + slice + row + 1u), c.tx);
    float code = lerp(lerp(c00, c10, c.ty), lerp(c01, c11, c.ty), c.tz);
    uint range = VolumeRangeLoad(l, brick);
    float lowest = f16tof32(range & 0xFFFFu);
    float highest = f16tof32(range >> 16u);
    return lowest + (highest - lowest) * (code * (1.0f / 15.0f));
}

// The region around the lookups of empty brick position c whose lookups read zero too: those of the
// bricks closer to it than the nearest allocated one. Lookups whose lowest voxel lies in brick b sit
// in [4 b + 0.5, 4 b + 4.5) voxels from the corner.
void BrickEmptyRegion(VolumeBrickLevel l, BrickCell c, AIR_OUT(float3) emptyMin, AIR_OUT(float3) emptyMax)
{
    uint bx = c.ix >> 2u;
    uint by = c.iy >> 2u;
    uint bz = c.iz >> 2u;
    uint position = (bz * l.bricksY + by) * l.bricksX + bx;
    uint reach = ((VolumeDistanceLoad(l, position >> 2u) >> ((position & 3u) * 8u)) & 0xFFu) - 1u;
    uint x0 = bx > reach ? bx - reach : 0u;
    uint y0 = by > reach ? by - reach : 0u;
    uint z0 = bz > reach ? bz - reach : 0u;
    float3 bricks = float3(float(min(bx + reach, l.bricksX - 1u) - x0 + 1u), float(min(by + reach, l.bricksY - 1u) - y0 + 1u),
                           float(min(bz + reach, l.bricksZ - 1u) - z0 + 1u));
    emptyMin = l.corner + (float3(float(x0), float(y0), float(z0)) * float(kBrickVoxels) + 0.5f) * l.voxelSize;
    emptyMax = emptyMin + bricks * (float(kBrickVoxels) * l.voxelSize);
}

// The density at world point p: trilinear over the level's voxels, zero outside them.
float BrickDensity(VolumeBrickLevel l, float3 p)
{
    BrickCell c;
    if (!BrickLocate(l, p, c))
    {
        return 0.0f;
    }
    uint brick = VolumeIndirectionLoad(l, BrickPosition(l, c));
    return brick == kNoBrick ? 0.0f : BrickRead(l, brick, c);
}

// The density at p (as BrickDensity) and whether a lookup there may read one above zero: where it
// cannot, emptyMin and emptyMax bound the region around p whose lookups cannot either
// (BrickEmptyRegion), so a march can jump to where its ray leaves it. Outside the voxels the region
// is p itself.
float BrickSample(VolumeBrickLevel l, float3 p, AIR_OUT(bool) occupied, AIR_OUT(float3) emptyMin,
                  AIR_OUT(float3) emptyMax)
{
    occupied = false;
    emptyMin = p;
    emptyMax = p;
    BrickCell c;
    if (!BrickLocate(l, p, c))
    {
        return 0.0f;
    }
    uint brick = VolumeIndirectionLoad(l, BrickPosition(l, c));
    if (brick == kNoBrick)
    {
        BrickEmptyRegion(l, c, emptyMin, emptyMax);
        return 0.0f;
    }
    occupied = true;
    return BrickRead(l, brick, c);
}

// Whether a lookup at p may read a density above zero, and the empty region around it where it
// cannot (BrickSample without the density).
bool BrickOccupied(VolumeBrickLevel l, float3 p, AIR_OUT(float3) emptyMin, AIR_OUT(float3) emptyMax)
{
    emptyMin = p;
    emptyMax = p;
    BrickCell c;
    if (!BrickLocate(l, p, c))
    {
        return false;
    }
    if (VolumeIndirectionLoad(l, BrickPosition(l, c)) != kNoBrick)
    {
        return true;
    }
    BrickEmptyRegion(l, c, emptyMin, emptyMax);
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
