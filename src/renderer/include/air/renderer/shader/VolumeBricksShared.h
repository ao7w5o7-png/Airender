// The real-time view of a volume (docs/architecture.md §8.13), shared by C++ and HLSL (see
// HlslCompat.h): its voxels as sparse bricks of 4-bit voxels, three levels of detail and a coarse
// grid of how much ambient light reaches inside, as renderer::BuildVolumeBricks makes them. The ray
// marcher and the moment shadow maps read them through VolumeBricks.hlsli. The layout follows the one
// Mueller describes for a production volume renderer (SIGGRAPH 2026): per level an indirection grid,
// an occupancy bit per brick, and 4^3 voxels per brick with the voxels one past it on the positive
// side, so that trilinear filtering never leaves the brick.
//
// Voxel coordinates of a level count from its padded grid's corner: the volume's voxels with a zero
// voxel around them (the path tracer's texture, VolumeShared.h, is the same grid at level 0). Voxel
// (i, j, k) has its centre at corner + (i + 0.5, j + 0.5, k + 0.5) * voxelSize; a lookup reads the
// 8 voxels around it from the brick that holds the lowest of them. Brick (x, y, z) covers voxels
// 4x .. 4x + 3 per axis and stores 4x .. 4x + 4 (5^3 voxels, x fastest), each as 4 bits between the
// brick's lowest and highest density (half floats). A brick is allocated when any of its 125 voxels
// is not zero; the others read zero.
#ifndef AIR_RENDERER_VOLUME_BRICKS_SHARED_H
#define AIR_RENDERER_VOLUME_BRICKS_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kBrickVoxels = 4;       // voxels per brick and axis
AIR_CONST uint kBrickStored = 5;       // voxels stored per brick and axis
AIR_CONST uint kBrickWords = 16;       // 125 4-bit voxels in 16 words
AIR_CONST uint kNoBrick = 0xFFFFFFFFu; // indirection of an empty brick
AIR_CONST uint kOccupancyX = 4;        // bricks per occupancy word and axis: 4 x 4 x 2 = 32 bits
AIR_CONST uint kOccupancyY = 4;
AIR_CONST uint kOccupancyZ = 2;
AIR_CONST uint kVolumeLevels = 3; // level l has voxels 2^l times as large as level 0
AIR_CONST uint kVolumeAoCell = 8; // level 0 voxels per ambient occlusion texel and axis

struct VolumeBrickLevel
{
    float3 corner;   // world position of the padded grid's corner
    float voxelSize; // metres
    uint voxelsX;    // the padded grid
    uint voxelsY;
    uint voxelsZ;
    uint bricksX; // ceil(voxels / 4)
    uint bricksY;
    uint bricksZ;
    uint wordsX; // occupancy words: ceil(bricks / (4, 4, 2))
    uint wordsY;
    uint wordsZ;
    uint brickCount; // allocated
    // Bindless StructuredBuffer<uint>s on the GPU, indices into the CPU renderers' tables there:
    uint indirection; // per brick position (x fastest): its brick, kNoBrick when empty
    uint occupancy;   // per word: bit (x & 3) + 4 (y & 3) + 16 (z & 1) of the bricks it covers
    uint bricks;      // kBrickWords per brick: voxel v in bits 4 (v & 7) .. of word v >> 3
    uint ranges;      // per brick: lowest density (half) | highest (half) << 16
    uint pad0;
    uint pad1;
};

// What the ray marcher needs of a volume besides its levels.
struct VolumeMarchData
{
    float3 boundsMin; // world box around every lookup that can read a density above zero, at any level
    float extinction; // per metre at density 1
    float3 boundsMax;
    float albedo;     // single scattering
    float3 aoCorner;  // the ambient occlusion grid's corner: texel centres at corner + (i + 0.5) * aoCell
    float anisotropy; // Henyey-Greenstein g
    float aoCell;     // metres
    uint aoX;         // texels
    uint aoY;
    uint aoZ;
    uint ao;     // Texture3D<float> (R8_UNORM): the volume's mean transmittance over all directions to
                 // light that diffuses through it (VolumeBricks.cpp)
    uint levels; // its first entry in the level table, kVolumeLevels of them
    uint pad0;
    uint pad1;
};

#ifdef __cplusplus
static_assert(sizeof(VolumeBrickLevel) == 80);
static_assert(sizeof(VolumeMarchData) == 80);
} // namespace air::gpu
#endif

#endif
