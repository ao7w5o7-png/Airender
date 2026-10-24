// A volume inside the scene (smoke, a cloud from a file), shared by C++ and HLSL (see HlslCompat.h):
// a box of voxels holding a density, scattering and absorbing light. renderer::BuildVolume makes it
// from an OpenVDB grid or a procedural plume; VolumeLookup.hlsli tracks rays through it.
// docs/architecture.md §8.12.
//
// Layout: the density texture holds sizeX x sizeY x sizeZ voxels plus a zero voxel on every side, so
// that sampling with clamping reads zero outside the box; texel (i + 1, j + 1, k + 1) is the voxel
// whose centre is at origin + (i + 0.5, j + 0.5, k + 0.5) * voxelSize. Majorant cells cover
// kVolumeCell voxels of the padded grid per axis, cell (x, y, z) at index (z * cellsY + y) * cellsX + x.
#ifndef AIR_RENDERER_VOLUME_SHARED_H
#define AIR_RENDERER_VOLUME_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kVolumeCell = 8; // voxels per majorant cell and axis

struct VolumeData
{
    float3 origin;   // world position of the voxels' corner (the box without its padding)
    float voxelSize; // metres
    uint sizeX;      // voxels, without the padding
    uint sizeY;
    uint sizeZ;
    uint enabled;     // 0: no volume (and no resources)
    float extinction; // per metre at density 1
    float albedo;     // single-scattering albedo
    float anisotropy; // Henyey-Greenstein g
    uint cellsX;      // majorant cells
    uint cellsY;
    uint cellsZ;
    uint density;   // bindless Texture3D<float> (GPU only)
    uint majorants; // bindless StructuredBuffer<float>: extinction per metre bounding each cell (GPU only)
};

#ifdef __cplusplus
static_assert(sizeof(VolumeData) == 64);
} // namespace air::gpu
#endif

#endif
