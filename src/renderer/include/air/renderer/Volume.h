#pragma once

// A volume inside the scene (docs/architecture.md §8.12): smoke, or a cloud of voxels from an OpenVDB
// file (Vdb.h), which absorbs and scatters light (Henyey-Greenstein). Its density is uploaded as a
// half-float 3D texture that the GPU path tracer (hardware trilinear filtering) and the CPU reference
// (the same filter, emulated) read through shader/VolumeLookup.hlsli, with a coarse grid of
// majorants for unbiased delta and ratio tracking.

#include "air/core/Types.h"
#include "air/renderer/shader/HlslCompat.h"
#include "air/renderer/shader/VolumeShared.h"

#include <optional>
#include <string>
#include <vector>

namespace air::renderer
{

struct Volume
{
    bool enabled = false;
    std::string file;             // an OpenVDB file; empty: a procedural plume of smoke
    std::string grid = "density"; // the grid to read from the file (its first float grid if absent)
    // Placement: the voxels' box is scaled to `size` metres along its longest side and centred on
    // `position` horizontally, its bottom at position.y; size 0 keeps the file's own placement,
    // moved by `position`.
    hlsl::float3 position = {0.0f, 0.0f, 0.0f};
    float size = 0.0f;
    float extinction = 20.0f; // per metre where the density is 1
    float albedo = 0.9f;      // single scattering (smoke: 0.5-0.95; clouds: 1)
    float anisotropy = 0.0f;  // Henyey-Greenstein g: > 0 forward (clouds about 0.8)
    u32 maxResolution = 256;  // longest side in voxels on the GPU (8 to 2048); larger grids are averaged down
    u32 seed = 1;             // of the procedural plume

    bool operator==(const Volume&) const = default;
};

// Densities on a grid of sizeX x sizeY x sizeZ voxels (x fastest) of `voxelSize` metres, the first
// voxel's corner at `origin`.
struct VolumeGrid
{
    std::vector<float> density;
    u32 sizeX = 0;
    u32 sizeY = 0;
    u32 sizeZ = 0;
    hlsl::float3 origin = {0.0f, 0.0f, 0.0f};
    float voxelSize = 1.0f;

    float At(u32 x, u32 y, u32 z) const { return density[(size_t(z) * sizeY + y) * sizeX + x]; }
};

// A rising plume of smoke, 1.6 m wide and 3.2 m tall at 2.5 cm voxels, from seeded noise; its bottom
// is centred on the origin.
VolumeGrid MakeSmokePlume(u32 seed);

// The grid averaged over blocks of `factor`^3 voxels (larger voxels, the same mean density).
VolumeGrid Downsample(const VolumeGrid& grid, u32 factor);

struct VolumeResources
{
    gpu::VolumeData data{}; // enabled == 0: no volume (and no texels); no bindless indices
    // R16_FLOAT texels, the voxels with a zero voxel on every side (VolumeShared.h), x fastest.
    std::vector<u16> density;
    std::vector<float> majorants; // cellsX * cellsY * cellsZ
    std::string error;            // why a file gave no volume
};

// The volume's voxels before they are placed: its file's grid, averaged down to maxResolution while
// it is read, or the plume. None, and why in `error`, when the file cannot be read.
std::optional<VolumeGrid> LoadVolumeGrid(const Volume& volume, std::string* error);

// Reads the file (or makes the plume), averages it down to maxResolution, places it and computes the
// majorants.
VolumeResources BuildVolume(const Volume& volume);
// The same for a grid at hand.
VolumeResources BuildVolume(const Volume& volume, const VolumeGrid& grid);

} // namespace air::renderer
