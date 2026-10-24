#pragma once

// OpenVDB files (docs/architecture.md §8.12), read without the OpenVDB library: the grids a file
// holds, and a scalar grid's voxels for a Volume (Volume.h). Reads the file format from version 222
// (OpenVDB 2.0) on: trees of 5-4-3 nodes holding float, double or half values (or floats saved as
// halves), uncompressed or compressed with zlib or Blosc (LZ4), active-mask compressed or not, and
// transforms that scale uniformly and translate (OpenVDB's ScaleMap and ScaleTranslateMap, their
// uniform variants, TranslationMap, and AffineMap without rotation).

#include "air/core/Types.h"
#include "air/renderer/Volume.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace air::renderer
{

struct VdbGridInfo
{
    std::string name;
    std::string valueType;  // of the tree: "float", "double", "vec3s", ...
    std::string nodes;      // the tree's node sizes: "5_4_3"
    bool halfFloat = false; // values stored as 16-bit floats
    std::string gridClass;  // "fog volume", "level set", ... (the grid's "class" metadata; may be empty)
    bool hasBounds = false; // the file stores the active voxels' index bounds
    i32 boundsMin[3] = {};
    i32 boundsMax[3] = {};
    u64 activeVoxels = 0;                  // when the file stores the count
    std::string map;                       // the transform's map type
    double voxelSize[3] = {1.0, 1.0, 1.0}; // world units per voxel along each index axis
    double translation[3] = {};            // world position of the voxel at index 0
    bool readable = false;                 // ReadVdbGrid can read it
    std::string why;                       // why not
};

struct VdbFileInfo
{
    u32 version = 0; // of the file format
    u32 libraryMajor = 0;
    u32 libraryMinor = 0;
    std::vector<VdbGridInfo> grids;
};

std::optional<VdbFileInfo> ReadVdbInfo(const std::filesystem::path& path, std::string* error);

// A scalar grid's voxels over the bounds of its active voxels, averaged over blocks of factor^3
// voxels (voxels outside the bounds count as empty) so that its longest side has at most
// maxResolution voxels, in world units (VolumeGrid). Negative values read as 0. The grid named
// `name`; when there is none and `name` is "density" or empty, the file's first readable grid.
std::optional<VolumeGrid> ReadVdbGrid(const std::filesystem::path& path, const std::string& name, u32 maxResolution,
                                      std::string* error);

} // namespace air::renderer
