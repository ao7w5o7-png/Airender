#pragma once

// CPU image + comparison utilities for golden image tests.
//   * PFM (float, linear HDR) is the reference format: exact, no tonemapping, trivial to parse.
//   * PPM (8-bit) is for quick viewing and diff heatmaps.
// CompareImages() gives the metrics an AI agent or CI uses to decide pass/fail.

#include "air/core/Types.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace air {

struct Image {
    u32 width = 0;
    u32 height = 0;
    std::vector<float> rgba; // width * height * 4, linear, row 0 = top

    Image() = default;
    Image(u32 w, u32 h) : width(w), height(h), rgba(size_t(w) * h * 4, 0.0f) {}

    float* At(u32 x, u32 y) { return &rgba[(size_t(y) * width + x) * 4]; }
    const float* At(u32 x, u32 y) const { return &rgba[(size_t(y) * width + x) * 4]; }
    bool Empty() const { return width == 0 || height == 0; }
};

bool WritePfm(const std::filesystem::path& path, const Image& image);
std::optional<Image> ReadPfm(const std::filesystem::path& path);
// Clamps to [0,1] and applies sRGB encoding (or writes the values as they are when the image is
// already display-encoded, e.g. a tonemapped frame).
bool WritePpm(const std::filesystem::path& path, const Image& image, bool encodeSrgb = true);
std::optional<Image> ReadPpm(const std::filesystem::path& path);
// Picks the reader from the extension (.pfm / .ppm).
std::optional<Image> ReadImage(const std::filesystem::path& path);

struct ImageCompareOptions {
    float pixelThreshold = 1e-3f; // per-channel absolute difference that counts a pixel as different
    float maxRmse = 1e-3f;        // pass criteria
    float maxDifferingFraction = 0.0f;
};

struct ImageCompareResult {
    bool sizeMismatch = false;
    double rmse = 0.0;
    double psnr = 0.0; // dB, +inf when identical
    double maxAbsDiff = 0.0;
    u64 differingPixels = 0;
    double differingFraction = 0.0;
    bool passed = false;
    Image diff; // per-pixel |a-b| scaled for visibility (red heatmap)

    std::string Summary() const;
};

ImageCompareResult CompareImages(const Image& reference, const Image& test, const ImageCompareOptions& options = {});

} // namespace air
