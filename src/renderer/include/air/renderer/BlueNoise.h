#pragma once

// Blue noise: a tileable texture whose values are a permutation of ranks with no low frequencies, so
// that thresholding it at any level gives evenly spread points (dithering, the jitter of ray marches).
// Made by Ulichney's void-and-cluster method ("The void-and-cluster method for dither array
// generation", 1993): points are ranked one by one, each in the largest void (or, among the first
// ones, removed from the tightest cluster) of a Gaussian energy on the torus.

#include "air/core/Types.h"

#include <vector>

namespace air::renderer
{

// size x size ranks 0 .. size^2 - 1 (row major), deterministic for a seed. size <= 256. About
// 0.1 s for 64 x 64 in a debug build.
std::vector<u16> MakeBlueNoise(u32 size, u32 seed);

// The 64 x 64 texture every renderer shares (seed 1), made on first use; thread safe.
inline constexpr u32 kBlueNoiseSize = 64;
const std::vector<u16>& BlueNoise();

} // namespace air::renderer
