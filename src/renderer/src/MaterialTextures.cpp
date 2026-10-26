// Procedural material textures (Scene.h): a height field per material whose slopes give the normal
// map, with colour, roughness and metallic laid out by the same pattern. Deterministic (integer hashes),
// so the images rendered from them match across platforms, and they repeat seamlessly.

#include "air/renderer/Scene.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace air::renderer
{
namespace
{

using hlsl::float3;

// A hash of a lattice point and a seed in [0, 1).
float Hash(int x, int y, u32 seed)
{
    u32 h = u32(x) * 0x8DA6B343u ^ u32(y) * 0xD8163841u ^ seed * 0xCB1AB31Fu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return float(h >> 8) * (1.0f / 16777216.0f);
}

// Value noise with a period of `period` cells, smoothly interpolated, at (x, y) in cells.
float Noise(float x, float y, int period, u32 seed)
{
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const int x0 = int(fx);
    const int y0 = int(fy);
    auto wrap = [&](int v) { return ((v % period) + period) % period; };
    auto smooth = [](float t) { return t * t * (3.0f - 2.0f * t); };
    const float tx = smooth(x - fx);
    const float ty = smooth(y - fy);
    const float a = Hash(wrap(x0), wrap(y0), seed);
    const float b = Hash(wrap(x0 + 1), wrap(y0), seed);
    const float c = Hash(wrap(x0), wrap(y0 + 1), seed);
    const float d = Hash(wrap(x0 + 1), wrap(y0 + 1), seed);
    return a + (b - a) * tx + (c - a) * ty + (a - b - c + d) * tx * ty;
}

// Fractal noise in [0, 1) over a texture of period 1 (u, v in [0, 1)), from `cells` cells per period.
float Fbm(float u, float v, int cells, int octaves, u32 seed)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float total = 0.0f;
    for (int o = 0; o < octaves; ++o)
    {
        const int period = cells << o;
        sum += Noise(u * float(period), v * float(period), period, seed + u32(o) * 977u) * amplitude;
        total += amplitude;
        amplitude *= 0.5f;
    }
    return sum / total;
}

float EncodeSrgb(float c)
{
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

u32 Pack(float r, float g, float b, float a)
{
    auto byte = [](float v) { return u32(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    return byte(r) | byte(g) << 8 | byte(b) << 16 | byte(a) << 24;
}

// What a material's pattern says about one texel (u, v in [0, 1) over the repeating texture).
struct Texel
{
    float height = 0.0f;            // in texture widths: the slope of `height` is the normal's
    float3 color{0.5f, 0.5f, 0.5f}; // linear
    float roughness = 0.5f;
    float metallic = 0.0f;
};

MaterialTextures Build(std::string name, u32 size, const std::function<Texel(float, float)>& pattern)
{
    MaterialTextures t;
    t.name = std::move(name);
    t.width = size;
    t.height = size;
    std::vector<Texel> texels(size_t(size) * size);
    for (u32 y = 0; y < size; ++y)
    {
        for (u32 x = 0; x < size; ++x)
        {
            texels[size_t(y) * size + x] = pattern((float(x) + 0.5f) / float(size), (float(y) + 0.5f) / float(size));
        }
    }
    t.color.resize(texels.size());
    t.surface.resize(texels.size());
    auto heightAt = [&](int x, int y)
    { return texels[size_t((y + int(size)) % int(size)) * size + size_t((x + int(size)) % int(size))].height; };
    for (u32 y = 0; y < size; ++y)
    {
        for (u32 x = 0; x < size; ++x)
        {
            const Texel& texel = texels[size_t(y) * size + x];
            // The slope along u and v (height and texture coordinates both in texture widths).
            const float du = (heightAt(int(x) + 1, int(y)) - heightAt(int(x) - 1, int(y))) * float(size) * 0.5f;
            const float dv = (heightAt(int(x), int(y) + 1) - heightAt(int(x), int(y) - 1)) * float(size) * 0.5f;
            const float3 n = hlsl::normalize(float3(-du, -dv, 1.0f));
            t.color[size_t(y) * size + x] =
                Pack(EncodeSrgb(texel.color.x), EncodeSrgb(texel.color.y), EncodeSrgb(texel.color.z), 1.0f);
            t.surface[size_t(y) * size + x] =
                Pack(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, texel.roughness, texel.metallic);
        }
    }
    return t;
}

// The distance (in texture widths) from (u, v) to the nearest edge of the cell it is in, on a grid of
// `columns` x `rows` cells whose odd rows are shifted by `shift` of a cell; also the cell's index.
float CellEdge(float u, float v, int columns, int rows, float shift, int& cell)
{
    const float row = std::floor(v * float(rows));
    const float offset = (int(row) % 2 != 0) ? shift : 0.0f;
    const float x = u * float(columns) + offset;
    const float column = std::floor(x);
    const int wrappedColumn = ((int(column) % columns) + columns) % columns;
    cell = int(row) * columns + wrappedColumn;
    const float fx = (x - column) / float(columns);
    const float fy = (v * float(rows) - row) / float(rows);
    return std::min(std::min(fx, 1.0f / float(columns) - fx), std::min(fy, 1.0f / float(rows) - fy));
}

} // namespace

MaterialTextures MakeBrickTextures(u32 size)
{
    return Build("Bricks", size,
                 [](float u, float v)
                 {
                     int cell = 0;
                     const float edge = CellEdge(u, v, 4, 8, 0.5f, cell);
                     const float mortar = 0.008f;
                     const float bevel = 0.006f;
                     Texel t;
                     const float grain = Fbm(u, v, 16, 5, 7u);
                     if (edge < mortar)
                     {
                         t.height = 0.0f;
                         t.color = float3(0.42f, 0.40f, 0.37f) * (0.8f + 0.4f * grain);
                         t.roughness = 0.95f;
                     }
                     else
                     {
                         const float shade = 0.75f + 0.5f * Hash(cell, 0, 11u);
                         t.height = 0.004f * std::min((edge - mortar) / bevel, 1.0f) + 0.0015f * grain;
                         t.color = float3(0.45f, 0.17f, 0.10f) * shade * (0.7f + 0.6f * grain);
                         t.roughness = 0.8f + 0.15f * grain;
                     }
                     return t;
                 });
}

MaterialTextures MakeTileTextures(u32 size)
{
    return Build("Tiles", size,
                 [](float u, float v)
                 {
                     int cell = 0;
                     const float edge = CellEdge(u, v, 4, 4, 0.0f, cell);
                     const float grout = 0.006f;
                     Texel t;
                     const float speckle = Fbm(u, v, 32, 3, 23u);
                     if (edge < grout)
                     {
                         t.height = 0.0f;
                         t.color = float3(0.30f, 0.29f, 0.27f) * (0.8f + 0.4f * speckle);
                         t.roughness = 0.9f;
                     }
                     else
                     {
                         const float tint = Hash(cell, 1, 31u);
                         t.height = 0.002f * std::min((edge - grout) / 0.004f, 1.0f);
                         t.color = (tint < 0.5f ? float3(0.62f, 0.68f, 0.72f) : float3(0.12f, 0.20f, 0.32f)) *
                                   (0.92f + 0.16f * speckle);
                         t.roughness = 0.08f + 0.1f * speckle; // glazed
                     }
                     return t;
                 });
}

MaterialTextures MakeMetalPlateTextures(u32 size)
{
    return Build("MetalPlates", size,
                 [](float u, float v)
                 {
                     int cell = 0;
                     const float edge = CellEdge(u, v, 2, 2, 0.0f, cell);
                     Texel t;
                     // Brushed along u: noise stretched 32 times along the strokes.
                     const float brushed = Fbm(u, v * 32.0f, 8, 4, 41u);
                     t.metallic = 1.0f;
                     t.color = float3(0.75f, 0.74f, 0.72f) * (0.9f + 0.1f * Hash(cell, 2, 43u));
                     t.roughness = 0.22f + 0.2f * brushed;
                     t.height = 0.0006f * brushed;
                     if (edge < 0.004f)
                     {
                         t.height = -0.002f; // the seam
                         t.color = float3(0.3f, 0.3f, 0.3f);
                         t.roughness = 0.6f;
                     }
                     // Rivets near the plate's corners: domes.
                     const float pu = std::fmod(u * 2.0f, 1.0f);
                     const float pv = std::fmod(v * 2.0f, 1.0f);
                     for (const float cu : {0.06f, 0.94f})
                     {
                         for (const float cv : {0.06f, 0.94f})
                         {
                             const float d = std::hypot(pu - cu, pv - cv) * 0.5f; // texture widths
                             const float radius = 0.012f;
                             if (d < radius)
                             {
                                 t.height = std::sqrt(radius * radius - d * d) * 0.6f;
                                 t.roughness = 0.35f;
                             }
                         }
                     }
                     return t;
                 });
}

} // namespace air::renderer
