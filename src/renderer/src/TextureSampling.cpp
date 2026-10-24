#include "air/renderer/TextureSampling.h"

#include "air/renderer/Transfer.h"

#include <algorithm>
#include <cmath>

namespace air::renderer
{
namespace
{

// The two texels along one axis and the weight of the second, as the filtering hardware computes
// them: the texel coordinate in fixed point with 8 fractional bits.
struct Taps
{
    int first = 0;
    int second = 0;
    float weight = 0.0f;
};

Taps Axis(float coordinate, u32 size, bool wrap)
{
    const float fixed = std::round((coordinate * float(size) - 0.5f) * 256.0f);
    const float whole = std::floor(fixed / 256.0f);
    Taps taps;
    taps.weight = (fixed - whole * 256.0f) / 256.0f;
    const int n = int(size);
    const int i = int(whole);
    if (wrap)
    {
        taps.first = ((i % n) + n) % n;
        taps.second = (((i + 1) % n) + n) % n;
    }
    else
    {
        taps.first = std::clamp(i, 0, n - 1);
        taps.second = std::clamp(i + 1, 0, n - 1);
    }
    return taps;
}

template <typename T, typename Load> T Trilinear(const Taps& x, const Taps& y, const Taps& z, Load&& load)
{
    auto row = [&](int yi, int zi)
    { return load(x.first, yi, zi) + (load(x.second, yi, zi) - load(x.first, yi, zi)) * x.weight; };
    auto slice = [&](int zi) { return row(y.first, zi) + (row(y.second, zi) - row(y.first, zi)) * y.weight; };
    return slice(z.first) + (slice(z.second) - slice(z.first)) * z.weight;
}

} // namespace

float SampleTexture3D(const u8* texels, u32 width, u32 height, u32 depth, float u, float v, float w)
{
    return Trilinear<float>(Axis(u, width, true), Axis(v, height, true), Axis(w, depth, true), [&](int x, int y, int z)
                            { return float(texels[(size_t(z) * height + size_t(y)) * width + size_t(x)]) / 255.0f; });
}

hlsl::float4 SampleTexture3DClamped(const u16* texels, u32 width, u32 height, u32 depth, float u, float v, float w)
{
    return Trilinear<hlsl::float4>(
        Axis(u, width, false), Axis(v, height, false), Axis(w, depth, false),
        [&](int x, int y, int z)
        {
            const u16* p = texels + ((size_t(z) * height + size_t(y)) * width + size_t(x)) * 4;
            return hlsl::float4(HalfToFloat(p[0]), HalfToFloat(p[1]), HalfToFloat(p[2]), HalfToFloat(p[3]));
        });
}

float SampleTexture3DClampedR16(const u16* texels, u32 width, u32 height, u32 depth, float u, float v, float w)
{
    return Trilinear<float>(Axis(u, width, false), Axis(v, height, false), Axis(w, depth, false),
                            [&](int x, int y, int z)
                            { return HalfToFloat(texels[(size_t(z) * height + size_t(y)) * width + size_t(x)]); });
}

} // namespace air::renderer
