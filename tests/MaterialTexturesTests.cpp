// Material textures (Scene.h MaterialTextures): the CPU's emulation of the GPU's sampling, the
// procedural sets, and what Validate asks of them.

#include "TestFramework.h"
#include "TestHelpers.h"

#include "air/renderer/Scene.h"
#include "air/renderer/TextureSampling.h"

#include <cmath>

using namespace air;

namespace
{

u32 Rgba(u32 r, u32 g, u32 b, u32 a)
{
    return r | g << 8 | b << 16 | a << 24;
}

bool Close(float a, float b, float tolerance = 2e-3f)
{
    return std::abs(a - b) <= tolerance;
}

} // namespace

// Texel centres return their texels, halfway between two the mean, past the edge the opposite texel
// (repeating), and sRGB texels decode before they are filtered.
TEST(MaterialTextures_SamplingWrapsAndDecodesBeforeFiltering)
{
    const u32 texels[2] = {Rgba(0, 0, 0, 255), Rgba(255, 128, 0, 255)};
    const hlsl::float4 first = renderer::SampleTexture2DWrapRGBA8(texels, 2, 1, false, 0.25f, 0.5f);
    EXPECT(Close(first.x, 0.0f) && Close(first.w, 1.0f));
    const hlsl::float4 middle = renderer::SampleTexture2DWrapRGBA8(texels, 2, 1, false, 0.5f, 0.5f);
    EXPECT(Close(middle.x, 0.5f) && Close(middle.y, 128.0f / 255.0f * 0.5f));
    const hlsl::float4 wrapped = renderer::SampleTexture2DWrapRGBA8(texels, 2, 1, false, 1.0f, 0.5f);
    EXPECT(Close(wrapped.x, 0.5f)); // between the last texel and the first
    const hlsl::float4 srgb = renderer::SampleTexture2DWrapRGBA8(texels, 2, 1, true, 0.5f, 0.5f);
    // (0 + decode(128/255)) / 2, not decode(the mean).
    EXPECT(Close(srgb.y, 0.5f * 0.2158605f) && Close(srgb.x, 0.5f));
}

// The procedural sets are deterministic, of the size asked for, with encoded unit normals that tilt
// somewhere, glazed tiles and metallic plates.
TEST(MaterialTextures_ProceduralSets)
{
    const renderer::MaterialTextures bricks = renderer::MakeBrickTextures(64);
    EXPECT(bricks.width == 64 && bricks.height == 64 && bricks.color.size() == 64 * 64 &&
           bricks.surface.size() == 64 * 64);
    EXPECT(bricks == renderer::MakeBrickTextures(64));
    u32 tilted = 0;
    for (u32 s : bricks.surface)
    {
        const float x = float(s & 0xFFu) / 255.0f * 2.0f - 1.0f;
        const float y = float((s >> 8) & 0xFFu) / 255.0f * 2.0f - 1.0f;
        EXPECT(x * x + y * y <= 1.0f + 1e-2f);
        tilted += std::abs(x) > 0.1f || std::abs(y) > 0.1f;
    }
    EXPECT(tilted > 0);
    const renderer::MaterialTextures tiles = renderer::MakeTileTextures(64);
    const u32 tileCentre = tiles.surface[32 * 64 + 40];
    EXPECT(float((tileCentre >> 16) & 0xFFu) / 255.0f < 0.3f); // glazed
    const renderer::MaterialTextures plates = renderer::MakeMetalPlateTextures(64);
    EXPECT((plates.surface[20 * 64 + 20] >> 24) == 255u); // metallic
}

TEST(MaterialTextures_ValidateChecksSetsAndReferences)
{
    renderer::SceneDesc scene = renderer::MakeCornellBox();
    scene.materials[0].textures = 0;
    EXPECT(!renderer::Validate(scene).empty()); // no such set
    scene.textureSets.push_back(renderer::MakeTileTextures(16));
    EXPECT(renderer::Validate(scene).empty());
    scene.textureSets[0].surface.pop_back();
    EXPECT(!renderer::Validate(scene).empty()); // texels missing
}
