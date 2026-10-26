// Neural texture compression (renderer/NeuralTextures.h): what a scene does with a neural set, the
// coarse copy, and, where LibNTC and a CUDA device are present, a round trip through the codec.

#include "TestFramework.h"
#include "TestHelpers.h"

#include "air/renderer/NeuralTextures.h"
#include "air/renderer/shader/NeuralTextureShared.h"

#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace air;

namespace
{

// A neural set's GPU data of the right shape (its values decode nothing sensible): 8 x 8 latents in
// 2 layers of 2 mips, and a coarse copy of `textures`.
std::shared_ptr<const renderer::NeuralTextureGpuData> FakeNeuralData(const renderer::MaterialTextures& textures)
{
    renderer::NeuralTextureGpuData data;
    data.latentWidth = 8;
    data.latentHeight = 8;
    data.latentLayers = 2;
    data.latentMips = 2;
    for (u32 layer = 0; layer < 2; ++layer)
    {
        data.latents.emplace_back(8 * 8 * 4, u8(17 * layer));
        data.latents.emplace_back(4 * 4 * 4, u8(17 * layer));
    }
    data.data.assign(gpu::kNeuralTextureConstantBytes + 64, 0);
    renderer::MakeNeuralTextureCoarseCopy(textures, data);
    return std::make_shared<const renderer::NeuralTextureGpuData>(std::move(data));
}

// The textured scene with its first set decoded on sample.
renderer::SceneDesc NeuralScene()
{
    renderer::SceneDesc scene = airtest::BuiltinScene("textured");
    scene.textureSets[0].neural = FakeNeuralData(scene.textureSets[0]);
    return scene;
}

std::filesystem::path TestCache()
{
    return std::filesystem::temp_directory_path() / "airender-ntc-tests";
}

} // namespace

// The coarse copy averages kNeuralTextureCoarseFactor^2 texels: the colour in linear light (half
// black, half white is 0.5 linear, 188 in sRGB), the surface as stored.
TEST(NeuralTextures_CoarseCopyAveragesInLinearLight)
{
    renderer::MaterialTextures t;
    t.width = 8;
    t.height = 4;
    for (u32 y = 0; y < 4; ++y)
    {
        for (u32 x = 0; x < 8; ++x)
        {
            const bool white = x % 2 == 0;
            t.color.push_back(white ? 0xFFFFFFFFu : 0xFF000000u);
            t.surface.push_back(white ? 0x10203040u : 0x30405060u);
        }
    }
    renderer::NeuralTextureGpuData data;
    renderer::MakeNeuralTextureCoarseCopy(t, data);
    EXPECT(data.coarseWidth == 2 && data.coarseHeight == 1);
    EXPECT(data.coarseColor.size() == 2 && data.coarseSurface.size() == 2);
    EXPECT((data.coarseColor[0] & 0xFFu) == 188u && (data.coarseColor[0] >> 24) == 255u);
    EXPECT(data.coarseSurface[1] == 0x20304050u);
}

// A neural set uploads its latents (a 2D array, every layer and mip), its data and its coarse copy
// instead of the full textures; Validate rejects one whose latents are missing.
TEST(NeuralTextures_SceneUploadsLatentsAndCoarseCopy)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    renderer::SceneDesc desc = NeuralScene();
    EXPECT(renderer::Validate(desc).empty());
    {
        renderer::Scene scene(*device, desc);
        EXPECT(scene.Get().neuralTextures);
        renderer::Scene plain(*device, airtest::BuiltinScene("textured"));
        EXPECT(!plain.Get().neuralTextures);
        // 512 x 512 RGBA8 twice replaced by the latents, the data and a 128 x 128 coarse copy.
        EXPECT(scene.GetGeometryBytes() + 2 * 512 * 512 * 4 - 2 * 128 * 128 * 4 - (2 * (256 + 64) + 400) ==
               plain.GetGeometryBytes());
    }
    auto broken = std::make_shared<renderer::NeuralTextureGpuData>(*desc.textureSets[0].neural);
    broken->latents.pop_back();
    desc.textureSets[0].neural = broken;
    EXPECT(!renderer::Validate(desc).empty());
}

// Off leaves a scene alone; without LibNTC (or a CUDA device) the others fail and say why, the scene
// unchanged.
TEST(NeuralTextures_ApplyLeavesTheSceneWhenItCannot)
{
    renderer::SceneDesc scene = airtest::BuiltinScene("textured");
    const renderer::SceneDesc original = scene;
    EXPECT(renderer::ApplyNeuralTextures(scene, {.mode = renderer::NeuralTextureMode::Off}));
    EXPECT(scene.textureSets == original.textureSets);
    std::string why;
    if (!renderer::NeuralTextureCodec::Get(&why))
    {
        std::string error;
        EXPECT(!renderer::ApplyNeuralTextures(scene, {.mode = renderer::NeuralTextureMode::OnLoad}, nullptr, &error));
        EXPECT(!error.empty());
        EXPECT(scene.textureSets == original.textureSets);
    }
}

// With LibNTC: a set compresses to a few bits per texel at a PSNR that reproduces it, comes back from
// the cache byte for byte, decodes to texels of its size close to the source's, and hands a shader
// latents of every layer and mip and its weights after the constants.
TEST(NeuralTextures_CodecRoundTrip)
{
    std::string why;
    renderer::NeuralTextureCodec* codec = renderer::NeuralTextureCodec::Get(&why);
    if (!codec)
    {
        std::printf("  skipped: %s\n", why.c_str());
        return;
    }
    std::error_code ignored;
    std::filesystem::remove_all(TestCache(), ignored);
    const renderer::MaterialTextures source = renderer::MakeTileTextures(64);
    std::string error;
    const auto compressed = codec->Compress(source, 4.0f, 2000, TestCache(), &error);
    EXPECT(compressed.has_value());
    if (!compressed)
    {
        std::printf("  %s\n", error.c_str());
        return;
    }
    EXPECT(!compressed->cached && compressed->psnr > 30.0f);
    EXPECT(compressed->bitsPerPixel > 2.0f && compressed->bitsPerPixel < 6.0f);
    const auto again = codec->Compress(source, 4.0f, 2000, TestCache(), &error);
    EXPECT(again && again->cached && again->file == compressed->file && again->psnr == compressed->psnr);

    renderer::MaterialTextures decoded = source;
    EXPECT(codec->Decompress(*compressed, decoded, &error));
    EXPECT(decoded.color.size() == source.color.size() && decoded.surface.size() == source.surface.size());
    double squares = 0.0;
    for (size_t i = 0; i < source.surface.size(); ++i)
    {
        const double d = double(source.surface[i] & 0xFFu) - double(decoded.surface[i] & 0xFFu);
        squares += d * d;
    }
    EXPECT(std::sqrt(squares / double(source.surface.size())) < 8.0); // the normal's x, out of 255

    const auto floats = codec->DecodeChannels(*compressed, &error);
    EXPECT(floats && floats->size() == source.color.size() * gpu::kNeuralTextureChannels);

    const auto gpuData = codec->PrepareInference(*compressed, &error);
    EXPECT(gpuData.has_value());
    if (gpuData)
    {
        EXPECT(gpuData->latentLayers >= 1 && gpuData->latentMips >= 2);
        EXPECT(gpuData->latents.size() == size_t(gpuData->latentLayers) * gpuData->latentMips);
        EXPECT(gpuData->latents[0].size() == size_t(gpuData->latentWidth) * gpuData->latentHeight * 4);
        EXPECT(gpuData->data.size() > gpu::kNeuralTextureConstantBytes + 1024);
    }
    std::filesystem::remove_all(TestCache(), ignored);
}

// With LibNTC: on sample, every set of the textured scene carries its GPU data and a coarse copy,
// and its texels are the decoded ones (what the CPU renders).
TEST(NeuralTextures_ApplyOnSample)
{
    if (!renderer::NeuralTextureCodec::Get())
    {
        std::printf("  skipped: no LibNTC\n");
        return;
    }
    renderer::SceneDesc scene = airtest::BuiltinScene("textured");
    for (renderer::MaterialTextures& t : scene.textureSets)
    {
        t = renderer::MakeTileTextures(32); // small, to compress fast
    }
    std::vector<renderer::NeuralTextureReport> report;
    std::string error;
    EXPECT(renderer::ApplyNeuralTextures(
        scene, {.mode = renderer::NeuralTextureMode::OnSample, .steps = 1000, .cache = TestCache()}, &report, &error));
    EXPECT(report.size() == scene.textureSets.size());
    for (const renderer::MaterialTextures& t : scene.textureSets)
    {
        EXPECT(t.neural && t.neural->coarseWidth == 8 && t.color != renderer::MakeTileTextures(32).color);
    }
    EXPECT(renderer::Validate(scene).empty());
    std::error_code ignored;
    std::filesystem::remove_all(TestCache(), ignored);
}

// The path tracer decodes a scene's neural sets with its AIR_PT_NTC kernels, accumulating and in real
// time (ReSTIR GI's twin), and only while the scene has such sets.
TEST(NeuralTextures_PathTracerUsesItsKernels)
{
    airtest::Headless h({"pathtracer", "tonemap"});
    h.device->WaitIdle();
    h.scene = std::make_unique<renderer::Scene>(*h.device, NeuralScene());
    h.camera = h.scene->Desc().camera;
    renderer::FrameResult frame = h.Frame();
    EXPECT(frame.validationErrors.empty());
    EXPECT(airtest::ContainsLine(frame.commands, "SetPipeline pipeline 'PathTracer.TraceNeural'"));
    EXPECT(h.runner->DescribeFeatures().Dump(0).find("\"neuralKernels\":true") != std::string::npos);
    EXPECT(h.runner->SetParameter("pathtracer.accumulate", "0"));
    frame = h.Frame();
    EXPECT(frame.validationErrors.empty());
    EXPECT(airtest::ContainsLine(frame.commands, "SetPipeline pipeline 'PathTracer.TraceGiNeural'"));
    h.device->WaitIdle();
    h.scene = std::make_unique<renderer::Scene>(*h.device, airtest::BuiltinScene("textured"));
    frame = h.Frame();
    EXPECT(frame.validationErrors.empty());
    EXPECT(!airtest::ContainsLine(frame.commands, "Neural'"));
    EXPECT(h.runner->DescribeFeatures().Dump(0).find("\"neuralKernels\":false") != std::string::npos);
}