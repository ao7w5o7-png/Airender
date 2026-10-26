// renderer::CheckNeuralTextures: a neural texture set decoded by LibNTC's shader code (NtcCheck.hlsl)
// against its decoding on CUDA.

#include "air/core/Log.h"
#include "air/rdg/RenderGraph.h"
#include "air/renderer/NeuralTextures.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/Transfer.h"
#include "air/renderer/shader/NeuralTextureShared.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace air::renderer
{
namespace
{

using rhi::Access;

// The set decoded at every texel by NtcCheck.cs; empty when the shader is missing.
std::vector<float> DecodeOnGpu(rhi::Device& device, const std::filesystem::path& dir, const NeuralTextureGpuData& set,
                               u32 width, u32 height)
{
    rhi::ComputePipelineDesc desc;
    desc.shader = LoadShader(device, dir, "NtcCheck.cs", "Main");
    desc.pushConstantBytes = sizeof(gpu::NeuralTextureCheckConstants);
    desc.debugName = "NtcCheck";
    if (desc.shader.bytecode.empty())
    {
        AIR_LOG(Warning, "Renderer", "NtcCheck.cs is missing: not checked");
        return {};
    }
    const rhi::PipelineHandle pipeline = device.CreateComputePipeline(desc);
    std::vector<std::span<const u8>> subresources(set.latents.begin(), set.latents.end());
    const rhi::TextureHandle latents = CreateTextureWithSubresources(device,
                                                                     {.format = rhi::Format::RGBA8_UNORM,
                                                                      .width = set.latentWidth,
                                                                      .height = set.latentHeight,
                                                                      .depthOrArraySize = set.latentLayers,
                                                                      .mipLevels = set.latentMips,
                                                                      .usage = rhi::TextureUsage::ShaderResource,
                                                                      .debugName = "NtcCheck.Latents"},
                                                                     subresources);
    const rhi::BufferHandle data =
        CreateBufferWithData(device, {.usage = rhi::BufferUsage::ShaderResource, .debugName = "NtcCheck.Data"},
                             set.data.data(), set.data.size());
    const u64 outputBytes = u64(width) * height * gpu::kNeuralTextureChannels * sizeof(float);
    const rhi::BufferHandle outputs =
        device.CreateBuffer({.size = outputBytes,
                             .usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::CopySrc,
                             .debugName = "NtcCheck.Outputs"});

    rdg::RenderGraph graph(device);
    const rdg::TextureRef latentsRef = graph.ImportTexture(latents, Access::SRVCompute, Access::SRVCompute);
    const rdg::BufferRef dataRef = graph.ImportBuffer(data, Access::SRVCompute, Access::SRVCompute);
    const rdg::BufferRef outputsRef = graph.ImportBuffer(outputs, Access::None, Access::UAVCompute);
    gpu::NeuralTextureCheckConstants constants{};
    constants.width = width;
    constants.height = height;
    graph.AddPass(
        "NtcCheck", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(latentsRef, Access::SRVCompute);
            b.Read(dataRef, Access::SRVCompute);
            b.Write(outputsRef, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            constants.latents = ctx.SRV(latentsRef);
            constants.data = ctx.SRV(dataRef);
            constants.outputs = ctx.UAV(outputsRef);
            ctx.cmd.SetPipeline(pipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        });
    device.WaitForFence(rhi::QueueType::Graphics, graph.Execute());
    const std::vector<u8> bytes = ReadbackBuffer(device, outputs, Access::UAVCompute);
    std::vector<float> values(size_t(outputBytes / sizeof(float)), 0.0f);
    std::memcpy(values.data(), bytes.data(), std::min<size_t>(bytes.size(), values.size() * sizeof(float)));
    device.DestroyBuffer(outputs);
    device.DestroyBuffer(data);
    device.DestroyTexture(latents);
    device.DestroyPipeline(pipeline);
    return values;
}

} // namespace

json::Value CheckNeuralTextures(rhi::Device& device, const std::filesystem::path& rendererShaderDir,
                                const NeuralTextureOptions& options)
{
    json::Value report = json::Value::MakeObject();
    std::string why;
    NeuralTextureCodec* codec = NeuralTextureCodec::Get(&why);
    if (!codec)
    {
        report["error"] = why;
        report["passed"] = false;
        return report;
    }
    const std::optional<SceneDesc> scene = MakeBuiltinScene("textured");
    AIR_CHECK(scene.has_value(), "the textured scene is missing");
    json::Value sets = json::Value::MakeArray();
    bool passed = true;
    for (const MaterialTextures& textures : scene->textureSets)
    {
        json::Value item = json::Value::MakeObject();
        item["name"] = textures.name;
        std::string error;
        const std::optional<CompressedTextures> compressed =
            codec->Compress(textures, options.bitsPerPixel, options.steps, options.cache, &error);
        const std::optional<std::vector<float>> cuda =
            compressed ? codec->DecodeChannels(*compressed, &error) : std::nullopt;
        const std::optional<NeuralTextureGpuData> data =
            cuda ? codec->PrepareInference(*compressed, &error) : std::nullopt;
        if (!data)
        {
            item["error"] = error;
            sets.Push(std::move(item));
            passed = false;
            continue;
        }
        const std::vector<float> shader =
            DecodeOnGpu(device, rendererShaderDir, *data, textures.width, textures.height);
        if (shader.size() != cuda->size())
        {
            item["error"] = "not run (shader missing)";
            sets.Push(std::move(item));
            passed = false;
            continue;
        }
        item["psnr"] = compressed->psnr;
        item["bitsPerPixel"] = compressed->bitsPerPixel;
        json::Value channels = json::Value::MakeArray();
        for (u32 c = 0; c < gpu::kNeuralTextureChannels; ++c)
        {
            double sum = 0.0;
            double squares = 0.0;
            double largest = 0.0;
            const size_t texels = cuda->size() / gpu::kNeuralTextureChannels;
            for (size_t t = 0; t < texels; ++t)
            {
                const size_t i = t * gpu::kNeuralTextureChannels + c;
                const double d = std::isfinite(shader[i]) ? double(shader[i]) - double((*cuda)[i]) : 1e30;
                sum += d;
                squares += d * d;
                largest = std::max(largest, std::abs(d));
            }
            const double mean = sum / double(texels);
            const double rms = std::sqrt(squares / double(texels));
            // Both decode the same int8 network, but the shader's latents are filtered by the texture
            // units and its layers round their inputs their own way: measured, the two differ by under
            // 1 / 255 RMS and 3e-4 on average (sets trained for fewer steps by up to 1.7 / 255 and 1e-3).
            // A wrong layout or offset shows as errors of 0.1 and more.
            const bool ok = std::abs(mean) < 2e-3 && rms < 1e-2;
            passed = passed && ok;
            json::Value channel = json::Value::MakeObject();
            channel["mean"] = mean;
            channel["rms"] = rms;
            channel["largest"] = largest;
            channel["passed"] = ok;
            channels.Push(std::move(channel));
        }
        item["channels"] = std::move(channels);
        sets.Push(std::move(item));
    }
    report["sets"] = std::move(sets);
    report["passed"] = passed;
    return report;
}

} // namespace air::renderer
