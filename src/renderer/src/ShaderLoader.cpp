#include "air/renderer/ShaderLoader.h"

#include "air/core/Log.h"

#include <fstream>
#include <iterator>

namespace air::renderer
{

rhi::ShaderBlob LoadShader(const rhi::Device& device, const std::filesystem::path& dir, const std::string& stem,
                           std::string entryPoint)
{
    rhi::ShaderBlob blob;
    blob.entryPoint = std::move(entryPoint);
    const char* extension = device.GetBackend() == rhi::Backend::Vulkan ? ".spv" : ".dxil";
    const std::filesystem::path path = dir / (stem + extension);
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        if (device.GetBackend() != rhi::Backend::Null)
        {
            AIR_LOG(Error, "Shader", "missing compiled shader {}", path.string());
        }
        return blob;
    }
    blob.bytecode.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return blob;
}

void LazyComputePipeline::Load(const rhi::Device& device, const std::filesystem::path& dir, const std::string& stem,
                               std::string debugName, u32 pushConstantBytes)
{
    const std::filesystem::path path = dir / (stem + (device.GetBackend() == rhi::Backend::Vulkan ? ".spv" : ".dxil"));
    mAvailable = device.GetBackend() == rhi::Backend::Null || std::filesystem::exists(path);
    if (mAvailable)
    {
        mDesc.shader = LoadShader(device, dir, stem, "Main");
    }
    mDesc.pushConstantBytes = pushConstantBytes;
    mDesc.debugName = std::move(debugName);
}

rhi::PipelineHandle LazyComputePipeline::Get(rhi::Device& device)
{
    if (!mPipeline.IsValid() && mAvailable)
    {
        mPipeline = device.CreateComputePipeline(mDesc);
        mAvailable = mPipeline.IsValid();
        mDesc.shader.bytecode = {}; // the driver has it now
    }
    return mPipeline;
}

void LazyComputePipeline::Destroy(rhi::Device& device)
{
    if (mPipeline.IsValid())
    {
        device.DestroyPipeline(mPipeline);
    }
    mPipeline = {};
    mAvailable = false;
}

} // namespace air::renderer
