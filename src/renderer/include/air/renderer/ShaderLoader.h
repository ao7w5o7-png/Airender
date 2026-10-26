#pragma once

#include "air/rhi/RHIDevice.h"

#include <filesystem>
#include <string>

namespace air::renderer
{

// Loads `<dir>/<stem>.dxil` (D3D12) or `<dir>/<stem>.spv` (Vulkan). On the Null backend a missing
// file is fine and yields an empty blob, so features can run headless without compiled shaders.
rhi::ShaderBlob LoadShader(const rhi::Device& device, const std::filesystem::path& dir, const std::string& stem,
                           std::string entryPoint);

// A compute pipeline whose shader is loaded up front and which is created the first time it is used:
// large kernels take the driver a while to compile, which then costs nothing while the feature that
// needs them is off. Use from the thread that builds frames.
class LazyComputePipeline
{
public:
    // Loads <dir>/<stem>.dxil|spv (entry point Main). Missing is fine: the pipeline is unavailable.
    void Load(const rhi::Device& device, const std::filesystem::path& dir, const std::string& stem,
              std::string debugName, u32 pushConstantBytes);
    // The shader exists (always on the Null backend) and creating the pipeline has not failed.
    bool IsAvailable() const { return mAvailable; }
    // The pipeline, created on first use; invalid when that fails.
    rhi::PipelineHandle Get(rhi::Device& device);
    void Destroy(rhi::Device& device);

private:
    rhi::ComputePipelineDesc mDesc;
    rhi::PipelineHandle mPipeline;
    bool mAvailable = false;
};

} // namespace air::renderer
