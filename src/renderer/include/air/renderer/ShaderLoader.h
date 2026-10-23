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

} // namespace air::renderer
