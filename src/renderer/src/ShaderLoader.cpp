#include "air/renderer/ShaderLoader.h"

#include "air/core/Log.h"

#include <fstream>
#include <iterator>

namespace air::renderer {

rhi::ShaderBlob LoadShader(const rhi::Device& device, const std::filesystem::path& dir, const std::string& stem,
                           std::string entryPoint) {
    rhi::ShaderBlob blob;
    blob.entryPoint = std::move(entryPoint);
    const char* extension = device.GetBackend() == rhi::Backend::Vulkan ? ".spv" : ".dxil";
    const std::filesystem::path path = dir / (stem + extension);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (device.GetBackend() != rhi::Backend::Null) {
            AIR_LOG(Error, "Shader", "missing compiled shader {}", path.string());
        }
        return blob;
    }
    blob.bytecode.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return blob;
}

} // namespace air::renderer
