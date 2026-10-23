#pragma once

// GPU-side scene data shared by all features. Features read it; the scene system (future: glTF
// loader, animation) writes it. Features never own scene resources.

#include "air/rhi/RHIDevice.h"

namespace air::renderer {

struct GpuScene {
    rhi::AccelStructHandle tlas;
    rhi::BufferHandle instances; // StructuredBuffer<InstanceData>, indexed by TLAS InstanceID
    rhi::BufferHandle materials; // StructuredBuffer<MaterialData>
    u64 version = 0;             // bumped whenever geometry, materials or transforms change
};

// Placeholder scene (one quad) so features can run before a real scene loader exists.
class DemoScene {
public:
    explicit DemoScene(rhi::Device& device);
    ~DemoScene();
    DemoScene(const DemoScene&) = delete;
    DemoScene& operator=(const DemoScene&) = delete;

    const GpuScene& Get() const { return mScene; }

private:
    rhi::Device& mDevice;
    GpuScene mScene;
    rhi::BufferHandle mVertices;
    rhi::BufferHandle mIndices;
    rhi::AccelStructHandle mBlas;
};

} // namespace air::renderer
