#include "air/renderer/Scene.h"

namespace air::renderer {

using rhi::BufferUsage;
using rhi::MemoryLocation;

DemoScene::DemoScene(rhi::Device& device) : mDevice(device) {
    const BufferUsage geometryUsage = BufferUsage::ShaderResource | BufferUsage::AccelStructBuildInput;
    mVertices = device.CreateBuffer({.size = 4 * 32,
                                     .stride = 32,
                                     .usage = geometryUsage,
                                     .memory = MemoryLocation::Upload,
                                     .debugName = "Demo.Vertices"});
    mIndices = device.CreateBuffer({.size = 6 * 4,
                                    .stride = 4,
                                    .usage = geometryUsage,
                                    .memory = MemoryLocation::Upload,
                                    .debugName = "Demo.Indices"});

    rhi::BlasDesc blas;
    blas.geometries.push_back(
        {.vertexBuffer = mVertices, .vertexCount = 4, .vertexStride = 32, .indexBuffer = mIndices, .indexCount = 6});
    blas.debugName = "Demo.Quad";
    mBlas = device.CreateBlas(blas);

    mScene.tlas = device.CreateTlas({.maxInstances = 1, .debugName = "Demo.TLAS"});
    mScene.instances = device.CreateBuffer({.size = 16,
                                            .stride = 16,
                                            .usage = BufferUsage::ShaderResource,
                                            .memory = MemoryLocation::Upload,
                                            .debugName = "Demo.Instances"});
    mScene.materials = device.CreateBuffer({.size = 32,
                                            .stride = 32,
                                            .usage = BufferUsage::ShaderResource,
                                            .memory = MemoryLocation::Upload,
                                            .debugName = "Demo.Materials"});
    mScene.version = 1;
}

DemoScene::~DemoScene() {
    mDevice.DestroyBuffer(mVertices);
    mDevice.DestroyBuffer(mIndices);
    mDevice.DestroyBuffer(mScene.instances);
    mDevice.DestroyBuffer(mScene.materials);
    mDevice.DestroyAccelStruct(mBlas);
    mDevice.DestroyAccelStruct(mScene.tlas);
}

} // namespace air::renderer
