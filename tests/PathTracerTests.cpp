#include "TestFramework.h"

#include "air/pathtracer/PathTracer.h"

#include <algorithm>

using namespace air;
using namespace air::rhi;

TEST(PathTracer_AccumulatesAcrossFramesHeadless) {
    auto device = CreateDevice({.backend = Backend::Null});
    device->SetCommandCapture(true);
    TaskSystem tasks(2);

    pt::PathTracer tracer(*device, {.width = 64, .height = 32, .samplesPerPixelPerFrame = 2, .targetSamples = 6});
    EXPECT(tracer.Initialize("does-not-exist"));

    pt::PathTracerScene scene;
    scene.tlas = device->CreateTlas({.maxInstances = 1, .debugName = "TLAS"});
    scene.instances = device->CreateBuffer({.size = 16, .usage = BufferUsage::ShaderResource, .debugName = "Instances"});
    scene.materials = device->CreateBuffer({.size = 32, .usage = BufferUsage::ShaderResource, .debugName = "Materials"});
    TextureHandle output = device->CreateTexture({.format = Format::RGBA8_UNORM, .width = 64, .height = 32,
                                                  .usage = TextureUsage::UnorderedAccess, .debugName = "Out"});
    Access outputAccess = Access::None;

    for (int frame = 0; frame < 3; ++frame) {
        rdg::RenderGraph graph(*device, &tasks);
        rdg::TextureRef out = graph.ImportTexture(output, outputAccess, Access::UAVCompute);
        outputAccess = Access::UAVCompute;
        tracer.AddPasses(graph, scene, out);
        graph.Execute();
        device->EndFrame();
    }
    EXPECT(tracer.GetAccumulatedSamples() == 6);
    EXPECT(tracer.IsConverged());

    const auto commands = device->TakeCapturedCommands();
    const auto count = [&](std::string_view needle) {
        return std::count_if(commands.begin(), commands.end(),
                             [&](const std::string& l) { return l.find(needle) != std::string::npos; });
    };
    EXPECT(count("DispatchRays 64x32x1") == 3);
    EXPECT(count("Dispatch 8x4x1") == 3);
    EXPECT(count("VALIDATION") == 0);

    tracer.SetCamera({});
    EXPECT(tracer.GetAccumulatedSamples() == 0);

    device->DestroyTexture(output);
    device->DestroyBuffer(scene.instances);
    device->DestroyBuffer(scene.materials);
    device->DestroyAccelStruct(scene.tlas);
    device->WaitIdle();
}
