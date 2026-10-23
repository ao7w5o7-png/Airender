// aircli: headless driver for the renderer. Meant to be run by humans and AI agents alike:
// everything it does is printed as text (graph dump, captured commands, memory stats).
//
//   aircli [--backend null|d3d12|vulkan] [--frames N] [--width W] [--height H]
//          [--dump-graph out.dot] [--dump-commands] [--shaders DIR]

#include "air/core/Log.h"
#include "air/core/TaskSystem.h"
#include "air/pathtracer/PathTracer.h"
#include "air/rdg/RenderGraph.h"
#include "air/rhi/RHIDevice.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>

using namespace air;

namespace {

struct Options {
    rhi::Backend backend = rhi::Backend::Null;
    u32 frames = 4;
    u32 width = 1280;
    u32 height = 720;
    std::string dumpGraph;
    bool dumpCommands = false;
    std::string shaderDir = "shaders";
};

[[noreturn]] void Usage() {
    std::puts("usage: aircli [--backend null|d3d12|vulkan] [--frames N] [--width W] [--height H]\n"
              "              [--dump-graph out.dot] [--dump-commands] [--shaders DIR]");
    std::exit(1);
}

Options Parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                Usage();
            }
            return argv[++i];
        };
        if (arg == "--backend") {
            std::string_view b = next();
            o.backend = b == "d3d12" ? rhi::Backend::D3D12 : b == "vulkan" ? rhi::Backend::Vulkan : rhi::Backend::Null;
        } else if (arg == "--frames") {
            o.frames = u32(std::stoul(std::string(next())));
        } else if (arg == "--width") {
            o.width = u32(std::stoul(std::string(next())));
        } else if (arg == "--height") {
            o.height = u32(std::stoul(std::string(next())));
        } else if (arg == "--dump-graph") {
            o.dumpGraph = next();
        } else if (arg == "--dump-commands") {
            o.dumpCommands = true;
        } else if (arg == "--shaders") {
            o.shaderDir = next();
        } else {
            Usage();
        }
    }
    return o;
}

// Placeholder scene: one BLAS/TLAS and empty instance/material buffers, enough to drive the passes.
struct DemoScene {
    pt::PathTracerScene gpu;
    rhi::BufferHandle vertices, indices;
    rhi::AccelStructHandle blas;
};

DemoScene CreateDemoScene(rhi::Device& device) {
    DemoScene s;
    rhi::BufferDesc vb{.size = 4 * 32, .stride = 32,
                       .usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::AccelStructBuildInput,
                       .memory = rhi::MemoryLocation::Upload, .debugName = "Demo.Vertices"};
    rhi::BufferDesc ib{.size = 6 * 4, .stride = 4,
                       .usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::AccelStructBuildInput,
                       .memory = rhi::MemoryLocation::Upload, .debugName = "Demo.Indices"};
    s.vertices = device.CreateBuffer(vb);
    s.indices = device.CreateBuffer(ib);

    rhi::BlasDesc blas;
    blas.geometries.push_back({.vertexBuffer = s.vertices, .vertexCount = 4, .vertexStride = 32,
                               .indexBuffer = s.indices, .indexCount = 6});
    blas.debugName = "Demo.Quad";
    s.blas = device.CreateBlas(blas);
    s.gpu.tlas = device.CreateTlas({.maxInstances = 1, .debugName = "Demo.TLAS"});
    s.gpu.instances = device.CreateBuffer({.size = 16, .stride = 16, .usage = rhi::BufferUsage::ShaderResource,
                                           .memory = rhi::MemoryLocation::Upload, .debugName = "Demo.Instances"});
    s.gpu.materials = device.CreateBuffer({.size = 32, .stride = 32, .usage = rhi::BufferUsage::ShaderResource,
                                           .memory = rhi::MemoryLocation::Upload, .debugName = "Demo.Materials"});
    return s;
}

void DestroyDemoScene(rhi::Device& device, const DemoScene& s) {
    device.DestroyBuffer(s.vertices);
    device.DestroyBuffer(s.indices);
    device.DestroyBuffer(s.gpu.instances);
    device.DestroyBuffer(s.gpu.materials);
    device.DestroyAccelStruct(s.blas);
    device.DestroyAccelStruct(s.gpu.tlas);
}

} // namespace

int main(int argc, char** argv) {
    const Options options = Parse(argc, argv);

    auto device = rhi::CreateDevice({.backend = options.backend});
    if (!device) {
        return 2;
    }
    device->SetCommandCapture(options.dumpCommands);
    TaskSystem tasks;
    AIR_LOG(Info, "aircli", "backend {} on '{}', {} worker threads", rhi::ToString(device->GetBackend()),
            device->GetCapabilities().adapterName, tasks.GetWorkerCount());

    pt::PathTracer tracer(*device, {.width = options.width, .height = options.height});
    if (!tracer.Initialize(options.shaderDir)) {
        return 3;
    }
    DemoScene scene = CreateDemoScene(*device);

    rhi::TextureHandle output = device->CreateTexture(
        {.format = rhi::Format::RGBA8_UNORM, .width = options.width, .height = options.height,
         .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::CopySrc, .debugName = "Output"});
    rhi::Access outputAccess = rhi::Access::None;

    for (u32 frame = 0; frame < options.frames; ++frame) {
        device->BeginFrame();
        rdg::RenderGraph graph(*device, &tasks);
        rdg::TextureRef out = graph.ImportTexture(output, outputAccess, rhi::Access::CopySrc);
        outputAccess = rhi::Access::CopySrc;
        tracer.AddPasses(graph, scene.gpu, out);
        graph.Execute();

        if (frame == 0) {
            std::fputs(graph.DumpText().c_str(), stdout);
            if (!options.dumpGraph.empty()) {
                std::ofstream(options.dumpGraph) << graph.DumpGraphviz();
                AIR_LOG(Info, "aircli", "wrote {}", options.dumpGraph);
            }
        }
        if (options.dumpCommands) {
            for (const std::string& line : device->TakeCapturedCommands()) {
                std::printf("frame %u: %s\n", frame, line.c_str());
            }
        }
        device->EndFrame();
    }
    AIR_LOG(Info, "aircli", "accumulated {} spp", tracer.GetAccumulatedSamples());

    DestroyDemoScene(*device, scene);
    device->DestroyTexture(output);
    device->WaitIdle();

    const rhi::MemoryStats stats = device->GetMemoryStats();
    for (u32 i = 0; i < u32(rhi::MemoryLocation::Count); ++i) {
        const auto& s = stats.perLocation[i];
        AIR_LOG(Info, "aircli", "memory {}: {} heaps, {} KiB reserved, {} KiB used, {} allocations",
                rhi::ToString(rhi::MemoryLocation(i)), s.heapCount, s.reservedBytes / 1024, s.usedBytes / 1024,
                s.allocationCount);
    }
    return 0;
}
