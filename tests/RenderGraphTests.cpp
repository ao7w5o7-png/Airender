#include "TestFramework.h"

#include "air/rdg/RenderGraph.h"

#include <algorithm>

using namespace air;
using namespace air::rhi;
using namespace air::rdg;

namespace {

TextureDesc UavTexture(const char* name, u32 size = 256) {
    return {.format = Format::RGBA16_FLOAT, .width = size, .height = size,
            .usage = TextureUsage::UnorderedAccess | TextureUsage::ShaderResource, .debugName = name};
}

bool Contains(const std::vector<std::string>& lines, std::string_view needle) {
    return std::any_of(lines.begin(), lines.end(), [&](const std::string& l) { return l.find(needle) != std::string::npos; });
}

bool HasValidationErrors(const std::vector<std::string>& lines) { return Contains(lines, "VALIDATION"); }

struct Fixture {
    std::unique_ptr<Device> device = CreateDevice({.backend = Backend::Null});
    PipelineHandle pso = device->CreateComputePipeline({.debugName = "Test"});
    TextureHandle output = device->CreateTexture(UavTexture("Output"));

    Fixture() { device->SetCommandCapture(true); }
    ~Fixture() {
        device->DestroyTexture(output);
        device->DestroyPipeline(pso);
        device->WaitIdle();
    }
    // A compute pass body that only dispatches.
    auto Dispatch() const {
        PipelineHandle p = pso;
        return [p](PassContext& ctx) {
            ctx.cmd.SetPipeline(p);
            ctx.cmd.Dispatch(1, 1, 1);
        };
    }
};

} // namespace

TEST(RenderGraph_CullsPassesWithUnusedOutputs) {
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef unused = graph.CreateTexture(UavTexture("Unused"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
    graph.AddPass("WritesUnused", PassFlags::Compute, [&](PassBuilder& b) { b.Write(unused, Access::UAVCompute); },
                  f.Dispatch());
    graph.AddPass("WritesOutput", PassFlags::Compute, [&](PassBuilder& b) { b.Write(out, Access::UAVCompute); },
                  f.Dispatch());
    graph.AddPass("Debug", PassFlags::Compute | PassFlags::NeverCull, [](PassBuilder&) {}, f.Dispatch());
    graph.Execute();

    EXPECT(graph.GetStats().culledPassCount == 1);
    auto commands = f.device->TakeCapturedCommands();
    EXPECT(!Contains(commands, "WritesUnused"));
    EXPECT(Contains(commands, "BeginEvent 'WritesOutput'"));
    EXPECT(Contains(commands, "BeginEvent 'Debug'"));
    EXPECT(!HasValidationErrors(commands));
}

TEST(RenderGraph_GeneratesTransitionsAndUavBarriers) {
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef temp = graph.CreateTexture(UavTexture("Temp"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
    graph.AddPass("A", PassFlags::Compute, [&](PassBuilder& b) { b.Write(temp, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("B", PassFlags::Compute, [&](PassBuilder& b) { b.Write(temp, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("C", PassFlags::Compute,
                  [&](PassBuilder& b) {
                      b.Read(temp, Access::SRVCompute);
                      b.Write(out, Access::UAVCompute);
                  },
                  f.Dispatch());
    graph.Execute();

    auto commands = f.device->TakeCapturedCommands();
    EXPECT(Contains(commands, "tex 'Temp' None -> UAVCompute discard"));       // first use
    EXPECT(Contains(commands, "tex 'Temp' UAVCompute -> UAVCompute"));         // UAV barrier A->B
    EXPECT(Contains(commands, "tex 'Temp' UAVCompute -> SRVCompute"));         // B->C
    EXPECT(Contains(commands, "tex 'Output' UAVCompute -> SRVCompute"));       // final state
    EXPECT(!HasValidationErrors(commands));
}

TEST(RenderGraph_AliasesNonOverlappingLifetimes) {
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef a = graph.CreateTexture(UavTexture("A"));
    TextureRef b = graph.CreateTexture(UavTexture("B"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::UAVCompute);
    // A lives in passes 0-1, B in passes 2-3: they can share memory.
    graph.AddPass("WriteA", PassFlags::Compute, [&](PassBuilder& p) { p.Write(a, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("ReadA", PassFlags::Compute,
                  [&](PassBuilder& p) {
                      p.Read(a, Access::SRVCompute);
                      p.Write(out, Access::UAVCompute);
                  },
                  f.Dispatch());
    graph.AddPass("WriteB", PassFlags::Compute, [&](PassBuilder& p) { p.Write(b, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("ReadB", PassFlags::Compute,
                  [&](PassBuilder& p) {
                      p.Read(b, Access::SRVCompute);
                      p.Write(out, Access::UAVCompute);
                  },
                  f.Dispatch());
    graph.Execute();

    const auto& stats = graph.GetStats();
    EXPECT(stats.transientResourceCount == 2);
    EXPECT(stats.transientHeapSize * 2 == stats.transientBytesUnaliased);
    EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));
}

TEST(RenderGraph_OverlappingLifetimesDoNotAlias) {
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef a = graph.CreateTexture(UavTexture("A"));
    TextureRef b = graph.CreateTexture(UavTexture("B"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::UAVCompute);
    graph.AddPass("WriteA", PassFlags::Compute, [&](PassBuilder& p) { p.Write(a, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("WriteB", PassFlags::Compute, [&](PassBuilder& p) { p.Write(b, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("Combine", PassFlags::Compute,
                  [&](PassBuilder& p) {
                      p.Read(a, Access::SRVCompute);
                      p.Read(b, Access::SRVCompute);
                      p.Write(out, Access::UAVCompute);
                  },
                  f.Dispatch());
    graph.Execute();
    EXPECT(graph.GetStats().transientHeapSize == graph.GetStats().transientBytesUnaliased);
}

TEST(RenderGraph_ParallelRecordingMatchesSerial) {
    auto run = [](bool parallel) {
        Fixture f;
        TaskSystem tasks(4);
        RenderGraph graph(*f.device, &tasks, {.parallelRecording = parallel, .passesPerCommandList = 2});
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        std::vector<TextureRef> chain;
        for (int i = 0; i < 16; ++i) {
            chain.push_back(graph.CreateTexture(UavTexture(("T" + std::to_string(i)).c_str(), 64)));
        }
        for (int i = 0; i < 16; ++i) {
            graph.AddPass("Step" + std::to_string(i), PassFlags::Compute,
                          [&, i](PassBuilder& b) {
                              if (i > 0) {
                                  b.Read(chain[i - 1], Access::SRVCompute);
                              }
                              b.Write(i == 15 ? out : chain[i], Access::UAVCompute);
                          },
                          f.Dispatch());
        }
        graph.Execute();
        EXPECT(graph.GetStats().commandListCount == 8);
        return f.device->TakeCapturedCommands();
    };
    const auto serial = run(false);
    const auto parallel = run(true);
    EXPECT(serial == parallel);
    EXPECT(!HasValidationErrors(parallel));
}

TEST(RenderGraph_ExtractedTextureOutlivesGraph) {
    Fixture f;
    TextureHandle history;
    {
        RenderGraph graph(*f.device);
        TextureRef t = graph.CreateTexture(UavTexture("History"));
        graph.AddPass("Write", PassFlags::Compute, [&](PassBuilder& b) { b.Write(t, Access::UAVCompute); },
                      f.Dispatch());
        graph.ExtractTexture(t, Access::SRVCompute, &history);
        graph.Execute();
    }
    f.device->EndFrame();
    f.device->EndFrame();
    f.device->EndFrame();
    EXPECT(history.IsValid());
    EXPECT(f.device->GetTextureDesc(history) != nullptr);
    f.device->DestroyTexture(history);
}

TEST(RenderGraph_TransientMemoryIsReleased) {
    Fixture f;
    {
        RenderGraph graph(*f.device);
        TextureRef t = graph.CreateTexture(UavTexture("Temp"));
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::UAVCompute);
        graph.AddPass("A", PassFlags::Compute, [&](PassBuilder& b) { b.Write(t, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass("B", PassFlags::Compute,
                      [&](PassBuilder& b) {
                          b.Read(t, Access::SRVCompute);
                          b.Write(out, Access::UAVCompute);
                      },
                      f.Dispatch());
        graph.Execute();
    }
    f.device->WaitIdle();
    const MemoryStats stats = f.device->GetMemoryStats();
    EXPECT(stats.liveTextures == 1); // only the fixture's Output
}
