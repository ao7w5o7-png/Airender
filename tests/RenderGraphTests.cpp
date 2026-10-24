#include "TestFramework.h"

#include "air/rdg/RenderGraph.h"
#include "air/rdg/TransientResourcePool.h"

#include <algorithm>

using namespace air;
using namespace air::rhi;
using namespace air::rdg;

namespace
{

TextureDesc UavTexture(const char* name, u32 size = 256)
{
    return {.format = Format::RGBA16_FLOAT,
            .width = size,
            .height = size,
            .usage = TextureUsage::UnorderedAccess | TextureUsage::ShaderResource,
            .debugName = name};
}

bool Contains(const std::vector<std::string>& lines, std::string_view needle)
{
    return std::any_of(lines.begin(), lines.end(),
                       [&](const std::string& l) { return l.find(needle) != std::string::npos; });
}

bool HasValidationErrors(const std::vector<std::string>& lines)
{
    return Contains(lines, "VALIDATION");
}

struct Fixture
{
    std::unique_ptr<Device> device = CreateDevice({.backend = Backend::Null});
    PipelineHandle pso = device->CreateComputePipeline({.debugName = "Test"});
    TextureHandle output = device->CreateTexture(UavTexture("Output"));

    Fixture() { device->SetCommandCapture(true); }
    ~Fixture()
    {
        device->DestroyTexture(output);
        device->DestroyPipeline(pso);
        device->WaitIdle();
    }
    // A compute pass body that only dispatches.
    auto Dispatch() const
    {
        PipelineHandle p = pso;
        return [p](PassContext& ctx)
        {
            ctx.cmd.SetPipeline(p);
            ctx.cmd.Dispatch(1, 1, 1);
        };
    }
};

} // namespace

TEST(RenderGraph_CullsPassesWithUnusedOutputs)
{
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef unused = graph.CreateTexture(UavTexture("Unused"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
    graph.AddPass(
        "WritesUnused", PassFlags::Compute, [&](PassBuilder& b) { b.Write(unused, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass(
        "WritesOutput", PassFlags::Compute, [&](PassBuilder& b) { b.Write(out, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("Debug", PassFlags::Compute | PassFlags::NeverCull, [](PassBuilder&) {}, f.Dispatch());
    graph.Execute();

    EXPECT(graph.GetStats().culledPassCount == 1);
    auto commands = f.device->TakeCapturedCommands();
    EXPECT(!Contains(commands, "WritesUnused"));
    EXPECT(Contains(commands, "BeginEvent 'WritesOutput'"));
    EXPECT(Contains(commands, "BeginEvent 'Debug'"));
    EXPECT(!HasValidationErrors(commands));
}

TEST(RenderGraph_GeneratesTransitionsAndUavBarriers)
{
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef temp = graph.CreateTexture(UavTexture("Temp"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
    graph.AddPass("A", PassFlags::Compute, [&](PassBuilder& b) { b.Write(temp, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("B", PassFlags::Compute, [&](PassBuilder& b) { b.Write(temp, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass(
        "C", PassFlags::Compute,
        [&](PassBuilder& b)
        {
            b.Read(temp, Access::SRVCompute);
            b.Write(out, Access::UAVCompute);
        },
        f.Dispatch());
    graph.Execute();

    auto commands = f.device->TakeCapturedCommands();
    EXPECT(Contains(commands, "tex 'Temp' None -> UAVCompute discard")); // first use
    EXPECT(Contains(commands, "tex 'Temp' UAVCompute -> UAVCompute"));   // UAV barrier A->B
    EXPECT(Contains(commands, "tex 'Temp' UAVCompute -> SRVCompute"));   // B->C
    EXPECT(Contains(commands, "tex 'Output' UAVCompute -> SRVCompute")); // final state
    EXPECT(!HasValidationErrors(commands));
}

TEST(RenderGraph_AliasesNonOverlappingLifetimes)
{
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef a = graph.CreateTexture(UavTexture("A"));
    TextureRef b = graph.CreateTexture(UavTexture("B"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::UAVCompute);
    // A lives in passes 0-1, B in passes 2-3: they can share memory.
    graph.AddPass("WriteA", PassFlags::Compute, [&](PassBuilder& p) { p.Write(a, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass(
        "ReadA", PassFlags::Compute,
        [&](PassBuilder& p)
        {
            p.Read(a, Access::SRVCompute);
            p.Write(out, Access::UAVCompute);
        },
        f.Dispatch());
    graph.AddPass("WriteB", PassFlags::Compute, [&](PassBuilder& p) { p.Write(b, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass(
        "ReadB", PassFlags::Compute,
        [&](PassBuilder& p)
        {
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

TEST(RenderGraph_OverlappingLifetimesDoNotAlias)
{
    Fixture f;
    RenderGraph graph(*f.device);
    TextureRef a = graph.CreateTexture(UavTexture("A"));
    TextureRef b = graph.CreateTexture(UavTexture("B"));
    TextureRef out = graph.ImportTexture(f.output, Access::None, Access::UAVCompute);
    graph.AddPass("WriteA", PassFlags::Compute, [&](PassBuilder& p) { p.Write(a, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass("WriteB", PassFlags::Compute, [&](PassBuilder& p) { p.Write(b, Access::UAVCompute); }, f.Dispatch());
    graph.AddPass(
        "Combine", PassFlags::Compute,
        [&](PassBuilder& p)
        {
            p.Read(a, Access::SRVCompute);
            p.Read(b, Access::SRVCompute);
            p.Write(out, Access::UAVCompute);
        },
        f.Dispatch());
    graph.Execute();
    EXPECT(graph.GetStats().transientHeapSize == graph.GetStats().transientBytesUnaliased);
}

TEST(RenderGraph_ParallelRecordingMatchesSerial)
{
    auto run = [](bool parallel)
    {
        Fixture f;
        TaskSystem tasks(4);
        RenderGraph graph(*f.device, &tasks, {.parallelRecording = parallel, .passesPerCommandList = 2});
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        std::vector<TextureRef> chain;
        for (int i = 0; i < 16; ++i)
        {
            chain.push_back(graph.CreateTexture(UavTexture(("T" + std::to_string(i)).c_str(), 64)));
        }
        for (int i = 0; i < 16; ++i)
        {
            graph.AddPass(
                "Step" + std::to_string(i), PassFlags::Compute,
                [&, i](PassBuilder& b)
                {
                    if (i > 0)
                    {
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

TEST(RenderGraph_ExtractedTextureOutlivesGraph)
{
    Fixture f;
    TextureHandle history;
    {
        RenderGraph graph(*f.device);
        TextureRef t = graph.CreateTexture(UavTexture("History"));
        graph.AddPass(
            "Write", PassFlags::Compute, [&](PassBuilder& b) { b.Write(t, Access::UAVCompute); }, f.Dispatch());
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

TEST(RenderGraph_TransientMemoryIsReleased)
{
    Fixture f;
    {
        RenderGraph graph(*f.device);
        TextureRef t = graph.CreateTexture(UavTexture("Temp"));
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::UAVCompute);
        graph.AddPass("A", PassFlags::Compute, [&](PassBuilder& b) { b.Write(t, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "B", PassFlags::Compute,
            [&](PassBuilder& b)
            {
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

// --- Async compute ---------------------------------------------------------------------------------

// An async pass without inputs only waits for the graphics prologue (the fork), so it overlaps the
// graphics pass declared before it; its consumer waits for it.
TEST(RenderGraph_AsyncComputeOverlapsAndIsJoined)
{
    Fixture f;
    {
        RenderGraph graph(*f.device);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        TextureRef pattern = graph.CreateTexture(UavTexture("Pattern"));
        graph.AddPass(
            "Before", PassFlags::Compute, [&](PassBuilder& b) { b.Write(out, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "Generate", PassFlags::Compute | PassFlags::AsyncCompute,
            [&](PassBuilder& b) { b.Write(pattern, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "Composite", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(pattern, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.Execute();
        EXPECT(graph.GetStats().asyncComputePassCount == 1);
        EXPECT(graph.GetStats().queueWaitCount == 2 && graph.GetStats().submissionCount == 4);
        EXPECT(graph.DumpText().find("(2) Compute: [1], waits for (0)\n") != std::string::npos); // not for "Before"
    }
    const auto lines = f.device->TakeCapturedCommands();
    EXPECT(Contains(lines, "[Compute] BeginEvent 'Generate'"));
    EXPECT(Contains(lines, "[Compute] Wait Graphics fence"));
    EXPECT(Contains(lines, "[Graphics] Wait Compute fence"));
    EXPECT(!HasValidationErrors(lines));
}

// Read after write across queues: the compute pass waits for the graphics producer, and the graphics
// consumer of its result waits for it.
TEST(RenderGraph_AsyncComputeWaitsForItsInputs)
{
    Fixture f;
    {
        RenderGraph graph(*f.device);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        TextureRef input = graph.CreateTexture(UavTexture("Input"));
        TextureRef result = graph.CreateTexture(UavTexture("Result"));
        graph.AddPass(
            "Produce", PassFlags::Compute, [&](PassBuilder& b) { b.Write(input, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "Filter", PassFlags::Compute | PassFlags::AsyncCompute,
            [&](PassBuilder& b)
            {
                b.Read(input, Access::SRVCompute);
                b.Write(result, Access::UAVCompute);
            },
            f.Dispatch());
        graph.AddPass(
            "Consume", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(result, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.Execute();
        EXPECT(graph.GetStats().queueWaitCount == 2);
    }
    const auto lines = f.device->TakeCapturedCommands();
    EXPECT(Contains(lines, "[Compute] Wait Graphics fence"));
    EXPECT(Contains(lines, "[Graphics] Wait Compute fence"));
    EXPECT(!HasValidationErrors(lines));
}

// A render target is in a state the compute queue cannot leave: the graphics queue makes the
// transition after rendering, and the compute pass waits for it.
TEST(RenderGraph_AsyncComputeHandsOverGraphicsOnlyStates)
{
    Fixture f;
    {
        RenderGraph graph(*f.device);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        TextureDesc desc = UavTexture("Rendered");
        desc.usage = desc.usage | TextureUsage::RenderTarget;
        TextureRef rendered = graph.CreateTexture(desc);
        graph.AddPass(
            "Draw", PassFlags::Raster, [&](PassBuilder& b) { b.Write(rendered, Access::RenderTarget); },
            [](PassContext&) {});
        graph.AddPass(
            "Analyse", PassFlags::Compute | PassFlags::AsyncCompute,
            [&](PassBuilder& b)
            {
                b.Read(rendered, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.Execute();
        // The dump shows the schedule: the hand-over ends the graphics submission the compute one waits for.
        const std::string dump = graph.DumpText();
        EXPECT(dump.find("(1) Graphics: [0]\n      then barrier Rendered: RenderTarget -> SRVCompute") !=
               std::string::npos);
        EXPECT(dump.find("(2) Compute: [1], waits for (1)") != std::string::npos);
    }
    const auto lines = f.device->TakeCapturedCommands();
    EXPECT(Contains(lines, "[Graphics] Barrier [tex 'Rendered' RenderTarget -> SRVCompute]"));
    EXPECT(!Contains(lines, "[Compute] Barrier [tex 'Rendered' RenderTarget"));
    EXPECT(Contains(lines, "[Compute] Wait Graphics fence"));
    EXPECT(!HasValidationErrors(lines));
}

// Write after read across queues: a graphics pass overwriting what an async pass reads waits for it.
TEST(RenderGraph_GraphicsWaitsForAsyncReader)
{
    Fixture f;
    TextureHandle shared = f.device->CreateTexture(UavTexture("Shared"));
    {
        RenderGraph graph(*f.device);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        TextureRef input = graph.ImportTexture(shared, Access::None, Access::SRVCompute);
        graph.AddPass(
            "Init", PassFlags::Compute, [&](PassBuilder& b) { b.Write(input, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "Read", PassFlags::Compute | PassFlags::AsyncCompute,
            [&](PassBuilder& b)
            {
                b.Read(input, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.AddPass(
            "Overwrite", PassFlags::Compute, [&](PassBuilder& b) { b.Write(input, Access::UAVCompute); }, f.Dispatch());
        graph.Execute();
    }
    const auto lines = f.device->TakeCapturedCommands();
    EXPECT(Contains(lines, "[Graphics] Wait Compute fence"));
    EXPECT(!HasValidationErrors(lines));
    f.device->DestroyTexture(shared);
}

// Transients on different queues share memory only when the cross-queue waits order every use of one
// before every use of the other; declaration order says nothing about when an async pass runs.
TEST(RenderGraph_AsyncTransientsAliasOnlyWhenWaitsOrderThem)
{
    {
        // "Early" is written on the compute queue while the graphics queue uses "Late": no order.
        Fixture f;
        RenderGraph graph(*f.device);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        TextureRef early = graph.CreateTexture(UavTexture("Early"));
        TextureRef late = graph.CreateTexture(UavTexture("Late"));
        graph.AddPass(
            "WriteEarly", PassFlags::Compute | PassFlags::AsyncCompute,
            [&](PassBuilder& b) { b.Write(early, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "WriteLate", PassFlags::Compute, [&](PassBuilder& b) { b.Write(late, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "UseLate", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(late, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.AddPass(
            "UseEarly", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(early, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.Execute();
        EXPECT(graph.GetStats().transientHeapSize == graph.GetStats().transientBytesUnaliased);
        EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));
    }
    {
        // "First" ends on the graphics queue before the async pass that writes "Second" starts (it
        // waits for that submission), and its reader waits for it: the two share memory.
        Fixture f;
        RenderGraph graph(*f.device);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        TextureRef first = graph.CreateTexture(UavTexture("First"));
        TextureRef middle = graph.CreateTexture(UavTexture("Middle"));
        TextureRef second = graph.CreateTexture(UavTexture("Second"));
        graph.AddPass(
            "WriteFirst", PassFlags::Compute, [&](PassBuilder& b) { b.Write(first, Access::UAVCompute); },
            f.Dispatch());
        graph.AddPass(
            "FirstToMiddle", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(first, Access::SRVCompute);
                b.Write(middle, Access::UAVCompute);
            },
            f.Dispatch());
        graph.AddPass(
            "MiddleToSecond", PassFlags::Compute | PassFlags::AsyncCompute,
            [&](PassBuilder& b)
            {
                b.Read(middle, Access::SRVCompute);
                b.Write(second, Access::UAVCompute);
            },
            f.Dispatch());
        graph.AddPass(
            "UseSecond", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(second, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.Execute();
        const auto& stats = graph.GetStats();
        EXPECT(stats.asyncComputePassCount == 1);
        EXPECT(stats.transientHeapSize < stats.transientBytesUnaliased); // First and Second share
        EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));
    }
}

// A TransientResourcePool keeps the heap and the placed resources: a second frame like the first
// creates nothing, and its first barriers start from the states the first frame left.
TEST(RenderGraph_TransientPoolReusesMemoryAcrossFrames)
{
    Fixture f;
    TransientResourcePool pool(*f.device);
    const u32 texturesBefore = f.device->GetMemoryStats().liveTextures;
    for (int frame = 0; frame < 3; ++frame)
    {
        {
            RenderGraph graph(*f.device, nullptr, {}, &pool);
            const Access initial = frame == 0 ? Access::None : Access::SRVCompute;
            TextureRef out = graph.ImportTexture(f.output, initial, Access::SRVCompute);
            TextureRef temp = graph.CreateTexture(UavTexture("Temp"));
            graph.AddPass(
                "Write", PassFlags::Compute, [&](PassBuilder& b) { b.Write(temp, Access::UAVCompute); }, f.Dispatch());
            graph.AddPass(
                "Read", PassFlags::Compute,
                [&](PassBuilder& b)
                {
                    b.Read(temp, Access::SRVCompute); // the frame leaves Temp in SRVCompute
                    b.Write(out, Access::UAVCompute);
                },
                f.Dispatch());
            graph.Execute();
            const auto& stats = graph.GetStats();
            EXPECT(stats.transientResourcesCreated == (frame == 0 ? 1u : 0u));
            EXPECT(stats.transientResourcesReused == (frame == 0 ? 0u : 1u));
        }
        f.device->EndFrame();
    }
    const auto lines = f.device->TakeCapturedCommands();
    EXPECT(Contains(lines, "Barrier [tex 'Temp' None -> UAVCompute discard]"));       // new
    EXPECT(Contains(lines, "Barrier [tex 'Temp' SRVCompute -> UAVCompute discard]")); // reused
    EXPECT(!HasValidationErrors(lines));
    EXPECT(pool.GetStats().heaps == 1 && pool.GetStats().heapsCreated == 0 && pool.GetStats().resources == 1);
    EXPECT(f.device->GetMemoryStats().liveTextures == texturesBefore + 1); // only the pooled Temp
}

// A pooled transient first used on the compute queue but last read by a pixel shader ends the frame
// in common, so the compute queue can take it back next frame.
TEST(RenderGraph_TransientPoolReturnsResourcesTheComputeQueueCanTake)
{
    Fixture f;
    TransientResourcePool pool(*f.device);
    for (int frame = 0; frame < 3; ++frame)
    {
        TextureHandle drawn;
        {
            RenderGraph graph(*f.device, nullptr, {}, &pool);
            TextureDesc desc = UavTexture("Shared");
            TextureRef shared = graph.CreateTexture(desc);
            TextureDesc targetDesc = UavTexture("Target");
            targetDesc.usage = targetDesc.usage | TextureUsage::RenderTarget;
            TextureRef target = graph.CreateTexture(targetDesc);
            graph.ExtractTexture(target, Access::SRVGraphics, &drawn);
            graph.AddPass(
                "Generate", PassFlags::Compute | PassFlags::AsyncCompute,
                [&](PassBuilder& b) { b.Write(shared, Access::UAVCompute); }, f.Dispatch());
            graph.AddPass(
                "Draw", PassFlags::Raster,
                [&](PassBuilder& b)
                {
                    b.Read(shared, Access::SRVGraphics); // graphics only: the compute queue cannot leave it
                    b.Write(target, Access::RenderTarget);
                },
                [](PassContext&) {});
            graph.Execute();
            EXPECT(graph.GetStats().transientResourcesReused == (frame == 0 ? 0u : 1u));
        }
        f.device->DestroyTexture(drawn);
        f.device->EndFrame();
    }
    const auto lines = f.device->TakeCapturedCommands();
    EXPECT(Contains(lines, "[tex 'Shared' SRVGraphics -> None]")); // among the graphics queue's final barriers
    EXPECT(Contains(lines, "[Compute] Barrier [tex 'Shared' None -> UAVCompute discard]"));
    EXPECT(!HasValidationErrors(lines));
    f.device->WaitIdle();
}

// What frames stop using is released after Config::retireAfterFrames frames; a bigger frame gets a
// bigger heap.
TEST(RenderGraph_TransientPoolRetiresUnusedMemory)
{
    Fixture f;
    TransientResourcePool pool(*f.device, {.heapGranularity = 64 << 10, .retireAfterFrames = 3});
    auto frame = [&](u32 size)
    {
        RenderGraph graph(*f.device, nullptr, {}, &pool);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::None);
        TextureDesc desc = UavTexture("Temp");
        desc.width = desc.height = size;
        TextureRef temp = graph.CreateTexture(desc);
        graph.AddPass(
            "Write", PassFlags::Compute, [&](PassBuilder& b) { b.Write(temp, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "Read", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(temp, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.Execute();
        f.device->TakeCapturedCommands();
    };
    frame(256); // needs a bigger heap than the 64 KiB granularity
    const u64 bigHeap = pool.GetStats().heapBytes;
    frame(16); // fits the big heap: no new one
    EXPECT(pool.GetStats().heaps == 1 && pool.GetStats().heapBytes == bigHeap);
    EXPECT(pool.GetStats().resources == 2); // the 256 and the 16 texture at offset 0
    for (int i = 0; i < 4; ++i)
    {
        frame(16);
    }
    EXPECT(pool.GetStats().resources == 1); // the 256 one went unused for 3 frames
    // Trim releases sooner (near the VRAM budget): here what the last frame did not use.
    frame(256);
    EXPECT(pool.GetStats().resources == 2);
    pool.Trim(2);
    EXPECT(pool.GetStats().resources == 1 && pool.GetStats().resourcesReleased == 1);
}

// Each frame's async pass rewrites a texture the previous frame's graphics pass read: the compute
// queue starts a frame at the graphics prologue (the fork), after the graphics queue's previous frame.
TEST(RenderGraph_AsyncComputeWaitsForThePreviousFrame)
{
    Fixture f;
    TextureHandle lut = f.device->CreateTexture(UavTexture("Lut"));
    for (int frame = 0; frame < 2; ++frame)
    {
        {
            RenderGraph graph(*f.device);
            const Access initial = frame == 0 ? Access::None : Access::SRVCompute;
            TextureRef out = graph.ImportTexture(f.output, initial, Access::SRVCompute);
            TextureRef table = graph.ImportTexture(lut, initial, Access::SRVCompute);
            graph.AddPass(
                "Update", PassFlags::Compute | PassFlags::AsyncCompute,
                [&](PassBuilder& b) { b.Write(table, Access::UAVCompute); }, f.Dispatch());
            graph.AddPass(
                "Use", PassFlags::Compute,
                [&](PassBuilder& b)
                {
                    b.Read(table, Access::SRVCompute);
                    b.Write(out, Access::UAVCompute);
                },
                f.Dispatch());
            graph.Execute();
            const std::string dump = graph.DumpText();
            EXPECT(dump.find("(0) Graphics: fork\n") != std::string::npos);
            EXPECT(dump.find("(1) Compute: [0], waits for (0)\n") != std::string::npos);
        }
        f.device->EndFrame();
    }
    EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));
    f.device->DestroyTexture(lut);
}

// With async compute disabled every pass runs on the graphics queue, as before.
TEST(RenderGraph_AsyncComputeCanBeDisabled)
{
    Fixture f;
    {
        RenderGraph graph(*f.device, nullptr, {.enableAsyncCompute = false});
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        graph.AddPass(
            "Generate", PassFlags::Compute | PassFlags::AsyncCompute,
            [&](PassBuilder& b) { b.Write(out, Access::UAVCompute); }, f.Dispatch());
        graph.Execute();
        EXPECT(graph.GetStats().asyncComputePassCount == 0 && graph.GetStats().queueWaitCount == 0);
    }
    const auto lines = f.device->TakeCapturedCommands();
    EXPECT(!Contains(lines, "[Compute]"));
}

// The Null device validates cross-queue use on its own: touching a resource on another queue without
// waiting for its previous use, or using a state the queue does not support, is reported.
TEST(NullDevice_ValidatesCrossQueueUse)
{
    Fixture f;
    auto submit = [&](QueueType queue, Barrier barrier)
    {
        auto cmd = f.device->CreateCommandList(queue);
        cmd->Begin();
        cmd->Barriers(std::span(&barrier, 1));
        cmd->End();
        CommandList* lists[] = {cmd.get()};
        return f.device->Submit(queue, lists);
    };
    const u64 written = submit(QueueType::Graphics, Barrier::Tex(f.output, Access::None, Access::UAVCompute));
    submit(QueueType::Compute, Barrier::Tex(f.output, Access::UAVCompute, Access::SRVCompute));
    EXPECT(Contains(f.device->TakeCapturedCommands(), "without waiting"));

    f.device->QueueWait(QueueType::Graphics, QueueType::Compute, 1);
    submit(QueueType::Graphics, Barrier::Tex(f.output, Access::SRVCompute, Access::UAVCompute));
    EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));
    (void)written;

    submit(QueueType::Compute, Barrier::Tex(f.output, Access::UAVCompute, Access::RenderTarget));
    EXPECT(Contains(f.device->TakeCapturedCommands(), "does not support"));
}
