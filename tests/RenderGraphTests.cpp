#include "TestFramework.h"

#include "air/rdg/RenderGraph.h"

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

// A transient touched by an async pass may be in use while any graphics pass runs, so it never
// shares memory, even with a transient whose declared lifetime does not overlap.
TEST(RenderGraph_AsyncTransientsAreNotAliased)
{
    for (bool async : {false, true})
    {
        Fixture f;
        RenderGraph graph(*f.device);
        TextureRef out = graph.ImportTexture(f.output, Access::None, Access::SRVCompute);
        TextureRef first = graph.CreateTexture(UavTexture("First"));
        TextureRef second = graph.CreateTexture(UavTexture("Second"));
        graph.AddPass(
            "WriteFirst", async ? PassFlags::Compute | PassFlags::AsyncCompute : PassFlags::Compute,
            [&](PassBuilder& b) { b.Write(first, Access::UAVCompute); }, f.Dispatch());
        graph.AddPass(
            "UseFirst", PassFlags::Compute,
            [&](PassBuilder& b)
            {
                b.Read(first, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.AddPass(
            "WriteSecond", PassFlags::Compute, [&](PassBuilder& b) { b.Write(second, Access::UAVCompute); },
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
        // Without async the two share memory; with it each has its own.
        EXPECT(async ? stats.transientHeapSize == stats.transientBytesUnaliased
                     : stats.transientHeapSize < stats.transientBytesUnaliased);
        EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));
    }
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
