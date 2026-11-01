// Upload paths: the frame's linear upload memory (rdg::UploadAllocator), the render graph's upload passes,
// and streaming on the copy queue (renderer::UploadQueue), all on the Null backend.

#include "TestHelpers.h"

#include "air/core/TaskSystem.h"
#include "air/rdg/RenderGraph.h"
#include "air/rdg/UploadAllocator.h"
#include "air/renderer/Scene.h"
#include "air/renderer/UploadQueue.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <numeric>
#include <tuple>
#include <vector>

using namespace air;
using namespace air::rhi;
using airtest::ContainsLine;

namespace
{

bool HasValidationErrors(const std::vector<std::string>& lines)
{
    return ContainsLine(lines, "VALIDATION");
}

struct Fixture
{
    std::unique_ptr<Device> device = CreateDevice({.backend = Backend::Null});
    PipelineHandle pso = device->CreateComputePipeline({.debugName = "Test"});

    Fixture() { device->SetCommandCapture(true); }
    ~Fixture()
    {
        device->DestroyPipeline(pso);
        device->WaitIdle();
    }
    auto Dispatch() const
    {
        PipelineHandle p = pso;
        return [p](rdg::PassContext& ctx)
        {
            ctx.cmd.SetPipeline(p);
            ctx.cmd.Dispatch(1, 1, 1);
        };
    }
};

BufferDesc ReadBuffer(const char* name, u64 size)
{
    return {.size = size, .stride = 4, .usage = BufferUsage::ShaderResource | BufferUsage::CopyDst, .debugName = name};
}

} // namespace

// --- rdg::UploadAllocator --------------------------------------------------------------------------

TEST(UploadAllocator_BumpsThroughAPageAligned)
{
    Fixture f;
    rdg::UploadAllocator uploads(*f.device);
    const auto a = uploads.Allocate(100);
    const auto b = uploads.Allocate(10, 512);
    const auto c = uploads.Allocate(4, 16);
    EXPECT(a.buffer == b.buffer && b.buffer == c.buffer); // one page
    EXPECT(a.offset == 0 && b.offset == 512 && c.offset == 528);
    EXPECT(static_cast<u8*>(b.cpu) - static_cast<u8*>(a.cpu) == 512);
    EXPECT(a.srv != kInvalidBindless && a.srv == c.srv);
    EXPECT(f.device->GetBufferDesc(a.buffer)->memory == MemoryLocation::DeviceLocalHost);
    uploads.EndFrame(0);
    const rdg::UploadAllocator::Stats stats = uploads.GetStats();
    EXPECT(stats.allocations == 3 && stats.bytes == 114 && stats.pagesUsed == 1 && stats.pagesCreated == 1);
}

TEST(UploadAllocator_ReusesPagesOnlyOnceTheGpuPassedTheirFrame)
{
    Fixture f;
    rdg::UploadAllocator uploads(*f.device, {.pageBytes = 1 << 20});
    const BufferHandle first = uploads.Allocate(1000).buffer;
    uploads.EndFrame(1000); // a fence the (Null) GPU has not reached: the page is still in flight
    const BufferHandle second = uploads.Allocate(1000).buffer;
    EXPECT(second != first);
    EXPECT(uploads.GetStats().pagesCreated == 1);
    uploads.EndFrame(0);
    EXPECT(uploads.GetStats().pagesCreated == 1);
    const BufferHandle third = uploads.Allocate(1000).buffer;
    EXPECT(third == second); // its frame is done
    // A frame that fills a page moves on to the next.
    const auto big = uploads.Allocate((1 << 20) - 500);
    EXPECT(big.buffer != third && big.offset == 0);
    uploads.EndFrame(0);
    EXPECT(uploads.GetStats().pagesUsed == 2 && uploads.GetStats().pages == 3); // the first is still in flight
}

TEST(UploadAllocator_GivesLargeAllocationsTheirOwnPage)
{
    Fixture f;
    rdg::UploadAllocator uploads(*f.device, {.pageBytes = 1 << 20});
    const auto small = uploads.Allocate(64);
    const auto large = uploads.Allocate(3 << 20);
    const auto after = uploads.Allocate(64);
    EXPECT(large.buffer != small.buffer && large.offset == 0);
    EXPECT(f.device->GetBufferDesc(large.buffer)->size >= (3u << 20));
    EXPECT(after.buffer == small.buffer); // small ones go on in their page
    uploads.EndFrame(0);
    const rdg::UploadAllocator::Stats stats = uploads.GetStats();
    EXPECT(stats.pagesReleased == 1 && stats.pages == 1); // the large one goes after its frame
}

TEST(UploadAllocator_AllocatesFromManyThreads)
{
    Fixture f;
    TaskSystem tasks(4);
    rdg::UploadAllocator uploads(*f.device, {.pageBytes = 64 * 1024});
    std::mutex mutex;
    std::vector<std::tuple<u32, u64, u64>> ranges; // page, begin, end
    tasks.ParallelFor(2000,
                      [&](u32 i)
                      {
                          const auto a = uploads.Allocate(100 + i % 7, 16);
                          std::memset(a.cpu, int(i), a.size);
                          std::lock_guard lock(mutex);
                          ranges.emplace_back(a.buffer.index, a.offset, a.offset + a.size);
                      });
    std::sort(ranges.begin(), ranges.end());
    bool disjoint = true;
    for (size_t i = 1; i < ranges.size(); ++i)
    {
        const auto& [page, begin, end] = ranges[i];
        disjoint = disjoint && (page != std::get<0>(ranges[i - 1]) || begin >= std::get<2>(ranges[i - 1]));
    }
    EXPECT(ranges.size() == 2000 && disjoint);
    uploads.EndFrame(0);
    EXPECT(uploads.GetStats().allocations == 2000);
}

// --- RenderGraph uploads ---------------------------------------------------------------------------

TEST(Upload_GraphCopiesUploadBufferInACopyPass)
{
    Fixture f;
    const std::vector<u32> data = {1, 2, 3, 4, 5, 6, 7, 8};
    {
        rdg::RenderGraph graph(*f.device);
        const rdg::BufferRef table = graph.UploadBuffer(ReadBuffer("Table", 0), data.data(), data.size() * 4);
        const rdg::BufferRef unused = graph.UploadBuffer(ReadBuffer("Unused", 0), data.data(), data.size() * 4);
        (void)unused;
        rdg::TextureRef out =
            graph.CreateTexture({.format = Format::RGBA16_FLOAT,
                                 .width = 8,
                                 .height = 8,
                                 .usage = TextureUsage::UnorderedAccess | TextureUsage::ShaderResource,
                                 .debugName = "Out"});
        rhi::TextureHandle extracted;
        graph.ExtractTexture(out, Access::SRVCompute, &extracted);
        graph.AddPass(
            "Shade", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(table, Access::SRVCompute);
                b.Write(out, Access::UAVCompute);
            },
            f.Dispatch());
        graph.Execute();
        f.device->DestroyTexture(extracted);
        EXPECT(graph.GetStats().uploadCount == 2);
        EXPECT(graph.GetStats().uploadBytes == 2 * data.size() * 4);
        EXPECT(graph.GetStats().culledPassCount == 1); // nothing reads "Unused"
    }
    const auto commands = f.device->TakeCapturedCommands();
    EXPECT(ContainsLine(commands, "BeginEvent 'Upload Table'"));
    EXPECT(!ContainsLine(commands, "Upload Unused"));
    EXPECT(ContainsLine(commands, "[buf 'Table' None -> CopyDst]"));
    EXPECT(ContainsLine(commands, "CopyBuffer buf 'Table'+0 <- buf 'Upload[0]'"));
    EXPECT(ContainsLine(commands, "[buf 'Table' CopyDst -> SRVCompute]"));
    EXPECT(!HasValidationErrors(commands));
}

TEST(Upload_GraphOrdersUploadToBufferBetweenItsReaders)
{
    Fixture f;
    const BufferHandle persistent = f.device->CreateBuffer(ReadBuffer("Persistent", 256));
    TextureHandle target = f.device->CreateTexture({.format = Format::RGBA16_FLOAT,
                                                    .width = 8,
                                                    .height = 8,
                                                    .usage = TextureUsage::UnorderedAccess,
                                                    .debugName = "Target"});
    rdg::UploadAllocator uploads(*f.device);
    for (int frame = 0; frame < 3; ++frame)
    {
        rdg::RenderGraph graph(*f.device, nullptr, {}, nullptr, &uploads);
        const rdg::BufferRef buffer =
            graph.ImportBuffer(persistent, frame == 0 ? Access::None : Access::SRVCompute, Access::SRVCompute);
        const rdg::TextureRef out = graph.ImportTexture(target, Access::None, Access::None);
        auto reader = [&](const char* name)
        {
            graph.AddPass(
                name, rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    b.Read(buffer, Access::SRVCompute);
                    b.Write(out, Access::UAVCompute);
                },
                f.Dispatch());
        };
        reader("Before");
        const u32 value = 42u + u32(frame);
        graph.UploadToBuffer(buffer, 64, &value, sizeof(value));
        reader("After");
        graph.Execute();
    }
    const auto commands = f.device->TakeCapturedCommands();
    EXPECT(ContainsLine(commands, "BeginEvent 'Upload Persistent'"));
    EXPECT(ContainsLine(commands, "CopyBuffer buf 'Persistent'+64 <- buf 'Upload[0]'"));
    EXPECT(ContainsLine(commands, "[buf 'Persistent' SRVCompute -> CopyDst]"));
    EXPECT(ContainsLine(commands, "[buf 'Persistent' CopyDst -> SRVCompute]"));
    EXPECT(!HasValidationErrors(commands));
    // Three frames with the Null GPU done at once: one page, written by each frame in turn.
    EXPECT(uploads.GetStats().pages == 1);
    f.device->DestroyTexture(target);
    f.device->DestroyBuffer(persistent);
}

TEST(Upload_GraphUploadTextureCopiesEveryLayer)
{
    Fixture f;
    std::vector<u8> texels(8 * 4 * 3 * 4); // 8 x 4 RGBA8, 3 layers: rows 256 bytes apart, layers 1024
    std::iota(texels.begin(), texels.end(), u8(0));
    rdg::RenderGraph graph(*f.device);
    const rdg::TextureRef texture = graph.UploadTexture({.format = Format::RGBA8_UNORM,
                                                         .width = 8,
                                                         .height = 4,
                                                         .depthOrArraySize = 3,
                                                         .usage = TextureUsage::ShaderResource,
                                                         .debugName = "Layers"},
                                                        texels.data());
    TextureHandle out;
    graph.ExtractTexture(texture, Access::SRVCompute, &out);
    graph.Execute();
    const auto commands = f.device->TakeCapturedCommands();
    EXPECT(ContainsLine(commands, "CopyBufferToTexture tex 'Layers' mip 0 <- buf 'Upload[0]'+0"));
    EXPECT(ContainsLine(commands, "CopyBufferToTexture tex 'Layers' mip 0 layer 1 <- buf 'Upload[0]'+1024"));
    EXPECT(ContainsLine(commands, "CopyBufferToTexture tex 'Layers' mip 0 layer 2 <- buf 'Upload[0]'+2048"));
    EXPECT(!HasValidationErrors(commands));
    f.device->DestroyTexture(out);
}

TEST(Upload_GraphPassesAllocateWhileTheyRecord)
{
    Fixture f;
    TaskSystem tasks(4);
    rdg::UploadAllocator uploads(*f.device);
    rdg::RenderGraph graph(*f.device, &tasks, {.passesPerCommandList = 1}, nullptr, &uploads);
    TextureHandle target = f.device->CreateTexture({.format = Format::RGBA16_FLOAT,
                                                    .width = 8,
                                                    .height = 8,
                                                    .usage = TextureUsage::UnorderedAccess,
                                                    .debugName = "Target"});
    const rdg::TextureRef out = graph.ImportTexture(target, Access::None, Access::UAVCompute);
    rdg::RenderGraph* g = &graph;
    const PipelineHandle pso = f.pso;
    for (int i = 0; i < 8; ++i)
    {
        graph.AddPass(
            "Pass", rdg::PassFlags::Compute, [&](rdg::PassBuilder& b) { b.Write(out, Access::UAVCompute); },
            [g, pso](rdg::PassContext& ctx)
            {
                const rdg::UploadAllocator::Allocation constants = g->AllocateUpload(64);
                std::memset(constants.cpu, 0, 64);
                ctx.cmd.SetPipeline(pso);
                ctx.cmd.PushConstants(&constants.srv, 4);
                ctx.cmd.Dispatch(1, 1, 1);
            });
    }
    graph.Execute();
    EXPECT(graph.GetStats().uploadBytes == 8 * 64);
    EXPECT(uploads.GetStats().allocations == 8);
    EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));
    f.device->DestroyTexture(target);
}

// --- renderer::UploadQueue -------------------------------------------------------------------------

TEST(UploadQueue_StreamsOnTheCopyQueueAndHandsOver)
{
    Fixture f;
    const BufferHandle buffer = f.device->CreateBuffer(ReadBuffer("Streamed", 4096));
    const TextureHandle texture =
        f.device->CreateTexture({.format = Format::RGBA8_UNORM,
                                 .width = 16,
                                 .height = 16,
                                 .mipLevels = 2,
                                 .usage = TextureUsage::ShaderResource | TextureUsage::CopyDst,
                                 .debugName = "StreamedTexture"});
    {
        renderer::UploadQueue uploads(*f.device);
        const std::vector<u8> bytes(4096, 7);
        uploads.WriteBuffer(buffer, 0, bytes);
        uploads.WriteTexture(texture, 0, 0, std::span(bytes).first(16 * 16 * 4));
        const renderer::UploadQueue::Ticket ticket =
            uploads.WriteTexture(texture, 1, 0, std::span(bytes).first(8 * 8 * 4));
        EXPECT(!uploads.IsComplete(ticket)); // not submitted yet
        EXPECT(uploads.MakeAvailable(QueueType::Graphics, Access::SRVCompute) == ticket);
        EXPECT(uploads.IsComplete(ticket));
        const renderer::UploadQueue::Stats stats = uploads.GetStats();
        EXPECT(stats.batches == 1 && stats.copies == 3 && stats.cpuWaits == 0);
    }
    const auto commands = f.device->TakeCapturedCommands();
    EXPECT(ContainsLine(commands, "[Copy] BeginEvent 'UploadQueue'"));
    EXPECT(ContainsLine(commands,
                        "[Copy] Barrier [buf 'Streamed' None -> CopyDst] [tex 'StreamedTexture' None -> CopyDst]"));
    EXPECT(ContainsLine(commands, "[Copy] CopyBuffer buf 'Streamed'+0 <- buf 'UploadQueue.Staging'+0"));
    EXPECT(ContainsLine(commands, "[Copy] CopyBufferToTexture tex 'StreamedTexture' mip 1"));
    EXPECT(ContainsLine(commands, "[Graphics] Wait Copy fence 1"));
    EXPECT(ContainsLine(
        commands, "[Graphics] Barrier [buf 'Streamed' None -> SRVCompute] [tex 'StreamedTexture' None -> SRVCompute]"));
    EXPECT(!HasValidationErrors(commands));
    f.device->DestroyTexture(texture);
    f.device->DestroyBuffer(buffer);
}

TEST(UploadQueue_WritesImmutableBuffersOnlyOnTheCopyQueue)
{
    Fixture f;
    BufferDesc desc = ReadBuffer("Immutable", 256);
    desc.usage |= BufferUsage::Immutable;
    const BufferHandle buffer = f.device->CreateBuffer(desc);
    const BufferHandle staging =
        f.device->CreateBuffer({.size = 256, .memory = MemoryLocation::Upload, .debugName = "Staging"});
    {
        renderer::UploadQueue uploads(*f.device);
        uploads.WriteBuffer(buffer, 0, std::vector<u8>(256, 1));
        uploads.MakeAvailable(QueueType::Graphics, Access::SRVCompute);
    }
    EXPECT(!HasValidationErrors(f.device->TakeCapturedCommands()));

    // D3D12 may pack it into a buffer shared with others, whose state only the copy queue may change.
    auto cmd = f.device->CreateCommandList(QueueType::Graphics);
    cmd->Begin();
    cmd->Barriers(std::array{Barrier::Buf(buffer, Access::SRVCompute, Access::CopyDst)});
    cmd->CopyBuffer(buffer, 0, staging, 0, 256);
    cmd->End();
    CommandList* lists[] = {cmd.get()};
    f.device->Submit(QueueType::Graphics, lists);
    EXPECT(ContainsLine(f.device->TakeCapturedCommands(),
                        "VALIDATION: CopyBuffer into Immutable buf 'Immutable' on the Graphics queue"));
    f.device->DestroyBuffer(staging);
    f.device->DestroyBuffer(buffer);
}

TEST(UploadQueue_ReportsUseWithoutWaitingForTheCopies)
{
    Fixture f;
    const BufferHandle buffer = f.device->CreateBuffer(ReadBuffer("Unsynchronized", 256));
    {
        renderer::UploadQueue uploads(*f.device);
        const std::vector<u8> bytes(256, 1);
        uploads.WriteBuffer(buffer, 0, bytes);
        uploads.Submit();
        // Graphics work that uses the buffer without WaitOnGpu() / MakeAvailable().
        auto cmd = f.device->CreateCommandList(QueueType::Graphics);
        cmd->Begin();
        cmd->Barriers(std::array{Barrier::Buf(buffer, Access::None, Access::SRVCompute)});
        cmd->End();
        CommandList* lists[] = {cmd.get()};
        f.device->Submit(QueueType::Graphics, lists);
    }
    EXPECT(ContainsLine(f.device->TakeCapturedCommands(), "was last used on the Copy queue"));
    f.device->DestroyBuffer(buffer);
}

TEST(UploadQueue_WaitsForStagingWhenTheRingIsFull)
{
    Fixture f;
    const BufferHandle buffer = f.device->CreateBuffer(ReadBuffer("Big", 8 << 20));
    renderer::UploadQueue uploads(*f.device, {.stagingBytes = 1 << 20, .submitBytes = 64 << 20});
    const std::vector<u8> chunk(400 * 1024, 3);
    for (u64 i = 0; i < 5; ++i)
    {
        uploads.WriteBuffer(buffer, i * chunk.size(), chunk); // the third one does not fit with the first two
    }
    uploads.MakeAvailable(QueueType::Graphics, Access::SRVCompute);
    const renderer::UploadQueue::Stats stats = uploads.GetStats();
    EXPECT(stats.cpuWaits == 2 && stats.batches == 3 && stats.dedicated == 0);
    EXPECT(stats.bytes == 5 * chunk.size());
    // Larger than the ring: its own staging buffer.
    const BufferHandle other = f.device->CreateBuffer(ReadBuffer("Other", 4 << 20));
    const std::vector<u8> large(3 << 20, 5);
    uploads.WriteBuffer(other, 0, large);
    uploads.MakeAvailable(QueueType::Graphics, Access::SRVCompute);
    EXPECT(uploads.GetStats().dedicated == 1);
    const auto commands = f.device->TakeCapturedCommands();
    EXPECT(ContainsLine(commands, "<- buf 'UploadQueue.Dedicated'+0 (3145728 bytes)"));
    EXPECT(!HasValidationErrors(commands));
    uploads.WaitOnCpu(uploads.Submit());
    f.device->DestroyBuffer(other);
    f.device->DestroyBuffer(buffer);
}

TEST(UploadQueue_SceneStreamsOnTheCopyQueue)
{
    Fixture f;
    {
        renderer::Scene scene(*f.device, renderer::MakeCornellBox());
        scene.Update();
    }
    const auto commands = f.device->TakeCapturedCommands();
    EXPECT(ContainsLine(commands, "[Copy] CopyBuffer"));
    EXPECT(ContainsLine(commands, "[Graphics] Wait Copy fence"));
    EXPECT(ContainsLine(commands, "[Graphics] BeginEvent 'Scene.BuildAccelStructs'"));
    EXPECT(!HasValidationErrors(commands));
}
