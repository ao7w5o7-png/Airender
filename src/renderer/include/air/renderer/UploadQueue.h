#pragma once

// Streaming uploads on the copy queue: CPU data goes through a ring of staging memory into device-local
// buffers and textures while the graphics and compute queues keep working, and the CPU does not wait
// for the GPU unless the ring is full.
//
//   UploadQueue uploads(device);
//   uploads.WriteBuffer(vertices, 0, bytes);           // any thread: staged now, copied later
//   uploads.WriteTexture(albedo, mip, layer, texels);
//   const UploadQueue::Ticket ticket = uploads.Submit();
//   uploads.WaitOnGpu(rhi::QueueType::Graphics, ticket); // graphics work submitted after this waits on the GPU
//   cmd->Barriers({Barrier::Tex(albedo, Access::None, Access::SRVCompute), ...});
//
// or, to stream without stalling anything, poll IsComplete(ticket) and start using the resources once
// it is (a resource that is not ready yet is not used, or replaced by a fallback).
//
//   * Staging: a ring of Config::stagingBytes of upload memory. A write is copied into it at once; the
//     recorded copies are submitted in batches, at Submit() and whenever Config::submitBytes have
//     accumulated, so the copy queue starts while the CPU still fills the ring. When the ring is full,
//     the writer waits on the CPU for the oldest batch (memory stays bounded however much is streamed).
//     A write larger than the ring gets its own staging buffer.
//   * States: the copies leave the resources in Access::None (D3D12 COMMON, where resources the copy
//     queue used decay to anyway). Whoever uses them transitions them on their queue, after waiting.
//   * The resources written must be idle: in Access::None, and no GPU work in flight using them.
//     Destroying one with copies in flight is an error: wait for its ticket first (the destructor
//     waits for every copy).
//
// Thread-safe. The copy queue has no timestamps: its work is not in the GPU timings.

#include "air/rhi/RHIDevice.h"

#include <deque>
#include <mutex>
#include <span>
#include <vector>

namespace air::renderer
{

class UploadQueue
{
public:
    struct Config
    {
        u64 stagingBytes = 64ull << 20;
        u64 submitBytes = 8ull << 20; // a batch is submitted once its copies reach this many bytes
    };

    struct Stats
    {
        u64 bytes = 0;       // written, in total
        u32 batches = 0;     // submissions to the copy queue
        u32 copies = 0;      // buffer and texture copies
        u32 cpuWaits = 0;    // times a writer waited for the ring to drain
        u32 dedicated = 0;   // writes larger than the ring, staged in their own buffer
        double waitMs = 0.0; // CPU time spent in those waits
    };

    // The batch a write goes into; a batch is complete when the copy queue has finished it. 0: none.
    using Ticket = u64;

    UploadQueue(rhi::Device& device, Config config);
    explicit UploadQueue(rhi::Device& device) : UploadQueue(device, Config{}) {}
    ~UploadQueue(); // waits for every copy
    UploadQueue(const UploadQueue&) = delete;
    UploadQueue& operator=(const UploadQueue&) = delete;

    // `dst` needs BufferUsage::CopyDst.
    Ticket WriteBuffer(rhi::BufferHandle dst, u64 dstOffset, std::span<const u8> data);
    // One subresource from tightly packed texels: rows of the mip's width, its height rows per slice of a
    // 3D texture. `dst` needs TextureUsage::CopyDst; `layer` is the slice of a 2D texture array.
    Ticket WriteTexture(rhi::TextureHandle dst, u32 mip, u32 layer, std::span<const u8> texels);

    // Submits what has been written; returns the ticket of everything written so far.
    Ticket Submit();
    // Hands every resource written since the last call to `queue` in `access`: submits the open batch,
    // makes `queue` wait for it on the GPU and submits there the transitions from Access::None. The CPU
    // does not wait. Returns the ticket of everything written so far.
    Ticket MakeAvailable(rhi::QueueType queue, rhi::Access access);
    bool IsComplete(Ticket ticket);
    // Blocks until the copy queue has finished `ticket` (submitting it first if needed).
    void WaitOnCpu(Ticket ticket);
    // Work submitted to `queue` after this call starts once the copies of `ticket` are done; the CPU does
    // not wait (submits `ticket` first if needed).
    void WaitOnGpu(rhi::QueueType queue, Ticket ticket);

    Stats GetStats() const;

private:
    struct Copy
    {
        rhi::BufferHandle buffer;   // destination: a buffer...
        rhi::TextureHandle texture; // ... or a texture
        u64 dstOffset = 0;
        u32 mip = 0;
        u32 layer = 0;
        rhi::BufferHandle source;
        u64 sourceOffset = 0;
        u64 size = 0;
    };
    struct Region // a batch's part of the staging ring
    {
        u64 begin = 0;
        u64 end = 0;
        Ticket ticket = 0;
    };
    struct Dedicated
    {
        rhi::BufferHandle buffer;
        Ticket ticket = 0;
    };

    // Staging memory for `size` bytes in the open batch; may submit and wait. Returns the CPU pointer.
    u8* StageLocked(u64 size, u64 alignment, rhi::BufferHandle* buffer, u64* offset);
    bool PlaceLocked(u64 size, u64 alignment, u64* offset);
    void RetireLocked();
    Ticket SubmitLocked();
    u64 FenceLocked(Ticket ticket); // submits `ticket` first if it is the open batch
    Ticket AddLocked(const Copy& copy);

    rhi::Device& mDevice;
    Config mConfig;
    mutable std::mutex mMutex;
    rhi::BufferHandle mStaging;
    u8* mStagingCpu = nullptr;
    u64 mHead = 0;               // where the next staging region starts
    std::deque<Region> mRegions; // live regions, oldest first
    std::vector<Dedicated> mDedicated;
    std::vector<Copy> mPending; // the open batch's copies
    u64 mPendingBytes = 0;
    std::vector<u64> mFences; // copy queue fence of batch (ticket) i + 1
    // Written since the last MakeAvailable().
    std::vector<rhi::BufferHandle> mWrittenBuffers;
    std::vector<rhi::TextureHandle> mWrittenTextures;
    Stats mStats;
};

} // namespace air::renderer
