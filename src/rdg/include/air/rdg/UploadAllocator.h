#pragma once

// Linear allocator of memory the CPU writes and the GPU reads, for one frame's data: constants and
// tables shaders read in place, and the sources of the render graph's uploads (RenderGraph::UploadBuffer).
//
//   * Pages: buffers of Config::pageBytes in Config::location (default DeviceLocalHost: video memory
//     the CPU writes through ReBAR / D3D12 GPU_UPLOAD, plain upload memory where there is none), each
//     with one raw bindless view (ByteAddressBuffer) for the shaders that read their data in place.
//   * Allocate() bumps an offset in the frame's current page and takes another page when it is full.
//     An allocation larger than a page gets a page of its own, released after the frame.
//   * EndFrame(fence): the pages the frame wrote come back once the GPU has passed `fence`, the graphics
//     fence of the frame's last submission (the render graph joins the compute queue's work into it).
//     The frames in flight thus write different pages; the pool grows to what they need, and pages
//     unused for Config::retireAfterFrames frames are released.
//
// FrameRunner owns one and hands it to every frame's RenderGraph, which ends the frame in Execute().
// Thread-safe: passes may allocate while they record. EndFrame() runs between frames.

#include "air/rhi/RHIDevice.h"

#include <mutex>
#include <vector>

namespace air::rdg
{

class UploadAllocator
{
public:
    struct Config
    {
        u64 pageBytes = 4ull << 20;
        rhi::MemoryLocation location = rhi::MemoryLocation::DeviceLocalHost;
        u32 retireAfterFrames = 120; // pages unused for this many frames are released
    };

    struct Allocation
    {
        rhi::BufferHandle buffer; // the page (BufferUsage::ShaderResource | CopySrc, stride 0)
        u64 offset = 0;
        u64 size = 0;
        // Mapped memory, write-combined: write it once, never read it back.
        void* cpu = nullptr;
        // The page's raw view: ByteAddressBuffer, the data at byte `offset`.
        rhi::BindlessIndex srv = rhi::kInvalidBindless;
    };

    struct Stats
    {
        u32 pages = 0;
        u64 pageBytes = 0;
        // The last frame ended with EndFrame():
        u64 bytes = 0; // allocated, without alignment padding
        u32 allocations = 0;
        u32 pagesUsed = 0;
        u32 pagesCreated = 0;
        u32 pagesReleased = 0;
    };

    UploadAllocator(rhi::Device& device, Config config);
    explicit UploadAllocator(rhi::Device& device) : UploadAllocator(device, Config{}) {}
    ~UploadAllocator();
    UploadAllocator(const UploadAllocator&) = delete;
    UploadAllocator& operator=(const UploadAllocator&) = delete;

    // `alignment` is a power of two (256: constant buffers; 512: texture copy sources).
    Allocation Allocate(u64 size, u64 alignment = 256);
    // Ends the frame, whose GPU work completes with `graphicsFence` on the graphics queue (0: nothing
    // was submitted, the pages are free at once).
    void EndFrame(u64 graphicsFence);

    Stats GetStats() const;

private:
    struct Page
    {
        rhi::BufferHandle buffer;
        u8* cpu = nullptr;
        u64 size = 0;
        rhi::BindlessIndex srv = rhi::kInvalidBindless;
        u64 fence = 0;    // graphics fence of the last frame that wrote it
        u64 lastUsed = 0; // allocator frame
        bool inUse = false;
        bool dedicated = false; // larger than a page: released after its frame
    };

    u32 TakePage(u64 bytes);

    rhi::Device& mDevice;
    Config mConfig;
    mutable std::mutex mMutex;
    std::vector<Page> mPages;
    std::vector<u32> mFramePages; // the pages this frame writes
    u32 mCurrent = ~0u;           // the page small allocations bump in
    u64 mOffset = 0;
    u64 mFrame = 1;
    Stats mStats;
    Stats mCounters; // this frame's, published to mStats by EndFrame()
};

} // namespace air::rdg
