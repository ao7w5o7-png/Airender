#pragma once

// Memory and placed resources for render graph transients, kept from frame to frame.
//
//   * Heaps: a frame's transients are placed in a pooled heap the GPU has finished with (the graphics
//     fence of the last frame that used it has passed), instead of in a new heap every frame. The
//     least recently used fitting heap is taken, which is the one laid out like this frame when the
//     frame does not change.
//   * Placed resources: a transient described exactly like one an earlier frame placed at the same
//     offset of the same heap reuses that resource and its bindless descriptors. The pool remembers
//     the state the frame left it in; the graph's first barrier of the next frame starts there.
//   * Heaps and resources unused for Config::retireAfterFrames frames are released (the device defers
//     the release until the GPU is done with them).
//
// Within a frame, transients share memory when their lifetimes do not overlap: the render graph
// computes the offsets, this pool only keeps what outlives the frame. FrameRunner owns one and hands
// it to every frame's RenderGraph. Single threaded: used by the thread that builds the graphs.

#include "air/rhi/RHIDevice.h"

#include <vector>

namespace air::rdg
{

class TransientResourcePool
{
public:
    struct Config
    {
        u64 heapGranularity = 16ull << 20; // heap sizes are rounded up to a multiple of this
        u32 retireAfterFrames = 120;       // heaps and resources unused for this many frames are released
    };

    struct Stats
    {
        u32 heaps = 0;
        u64 heapBytes = 0;
        u32 resources = 0; // placed resources kept for later frames
        // The last frame ended with EndFrame():
        u32 heapsCreated = 0;
        u32 resourcesCreated = 0;
        u32 resourcesReused = 0;
        u32 heapsReleased = 0;
        u32 resourcesReleased = 0;
    };

    // A placed resource handed to the frame.
    struct Placement
    {
        rhi::TextureHandle texture;
        rhi::BufferHandle buffer;
        rhi::Access state = rhi::Access::None; // the state it is in now: the frame's first barrier starts here
        u32 entry = 0;                         // for SetFinalState()
        bool reused = false;                   // placed by an earlier frame
    };

    TransientResourcePool(rhi::Device& device, Config config);
    explicit TransientResourcePool(rhi::Device& device) : TransientResourcePool(device, Config{}) {}
    ~TransientResourcePool();
    TransientResourcePool(const TransientResourcePool&) = delete;
    TransientResourcePool& operator=(const TransientResourcePool&) = delete;

    // A heap of at least `bytes` for this frame's transients; returns its slot for the calls below.
    // `created`: whether it is a new heap.
    u32 AcquireHeap(u64 bytes, bool* created = nullptr);
    rhi::HeapHandle GetHeap(u32 slot) const { return mHeaps[slot].heap; }
    // A resource placed at `offset` of the frame's heap. `firstQueue` runs its first barrier: a pooled
    // resource left in a state that queue cannot transition from is not reused.
    Placement AcquireTexture(u32 slot, u64 offset, const rhi::TextureDesc& desc, rhi::QueueType firstQueue);
    Placement AcquireBuffer(u32 slot, u64 offset, const rhi::BufferDesc& desc, rhi::QueueType firstQueue);
    // The state the frame leaves a placed resource in.
    void SetFinalState(u32 slot, u32 entry, rhi::Access state);
    // Ends the frame, whose GPU work completes with `graphicsFence` on the graphics queue. `slot` is
    // the frame's heap, ~0u for a frame without transients. Releases what has been unused for long.
    void EndFrame(u32 slot, u64 graphicsFence);
    // Releases now what has been unused for `unusedFrames` frames (frames in flight + 1 keeps what
    // the next frames need), e.g. when the process nears its VRAM budget. Between frames only.
    void Trim(u32 unusedFrames);

    const Stats& GetStats() const { return mStats; }

private:
    struct Entry
    {
        u64 offset = 0;
        bool isTexture = true;
        rhi::TextureDesc textureDesc;
        rhi::BufferDesc bufferDesc;
        rhi::TextureHandle texture;
        rhi::BufferHandle buffer;
        rhi::Access state = rhi::Access::None;
        u64 lastUsed = 0; // pool frame
        bool inUse = false;
    };
    struct Heap
    {
        rhi::HeapHandle heap;
        u64 size = 0;
        u64 fence = 0;    // graphics fence of the last frame that used it
        u64 lastUsed = 0; // pool frame
        bool inUse = false;
        std::vector<Entry> entries;
    };

    Placement Acquire(u32 slot, u64 offset, const rhi::TextureDesc* textureDesc, const rhi::BufferDesc* bufferDesc,
                      rhi::QueueType firstQueue);
    void Release(Entry& entry);
    void ReleaseIdle(u32 unusedFrames);
    void UpdateTotals();

    rhi::Device& mDevice;
    Config mConfig;
    std::vector<Heap> mHeaps;
    u64 mFrame = 1;
    Stats mStats;
    Stats mCounters; // this frame's counts, published to mStats by EndFrame()
};

} // namespace air::rdg
