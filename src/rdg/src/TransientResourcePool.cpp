#include "air/rdg/TransientResourcePool.h"

#include "air/core/Assert.h"

#include <algorithm>
#include <utility>

namespace air::rdg
{

namespace
{

bool SameDesc(const rhi::TextureDesc& a, const rhi::TextureDesc& b)
{
    return a.dimension == b.dimension && a.format == b.format && a.width == b.width && a.height == b.height &&
           a.depthOrArraySize == b.depthOrArraySize && a.mipLevels == b.mipLevels && a.sampleCount == b.sampleCount &&
           a.usage == b.usage &&
           std::equal(std::begin(a.clearColor), std::end(a.clearColor), std::begin(b.clearColor)) &&
           a.debugName == b.debugName;
}

bool SameDesc(const rhi::BufferDesc& a, const rhi::BufferDesc& b)
{
    return a.size == b.size && a.stride == b.stride && a.usage == b.usage && a.memory == b.memory &&
           a.debugName == b.debugName;
}

u64 AlignUp(u64 value, u64 alignment)
{
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

} // namespace

TransientResourcePool::TransientResourcePool(rhi::Device& device, Config config) : mDevice(device), mConfig(config) {}

TransientResourcePool::~TransientResourcePool()
{
    for (Heap& heap : mHeaps)
    {
        for (Entry& entry : heap.entries)
        {
            Release(entry);
        }
        mDevice.DestroyHeap(heap.heap);
    }
}

u32 TransientResourcePool::AcquireHeap(u64 bytes, bool* created)
{
    if (created)
    {
        *created = false;
    }
    u32 best = ~0u;
    for (u32 i = 0; i < mHeaps.size(); ++i)
    {
        const Heap& heap = mHeaps[i];
        if (heap.inUse || heap.size < bytes || !mDevice.IsFenceComplete(rhi::QueueType::Graphics, heap.fence))
        {
            continue;
        }
        if (best == ~0u || heap.lastUsed < mHeaps[best].lastUsed)
        {
            best = i;
        }
    }
    if (best == ~0u)
    {
        Heap heap;
        heap.size = AlignUp(bytes, mConfig.heapGranularity);
        heap.heap = mDevice.CreateHeap(heap.size, rhi::MemoryLocation::DeviceLocal, "RDG Transient Pool");
        AIR_CHECK(heap.heap.IsValid(), "failed to allocate a {} byte transient heap", heap.size);
        mHeaps.push_back(std::move(heap));
        best = u32(mHeaps.size() - 1);
        ++mCounters.heapsCreated;
        if (created)
        {
            *created = true;
        }
    }
    Heap& heap = mHeaps[best];
    heap.inUse = true;
    heap.lastUsed = mFrame;
    return best;
}

TransientResourcePool::Placement
TransientResourcePool::AcquireTexture(u32 slot, u64 offset, const rhi::TextureDesc& desc, rhi::QueueType firstQueue)
{
    return Acquire(slot, offset, &desc, nullptr, firstQueue);
}

TransientResourcePool::Placement TransientResourcePool::AcquireBuffer(u32 slot, u64 offset, const rhi::BufferDesc& desc,
                                                                      rhi::QueueType firstQueue)
{
    return Acquire(slot, offset, nullptr, &desc, firstQueue);
}

TransientResourcePool::Placement TransientResourcePool::Acquire(u32 slot, u64 offset,
                                                                const rhi::TextureDesc* textureDesc,
                                                                const rhi::BufferDesc* bufferDesc,
                                                                rhi::QueueType firstQueue)
{
    Heap& heap = mHeaps[slot];
    AIR_CHECK(heap.inUse, "TransientResourcePool: heap slot {} was not acquired this frame", slot);
    for (u32 i = 0; i < heap.entries.size(); ++i)
    {
        Entry& entry = heap.entries[i];
        const bool same =
            entry.offset == offset && entry.isTexture == (textureDesc != nullptr) &&
            (textureDesc ? SameDesc(entry.textureDesc, *textureDesc) : SameDesc(entry.bufferDesc, *bufferDesc));
        // The compute queue cannot transition out of graphics-only states (a render target, ...).
        if (entry.inUse || !same || !rhi::IsAccessLegalOnQueue(entry.state, firstQueue))
        {
            continue;
        }
        entry.inUse = true;
        entry.lastUsed = mFrame;
        ++mCounters.resourcesReused;
        return {entry.texture, entry.buffer, entry.state, i, true};
    }
    Entry entry;
    entry.offset = offset;
    entry.isTexture = textureDesc != nullptr;
    if (textureDesc)
    {
        entry.textureDesc = *textureDesc;
        entry.texture = mDevice.CreatePlacedTexture(*textureDesc, heap.heap, offset);
        AIR_CHECK(entry.texture.IsValid(), "failed to place texture '{}'", textureDesc->debugName);
    }
    else
    {
        entry.bufferDesc = *bufferDesc;
        entry.buffer = mDevice.CreatePlacedBuffer(*bufferDesc, heap.heap, offset);
        AIR_CHECK(entry.buffer.IsValid(), "failed to place buffer '{}'", bufferDesc->debugName);
    }
    entry.inUse = true;
    entry.lastUsed = mFrame;
    heap.entries.push_back(std::move(entry));
    ++mCounters.resourcesCreated;
    const Entry& added = heap.entries.back();
    return {added.texture, added.buffer, rhi::Access::None, u32(heap.entries.size() - 1), false};
}

void TransientResourcePool::SetFinalState(u32 slot, u32 entry, rhi::Access state)
{
    mHeaps[slot].entries[entry].state = state;
}

void TransientResourcePool::Release(Entry& entry)
{
    if (entry.isTexture)
    {
        mDevice.DestroyTexture(entry.texture);
    }
    else
    {
        mDevice.DestroyBuffer(entry.buffer);
    }
}

void TransientResourcePool::EndFrame(u32 slot, u64 graphicsFence)
{
    if (slot != ~0u)
    {
        Heap& heap = mHeaps[slot];
        heap.inUse = false;
        heap.fence = graphicsFence;
        for (Entry& entry : heap.entries)
        {
            entry.inUse = false;
        }
    }
    // Release what has been idle for long: a resource that stopped being placed (a pass went away, a
    // resize), then a heap no frame fitted in (the frame shrank, or grew past it).
    ReleaseIdle(mConfig.retireAfterFrames);
    ++mFrame;

    mStats = mCounters;
    mCounters = {};
    UpdateTotals();
}

void TransientResourcePool::Trim(u32 unusedFrames)
{
    ReleaseIdle(unusedFrames);
    mStats.heapsReleased += std::exchange(mCounters.heapsReleased, 0u);
    mStats.resourcesReleased += std::exchange(mCounters.resourcesReleased, 0u);
    UpdateTotals();
}

void TransientResourcePool::UpdateTotals()
{
    mStats.heaps = 0;
    mStats.heapBytes = 0;
    mStats.resources = 0;
    for (const Heap& heap : mHeaps)
    {
        ++mStats.heaps;
        mStats.heapBytes += heap.size;
        mStats.resources += u32(heap.entries.size());
    }
}

void TransientResourcePool::ReleaseIdle(u32 unusedFrames)
{
    const auto idle = [&](u64 lastUsed) { return mFrame - lastUsed >= unusedFrames; };
    for (Heap& heap : mHeaps)
    {
        std::erase_if(heap.entries,
                      [&](Entry& entry)
                      {
                          if (!idle(entry.lastUsed))
                          {
                              return false;
                          }
                          Release(entry);
                          ++mCounters.resourcesReleased;
                          return true;
                      });
    }
    std::erase_if(mHeaps,
                  [&](Heap& heap)
                  {
                      if (heap.inUse || !idle(heap.lastUsed))
                      {
                          return false;
                      }
                      for (Entry& entry : heap.entries)
                      {
                          Release(entry);
                          ++mCounters.resourcesReleased;
                      }
                      mDevice.DestroyHeap(heap.heap);
                      ++mCounters.heapsReleased;
                      return true;
                  });
}

} // namespace air::rdg
