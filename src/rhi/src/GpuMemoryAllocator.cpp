#include "air/rhi/GpuMemoryAllocator.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include <algorithm>
#include <bit>

namespace air::rhi {

// ===========================================================================
// TlsfAllocator
// ===========================================================================

TlsfAllocator::TlsfAllocator(u64 capacity) : mCapacity(capacity & ~(kMinBlockSize - 1)) {
    AIR_CHECK(mCapacity >= kMinBlockSize, "TLSF capacity {} too small", capacity);
    for (auto& row : mHeads) {
        for (u32& head : row) {
            head = kNull;
        }
    }
    const u32 root = NewNode();
    mNodes[root].offset = 0;
    mNodes[root].size = mCapacity;
    InsertFree(root);
}

void TlsfAllocator::MapInsert(u64 size, u32& fl, u32& sl) {
    const u32 log2 = u32(std::bit_width(size)) - 1;
    sl = u32(size >> (log2 - kSecondLevelLog2)) - kSecondLevelCount;
    fl = log2 - kFirstLevelShift;
}

void TlsfAllocator::MapSearch(u64 size, u32& fl, u32& sl) {
    const u32 log2 = u32(std::bit_width(size)) - 1;
    // Round up to the next size class so every block in the found list is large enough.
    size += (1ull << (log2 - kSecondLevelLog2)) - 1;
    MapInsert(size, fl, sl);
}

u32 TlsfAllocator::FindSuitable(u32 fl, u32 sl) const {
    if (fl >= kFirstLevelCount) {
        return kNull;
    }
    u32 slMap = mSecondLevelBitmap[fl] & (~0u << sl);
    if (slMap == 0) {
        const u64 flMap = fl + 1 < 64 ? mFirstLevelBitmap & (~0ull << (fl + 1)) : 0;
        if (flMap == 0) {
            return kNull;
        }
        fl = u32(std::countr_zero(flMap));
        slMap = mSecondLevelBitmap[fl];
    }
    sl = u32(std::countr_zero(slMap));
    return mHeads[fl][sl];
}

void TlsfAllocator::InsertFree(u32 index) {
    Node& node = mNodes[index];
    u32 fl, sl;
    MapInsert(node.size, fl, sl);
    node.free = true;
    node.prevFree = kNull;
    node.nextFree = mHeads[fl][sl];
    if (node.nextFree != kNull) {
        mNodes[node.nextFree].prevFree = index;
    }
    mHeads[fl][sl] = index;
    mFirstLevelBitmap |= 1ull << fl;
    mSecondLevelBitmap[fl] |= 1u << sl;
}

void TlsfAllocator::RemoveFree(u32 index) {
    Node& node = mNodes[index];
    u32 fl, sl;
    MapInsert(node.size, fl, sl);
    if (node.prevFree != kNull) {
        mNodes[node.prevFree].nextFree = node.nextFree;
    }
    if (node.nextFree != kNull) {
        mNodes[node.nextFree].prevFree = node.prevFree;
    }
    if (mHeads[fl][sl] == index) {
        mHeads[fl][sl] = node.nextFree;
        if (node.nextFree == kNull) {
            mSecondLevelBitmap[fl] &= ~(1u << sl);
            if (mSecondLevelBitmap[fl] == 0) {
                mFirstLevelBitmap &= ~(1ull << fl);
            }
        }
    }
    node.prevFree = node.nextFree = kNull;
    node.free = false;
}

u32 TlsfAllocator::NewNode() {
    u32 index;
    if (!mFreeNodeSlots.empty()) {
        index = mFreeNodeSlots.back();
        mFreeNodeSlots.pop_back();
        mNodes[index] = Node{};
    } else {
        index = u32(mNodes.size());
        mNodes.emplace_back();
    }
    mNodes[index].inUse = true;
    return index;
}

void TlsfAllocator::ReleaseNode(u32 index) {
    mNodes[index] = Node{};
    mFreeNodeSlots.push_back(index);
}

std::optional<TlsfAllocator::Allocation> TlsfAllocator::Allocate(u64 size, u64 alignment) {
    if (size == 0) {
        return std::nullopt;
    }
    alignment = alignment < kMinBlockSize ? kMinBlockSize : alignment;
    AIR_CHECK(IsPow2(alignment), "alignment {} is not a power of two", alignment);
    size = AlignUp(size, kMinBlockSize);

    // Offsets are multiples of kMinBlockSize, so worst-case front padding is alignment - kMinBlockSize.
    const u64 searchSize = size + alignment - kMinBlockSize;
    if (searchSize > mCapacity) {
        return std::nullopt;
    }
    u32 fl, sl;
    MapSearch(searchSize, fl, sl);
    const u32 index = FindSuitable(fl, sl);
    if (index == kNull) {
        return std::nullopt;
    }
    RemoveFree(index);

    // Split off front padding as its own free block. The previous physical block is never free
    // (free neighbours are always coalesced), so no merge is needed.
    const u64 alignedOffset = AlignUp(mNodes[index].offset, alignment);
    const u64 padding = alignedOffset - mNodes[index].offset;
    if (padding > 0) {
        const u32 front = NewNode();
        Node& node = mNodes[index];
        mNodes[front].offset = node.offset;
        mNodes[front].size = padding;
        mNodes[front].prevPhys = node.prevPhys;
        mNodes[front].nextPhys = index;
        if (node.prevPhys != kNull) {
            mNodes[node.prevPhys].nextPhys = front;
        }
        node.prevPhys = front;
        node.offset += padding;
        node.size -= padding;
        InsertFree(front);
    }

    // Split off the tail.
    if (mNodes[index].size - size >= kMinBlockSize) {
        const u32 tail = NewNode();
        Node& node = mNodes[index];
        mNodes[tail].offset = node.offset + size;
        mNodes[tail].size = node.size - size;
        mNodes[tail].prevPhys = index;
        mNodes[tail].nextPhys = node.nextPhys;
        if (node.nextPhys != kNull) {
            mNodes[node.nextPhys].prevPhys = tail;
        }
        node.nextPhys = tail;
        node.size = size;
        InsertFree(tail);
    }

    Node& node = mNodes[index];
    mUsed += node.size;
    ++mAllocationCount;
    return Allocation{node.offset, node.size, index};
}

void TlsfAllocator::Free(u32 index) {
    AIR_CHECK(index < mNodes.size() && mNodes[index].inUse && !mNodes[index].free, "TLSF double free of node {}",
              index);
    mUsed -= mNodes[index].size;
    --mAllocationCount;

    const u32 prev = mNodes[index].prevPhys;
    if (prev != kNull && mNodes[prev].free) {
        RemoveFree(prev);
        Node& node = mNodes[index];
        node.offset = mNodes[prev].offset;
        node.size += mNodes[prev].size;
        node.prevPhys = mNodes[prev].prevPhys;
        if (node.prevPhys != kNull) {
            mNodes[node.prevPhys].nextPhys = index;
        }
        ReleaseNode(prev);
    }
    const u32 next = mNodes[index].nextPhys;
    if (next != kNull && mNodes[next].free) {
        RemoveFree(next);
        Node& node = mNodes[index];
        node.size += mNodes[next].size;
        node.nextPhys = mNodes[next].nextPhys;
        if (node.nextPhys != kNull) {
            mNodes[node.nextPhys].prevPhys = index;
        }
        ReleaseNode(next);
    }
    InsertFree(index);
}

bool TlsfAllocator::Validate() const {
    u32 first = kNull;
    for (u32 i = 0; i < mNodes.size(); ++i) {
        if (mNodes[i].inUse && mNodes[i].prevPhys == kNull) {
            if (first != kNull) {
                return false; // two chain heads
            }
            first = i;
        }
    }
    if (first == kNull) {
        return false;
    }
    u64 expectedOffset = 0;
    u64 used = 0;
    u32 freeInChain = 0;
    bool prevFree = false;
    for (u32 i = first; i != kNull; i = mNodes[i].nextPhys) {
        const Node& node = mNodes[i];
        if (node.offset != expectedOffset || node.size == 0 || (node.free && prevFree)) {
            return false;
        }
        if (node.nextPhys != kNull && mNodes[node.nextPhys].prevPhys != i) {
            return false;
        }
        expectedOffset += node.size;
        used += node.free ? 0 : node.size;
        freeInChain += node.free ? 1 : 0;
        prevFree = node.free;
    }
    u32 freeInLists = 0;
    for (u32 fl = 0; fl < kFirstLevelCount; ++fl) {
        for (u32 sl = 0; sl < kSecondLevelCount; ++sl) {
            const bool bit = (mSecondLevelBitmap[fl] >> sl) & 1u;
            if (bit != (mHeads[fl][sl] != kNull)) {
                return false;
            }
            for (u32 i = mHeads[fl][sl]; i != kNull; i = mNodes[i].nextFree) {
                u32 nodeFl, nodeSl;
                MapInsert(mNodes[i].size, nodeFl, nodeSl);
                if (!mNodes[i].free || nodeFl != fl || nodeSl != sl) {
                    return false;
                }
                ++freeInLists;
            }
        }
    }
    return expectedOffset == mCapacity && used == mUsed && freeInChain == freeInLists;
}

// ===========================================================================
// GpuMemoryAllocator
// ===========================================================================

GpuMemoryAllocator::GpuMemoryAllocator(IGpuHeapProvider& provider, Config config)
    : mProvider(provider), mConfig(config) {}

GpuMemoryAllocator::~GpuMemoryAllocator() {
    for (auto& [key, pool] : mPools) {
        if (pool->allocationCount > 0) {
            AIR_LOG(Warning, "GpuMemory", "pool {} ({}) destroyed with {} live allocations ({} bytes)", key,
                    ToString(pool->location), pool->allocationCount, pool->usedBytes);
        }
        for (Block& block : pool->blocks) {
            if (block.alive) {
                mProvider.DestroyHeap(key, block.heap);
            }
        }
    }
}

GpuMemoryAllocator::Pool& GpuMemoryAllocator::GetPool(u32 memoryTypeKey, MemoryLocation location) {
    std::lock_guard lock(mPoolsMutex);
    auto& pool = mPools[memoryTypeKey];
    if (!pool) {
        pool = std::make_unique<Pool>();
        pool->location = location;
    }
    AIR_CHECK(pool->location == location, "memory type key {} used with two memory locations", memoryTypeKey);
    return *pool;
}

GpuMemoryAllocator::Pool* GpuMemoryAllocator::FindPool(u32 memoryTypeKey) const {
    std::lock_guard lock(mPoolsMutex);
    auto it = mPools.find(memoryTypeKey);
    return it == mPools.end() ? nullptr : it->second.get();
}

u32 GpuMemoryAllocator::AddBlock(Pool& pool, Block block) {
    block.alive = true;
    if (!pool.freeBlockSlots.empty()) {
        const u32 index = pool.freeBlockSlots.back();
        pool.freeBlockSlots.pop_back();
        pool.blocks[index] = std::move(block);
        return index;
    }
    pool.blocks.push_back(std::move(block));
    return u32(pool.blocks.size() - 1);
}

std::optional<GpuAllocation> GpuMemoryAllocator::Allocate(const Request& request) {
    AIR_CHECK(request.size > 0, "zero sized GPU allocation");
    Pool& pool = GetPool(request.memoryTypeKey, request.location);

    GpuAllocation result;
    result.memoryTypeKey = request.memoryTypeKey;
    result.location = request.location;

    const bool dedicated = request.forceDedicated || request.size >= mConfig.dedicatedThreshold;

    std::lock_guard lock(pool.mutex);
    if (dedicated) {
        const u64 size = AlignUp(request.size, TlsfAllocator::kMinBlockSize);
        auto heap = mProvider.CreateHeap(request.memoryTypeKey, request.location, size);
        if (!heap) {
            AIR_LOG(Error, "GpuMemory", "out of memory: dedicated {} bytes in {}", size, ToString(request.location));
            return std::nullopt;
        }
        result.heapIndex = AddBlock(pool, Block{*heap, nullptr});
        result.dedicated = true;
        result.nativeHeap = heap->native;
        result.mapped = heap->mapped;
        result.offset = 0;
        result.size = size;
        pool.usedBytes += size;
        ++pool.allocationCount;
        ++pool.dedicatedCount;
        return result;
    }

    auto fill = [&](u32 blockIndex, const TlsfAllocator::Allocation& a) {
        const Block& block = pool.blocks[blockIndex];
        result.heapIndex = blockIndex;
        result.tlsfNode = a.node;
        result.nativeHeap = block.heap.native;
        result.mapped = block.heap.mapped ? block.heap.mapped + a.offset : nullptr;
        result.offset = a.offset;
        result.size = a.size;
        pool.usedBytes += a.size;
        ++pool.allocationCount;
    };

    for (u32 i = 0; i < pool.blocks.size(); ++i) {
        Block& block = pool.blocks[i];
        if (!block.alive || !block.tlsf) {
            continue;
        }
        if (auto a = block.tlsf->Allocate(request.size, request.alignment)) {
            fill(i, *a);
            return result;
        }
    }

    const bool hostVisible = request.location == MemoryLocation::Upload || request.location == MemoryLocation::Readback;
    u64 heapSize = hostVisible ? mConfig.hostVisibleHeapSize : mConfig.defaultHeapSize;
    heapSize = std::max(heapSize, AlignUp(request.size + request.alignment, TlsfAllocator::kMinBlockSize));
    auto heap = mProvider.CreateHeap(request.memoryTypeKey, request.location, heapSize);
    if (!heap) {
        AIR_LOG(Error, "GpuMemory", "out of memory: heap of {} bytes in {}", heapSize, ToString(request.location));
        return std::nullopt;
    }
    const u32 blockIndex = AddBlock(pool, Block{*heap, std::make_unique<TlsfAllocator>(heap->size)});
    auto a = pool.blocks[blockIndex].tlsf->Allocate(request.size, request.alignment);
    AIR_CHECK(a.has_value(), "fresh heap could not satisfy allocation");
    fill(blockIndex, *a);
    return result;
}

void GpuMemoryAllocator::Free(const GpuAllocation& allocation) {
    if (!allocation.IsValid()) {
        return;
    }
    Pool* pool = FindPool(allocation.memoryTypeKey);
    AIR_CHECK(pool, "freeing allocation from unknown memory type {}", allocation.memoryTypeKey);

    std::lock_guard lock(pool->mutex);
    AIR_CHECK(allocation.heapIndex < pool->blocks.size() && pool->blocks[allocation.heapIndex].alive,
              "freeing allocation from dead heap");
    Block& block = pool->blocks[allocation.heapIndex];
    pool->usedBytes -= allocation.size;
    --pool->allocationCount;

    auto destroyBlock = [&](u32 index) {
        mProvider.DestroyHeap(allocation.memoryTypeKey, pool->blocks[index].heap);
        pool->blocks[index] = Block{};
        pool->freeBlockSlots.push_back(index);
    };

    if (allocation.dedicated) {
        --pool->dedicatedCount;
        destroyBlock(allocation.heapIndex);
        return;
    }

    block.tlsf->Free(allocation.tlsfNode);
    if (mConfig.freeEmptyHeaps && block.tlsf->IsEmpty()) {
        // Keep one empty heap around to avoid create/destroy thrashing.
        for (u32 i = 0; i < pool->blocks.size(); ++i) {
            const Block& other = pool->blocks[i];
            if (i != allocation.heapIndex && other.alive && other.tlsf && other.tlsf->IsEmpty()) {
                destroyBlock(allocation.heapIndex);
                return;
            }
        }
    }
}

GpuMemoryAllocator::Stats GpuMemoryAllocator::GetStats(MemoryLocation location) const {
    Stats stats;
    std::lock_guard lock(mPoolsMutex);
    for (const auto& [key, pool] : mPools) {
        if (pool->location != location) {
            continue;
        }
        std::lock_guard poolLock(pool->mutex);
        for (const Block& block : pool->blocks) {
            if (block.alive) {
                stats.reservedBytes += block.heap.size;
                ++stats.heapCount;
            }
        }
        stats.usedBytes += pool->usedBytes;
        stats.allocationCount += pool->allocationCount;
        stats.dedicatedCount += pool->dedicatedCount;
    }
    return stats;
}

GpuMemoryAllocator::Stats GpuMemoryAllocator::GetTotalStats() const {
    Stats total;
    for (u32 i = 0; i < u32(MemoryLocation::Count); ++i) {
        const Stats s = GetStats(MemoryLocation(i));
        total.reservedBytes += s.reservedBytes;
        total.usedBytes += s.usedBytes;
        total.heapCount += s.heapCount;
        total.allocationCount += s.allocationCount;
        total.dedicatedCount += s.dedicatedCount;
    }
    return total;
}

} // namespace air::rhi
