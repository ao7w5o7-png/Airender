#pragma once

// GPU memory management, backend independent.
//
//   TlsfAllocator       O(1) offset allocator (Two-Level Segregated Fit) over one address range.
//                       Knows nothing about GPUs; manages offsets only. Not thread-safe.
//   GpuMemoryAllocator  Pools of large heaps per "memory type key", sub-allocated with TLSF.
//                       Large requests get a dedicated heap. Thread-safe (one mutex per pool).
//
// The backend decides what a memory type key means: on Vulkan it is the memoryTypeIndex, on D3D12
// it encodes (heap type, heap tier category: buffers / RT-DS textures / other textures).
// The backend also supplies IGpuHeapProvider, which creates the actual ID3D12Heap / VkDeviceMemory.

#include "air/core/Types.h"
#include "air/rhi/RHITypes.h"

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace air::rhi
{

class TlsfAllocator
{
public:
    static constexpr u64 kMinBlockSize = 256; // allocation granularity (D3D12 CB alignment)

    struct Allocation
    {
        u64 offset = 0;
        u64 size = 0;
        u32 node = ~0u; // opaque id passed back to Free()
    };

    explicit TlsfAllocator(u64 capacity);

    std::optional<Allocation> Allocate(u64 size, u64 alignment);
    void Free(u32 node);

    u64 GetCapacity() const { return mCapacity; }
    u64 GetUsedBytes() const { return mUsed; }
    u32 GetAllocationCount() const { return mAllocationCount; }
    bool IsEmpty() const { return mAllocationCount == 0; }
    // Debug: verifies the physical block chain and free lists. Returns false on corruption.
    bool Validate() const;

private:
    static constexpr u32 kSecondLevelLog2 = 3;
    static constexpr u32 kSecondLevelCount = 1u << kSecondLevelLog2;
    static constexpr u32 kFirstLevelShift = 8; // log2(kMinBlockSize)
    static constexpr u32 kFirstLevelCount = 64 - kFirstLevelShift;
    static constexpr u32 kNull = ~0u;

    struct Node
    {
        u64 offset = 0;
        u64 size = 0;
        u32 prevPhys = kNull;
        u32 nextPhys = kNull;
        u32 prevFree = kNull;
        u32 nextFree = kNull;
        bool free = false;
        bool inUse = false; // slot holds a live block
    };

    static void MapInsert(u64 size, u32& fl, u32& sl);
    static void MapSearch(u64 size, u32& fl, u32& sl);
    u32 FindSuitable(u32 fl, u32 sl) const;
    void InsertFree(u32 node);
    void RemoveFree(u32 node);
    u32 NewNode();
    void ReleaseNode(u32 node);

    u64 mCapacity;
    u64 mUsed = 0;
    u32 mAllocationCount = 0;
    std::vector<Node> mNodes;
    std::vector<u32> mFreeNodeSlots;
    u64 mFirstLevelBitmap = 0;
    u32 mSecondLevelBitmap[kFirstLevelCount] = {};
    u32 mHeads[kFirstLevelCount][kSecondLevelCount];
};

// A heap created by the backend (ID3D12Heap, VkDeviceMemory, or CPU memory for the Null backend).
struct GpuHeap
{
    void* native = nullptr;
    u8* mapped = nullptr; // persistent CPU mapping for host visible memory
    u64 size = 0;
};

class IGpuHeapProvider
{
public:
    virtual ~IGpuHeapProvider() = default;
    virtual std::optional<GpuHeap> CreateHeap(u32 memoryTypeKey, MemoryLocation location, u64 size) = 0;
    virtual void DestroyHeap(u32 memoryTypeKey, const GpuHeap& heap) = 0;
};

struct GpuAllocation
{
    void* nativeHeap = nullptr;
    u8* mapped = nullptr; // already offset; nullptr for device-only memory
    u64 offset = 0;
    u64 size = 0;
    u32 memoryTypeKey = 0;
    MemoryLocation location = MemoryLocation::DeviceLocal;

    bool IsValid() const { return nativeHeap != nullptr; }

private:
    friend class GpuMemoryAllocator;
    u32 mHeapIndex = 0;
    u32 mTlsfNode = ~0u;
    bool mDedicated = false;
};

class GpuMemoryAllocator
{
public:
    struct Config
    {
        u64 defaultHeapSize = 256ull << 20;    // 256 MiB blocks
        u64 dedicatedThreshold = 64ull << 20;  // >= 64 MiB gets its own heap
        u64 hostVisibleHeapSize = 64ull << 20; // smaller blocks for Upload / Readback
        bool freeEmptyHeaps = true;            // keep at most one empty heap per pool
    };

    struct Request
    {
        u32 memoryTypeKey = 0;
        MemoryLocation location = MemoryLocation::DeviceLocal;
        u64 size = 0;
        u64 alignment = 256;
        bool forceDedicated = false;
    };

    struct Stats
    {
        u64 reservedBytes = 0; // sum of heap sizes
        u64 usedBytes = 0;     // sum of live allocation sizes
        u32 heapCount = 0;
        u32 allocationCount = 0;
        u32 dedicatedCount = 0;
    };

    GpuMemoryAllocator(IGpuHeapProvider& provider, Config config);
    explicit GpuMemoryAllocator(IGpuHeapProvider& provider) : GpuMemoryAllocator(provider, Config{}) {}
    ~GpuMemoryAllocator();

    std::optional<GpuAllocation> Allocate(const Request& request);
    void Free(const GpuAllocation& allocation);

    // Per heap block usage, for memory visualisation (fragmentation, dedicated heaps).
    struct HeapUsage
    {
        u64 id = 0; // the native heap's address: ResourceInfo::heapId of the resources in it
        u64 size = 0;
        u64 usedBytes = 0;
        u32 allocationCount = 0;
        bool dedicated = false;
    };
    struct PoolStats
    {
        u32 memoryTypeKey = 0;
        MemoryLocation location = MemoryLocation::DeviceLocal;
        Stats stats;
        std::vector<HeapUsage> heaps;
    };

    Stats GetStats(MemoryLocation location) const;
    Stats GetTotalStats() const;
    std::vector<PoolStats> GetPoolStats() const; // sorted by memory type key

private:
    struct Block
    {
        GpuHeap heap;
        std::unique_ptr<TlsfAllocator> tlsf; // null for dedicated heaps
        bool alive = false;
    };
    struct Pool
    {
        mutable std::mutex mutex;
        MemoryLocation location = MemoryLocation::DeviceLocal;
        std::vector<Block> blocks;
        std::vector<u32> freeBlockSlots;
        u64 usedBytes = 0;
        u32 allocationCount = 0;
        u32 dedicatedCount = 0;
    };

    Pool& GetPool(u32 memoryTypeKey, MemoryLocation location);
    Pool* FindPool(u32 memoryTypeKey) const;
    u32 AddBlock(Pool& pool, Block block);

    IGpuHeapProvider& mProvider;
    Config mConfig;
    mutable std::mutex mPoolsMutex; // guards the map only; pools have their own locks
    std::unordered_map<u32, std::unique_ptr<Pool>> mPools;
};

} // namespace air::rhi
