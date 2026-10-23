#include "TestFramework.h"

#include "air/rhi/GpuMemoryAllocator.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <random>
#include <thread>

using namespace air;
using namespace air::rhi;

namespace {

bool NoOverlap(std::vector<TlsfAllocator::Allocation> allocations) {
    std::sort(allocations.begin(), allocations.end(), [](auto& a, auto& b) { return a.offset < b.offset; });
    for (size_t i = 1; i < allocations.size(); ++i) {
        if (allocations[i - 1].offset + allocations[i - 1].size > allocations[i].offset) {
            return false;
        }
    }
    return true;
}

class CountingProvider final : public IGpuHeapProvider {
public:
    std::optional<GpuHeap> CreateHeap(u32, MemoryLocation location, u64 size) override {
        std::lock_guard lock(mutex);
        ++created;
        GpuHeap heap;
        heap.size = size;
        heap.mapped = location == MemoryLocation::DeviceLocal ? nullptr : new u8[size];
        heap.native = heap.mapped ? static_cast<void*>(heap.mapped) : new int(0);
        return heap;
    }
    void DestroyHeap(u32, const GpuHeap& heap) override {
        std::lock_guard lock(mutex);
        ++destroyed;
        if (heap.mapped) {
            delete[] heap.mapped;
        } else {
            delete static_cast<int*>(heap.native);
        }
    }
    std::mutex mutex;
    int created = 0;
    int destroyed = 0;
};

} // namespace

TEST(Tlsf_AlignedNonOverlappingAndCoalesces) {
    TlsfAllocator tlsf(1 << 20);
    std::vector<TlsfAllocator::Allocation> allocations;
    const u64 alignments[] = {256, 4096, 65536};
    for (int i = 0; i < 12; ++i) {
        auto a = tlsf.Allocate(1000 + i * 3000, alignments[i % 3]);
        EXPECT(a.has_value());
        if (a) {
            EXPECT(a->offset % alignments[i % 3] == 0);
            allocations.push_back(*a);
        }
    }
    EXPECT(tlsf.Validate());
    EXPECT(NoOverlap(allocations));
    for (auto& a : allocations) {
        tlsf.Free(a.node);
    }
    EXPECT(tlsf.Validate());
    EXPECT(tlsf.IsEmpty());
    // Everything coalesced back: the whole range is allocatable again.
    auto all = tlsf.Allocate(1 << 20, 256);
    EXPECT(all.has_value() && all->offset == 0);
}

TEST(Tlsf_FailsCleanlyWhenFull) {
    TlsfAllocator tlsf(64 * 1024);
    auto a = tlsf.Allocate(60 * 1024, 256);
    EXPECT(a.has_value());
    EXPECT(!tlsf.Allocate(8 * 1024, 256).has_value());
    EXPECT(tlsf.Validate());
}

TEST(Tlsf_RandomStress) {
    std::mt19937 rng(1234);
    TlsfAllocator tlsf(64ull << 20);
    std::vector<TlsfAllocator::Allocation> live;
    for (int step = 0; step < 20000; ++step) {
        if (live.empty() || rng() % 3 != 0) {
            const u64 size = 1 + rng() % (512 * 1024);
            const u64 align = 256ull << (rng() % 9);
            if (auto a = tlsf.Allocate(size, align)) {
                EXPECT(a->offset % align == 0 && a->size >= size);
                live.push_back(*a);
            }
        } else {
            const size_t i = rng() % live.size();
            tlsf.Free(live[i].node);
            live[i] = live.back();
            live.pop_back();
        }
        if (step % 2000 == 0) {
            EXPECT(tlsf.Validate());
            EXPECT(NoOverlap(live));
        }
    }
    for (auto& a : live) {
        tlsf.Free(a.node);
    }
    EXPECT(tlsf.Validate() && tlsf.IsEmpty());
}

TEST(GpuAllocator_PoolsDedicatedAndMapping) {
    CountingProvider provider;
    {
        GpuMemoryAllocator allocator(provider, {.defaultHeapSize = 1 << 20, .dedicatedThreshold = 512 << 10,
                                                .hostVisibleHeapSize = 1 << 20});
        auto a = allocator.Allocate({.memoryTypeKey = 0, .location = MemoryLocation::DeviceLocal, .size = 1000});
        auto b = allocator.Allocate({.memoryTypeKey = 0, .location = MemoryLocation::DeviceLocal, .size = 1000});
        EXPECT(a && b && a->nativeHeap == b->nativeHeap); // same block
        EXPECT(provider.created == 1);

        auto big = allocator.Allocate({.memoryTypeKey = 0, .location = MemoryLocation::DeviceLocal, .size = 600 << 10});
        EXPECT(big && big->nativeHeap != a->nativeHeap && big->offset == 0); // dedicated
        EXPECT(allocator.GetStats(MemoryLocation::DeviceLocal).dedicatedCount == 1);

        auto up = allocator.Allocate({.memoryTypeKey = 4, .location = MemoryLocation::Upload, .size = 300, .alignment = 256});
        auto up2 = allocator.Allocate({.memoryTypeKey = 4, .location = MemoryLocation::Upload, .size = 300, .alignment = 256});
        EXPECT(up && up2 && up->mapped && up2->mapped);
        EXPECT(up2->mapped - up->mapped == std::ptrdiff_t(up2->offset) - std::ptrdiff_t(up->offset));

        allocator.Free(*big);
        EXPECT(provider.destroyed == 1);
        allocator.Free(*a);
        allocator.Free(*b);
        allocator.Free(*up);
        allocator.Free(*up2);
        const auto total = allocator.GetTotalStats();
        EXPECT(total.allocationCount == 0 && total.usedBytes == 0);
    }
    EXPECT(provider.created == provider.destroyed);
}

TEST(GpuAllocator_ThreadSafeConcurrentUse) {
    CountingProvider provider;
    GpuMemoryAllocator allocator(provider, {.defaultHeapSize = 4 << 20, .dedicatedThreshold = 1 << 20,
                                            .hostVisibleHeapSize = 1 << 20});
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937 rng(t);
            std::vector<GpuAllocation> live;
            for (int i = 0; i < 2000; ++i) {
                if (live.size() < 32 && rng() % 2) {
                    const u32 key = rng() % 3;
                    auto a = allocator.Allocate({.memoryTypeKey = key, .location = MemoryLocation::DeviceLocal,
                                                 .size = 256 + rng() % (256 << 10)});
                    if (a) {
                        live.push_back(*a);
                    } else {
                        ++failures;
                    }
                } else if (!live.empty()) {
                    allocator.Free(live.back());
                    live.pop_back();
                }
            }
            for (auto& a : live) {
                allocator.Free(a);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    EXPECT(failures.load() == 0);
    EXPECT(allocator.GetTotalStats().allocationCount == 0);
}
