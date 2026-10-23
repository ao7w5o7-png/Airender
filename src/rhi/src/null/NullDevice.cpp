// Null RHI backend: no GPU. Resources get real memory bookkeeping (through GpuMemoryAllocator),
// host visible buffers get real CPU memory, and every command is captured as text. Barriers are
// validated against tracked resource state. Used by tests, CI and headless AI debugging sessions.

#include "../Backends.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include <algorithm>
#include <atomic>
#include <format>
#include <functional>
#include <unordered_map>
#include <utility>

namespace air::rhi {
namespace {

enum class HeapCategory : u32 { Buffer = 0, RenderTargetTexture = 1, Texture = 2 };

// Same encoding the D3D12 backend uses: heap tier 1 separates these categories.
u32 MemoryTypeKey(MemoryLocation location, HeapCategory category) {
    return u32(location) * 4 + u32(category);
}

class NullHeapProvider final : public IGpuHeapProvider {
public:
    std::optional<GpuHeap> CreateHeap(u32 memoryTypeKey, MemoryLocation location, u64 size) override {
        GpuHeap heap;
        heap.size = size;
        const bool hostVisible = location != MemoryLocation::DeviceLocal;
        if (hostVisible) {
            heap.mapped = new u8[size]();
            heap.native = heap.mapped;
        } else {
            heap.native = new u64(size); // placeholder identity for device-only memory
        }
        return heap;
    }
    void DestroyHeap(u32 memoryTypeKey, const GpuHeap& heap) override {
        if (heap.mapped) {
            delete[] heap.mapped;
        } else {
            delete static_cast<u64*>(heap.native);
        }
    }
};

struct NullBuffer {
    BufferDesc desc;
    GpuAllocation allocation;
    bool placed = false;
    BindlessIndex srv = kInvalidBindless;
    BindlessIndex uav = kInvalidBindless;
};

struct NullTexture {
    TextureDesc desc;
    GpuAllocation allocation;
    bool placed = false;
    BindlessIndex srv = kInvalidBindless;
    BindlessIndex uav = kInvalidBindless;
};

struct NullHeap {
    u64 size = 0;
    MemoryLocation location = MemoryLocation::DeviceLocal;
    GpuAllocation allocation;
    std::string name;
};

struct NullAccelStruct {
    bool isTlas = false;
    BlasDesc blas;
    TlasDesc tlas;
    BindlessIndex srv = kInvalidBindless;
};

struct NullPipeline {
    bool rayTracing = false;
    std::string name;
};

class NullDevice;

struct RecordedCommand {
    std::string text;
    std::vector<Barrier> barriers; // non-empty for barrier batches, validated at submit
};

class NullCommandList final : public CommandList {
public:
    NullCommandList(NullDevice& device, QueueType queue) : mDevice(device), mQueue(queue) {}

    QueueType GetQueueType() const override { return mQueue; }
    void Begin() override {
        AIR_CHECK(!mRecording, "Begin() on a command list that is already recording");
        mCommands.clear();
        mRecording = true;
        mEventDepth = 0;
    }
    void End() override {
        AIR_CHECK(mRecording, "End() without Begin()");
        AIR_CHECK(mEventDepth == 0, "{} unbalanced BeginEvent()", mEventDepth);
        mRecording = false;
    }

    void Barriers(std::span<const Barrier> barriers) override;
    void CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size) override;
    void CopyBufferToTexture(TextureHandle dst, u32 mip, BufferHandle src, u64 srcOffset) override;
    void CopyTexture(TextureHandle dst, TextureHandle src) override;
    void ClearTexture(TextureHandle texture, const float color[4]) override;
    void SetPipeline(PipelineHandle pipeline) override;
    void PushConstants(const void* data, u32 size) override {
        AIR_CHECK(size <= 128, "push constants limited to 128 bytes, got {}", size);
        Record(std::format("PushConstants {} bytes", size));
    }
    void Dispatch(u32 x, u32 y, u32 z) override {
        AIR_CHECK(mPipeline.IsValid(), "Dispatch without pipeline");
        Record(std::format("Dispatch {}x{}x{}", x, y, z));
    }
    void DispatchRays(u32 w, u32 h, u32 d) override {
        AIR_CHECK(mPipeline.IsValid(), "DispatchRays without pipeline");
        Record(std::format("DispatchRays {}x{}x{}", w, h, d));
    }
    void BuildBlas(AccelStructHandle blas, BufferHandle scratch) override;
    void BuildTlas(AccelStructHandle tlas, std::span<const TlasInstance> instances, BufferHandle scratch) override;
    void BeginEvent(std::string_view name) override {
        ++mEventDepth;
        Record(std::format("BeginEvent '{}'", name));
    }
    void EndEvent() override {
        AIR_CHECK(mEventDepth > 0, "EndEvent without BeginEvent");
        --mEventDepth;
        Record("EndEvent");
    }

    std::vector<RecordedCommand>& Commands() { return mCommands; }

private:
    void Record(std::string text, std::vector<Barrier> barriers = {}) {
        AIR_CHECK(mRecording, "recording into a command list outside Begin()/End()");
        mCommands.push_back({std::move(text), std::move(barriers)});
    }

    NullDevice& mDevice;
    QueueType mQueue;
    bool mRecording = false;
    u32 mEventDepth = 0;
    PipelineHandle mPipeline;
    std::vector<RecordedCommand> mCommands;
};

const char* QueueName(QueueType queue) {
    switch (queue) {
    case QueueType::Graphics: return "Graphics";
    case QueueType::Compute: return "Compute";
    case QueueType::Copy: return "Copy";
    case QueueType::Count: break;
    }
    return "?";
}

class NullDevice final : public Device {
public:
    explicit NullDevice(const DeviceDesc& desc) : mDesc(desc), mAllocator(mHeapProvider) {
        mCaps.rayTracing = true; // pretend, so ray tracing code paths can be exercised headless
        mCaps.bindless = true;
        mCaps.rebar = true;
        mCaps.dedicatedVideoMemory = 8ull << 30;
        mCaps.adapterName = "Null Adapter";
    }

    ~NullDevice() override { WaitIdle(); }

    Backend GetBackend() const override { return Backend::Null; }
    const DeviceCapabilities& GetCapabilities() const override { return mCaps; }

    // --- resources -------------------------------------------------------------------------
    BufferHandle CreateBuffer(const BufferDesc& desc) override {
        AIR_CHECK(desc.size > 0, "buffer '{}' has zero size", desc.debugName);
        NullBuffer buffer;
        buffer.desc = desc;
        const AllocationInfo info = GetBufferAllocationInfo(desc);
        auto allocation = mAllocator.Allocate(
            {MemoryTypeKey(desc.memory, HeapCategory::Buffer), desc.memory, info.size, info.alignment});
        if (!allocation) {
            return {};
        }
        buffer.allocation = *allocation;
        return mBuffers.Allocate(std::move(buffer));
    }

    TextureHandle CreateTexture(const TextureDesc& desc) override {
        NullTexture texture;
        texture.desc = desc;
        const AllocationInfo info = GetTextureAllocationInfo(desc);
        auto allocation = mAllocator.Allocate({MemoryTypeKey(MemoryLocation::DeviceLocal, TextureCategory(desc)),
                                               MemoryLocation::DeviceLocal, info.size, info.alignment});
        if (!allocation) {
            return {};
        }
        texture.allocation = *allocation;
        return mTextures.Allocate(std::move(texture));
    }

    void DestroyBuffer(BufferHandle handle) override {
        DeferDelete([this, handle] {
            NullBuffer* buffer = mBuffers.Get(handle);
            AIR_CHECK(buffer, "DestroyBuffer on stale handle");
            if (!buffer->placed) {
                mAllocator.Free(buffer->allocation);
            }
            ReleaseBindless(buffer->srv);
            ReleaseBindless(buffer->uav);
            ForgetState(StateKey(handle));
            mBuffers.Free(handle);
        });
    }

    void DestroyTexture(TextureHandle handle) override {
        DeferDelete([this, handle] {
            NullTexture* texture = mTextures.Get(handle);
            AIR_CHECK(texture, "DestroyTexture on stale handle");
            if (!texture->placed) {
                mAllocator.Free(texture->allocation);
            }
            ReleaseBindless(texture->srv);
            ReleaseBindless(texture->uav);
            ForgetState(StateKey(handle));
            mTextures.Free(handle);
        });
    }

    const BufferDesc* GetBufferDesc(BufferHandle handle) override {
        NullBuffer* buffer = mBuffers.Get(handle);
        return buffer ? &buffer->desc : nullptr;
    }
    const TextureDesc* GetTextureDesc(TextureHandle handle) override {
        NullTexture* texture = mTextures.Get(handle);
        return texture ? &texture->desc : nullptr;
    }
    void* GetMappedPointer(BufferHandle handle) override {
        NullBuffer* buffer = mBuffers.Get(handle);
        return buffer ? buffer->allocation.mapped : nullptr;
    }

    AllocationInfo GetTextureAllocationInfo(const TextureDesc& desc) override {
        u64 size = 0;
        u32 w = desc.width, h = desc.height;
        const u32 layers =
            desc.dimension == TextureDimension::TexCube ? 6 * desc.depthOrArraySize : desc.depthOrArraySize;
        for (u32 mip = 0; mip < desc.mipLevels; ++mip) {
            size += u64(w) * h * FormatBytesPerPixel(desc.format);
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
        }
        size *= u64(layers) * desc.sampleCount;
        return {AlignUp(std::max<u64>(size, 1), 64 * 1024), 64 * 1024}; // D3D12 default placement alignment
    }
    AllocationInfo GetBufferAllocationInfo(const BufferDesc& desc) override { return {AlignUp(desc.size, 256), 256}; }

    HeapHandle CreateHeap(u64 size, MemoryLocation location, std::string_view debugName) override {
        NullHeap heap;
        heap.size = size;
        heap.location = location;
        heap.name = std::string(debugName);
        // Transient heaps hold every resource category (D3D12 heap tier 2 / Vulkan).
        auto allocation = mAllocator.Allocate(
            {MemoryTypeKey(location, HeapCategory::Texture) + 100, location, size, 64 * 1024, /*forceDedicated*/ true});
        if (!allocation) {
            return {};
        }
        heap.allocation = *allocation;
        return mHeaps.Allocate(std::move(heap));
    }

    void DestroyHeap(HeapHandle handle) override {
        DeferDelete([this, handle] {
            NullHeap* heap = mHeaps.Get(handle);
            AIR_CHECK(heap, "DestroyHeap on stale handle");
            mAllocator.Free(heap->allocation);
            mHeaps.Free(handle);
        });
    }

    TextureHandle CreatePlacedTexture(const TextureDesc& desc, HeapHandle heapHandle, u64 offset) override {
        NullHeap* heap = mHeaps.Get(heapHandle);
        AIR_CHECK(heap, "CreatePlacedTexture: invalid heap");
        const AllocationInfo info = GetTextureAllocationInfo(desc);
        AIR_CHECK(offset % info.alignment == 0, "placed texture '{}' misaligned offset {}", desc.debugName, offset);
        AIR_CHECK(offset + info.size <= heap->size, "placed texture '{}' overflows heap '{}'", desc.debugName,
                  heap->name);
        NullTexture texture;
        texture.desc = desc;
        texture.placed = true;
        texture.allocation = heap->allocation;
        texture.allocation.offset += offset;
        texture.allocation.size = info.size;
        return mTextures.Allocate(std::move(texture));
    }

    BufferHandle CreatePlacedBuffer(const BufferDesc& desc, HeapHandle heapHandle, u64 offset) override {
        NullHeap* heap = mHeaps.Get(heapHandle);
        AIR_CHECK(heap, "CreatePlacedBuffer: invalid heap");
        const AllocationInfo info = GetBufferAllocationInfo(desc);
        AIR_CHECK(offset % info.alignment == 0 && offset + info.size <= heap->size, "bad placement for buffer '{}'",
                  desc.debugName);
        NullBuffer buffer;
        buffer.desc = desc;
        buffer.placed = true;
        buffer.allocation = heap->allocation;
        buffer.allocation.offset += offset;
        buffer.allocation.size = info.size;
        if (buffer.allocation.mapped) {
            buffer.allocation.mapped += offset;
        }
        return mBuffers.Allocate(std::move(buffer));
    }

    BindlessIndex GetBindlessSRV(TextureHandle handle) override {
        NullTexture* t = mTextures.Get(handle);
        AIR_CHECK(t, "GetBindlessSRV: invalid texture");
        return LazyBindless(t->srv);
    }
    BindlessIndex GetBindlessUAV(TextureHandle handle, u32 mip) override {
        NullTexture* t = mTextures.Get(handle);
        AIR_CHECK(t, "GetBindlessUAV: invalid texture");
        AIR_CHECK(Any(t->desc.usage & TextureUsage::UnorderedAccess), "texture '{}' lacks UnorderedAccess usage",
                  t->desc.debugName);
        return LazyBindless(t->uav);
    }
    BindlessIndex GetBindlessSRV(BufferHandle handle) override {
        NullBuffer* b = mBuffers.Get(handle);
        AIR_CHECK(b, "GetBindlessSRV: invalid buffer");
        return LazyBindless(b->srv);
    }
    BindlessIndex GetBindlessUAV(BufferHandle handle) override {
        NullBuffer* b = mBuffers.Get(handle);
        AIR_CHECK(b, "GetBindlessUAV: invalid buffer");
        return LazyBindless(b->uav);
    }
    BindlessIndex GetBindlessSRV(AccelStructHandle handle) override {
        NullAccelStruct* as = mAccelStructs.Get(handle);
        AIR_CHECK(as && as->isTlas, "GetBindlessSRV: invalid TLAS");
        return LazyBindless(as->srv);
    }

    AccelStructHandle CreateBlas(const BlasDesc& desc) override {
        NullAccelStruct as;
        as.blas = desc;
        return mAccelStructs.Allocate(std::move(as));
    }
    AccelStructHandle CreateTlas(const TlasDesc& desc) override {
        NullAccelStruct as;
        as.isTlas = true;
        as.tlas = desc;
        return mAccelStructs.Allocate(std::move(as));
    }
    void DestroyAccelStruct(AccelStructHandle handle) override {
        DeferDelete([this, handle] {
            if (NullAccelStruct* as = mAccelStructs.Get(handle)) {
                ReleaseBindless(as->srv);
            }
            mAccelStructs.Free(handle);
        });
    }
    u64 GetAccelStructScratchSize(AccelStructHandle handle) override {
        NullAccelStruct* as = mAccelStructs.Get(handle);
        AIR_CHECK(as, "GetAccelStructScratchSize: invalid handle");
        if (as->isTlas) {
            return AlignUp(u64(as->tlas.maxInstances) * 64 + 256, 256);
        }
        u64 triangles = 0;
        for (const auto& g : as->blas.geometries) {
            triangles += g.indexCount / 3;
        }
        return AlignUp(triangles * 64 + 256, 256);
    }

    PipelineHandle CreateComputePipeline(const ComputePipelineDesc& desc) override {
        return mPipelines.Allocate(NullPipeline{false, desc.debugName});
    }
    PipelineHandle CreateRayTracingPipeline(const RayTracingPipelineDesc& desc) override {
        AIR_CHECK(!desc.rayGen.empty(), "ray tracing pipeline '{}' has no raygen shader", desc.debugName);
        return mPipelines.Allocate(NullPipeline{true, desc.debugName});
    }
    void DestroyPipeline(PipelineHandle handle) override {
        DeferDelete([this, handle] { mPipelines.Free(handle); });
    }

    // --- submission ------------------------------------------------------------------------
    std::unique_ptr<CommandList> CreateCommandList(QueueType queue) override {
        return std::make_unique<NullCommandList>(*this, queue);
    }

    u64 Submit(QueueType queue, std::span<CommandList* const> lists) override {
        std::lock_guard lock(mSubmitMutex);
        for (CommandList* list : lists) {
            AIR_CHECK(list->GetQueueType() == queue, "command list submitted to the wrong queue");
            for (RecordedCommand& command : static_cast<NullCommandList*>(list)->Commands()) {
                for (const Barrier& barrier : command.barriers) {
                    ValidateBarrierLocked(queue, barrier);
                }
                if (mCapture) {
                    mCaptured.push_back(std::format("[{}] {}", QueueName(queue), command.text));
                }
            }
        }
        return ++mFence[size_t(queue)]; // the null GPU finishes instantly
    }

    bool IsFenceComplete(QueueType queue, u64 value) override {
        std::lock_guard lock(mSubmitMutex);
        return value <= mFence[size_t(queue)];
    }
    void WaitForFence(QueueType, u64) override {}

    void WaitIdle() override { FlushDeferredDeletes(~0ull); }

    void BeginFrame() override {}
    void EndFrame() override {
        const u64 frame = mFrameIndex.fetch_add(1) + 1;
        // A real GPU may still be working on the last framesInFlight frames.
        if (frame >= mDesc.framesInFlight) {
            FlushDeferredDeletes(frame - mDesc.framesInFlight);
        }
    }
    u64 GetFrameIndex() const override { return mFrameIndex.load(); }

    MemoryStats GetMemoryStats() override {
        MemoryStats stats;
        for (u32 i = 0; i < u32(MemoryLocation::Count); ++i) {
            stats.perLocation[i] = mAllocator.GetStats(MemoryLocation(i));
        }
        stats.liveBuffers = mBuffers.LiveCount();
        stats.liveTextures = mTextures.LiveCount();
        return stats;
    }

    void SetCommandCapture(bool enabled) override {
        std::lock_guard lock(mSubmitMutex);
        mCapture = enabled;
    }
    std::vector<std::string> TakeCapturedCommands() override {
        std::lock_guard lock(mSubmitMutex);
        return std::exchange(mCaptured, {});
    }

    // --- helpers used by the command list ----------------------------------------------------
    std::string Describe(TextureHandle handle) {
        NullTexture* t = mTextures.Get(handle);
        return t ? std::format("tex '{}'", t->desc.debugName) : "tex <invalid>";
    }
    std::string Describe(BufferHandle handle) {
        NullBuffer* b = mBuffers.Get(handle);
        return b ? std::format("buf '{}'", b->desc.debugName) : "buf <invalid>";
    }
    std::string Describe(AccelStructHandle handle) {
        NullAccelStruct* as = mAccelStructs.Get(handle);
        if (!as) {
            return "as <invalid>";
        }
        return as->isTlas ? std::format("tlas '{}'", as->tlas.debugName) : std::format("blas '{}'", as->blas.debugName);
    }
    std::string Describe(PipelineHandle handle) {
        NullPipeline* p = mPipelines.Get(handle);
        return p ? std::format("pipeline '{}'", p->name) : "pipeline <invalid>";
    }
    bool IsValid(TextureHandle h) { return mTextures.Get(h) != nullptr; }
    bool IsValid(BufferHandle h) { return mBuffers.Get(h) != nullptr; }

private:
    static HeapCategory TextureCategory(const TextureDesc& desc) {
        return Any(desc.usage & (TextureUsage::RenderTarget | TextureUsage::DepthStencil))
                   ? HeapCategory::RenderTargetTexture
                   : HeapCategory::Texture;
    }

    static u64 StateKey(TextureHandle h) { return (1ull << 62) | (u64(h.index) << 32) | h.generation; }
    static u64 StateKey(BufferHandle h) { return (2ull << 62) | (u64(h.index) << 32) | h.generation; }

    void ValidateBarrierLocked(QueueType queue, const Barrier& barrier) {
        if (barrier.type == Barrier::Type::Aliasing || barrier.type == Barrier::Type::AccelStruct) {
            return;
        }
        const u64 key = barrier.type == Barrier::Type::Texture ? StateKey(barrier.texture) : StateKey(barrier.buffer);
        auto [it, inserted] = mTrackedState.try_emplace(key, Access::None);
        if (!barrier.discard && it->second != barrier.before) {
            const std::string name =
                barrier.type == Barrier::Type::Texture ? Describe(barrier.texture) : Describe(barrier.buffer);
            const std::string message = std::format("VALIDATION: barrier on {} expects {} but resource is in {}", name,
                                                    ToString(barrier.before), ToString(it->second));
            AIR_LOG(Error, "NullRHI", "{}", message);
            if (mCapture) {
                mCaptured.push_back(std::format("[{}] {}", QueueName(queue), message));
            }
        }
        it->second = barrier.after;
    }

    void ForgetState(u64 key) {
        std::lock_guard lock(mSubmitMutex);
        mTrackedState.erase(key);
    }

    void DeferDelete(std::function<void()> fn) {
        std::lock_guard lock(mDeferredMutex);
        mDeferred.push_back({mFrameIndex.load(), std::move(fn)});
    }

    void FlushDeferredDeletes(u64 completedFrame) {
        std::vector<std::function<void()>> ready;
        {
            std::lock_guard lock(mDeferredMutex);
            auto it = std::stable_partition(mDeferred.begin(), mDeferred.end(),
                                            [&](const auto& d) { return d.first > completedFrame; });
            for (auto r = it; r != mDeferred.end(); ++r) {
                ready.push_back(std::move(r->second));
            }
            mDeferred.erase(it, mDeferred.end());
        }
        for (auto& fn : ready) {
            fn();
        }
    }

    BindlessIndex LazyBindless(BindlessIndex& slot) {
        std::lock_guard lock(mBindlessMutex);
        if (slot == kInvalidBindless) {
            if (!mFreeBindless.empty()) {
                slot = mFreeBindless.back();
                mFreeBindless.pop_back();
            } else {
                slot = mNextBindless++;
            }
        }
        return slot;
    }
    void ReleaseBindless(BindlessIndex index) {
        if (index == kInvalidBindless) {
            return;
        }
        std::lock_guard lock(mBindlessMutex);
        mFreeBindless.push_back(index);
    }

    DeviceDesc mDesc;
    DeviceCapabilities mCaps;
    NullHeapProvider mHeapProvider;
    GpuMemoryAllocator mAllocator;

    HandlePool<BufferTag, NullBuffer> mBuffers;
    HandlePool<TextureTag, NullTexture> mTextures;
    HandlePool<HeapTag, NullHeap> mHeaps;
    HandlePool<AccelStructTag, NullAccelStruct> mAccelStructs;
    HandlePool<PipelineTag, NullPipeline> mPipelines;

    std::mutex mSubmitMutex; // guards fences, tracked state and capture
    u64 mFence[size_t(QueueType::Count)] = {};
    std::unordered_map<u64, Access> mTrackedState;
    bool mCapture = false;
    std::vector<std::string> mCaptured;

    std::mutex mDeferredMutex;
    std::vector<std::pair<u64, std::function<void()>>> mDeferred;
    std::atomic<u64> mFrameIndex{0};

    std::mutex mBindlessMutex;
    BindlessIndex mNextBindless = 0;
    std::vector<BindlessIndex> mFreeBindless;
};

// --- NullCommandList out-of-line members (need NullDevice) -----------------------------------

void NullCommandList::Barriers(std::span<const Barrier> barriers) {
    if (barriers.empty()) {
        return;
    }
    std::string text = "Barrier";
    for (const Barrier& b : barriers) {
        switch (b.type) {
        case Barrier::Type::Texture:
            AIR_CHECK(mDevice.IsValid(b.texture), "barrier on invalid texture");
            text += std::format(" [{} {} -> {}{}]", mDevice.Describe(b.texture), ToString(b.before), ToString(b.after),
                                b.discard ? " discard" : "");
            break;
        case Barrier::Type::Buffer:
            AIR_CHECK(mDevice.IsValid(b.buffer), "barrier on invalid buffer");
            text += std::format(" [{} {} -> {}]", mDevice.Describe(b.buffer), ToString(b.before), ToString(b.after));
            break;
        case Barrier::Type::AccelStruct:
            text +=
                std::format(" [{} {} -> {}]", mDevice.Describe(b.accelStruct), ToString(b.before), ToString(b.after));
            break;
        case Barrier::Type::Aliasing: text += " [aliasing]"; break;
        }
    }
    Record(std::move(text), std::vector<Barrier>(barriers.begin(), barriers.end()));
}

void NullCommandList::CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size) {
    Record(std::format("CopyBuffer {}+{} <- {}+{} ({} bytes)", mDevice.Describe(dst), dstOffset, mDevice.Describe(src),
                       srcOffset, size));
}
void NullCommandList::CopyBufferToTexture(TextureHandle dst, u32 mip, BufferHandle src, u64 srcOffset) {
    Record(std::format("CopyBufferToTexture {} mip {} <- {}+{}", mDevice.Describe(dst), mip, mDevice.Describe(src),
                       srcOffset));
}
void NullCommandList::CopyTexture(TextureHandle dst, TextureHandle src) {
    Record(std::format("CopyTexture {} <- {}", mDevice.Describe(dst), mDevice.Describe(src)));
}
void NullCommandList::ClearTexture(TextureHandle texture, const float c[4]) {
    Record(std::format("ClearTexture {} ({}, {}, {}, {})", mDevice.Describe(texture), c[0], c[1], c[2], c[3]));
}
void NullCommandList::SetPipeline(PipelineHandle pipeline) {
    mPipeline = pipeline;
    Record(std::format("SetPipeline {}", mDevice.Describe(pipeline)));
}
void NullCommandList::BuildBlas(AccelStructHandle blas, BufferHandle scratch) {
    Record(std::format("BuildBlas {} scratch {}", mDevice.Describe(blas), mDevice.Describe(scratch)));
}
void NullCommandList::BuildTlas(AccelStructHandle tlas, std::span<const TlasInstance> instances, BufferHandle scratch) {
    Record(std::format("BuildTlas {} instances {} scratch {}", mDevice.Describe(tlas), instances.size(),
                       mDevice.Describe(scratch)));
}

} // namespace

std::unique_ptr<Device> CreateNullDevice(const DeviceDesc& desc) {
    return std::make_unique<NullDevice>(desc);
}

} // namespace air::rhi
