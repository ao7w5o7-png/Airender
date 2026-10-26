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

namespace air::rhi
{
namespace
{

enum class HeapCategory : u32
{
    Buffer = 0,
    RenderTargetTexture = 1,
    Texture = 2
};

// Same encoding the D3D12 backend uses: heap tier 1 separates these categories.
u32 MemoryTypeKey(MemoryLocation location, HeapCategory category)
{
    return u32(location) * 4 + u32(category);
}

class NullHeapProvider final : public IGpuHeapProvider
{
public:
    std::optional<GpuHeap> CreateHeap(u32 memoryTypeKey, MemoryLocation location, u64 size) override
    {
        GpuHeap heap;
        heap.size = size;
        const bool hostVisible = location != MemoryLocation::DeviceLocal;
        if (hostVisible)
        {
            heap.mapped = new u8[size]();
            heap.native = heap.mapped;
        }
        else
        {
            heap.native = new u64(size); // placeholder identity for device-only memory
        }
        return heap;
    }
    void DestroyHeap(u32 memoryTypeKey, const GpuHeap& heap) override
    {
        if (heap.mapped)
        {
            delete[] heap.mapped;
        }
        else
        {
            delete static_cast<u64*>(heap.native);
        }
    }
};

struct NullBuffer
{
    BufferDesc desc;
    GpuAllocation allocation;
    bool placed = false;
    BindlessIndex srv = kInvalidBindless;
    BindlessIndex uav = kInvalidBindless;
    bool destroyed = false; // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
};

struct NullTexture
{
    TextureDesc desc;
    GpuAllocation allocation;
    bool placed = false;
    BindlessIndex srv = kInvalidBindless;
    BindlessIndex uav = kInvalidBindless;
    bool destroyed = false; // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
};

struct NullHeap
{
    u64 size = 0;
    MemoryLocation location = MemoryLocation::DeviceLocal;
    GpuAllocation allocation;
    std::string name;
    bool destroyed = false; // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
};

struct NullAccelStruct
{
    bool isTlas = false;
    bool destroyed = false; // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
    BlasDesc blas;
    TlasDesc tlas;
    BindlessIndex srv = kInvalidBindless;
};

struct NullPipeline
{
    bool rayTracing = false;
    std::string name;
};

class NullDevice;

// A texture state a command relies on (e.g. a render pass target must be in RenderTarget).
struct StateRequirement
{
    TextureHandle texture;
    Access access = Access::None;
};

struct RecordedCommand
{
    std::string text;
    std::vector<Barrier> barriers;              // non-empty for barrier batches, validated at submit
    std::vector<StateRequirement> requirements; // validated at submit
    enum class Event : u8
    {
        None,
        Begin,
        End
    } event = Event::None;
    std::string eventName;
};

class NullCommandList final : public CommandList
{
public:
    NullCommandList(NullDevice& device, QueueType queue) : mDevice(device), mQueue(queue) {}

    QueueType GetQueueType() const override { return mQueue; }
    void Begin() override
    {
        AIR_CHECK(!mRecording, "Begin() on a command list that is already recording");
        mCommands.clear();
        mRecording = true;
        mEventDepth = 0;
    }
    void End() override
    {
        AIR_CHECK(mRecording, "End() without Begin()");
        AIR_CHECK(mEventDepth == 0, "{} unbalanced BeginEvent()", mEventDepth);
        AIR_CHECK(!mRenderPassTarget.IsValid(), "End() inside a render pass");
        mRecording = false;
    }

    void Barriers(std::span<const Barrier> barriers) override;
    void CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size) override;
    void CopyBufferToTexture(TextureHandle dst, u32 mip, u32 layer, BufferHandle src, u64 srcOffset) override;
    void CopyTexture(TextureHandle dst, TextureHandle src) override;
    void CopyTextureToBuffer(BufferHandle dst, u64 dstOffset, TextureHandle src, u32 mip) override;
    void ClearTexture(TextureHandle texture, const float color[4]) override;
    void SetPipeline(PipelineHandle pipeline) override;
    void PushConstants(const void* data, u32 size) override
    {
        AIR_CHECK(size <= 128, "push constants limited to 128 bytes, got {}", size);
        Record(std::format("PushConstants {} bytes", size));
    }
    void Dispatch(u32 x, u32 y, u32 z) override
    {
        AIR_CHECK(mPipeline.IsValid(), "Dispatch without pipeline");
        Record(std::format("Dispatch {}x{}x{}", x, y, z));
    }
    void DispatchRays(u32 w, u32 h, u32 d) override
    {
        AIR_CHECK(mPipeline.IsValid(), "DispatchRays without pipeline");
        Record(std::format("DispatchRays {}x{}x{}", w, h, d));
    }
    void BuildBlas(AccelStructHandle blas, BufferHandle scratch) override;
    void BuildTlas(AccelStructHandle tlas, std::span<const TlasInstance> instances, BufferHandle scratch) override;
    void ConvertMatrices(std::span<const MatrixConversion> conversions) override;
    void BeginRenderPass(TextureHandle target, const float* clearColor) override;
    void EndRenderPass() override
    {
        AIR_CHECK(mRenderPassTarget.IsValid(), "EndRenderPass without BeginRenderPass");
        mRenderPassTarget = {};
        Record("EndRenderPass");
    }
    void* GetNativeHandle() override { return nullptr; }
    void BeginEvent(std::string_view name) override
    {
        ++mEventDepth;
        Record(std::format("BeginEvent '{}'", name));
        mCommands.back().event = RecordedCommand::Event::Begin;
        mCommands.back().eventName = std::string(name);
    }
    void EndEvent() override
    {
        AIR_CHECK(mEventDepth > 0, "EndEvent without BeginEvent");
        --mEventDepth;
        Record("EndEvent");
        mCommands.back().event = RecordedCommand::Event::End;
    }

    std::vector<RecordedCommand>& Commands() { return mCommands; }

private:
    void Record(std::string text, std::vector<Barrier> barriers = {}, std::vector<StateRequirement> requirements = {})
    {
        AIR_CHECK(mRecording, "recording into a command list outside Begin()/End()");
        mCommands.push_back({std::move(text), std::move(barriers), std::move(requirements)});
    }

    NullDevice& mDevice;
    QueueType mQueue;
    bool mRecording = false;
    u32 mEventDepth = 0;
    PipelineHandle mPipeline;
    TextureHandle mRenderPassTarget;
    std::vector<RecordedCommand> mCommands;
};

class NullSwapchain final : public Swapchain
{
public:
    NullSwapchain(NullDevice& device, const SwapchainDesc& desc);
    ~NullSwapchain() override { DestroyBuffers(); }

    TextureHandle GetCurrentTexture() override { return mBuffers[mCurrent]; }
    bool Present() override;
    void Resize(u32 width, u32 height) override;
    void SetVsync(bool vsync) override { mDesc.vsync = vsync; }
    const SwapchainDesc& GetDesc() const override { return mDesc; }

private:
    void CreateBuffers();
    void DestroyBuffers();

    NullDevice& mDevice;
    SwapchainDesc mDesc;
    std::vector<TextureHandle> mBuffers;
    u32 mCurrent = 0;
};

const char* QueueName(QueueType queue)
{
    switch (queue)
    {
    case QueueType::Graphics: return "Graphics";
    case QueueType::Compute: return "Compute";
    case QueueType::Copy: return "Copy";
    case QueueType::Count: break;
    }
    return "?";
}

class NullDevice final : public Device
{
public:
    explicit NullDevice(const DeviceDesc& desc) : mDesc(desc), mAllocator(mHeapProvider)
    {
        mCaps.rayTracing = true;    // pretend, so ray tracing code paths can be exercised headless
        mCaps.linearAlgebra = true; // likewise for the matrix conversions of neural network passes
        mCaps.bindless = true;
        mCaps.rebar = true;
        mCaps.dedicatedVideoMemory = 8ull << 30;
        mCaps.adapterName = "Null Adapter";
    }

    ~NullDevice() override { WaitIdle(); }

    Backend GetBackend() const override { return Backend::Null; }
    const DeviceCaps& GetCapabilities() const override { return mCaps; }

    // --- resources -------------------------------------------------------------------------
    BufferHandle CreateBuffer(const BufferDesc& desc) override
    {
        AIR_CHECK(desc.size > 0, "buffer '{}' has zero size", desc.debugName);
        NullBuffer buffer;
        buffer.desc = desc;
        const AllocationInfo info = GetBufferAllocationInfo(desc);
        auto allocation = mAllocator.Allocate(
            {MemoryTypeKey(desc.memory, HeapCategory::Buffer), desc.memory, info.size, info.alignment});
        if (!allocation)
        {
            return {};
        }
        buffer.allocation = *allocation;
        return mBuffers.Allocate(std::move(buffer));
    }

    TextureHandle CreateTexture(const TextureDesc& desc) override
    {
        NullTexture texture;
        texture.desc = desc;
        const AllocationInfo info = GetTextureAllocationInfo(desc);
        auto allocation = mAllocator.Allocate({MemoryTypeKey(MemoryLocation::DeviceLocal, TextureCategory(desc)),
                                               MemoryLocation::DeviceLocal, info.size, info.alignment});
        if (!allocation)
        {
            return {};
        }
        texture.allocation = *allocation;
        return mTextures.Allocate(std::move(texture));
    }

    void DestroyBuffer(BufferHandle handle) override
    {
        mBuffers.Modify(handle, [](NullBuffer& r) { r.destroyed = true; });
        DeferDelete(
            [this, handle]
            {
                NullBuffer* buffer = mBuffers.Get(handle);
                AIR_CHECK(buffer, "DestroyBuffer on stale handle");
                if (!buffer->placed)
                {
                    mAllocator.Free(buffer->allocation);
                }
                ReleaseBindless(buffer->srv);
                ReleaseBindless(buffer->uav);
                ForgetState(StateKey(handle));
                mBuffers.Free(handle);
            });
    }

    void DestroyTexture(TextureHandle handle) override
    {
        mTextures.Modify(handle, [](NullTexture& r) { r.destroyed = true; });
        DeferDelete(
            [this, handle]
            {
                NullTexture* texture = mTextures.Get(handle);
                AIR_CHECK(texture, "DestroyTexture on stale handle");
                if (!texture->placed)
                {
                    mAllocator.Free(texture->allocation);
                }
                ReleaseBindless(texture->srv);
                ReleaseBindless(texture->uav);
                ForgetState(StateKey(handle));
                mTextures.Free(handle);
            });
    }

    const BufferDesc* GetBufferDesc(BufferHandle handle) override
    {
        NullBuffer* buffer = mBuffers.Get(handle);
        return buffer ? &buffer->desc : nullptr;
    }
    const TextureDesc* GetTextureDesc(TextureHandle handle) override
    {
        NullTexture* texture = mTextures.Get(handle);
        return texture ? &texture->desc : nullptr;
    }
    void* GetMappedPointer(BufferHandle handle) override
    {
        NullBuffer* buffer = mBuffers.Get(handle);
        return buffer ? buffer->allocation.mapped : nullptr;
    }

    AllocationInfo GetTextureAllocationInfo(const TextureDesc& desc) override
    {
        u64 size = 0;
        u32 w = desc.width, h = desc.height;
        const u32 layers =
            desc.dimension == TextureDimension::TexCube ? 6 * desc.depthOrArraySize : desc.depthOrArraySize;
        for (u32 mip = 0; mip < desc.mipLevels; ++mip)
        {
            size += u64(w) * h * FormatBytesPerPixel(desc.format);
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
        }
        size *= u64(layers) * desc.sampleCount;
        return {AlignUp(std::max<u64>(size, 1), 64 * 1024), 64 * 1024}; // D3D12 default placement alignment
    }
    AllocationInfo GetBufferAllocationInfo(const BufferDesc& desc) override { return {AlignUp(desc.size, 256), 256}; }

    HeapHandle CreateHeap(u64 size, MemoryLocation location, std::string_view debugName) override
    {
        NullHeap heap;
        heap.size = size;
        heap.location = location;
        heap.name = std::string(debugName);
        // Transient heaps hold every resource category (D3D12 heap tier 2 / Vulkan).
        auto allocation = mAllocator.Allocate(
            {MemoryTypeKey(location, HeapCategory::Texture) + 100, location, size, 64 * 1024, /*forceDedicated*/ true});
        if (!allocation)
        {
            return {};
        }
        heap.allocation = *allocation;
        return mHeaps.Allocate(std::move(heap));
    }

    void DestroyHeap(HeapHandle handle) override
    {
        mHeaps.Modify(handle, [](NullHeap& r) { r.destroyed = true; });
        DeferDelete(
            [this, handle]
            {
                NullHeap* heap = mHeaps.Get(handle);
                AIR_CHECK(heap, "DestroyHeap on stale handle");
                mAllocator.Free(heap->allocation);
                mHeaps.Free(handle);
            });
    }

    TextureHandle CreatePlacedTexture(const TextureDesc& desc, HeapHandle heapHandle, u64 offset) override
    {
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

    BufferHandle CreatePlacedBuffer(const BufferDesc& desc, HeapHandle heapHandle, u64 offset) override
    {
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
        if (buffer.allocation.mapped)
        {
            buffer.allocation.mapped += offset;
        }
        return mBuffers.Allocate(std::move(buffer));
    }

    BindlessIndex GetBindlessSRV(TextureHandle handle) override
    {
        NullTexture* t = mTextures.Get(handle);
        AIR_CHECK(t, "GetBindlessSRV: invalid texture");
        return LazyBindless(t->srv);
    }
    BindlessIndex GetBindlessUAV(TextureHandle handle, u32 mip) override
    {
        NullTexture* t = mTextures.Get(handle);
        AIR_CHECK(t, "GetBindlessUAV: invalid texture");
        AIR_CHECK(Any(t->desc.usage & TextureUsage::UnorderedAccess), "texture '{}' lacks UnorderedAccess usage",
                  t->desc.debugName);
        return LazyBindless(t->uav);
    }
    BindlessIndex GetBindlessSRV(BufferHandle handle) override
    {
        NullBuffer* b = mBuffers.Get(handle);
        AIR_CHECK(b, "GetBindlessSRV: invalid buffer");
        return LazyBindless(b->srv);
    }
    BindlessIndex GetBindlessUAV(BufferHandle handle) override
    {
        NullBuffer* b = mBuffers.Get(handle);
        AIR_CHECK(b, "GetBindlessUAV: invalid buffer");
        return LazyBindless(b->uav);
    }
    BindlessIndex GetBindlessSRV(AccelStructHandle handle) override
    {
        NullAccelStruct* as = mAccelStructs.Get(handle);
        AIR_CHECK(as && as->isTlas, "GetBindlessSRV: invalid TLAS");
        return LazyBindless(as->srv);
    }
    BindlessIndex AllocateBindlessSlot() override
    {
        BindlessIndex slot = kInvalidBindless;
        return LazyBindless(slot);
    }
    void FreeBindlessSlot(BindlessIndex index) override
    {
        DeferDelete([this, index] { ReleaseBindless(index); });
    }
    DescriptorHandles GetBindlessDescriptor(BindlessIndex index) override
    {
        // Synthetic addresses: unique per slot, never dereferenced.
        constexpr u64 kDescriptorSize = 32;
        return {0x10000000ull + index * kDescriptorSize, 0x20000000ull + index * kDescriptorSize};
    }

    AccelStructHandle CreateBlas(const BlasDesc& desc) override
    {
        NullAccelStruct as;
        as.blas = desc;
        return mAccelStructs.Allocate(std::move(as));
    }
    AccelStructHandle CreateTlas(const TlasDesc& desc) override
    {
        NullAccelStruct as;
        as.isTlas = true;
        as.tlas = desc;
        return mAccelStructs.Allocate(std::move(as));
    }
    void DestroyAccelStruct(AccelStructHandle handle) override
    {
        mAccelStructs.Modify(handle, [](NullAccelStruct& r) { r.destroyed = true; });
        DeferDelete(
            [this, handle]
            {
                if (NullAccelStruct* as = mAccelStructs.Get(handle))
                {
                    ReleaseBindless(as->srv);
                }
                mAccelStructs.Free(handle);
            });
    }
    u64 GetMatrixSize(u32 rows, u32 columns, MatrixType type, MatrixLayout layout, u32 stride) override
    {
        return PackedMatrixSize(rows, columns, type, layout, stride);
    }
    u64 GetAccelStructScratchSize(AccelStructHandle handle) override
    {
        NullAccelStruct* as = mAccelStructs.Get(handle);
        AIR_CHECK(as, "GetAccelStructScratchSize: invalid handle");
        if (as->isTlas)
        {
            return AlignUp(u64(as->tlas.maxInstances) * 64 + 256, 256);
        }
        u64 triangles = 0;
        for (const auto& g : as->blas.geometries)
        {
            triangles += g.indexCount / 3;
        }
        return AlignUp(triangles * 64 + 256, 256);
    }

    PipelineHandle CreateComputePipeline(const ComputePipelineDesc& desc) override
    {
        return mPipelines.Allocate(NullPipeline{false, desc.debugName});
    }
    PipelineHandle CreateRayTracingPipeline(const RayTracingPipelineDesc& desc) override
    {
        AIR_CHECK(!desc.rayGen.empty(), "ray tracing pipeline '{}' has no raygen shader", desc.debugName);
        return mPipelines.Allocate(NullPipeline{true, desc.debugName});
    }
    void DestroyPipeline(PipelineHandle handle) override
    {
        DeferDelete([this, handle] { mPipelines.Free(handle); });
    }

    // --- presentation and interop -----------------------------------------------------------
    std::unique_ptr<Swapchain> CreateSwapchain(const SwapchainDesc& desc) override
    {
        AIR_CHECK(desc.width > 0 && desc.height > 0 && desc.bufferCount >= 2, "invalid swapchain {}x{} x{}", desc.width,
                  desc.height, desc.bufferCount);
        return std::make_unique<NullSwapchain>(*this, desc);
    }
    void* GetNativeObject(NativeObject) override { return nullptr; }
    void* GetNativeResource(TextureHandle) override { return nullptr; }
    void* GetNativeResource(BufferHandle) override { return nullptr; }

    // --- submission ------------------------------------------------------------------------
    std::unique_ptr<CommandList> CreateCommandList(QueueType queue) override
    {
        return std::make_unique<NullCommandList>(*this, queue);
    }

    u64 Submit(QueueType queue, std::span<CommandList* const> lists) override
    {
        std::lock_guard lock(mSubmitMutex);
        for (CommandList* list : lists)
        {
            AIR_CHECK(list->GetQueueType() == queue, "command list submitted to the wrong queue");
            for (RecordedCommand& command : static_cast<NullCommandList*>(list)->Commands())
            {
                for (const Barrier& barrier : command.barriers)
                {
                    ValidateBarrierLocked(queue, barrier, mFence[size_t(queue)] + 1);
                }
                for (const StateRequirement& requirement : command.requirements)
                {
                    ValidateStateLocked(QueueName(queue), requirement.texture, requirement.access, command.text);
                }
                if (mProfiling && command.event == RecordedCommand::Event::Begin)
                {
                    mFrameScopes.push_back({command.eventName, queue, mScopeDepth++, 0.0, 0.0});
                }
                else if (mProfiling && command.event == RecordedCommand::Event::End && mScopeDepth > 0)
                {
                    --mScopeDepth;
                }
                if (mCapture)
                {
                    mCaptured.push_back(std::format("[{}] {}", QueueName(queue), command.text));
                }
            }
        }
        return ++mFence[size_t(queue)]; // the null GPU finishes instantly
    }

    void QueueWait(QueueType waiter, QueueType signaler, u64 value) override
    {
        std::lock_guard lock(mSubmitMutex);
        if (value > mFence[size_t(signaler)])
        {
            Report(waiter, std::format("VALIDATION: {} waits for {} fence {}, which was never submitted (the GPU "
                                       "would hang)",
                                       QueueName(waiter), QueueName(signaler), value));
        }
        u64& waited = mWaited[size_t(waiter)][size_t(signaler)];
        waited = std::max(waited, value);
        if (mCapture)
        {
            mCaptured.push_back(std::format("[{}] Wait {} fence {}", QueueName(waiter), QueueName(signaler), value));
        }
    }

    bool IsFenceComplete(QueueType queue, u64 value) override
    {
        std::lock_guard lock(mSubmitMutex);
        return value <= mFence[size_t(queue)];
    }
    void WaitForFence(QueueType, u64) override {}

    void WaitIdle() override { FlushDeferredDeletes(~0ull); }

    void BeginFrame() override {}
    void EndFrame() override
    {
        {
            std::lock_guard lock(mSubmitMutex);
            if (mProfiling)
            {
                mLastTimings.frameIndex = mFrameIndex.load();
                mLastTimings.frameMs = 0.0;
                mLastTimings.scopes = std::exchange(mFrameScopes, {});
            }
            mScopeDepth = 0;
        }
        const u64 frame = mFrameIndex.fetch_add(1) + 1;
        // A real GPU may still be working on the last framesInFlight frames.
        if (frame >= mDesc.framesInFlight)
        {
            FlushDeferredDeletes(frame - mDesc.framesInFlight);
        }
    }
    u64 GetFrameIndex() const override { return mFrameIndex.load(); }
    u32 GetFramesInFlight() const override { return mDesc.framesInFlight; }

    MemoryStats GetMemoryStats() override
    {
        MemoryStats stats;
        for (u32 i = 0; i < u32(MemoryLocation::Count); ++i)
        {
            stats.perLocation[i] = mAllocator.GetStats(MemoryLocation(i));
        }
        stats.pools = mAllocator.GetPoolStats();
        stats.local = {mCaps.dedicatedVideoMemory,
                       stats.perLocation[size_t(MemoryLocation::DeviceLocal)].reservedBytes};
        stats.liveBuffers = mBuffers.LiveCount();
        stats.liveTextures = mTextures.LiveCount();
        stats.liveAccelStructs = mAccelStructs.LiveCount();
        stats.livePipelines = mPipelines.LiveCount();
        {
            std::lock_guard lock(mBindlessMutex);
            stats.bindlessDescriptors = mNextBindless - u32(mFreeBindless.size());
        }
        return stats;
    }

    std::vector<ResourceInfo> GetResourceList() override
    {
        std::vector<ResourceInfo> list;
        auto heapId = [](const GpuAllocation& allocation) { return reinterpret_cast<u64>(allocation.nativeHeap); };
        mBuffers.ForEach(
            [&](BufferHandle, const NullBuffer& b)
            {
                if (b.destroyed)
                {
                    return;
                }
                list.push_back({ResourceInfo::Kind::Buffer, b.desc.debugName, b.allocation.size, b.desc.memory,
                                b.placed, heapId(b.allocation), b.allocation.offset});
            });
        mTextures.ForEach(
            [&](TextureHandle, const NullTexture& tex)
            {
                if (tex.destroyed)
                {
                    return;
                }
                list.push_back({ResourceInfo::Kind::Texture, tex.desc.debugName, tex.allocation.size,
                                MemoryLocation::DeviceLocal, tex.placed, heapId(tex.allocation),
                                tex.allocation.offset});
            });
        mHeaps.ForEach(
            [&](HeapHandle, const NullHeap& h)
            {
                if (h.destroyed)
                {
                    return;
                }
                list.push_back({ResourceInfo::Kind::Heap, h.name, h.size, h.location, false, heapId(h.allocation),
                                h.allocation.offset});
            });
        mAccelStructs.ForEach(
            [&](AccelStructHandle, const NullAccelStruct& as)
            {
                if (as.destroyed)
                {
                    return;
                }
                list.push_back({ResourceInfo::Kind::AccelStruct, as.isTlas ? as.tlas.debugName : as.blas.debugName, 0,
                                MemoryLocation::DeviceLocal, false});
            });
        return list;
    }

    void SetGpuProfiling(bool enabled) override
    {
        std::lock_guard lock(mSubmitMutex);
        mProfiling = enabled;
    }
    GpuFrameTimings GetGpuTimings() override
    {
        std::lock_guard lock(mSubmitMutex);
        return mLastTimings;
    }

    void SetCommandCapture(bool enabled) override
    {
        std::lock_guard lock(mSubmitMutex);
        mCapture = enabled;
    }
    std::vector<std::string> TakeCapturedCommands() override
    {
        std::lock_guard lock(mSubmitMutex);
        return std::exchange(mCaptured, {});
    }

    // --- helpers used by the command list ----------------------------------------------------
    std::string Describe(TextureHandle handle)
    {
        NullTexture* t = mTextures.Get(handle);
        return t ? std::format("tex '{}'", t->desc.debugName) : "tex <invalid>";
    }
    std::string Describe(BufferHandle handle)
    {
        NullBuffer* b = mBuffers.Get(handle);
        return b ? std::format("buf '{}'", b->desc.debugName) : "buf <invalid>";
    }
    std::string Describe(AccelStructHandle handle)
    {
        NullAccelStruct* as = mAccelStructs.Get(handle);
        if (!as)
        {
            return "as <invalid>";
        }
        return as->isTlas ? std::format("tlas '{}'", as->tlas.debugName) : std::format("blas '{}'", as->blas.debugName);
    }
    std::string Describe(PipelineHandle handle)
    {
        NullPipeline* p = mPipelines.Get(handle);
        return p ? std::format("pipeline '{}'", p->name) : "pipeline <invalid>";
    }
    bool IsValid(TextureHandle h) { return mTextures.Get(h) != nullptr; }
    bool IsValid(BufferHandle h) { return mBuffers.Get(h) != nullptr; }

    // Swapchain support: back buffers start in Present, and Present() checks they are back there.
    void SetTrackedState(TextureHandle texture, Access access)
    {
        std::lock_guard lock(mSubmitMutex);
        mTrackedState[StateKey(texture)] = access;
    }
    void ValidatePresent(TextureHandle texture)
    {
        std::lock_guard lock(mSubmitMutex);
        ValidateStateLocked("Present", texture, Access::Present, "Present");
        if (mCapture)
        {
            mCaptured.push_back(std::format("[Present] {}", Describe(texture)));
        }
    }

private:
    static HeapCategory TextureCategory(const TextureDesc& desc)
    {
        return Any(desc.usage & (TextureUsage::RenderTarget | TextureUsage::DepthStencil))
                   ? HeapCategory::RenderTargetTexture
                   : HeapCategory::Texture;
    }

    static u64 StateKey(TextureHandle h) { return (1ull << 62) | (u64(h.index) << 32) | h.generation; }
    static u64 StateKey(BufferHandle h) { return (2ull << 62) | (u64(h.index) << 32) | h.generation; }

    void Report(QueueType queue, const std::string& message)
    {
        AIR_LOG(Error, "NullRHI", "{}", message);
        if (mCapture)
        {
            mCaptured.push_back(std::format("[{}] {}", QueueName(queue), message));
        }
    }

    // `fence` is the value the submission containing the barrier will signal on `queue`.
    void ValidateBarrierLocked(QueueType queue, const Barrier& barrier, u64 fence)
    {
        if (barrier.type == Barrier::Type::Aliasing || barrier.type == Barrier::Type::AccelStruct)
        {
            return;
        }
        const u64 key = barrier.type == Barrier::Type::Texture ? StateKey(barrier.texture) : StateKey(barrier.buffer);
        const auto describe = [&]
        { return barrier.type == Barrier::Type::Texture ? Describe(barrier.texture) : Describe(barrier.buffer); };
        // States a queue cannot use (render targets, pixel-shader reads, ... on the compute queue).
        if (!IsAccessLegalOnQueue(barrier.after, queue) ||
            (!barrier.discard && !IsAccessLegalOnQueue(barrier.before, queue)))
        {
            Report(queue, std::format("VALIDATION: barrier on {} ({} -> {}) uses a state the {} queue does not support",
                                      describe(), ToString(barrier.before), ToString(barrier.after), QueueName(queue)));
        }
        // A resource last used on another queue must not be touched before waiting for that use.
        LastUse& last = mLastUse[key];
        if (last.fence != 0 && last.queue != queue && mWaited[size_t(queue)][size_t(last.queue)] < last.fence)
        {
            Report(queue, std::format("VALIDATION: {} was last used on the {} queue (fence {}) and is used on the {} "
                                      "queue without waiting for it",
                                      describe(), QueueName(last.queue), last.fence, QueueName(queue)));
        }
        last = {queue, fence};
        // A discarding barrier too: like D3D12 it transitions from `before`, which for a reused
        // placed resource is the state an earlier frame left it in.
        auto [it, inserted] = mTrackedState.try_emplace(key, Access::None);
        if (it->second != barrier.before)
        {
            const std::string name =
                barrier.type == Barrier::Type::Texture ? Describe(barrier.texture) : Describe(barrier.buffer);
            const std::string message = std::format("VALIDATION: barrier on {} expects {} but resource is in {}", name,
                                                    ToString(barrier.before), ToString(it->second));
            AIR_LOG(Error, "NullRHI", "{}", message);
            if (mCapture)
            {
                mCaptured.push_back(std::format("[{}] {}", QueueName(queue), message));
            }
        }
        it->second = barrier.after;
    }

    void ValidateStateLocked(std::string_view where, TextureHandle texture, Access expected, std::string_view what)
    {
        const auto it = mTrackedState.find(StateKey(texture));
        const Access actual = it == mTrackedState.end() ? Access::None : it->second;
        if (actual != expected)
        {
            const std::string message = std::format("VALIDATION: '{}' needs {} in {} but it is in {}", what,
                                                    Describe(texture), ToString(expected), ToString(actual));
            AIR_LOG(Error, "NullRHI", "{}", message);
            if (mCapture)
            {
                mCaptured.push_back(std::format("[{}] {}", where, message));
            }
        }
    }

    void ForgetState(u64 key)
    {
        std::lock_guard lock(mSubmitMutex);
        mTrackedState.erase(key);
        mLastUse.erase(key);
    }

    void DeferDelete(std::function<void()> fn)
    {
        std::lock_guard lock(mDeferredMutex);
        mDeferred.push_back({mFrameIndex.load(), std::move(fn)});
    }

    void FlushDeferredDeletes(u64 completedFrame)
    {
        std::vector<std::function<void()>> ready;
        {
            std::lock_guard lock(mDeferredMutex);
            auto it = std::stable_partition(mDeferred.begin(), mDeferred.end(),
                                            [&](const auto& d) { return d.first > completedFrame; });
            for (auto r = it; r != mDeferred.end(); ++r)
            {
                ready.push_back(std::move(r->second));
            }
            mDeferred.erase(it, mDeferred.end());
        }
        for (auto& fn : ready)
        {
            fn();
        }
    }

    BindlessIndex LazyBindless(BindlessIndex& slot)
    {
        std::lock_guard lock(mBindlessMutex);
        if (slot == kInvalidBindless)
        {
            if (!mFreeBindless.empty())
            {
                slot = mFreeBindless.back();
                mFreeBindless.pop_back();
            }
            else
            {
                slot = mNextBindless++;
            }
        }
        return slot;
    }
    void ReleaseBindless(BindlessIndex index)
    {
        if (index == kInvalidBindless)
        {
            return;
        }
        std::lock_guard lock(mBindlessMutex);
        mFreeBindless.push_back(index);
    }

    DeviceDesc mDesc;
    DeviceCaps mCaps;
    NullHeapProvider mHeapProvider;
    GpuMemoryAllocator mAllocator;

    HandlePool<BufferTag, NullBuffer> mBuffers;
    HandlePool<TextureTag, NullTexture> mTextures;
    HandlePool<HeapTag, NullHeap> mHeaps;
    HandlePool<AccelStructTag, NullAccelStruct> mAccelStructs;
    HandlePool<PipelineTag, NullPipeline> mPipelines;

    std::mutex mSubmitMutex; // guards fences, tracked state and capture
    u64 mFence[size_t(QueueType::Count)] = {};
    // mWaited[waiter][signaler]: the highest signaler fence the waiter queue waited for.
    u64 mWaited[size_t(QueueType::Count)][size_t(QueueType::Count)] = {};
    struct LastUse
    {
        QueueType queue = QueueType::Graphics;
        u64 fence = 0; // 0: never used
    };
    std::unordered_map<u64, LastUse> mLastUse; // barrier-visible uses, by StateKey
    std::unordered_map<u64, Access> mTrackedState;
    bool mCapture = false;
    std::vector<std::string> mCaptured;
    bool mProfiling = false;
    u32 mScopeDepth = 0;
    std::vector<GpuTimingScope> mFrameScopes;
    GpuFrameTimings mLastTimings;

    std::mutex mDeferredMutex;
    std::vector<std::pair<u64, std::function<void()>>> mDeferred;
    std::atomic<u64> mFrameIndex{0};

    mutable std::mutex mBindlessMutex;
    BindlessIndex mNextBindless = 0;
    std::vector<BindlessIndex> mFreeBindless;
};

// --- NullCommandList out-of-line members (need NullDevice) -----------------------------------

void NullCommandList::Barriers(std::span<const Barrier> barriers)
{
    if (barriers.empty())
    {
        return;
    }
    std::string text = "Barrier";
    for (const Barrier& b : barriers)
    {
        switch (b.type)
        {
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

void NullCommandList::CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size)
{
    Record(std::format("CopyBuffer {}+{} <- {}+{} ({} bytes)", mDevice.Describe(dst), dstOffset, mDevice.Describe(src),
                       srcOffset, size));
}
void NullCommandList::CopyBufferToTexture(TextureHandle dst, u32 mip, u32 layer, BufferHandle src, u64 srcOffset)
{
    Record(layer == 0 ? std::format("CopyBufferToTexture {} mip {} <- {}+{}", mDevice.Describe(dst), mip,
                                    mDevice.Describe(src), srcOffset)
                      : std::format("CopyBufferToTexture {} mip {} layer {} <- {}+{}", mDevice.Describe(dst), mip,
                                    layer, mDevice.Describe(src), srcOffset));
}
void NullCommandList::CopyTexture(TextureHandle dst, TextureHandle src)
{
    Record(std::format("CopyTexture {} <- {}", mDevice.Describe(dst), mDevice.Describe(src)));
}
void NullCommandList::CopyTextureToBuffer(BufferHandle dst, u64 dstOffset, TextureHandle src, u32 mip)
{
    Record(std::format("CopyTextureToBuffer {}+{} <- {} mip {}", mDevice.Describe(dst), dstOffset,
                       mDevice.Describe(src), mip));
}
void NullCommandList::BeginRenderPass(TextureHandle target, const float* clearColor)
{
    AIR_CHECK(!mRenderPassTarget.IsValid(), "BeginRenderPass inside a render pass");
    AIR_CHECK(mDevice.IsValid(target), "BeginRenderPass on an invalid texture");
    mRenderPassTarget = target;
    std::string text = std::format("BeginRenderPass {}", mDevice.Describe(target));
    if (clearColor)
    {
        text += std::format(" clear ({}, {}, {}, {})", clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    }
    Record(std::move(text), {}, {{target, Access::RenderTarget}});
}
void NullCommandList::ClearTexture(TextureHandle texture, const float c[4])
{
    Record(std::format("ClearTexture {} ({}, {}, {}, {})", mDevice.Describe(texture), c[0], c[1], c[2], c[3]));
}
void NullCommandList::SetPipeline(PipelineHandle pipeline)
{
    mPipeline = pipeline;
    Record(std::format("SetPipeline {}", mDevice.Describe(pipeline)));
}
void NullCommandList::BuildBlas(AccelStructHandle blas, BufferHandle scratch)
{
    Record(std::format("BuildBlas {} scratch {}", mDevice.Describe(blas), mDevice.Describe(scratch)));
}
void NullCommandList::BuildTlas(AccelStructHandle tlas, std::span<const TlasInstance> instances, BufferHandle scratch)
{
    Record(std::format("BuildTlas {} instances {} scratch {}", mDevice.Describe(tlas), instances.size(),
                       mDevice.Describe(scratch)));
}
void NullCommandList::ConvertMatrices(std::span<const MatrixConversion> conversions)
{
    for (const MatrixConversion& c : conversions)
    {
        AIR_CHECK(mDevice.IsValid(c.src) && mDevice.IsValid(c.dst), "ConvertMatrices: invalid buffer");
        AIR_CHECK(c.srcOffset % 128 == 0 && c.dstOffset % 128 == 0,
                  "ConvertMatrices: offsets must be 128-byte aligned");
        Record(std::format("ConvertMatrix {}x{} {}+{} {} <- {}+{} {}", c.rows, c.columns, mDevice.Describe(c.dst),
                           c.dstOffset, ToString(c.dstLayout), mDevice.Describe(c.src), c.srcOffset,
                           ToString(c.srcLayout)));
    }
}

// --- NullSwapchain ---------------------------------------------------------------------------

NullSwapchain::NullSwapchain(NullDevice& device, const SwapchainDesc& desc) : mDevice(device), mDesc(desc)
{
    CreateBuffers();
}

void NullSwapchain::CreateBuffers()
{
    for (u32 i = 0; i < mDesc.bufferCount; ++i)
    {
        TextureDesc desc;
        desc.format = mDesc.format;
        desc.width = mDesc.width;
        desc.height = mDesc.height;
        desc.usage = TextureUsage::RenderTarget | TextureUsage::CopyDst | TextureUsage::Present;
        desc.debugName = std::format("Swapchain[{}]", i);
        const TextureHandle texture = mDevice.CreateTexture(desc);
        mDevice.SetTrackedState(texture, Access::Present);
        mBuffers.push_back(texture);
    }
    mCurrent = 0;
}

void NullSwapchain::DestroyBuffers()
{
    for (TextureHandle texture : mBuffers)
    {
        mDevice.DestroyTexture(texture);
    }
    mBuffers.clear();
}

bool NullSwapchain::Present()
{
    mDevice.ValidatePresent(mBuffers[mCurrent]);
    mCurrent = (mCurrent + 1) % u32(mBuffers.size());
    return true;
}

void NullSwapchain::Resize(u32 width, u32 height)
{
    mDevice.WaitIdle();
    DestroyBuffers();
    mDesc.width = width;
    mDesc.height = height;
    CreateBuffers();
}

} // namespace

std::unique_ptr<Device> CreateNullDevice(const DeviceDesc& desc)
{
    return std::make_unique<NullDevice>(desc);
}

} // namespace air::rhi
