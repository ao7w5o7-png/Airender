#pragma once

// RHI device and command list interfaces.
//
// Threading contract (same split UE uses between FRHICommandList and the dynamic RHI):
//   * Device: every function is thread-safe. Resource creation/destruction may happen on any thread.
//     Destroy*() is deferred until the GPU has finished every frame that could still reference it.
//   * CommandList: NOT thread-safe. One thread records one list at a time. Record different lists
//     on different threads in parallel and submit them in the order you want them executed.
//   * Submit(): thread-safe; submissions to one queue are serialised in call order.

#include "air/rhi/GpuMemoryAllocator.h"
#include "air/rhi/RHITypes.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace air::rhi {

class CommandList {
public:
    virtual ~CommandList() = default;

    virtual QueueType GetQueueType() const = 0;

    virtual void Begin() = 0;
    virtual void End() = 0;

    virtual void Barriers(std::span<const Barrier> barriers) = 0;

    virtual void CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size) = 0;
    virtual void CopyBufferToTexture(TextureHandle dst, u32 mip, BufferHandle src, u64 srcOffset) = 0;
    virtual void CopyTexture(TextureHandle dst, TextureHandle src) = 0;
    virtual void ClearTexture(TextureHandle texture, const float color[4]) = 0;

    virtual void SetPipeline(PipelineHandle pipeline) = 0;
    // Root constants (D3D12) / push constants (Vulkan). Shaders read bindless indices from here.
    virtual void PushConstants(const void* data, u32 size) = 0;
    virtual void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) = 0;
    virtual void DispatchRays(u32 width, u32 height, u32 depth) = 0;

    virtual void BuildBlas(AccelStructHandle blas, BufferHandle scratch) = 0;
    virtual void BuildTlas(AccelStructHandle tlas, std::span<const TlasInstance> instances, BufferHandle scratch) = 0;

    // Debug markers, visible in PIX / RenderDoc / Nsight.
    virtual void BeginEvent(std::string_view name) = 0;
    virtual void EndEvent() = 0;
};

struct DeviceDesc {
    Backend backend = Backend::Null;
    bool enableValidation = true; // D3D12 debug layer + GBV / Vulkan validation layers
    bool preferHighPerformanceAdapter = true;
    u32 framesInFlight = 2;
};

struct MemoryStats {
    GpuMemoryAllocator::Stats perLocation[size_t(MemoryLocation::Count)];
    u32 liveBuffers = 0;
    u32 liveTextures = 0;
};

class Device {
public:
    virtual ~Device() = default;

    virtual Backend GetBackend() const = 0;
    virtual const DeviceCapabilities& GetCapabilities() const = 0;

    // Resources
    virtual BufferHandle CreateBuffer(const BufferDesc& desc) = 0;
    virtual TextureHandle CreateTexture(const TextureDesc& desc) = 0;
    virtual void DestroyBuffer(BufferHandle buffer) = 0;
    virtual void DestroyTexture(TextureHandle texture) = 0;
    virtual const BufferDesc* GetBufferDesc(BufferHandle buffer) = 0;
    virtual const TextureDesc* GetTextureDesc(TextureHandle texture) = 0;
    // CPU pointer for Upload / Readback / DeviceLocalHost buffers, nullptr otherwise.
    virtual void* GetMappedPointer(BufferHandle buffer) = 0;

    // Placed resources: the render graph aliases transient resources inside one heap.
    virtual AllocationInfo GetTextureAllocationInfo(const TextureDesc& desc) = 0;
    virtual AllocationInfo GetBufferAllocationInfo(const BufferDesc& desc) = 0;
    virtual HeapHandle CreateHeap(u64 size, MemoryLocation location, std::string_view debugName) = 0;
    virtual void DestroyHeap(HeapHandle heap) = 0;
    virtual TextureHandle CreatePlacedTexture(const TextureDesc& desc, HeapHandle heap, u64 offset) = 0;
    virtual BufferHandle CreatePlacedBuffer(const BufferDesc& desc, HeapHandle heap, u64 offset) = 0;

    // Bindless views. Indices stay valid until the resource is destroyed.
    virtual BindlessIndex GetBindlessSRV(TextureHandle texture) = 0;
    virtual BindlessIndex GetBindlessUAV(TextureHandle texture, u32 mip = 0) = 0;
    virtual BindlessIndex GetBindlessSRV(BufferHandle buffer) = 0;
    virtual BindlessIndex GetBindlessUAV(BufferHandle buffer) = 0;
    virtual BindlessIndex GetBindlessSRV(AccelStructHandle tlas) = 0;

    // Ray tracing
    virtual AccelStructHandle CreateBlas(const BlasDesc& desc) = 0;
    virtual AccelStructHandle CreateTlas(const TlasDesc& desc) = 0;
    virtual void DestroyAccelStruct(AccelStructHandle as) = 0;
    virtual u64 GetAccelStructScratchSize(AccelStructHandle as) = 0;

    // Pipelines
    virtual PipelineHandle CreateComputePipeline(const ComputePipelineDesc& desc) = 0;
    virtual PipelineHandle CreateRayTracingPipeline(const RayTracingPipelineDesc& desc) = 0;
    virtual void DestroyPipeline(PipelineHandle pipeline) = 0;

    // Commands
    virtual std::unique_ptr<CommandList> CreateCommandList(QueueType queue) = 0;
    // Returns a fence value that completes when the GPU finished these lists.
    virtual u64 Submit(QueueType queue, std::span<CommandList* const> lists) = 0;
    virtual bool IsFenceComplete(QueueType queue, u64 fenceValue) = 0;
    virtual void WaitForFence(QueueType queue, u64 fenceValue) = 0;
    virtual void WaitIdle() = 0;

    // Frame pacing: EndFrame() advances the frame index and releases deferred deletions whose frame
    // the GPU has completed. Call from the thread that owns the frame loop.
    virtual void BeginFrame() = 0;
    virtual void EndFrame() = 0;
    virtual u64 GetFrameIndex() const = 0;

    virtual MemoryStats GetMemoryStats() = 0;

    // Command capture for debugging and tests (and for an AI agent inspecting what a frame did).
    // When enabled, every submitted command becomes one human readable line, e.g.
    //   "[Graphics] Barrier tex 'SceneColor' UAVCompute -> SRVCompute"
    // Backends that validate state also emit "VALIDATION: ..." lines.
    virtual void SetCommandCapture(bool enabled) = 0;
    virtual std::vector<std::string> TakeCapturedCommands() = 0;
};

std::unique_ptr<Device> CreateDevice(const DeviceDesc& desc);
const char* ToString(Backend backend);

} // namespace air::rhi
