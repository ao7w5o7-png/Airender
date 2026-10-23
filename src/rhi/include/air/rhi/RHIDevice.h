#pragma once

// RHI device and command list interfaces.
//
// Threading contract:
//   * Device: every function is thread-safe. Resource creation/destruction may happen on any thread.
//     Destroy*() is deferred until the GPU has finished every frame that could still reference it.
//   * CommandList: NOT thread-safe. One thread records one list at a time. Record different lists
//     on different threads in parallel and submit them in the order you want them executed.
//   * Submit(): thread-safe; submissions to one queue are serialised in call order.
//   * Queues run concurrently. Work on one queue that depends on another's waits for it on the GPU
//     with QueueWait(); the render graph inserts those waits for async compute passes.

#include "air/rhi/GpuMemoryAllocator.h"
#include "air/rhi/RHITypes.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace air::rhi
{

class CommandList
{
public:
    virtual ~CommandList() = default;

    virtual QueueType GetQueueType() const = 0;

    virtual void Begin() = 0;
    virtual void End() = 0;

    virtual void Barriers(std::span<const Barrier> barriers) = 0;

    virtual void CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size) = 0;
    // Rows in `src` start TextureCopyRowPitch(format, mipWidth) bytes apart; the slices of a 3D
    // texture follow each other, mipHeight rows each.
    virtual void CopyBufferToTexture(TextureHandle dst, u32 mip, BufferHandle src, u64 srcOffset) = 0;
    virtual void CopyTexture(TextureHandle dst, TextureHandle src) = 0;
    // Rows in `dst` start TextureCopyRowPitch(format, mipWidth) bytes apart.
    virtual void CopyTextureToBuffer(BufferHandle dst, u64 dstOffset, TextureHandle src, u32 mip) = 0;
    virtual void ClearTexture(TextureHandle texture, const float color[4]) = 0;

    virtual void SetPipeline(PipelineHandle pipeline) = 0;
    // Root constants (D3D12) / push constants (Vulkan). Shaders read bindless indices from here.
    virtual void PushConstants(const void* data, u32 size) = 0;
    virtual void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) = 0;
    virtual void DispatchRays(u32 width, u32 height, u32 depth) = 0;

    virtual void BuildBlas(AccelStructHandle blas, BufferHandle scratch) = 0;
    virtual void BuildTlas(AccelStructHandle tlas, std::span<const TlasInstance> instances, BufferHandle scratch) = 0;

    // Raster: binds `target` (in Access::RenderTarget) as the only render target with a full-size
    // viewport and scissor. clearColor == nullptr keeps the contents.
    virtual void BeginRenderPass(TextureHandle target, const float* clearColor) = 0;
    virtual void EndRenderPass() = 0;

    // Debug markers, visible in PIX / RenderDoc / Nsight.
    virtual void BeginEvent(std::string_view name) = 0;
    virtual void EndEvent() = 0;

    // The backend command list (ID3D12GraphicsCommandList4*, VkCommandBuffer; nullptr on Null) for
    // third-party SDKs such as ImGui or DLSS. Native code may change any pipeline state: the next
    // SetPipeline() re-binds everything this interface relies on.
    virtual void* GetNativeHandle() = 0;
};

struct SwapchainDesc
{
    void* window = nullptr; // HWND on Windows; ignored by the Null backend
    u32 width = 0;
    u32 height = 0;
    Format format = Format::BGRA8_UNORM;
    u32 bufferCount = 3;
    bool vsync = true;
};

// Presentation surface. Not thread-safe: use it from the thread that owns the frame loop.
class Swapchain
{
public:
    virtual ~Swapchain() = default;

    // This frame's back buffer (usage RenderTarget | CopyDst). It is in Access::Present between
    // frames, so transition it to RenderTarget before drawing and back to Present before Present().
    virtual TextureHandle GetCurrentTexture() = 0;
    virtual bool Present() = 0;
    // Waits for the GPU and recreates the back buffers. Handles from GetCurrentTexture() become invalid.
    virtual void Resize(u32 width, u32 height) = 0;
    virtual void SetVsync(bool vsync) = 0;
    virtual const SwapchainDesc& GetDesc() const = 0;
};

// Objects handed to third-party SDKs (ImGui, DLSS). nullptr when the backend has no such object.
enum class NativeObject : u8
{
    Device,                 // ID3D12Device5* / VkDevice
    Adapter,                // IDXGIAdapter1* / VkPhysicalDevice
    GraphicsQueue,          // ID3D12CommandQueue* / VkQueue
    ResourceDescriptorHeap, // shader visible ID3D12DescriptorHeap* holding every bindless index
};

// CPU / GPU descriptor handles (D3D12_CPU_DESCRIPTOR_HANDLE::ptr, D3D12_GPU_DESCRIPTOR_HANDLE::ptr).
struct DescriptorHandles
{
    u64 cpu = 0;
    u64 gpu = 0;
};

struct DeviceDesc
{
    Backend backend = Backend::Null;
    bool enableValidation = true;     // D3D12 debug layer / Vulkan validation layers
    bool enableGpuValidation = false; // D3D12 GPU-based validation (slow, catches bindless misuse)
    bool preferHighPerformanceAdapter = true;
    u32 framesInFlight = 2;
};

// What the OS says about video memory (DXGI QueryVideoMemoryInfo / VK_EXT_memory_budget).
struct MemoryBudget
{
    u64 budgetBytes = 0; // how much the process may use before the OS starts demoting
    u64 usageBytes = 0;  // what the process uses right now (all allocations, not just ours)
};

struct MemoryStats
{
    GpuMemoryAllocator::Stats perLocation[size_t(MemoryLocation::Count)];
    std::vector<GpuMemoryAllocator::PoolStats> pools; // per memory type, per heap block
    MemoryBudget local;                               // VRAM
    MemoryBudget nonLocal;                            // system memory visible to the GPU
    u32 liveBuffers = 0;
    u32 liveTextures = 0;
    u32 liveAccelStructs = 0;
    u32 livePipelines = 0;
    u64 accelStructBytes = 0;
    u32 bindlessDescriptors = 0; // allocated slots in the bindless heap
};

// One live GPU resource, for memory inspection.
struct ResourceInfo
{
    enum class Kind : u8
    {
        Buffer,
        Texture,
        AccelStruct,
        Heap
    };
    Kind kind = Kind::Buffer;
    std::string name;
    u64 sizeBytes = 0;
    MemoryLocation location = MemoryLocation::DeviceLocal;
    bool placedInHeap = false; // lives inside a CreateHeap() heap (render graph transient memory)
    // Where the memory is: the allocator heap (GpuMemoryAllocator::HeapUsage::id) and the byte offset
    // in it; resources placed in a CreateHeap() heap share that heap's id. 0 when not known.
    u64 heapId = 0;
    u64 heapOffset = 0;
};

// GPU duration of one BeginEvent/EndEvent scope.
struct GpuTimingScope
{
    std::string name;
    QueueType queue = QueueType::Graphics; // scopes on different queues may overlap in time
    u32 depth = 0;                         // nesting level within its queue
    double beginMs = 0.0;                  // relative to the frame's first timestamp
    double durationMs = 0.0;
};

struct GpuFrameTimings
{
    u64 frameIndex = ~0ull; // ~0: nothing measured yet
    double frameMs = 0.0;   // first timestamp to last timestamp
    // Time some queue was inside a scope: frameMs without the gaps where queues waited, for another
    // queue or for a swapchain buffer (at vsync, a submission that writes one waits on the GPU).
    double busyMs = 0.0;
    std::vector<GpuTimingScope> scopes;
};

// The time covered by at least one scope (the union of their intervals, any queue or depth).
double ComputeBusyMs(std::span<const GpuTimingScope> scopes);

class Device
{
public:
    virtual ~Device() = default;

    virtual Backend GetBackend() const = 0;
    virtual const DeviceCaps& GetCapabilities() const = 0;

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
    // Raw slots in the bindless heap for code that writes its own descriptors (ImGui textures).
    // FreeBindlessSlot() is deferred like resource destruction.
    virtual BindlessIndex AllocateBindlessSlot() = 0;
    virtual void FreeBindlessSlot(BindlessIndex index) = 0;
    virtual DescriptorHandles GetBindlessDescriptor(BindlessIndex index) = 0;

    // Ray tracing
    virtual AccelStructHandle CreateBlas(const BlasDesc& desc) = 0;
    virtual AccelStructHandle CreateTlas(const TlasDesc& desc) = 0;
    virtual void DestroyAccelStruct(AccelStructHandle as) = 0;
    virtual u64 GetAccelStructScratchSize(AccelStructHandle as) = 0;

    // Pipelines
    virtual PipelineHandle CreateComputePipeline(const ComputePipelineDesc& desc) = 0;
    virtual PipelineHandle CreateRayTracingPipeline(const RayTracingPipelineDesc& desc) = 0;
    virtual void DestroyPipeline(PipelineHandle pipeline) = 0;

    // Presentation
    virtual std::unique_ptr<Swapchain> CreateSwapchain(const SwapchainDesc& desc) = 0;

    // Interop with third-party SDKs (see NativeObject). The pointers are not AddRef'd.
    virtual void* GetNativeObject(NativeObject object) = 0;
    virtual void* GetNativeResource(TextureHandle texture) = 0; // ID3D12Resource* / VkImage
    virtual void* GetNativeResource(BufferHandle buffer) = 0;   // ID3D12Resource* / VkBuffer

    // Commands
    virtual std::unique_ptr<CommandList> CreateCommandList(QueueType queue) = 0;
    // Returns a fence value that completes when the GPU finished these lists.
    virtual u64 Submit(QueueType queue, std::span<CommandList* const> lists) = 0;
    // GPU-side wait, no CPU blocking: work submitted to `waiter` after this call starts once
    // `signaler` has completed `fenceValue` (a value Submit() returned for that queue).
    virtual void QueueWait(QueueType waiter, QueueType signaler, u64 fenceValue) = 0;
    virtual bool IsFenceComplete(QueueType queue, u64 fenceValue) = 0;
    virtual void WaitForFence(QueueType queue, u64 fenceValue) = 0;
    virtual void WaitIdle() = 0;

    // Frame pacing: EndFrame() advances the frame index and releases deferred deletions whose frame
    // the GPU has completed. Call from the thread that owns the frame loop.
    virtual void BeginFrame() = 0;
    virtual void EndFrame() = 0;
    virtual u64 GetFrameIndex() const = 0;
    // Frames the CPU may record ahead of the GPU. Per-frame upload rings need this many + 1 slots.
    virtual u32 GetFramesInFlight() const = 0;

    virtual MemoryStats GetMemoryStats() = 0;
    virtual std::vector<ResourceInfo> GetResourceList() = 0;

    // GPU profiling: when enabled, every BeginEvent/EndEvent pair also writes GPU timestamps.
    // GetGpuTimings() returns the latest frame whose GPU work has completed (a few frames behind
    // the CPU). The Null backend reports the scopes with zero durations.
    virtual void SetGpuProfiling(bool enabled) = 0;
    virtual GpuFrameTimings GetGpuTimings() = 0;

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
