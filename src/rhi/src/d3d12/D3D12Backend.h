#pragma once

// Direct3D 12 backend internals, shared by the files in src/rhi/src/d3d12/. Not a public header.
//
//   D3D12Device.cpp        adapter / device / queues / memory / resources / descriptors / frames
//   D3D12CommandList.cpp   command recording and barrier translation
//   D3D12RayTracing.cpp    acceleration structures, ray tracing state objects, shader tables
//   D3D12Swapchain.cpp     DXGI flip-model swapchain
//
// Design:
//   * Requires an SM 6.6 + resource binding tier 3 GPU (bindless via ResourceDescriptorHeap) and
//     reports ray tracing when DXR 1.1 is available.
//   * One global root signature for compute and ray tracing: 32 root constants at b0 (push
//     constants) plus static samplers s0..s2; every resource is a bindless descriptor index.
//   * Memory comes from GpuMemoryAllocator: ID3D12Heap blocks per (heap type, tier-1 category),
//     sub-allocated with TLSF; every resource is a placed resource.
//   * Barriers use the legacy ResourceBarrier API. Host visible buffers keep their fixed heap state
//     (GENERIC_READ / COPY_DEST) and never get barriers.
//   * The debug layer reports errors through ID3D12InfoQueue1; they become "VALIDATION:" lines in
//     the captured command stream, exactly like the Null backend's own checks.

#include "../Backends.h"

#include "air/core/HandlePool.h"

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace air::rhi::d3d12
{

using Microsoft::WRL::ComPtr;

constexpr u32 kRootConstantCount = 32; // 128 bytes of push constants
constexpr u32 kBindlessCapacity = 1u << 18;
constexpr u32 kRtvCapacity = 1024;
constexpr u64 kPlacementAlignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
constexpr u32 kMaxTimestampsPerFrame = 8192;
constexpr u32 kTimestampBlock = 64; // timestamps a command list reserves at a time

// D3D12 methods that return structs have a different C++ ABI outside MSVC (MinGW): these wrappers
// call the right variant.
inline D3D12_CPU_DESCRIPTOR_HANDLE CpuHeapStart(ID3D12DescriptorHeap* heap)
{
#if defined(_MSC_VER) || !defined(_WIN32)
    return heap->GetCPUDescriptorHandleForHeapStart();
#else
    D3D12_CPU_DESCRIPTOR_HANDLE handle;
    heap->GetCPUDescriptorHandleForHeapStart(&handle);
    return handle;
#endif
}
inline D3D12_GPU_DESCRIPTOR_HANDLE GpuHeapStart(ID3D12DescriptorHeap* heap)
{
#if defined(_MSC_VER) || !defined(_WIN32)
    return heap->GetGPUDescriptorHandleForHeapStart();
#else
    D3D12_GPU_DESCRIPTOR_HANDLE handle;
    heap->GetGPUDescriptorHandleForHeapStart(&handle);
    return handle;
#endif
}
inline D3D12_RESOURCE_ALLOCATION_INFO ResourceAllocationInfo(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc)
{
#if defined(_MSC_VER) || !defined(_WIN32)
    return device->GetResourceAllocationInfo(0, 1, &desc);
#else
    D3D12_RESOURCE_ALLOCATION_INFO info;
    device->GetResourceAllocationInfo(&info, 0, 1, &desc);
    return info;
#endif
}

DXGI_FORMAT ToDxgi(Format format);
std::string ToUtf8(const wchar_t* text);
std::string HResultToString(HRESULT hr);

// Tier-1 heap categories, encoded in the allocator's memory type key as location * 4 + category.
enum class HeapCategory : u32
{
    Buffer = 0,
    RenderTargetTexture = 1,
    Texture = 2,
    All = 3
};

struct D3D12Buffer
{
    BufferDesc desc;
    ComPtr<ID3D12Resource> resource;
    GpuAllocation allocation;
    bool ownsAllocation = true;
    bool hostVisible = false; // fixed heap state, no barriers
    u8* mapped = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    BindlessIndex srv = kInvalidBindless;
    BindlessIndex uav = kInvalidBindless;
    bool destroyed = false; // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
};

struct D3D12Texture
{
    TextureDesc desc;
    ComPtr<ID3D12Resource> resource;
    GpuAllocation allocation;
    bool ownsAllocation = true;
    BindlessIndex srv = kInvalidBindless;
    std::vector<BindlessIndex> uavs; // per mip
    u32 rtv = ~0u;                   // index in the RTV heap
    bool destroyed = false;          // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
};

struct D3D12Heap
{
    GpuAllocation allocation;
    u64 size = 0;
    std::string name;
    bool destroyed = false; // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
};

struct D3D12AccelStruct
{
    bool isTlas = false;
    bool destroyed = false; // Destroy*() called, the release waits for the GPU (hidden from GetResourceList)
    BlasDesc blas;
    TlasDesc tlas;
    ComPtr<ID3D12Resource> resource;
    GpuAllocation allocation;
    D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    u64 scratchSize = 0;
    BindlessIndex srv = kInvalidBindless;
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometries; // BLAS
    // TLAS: instance descriptors are written into a ring of upload memory, one slot per build,
    // so a build never overwrites instances the GPU may still read.
    ComPtr<ID3D12Resource> instanceRing;
    u8* instanceMapped = nullptr;
    u32 ringSlots = 0;
    u32 nextSlot = 0; // advanced with std::atomic_ref
};

struct D3D12Pipeline
{
    bool rayTracing = false;
    std::string name;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12StateObject> stateObject;
    ComPtr<ID3D12Resource> shaderTable; // upload heap
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE rayGen{};
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE miss{};
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE hitGroups{};
};

// GPU timestamps of one frame. Command lists reserve blocks of query indices, resolve them into
// their queue's readback buffer at End(), and append their scopes at Submit() (so scopes are in
// GPU order).
struct TimestampScope
{
    std::string name;
    QueueType queue = QueueType::Graphics;
    u32 depth = 0;
    u32 begin = 0; // query indices in the queue's heap
    u32 end = 0;
};

// Each timed queue resolves into its own buffer: queues running at the same time must not write
// one resource, even at different offsets.
struct TimestampQueries
{
    ComPtr<ID3D12QueryHeap> heap;
    ComPtr<ID3D12Resource> readback;
    const u64* mapped = nullptr;
    std::atomic<u32> next{0};
};

struct TimestampFrame
{
    TimestampQueries queues[size_t(QueueType::Count)]; // the copy queue's stays empty: copies are not timed
    std::mutex mutex;                                  // guards scopes and frame
    std::vector<TimestampScope> scopes;
    u64 frame = ~0ull;
};

// A command allocator + command list pair. Recycled once the GPU has finished with it.
// Built against the preview Agility SDK (AIR_D3D12_PREVIEW): Shader Model 6.10 linear algebra.
#ifdef D3D12_PREVIEW_SDK_VERSION
#define AIR_D3D12_LINEAR_ALGEBRA 1
#else
#define AIR_D3D12_LINEAR_ALGEBRA 0
#endif

struct CommandContext
{
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList4> list;
#if AIR_D3D12_LINEAR_ALGEBRA
    ComPtr<ID3D12GraphicsCommandListPreview> preview; // ConvertLinearAlgebraMatrix, with linearAlgebra
#endif
    u64 fence = 0; // fence value on its queue that releases it
};

struct Queue
{
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    std::mutex mutex; // guards submission and lastSignaled
    u64 lastSignaled = 0;
    std::mutex poolMutex;                                  // guards contexts
    std::vector<std::unique_ptr<CommandContext>> contexts; // free contexts
};

class D3D12Device;

class D3D12CommandList final : public CommandList
{
public:
    D3D12CommandList(D3D12Device& device, QueueType queue) : mDevice(device), mQueue(queue) {}
    ~D3D12CommandList() override;

    QueueType GetQueueType() const override { return mQueue; }
    void Begin() override;
    void End() override;

    void Barriers(std::span<const Barrier> barriers) override;
    void CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size) override;
    void CopyBufferToTexture(TextureHandle dst, u32 mip, u32 layer, BufferHandle src, u64 srcOffset) override;
    void CopyTexture(TextureHandle dst, TextureHandle src) override;
    void CopyTextureToBuffer(BufferHandle dst, u64 dstOffset, TextureHandle src, u32 mip) override;
    void ClearTexture(TextureHandle texture, const float color[4]) override;

    void SetPipeline(PipelineHandle pipeline) override;
    void PushConstants(const void* data, u32 size) override;
    void Dispatch(u32 groupsX, u32 groupsY, u32 groupsZ) override;
    void DispatchRays(u32 width, u32 height, u32 depth) override;

    void BuildBlas(AccelStructHandle blas, BufferHandle scratch) override;
    void BuildTlas(AccelStructHandle tlas, std::span<const TlasInstance> instances, BufferHandle scratch) override;
    void ConvertMatrices(std::span<const MatrixConversion> conversions) override;

    void BeginRenderPass(TextureHandle target, const float* clearColor) override;
    void EndRenderPass() override;
    void BeginEvent(std::string_view name) override;
    void EndEvent() override;
    void* GetNativeHandle() override;

    // Used by D3D12Device::Submit: hands over the recorded list, its captured text and its
    // timestamp scopes.
    std::unique_ptr<CommandContext> DetachContext();
    std::vector<std::string>& Captured() { return mCaptured; }
    TimestampFrame* Timestamps() const { return mTimestamps; }
    std::vector<TimestampScope>& TimestampScopes() { return mClosedScopes; }

private:
    void BindRootState();
    void Capture(std::string text);
    u32 AllocateTimestamp();

    struct OpenScope
    {
        std::string name;
        u32 begin;
    };

    D3D12Device& mDevice;
    QueueType mQueue;
    std::unique_ptr<CommandContext> mContext;
    ID3D12GraphicsCommandList4* mList = nullptr;
    bool mRecording = false;
    bool mRootStateBound = false; // false after Begin() or after native code got the list
    bool mCapture = false;
    u32 mEventDepth = 0;
    const D3D12Pipeline* mRayTracingPipeline = nullptr;
    std::vector<std::string> mCaptured;

    // Profiling (graphics / compute queues only)
    TimestampFrame* mTimestamps = nullptr;
    struct TimestampBlock
    {
        u32 begin = 0;
        u32 used = 0; // next free index; the block ends at begin + kTimestampBlock
    };
    std::vector<TimestampBlock> mTimestampBlocks;
    std::vector<OpenScope> mOpenScopes;
    std::vector<TimestampScope> mClosedScopes;
};

class D3D12Device final : public Device
{
public:
    explicit D3D12Device(const DeviceDesc& desc);
    ~D3D12Device() override;

    bool Initialize();

    // --- Device interface ---------------------------------------------------------------------
    Backend GetBackend() const override { return Backend::D3D12; }
    const DeviceCaps& GetCapabilities() const override { return mCaps; }

    BufferHandle CreateBuffer(const BufferDesc& desc) override;
    TextureHandle CreateTexture(const TextureDesc& desc) override;
    void DestroyBuffer(BufferHandle buffer) override;
    void DestroyTexture(TextureHandle texture) override;
    const BufferDesc* GetBufferDesc(BufferHandle buffer) override;
    const TextureDesc* GetTextureDesc(TextureHandle texture) override;
    void* GetMappedPointer(BufferHandle buffer) override;

    AllocationInfo GetTextureAllocationInfo(const TextureDesc& desc) override;
    AllocationInfo GetBufferAllocationInfo(const BufferDesc& desc) override;
    HeapHandle CreateHeap(u64 size, MemoryLocation location, std::string_view debugName) override;
    void DestroyHeap(HeapHandle heap) override;
    TextureHandle CreatePlacedTexture(const TextureDesc& desc, HeapHandle heap, u64 offset) override;
    BufferHandle CreatePlacedBuffer(const BufferDesc& desc, HeapHandle heap, u64 offset) override;

    BindlessIndex GetBindlessSRV(TextureHandle texture) override;
    BindlessIndex GetBindlessUAV(TextureHandle texture, u32 mip) override;
    BindlessIndex GetBindlessSRV(BufferHandle buffer) override;
    BindlessIndex GetBindlessUAV(BufferHandle buffer) override;
    BindlessIndex GetBindlessSRV(AccelStructHandle tlas) override;
    BindlessIndex AllocateBindlessSlot() override;
    void FreeBindlessSlot(BindlessIndex index) override;
    DescriptorHandles GetBindlessDescriptor(BindlessIndex index) override;

    AccelStructHandle CreateBlas(const BlasDesc& desc) override;
    AccelStructHandle CreateTlas(const TlasDesc& desc) override;
    void DestroyAccelStruct(AccelStructHandle as) override;
    u64 GetAccelStructScratchSize(AccelStructHandle as) override;
    u64 GetMatrixSize(u32 rows, u32 columns, MatrixType type, MatrixLayout layout, u32 stride) override;

    PipelineHandle CreateComputePipeline(const ComputePipelineDesc& desc) override;
    PipelineHandle CreateRayTracingPipeline(const RayTracingPipelineDesc& desc) override;
    void DestroyPipeline(PipelineHandle pipeline) override;

    std::unique_ptr<Swapchain> CreateSwapchain(const SwapchainDesc& desc) override;
    void* GetNativeObject(NativeObject object) override;
    void* GetNativeResource(TextureHandle texture) override;
    void* GetNativeResource(BufferHandle buffer) override;

    std::unique_ptr<CommandList> CreateCommandList(QueueType queue) override;
    u64 Submit(QueueType queue, std::span<CommandList* const> lists) override;
    void QueueWait(QueueType waiter, QueueType signaler, u64 fenceValue) override;
    bool IsFenceComplete(QueueType queue, u64 fenceValue) override;
    void WaitForFence(QueueType queue, u64 fenceValue) override;
    void WaitIdle() override;

    void BeginFrame() override;
    void EndFrame() override;
    u64 GetFrameIndex() const override { return mFrameIndex.load(); }
    u32 GetFramesInFlight() const override { return mDesc.framesInFlight; }

    MemoryStats GetMemoryStats() override;
    std::vector<ResourceInfo> GetResourceList() override;
    void SetGpuProfiling(bool enabled) override;
    GpuFrameTimings GetGpuTimings() override;
    void SetCommandCapture(bool enabled) override;
    std::vector<std::string> TakeCapturedCommands() override;

    // --- backend internals ----------------------------------------------------------------------
    ID3D12Device5* Native() const { return mDevice.Get(); }
    IDXGIFactory6* Factory() const { return mFactory.Get(); }
    Queue& GetQueue(QueueType type) { return mQueues[size_t(type)]; }
    ID3D12RootSignature* RootSignature() const { return mRootSignature.Get(); }
    ID3D12DescriptorHeap* BindlessHeap() const { return mBindlessHeap.Get(); }
    bool AllowTearing() const { return mAllowTearing; }
    bool CaptureEnabled() const { return mCapture.load(std::memory_order_relaxed); }

    D3D12Buffer* Buffer(BufferHandle h) { return mBuffers.Get(h); }
    D3D12Texture* Texture(TextureHandle h) { return mTextures.Get(h); }
    D3D12AccelStruct* AccelStruct(AccelStructHandle h) { return mAccelStructs.Get(h); }
    D3D12Pipeline* Pipeline(PipelineHandle h) { return mPipelines.Get(h); }

    std::unique_ptr<CommandContext> AcquireContext(QueueType queue);
    void ReleaseContext(QueueType queue, std::unique_ptr<CommandContext> context);

    D3D12_CPU_DESCRIPTOR_HANDLE GetRtv(TextureHandle texture);
    // CPU-only copy of a bindless UAV descriptor (ClearUnorderedAccessView needs one).
    D3D12_CPU_DESCRIPTOR_HANDLE GetUavCpuMirror(BindlessIndex index) const;

    // Swapchain back buffers: wraps a resource the device does not own memory for.
    TextureHandle RegisterExternalTexture(ComPtr<ID3D12Resource> resource, const TextureDesc& desc);

    // Timestamp storage for the frame being recorded; nullptr when profiling is off.
    TimestampFrame* CurrentTimestampFrame();
    void AppendTimestampScopes(TimestampFrame* frame, std::vector<TimestampScope>& scopes, QueueType queue);

    void ReportValidation(std::string message);
    void LogDeviceRemoved(HRESULT hr, std::string_view where);

private:
    struct DeferredDelete
    {
        u64 frame;
        std::function<void()> fn;
    };
    struct FrameRecord
    {
        u64 frame;
        u64 fences[size_t(QueueType::Count)];
    };

    bool CreateDeviceAndQueues();
    bool CreateRootSignature();
    bool CreateDescriptorHeaps();
    void QueryCapabilities();
    void EnableInfoQueueCallback();

    D3D12_RESOURCE_DESC MakeBufferResourceDesc(const BufferDesc& desc) const;
    D3D12_RESOURCE_DESC MakeTextureResourceDesc(const TextureDesc& desc) const;
    D3D12_HEAP_TYPE HeapTypeFor(MemoryLocation location) const;
    BufferHandle FinishBuffer(const BufferDesc& desc, const GpuAllocation& allocation, ID3D12Heap* heap, u64 offset,
                              bool ownsAllocation);
    TextureHandle FinishTexture(const TextureDesc& desc, const GpuAllocation& allocation, ID3D12Heap* heap, u64 offset,
                                bool ownsAllocation);

    BindlessIndex AllocateDescriptor();
    void ReleaseDescriptor(BindlessIndex index);
    D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(BindlessIndex index) const;

    void DeferDelete(std::function<void()> fn);
    void FlushDeferredDeletes(u64 completedFrame);
    void RetireCompletedFrames(bool waitForOldest);
    void ReadTimestamps(u64 frame);

    DeviceDesc mDesc;
    DeviceCaps mCaps;
    ComPtr<IDXGIFactory6> mFactory;
    ComPtr<IDXGIAdapter1> mAdapter;
    ComPtr<ID3D12Device5> mDevice;
#if AIR_D3D12_LINEAR_ALGEBRA
    ComPtr<ID3D12DevicePreview> mDevicePreview; // with mCaps.linearAlgebra
#endif
    ComPtr<ID3D12InfoQueue1> mInfoQueue;
    DWORD mInfoQueueCookie = 0;
    bool mGpuUploadHeap = false;
    bool mAllowTearing = false;
    Queue mQueues[size_t(QueueType::Count)];

    ComPtr<ID3D12RootSignature> mRootSignature;

    class HeapProvider;
    std::unique_ptr<HeapProvider> mHeapProvider;
    std::unique_ptr<GpuMemoryAllocator> mAllocator;

    HandlePool<BufferTag, D3D12Buffer> mBuffers;
    HandlePool<TextureTag, D3D12Texture> mTextures;
    HandlePool<HeapTag, D3D12Heap> mHeaps;
    HandlePool<AccelStructTag, D3D12AccelStruct> mAccelStructs;
    HandlePool<PipelineTag, D3D12Pipeline> mPipelines;

    ComPtr<ID3D12DescriptorHeap> mBindlessHeap;  // shader visible
    ComPtr<ID3D12DescriptorHeap> mUavMirrorHeap; // CPU only, same indices as mBindlessHeap
    ComPtr<ID3D12DescriptorHeap> mRtvHeap;       // CPU only
    u32 mDescriptorSize = 0;
    u32 mRtvDescriptorSize = 0;
    std::mutex mViewMutex;       // guards lazy view creation (D3D12Buffer/Texture/AccelStruct view indices)
    std::mutex mDescriptorMutex; // guards the free lists below
    u32 mNextDescriptor = 0;
    std::vector<BindlessIndex> mFreeDescriptors;
    u32 mNextRtv = 0;
    std::vector<u32> mFreeRtvs;

    std::mutex mDeferredMutex; // guards mDeferred and mPendingFrames
    std::vector<DeferredDelete> mDeferred;
    std::deque<FrameRecord> mPendingFrames;
    std::atomic<u64> mFrameIndex{0};

    std::atomic<bool> mProfiling{false};
    std::vector<std::unique_ptr<TimestampFrame>> mTimestampFrames; // framesInFlight + 1
    double mTimestampFrequency = 1.0;                              // ticks per millisecond
    std::mutex mTimingsMutex;                                      // guards mLastTimings
    GpuFrameTimings mLastTimings;

    std::atomic<bool> mCapture{false};
    std::mutex mCaptureMutex; // guards mCaptured
    std::vector<std::string> mCaptured;
};

std::unique_ptr<Swapchain> CreateD3D12Swapchain(D3D12Device& device, const SwapchainDesc& desc);
D3D12_RESOURCE_STATES ToStates(Access access);

#if AIR_D3D12_LINEAR_ALGEBRA
inline D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT ToD3D12(MatrixLayout layout)
{
    switch (layout)
    {
    case MatrixLayout::RowMajor: return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
    case MatrixLayout::ColumnMajor: return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_COLUMN_MAJOR;
    case MatrixLayout::MulOptimal: return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_MUL_OPTIMAL;
    case MatrixLayout::OuterProductOptimal: return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_OUTER_PRODUCT_OPTIMAL;
    }
    return D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT_ROW_MAJOR;
}

inline D3D12_LINEAR_ALGEBRA_DATATYPE ToD3D12(MatrixType type)
{
    return type == MatrixType::Float16 ? D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16 : D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
}
#endif

#define AIR_D3D12_CHECK(expr, ...)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        const HRESULT air_hr_ = (expr);                                                                                \
        AIR_CHECK(SUCCEEDED(air_hr_), "{} failed: {}", #expr, ::air::rhi::d3d12::HResultToString(air_hr_));            \
    } while (false)

} // namespace air::rhi::d3d12
