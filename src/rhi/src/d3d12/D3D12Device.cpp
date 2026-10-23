// Direct3D 12 device: adapter selection, queues, memory, resources, descriptors and frame pacing.
// See D3D12Backend.h for the overall design.

#include "D3D12Backend.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include <algorithm>
#include <format>
#include <utility>

namespace air::rhi
{

std::unique_ptr<Device> CreateD3D12Device(const DeviceDesc& desc)
{
    auto device = std::make_unique<d3d12::D3D12Device>(desc);
    if (!device->Initialize())
    {
        return nullptr;
    }
    return device;
}

} // namespace air::rhi

namespace air::rhi::d3d12
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

DXGI_FORMAT ToDxgi(Format format)
{
    switch (format)
    {
    case Format::Unknown: return DXGI_FORMAT_UNKNOWN;
    case Format::R8_UNORM: return DXGI_FORMAT_R8_UNORM;
    case Format::RG8_UNORM: return DXGI_FORMAT_R8G8_UNORM;
    case Format::RGBA8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA8_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case Format::BGRA8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case Format::R16_FLOAT: return DXGI_FORMAT_R16_FLOAT;
    case Format::RG16_FLOAT: return DXGI_FORMAT_R16G16_FLOAT;
    case Format::RGBA16_FLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case Format::R32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case Format::RG32_FLOAT: return DXGI_FORMAT_R32G32_FLOAT;
    case Format::RGB32_FLOAT: return DXGI_FORMAT_R32G32B32_FLOAT;
    case Format::RGBA32_FLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case Format::R32_UINT: return DXGI_FORMAT_R32_UINT;
    case Format::RG32_UINT: return DXGI_FORMAT_R32G32_UINT;
    case Format::RGBA32_UINT: return DXGI_FORMAT_R32G32B32A32_UINT;
    case Format::R11G11B10_FLOAT: return DXGI_FORMAT_R11G11B10_FLOAT;
    case Format::D32_FLOAT: return DXGI_FORMAT_D32_FLOAT;
    case Format::D24_UNORM_S8_UINT: return DXGI_FORMAT_D24_UNORM_S8_UINT;
    }
    return DXGI_FORMAT_UNKNOWN;
}

namespace
{

// Resource format: depth textures read as SRVs and sRGB textures with UAVs need typeless storage.
DXGI_FORMAT ResourceFormat(const TextureDesc& desc)
{
    const bool srv = Any(desc.usage & TextureUsage::ShaderResource);
    switch (desc.format)
    {
    case Format::D32_FLOAT: return srv ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_D32_FLOAT;
    case Format::D24_UNORM_S8_UINT: return srv ? DXGI_FORMAT_R24G8_TYPELESS : DXGI_FORMAT_D24_UNORM_S8_UINT;
    case Format::RGBA8_SRGB:
        return Any(desc.usage & TextureUsage::UnorderedAccess) ? DXGI_FORMAT_R8G8B8A8_TYPELESS
                                                               : DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    default: return ToDxgi(desc.format);
    }
}

DXGI_FORMAT SrvFormat(Format format)
{
    switch (format)
    {
    case Format::D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case Format::D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    default: return ToDxgi(format);
    }
}

DXGI_FORMAT UavFormat(Format format)
{
    return format == Format::RGBA8_SRGB ? DXGI_FORMAT_R8G8B8A8_UNORM : ToDxgi(format);
}

u32 MemoryTypeKey(MemoryLocation location, HeapCategory category)
{
    return u32(location) * 4 + u32(category);
}

HeapCategory TextureCategory(const TextureDesc& desc)
{
    return Any(desc.usage & (TextureUsage::RenderTarget | TextureUsage::DepthStencil))
               ? HeapCategory::RenderTargetTexture
               : HeapCategory::Texture;
}

D3D12_RESOURCE_STATES InitialBufferState(MemoryLocation location, bool hostVisibleIsUpload)
{
    switch (location)
    {
    case MemoryLocation::Upload: return D3D12_RESOURCE_STATE_GENERIC_READ;
    case MemoryLocation::Readback: return D3D12_RESOURCE_STATE_COPY_DEST;
    case MemoryLocation::DeviceLocalHost:
        return hostVisibleIsUpload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON;
    default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

std::wstring ToWide(std::string_view text)
{
    if (text.empty())
    {
        return {};
    }
    const int length = MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), nullptr, 0);
    std::wstring wide(size_t(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), wide.data(), length);
    return wide;
}

void SetName(ID3D12Object* object, std::string_view name)
{
    if (object && !name.empty())
    {
        object->SetName(ToWide(name).c_str());
    }
}

void __stdcall InfoQueueCallback(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID,
                                 LPCSTR description, void* context)
{
    auto* device = static_cast<D3D12Device*>(context);
    if (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || severity == D3D12_MESSAGE_SEVERITY_ERROR)
    {
        device->ReportValidation(std::format("VALIDATION: D3D12: {}", description));
    }
    else if (severity == D3D12_MESSAGE_SEVERITY_WARNING)
    {
        AIR_LOG(Warning, "D3D12", "{}", description);
    }
}

} // namespace

std::string ToUtf8(const wchar_t* text)
{
    if (!text || !*text)
    {
        return {};
    }
    const int length = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string utf8(size_t(std::max(length - 1, 0)), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, utf8.data(), length, nullptr, nullptr);
    return utf8;
}

std::string HResultToString(HRESULT hr)
{
    return std::format("HRESULT 0x{:08X}", u32(hr));
}

D3D12_RESOURCE_STATES ToStates(Access access)
{
    if (access == Access::None)
    {
        return D3D12_RESOURCE_STATE_COMMON;
    }
    D3D12_RESOURCE_STATES states = D3D12_RESOURCE_STATE_COMMON;
    auto add = [&](Access bit, D3D12_RESOURCE_STATES state)
    {
        if (Any(access & bit))
        {
            states |= state;
        }
    };
    add(Access::CopySrc, D3D12_RESOURCE_STATE_COPY_SOURCE);
    add(Access::CopyDst, D3D12_RESOURCE_STATE_COPY_DEST);
    add(Access::VertexOrIndex, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | D3D12_RESOURCE_STATE_INDEX_BUFFER);
    add(Access::Constant, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    // SRVs are always readable from every stage: avoids pixel <-> non-pixel transitions when an
    // image produced by compute is displayed by a pixel shader (editor viewport).
    add(Access::SRVGraphics | Access::SRVCompute, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    add(Access::UAVGraphics | Access::UAVCompute, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    add(Access::RenderTarget, D3D12_RESOURCE_STATE_RENDER_TARGET);
    add(Access::DepthRead, D3D12_RESOURCE_STATE_DEPTH_READ);
    add(Access::DepthWrite, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    add(Access::IndirectArgs, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    add(Access::AccelStructRead | Access::AccelStructWrite, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    add(Access::Present, D3D12_RESOURCE_STATE_PRESENT);
    return states;
}

// ---------------------------------------------------------------------------
// Memory: ID3D12Heap blocks for GpuMemoryAllocator
// ---------------------------------------------------------------------------

class D3D12Device::HeapProvider final : public IGpuHeapProvider
{
public:
    explicit HeapProvider(D3D12Device& device) : mDevice(device) {}

    std::optional<GpuHeap> CreateHeap(u32 memoryTypeKey, MemoryLocation location, u64 size) override
    {
        D3D12_HEAP_DESC desc = {};
        desc.SizeInBytes = size;
        desc.Properties.Type = mDevice.HeapTypeFor(location);
        desc.Alignment = kPlacementAlignment;
        switch (HeapCategory(memoryTypeKey % 4))
        {
        case HeapCategory::Buffer: desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS; break;
        case HeapCategory::RenderTargetTexture: desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES; break;
        case HeapCategory::Texture: desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES; break;
        case HeapCategory::All: desc.Flags = D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES; break;
        }
        if (location == MemoryLocation::DeviceLocal)
        {
            desc.Flags |= D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
        }
        ID3D12Heap* heap = nullptr;
        const HRESULT hr = mDevice.Native()->CreateHeap(&desc, IID_PPV_ARGS(&heap));
        if (FAILED(hr))
        {
            AIR_LOG(Error, "D3D12", "CreateHeap({} bytes, {}) failed: {}", size, ToString(location),
                    HResultToString(hr));
            return std::nullopt;
        }
        if (location == MemoryLocation::DeviceLocal || location == MemoryLocation::DeviceLocalHost)
        {
            // Ask the OS to keep our heaps in video memory when other processes (graphics debuggers,
            // IDEs) have filled it: otherwise new heaps may be placed in system memory and every pass
            // runs at PCIe bandwidth. Their idle allocations get paged out instead.
            ComPtr<ID3D12Device1> device1;
            if (SUCCEEDED(mDevice.Native()->QueryInterface(IID_PPV_ARGS(&device1))))
            {
                ID3D12Pageable* pageable = heap;
                const D3D12_RESIDENCY_PRIORITY priority = D3D12_RESIDENCY_PRIORITY_HIGH;
                device1->SetResidencyPriority(1, &pageable, &priority);
            }
        }
        GpuHeap result;
        result.native = heap;
        result.size = size;
        return result;
    }

    void DestroyHeap(u32, const GpuHeap& heap) override { static_cast<ID3D12Heap*>(heap.native)->Release(); }

private:
    D3D12Device& mDevice;
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

D3D12Device::D3D12Device(const DeviceDesc& desc) : mDesc(desc) {}

D3D12Device::~D3D12Device()
{
    if (mDevice)
    {
        WaitIdle();
    }
    if (mInfoQueue && mInfoQueueCookie)
    {
        mInfoQueue->UnregisterMessageCallback(mInfoQueueCookie);
    }
    for (Queue& queue : mQueues)
    {
        queue.contexts.clear();
    }
    mTimestampFrames.clear();
    mAllocator.reset(); // releases the remaining heaps
}

bool D3D12Device::Initialize()
{
    if (!CreateDeviceAndQueues() || !CreateRootSignature() || !CreateDescriptorHeaps())
    {
        return false;
    }
    QueryCapabilities();
    mHeapProvider = std::make_unique<HeapProvider>(*this);
    mAllocator = std::make_unique<GpuMemoryAllocator>(*mHeapProvider);

    // Timestamp storage: one slot per frame that can be in flight, plus the one being recorded.
    u64 frequency = 0;
    if (SUCCEEDED(mQueues[size_t(QueueType::Graphics)].queue->GetTimestampFrequency(&frequency)) && frequency)
    {
        mTimestampFrequency = double(frequency) / 1000.0;
    }
    for (u32 i = 0; i < mDesc.framesInFlight + 1; ++i)
    {
        auto frame = std::make_unique<TimestampFrame>();
        D3D12_QUERY_HEAP_DESC queryDesc = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kMaxTimestampsPerFrame, 0};
        AIR_D3D12_CHECK(mDevice->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&frame->heap)));
        D3D12_HEAP_PROPERTIES readback = {D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC bufferDesc = MakeBufferResourceDesc({.size = kMaxTimestampsPerFrame * 8});
        AIR_D3D12_CHECK(mDevice->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&frame->readback)));
        void* mapped = nullptr;
        AIR_D3D12_CHECK(frame->readback->Map(0, nullptr, &mapped));
        frame->mapped = static_cast<const u64*>(mapped);
        SetName(frame->readback.Get(), "Timestamps");
        mTimestampFrames.push_back(std::move(frame));
    }

    AIR_LOG(Info, "D3D12", "device '{}' ({} MiB VRAM, SM {}.{}, ray tracing tier {}.{}, SER {}, GPU upload heap {})",
            mCaps.adapterName, mCaps.dedicatedVideoMemory >> 20, mCaps.shaderModel >> 4, mCaps.shaderModel & 0xF,
            mCaps.rayTracingTier / 10, mCaps.rayTracingTier % 10, mCaps.shaderExecutionReordering ? "yes" : "no",
            mGpuUploadHeap ? "yes" : "no");
    return true;
}

bool D3D12Device::CreateDeviceAndQueues()
{
    if (mDesc.enableValidation)
    {
        ComPtr<ID3D12Debug1> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
        {
            debug->EnableDebugLayer();
            debug->SetEnableGPUBasedValidation(mDesc.enableGpuValidation);
        }
        else
        {
            AIR_LOG(Warning, "D3D12", "debug layer unavailable (install the Graphics Tools optional feature)");
        }
        // Device removed extended data: breadcrumbs + page fault info in GetDeviceRemovedReason logs.
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred))))
        {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        }
    }

    const UINT factoryFlags = mDesc.enableValidation ? DXGI_CREATE_FACTORY_DEBUG : 0;
    if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&mFactory))))
    {
        AIR_LOG(Error, "D3D12", "CreateDXGIFactory2 failed (needs Windows 10 1803+)");
        return false;
    }
    BOOL tearing = FALSE;
    if (SUCCEEDED(mFactory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
    {
        mAllowTearing = tearing != FALSE;
    }

    const DXGI_GPU_PREFERENCE preference =
        mDesc.preferHighPerformanceAdapter ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_UNSPECIFIED;
    for (UINT i = 0;; ++i)
    {
        ComPtr<IDXGIAdapter1> adapter;
        if (mFactory->EnumAdapterByGpuPreference(i, preference, IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND)
        {
            break;
        }
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        {
            continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&mDevice))))
        {
            mAdapter = adapter;
            mCaps.adapterName = ToUtf8(desc.Description);
            mCaps.dedicatedVideoMemory = desc.DedicatedVideoMemory;
            break;
        }
    }
    if (!mDevice)
    {
        AIR_LOG(Error, "D3D12", "no hardware adapter supports Direct3D 12 feature level 12_0 with ID3D12Device5");
        return false;
    }
    SetName(mDevice.Get(), "Airender");

    // Bindless (SM 6.6 ResourceDescriptorHeap) is not optional: every shader uses it.
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    if (FAILED(mDevice->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))) ||
        shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
        FAILED(mDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3 ||
        options.ResourceHeapTier < D3D12_RESOURCE_HEAP_TIER_2)
    {
        AIR_LOG(Error, "D3D12",
                "'{}' lacks shader model 6.6, resource binding tier 3 or resource heap tier 2 (update the driver; "
                "the Agility SDK runtime must sit in .\\D3D12\\ next to the executable)",
                mCaps.adapterName);
        return false;
    }

    if (mDesc.enableValidation && SUCCEEDED(mDevice.As(&mInfoQueue)))
    {
        EnableInfoQueueCallback();
    }

    static constexpr D3D12_COMMAND_LIST_TYPE kTypes[] = {D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY};
    static constexpr const char* kNames[] = {"Graphics Queue", "Compute Queue", "Copy Queue"};
    for (size_t i = 0; i < size_t(QueueType::Count); ++i)
    {
        D3D12_COMMAND_QUEUE_DESC desc = {};
        desc.Type = kTypes[i];
        AIR_D3D12_CHECK(mDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&mQueues[i].queue)));
        AIR_D3D12_CHECK(mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mQueues[i].fence)));
        SetName(mQueues[i].queue.Get(), kNames[i]);
    }
    return true;
}

void D3D12Device::EnableInfoQueueCallback()
{
    // Clearing with a value that differs from the optimized clear value is legal and only slower.
    D3D12_MESSAGE_ID denied[] = {D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE};
    D3D12_INFO_QUEUE_FILTER filter = {};
    filter.DenyList.NumIDs = UINT(std::size(denied));
    filter.DenyList.pIDList = denied;
    mInfoQueue->PushStorageFilter(&filter);
    if (FAILED(mInfoQueue->RegisterMessageCallback(&InfoQueueCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, this,
                                                   &mInfoQueueCookie)))
    {
        AIR_LOG(Warning, "D3D12", "ID3D12InfoQueue1 callbacks unavailable; validation errors only reach the debugger");
    }
}

bool D3D12Device::CreateRootSignature()
{
    D3D12_ROOT_PARAMETER1 constants = {};
    constants.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    constants.Constants.ShaderRegister = 0;
    constants.Constants.RegisterSpace = 0;
    constants.Constants.Num32BitValues = kRootConstantCount;
    constants.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // s0 linear clamp, s1 point clamp, s2 linear wrap.
    D3D12_STATIC_SAMPLER_DESC samplers[3] = {};
    const D3D12_FILTER filters[3] = {D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_FILTER_MIN_MAG_MIP_POINT,
                                     D3D12_FILTER_MIN_MAG_MIP_LINEAR};
    for (u32 i = 0; i < 3; ++i)
    {
        const D3D12_TEXTURE_ADDRESS_MODE mode =
            i == 2 ? D3D12_TEXTURE_ADDRESS_MODE_WRAP : D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].Filter = filters[i];
        samplers[i].AddressU = samplers[i].AddressV = samplers[i].AddressW = mode;
        samplers[i].MaxLOD = D3D12_FLOAT32_MAX;
        samplers[i].ShaderRegister = i;
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = 1;
    desc.Desc_1_1.pParameters = &constants;
    desc.Desc_1_1.NumStaticSamplers = 3;
    desc.Desc_1_1.pStaticSamplers = samplers;
    desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;

    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;
    if (FAILED(D3D12SerializeVersionedRootSignature(&desc, &blob, &error)))
    {
        AIR_LOG(Error, "D3D12", "root signature: {}",
                error ? static_cast<const char*>(error->GetBufferPointer()) : "unknown error");
        return false;
    }
    AIR_D3D12_CHECK(mDevice->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&mRootSignature)));
    SetName(mRootSignature.Get(), "Bindless Root Signature");
    return true;
}

bool D3D12Device::CreateDescriptorHeaps()
{
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = kBindlessCapacity;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    AIR_D3D12_CHECK(mDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&mBindlessHeap)));
    SetName(mBindlessHeap.Get(), "Bindless Heap");

    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    AIR_D3D12_CHECK(mDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&mUavMirrorHeap)));

    D3D12_DESCRIPTOR_HEAP_DESC rtv = {};
    rtv.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv.NumDescriptors = kRtvCapacity;
    AIR_D3D12_CHECK(mDevice->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&mRtvHeap)));

    mDescriptorSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    mRtvDescriptorSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    return true;
}

void D3D12Device::QueryCapabilities()
{
    mCaps.bindless = true; // checked in CreateDeviceAndQueues

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    if (SUCCEEDED(mDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5))))
    {
        mCaps.rayTracing = options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
        mCaps.rayTracingTier = u32(options5.RaytracingTier);
    }
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {D3D_SHADER_MODEL_6_9}; // returns the highest supported
    if (SUCCEEDED(mDevice->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))))
    {
        mCaps.shaderModel = u32(shaderModel.HighestShaderModel);
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS22 options22 = {};
    const bool serReorders =
        SUCCEEDED(mDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS22, &options22, sizeof(options22))) &&
        options22.ShaderExecutionReorderingActuallyReorders;
    mCaps.shaderExecutionReordering = mCaps.shaderModel >= u32(D3D_SHADER_MODEL_6_9) &&
                                      mCaps.rayTracingTier >= u32(D3D12_RAYTRACING_TIER_1_2) && serReorders;
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = {};
    if (SUCCEEDED(mDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7))))
    {
        mCaps.meshShaders = options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 options16 = {};
    if (SUCCEEDED(mDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &options16, sizeof(options16))))
    {
        mGpuUploadHeap = options16.GPUUploadHeapSupported != FALSE;
    }
    mCaps.rebar = mGpuUploadHeap;
    if (!mCaps.rayTracing)
    {
        AIR_LOG(Warning, "D3D12", "'{}' has no DXR 1.1 support; the path tracer will refuse to run", mCaps.adapterName);
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

D3D12_HEAP_TYPE D3D12Device::HeapTypeFor(MemoryLocation location) const
{
    switch (location)
    {
    case MemoryLocation::Upload: return D3D12_HEAP_TYPE_UPLOAD;
    case MemoryLocation::Readback: return D3D12_HEAP_TYPE_READBACK;
    case MemoryLocation::DeviceLocalHost: return mGpuUploadHeap ? D3D12_HEAP_TYPE_GPU_UPLOAD : D3D12_HEAP_TYPE_UPLOAD;
    default: return D3D12_HEAP_TYPE_DEFAULT;
    }
}

D3D12_RESOURCE_DESC D3D12Device::MakeBufferResourceDesc(const BufferDesc& desc) const
{
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = AlignUp(std::max<u64>(desc.size, 4), 256);
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (Any(desc.usage & (BufferUsage::UnorderedAccess | BufferUsage::AccelStructStorage)))
    {
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    return rd;
}

D3D12_RESOURCE_DESC D3D12Device::MakeTextureResourceDesc(const TextureDesc& desc) const
{
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = desc.dimension == TextureDimension::Tex3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D
                                                             : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = desc.width;
    rd.Height = desc.height;
    rd.DepthOrArraySize =
        u16(desc.dimension == TextureDimension::TexCube ? 6 * desc.depthOrArraySize : desc.depthOrArraySize);
    rd.MipLevels = u16(desc.mipLevels);
    rd.Format = ResourceFormat(desc);
    rd.SampleDesc.Count = desc.sampleCount;
    if (Any(desc.usage & TextureUsage::UnorderedAccess))
    {
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if (Any(desc.usage & TextureUsage::RenderTarget))
    {
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if (Any(desc.usage & TextureUsage::DepthStencil))
    {
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if (!Any(desc.usage & TextureUsage::ShaderResource))
        {
            rd.Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        }
    }
    return rd;
}

AllocationInfo D3D12Device::GetBufferAllocationInfo(const BufferDesc& desc)
{
    return {AlignUp(MakeBufferResourceDesc(desc).Width, kPlacementAlignment), kPlacementAlignment};
}

AllocationInfo D3D12Device::GetTextureAllocationInfo(const TextureDesc& desc)
{
    const D3D12_RESOURCE_DESC rd = MakeTextureResourceDesc(desc);
    const D3D12_RESOURCE_ALLOCATION_INFO info = ResourceAllocationInfo(mDevice.Get(), rd);
    return {info.SizeInBytes, std::max<u64>(info.Alignment, kPlacementAlignment)};
}

BufferHandle D3D12Device::FinishBuffer(const BufferDesc& desc, const GpuAllocation& allocation, ID3D12Heap* heap,
                                       u64 offset, bool ownsAllocation)
{
    D3D12Buffer buffer;
    buffer.desc = desc;
    buffer.allocation = allocation;
    buffer.ownsAllocation = ownsAllocation;
    buffer.hostVisible = desc.memory != MemoryLocation::DeviceLocal;
    const D3D12_RESOURCE_DESC rd = MakeBufferResourceDesc(desc);
    const bool uploadLike = HeapTypeFor(desc.memory) == D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_STATES initial = InitialBufferState(desc.memory, uploadLike);
    if (Any(desc.usage & BufferUsage::AccelStructStorage))
    {
        initial = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    }
    const HRESULT hr =
        mDevice->CreatePlacedResource(heap, offset, &rd, initial, nullptr, IID_PPV_ARGS(&buffer.resource));
    if (FAILED(hr))
    {
        AIR_LOG(Error, "D3D12", "buffer '{}' ({} bytes): CreatePlacedResource failed: {}", desc.debugName, desc.size,
                HResultToString(hr));
        if (ownsAllocation)
        {
            mAllocator->Free(allocation);
        }
        return {};
    }
    SetName(buffer.resource.Get(), desc.debugName);
    buffer.address = buffer.resource->GetGPUVirtualAddress();
    if (buffer.hostVisible)
    {
        void* mapped = nullptr;
        const D3D12_RANGE noRead = {0, 0};
        AIR_D3D12_CHECK(buffer.resource->Map(0, desc.memory == MemoryLocation::Readback ? nullptr : &noRead, &mapped));
        buffer.mapped = static_cast<u8*>(mapped);
    }
    return mBuffers.Allocate(std::move(buffer));
}

BufferHandle D3D12Device::CreateBuffer(const BufferDesc& desc)
{
    AIR_CHECK(desc.size > 0, "buffer '{}' has zero size", desc.debugName);
    const AllocationInfo info = GetBufferAllocationInfo(desc);
    auto allocation = mAllocator->Allocate(
        {MemoryTypeKey(desc.memory, HeapCategory::Buffer), desc.memory, info.size, info.alignment});
    if (!allocation)
    {
        AIR_LOG(Error, "D3D12", "out of {} memory for buffer '{}' ({} bytes)", ToString(desc.memory), desc.debugName,
                desc.size);
        return {};
    }
    return FinishBuffer(desc, *allocation, static_cast<ID3D12Heap*>(allocation->nativeHeap), allocation->offset, true);
}

TextureHandle D3D12Device::FinishTexture(const TextureDesc& desc, const GpuAllocation& allocation, ID3D12Heap* heap,
                                         u64 offset, bool ownsAllocation)
{
    D3D12Texture texture;
    texture.desc = desc;
    texture.allocation = allocation;
    texture.ownsAllocation = ownsAllocation;
    texture.uavs.assign(desc.mipLevels, kInvalidBindless);
    const D3D12_RESOURCE_DESC rd = MakeTextureResourceDesc(desc);
    D3D12_CLEAR_VALUE clear = {};
    const D3D12_CLEAR_VALUE* clearPtr = nullptr;
    if (Any(desc.usage & TextureUsage::RenderTarget))
    {
        clear.Format = ToDxgi(desc.format);
        std::copy(std::begin(desc.clearColor), std::end(desc.clearColor), clear.Color);
        clearPtr = &clear;
    }
    else if (Any(desc.usage & TextureUsage::DepthStencil))
    {
        clear.Format = ToDxgi(desc.format);
        clear.DepthStencil.Depth = desc.clearColor[0];
        clearPtr = &clear;
    }
    const HRESULT hr = mDevice->CreatePlacedResource(heap, offset, &rd, D3D12_RESOURCE_STATE_COMMON, clearPtr,
                                                     IID_PPV_ARGS(&texture.resource));
    if (FAILED(hr))
    {
        AIR_LOG(Error, "D3D12", "texture '{}' ({}x{} {}): CreatePlacedResource failed: {}", desc.debugName, desc.width,
                desc.height, ToString(desc.format), HResultToString(hr));
        if (ownsAllocation)
        {
            mAllocator->Free(allocation);
        }
        return {};
    }
    SetName(texture.resource.Get(), desc.debugName);
    return mTextures.Allocate(std::move(texture));
}

TextureHandle D3D12Device::CreateTexture(const TextureDesc& desc)
{
    const AllocationInfo info = GetTextureAllocationInfo(desc);
    auto allocation = mAllocator->Allocate({MemoryTypeKey(MemoryLocation::DeviceLocal, TextureCategory(desc)),
                                            MemoryLocation::DeviceLocal, info.size, info.alignment});
    if (!allocation)
    {
        AIR_LOG(Error, "D3D12", "out of video memory for texture '{}'", desc.debugName);
        return {};
    }
    return FinishTexture(desc, *allocation, static_cast<ID3D12Heap*>(allocation->nativeHeap), allocation->offset, true);
}

TextureHandle D3D12Device::RegisterExternalTexture(ComPtr<ID3D12Resource> resource, const TextureDesc& desc)
{
    D3D12Texture texture;
    texture.desc = desc;
    texture.resource = std::move(resource);
    texture.ownsAllocation = false;
    texture.uavs.assign(desc.mipLevels, kInvalidBindless);
    SetName(texture.resource.Get(), desc.debugName);
    return mTextures.Allocate(std::move(texture));
}

void D3D12Device::DestroyBuffer(BufferHandle handle)
{
    mBuffers.Modify(handle, [](D3D12Buffer& r) { r.destroyed = true; });
    DeferDelete(
        [this, handle]
        {
            D3D12Buffer* buffer = mBuffers.Get(handle);
            AIR_CHECK(buffer, "DestroyBuffer on stale handle");
            ReleaseDescriptor(buffer->srv);
            ReleaseDescriptor(buffer->uav);
            buffer->resource.Reset();
            if (buffer->ownsAllocation)
            {
                mAllocator->Free(buffer->allocation);
            }
            mBuffers.Free(handle);
        });
}

void D3D12Device::DestroyTexture(TextureHandle handle)
{
    mTextures.Modify(handle, [](D3D12Texture& r) { r.destroyed = true; });
    DeferDelete(
        [this, handle]
        {
            D3D12Texture* texture = mTextures.Get(handle);
            AIR_CHECK(texture, "DestroyTexture on stale handle");
            ReleaseDescriptor(texture->srv);
            for (BindlessIndex uav : texture->uavs)
            {
                ReleaseDescriptor(uav);
            }
            if (texture->rtv != ~0u)
            {
                std::lock_guard lock(mDescriptorMutex);
                mFreeRtvs.push_back(texture->rtv);
            }
            texture->resource.Reset();
            if (texture->ownsAllocation)
            {
                mAllocator->Free(texture->allocation);
            }
            mTextures.Free(handle);
        });
}

const BufferDesc* D3D12Device::GetBufferDesc(BufferHandle handle)
{
    D3D12Buffer* buffer = mBuffers.Get(handle);
    return buffer ? &buffer->desc : nullptr;
}

const TextureDesc* D3D12Device::GetTextureDesc(TextureHandle handle)
{
    D3D12Texture* texture = mTextures.Get(handle);
    return texture ? &texture->desc : nullptr;
}

void* D3D12Device::GetMappedPointer(BufferHandle handle)
{
    D3D12Buffer* buffer = mBuffers.Get(handle);
    return buffer ? buffer->mapped : nullptr;
}

HeapHandle D3D12Device::CreateHeap(u64 size, MemoryLocation location, std::string_view debugName)
{
    auto allocation = mAllocator->Allocate(
        {MemoryTypeKey(location, HeapCategory::All), location, size, kPlacementAlignment, /*forceDedicated*/ true});
    if (!allocation)
    {
        return {};
    }
    SetName(static_cast<ID3D12Heap*>(allocation->nativeHeap), debugName);
    return mHeaps.Allocate(D3D12Heap{*allocation, size, std::string(debugName)});
}

void D3D12Device::DestroyHeap(HeapHandle handle)
{
    mHeaps.Modify(handle, [](D3D12Heap& r) { r.destroyed = true; });
    DeferDelete(
        [this, handle]
        {
            D3D12Heap* heap = mHeaps.Get(handle);
            AIR_CHECK(heap, "DestroyHeap on stale handle");
            mAllocator->Free(heap->allocation);
            mHeaps.Free(handle);
        });
}

TextureHandle D3D12Device::CreatePlacedTexture(const TextureDesc& desc, HeapHandle heapHandle, u64 offset)
{
    D3D12Heap* heap = mHeaps.Get(heapHandle);
    AIR_CHECK(heap, "CreatePlacedTexture: invalid heap");
    GpuAllocation allocation = heap->allocation;
    allocation.offset += offset;
    allocation.size = GetTextureAllocationInfo(desc).size;
    AIR_CHECK(offset + allocation.size <= heap->size, "placed texture '{}' overflows heap '{}'", desc.debugName,
              heap->name);
    return FinishTexture(desc, allocation, static_cast<ID3D12Heap*>(heap->allocation.nativeHeap),
                         heap->allocation.offset + offset, false);
}

BufferHandle D3D12Device::CreatePlacedBuffer(const BufferDesc& desc, HeapHandle heapHandle, u64 offset)
{
    D3D12Heap* heap = mHeaps.Get(heapHandle);
    AIR_CHECK(heap, "CreatePlacedBuffer: invalid heap");
    GpuAllocation allocation = heap->allocation;
    allocation.offset += offset;
    allocation.size = GetBufferAllocationInfo(desc).size;
    AIR_CHECK(offset + allocation.size <= heap->size, "placed buffer '{}' overflows heap '{}'", desc.debugName,
              heap->name);
    BufferDesc placed = desc;
    placed.memory = heap->allocation.location;
    return FinishBuffer(placed, allocation, static_cast<ID3D12Heap*>(heap->allocation.nativeHeap),
                        heap->allocation.offset + offset, false);
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

BindlessIndex D3D12Device::AllocateDescriptor()
{
    std::lock_guard lock(mDescriptorMutex);
    if (!mFreeDescriptors.empty())
    {
        const BindlessIndex index = mFreeDescriptors.back();
        mFreeDescriptors.pop_back();
        return index;
    }
    AIR_CHECK(mNextDescriptor < kBindlessCapacity, "bindless heap full ({} descriptors)", kBindlessCapacity);
    return mNextDescriptor++;
}

void D3D12Device::ReleaseDescriptor(BindlessIndex index)
{
    if (index == kInvalidBindless)
    {
        return;
    }
    std::lock_guard lock(mDescriptorMutex);
    mFreeDescriptors.push_back(index);
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::CpuHandle(BindlessIndex index) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = CpuHeapStart(mBindlessHeap.Get());
    handle.ptr += SIZE_T(index) * mDescriptorSize;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::GetUavCpuMirror(BindlessIndex index) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = CpuHeapStart(mUavMirrorHeap.Get());
    handle.ptr += SIZE_T(index) * mDescriptorSize;
    return handle;
}

BindlessIndex D3D12Device::GetBindlessSRV(TextureHandle handle)
{
    D3D12Texture* t = mTextures.Get(handle);
    AIR_CHECK(t, "GetBindlessSRV: invalid texture");
    std::lock_guard lock(mViewMutex);
    if (t->srv == kInvalidBindless)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = SrvFormat(t->desc.format);
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        switch (t->desc.dimension)
        {
        case TextureDimension::Tex2D:
            if (t->desc.depthOrArraySize > 1)
            {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                srv.Texture2DArray.MipLevels = t->desc.mipLevels;
                srv.Texture2DArray.ArraySize = t->desc.depthOrArraySize;
            }
            else
            {
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srv.Texture2D.MipLevels = t->desc.mipLevels;
            }
            break;
        case TextureDimension::Tex3D:
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srv.Texture3D.MipLevels = t->desc.mipLevels;
            break;
        case TextureDimension::TexCube:
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srv.TextureCube.MipLevels = t->desc.mipLevels;
            break;
        }
        t->srv = AllocateDescriptor();
        mDevice->CreateShaderResourceView(t->resource.Get(), &srv, CpuHandle(t->srv));
    }
    return t->srv;
}

BindlessIndex D3D12Device::GetBindlessUAV(TextureHandle handle, u32 mip)
{
    D3D12Texture* t = mTextures.Get(handle);
    AIR_CHECK(t, "GetBindlessUAV: invalid texture");
    AIR_CHECK(Any(t->desc.usage & TextureUsage::UnorderedAccess), "texture '{}' lacks UnorderedAccess usage",
              t->desc.debugName);
    AIR_CHECK(mip < t->desc.mipLevels, "texture '{}' has no mip {}", t->desc.debugName, mip);
    std::lock_guard lock(mViewMutex);
    if (t->uavs[mip] == kInvalidBindless)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
        uav.Format = UavFormat(t->desc.format);
        if (t->desc.dimension == TextureDimension::Tex3D)
        {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            uav.Texture3D.MipSlice = mip;
            uav.Texture3D.WSize = ~0u;
        }
        else if (t->desc.depthOrArraySize > 1 || t->desc.dimension == TextureDimension::TexCube)
        {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            uav.Texture2DArray.MipSlice = mip;
            uav.Texture2DArray.ArraySize = t->desc.dimension == TextureDimension::TexCube ? 6 * t->desc.depthOrArraySize
                                                                                          : t->desc.depthOrArraySize;
        }
        else
        {
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uav.Texture2D.MipSlice = mip;
        }
        const BindlessIndex index = AllocateDescriptor();
        mDevice->CreateUnorderedAccessView(t->resource.Get(), nullptr, &uav, CpuHandle(index));
        mDevice->CreateUnorderedAccessView(t->resource.Get(), nullptr, &uav, GetUavCpuMirror(index));
        t->uavs[mip] = index;
    }
    return t->uavs[mip];
}

BindlessIndex D3D12Device::GetBindlessSRV(BufferHandle handle)
{
    D3D12Buffer* b = mBuffers.Get(handle);
    AIR_CHECK(b, "GetBindlessSRV: invalid buffer");
    std::lock_guard lock(mViewMutex);
    if (b->srv == kInvalidBindless)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (b->desc.stride > 0)
        {
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.Buffer.NumElements = UINT(b->desc.size / b->desc.stride);
            srv.Buffer.StructureByteStride = b->desc.stride;
        }
        else
        {
            srv.Format = DXGI_FORMAT_R32_TYPELESS;
            srv.Buffer.NumElements = UINT(AlignUp(b->desc.size, 4) / 4);
            srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        }
        b->srv = AllocateDescriptor();
        mDevice->CreateShaderResourceView(b->resource.Get(), &srv, CpuHandle(b->srv));
    }
    return b->srv;
}

BindlessIndex D3D12Device::GetBindlessUAV(BufferHandle handle)
{
    D3D12Buffer* b = mBuffers.Get(handle);
    AIR_CHECK(b, "GetBindlessUAV: invalid buffer");
    AIR_CHECK(Any(b->desc.usage & BufferUsage::UnorderedAccess), "buffer '{}' lacks UnorderedAccess usage",
              b->desc.debugName);
    std::lock_guard lock(mViewMutex);
    if (b->uav == kInvalidBindless)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        if (b->desc.stride > 0)
        {
            uav.Format = DXGI_FORMAT_UNKNOWN;
            uav.Buffer.NumElements = UINT(b->desc.size / b->desc.stride);
            uav.Buffer.StructureByteStride = b->desc.stride;
        }
        else
        {
            uav.Format = DXGI_FORMAT_R32_TYPELESS;
            uav.Buffer.NumElements = UINT(AlignUp(b->desc.size, 4) / 4);
            uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        }
        b->uav = AllocateDescriptor();
        mDevice->CreateUnorderedAccessView(b->resource.Get(), nullptr, &uav, CpuHandle(b->uav));
        mDevice->CreateUnorderedAccessView(b->resource.Get(), nullptr, &uav, GetUavCpuMirror(b->uav));
    }
    return b->uav;
}

BindlessIndex D3D12Device::AllocateBindlessSlot()
{
    return AllocateDescriptor();
}

void D3D12Device::FreeBindlessSlot(BindlessIndex index)
{
    DeferDelete([this, index] { ReleaseDescriptor(index); });
}

DescriptorHandles D3D12Device::GetBindlessDescriptor(BindlessIndex index)
{
    AIR_CHECK(index < kBindlessCapacity, "bindless index {} out of range", index);
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = GpuHeapStart(mBindlessHeap.Get());
    gpu.ptr += u64(index) * mDescriptorSize;
    return {u64(CpuHandle(index).ptr), u64(gpu.ptr)};
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::GetRtv(TextureHandle handle)
{
    D3D12Texture* t = mTextures.Get(handle);
    AIR_CHECK(t, "render target: invalid texture");
    AIR_CHECK(Any(t->desc.usage & TextureUsage::RenderTarget), "texture '{}' lacks RenderTarget usage",
              t->desc.debugName);
    std::lock_guard lock(mViewMutex);
    if (t->rtv == ~0u)
    {
        {
            std::lock_guard freeLock(mDescriptorMutex);
            if (!mFreeRtvs.empty())
            {
                t->rtv = mFreeRtvs.back();
                mFreeRtvs.pop_back();
            }
            else
            {
                AIR_CHECK(mNextRtv < kRtvCapacity, "RTV heap full");
                t->rtv = mNextRtv++;
            }
        }
        D3D12_RENDER_TARGET_VIEW_DESC rtv = {};
        rtv.Format = ToDxgi(t->desc.format);
        rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = CpuHeapStart(mRtvHeap.Get());
        cpu.ptr += SIZE_T(t->rtv) * mRtvDescriptorSize;
        mDevice->CreateRenderTargetView(t->resource.Get(), &rtv, cpu);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = CpuHeapStart(mRtvHeap.Get());
    cpu.ptr += SIZE_T(t->rtv) * mRtvDescriptorSize;
    return cpu;
}

// ---------------------------------------------------------------------------
// Interop
// ---------------------------------------------------------------------------

std::unique_ptr<Swapchain> D3D12Device::CreateSwapchain(const SwapchainDesc& desc)
{
    return CreateD3D12Swapchain(*this, desc);
}

void* D3D12Device::GetNativeObject(NativeObject object)
{
    switch (object)
    {
    case NativeObject::Device: return mDevice.Get();
    case NativeObject::Adapter: return mAdapter.Get();
    case NativeObject::GraphicsQueue: return mQueues[size_t(QueueType::Graphics)].queue.Get();
    case NativeObject::ResourceDescriptorHeap: return mBindlessHeap.Get();
    }
    return nullptr;
}

void* D3D12Device::GetNativeResource(TextureHandle handle)
{
    D3D12Texture* t = mTextures.Get(handle);
    return t ? t->resource.Get() : nullptr;
}

void* D3D12Device::GetNativeResource(BufferHandle handle)
{
    D3D12Buffer* b = mBuffers.Get(handle);
    return b ? b->resource.Get() : nullptr;
}

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------

std::unique_ptr<CommandList> D3D12Device::CreateCommandList(QueueType queue)
{
    return std::make_unique<D3D12CommandList>(*this, queue);
}

std::unique_ptr<CommandContext> D3D12Device::AcquireContext(QueueType type)
{
    Queue& queue = mQueues[size_t(type)];
    {
        std::lock_guard lock(queue.poolMutex);
        const u64 completed = queue.fence->GetCompletedValue();
        for (auto it = queue.contexts.begin(); it != queue.contexts.end(); ++it)
        {
            if ((*it)->fence <= completed)
            {
                std::unique_ptr<CommandContext> context = std::move(*it);
                queue.contexts.erase(it);
                return context;
            }
        }
    }
    static constexpr D3D12_COMMAND_LIST_TYPE kTypes[] = {D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY};
    auto context = std::make_unique<CommandContext>();
    AIR_D3D12_CHECK(mDevice->CreateCommandAllocator(kTypes[size_t(type)], IID_PPV_ARGS(&context->allocator)));
    AIR_D3D12_CHECK(mDevice->CreateCommandList1(0, kTypes[size_t(type)], D3D12_COMMAND_LIST_FLAG_NONE,
                                                IID_PPV_ARGS(&context->list)));
    return context;
}

void D3D12Device::ReleaseContext(QueueType type, std::unique_ptr<CommandContext> context)
{
    Queue& queue = mQueues[size_t(type)];
    std::lock_guard lock(queue.poolMutex);
    queue.contexts.push_back(std::move(context));
}

u64 D3D12Device::Submit(QueueType type, std::span<CommandList* const> lists)
{
    Queue& queue = mQueues[size_t(type)];
    std::vector<std::unique_ptr<CommandContext>> contexts;
    std::vector<ID3D12CommandList*> native;
    std::vector<std::string> captured;
    for (CommandList* list : lists)
    {
        AIR_CHECK(list->GetQueueType() == type, "command list submitted to the wrong queue");
        auto* d3dList = static_cast<D3D12CommandList*>(list);
        contexts.push_back(d3dList->DetachContext());
        AIR_CHECK(contexts.back(), "Submit: command list was not recorded (Begin/End)");
        native.push_back(contexts.back()->list.Get());
        if (TimestampFrame* timestamps = d3dList->Timestamps())
        {
            AppendTimestampScopes(timestamps, d3dList->TimestampScopes());
        }
        static constexpr const char* kQueueNames[] = {"Graphics", "Compute", "Copy"};
        for (std::string& line : d3dList->Captured())
        {
            captured.push_back(std::format("[{}] {}", kQueueNames[size_t(type)], line));
        }
        d3dList->Captured().clear();
    }

    u64 value = 0;
    {
        std::lock_guard lock(queue.mutex);
        queue.queue->ExecuteCommandLists(UINT(native.size()), native.data());
        value = ++queue.lastSignaled;
        const HRESULT hr = queue.queue->Signal(queue.fence.Get(), value);
        if (FAILED(hr))
        {
            LogDeviceRemoved(hr, "Signal");
        }
    }
    for (auto& context : contexts)
    {
        context->fence = value;
        ReleaseContext(type, std::move(context));
    }
    if (!captured.empty() && CaptureEnabled())
    {
        std::lock_guard lock(mCaptureMutex);
        for (std::string& line : captured)
        {
            mCaptured.push_back(std::move(line));
        }
    }
    return value;
}

bool D3D12Device::IsFenceComplete(QueueType type, u64 value)
{
    return mQueues[size_t(type)].fence->GetCompletedValue() >= value;
}

void D3D12Device::WaitForFence(QueueType type, u64 value)
{
    Queue& queue = mQueues[size_t(type)];
    if (queue.fence->GetCompletedValue() >= value)
    {
        return;
    }
    // A null event blocks until the fence reaches the value.
    const HRESULT hr = queue.fence->SetEventOnCompletion(value, nullptr);
    if (FAILED(hr))
    {
        LogDeviceRemoved(hr, "WaitForFence");
    }
}

void D3D12Device::WaitIdle()
{
    for (size_t i = 0; i < size_t(QueueType::Count); ++i)
    {
        Queue& queue = mQueues[i];
        u64 value = 0;
        {
            std::lock_guard lock(queue.mutex);
            value = ++queue.lastSignaled;
            queue.queue->Signal(queue.fence.Get(), value);
        }
        WaitForFence(QueueType(i), value);
    }
    std::deque<FrameRecord> retired;
    {
        std::lock_guard lock(mDeferredMutex);
        retired.swap(mPendingFrames);
    }
    for (const FrameRecord& record : retired)
    {
        ReadTimestamps(record.frame);
    }
    FlushDeferredDeletes(~0ull);
}

// ---------------------------------------------------------------------------
// Frames and deferred deletion
// ---------------------------------------------------------------------------

void D3D12Device::BeginFrame()
{
    RetireCompletedFrames(/*waitForOldest*/ true);
}

void D3D12Device::EndFrame()
{
    FrameRecord record;
    record.frame = mFrameIndex.load();
    for (size_t i = 0; i < size_t(QueueType::Count); ++i)
    {
        std::lock_guard lock(mQueues[i].mutex);
        record.fences[i] = mQueues[i].lastSignaled;
    }
    {
        std::lock_guard lock(mDeferredMutex);
        mPendingFrames.push_back(record);
    }
    mFrameIndex.fetch_add(1);
    RetireCompletedFrames(/*waitForOldest*/ false);
}

void D3D12Device::RetireCompletedFrames(bool waitForOldest)
{
    for (;;)
    {
        FrameRecord record;
        bool mustWait = false;
        {
            std::lock_guard lock(mDeferredMutex);
            if (mPendingFrames.empty())
            {
                return;
            }
            record = mPendingFrames.front();
            mustWait = waitForOldest && mPendingFrames.size() >= mDesc.framesInFlight;
        }
        bool complete = true;
        for (size_t i = 0; i < size_t(QueueType::Count); ++i)
        {
            if (mustWait)
            {
                WaitForFence(QueueType(i), record.fences[i]);
            }
            else if (!IsFenceComplete(QueueType(i), record.fences[i]))
            {
                complete = false;
            }
        }
        if (!complete)
        {
            return;
        }
        {
            std::lock_guard lock(mDeferredMutex);
            if (mPendingFrames.empty() || mPendingFrames.front().frame != record.frame)
            {
                continue; // retired concurrently (WaitIdle)
            }
            mPendingFrames.pop_front();
        }
        ReadTimestamps(record.frame);
        FlushDeferredDeletes(record.frame);
    }
}

void D3D12Device::DeferDelete(std::function<void()> fn)
{
    std::lock_guard lock(mDeferredMutex);
    mDeferred.push_back({mFrameIndex.load(), std::move(fn)});
}

void D3D12Device::FlushDeferredDeletes(u64 completedFrame)
{
    std::vector<std::function<void()>> ready;
    {
        std::lock_guard lock(mDeferredMutex);
        auto it = std::stable_partition(mDeferred.begin(), mDeferred.end(),
                                        [&](const DeferredDelete& d) { return d.frame > completedFrame; });
        for (auto r = it; r != mDeferred.end(); ++r)
        {
            ready.push_back(std::move(r->fn));
        }
        mDeferred.erase(it, mDeferred.end());
    }
    for (auto& fn : ready)
    {
        fn();
    }
}

// ---------------------------------------------------------------------------
// GPU profiling
// ---------------------------------------------------------------------------

void D3D12Device::SetGpuProfiling(bool enabled)
{
    mProfiling.store(enabled);
}

TimestampFrame* D3D12Device::CurrentTimestampFrame()
{
    if (!mProfiling.load(std::memory_order_relaxed))
    {
        return nullptr;
    }
    const u64 frame = mFrameIndex.load();
    TimestampFrame* slot = mTimestampFrames[frame % mTimestampFrames.size()].get();
    std::lock_guard lock(slot->mutex);
    if (slot->frame != frame)
    {
        slot->frame = frame;
        slot->scopes.clear();
        slot->next.store(0);
    }
    return slot;
}

void D3D12Device::AppendTimestampScopes(TimestampFrame* frame, std::vector<TimestampScope>& scopes)
{
    std::lock_guard lock(frame->mutex);
    for (TimestampScope& scope : scopes)
    {
        frame->scopes.push_back(std::move(scope));
    }
    scopes.clear();
}

void D3D12Device::ReadTimestamps(u64 frame)
{
    TimestampFrame* slot = mTimestampFrames[frame % mTimestampFrames.size()].get();
    GpuFrameTimings timings;
    {
        std::lock_guard lock(slot->mutex);
        if (slot->frame != frame || slot->scopes.empty())
        {
            return;
        }
        u64 first = ~0ull;
        u64 last = 0;
        for (const TimestampScope& scope : slot->scopes)
        {
            first = std::min(first, slot->mapped[scope.begin]);
            last = std::max(last, slot->mapped[scope.end]);
        }
        timings.frameIndex = frame;
        timings.frameMs = double(last - first) / mTimestampFrequency;
        for (const TimestampScope& scope : slot->scopes)
        {
            const u64 begin = slot->mapped[scope.begin];
            const u64 end = std::max(begin, slot->mapped[scope.end]);
            timings.scopes.push_back({scope.name, scope.depth, double(begin - first) / mTimestampFrequency,
                                      double(end - begin) / mTimestampFrequency});
        }
        slot->scopes.clear();
        slot->frame = ~0ull;
    }
    // Scopes close inner-first; present them in start order (parents before children).
    std::stable_sort(timings.scopes.begin(), timings.scopes.end(), [](const GpuTimingScope& a, const GpuTimingScope& b)
                     { return a.beginMs < b.beginMs || (a.beginMs == b.beginMs && a.depth < b.depth); });
    std::lock_guard lock(mTimingsMutex);
    mLastTimings = std::move(timings);
}

GpuFrameTimings D3D12Device::GetGpuTimings()
{
    std::lock_guard lock(mTimingsMutex);
    return mLastTimings;
}

// ---------------------------------------------------------------------------
// Statistics, capture, diagnostics
// ---------------------------------------------------------------------------

MemoryStats D3D12Device::GetMemoryStats()
{
    MemoryStats stats;
    for (u32 i = 0; i < u32(MemoryLocation::Count); ++i)
    {
        stats.perLocation[i] = mAllocator->GetStats(MemoryLocation(i));
    }
    stats.pools = mAllocator->GetPoolStats();
    ComPtr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(mAdapter.As(&adapter3)))
    {
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        {
            stats.local = {info.Budget, info.CurrentUsage};
        }
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info)))
        {
            stats.nonLocal = {info.Budget, info.CurrentUsage};
        }
    }
    stats.liveBuffers = mBuffers.LiveCount();
    stats.liveTextures = mTextures.LiveCount();
    stats.liveAccelStructs = mAccelStructs.LiveCount();
    stats.livePipelines = mPipelines.LiveCount();
    mAccelStructs.ForEach([&](AccelStructHandle, const D3D12AccelStruct& as)
                          { stats.accelStructBytes += as.allocation.size; });
    {
        std::lock_guard lock(mDescriptorMutex);
        stats.bindlessDescriptors = mNextDescriptor - u32(mFreeDescriptors.size());
    }
    return stats;
}

std::vector<ResourceInfo> D3D12Device::GetResourceList()
{
    std::vector<ResourceInfo> list;
    mBuffers.ForEach(
        [&](BufferHandle, const D3D12Buffer& b)
        {
            if (b.destroyed)
            {
                return;
            }
            list.push_back(
                {ResourceInfo::Kind::Buffer, b.desc.debugName, b.allocation.size, b.desc.memory, !b.ownsAllocation});
        });
    mTextures.ForEach(
        [&](TextureHandle, const D3D12Texture& t)
        {
            if (t.destroyed)
            {
                return;
            }
            list.push_back({ResourceInfo::Kind::Texture, t.desc.debugName, t.allocation.size,
                            MemoryLocation::DeviceLocal, !t.ownsAllocation && t.allocation.IsValid()});
        });
    mHeaps.ForEach(
        [&](HeapHandle, const D3D12Heap& h)
        {
            if (h.destroyed)
            {
                return;
            }
            list.push_back({ResourceInfo::Kind::Heap, h.name, h.size, h.allocation.location, false});
        });
    mAccelStructs.ForEach(
        [&](AccelStructHandle, const D3D12AccelStruct& as)
        {
            if (as.destroyed)
            {
                return;
            }
            list.push_back({ResourceInfo::Kind::AccelStruct, as.isTlas ? as.tlas.debugName : as.blas.debugName,
                            as.allocation.size, MemoryLocation::DeviceLocal, false});
        });
    return list;
}

void D3D12Device::SetCommandCapture(bool enabled)
{
    mCapture.store(enabled);
}

std::vector<std::string> D3D12Device::TakeCapturedCommands()
{
    std::lock_guard lock(mCaptureMutex);
    return std::exchange(mCaptured, {});
}

void D3D12Device::ReportValidation(std::string message)
{
    AIR_LOG(Error, "D3D12", "{}", message);
    std::lock_guard lock(mCaptureMutex);
    mCaptured.push_back(std::move(message));
}

void D3D12Device::LogDeviceRemoved(HRESULT hr, std::string_view where)
{
    const HRESULT reason = mDevice->GetDeviceRemovedReason();
    AIR_LOG(Error, "D3D12", "{} failed: {} (device removed reason: {})", where, HResultToString(hr),
            HResultToString(reason));
    ComPtr<ID3D12DeviceRemovedExtendedData1> dred;
    if (FAILED(mDevice.As(&dred)))
    {
        return;
    }
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 breadcrumbs = {};
    if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput1(&breadcrumbs)))
    {
        for (const D3D12_AUTO_BREADCRUMB_NODE1* node = breadcrumbs.pHeadAutoBreadcrumbNode; node; node = node->pNext)
        {
            const u32 done = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
            if (done < node->BreadcrumbCount)
            {
                AIR_LOG(Error, "D3D12", "  unfinished: list '{}' on queue '{}', {} of {} operations completed",
                        node->pCommandListDebugNameA ? node->pCommandListDebugNameA : "?",
                        node->pCommandQueueDebugNameA ? node->pCommandQueueDebugNameA : "?", done,
                        node->BreadcrumbCount);
            }
        }
    }
    D3D12_DRED_PAGE_FAULT_OUTPUT1 pageFault = {};
    if (SUCCEEDED(dred->GetPageFaultAllocationOutput1(&pageFault)) && pageFault.PageFaultVA)
    {
        AIR_LOG(Error, "D3D12", "  page fault at GPU VA 0x{:X}", u64(pageFault.PageFaultVA));
    }
}

} // namespace air::rhi::d3d12
