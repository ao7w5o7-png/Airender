// Direct3D 12 pipelines: compute PSOs, DXR acceleration structures, state objects and shader tables.

#include "D3D12Backend.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include <atomic>
#include <cstring>
#include <format>

namespace air::rhi::d3d12 {

namespace {

std::wstring Widen(std::string_view text) {
    return std::wstring(text.begin(), text.end()); // entry point names are ASCII
}

D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS BuildFlags(bool allowUpdate, bool fastTrace) {
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags =
        fastTrace ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                  : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    if (allowUpdate) {
        flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }
    return flags;
}

ComPtr<ID3D12Resource> CreateUploadBuffer(ID3D12Device5* device, u64 size, std::string_view name) {
    D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = AlignUp(size, 256);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> resource;
    AIR_D3D12_CHECK(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&resource)));
    const std::wstring wide = Widen(name);
    resource->SetName(wide.c_str());
    return resource;
}

} // namespace

// ---------------------------------------------------------------------------
// Acceleration structures
// ---------------------------------------------------------------------------

AccelStructHandle D3D12Device::CreateBlas(const BlasDesc& desc) {
    D3D12AccelStruct as;
    as.blas = desc;
    for (const RayTracingGeometryTriangles& g : desc.geometries) {
        D3D12Buffer* vertices = mBuffers.Get(g.vertexBuffer);
        D3D12Buffer* indices = mBuffers.Get(g.indexBuffer);
        AIR_CHECK(vertices, "BLAS '{}': invalid vertex buffer", desc.debugName);
        D3D12_RAYTRACING_GEOMETRY_DESC geometry = {};
        geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry.Flags = g.opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        geometry.Triangles.VertexBuffer.StartAddress = vertices->address + g.vertexOffset;
        geometry.Triangles.VertexBuffer.StrideInBytes = g.vertexStride;
        geometry.Triangles.VertexCount = g.vertexCount;
        geometry.Triangles.VertexFormat = ToDxgi(g.vertexFormat);
        if (indices) {
            geometry.Triangles.IndexBuffer = indices->address + g.indexOffset;
            geometry.Triangles.IndexCount = g.indexCount;
            geometry.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        }
        as.geometries.push_back(geometry);
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.Flags = BuildFlags(desc.allowUpdate, desc.preferFastTrace);
    inputs.NumDescs = UINT(as.geometries.size());
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.pGeometryDescs = as.geometries.data();
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild = {};
    mDevice->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);
    as.scratchSize = AlignUp(prebuild.ScratchDataSizeInBytes, 256);

    const BufferDesc storage = {.size = AlignUp(prebuild.ResultDataMaxSizeInBytes, 256),
                                .usage = BufferUsage::AccelStructStorage,
                                .debugName = desc.debugName};
    const AllocationInfo info = GetBufferAllocationInfo(storage);
    auto allocation = mAllocator->Allocate({u32(MemoryLocation::DeviceLocal) * 4 + u32(HeapCategory::Buffer),
                                            MemoryLocation::DeviceLocal, info.size, info.alignment});
    if (!allocation) {
        AIR_LOG(Error, "D3D12", "out of video memory for BLAS '{}'", desc.debugName);
        return {};
    }
    const D3D12_RESOURCE_DESC rd = MakeBufferResourceDesc(storage);
    AIR_D3D12_CHECK(mDevice->CreatePlacedResource(static_cast<ID3D12Heap*>(allocation->nativeHeap), allocation->offset,
                                                  &rd, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr,
                                                  IID_PPV_ARGS(&as.resource)));
    as.resource->SetName(Widen(desc.debugName).c_str());
    as.allocation = *allocation;
    as.address = as.resource->GetGPUVirtualAddress();
    return mAccelStructs.Allocate(std::move(as));
}

AccelStructHandle D3D12Device::CreateTlas(const TlasDesc& desc) {
    AIR_CHECK(desc.maxInstances > 0, "TLAS '{}' needs maxInstances > 0", desc.debugName);
    D3D12AccelStruct as;
    as.isTlas = true;
    as.tlas = desc;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.Flags = BuildFlags(desc.allowUpdate, true);
    inputs.NumDescs = desc.maxInstances;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild = {};
    mDevice->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);
    as.scratchSize = AlignUp(prebuild.ScratchDataSizeInBytes, 256);

    const BufferDesc storage = {.size = AlignUp(prebuild.ResultDataMaxSizeInBytes, 256),
                                .usage = BufferUsage::AccelStructStorage,
                                .debugName = desc.debugName};
    const AllocationInfo info = GetBufferAllocationInfo(storage);
    auto allocation = mAllocator->Allocate({u32(MemoryLocation::DeviceLocal) * 4 + u32(HeapCategory::Buffer),
                                            MemoryLocation::DeviceLocal, info.size, info.alignment});
    if (!allocation) {
        AIR_LOG(Error, "D3D12", "out of video memory for TLAS '{}'", desc.debugName);
        return {};
    }
    const D3D12_RESOURCE_DESC rd = MakeBufferResourceDesc(storage);
    AIR_D3D12_CHECK(mDevice->CreatePlacedResource(static_cast<ID3D12Heap*>(allocation->nativeHeap), allocation->offset,
                                                  &rd, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr,
                                                  IID_PPV_ARGS(&as.resource)));
    as.resource->SetName(Widen(desc.debugName).c_str());
    as.allocation = *allocation;
    as.address = as.resource->GetGPUVirtualAddress();

    // Enough slots that a build never reuses instance memory the GPU may still read.
    as.ringSlots = mDesc.framesInFlight + 2;
    const u64 slotSize = AlignUp(u64(desc.maxInstances) * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), 256);
    as.instanceRing = CreateUploadBuffer(mDevice.Get(), slotSize * as.ringSlots, desc.debugName + ".Instances");
    void* mapped = nullptr;
    const D3D12_RANGE noRead = {0, 0};
    AIR_D3D12_CHECK(as.instanceRing->Map(0, &noRead, &mapped));
    as.instanceMapped = static_cast<u8*>(mapped);
    return mAccelStructs.Allocate(std::move(as));
}

void D3D12Device::DestroyAccelStruct(AccelStructHandle handle) {
    DeferDelete([this, handle] {
        D3D12AccelStruct* as = mAccelStructs.Get(handle);
        AIR_CHECK(as, "DestroyAccelStruct on stale handle");
        ReleaseDescriptor(as->srv);
        as->resource.Reset();
        as->instanceRing.Reset();
        mAllocator->Free(as->allocation);
        mAccelStructs.Free(handle);
    });
}

u64 D3D12Device::GetAccelStructScratchSize(AccelStructHandle handle) {
    D3D12AccelStruct* as = mAccelStructs.Get(handle);
    AIR_CHECK(as, "GetAccelStructScratchSize: invalid handle");
    return as->scratchSize;
}

BindlessIndex D3D12Device::GetBindlessSRV(AccelStructHandle handle) {
    D3D12AccelStruct* as = mAccelStructs.Get(handle);
    AIR_CHECK(as && as->isTlas, "GetBindlessSRV: invalid TLAS");
    std::lock_guard lock(mViewMutex);
    if (as->srv == kInvalidBindless) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.RaytracingAccelerationStructure.Location = as->address;
        as->srv = AllocateDescriptor();
        mDevice->CreateShaderResourceView(nullptr, &srv, CpuHandle(as->srv));
    }
    return as->srv;
}

void D3D12CommandList::BuildBlas(AccelStructHandle handle, BufferHandle scratchHandle) {
    D3D12AccelStruct* as = mDevice.AccelStruct(handle);
    D3D12Buffer* scratch = mDevice.Buffer(scratchHandle);
    AIR_CHECK(as && !as->isTlas, "BuildBlas: invalid BLAS");
    AIR_CHECK(scratch && scratch->desc.size >= as->scratchSize, "BuildBlas '{}': scratch buffer too small",
              as->blas.debugName);
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = {};
    build.DestAccelerationStructureData = as->address;
    build.ScratchAccelerationStructureData = scratch->address;
    build.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    build.Inputs.Flags = BuildFlags(as->blas.allowUpdate, as->blas.preferFastTrace);
    build.Inputs.NumDescs = UINT(as->geometries.size());
    build.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    build.Inputs.pGeometryDescs = as->geometries.data();
    mList->BuildRaytracingAccelerationStructure(&build, 0, nullptr);

    // Builds are always followed by a UAV barrier: the result feeds TLAS builds / traces and the
    // scratch buffer is usually reused by the next build.
    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barriers[0].UAV.pResource = as->resource.Get();
    barriers[1].UAV.pResource = scratch->resource.Get();
    mList->ResourceBarrier(2, barriers);
    Capture(std::format("BuildBlas '{}'", as->blas.debugName));
}

void D3D12CommandList::BuildTlas(AccelStructHandle handle, std::span<const TlasInstance> instances,
                                 BufferHandle scratchHandle) {
    D3D12AccelStruct* as = mDevice.AccelStruct(handle);
    D3D12Buffer* scratch = mDevice.Buffer(scratchHandle);
    AIR_CHECK(as && as->isTlas, "BuildTlas: invalid TLAS");
    AIR_CHECK(instances.size() <= as->tlas.maxInstances, "TLAS '{}': {} instances > maxInstances {}",
              as->tlas.debugName, instances.size(), as->tlas.maxInstances);
    AIR_CHECK(scratch && scratch->desc.size >= as->scratchSize, "BuildTlas '{}': scratch buffer too small",
              as->tlas.debugName);

    const u64 slotSize = AlignUp(u64(as->tlas.maxInstances) * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), 256);
    const u32 slot = std::atomic_ref<u32>(as->nextSlot).fetch_add(1) % as->ringSlots;
    auto* descs = reinterpret_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(as->instanceMapped + slot * slotSize);
    for (size_t i = 0; i < instances.size(); ++i) {
        const TlasInstance& in = instances[i];
        D3D12AccelStruct* blas = mDevice.AccelStruct(in.blas);
        AIR_CHECK(blas && !blas->isTlas, "TLAS '{}': instance {} has an invalid BLAS", as->tlas.debugName, i);
        D3D12_RAYTRACING_INSTANCE_DESC desc = {};
        std::memcpy(desc.Transform, in.transform, sizeof(desc.Transform));
        desc.InstanceID = in.instanceId & 0xFFFFFF;
        desc.InstanceMask = in.mask & 0xFF;
        desc.InstanceContributionToHitGroupIndex = in.hitGroupOffset & 0xFFFFFF;
        desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
        desc.AccelerationStructure = blas->address;
        descs[i] = desc;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = {};
    build.DestAccelerationStructureData = as->address;
    build.ScratchAccelerationStructureData = scratch->address;
    build.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    build.Inputs.Flags = BuildFlags(as->tlas.allowUpdate, true);
    build.Inputs.NumDescs = UINT(instances.size());
    build.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    build.Inputs.InstanceDescs = as->instanceRing->GetGPUVirtualAddress() + slot * slotSize;
    mList->BuildRaytracingAccelerationStructure(&build, 0, nullptr);

    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barriers[0].UAV.pResource = as->resource.Get();
    barriers[1].UAV.pResource = scratch->resource.Get();
    mList->ResourceBarrier(2, barriers);
    Capture(std::format("BuildTlas '{}' instances {}", as->tlas.debugName, instances.size()));
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

PipelineHandle D3D12Device::CreateComputePipeline(const ComputePipelineDesc& desc) {
    if (desc.shader.bytecode.empty()) {
        AIR_LOG(Error, "D3D12", "compute pipeline '{}': no shader bytecode (was the shader compiled?)", desc.debugName);
        return {};
    }
    AIR_CHECK(desc.pushConstantBytes <= kRootConstantCount * 4, "pipeline '{}': {} push constant bytes > {}",
              desc.debugName, desc.pushConstantBytes, kRootConstantCount * 4);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
    pso.pRootSignature = mRootSignature.Get();
    pso.CS = {desc.shader.bytecode.data(), desc.shader.bytecode.size()};
    D3D12Pipeline pipeline;
    pipeline.name = desc.debugName;
    const HRESULT hr = mDevice->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline.pso));
    if (FAILED(hr)) {
        AIR_LOG(Error, "D3D12", "compute pipeline '{}': CreateComputePipelineState failed: {}", desc.debugName,
                HResultToString(hr));
        return {};
    }
    pipeline.pso->SetName(Widen(desc.debugName).c_str());
    return mPipelines.Allocate(std::move(pipeline));
}

PipelineHandle D3D12Device::CreateRayTracingPipeline(const RayTracingPipelineDesc& desc) {
    if (!mCaps.rayTracing) {
        AIR_LOG(Error, "D3D12", "ray tracing pipeline '{}': device has no DXR 1.1", desc.debugName);
        return {};
    }
    if (desc.library.bytecode.empty()) {
        AIR_LOG(Error, "D3D12", "ray tracing pipeline '{}': no library bytecode (was the shader compiled?)",
                desc.debugName);
        return {};
    }

    // Wide strings must outlive CreateStateObject.
    std::vector<std::wstring> names;
    names.reserve(desc.hitGroups.size() * 3 + desc.missShaders.size() + 1);
    auto keep = [&](std::string_view s) -> LPCWSTR {
        if (s.empty()) {
            return nullptr;
        }
        names.push_back(Widen(s));
        return names.back().c_str();
    };

    std::vector<D3D12_STATE_SUBOBJECT> subobjects;
    subobjects.reserve(4 + desc.hitGroups.size());

    D3D12_DXIL_LIBRARY_DESC library = {};
    library.DXILLibrary = {desc.library.bytecode.data(), desc.library.bytecode.size()};
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library}); // exports everything

    std::vector<D3D12_HIT_GROUP_DESC> hitGroups(desc.hitGroups.size());
    for (size_t i = 0; i < desc.hitGroups.size(); ++i) {
        hitGroups[i].HitGroupExport = keep(desc.hitGroups[i].name);
        hitGroups[i].Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
        hitGroups[i].ClosestHitShaderImport = keep(desc.hitGroups[i].closestHit);
        hitGroups[i].AnyHitShaderImport = keep(desc.hitGroups[i].anyHit);
        subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hitGroups[i]});
    }

    D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {desc.maxPayloadBytes, desc.maxAttributeBytes};
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderConfig});

    D3D12_GLOBAL_ROOT_SIGNATURE globalRoot = {mRootSignature.Get()};
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &globalRoot});

    D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig = {desc.maxRecursionDepth};
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipelineConfig});

    D3D12_STATE_OBJECT_DESC stateDesc = {};
    stateDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    stateDesc.NumSubobjects = UINT(subobjects.size());
    stateDesc.pSubobjects = subobjects.data();

    D3D12Pipeline pipeline;
    pipeline.rayTracing = true;
    pipeline.name = desc.debugName;
    const HRESULT hr = mDevice->CreateStateObject(&stateDesc, IID_PPV_ARGS(&pipeline.stateObject));
    if (FAILED(hr)) {
        AIR_LOG(Error, "D3D12",
                "ray tracing pipeline '{}': CreateStateObject failed: {} (the debug layer log has "
                "the details)",
                desc.debugName, HResultToString(hr));
        return {};
    }
    pipeline.stateObject->SetName(Widen(desc.debugName).c_str());

    ComPtr<ID3D12StateObjectProperties> properties;
    AIR_D3D12_CHECK(pipeline.stateObject.As(&properties));
    auto identifier = [&](std::string_view name) -> const void* {
        const std::wstring wide = Widen(name);
        const void* id = properties->GetShaderIdentifier(wide.c_str());
        AIR_CHECK(id, "ray tracing pipeline '{}': no export named '{}'", desc.debugName, name);
        return id;
    };

    // Shader table: [raygen][miss...][hit groups...]; each table starts 64-byte aligned and each
    // record is one 32-byte identifier (no local root arguments: everything is bindless).
    constexpr u64 kRecord = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
    static_assert(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES == kRecord);
    constexpr u64 kTableAlign = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    const u64 rayGenOffset = 0;
    const u64 missOffset = AlignUp(rayGenOffset + kRecord, kTableAlign);
    const u64 hitOffset = AlignUp(missOffset + kRecord * std::max<size_t>(desc.missShaders.size(), 1), kTableAlign);
    const u64 total = hitOffset + kRecord * std::max<size_t>(desc.hitGroups.size(), 1);
    pipeline.shaderTable = CreateUploadBuffer(mDevice.Get(), total, desc.debugName + ".ShaderTable");
    u8* table = nullptr;
    const D3D12_RANGE noRead = {0, 0};
    AIR_D3D12_CHECK(pipeline.shaderTable->Map(0, &noRead, reinterpret_cast<void**>(&table)));
    std::memcpy(table + rayGenOffset, identifier(desc.rayGen), kRecord);
    for (size_t i = 0; i < desc.missShaders.size(); ++i) {
        std::memcpy(table + missOffset + i * kRecord, identifier(desc.missShaders[i]), kRecord);
    }
    for (size_t i = 0; i < desc.hitGroups.size(); ++i) {
        std::memcpy(table + hitOffset + i * kRecord, identifier(desc.hitGroups[i].name), kRecord);
    }
    pipeline.shaderTable->Unmap(0, nullptr);

    const D3D12_GPU_VIRTUAL_ADDRESS base = pipeline.shaderTable->GetGPUVirtualAddress();
    pipeline.rayGen = {base + rayGenOffset, kRecord};
    pipeline.miss = {base + missOffset, kRecord * desc.missShaders.size(), kRecord};
    pipeline.hitGroups = {base + hitOffset, kRecord * desc.hitGroups.size(), kRecord};
    return mPipelines.Allocate(std::move(pipeline));
}

void D3D12Device::DestroyPipeline(PipelineHandle handle) {
    DeferDelete([this, handle] { mPipelines.Free(handle); });
}

} // namespace air::rhi::d3d12
