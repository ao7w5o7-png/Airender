#include "air/renderer/Scene.h"

#include "air/core/Assert.h"
#include "air/core/Profiler.h"
#include "air/renderer/Transfer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <numbers>

namespace air::renderer
{

using rhi::Access;
using rhi::Barrier;
using rhi::BufferUsage;
using rhi::MemoryLocation;

namespace
{

// Rows of the inverse transpose of the upper 3x3 (cofactor matrix / determinant).
void NormalMatrix(const float m[3][4], hlsl::float4 out[3])
{
    const float3 r0(m[0][0], m[0][1], m[0][2]);
    const float3 r1(m[1][0], m[1][1], m[1][2]);
    const float3 r2(m[2][0], m[2][1], m[2][2]);
    const float3 c0 = hlsl::cross(r1, r2);
    const float3 c1 = hlsl::cross(r2, r0);
    const float3 c2 = hlsl::cross(r0, r1);
    const float det = hlsl::dot(r0, c0);
    AIR_CHECK(std::abs(det) > 1e-12f, "instance transform is singular");
    const float inv = 1.0f / det;
    out[0] = {c0 * inv, 0.0f};
    out[1] = {c1 * inv, 0.0f};
    out[2] = {c2 * inv, 0.0f};
}

hlsl::float4 Row(const float m[3][4], int r)
{
    return {m[r][0], m[r][1], m[r][2], m[r][3]};
}

void WriteBuffer(rhi::Device& device, rhi::BufferHandle buffer, const void* data, u64 size)
{
    void* mapped = device.GetMappedPointer(buffer);
    AIR_CHECK(mapped, "buffer is not host visible");
    std::memcpy(mapped, data, size);
}

} // namespace

float3 TransformPoint(const float m[3][4], float3 p)
{
    return {m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
            m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]};
}

namespace
{

float EmittedLuminance(const SceneDesc& desc, const Instance& instance)
{
    return hlsl::dot(desc.materials[instance.material].ToGpu().emission, float3(0.2126f, 0.7152f, 0.0722f));
}

} // namespace

std::vector<u32> FirstLightIndices(const SceneDesc& desc)
{
    std::vector<u32> first;
    u32 next = 0;
    for (const Instance& instance : desc.instances)
    {
        if (EmittedLuminance(desc, instance) > 0.0f)
        {
            first.push_back(next);
            next += u32(desc.meshes[instance.mesh].indices.size() / 3);
        }
        else
        {
            first.push_back(gpu::kInvalidIndex);
        }
    }
    return first;
}

gpu::InstanceData MakeInstanceData(const Instance& in, u32 vertexBuffer, u32 indexBuffer)
{
    gpu::InstanceData data;
    data.vertexBuffer = vertexBuffer;
    data.indexBuffer = indexBuffer;
    data.materialIndex = in.material;
    data.reserved = gpu::kInvalidIndex;
    data.objectToWorld0 = Row(in.transform, 0);
    data.objectToWorld1 = Row(in.transform, 1);
    data.objectToWorld2 = Row(in.transform, 2);
    hlsl::float4 normal[3];
    NormalMatrix(in.transform, normal);
    data.normalToWorld0 = normal[0];
    data.normalToWorld1 = normal[1];
    data.normalToWorld2 = normal[2];
    return data;
}

// Probability proportional to emitted power, so bright and large emitters get most light samples.
std::vector<gpu::LightTriangle> BuildLightTriangles(const SceneDesc& desc, double* totalPowerOut)
{
    std::vector<gpu::LightTriangle> lights;
    double totalPower = 0.0;
    for (size_t i = 0; i < desc.instances.size(); ++i)
    {
        const Instance& in = desc.instances[i];
        const Mesh& mesh = desc.meshes[in.mesh];
        const gpu::MaterialData material = desc.materials[in.material].ToGpu();
        const float luminance = EmittedLuminance(desc, in);
        if (luminance <= 0.0f)
        {
            continue; // not in the list (FirstLightIndices maps it to kInvalidIndex)
        }
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3)
        {
            gpu::LightTriangle light;
            const float3 p0 = TransformPoint(in.transform, mesh.vertices[mesh.indices[t]].position);
            const float3 p1 = TransformPoint(in.transform, mesh.vertices[mesh.indices[t + 1]].position);
            const float3 p2 = TransformPoint(in.transform, mesh.vertices[mesh.indices[t + 2]].position);
            light.p0 = p0;
            light.e1 = p1 - p0;
            light.e2 = p2 - p0;
            light.area = 0.5f * hlsl::length(hlsl::cross(light.e1, light.e2));
            light.radiance = material.emission;
            // Power of a one-sided Lambertian emitter, L * A * pi; normalized below.
            light.probability = luminance * light.area * std::numbers::pi_v<float>;
            totalPower += light.probability;
            lights.push_back(light);
        }
    }
    // Normalize, then build the alias table (Vose): every entry ends up with a threshold and an alias
    // such that drawing an entry uniformly and keeping it with probability `aliasThreshold` selects
    // light j with exactly probability[j]. Doubles keep the redistribution exact for large lists.
    const size_t count = lights.size();
    std::vector<double> scaled(count);
    std::vector<u32> small;
    std::vector<u32> large;
    for (size_t i = 0; i < count; ++i)
    {
        const double p = totalPower > 0.0 ? lights[i].probability / totalPower : 0.0;
        lights[i].probability = float(p);
        lights[i].aliasThreshold = 1.0f;
        lights[i].alias = u32(i);
        scaled[i] = p * double(count);
        (scaled[i] < 1.0 ? small : large).push_back(u32(i));
    }
    while (!small.empty() && !large.empty())
    {
        const u32 s = small.back();
        small.pop_back();
        const u32 l = large.back();
        lights[s].aliasThreshold = float(scaled[s]);
        lights[s].alias = l;
        scaled[l] -= 1.0 - scaled[s];
        if (scaled[l] < 1.0)
        {
            large.pop_back();
            small.push_back(l);
        }
    }
    // Whatever is left is 1 up to rounding: keep itself.
    for (u32 i : small)
    {
        lights[i].aliasThreshold = 1.0f;
    }
    for (u32 i : large)
    {
        lights[i].aliasThreshold = 1.0f;
    }
    if (totalPowerOut)
    {
        *totalPowerOut = totalPower;
    }
    return lights;
}

Scene::Scene(rhi::Device& device, SceneDesc desc) : mDevice(device), mDesc(std::move(desc))
{
    const std::string problem = Validate(mDesc);
    AIR_CHECK(problem.empty(), "scene '{}': {}", mDesc.name, problem);
    Upload();
    const u32 slots = mDevice.GetFramesInFlight() + 1;
    for (u32 i = 0; i < slots; ++i)
    {
        mMaterialRing.push_back(mDevice.CreateBuffer({.size = mDesc.materials.size() * sizeof(gpu::MaterialData),
                                                      .stride = sizeof(gpu::MaterialData),
                                                      .usage = BufferUsage::ShaderResource,
                                                      .memory = MemoryLocation::DeviceLocalHost,
                                                      .debugName = std::format("{}.Materials[{}]", mDesc.name, i)}));
    }
    mLightRing.assign(slots, {});
    mLightRingCapacity.assign(slots, 0);
    mLightTreeRing.assign(slots, {});
    mLightTreeRingCapacity.assign(slots, 0);
    mSkyRing.assign(slots, {});
    mSkyRingCapacity.assign(slots, 0);
    for (u32 i = 0; i < slots; ++i)
    {
        mInstanceLightRing.push_back(
            mDevice.CreateBuffer({.size = mDesc.instances.size() * sizeof(u32),
                                  .stride = sizeof(u32),
                                  .usage = BufferUsage::ShaderResource,
                                  .memory = MemoryLocation::DeviceLocalHost,
                                  .debugName = std::format("{}.InstanceLights[{}]", mDesc.name, i)}));
    }
    Update();
}

Scene::~Scene()
{
    for (GpuMesh& mesh : mMeshes)
    {
        mDevice.DestroyBuffer(mesh.vertices);
        mDevice.DestroyBuffer(mesh.indices);
        mDevice.DestroyAccelStruct(mesh.blas);
    }
    for (rhi::BufferHandle buffer : mMaterialRing)
    {
        mDevice.DestroyBuffer(buffer);
    }
    for (const auto* ring : {&mLightRing, &mLightTreeRing, &mSkyRing})
    {
        for (rhi::BufferHandle buffer : *ring)
        {
            if (buffer.IsValid())
            {
                mDevice.DestroyBuffer(buffer);
            }
        }
    }
    for (rhi::BufferHandle buffer : mInstanceLightRing)
    {
        mDevice.DestroyBuffer(buffer);
    }
    DestroyClouds();
    DestroyAerial();
    mDevice.DestroyBuffer(mInstances);
    mDevice.DestroyAccelStruct(mGpu.tlas);
}

void Scene::Upload()
{
    AIR_PROFILE_SCOPE("Scene.Upload");
    const BufferUsage geometryUsage =
        BufferUsage::ShaderResource | BufferUsage::AccelStructBuildInput | BufferUsage::CopyDst;
    // One staging buffer for all geometry and instance data.
    u64 stagingSize = mDesc.instances.size() * sizeof(gpu::InstanceData);
    for (const Mesh& mesh : mDesc.meshes)
    {
        stagingSize += mesh.vertices.size() * sizeof(gpu::Vertex) + mesh.indices.size() * sizeof(u32);
    }
    const rhi::BufferHandle staging = mDevice.CreateBuffer({.size = stagingSize,
                                                            .usage = BufferUsage::CopySrc,
                                                            .memory = MemoryLocation::Upload,
                                                            .debugName = mDesc.name + ".Staging"});
    auto* stagingBytes = static_cast<u8*>(mDevice.GetMappedPointer(staging));
    AIR_CHECK(stagingBytes, "staging buffer is not host visible");
    u64 stagingOffset = 0;

    auto cmd = mDevice.CreateCommandList(rhi::QueueType::Graphics);
    cmd->Begin();
    cmd->BeginEvent("Scene.Upload");
    std::vector<Barrier> toShaderRead;
    auto uploadBuffer = [&](const void* data, u64 size, u32 stride, BufferUsage usage, std::string name)
    {
        const rhi::BufferHandle buffer =
            mDevice.CreateBuffer({.size = size, .stride = stride, .usage = usage, .debugName = std::move(name)});
        AIR_CHECK(buffer.IsValid(), "out of device memory uploading scene '{}'", mDesc.name);
        std::memcpy(stagingBytes + stagingOffset, data, size);
        cmd->Barriers(std::array{Barrier::Buf(buffer, Access::None, Access::CopyDst)});
        cmd->CopyBuffer(buffer, 0, staging, stagingOffset, size);
        toShaderRead.push_back(Barrier::Buf(buffer, Access::CopyDst, Access::SRVCompute));
        stagingOffset += size;
        mGeometryBytes += size;
        return buffer;
    };

    u64 scratchSize = 0;
    for (const Mesh& mesh : mDesc.meshes)
    {
        GpuMesh gpu;
        gpu.vertices = uploadBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(gpu::Vertex),
                                    sizeof(gpu::Vertex), geometryUsage, mesh.name + ".Vertices");
        gpu.indices = uploadBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(u32), sizeof(u32), geometryUsage,
                                   mesh.name + ".Indices");
        rhi::BlasDesc blas;
        blas.geometries.push_back({.vertexBuffer = gpu.vertices,
                                   .vertexCount = u32(mesh.vertices.size()),
                                   .vertexStride = sizeof(gpu::Vertex),
                                   .indexBuffer = gpu.indices,
                                   .indexCount = u32(mesh.indices.size())});
        blas.debugName = mesh.name + ".BLAS";
        gpu.blas = mDevice.CreateBlas(blas);
        AIR_CHECK(gpu.blas.IsValid(), "failed to create BLAS for mesh '{}'", mesh.name);
        scratchSize = std::max(scratchSize, mDevice.GetAccelStructScratchSize(gpu.blas));
        mTriangleCount += u32(mesh.indices.size() / 3);
        mMeshes.push_back(gpu);
    }

    std::vector<gpu::InstanceData> instances;
    std::vector<rhi::TlasInstance> tlasInstances;
    for (size_t i = 0; i < mDesc.instances.size(); ++i)
    {
        const Instance& in = mDesc.instances[i];
        instances.push_back(MakeInstanceData(in, mDevice.GetBindlessSRV(mMeshes[in.mesh].vertices),
                                             mDevice.GetBindlessSRV(mMeshes[in.mesh].indices)));
        rhi::TlasInstance tlas;
        std::memcpy(tlas.transform, in.transform, sizeof(tlas.transform));
        tlas.instanceId = u32(i);
        tlas.blas = mMeshes[in.mesh].blas;
        tlasInstances.push_back(tlas);
    }
    mInstances = uploadBuffer(instances.data(), instances.size() * sizeof(gpu::InstanceData), sizeof(gpu::InstanceData),
                              BufferUsage::ShaderResource | BufferUsage::CopyDst, mDesc.name + ".Instances");
    cmd->Barriers(toShaderRead);

    mGpu.tlas = mDevice.CreateTlas(
        {.maxInstances = u32(tlasInstances.size()), .allowUpdate = false, .debugName = mDesc.name + ".TLAS"});
    AIR_CHECK(mGpu.tlas.IsValid(), "failed to create the TLAS of scene '{}'", mDesc.name);
    scratchSize = std::max(scratchSize, mDevice.GetAccelStructScratchSize(mGpu.tlas));
    const rhi::BufferHandle scratch = mDevice.CreateBuffer({.size = std::max<u64>(scratchSize, 256),
                                                            .usage = BufferUsage::UnorderedAccess,
                                                            .debugName = mDesc.name + ".BuildScratch"});
    cmd->Barriers(std::array{Barrier::Buf(scratch, Access::None, Access::UAVCompute)});
    // Builds that share the scratch buffer are serialized by the UAV barrier each build ends with.
    for (const GpuMesh& mesh : mMeshes)
    {
        cmd->BuildBlas(mesh.blas, scratch);
    }
    cmd->BuildTlas(mGpu.tlas, tlasInstances, scratch);
    cmd->EndEvent();
    cmd->End();

    rhi::CommandList* lists[] = {cmd.get()};
    const u64 fence = mDevice.Submit(rhi::QueueType::Graphics, lists);
    mDevice.WaitForFence(rhi::QueueType::Graphics, fence);
    mDevice.DestroyBuffer(staging);
    mDevice.DestroyBuffer(scratch);
    mGpu.instances = mInstances;
}

void Scene::SetMaterial(u32 index, const Material& material)
{
    AIR_CHECK(index < mDesc.materials.size(), "SetMaterial: index {} out of range", index);
    if (!(mDesc.materials[index] == material))
    {
        mDesc.materials[index] = material;
        mDirty = true;
    }
}

void Scene::SetSky(const Sky& sky)
{
    if (!(mDesc.sky == sky))
    {
        mDesc.sky = sky;
        mDirty = true;
    }
}

void Scene::SetClouds(const Clouds& clouds)
{
    if (!(mDesc.clouds == clouds))
    {
        mDesc.clouds = clouds;
        mDirty = true;
    }
}

void Scene::SetEnvironment(float3 radiance)
{
    if (!(mDesc.environment == radiance))
    {
        mDesc.environment = radiance;
        mDirty = true;
    }
}

void Scene::Update()
{
    if (!mDirty)
    {
        return;
    }
    mRingSlot = (mRingSlot + 1) % u32(mMaterialRing.size());
    WriteDynamicData();
    ++mGpu.version;
    mDirty = false;
}

void Scene::WriteDynamicData()
{
    AIR_PROFILE_SCOPE("Scene.WriteDynamicData");
    std::vector<gpu::MaterialData> materials;
    materials.reserve(mDesc.materials.size());
    for (const Material& m : mDesc.materials)
    {
        materials.push_back(m.ToGpu());
    }
    WriteBuffer(mDevice, mMaterialRing[mRingSlot], materials.data(), materials.size() * sizeof(gpu::MaterialData));
    mGpu.materials = mMaterialRing[mRingSlot];

    double totalPower = 0.0;
    std::vector<gpu::LightTriangle> lights = BuildLightTriangles(mDesc, &totalPower);
    const std::vector<gpu::LightTreeNode> tree = BuildLightTree(lights); // also sets the lights' trails
    if (lights.empty())
    {
        lights.emplace_back(); // keep a valid buffer to bind
    }

    // Grows a ring slot to hold `count` elements; the old buffer goes once the GPU is done with it.
    auto ensure = [&](std::vector<rhi::BufferHandle>& ring, std::vector<u32>& capacity, size_t count, u32 stride,
                      const char* what)
    {
        if (capacity[mRingSlot] >= count)
        {
            return;
        }
        if (ring[mRingSlot].IsValid())
        {
            mDevice.DestroyBuffer(ring[mRingSlot]);
        }
        ring[mRingSlot] = mDevice.CreateBuffer({.size = count * stride,
                                                .stride = stride,
                                                .usage = BufferUsage::ShaderResource,
                                                .memory = MemoryLocation::DeviceLocalHost,
                                                .debugName = std::format("{}.{}[{}]", mDesc.name, what, mRingSlot)});
        capacity[mRingSlot] = u32(count);
    };
    ensure(mLightRing, mLightRingCapacity, lights.size(), sizeof(gpu::LightTriangle), "Lights");
    WriteBuffer(mDevice, mLightRing[mRingSlot], lights.data(), lights.size() * sizeof(gpu::LightTriangle));
    mGpu.lights = mLightRing[mRingSlot];
    ensure(mLightTreeRing, mLightTreeRingCapacity, tree.size(), sizeof(gpu::LightTreeNode), "LightTree");
    WriteBuffer(mDevice, mLightTreeRing[mRingSlot], tree.data(), tree.size() * sizeof(gpu::LightTreeNode));
    mGpu.lightTree = mLightTreeRing[mRingSlot];
    const std::vector<u32> firstLights = FirstLightIndices(mDesc);
    WriteBuffer(mDevice, mInstanceLightRing[mRingSlot], firstLights.data(), firstLights.size() * sizeof(u32));
    mGpu.instanceLights = mInstanceLightRing[mRingSlot];
    mGpu.lightCount = totalPower > 0.0 ? u32(lights.size()) : 0u;
    mGpu.lightPower = float(totalPower);

    // The sky table is only rebuilt when the sky changes; every slot gets a copy.
    if (!mSkyTableBuilt || !(mSkyTableFor == mDesc.sky))
    {
        mSkyTable = BuildSkyTable(mDesc.sky, mDesc.environment);
        mSkyTableFor = mDesc.sky;
        mSkyTableBuilt = true;
        DestroyAerial();
        if (!mSkyTable.aerialInscatter.empty())
        {
            auto upload = [&](const std::vector<u16>& texels, const char* what)
            {
                return CreateTextureWithData(mDevice,
                                             {.dimension = rhi::TextureDimension::Tex3D,
                                              .format = rhi::Format::RGBA16_FLOAT,
                                              .width = mSkyTable.aerialWidth,
                                              .height = mSkyTable.aerialHeight,
                                              .depthOrArraySize = mSkyTable.data.aerialSlices,
                                              .debugName = std::format("{}.{}", mDesc.name, what)},
                                             texels.data());
            };
            mGpu.aerialInscatter = upload(mSkyTable.aerialInscatter, "AerialInscatter");
            mGpu.aerialTransmittance = upload(mSkyTable.aerialTransmittance, "AerialTransmittance");
        }
    }
    mSkyTable.data.uniformRadiance = mDesc.environment;
    const std::vector<float> placeholder(3, 0.0f);
    const std::vector<float>& skyTable = mSkyTable.radiance.empty() ? placeholder : mSkyTable.radiance;
    ensure(mSkyRing, mSkyRingCapacity, skyTable.size(), sizeof(float), "SkyTable");
    WriteBuffer(mDevice, mSkyRing[mRingSlot], skyTable.data(), skyTable.size() * sizeof(float));
    mGpu.skyTable = mSkyRing[mRingSlot];
    mGpu.sky = mSkyTable.data;

    // The clouds are baked and uploaded only when they change (the old resources go once the GPU
    // is done with them).
    if (!mCloudsBuilt || !(mCloudsFor == mDesc.clouds))
    {
        DestroyClouds();
        const CloudVolume volume = BuildCloudVolume(mDesc.clouds);
        mGpu.clouds = volume.data;
        if (volume.data.enabled != 0)
        {
            const gpu::CloudData& c = volume.data;
            mGpu.cloudBase = CreateTextureWithData(mDevice,
                                                   {.dimension = rhi::TextureDimension::Tex3D,
                                                    .format = rhi::Format::R8_UNORM,
                                                    .width = c.gridX,
                                                    .height = c.gridZ,
                                                    .depthOrArraySize = c.gridY + 2,
                                                    .debugName = std::format("{}.CloudBase", mDesc.name)},
                                                   volume.base.data());
            mGpu.cloudDetail = CreateTextureWithData(mDevice,
                                                     {.dimension = rhi::TextureDimension::Tex3D,
                                                      .format = rhi::Format::R8_UNORM,
                                                      .width = volume.detailSize,
                                                      .height = volume.detailSize,
                                                      .depthOrArraySize = volume.detailSize,
                                                      .debugName = std::format("{}.CloudDetail", mDesc.name)},
                                                     volume.detail.data());
            mGpu.cloudMajorants =
                CreateBufferWithData(mDevice,
                                     {.stride = sizeof(float),
                                      .usage = BufferUsage::ShaderResource,
                                      .debugName = std::format("{}.CloudMajorants", mDesc.name)},
                                     volume.majorants.data(), volume.majorants.size() * sizeof(float));
        }
        mCloudsFor = mDesc.clouds;
        mCloudsBuilt = true;
    }
}

void Scene::DestroyAerial()
{
    if (mGpu.aerialInscatter.IsValid())
    {
        mDevice.DestroyTexture(mGpu.aerialInscatter);
        mDevice.DestroyTexture(mGpu.aerialTransmittance);
    }
    mGpu.aerialInscatter = {};
    mGpu.aerialTransmittance = {};
}

void Scene::DestroyClouds()
{
    if (mGpu.cloudBase.IsValid())
    {
        mDevice.DestroyTexture(mGpu.cloudBase);
        mDevice.DestroyTexture(mGpu.cloudDetail);
        mDevice.DestroyBuffer(mGpu.cloudMajorants);
    }
    mGpu.cloudBase = {};
    mGpu.cloudDetail = {};
    mGpu.cloudMajorants = {};
}

} // namespace air::renderer
