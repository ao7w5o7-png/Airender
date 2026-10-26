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
    gpu::InstanceData data{};
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
    u32 group = 0;
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
        const u32 thisGroup = group++;
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3)
        {
            gpu::LightTriangle light{};
            const float3 p0 = TransformPoint(in.transform, mesh.vertices[mesh.indices[t]].position);
            const float3 p1 = TransformPoint(in.transform, mesh.vertices[mesh.indices[t + 1]].position);
            const float3 p2 = TransformPoint(in.transform, mesh.vertices[mesh.indices[t + 2]].position);
            light.p0 = p0;
            light.e1 = p1 - p0;
            light.e2 = p2 - p0;
            light.area = 0.5f * hlsl::length(hlsl::cross(light.e1, light.e2));
            light.radiance = material.emission;
            light.group = thisGroup;
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

std::vector<gpu::LightGroup> BuildLightGroups(const std::vector<gpu::LightTriangle>& lights, const SceneDesc& desc)
{
    std::vector<u32> instances; // the emissive ones, one per group
    for (size_t i = 0; i < desc.instances.size(); ++i)
    {
        if (EmittedLuminance(desc, desc.instances[i]) > 0.0f)
        {
            instances.push_back(u32(i));
        }
    }
    std::vector<gpu::LightGroup> groups;
    for (size_t first = 0; first < lights.size();)
    {
        size_t end = first;
        double area = 0.0;
        double centre[3] = {};
        double radiance[3] = {};
        float3 normalArea(0.0f);
        while (end < lights.size() && lights[end].group == lights[first].group)
        {
            const gpu::LightTriangle& t = lights[end++];
            const float3 c = t.p0 + (t.e1 + t.e2) * (1.0f / 3.0f);
            area += t.area;
            centre[0] += double(c.x) * t.area;
            centre[1] += double(c.y) * t.area;
            centre[2] += double(c.z) * t.area;
            radiance[0] += double(t.radiance.x) * t.area;
            radiance[1] += double(t.radiance.y) * t.area;
            radiance[2] += double(t.radiance.z) * t.area;
            normalArea += hlsl::cross(t.e1, t.e2) * 0.5f;
        }
        gpu::LightGroup g{};
        if (area > 0.0)
        {
            g.centre = float3(float(centre[0] / area), float(centre[1] / area), float(centre[2] / area));
            g.radiance = float3(float(radiance[0] / area), float(radiance[1] / area), float(radiance[2] / area));
        }
        else
        {
            g.centre = lights[first].p0;
            g.radiance = lights[first].radiance;
        }
        g.area = float(area);
        g.normalArea = normalArea;
        g.firstLight = u32(first);
        g.lightCount = u32(end - first);
        for (size_t i = first; i < end; ++i)
        {
            const gpu::LightTriangle& t = lights[i];
            for (const float3 p : {t.p0, t.p0 + t.e1, t.p0 + t.e2})
            {
                g.radius = std::max(g.radius, hlsl::length(p - g.centre));
            }
        }
        g.power = hlsl::dot(g.radiance, float3(0.2126f, 0.7152f, 0.0722f)) * g.area * std::numbers::pi_v<float>;
        g.instance = lights[first].group < instances.size() ? instances[lights[first].group] : gpu::kInvalidIndex;
        groups.push_back(g);
        first = end;
    }
    return groups;
}

Scene::Scene(rhi::Device& device, SceneDesc desc) : mDevice(device), mDesc(std::move(desc))
{
    const std::string problem = Validate(mDesc);
    AIR_CHECK(problem.empty(), "scene '{}': {}", mDesc.name, problem);
    SceneBounds(mDesc, mGpu.boundsMin, mGpu.boundsMax);
    Upload();
    UploadTextures();
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
    mLightGroupRing.assign(slots, {});
    mLightGroupRingCapacity.assign(slots, 0);
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
    for (const auto* ring : {&mLightRing, &mLightTreeRing, &mLightGroupRing, &mSkyRing})
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
    for (LoadedVolume& volume : mVolumes)
    {
        DestroyVolume(volume);
    }
    for (rhi::BufferHandle table : {mGpu.volumeTable, mGpu.volumeMarchTable, mGpu.volumeLevelTable})
    {
        if (table.IsValid())
        {
            mDevice.DestroyBuffer(table);
        }
    }
    DestroyAerial();
    for (rhi::TextureHandle texture : mTextures)
    {
        mDevice.DestroyTexture(texture);
    }
    mDevice.DestroyBuffer(mGpu.textureSetTable);
    mDevice.DestroyBuffer(mInstances);
    mDevice.DestroyAccelStruct(mGpu.tlas);
}

void Scene::UploadTextures()
{
    std::vector<gpu::TextureSetData> table;
    for (const MaterialTextures& t : mDesc.textureSets)
    {
        auto make = [&](const std::vector<u32>& texels, rhi::Format format, const char* kind)
        {
            const rhi::TextureHandle texture =
                CreateTextureWithData(mDevice,
                                      {.format = format,
                                       .width = t.width,
                                       .height = t.height,
                                       .usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopyDst,
                                       .debugName = std::format("{}.{}", t.name, kind)},
                                      texels.data());
            AIR_CHECK(texture.IsValid(), "out of device memory uploading the textures '{}'", t.name);
            mTextures.push_back(texture);
            mGeometryBytes += texels.size() * sizeof(u32);
            return mDevice.GetBindlessSRV(texture);
        };
        gpu::TextureSetData data{};
        data.color = make(t.color, rhi::Format::RGBA8_SRGB, "Color");
        data.surface = make(t.surface, rhi::Format::RGBA8_UNORM, "Surface");
        table.push_back(data);
    }
    if (table.empty())
    {
        table.push_back({gpu::kInvalidIndex, gpu::kInvalidIndex, 0, 0});
    }
    mGpu.textureSetTable = CreateBufferWithData(mDevice,
                                                {.stride = sizeof(gpu::TextureSetData),
                                                 .usage = BufferUsage::ShaderResource,
                                                 .debugName = mDesc.name + ".TextureSets"},
                                                table.data(), table.size() * sizeof(gpu::TextureSetData));
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

void Scene::SetVolumes(std::vector<Volume> volumes)
{
    AIR_CHECK(volumes.size() <= gpu::kMaxVolumes, "SetVolumes: {} volumes, at most {}", volumes.size(),
              gpu::kMaxVolumes);
    if (!(mDesc.volumes == volumes))
    {
        mDesc.volumes = std::move(volumes);
        mDirty = true;
    }
}

void Scene::SetVolume(u32 index, const Volume& volume)
{
    AIR_CHECK(index < mDesc.volumes.size(), "SetVolume: index {} out of range", index);
    if (!(mDesc.volumes[index] == volume))
    {
        mDesc.volumes[index] = volume;
        mDirty = true;
    }
}

std::string_view Scene::VolumeError(size_t index) const
{
    return index < mVolumes.size() ? std::string_view(mVolumes[index].error) : std::string_view();
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
    mGpu.lightGroups = BuildLightGroups(lights, mDesc);
    mGpu.lightTriangles = lights;
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
    const gpu::LightGroup noGroup{};
    const size_t groups = std::max<size_t>(mGpu.lightGroups.size(), 1);
    ensure(mLightGroupRing, mLightGroupRingCapacity, groups, sizeof(gpu::LightGroup), "LightGroups");
    WriteBuffer(mDevice, mLightGroupRing[mRingSlot], mGpu.lightGroups.empty() ? &noGroup : mGpu.lightGroups.data(),
                groups * sizeof(gpu::LightGroup));
    mGpu.lightGroupTable = mLightGroupRing[mRingSlot];
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

    // The volumes likewise, each on its own.
    const bool volumesChanged =
        !mVolumesBuilt || mVolumes.size() != mDesc.volumes.size() ||
        !std::equal(mVolumes.begin(), mVolumes.end(), mDesc.volumes.begin(),
                    [](const LoadedVolume& loaded, const Volume& desc) { return loaded.desc == desc; });
    if (volumesChanged)
    {
        UpdateVolumes();
    }
}

void Scene::UpdateVolumes()
{
    AIR_PROFILE_SCOPE("Scene.UpdateVolumes");
    // A volume that did not change keeps its resources; one that did is built again, from the voxels
    // of an earlier volume with the same source if there is one, else from its file or plume.
    std::vector<LoadedVolume> previous = std::move(mVolumes);
    mVolumes.clear();
    mVolumes.resize(mDesc.volumes.size());
    std::vector<bool> reused(mDesc.volumes.size(), false);
    std::vector<bool> taken(previous.size(), false);
    for (size_t i = 0; i < mDesc.volumes.size(); ++i)
    {
        for (size_t j = 0; j < previous.size(); ++j)
        {
            if (!taken[j] && previous[j].desc == mDesc.volumes[i])
            {
                mVolumes[i] = std::move(previous[j]);
                previous[j] = {}; // its handles now belong to mVolumes[i]
                taken[j] = true;
                reused[i] = true;
                break;
            }
        }
    }
    auto sameSource = [](const Volume& a, const Volume& b)
    { return a.file == b.file && a.grid == b.grid && a.maxResolution == b.maxResolution && a.seed == b.seed; };
    // The voxels of a volume with this source, kept or being built.
    auto findGrid = [&](const Volume& desc) -> std::shared_ptr<const VolumeGrid>
    {
        for (const std::vector<LoadedVolume>* list : {&mVolumes, &previous})
        {
            for (const LoadedVolume& volume : *list)
            {
                if (volume.grid && sameSource(volume.desc, desc))
                {
                    return volume.grid;
                }
            }
        }
        return nullptr;
    };
    auto keptBytes = [&]
    {
        std::vector<const VolumeGrid*> distinct;
        size_t bytes = 0;
        for (const LoadedVolume& volume : mVolumes)
        {
            if (volume.grid && std::find(distinct.begin(), distinct.end(), volume.grid.get()) == distinct.end())
            {
                distinct.push_back(volume.grid.get());
                bytes += volume.grid->density.size() * sizeof(float);
            }
        }
        return bytes;
    };
    for (size_t i = 0; i < mDesc.volumes.size(); ++i)
    {
        if (reused[i])
        {
            continue;
        }
        const Volume& desc = mDesc.volumes[i];
        LoadedVolume& out = mVolumes[i];
        out.desc = desc;
        if (!desc.enabled)
        {
            continue;
        }
        std::shared_ptr<const VolumeGrid> grid = findGrid(desc);
        if (!grid)
        {
            std::optional<VolumeGrid> loaded = LoadVolumeGrid(desc, &out.error);
            if (!loaded)
            {
                AIR_LOG(Error, "Volume", "{}", out.error);
                continue;
            }
            grid = std::make_shared<const VolumeGrid>(std::move(*loaded));
        }
        const VolumeResources built = BuildVolume(desc, *grid);
        const gpu::VolumeData& v = built.data;
        out.density = CreateTextureWithData(mDevice,
                                            {.dimension = rhi::TextureDimension::Tex3D,
                                             .format = rhi::Format::R16_FLOAT,
                                             .width = v.sizeX + 2,
                                             .height = v.sizeY + 2,
                                             .depthOrArraySize = v.sizeZ + 2,
                                             .debugName = std::format("{}.Volume{}.Density", mDesc.name, i)},
                                            built.density.data());
        out.majorants = CreateBufferWithData(mDevice,
                                             {.stride = sizeof(float),
                                              .usage = BufferUsage::ShaderResource,
                                              .debugName = std::format("{}.Volume{}.Majorants", mDesc.name, i)},
                                             built.majorants.data(), built.majorants.size() * sizeof(float));
        out.data = v;
        out.data.density = mDevice.GetBindlessSRV(out.density);
        out.data.majorants = mDevice.GetBindlessSRV(out.majorants);
        // The bricks; empty tables get one element so that every buffer can be bound.
        auto upload = [&](const std::vector<u32>& words, std::string name)
        {
            const u32 none = 0;
            const rhi::BufferHandle buffer = CreateBufferWithData(
                mDevice, {.stride = sizeof(u32), .usage = BufferUsage::ShaderResource, .debugName = std::move(name)},
                words.empty() ? &none : words.data(), std::max<size_t>(words.size(), 1) * sizeof(u32));
            out.brickBuffers.push_back(buffer);
            return mDevice.GetBindlessSRV(buffer);
        };
        for (u32 level = 0; level < gpu::kVolumeLevels; ++level)
        {
            const VolumeBricks::Level& source = built.bricks.levels[level];
            gpu::VolumeBrickLevel& l = out.levels[level];
            l = source.data;
            const std::string name = std::format("{}.Volume{}.Level{}", mDesc.name, i, level);
            l.indirection = upload(source.indirection, name + ".Indirection");
            l.distance = upload(source.distance, name + ".Distance");
            l.bricks = upload(source.bricks, name + ".Bricks");
            l.ranges = upload(source.ranges, name + ".Ranges");
        }
        const VolumeBricks& bricks = built.bricks;
        out.ao = CreateTextureWithData(mDevice,
                                       {.dimension = rhi::TextureDimension::Tex3D,
                                        .format = rhi::Format::R8_UNORM,
                                        .width = bricks.data.aoX,
                                        .height = bricks.data.aoY,
                                        .depthOrArraySize = bricks.data.aoZ,
                                        .debugName = std::format("{}.Volume{}.Occlusion", mDesc.name, i)},
                                       bricks.ao.data());
        out.march = bricks.data;
        out.march.ao = mDevice.GetBindlessSRV(out.ao);
        if (findGrid(desc) == grid || keptBytes() + grid->density.size() * sizeof(float) <= (size_t(512) << 20))
        {
            out.grid = std::move(grid);
        }
    }
    for (LoadedVolume& old : previous)
    {
        DestroyVolume(old);
    }

    // The tables the GPU reads: the volumes that loaded.
    mGpu.volumes.clear();
    mGpu.volumeMarch.clear();
    mGpu.volumeLevels.clear();
    for (const LoadedVolume& volume : mVolumes)
    {
        if (volume.data.enabled != 0)
        {
            mGpu.volumes.push_back(volume.data);
            gpu::VolumeMarchData march = volume.march;
            march.levels = u32(mGpu.volumeLevels.size());
            mGpu.volumeMarch.push_back(march);
            mGpu.volumeLevels.insert(mGpu.volumeLevels.end(), std::begin(volume.levels), std::end(volume.levels));
        }
    }
    for (rhi::BufferHandle* table : {&mGpu.volumeTable, &mGpu.volumeMarchTable, &mGpu.volumeLevelTable})
    {
        if (table->IsValid())
        {
            mDevice.DestroyBuffer(*table);
            *table = {};
        }
    }
    if (!mGpu.volumes.empty())
    {
        auto table = [&](const void* data, size_t count, u32 stride, const char* what)
        {
            return CreateBufferWithData(mDevice,
                                        {.stride = stride,
                                         .usage = BufferUsage::ShaderResource,
                                         .debugName = std::format("{}.{}", mDesc.name, what)},
                                        data, count * stride);
        };
        mGpu.volumeTable = table(mGpu.volumes.data(), mGpu.volumes.size(), sizeof(gpu::VolumeData), "Volumes");
        mGpu.volumeMarchTable =
            table(mGpu.volumeMarch.data(), mGpu.volumeMarch.size(), sizeof(gpu::VolumeMarchData), "VolumeMarch");
        mGpu.volumeLevelTable =
            table(mGpu.volumeLevels.data(), mGpu.volumeLevels.size(), sizeof(gpu::VolumeBrickLevel), "VolumeLevels");
    }
    mVolumesBuilt = true;
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

void Scene::DestroyVolume(LoadedVolume& volume)
{
    if (volume.density.IsValid())
    {
        mDevice.DestroyTexture(volume.density);
    }
    if (volume.majorants.IsValid())
    {
        mDevice.DestroyBuffer(volume.majorants);
    }
    for (rhi::BufferHandle buffer : volume.brickBuffers)
    {
        mDevice.DestroyBuffer(buffer);
    }
    if (volume.ao.IsValid())
    {
        mDevice.DestroyTexture(volume.ao);
    }
    volume.density = {};
    volume.majorants = {};
    volume.brickBuffers.clear();
    volume.ao = {};
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
