#include "ReferenceScene.h"

#include "air/core/Assert.h"
#include "air/features/pathtracer/Bsdf.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace air::pt::reference
{

Bvh::Bvh(std::vector<Triangle> triangles) : mTriangles(std::move(triangles))
{
    AIR_CHECK(!mTriangles.empty(), "reference renderer: scene has no triangles");
    mNodes.reserve(mTriangles.size() * 2);
    mNodes.emplace_back();
    Build(0, 0, u32(mTriangles.size()));
}

bool Bvh::Intersect(float3 origin, float3 direction, float tMax, bool anyHit, HitInfo& hit, float tMin, u32 skip) const
{
    const float3 inv(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
    hit.hit = false;
    hit.t = tMax;
    u32 stack[64];
    u32 top = 0;
    stack[top++] = 0;
    while (top > 0)
    {
        const BvhNode& node = mNodes[stack[--top]];
        if (!HitsBox(node.bounds, origin, inv, hit.t))
        {
            continue;
        }
        if (node.count > 0)
        {
            for (u32 i = node.first; i < node.first + node.count; ++i)
            {
                if (mTriangles[i].instance != skip && IntersectTriangle(mTriangles[i], origin, direction, tMin, hit) &&
                    anyHit)
                {
                    return true;
                }
            }
        }
        else
        {
            AIR_DCHECK(top + 2 <= 64, "BVH too deep");
            stack[top++] = node.first;
            stack[top++] = node.first + 1;
        }
    }
    return hit.hit;
}

void Bvh::Build(u32 nodeIndex, u32 begin, u32 end)
{
    Bounds bounds, centroids;
    for (u32 i = begin; i < end; ++i)
    {
        const Triangle& t = mTriangles[i];
        bounds.Grow(t.p0);
        bounds.Grow(t.p0 + t.e1);
        bounds.Grow(t.p0 + t.e2);
        centroids.Grow(t.p0 + (t.e1 + t.e2) * (1.0f / 3.0f));
    }
    mNodes[nodeIndex].bounds = bounds;
    const float3 extent = centroids.hi - centroids.lo;
    if (end - begin <= 4 || std::max(extent.x, std::max(extent.y, extent.z)) <= 0.0f)
    {
        mNodes[nodeIndex].first = begin;
        mNodes[nodeIndex].count = end - begin;
        return;
    }
    const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : extent.y >= extent.z ? 1 : 2;
    auto key = [axis](const Triangle& t)
    {
        const float3 c = t.p0 + (t.e1 + t.e2) * (1.0f / 3.0f);
        return axis == 0 ? c.x : axis == 1 ? c.y : c.z;
    };
    const u32 mid = begin + (end - begin) / 2;
    std::nth_element(mTriangles.begin() + begin, mTriangles.begin() + mid, mTriangles.begin() + end,
                     [&](const Triangle& a, const Triangle& b) { return key(a) < key(b); });
    const u32 left = u32(mNodes.size());
    mNodes.emplace_back();
    mNodes.emplace_back();
    mNodes[nodeIndex].first = left;
    mNodes[nodeIndex].count = 0;
    Build(left, begin, mid);
    Build(left + 1, mid, end);
}

bool Bvh::HitsBox(const Bounds& b, float3 origin, float3 inv, float tMax)
{
    const float3 t0 = (b.lo - origin) * inv;
    const float3 t1 = (b.hi - origin) * inv;
    const float3 tNear = min(t0, t1);
    const float3 tFar = max(t0, t1);
    const float enter = std::max(std::max(tNear.x, tNear.y), std::max(tNear.z, 0.0f));
    const float exit = std::min(std::min(tFar.x, tFar.y), std::min(tFar.z, tMax));
    return enter <= exit;
}

// Moller-Trumbore. Barycentrics are those of vertices 1 and 2, like DXR.
bool Bvh::IntersectTriangle(const Triangle& tri, float3 origin, float3 direction, float tMin, HitInfo& hit)
{
    const float3 p = cross(direction, tri.e2);
    const float det = dot(tri.e1, p);
    if (std::abs(det) < 1e-12f)
    {
        return false;
    }
    const float invDet = 1.0f / det;
    const float3 s = origin - tri.p0;
    const float u = dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }
    const float3 q = cross(s, tri.e1);
    const float v = dot(direction, q) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }
    const float t = dot(tri.e2, q) * invDet;
    if (t <= tMin || t >= hit.t)
    {
        return false;
    }
    hit.hit = true;
    hit.t = t;
    hit.instance = tri.instance;
    hit.primitive = tri.primitive;
    hit.barycentrics = float2(u, v);
    return true;
}

SceneTables BuildSceneTables(const renderer::SceneDesc& scene)
{
    SceneTables tables;
    tables.instanceLights = renderer::FirstLightIndices(scene);
    std::vector<Triangle> triangles;
    for (size_t i = 0; i < scene.instances.size(); ++i)
    {
        const renderer::Instance& in = scene.instances[i];
        tables.instances.push_back(renderer::MakeInstanceData(in, in.mesh, in.mesh));
        const renderer::Mesh& mesh = scene.meshes[in.mesh];
        for (u32 t = 0; t + 2 < mesh.indices.size(); t += 3)
        {
            const float3 p0 = renderer::TransformPoint(in.transform, mesh.vertices[mesh.indices[t]].position);
            const float3 p1 = renderer::TransformPoint(in.transform, mesh.vertices[mesh.indices[t + 1]].position);
            const float3 p2 = renderer::TransformPoint(in.transform, mesh.vertices[mesh.indices[t + 2]].position);
            triangles.push_back({p0, p1 - p0, p2 - p0, u32(i), t / 3});
        }
    }
    for (const renderer::Mesh& mesh : scene.meshes)
    {
        tables.vertices.emplace_back(mesh.vertices);
        tables.indices.emplace_back(mesh.indices);
    }
    for (const renderer::Material& m : scene.materials)
    {
        tables.materials.push_back(m.ToGpu());
    }
    tables.lights = renderer::BuildLightTriangles(scene, &tables.lightPower);
    tables.lightTree = renderer::BuildLightTree(tables.lights);
    tables.lightGroups = renderer::BuildLightGroups(tables.lights, scene);
    if (tables.lights.empty())
    {
        tables.lights.emplace_back(); // like the GPU list: NEE reads entry 0 when no light is chosen
    }
    tables.lut = GetBsdfLut();
    tables.sky = renderer::BuildSkyTable(scene.sky, scene.environment);
    tables.clouds = renderer::BuildCloudVolume(scene.clouds);
    for (const renderer::Volume& volume : scene.volumes)
    {
        renderer::VolumeResources built = renderer::BuildVolume(volume);
        if (built.data.enabled != 0)
        {
            built.data.density = u32(tables.volumes.size());
            built.data.majorants = u32(tables.volumes.size());
            tables.volumes.push_back(std::move(built));
        }
    }
    tables.bvh = std::make_unique<Bvh>(std::move(triangles));
    return tables;
}

CameraData ToCameraData(const renderer::Camera& camera, float aspect)
{
    CameraData out;
    out.position = {camera.position, std::tan(camera.verticalFov * 0.5f)};
    out.forward = {camera.Forward(), aspect};
    out.right = {camera.Right(), 0.0f};
    out.up = {camera.Up(), 0.0f};
    return out;
}

ReferenceLighting LightingFor(const SceneTables& tables, const renderer::Camera& camera,
                              const ReferenceSettings& settings)
{
    ReferenceLighting out;
    out.sky = tables.sky.data;
    out.exposure = settings.applyExposure ? renderer::ExposureScale(camera.ev100) : 1.0f;
    if (settings.furnace)
    {
        // Same convention as the GPU path tracer: a uniform sky mapped to 0.18, no sun.
        const float3 uniform = out.sky.width == 0 ? out.sky.uniformRadiance : float3(1.0f);
        const float environment = dot(uniform, float3(0.2126f, 0.7152f, 0.0722f));
        out.sky.width = 0;
        out.sky.height = 0;
        out.sky.uniformRadiance = environment > 0.0f ? uniform : float3(1.0f);
        out.sky.sunRadiance = float3(0.0f);
        out.sky.aerialDistance = 0.0f;
        out.exposure = settings.applyExposure ? 0.18f / (environment > 0.0f ? environment : 1.0f) : 1.0f;
    }
    return out;
}

} // namespace air::pt::reference
