// CPU host for Integrator.hlsli: scene tables in std::vectors and a median-split BVH.

#include "air/features/pathtracer/ReferenceRenderer.h"

#include "air/core/Assert.h"
#include "air/features/pathtracer/Bsdf.h"
#include "air/renderer/shader/SceneShared.h"

#include "PathTracerShared.h"

#include <algorithm>
#include <cmath>
#include <span>

namespace air::pt
{
namespace
{

using namespace air::hlsl;
using namespace air::gpu;
using namespace air::pt::gpu;

// --- BVH over world-space triangles --------------------------------------------------------------

struct Triangle
{
    float3 p0, e1, e2;
    u32 instance;
    u32 primitive;
};

struct Bounds
{
    float3 lo = float3(1e30f);
    float3 hi = float3(-1e30f);
    void Grow(float3 p)
    {
        lo = min(lo, p);
        hi = max(hi, p);
    }
};

struct BvhNode
{
    Bounds bounds;
    u32 first = 0; // leaf: first triangle; inner: index of the left child (right = first + 1)
    u32 count = 0; // > 0: leaf
};

class Bvh
{
public:
    explicit Bvh(std::vector<Triangle> triangles) : mTriangles(std::move(triangles))
    {
        AIR_CHECK(!mTriangles.empty(), "reference renderer: scene has no triangles");
        mNodes.reserve(mTriangles.size() * 2);
        mNodes.emplace_back();
        Build(0, 0, u32(mTriangles.size()));
    }

    // Closest hit with t in (0, tMax). anyHit: stop at the first hit.
    bool Intersect(float3 origin, float3 direction, float tMax, bool anyHit, HitInfo& hit) const
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
                    if (IntersectTriangle(mTriangles[i], origin, direction, hit) && anyHit)
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

private:
    void Build(u32 nodeIndex, u32 begin, u32 end)
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

    static bool HitsBox(const Bounds& b, float3 origin, float3 inv, float tMax)
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
    static bool IntersectTriangle(const Triangle& tri, float3 origin, float3 direction, HitInfo& hit)
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
        if (t <= 0.0f || t >= hit.t)
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

    std::vector<Triangle> mTriangles;
    std::vector<BvhNode> mNodes;
};

// --- Integrator host -----------------------------------------------------------------------------

struct SceneTables
{
    std::vector<InstanceData> instances; // vertexBuffer / indexBuffer = mesh index
    std::vector<std::span<const Vertex>> vertices;
    std::vector<std::span<const u32>> indices;
    std::vector<MaterialData> materials;
    std::vector<LightTriangle> lights;
    std::span<const float> lut;
    const Bvh* bvh = nullptr;
};

// The shader code becomes member functions; the hooks read the tables above.
struct Kernel
{
    const SceneTables& tables;

    float BsdfLutLoad(uint index) const { return tables.lut[index]; }
    InstanceData LoadInstance(uint instance) const { return tables.instances[instance]; }
    uint LoadIndex(InstanceData instance, uint index) const { return tables.indices[instance.indexBuffer][index]; }
    Vertex LoadVertex(InstanceData instance, uint index) const { return tables.vertices[instance.vertexBuffer][index]; }
    MaterialData LoadMaterial(uint material) const { return tables.materials[material]; }
    LightTriangle LoadLight(uint light) const { return tables.lights[light]; }
    HitInfo TraceClosest(float3 origin, float3 direction) const
    {
        HitInfo hit{};
        tables.bvh->Intersect(origin, direction, 1e30f, false, hit);
        return hit;
    }
    bool TraceVisible(float3 origin, float3 direction, float tMax) const
    {
        HitInfo hit{};
        return !tables.bvh->Intersect(origin, direction, tMax, true, hit);
    }

// NOLINTBEGIN(readability-identifier-naming)
#include "Integrator.hlsli"
    // NOLINTEND(readability-identifier-naming)
};

CameraData ToCameraData(const renderer::Camera& camera, float aspect)
{
    CameraData out;
    out.position = {camera.position, std::tan(camera.verticalFov * 0.5f)};
    out.forward = {camera.Forward(), aspect};
    out.right = {camera.Right(), 0.0f};
    out.up = {camera.Up(), 0.0f};
    return out;
}

} // namespace

Image RenderReference(const renderer::SceneDesc& scene, const renderer::Camera& camera,
                      const ReferenceSettings& settings)
{
    const std::string problem = renderer::Validate(scene);
    AIR_CHECK(problem.empty(), "reference renderer: {}", problem);

    SceneTables tables;
    const std::vector<u32> firstLights = renderer::FirstLightIndices(scene);
    std::vector<Triangle> triangles;
    for (size_t i = 0; i < scene.instances.size(); ++i)
    {
        const renderer::Instance& in = scene.instances[i];
        tables.instances.push_back(renderer::MakeInstanceData(in, firstLights[i], in.mesh, in.mesh));
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
    double lightPower = 0.0;
    tables.lights = renderer::BuildLightTriangles(scene, &lightPower);
    tables.lut = GetBsdfLut();
    const Bvh bvh(std::move(triangles));
    tables.bvh = &bvh;

    Kernel::IntegratorParams params;
    params.environment = scene.environment;
    params.lightCount = lightPower > 0.0 ? u32(tables.lights.size()) : 0u;
    params.maxBounces = settings.maxBounces;
    params.flags = (settings.furnace ? kFlagFurnace : 0u) | (settings.nextEventEstimation ? 0u : kFlagNoNee);
    float exposure = settings.applyExposure ? renderer::ExposureScale(camera.ev100) : 1.0f;
    if (settings.furnace)
    {
        // Same convention as the GPU path tracer: the environment maps to 0.18.
        const float environment = dot(scene.environment, float3(0.2126f, 0.7152f, 0.0722f));
        params.environment = environment > 0.0f ? scene.environment : float3(1.0f);
        exposure = settings.applyExposure ? 0.18f / (environment > 0.0f ? environment : 1.0f) : 1.0f;
    }

    const float2 size(float(settings.width), float(settings.height));
    const CameraData cameraData = ToCameraData(camera, size.x / size.y);
    Image image(settings.width, settings.height);
    auto renderRow = [&](u32 y)
    {
        Kernel kernel{tables};
        for (u32 x = 0; x < settings.width; ++x)
        {
            // Same seeding as PathTracer.hlsl for frame 0.
            uint seed = kernel.Pcg(x + kernel.Pcg(y + kernel.Pcg(0u)));
            double sum[3] = {};
            for (u32 s = 0; s < settings.samplesPerPixel; ++s)
            {
                const float jx = kernel.Rand(seed);
                const float jy = kernel.Rand(seed);
                const float3 direction =
                    kernel.CameraRayDirection(cameraData, float2(float(x) + jx, float(y) + jy), size);
                const Kernel::PathSample sample = kernel.TracePath(camera.position, direction, params, seed);
                sum[0] += sample.radiance.x;
                sum[1] += sample.radiance.y;
                sum[2] += sample.radiance.z;
            }
            float* out = image.At(x, y);
            for (int c = 0; c < 3; ++c)
            {
                out[c] = float(sum[c] / settings.samplesPerPixel) * exposure;
            }
            out[3] = 1.0f;
        }
    };
    if (settings.tasks)
    {
        settings.tasks->ParallelFor(settings.height, renderRow);
    }
    else
    {
        for (u32 y = 0; y < settings.height; ++y)
        {
            renderRow(y);
        }
    }
    return image;
}

} // namespace air::pt
