// Camera math, mesh builders and the built-in scenes.

#include "air/renderer/Scene.h"

#include "SceneBuilders.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>

namespace air::renderer
{

using hlsl::cross;
using hlsl::dot;
using hlsl::normalize;

float3 Camera::Forward() const
{
    return {-std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch)};
}

float3 Camera::Right() const
{
    return {std::cos(yaw), 0.0f, -std::sin(yaw)};
}

float3 Camera::Up() const
{
    return cross(Right(), Forward());
}

float ExposureScale(float ev100)
{
    return 1.0f / (1.2f * std::exp2(ev100));
}

gpu::MaterialData Material::ToGpu() const
{
    gpu::MaterialData out{};
    out.baseColor = baseColor;
    out.roughness = roughness;
    out.metallic = metallic;
    const float luminance = dot(emissionColor, float3(0.2126f, 0.7152f, 0.0722f));
    out.emission = luminance > 0.0f && emissionNits > 0.0f ? emissionColor * (emissionNits / luminance) : float3(0.0f);
    return out;
}

Mesh MakeQuad(std::string name, float3 p0, float3 p1, float3 p2, float3 p3, float3 normal)
{
    Mesh mesh;
    mesh.name = std::move(name);
    const float3 n = normalize(normal);
    for (const float3& p : {p0, p1, p2, p3})
    {
        mesh.vertices.push_back({p, 0.0f, n, 0.0f});
    }
    mesh.vertices[1].u = mesh.vertices[2].u = 1.0f;
    mesh.vertices[2].v = mesh.vertices[3].v = 1.0f;
    if (dot(cross(p1 - p0, p2 - p0), n) >= 0.0f)
    {
        mesh.indices = {0, 1, 2, 0, 2, 3};
    }
    else
    {
        mesh.indices = {0, 2, 1, 0, 3, 2};
    }
    return mesh;
}

Mesh MakeBox(std::string name, float3 h)
{
    Mesh mesh;
    mesh.name = std::move(name);
    // For each face: normal and two tangent axes; corners = n*h +- t*h +- b*h.
    const float3 axes[6][3] = {
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}}, {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}, {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},  {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},
    };
    for (const auto& face : axes)
    {
        const float3 n = face[0] * h;
        const float3 t = face[1] * h;
        const float3 b = face[2] * h;
        Mesh quad = MakeQuad("", n - t - b, n + t - b, n + t + b, n - t + b, face[0]);
        const u32 base = u32(mesh.vertices.size());
        mesh.vertices.insert(mesh.vertices.end(), quad.vertices.begin(), quad.vertices.end());
        for (u32 index : quad.indices)
        {
            mesh.indices.push_back(base + index);
        }
    }
    return mesh;
}

Mesh MakeSphere(std::string name, float radius, u32 segments)
{
    Mesh mesh;
    mesh.name = std::move(name);
    const u32 rings = std::max(segments / 2, 2u);
    const u32 columns = std::max(segments, 3u);
    for (u32 r = 0; r <= rings; ++r)
    {
        const float theta = std::numbers::pi_v<float> * float(r) / float(rings);
        for (u32 c = 0; c <= columns; ++c)
        {
            const float phi = 2.0f * std::numbers::pi_v<float> * float(c) / float(columns);
            const float3 n(std::sin(theta) * std::cos(phi), std::cos(theta), -std::sin(theta) * std::sin(phi));
            mesh.vertices.push_back({n * radius, float(c) / float(columns), n, float(r) / float(rings)});
        }
    }
    const u32 stride = columns + 1;
    for (u32 r = 0; r < rings; ++r)
    {
        for (u32 c = 0; c < columns; ++c)
        {
            const u32 a = r * stride + c;
            const u32 b = a + stride;
            // Counter-clockwise seen from outside: cross(e1, e2) points outward.
            if (r != 0)
            {
                mesh.indices.insert(mesh.indices.end(), {a, b, a + 1});
            }
            if (r != rings - 1)
            {
                mesh.indices.insert(mesh.indices.end(), {a + 1, b, b + 1});
            }
        }
    }
    return mesh;
}

void SetTransform(Instance& instance, float3 translation, float yawRadians, float scale)
{
    const float c = std::cos(yawRadians) * scale;
    const float s = std::sin(yawRadians) * scale;
    const float rows[3][4] = {
        {c, 0.0f, s, translation.x},
        {0.0f, scale, 0.0f, translation.y},
        {-s, 0.0f, c, translation.z},
    };
    std::copy(&rows[0][0], &rows[0][0] + 12, &instance.transform[0][0]);
}

namespace detail
{

u32 AddMaterial(SceneDesc& scene, Material material)
{
    scene.materials.push_back(std::move(material));
    return u32(scene.materials.size() - 1);
}

u32 AddMesh(SceneDesc& scene, Mesh mesh)
{
    scene.meshes.push_back(std::move(mesh));
    return u32(scene.meshes.size() - 1);
}

Instance& AddInstance(SceneDesc& scene, std::string name, u32 mesh, u32 material)
{
    Instance instance;
    instance.name = std::move(name);
    instance.mesh = mesh;
    instance.material = material;
    scene.instances.push_back(instance);
    return scene.instances.back();
}

void SetBasis(Instance& instance, float3 x, float3 y, float3 z, float3 translation)
{
    const float rows[3][4] = {
        {x.x, y.x, z.x, translation.x},
        {x.y, y.y, z.y, translation.y},
        {x.z, y.z, z.z, translation.z},
    };
    std::copy(&rows[0][0], &rows[0][0] + 12, &instance.transform[0][0]);
}

void LookAt(Camera& camera, float3 eye, float3 target)
{
    const float3 d = normalize(target - eye);
    camera.position = eye;
    camera.yaw = std::atan2(-d.x, -d.z);
    camera.pitch = std::asin(std::clamp(d.y, -1.0f, 1.0f));
}

Mesh MakeDisc(std::string name, float3 center, float3 normal, float radius, u32 segments)
{
    Mesh mesh;
    mesh.name = std::move(name);
    const float3 n = normalize(normal);
    // Any tangent frame; the fan is wound so that cross(e1, e2) points along `normal`.
    const float3 helper = std::abs(n.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 t = normalize(cross(helper, n));
    const float3 b = cross(n, t);
    mesh.vertices.push_back({center, 0.5f, n, 0.5f});
    for (u32 i = 0; i < segments; ++i)
    {
        const float a = 2.0f * std::numbers::pi_v<float> * float(i) / float(segments);
        const float3 p = center + (t * std::cos(a) + b * std::sin(a)) * radius;
        mesh.vertices.push_back({p, 0.5f + 0.5f * std::cos(a), n, 0.5f + 0.5f * std::sin(a)});
    }
    for (u32 i = 0; i < segments; ++i)
    {
        const u32 a = 1 + i;
        const u32 c = 1 + (i + 1) % segments;
        if (dot(cross(mesh.vertices[a].position - center, mesh.vertices[c].position - center), n) >= 0.0f)
        {
            mesh.indices.insert(mesh.indices.end(), {0u, a, c});
        }
        else
        {
            mesh.indices.insert(mesh.indices.end(), {0u, c, a});
        }
    }
    return mesh;
}

float3 Hue(float h)
{
    const float r = std::clamp(std::abs(h * 6.0f - 3.0f) - 1.0f, 0.0f, 1.0f);
    const float g = std::clamp(2.0f - std::abs(h * 6.0f - 2.0f), 0.0f, 1.0f);
    const float b = std::clamp(2.0f - std::abs(h * 6.0f - 4.0f), 0.0f, 1.0f);
    return {r, g, b};
}

} // namespace detail

using namespace detail;

// A 2 m Cornell box (x, z in [-1, 1], y in [0, 2], open towards +z) lit by a 3000 nit ceiling
// panel, with a diffuse tall box and a rough gold sphere. Reflectances are those of the
// original Cornell measurements converted to linear sRGB.
SceneDesc MakeCornellBox()
{
    SceneDesc scene;
    scene.name = "cornell_box";
    const u32 white = AddMaterial(scene, {.name = "White", .baseColor = {0.725f, 0.71f, 0.68f}, .roughness = 0.9f});
    const u32 red = AddMaterial(scene, {.name = "Red", .baseColor = {0.63f, 0.065f, 0.05f}, .roughness = 0.9f});
    const u32 green = AddMaterial(scene, {.name = "Green", .baseColor = {0.14f, 0.45f, 0.091f}, .roughness = 0.9f});
    const u32 light = AddMaterial(scene, {.name = "Light",
                                          .baseColor = {0.78f, 0.78f, 0.78f},
                                          .roughness = 0.9f,
                                          .emissionColor = {1.0f, 0.85f, 0.65f},
                                          .emissionNits = 3000.0f});
    const u32 gold =
        AddMaterial(scene, {.name = "Gold", .baseColor = {1.0f, 0.71f, 0.29f}, .roughness = 0.3f, .metallic = 1.0f});

    const float3 up(0, 1, 0), down(0, -1, 0);
    AddInstance(scene, "Floor", AddMesh(scene, MakeQuad("Floor", {-1, 0, 1}, {1, 0, 1}, {1, 0, -1}, {-1, 0, -1}, up)),
                white);
    AddInstance(scene, "Ceiling",
                AddMesh(scene, MakeQuad("Ceiling", {-1, 2, 1}, {1, 2, 1}, {1, 2, -1}, {-1, 2, -1}, down)), white);
    AddInstance(scene, "BackWall",
                AddMesh(scene, MakeQuad("BackWall", {-1, 0, -1}, {1, 0, -1}, {1, 2, -1}, {-1, 2, -1}, {0, 0, 1})),
                white);
    AddInstance(scene, "LeftWall",
                AddMesh(scene, MakeQuad("LeftWall", {-1, 0, 1}, {-1, 0, -1}, {-1, 2, -1}, {-1, 2, 1}, {1, 0, 0})), red);
    AddInstance(scene, "RightWall",
                AddMesh(scene, MakeQuad("RightWall", {1, 0, 1}, {1, 0, -1}, {1, 2, -1}, {1, 2, 1}, {-1, 0, 0})), green);
    AddInstance(scene, "Light",
                AddMesh(scene, MakeQuad("Light", {-0.25f, 1.995f, 0.2f}, {0.25f, 1.995f, 0.2f}, {0.25f, 1.995f, -0.2f},
                                        {-0.25f, 1.995f, -0.2f}, down)),
                light);
    Instance& tall = AddInstance(scene, "TallBox", AddMesh(scene, MakeBox("TallBox", {0.3f, 0.6f, 0.3f})), white);
    SetTransform(tall, {-0.35f, 0.6f, -0.3f}, 0.3f);
    Instance& sphere = AddInstance(scene, "Sphere", AddMesh(scene, MakeSphere("Sphere", 0.35f, 64)), gold);
    SetTransform(sphere, {0.4f, 0.35f, 0.3f});

    scene.camera.position = {0.0f, 1.0f, 3.8f};
    scene.camera.verticalFov = 0.6911f; // 39.6 degrees
    scene.camera.ev100 = 8.0f;
    return scene;
}

// Two rows of spheres, roughness 0 .. 1: dielectric (front) and metal (back), in a uniform
// 1000 nit environment. Set pathtracer.furnace=true to make every material white: an
// energy-conserving BSDF then renders every sphere exactly as bright as the background.
SceneDesc MakeFurnaceScene()
{
    SceneDesc scene;
    scene.name = "furnace";
    const u32 sphere = AddMesh(scene, MakeSphere("Sphere", 0.2f, 64));
    for (u32 i = 0; i < 6; ++i)
    {
        const float roughness = float(i) / 5.0f;
        for (u32 metal = 0; metal < 2; ++metal)
        {
            const u32 material = AddMaterial(scene, {.name = std::format("{}{}", metal ? "Metal" : "Dielectric", i),
                                                     .baseColor = metal ? float3(0.95f, 0.64f, 0.54f) : float3(0.8f),
                                                     .roughness = roughness,
                                                     .metallic = metal ? 1.0f : 0.0f});
            Instance& instance = AddInstance(scene, scene.materials[material].name, sphere, material);
            SetTransform(instance, {-1.25f + 0.5f * float(i), 0.25f + 0.5f * float(metal), -0.5f * float(metal)});
        }
    }
    scene.environment = {1000.0f, 1000.0f, 1000.0f};
    scene.camera.position = {0.0f, 0.5f, 3.2f};
    scene.camera.pitch = -0.05f;
    scene.camera.verticalFov = 0.7f;
    scene.camera.ev100 = 9.0f;
    return scene;
}

namespace
{

struct BuiltinScene
{
    const char* name;
    SceneDesc (*make)();
};

constexpr BuiltinScene kBuiltinScenes[] = {
    {"cornell_box", MakeCornellBox},
    {"furnace", MakeFurnaceScene},
    {"veach_mis", detail::MakeVeachMis},
    {"material_grid", detail::MakeMaterialGrid},
    {"sphere_field", detail::MakeSphereField},
    {"many_lights", detail::MakeManyLights},
    {"door", detail::MakeDoor},
    {"terrain", detail::MakeTerrain},
    {"sunset", detail::MakeSunset},
    {"cumulus", detail::MakeCumulus},
    {"smoke", detail::MakeSmoke},
};

} // namespace

std::vector<std::string> BuiltinSceneNames()
{
    std::vector<std::string> names;
    for (const BuiltinScene& scene : kBuiltinScenes)
    {
        names.emplace_back(scene.name);
    }
    return names;
}

std::optional<SceneDesc> MakeBuiltinScene(std::string_view name)
{
    for (const BuiltinScene& scene : kBuiltinScenes)
    {
        if (name == scene.name)
        {
            return scene.make();
        }
    }
    return std::nullopt;
}

float HighestPoint(const SceneDesc& desc)
{
    float highest = -1e30f;
    for (const Instance& instance : desc.instances)
    {
        if (instance.mesh >= desc.meshes.size())
        {
            continue;
        }
        for (const gpu::Vertex& v : desc.meshes[instance.mesh].vertices)
        {
            highest = std::max(highest, TransformPoint(instance.transform, v.position).y);
        }
    }
    return highest;
}

void SceneBounds(const SceneDesc& desc, float3& lo, float3& hi)
{
    lo = float3(1e30f);
    hi = float3(-1e30f);
    for (const Instance& instance : desc.instances)
    {
        if (instance.mesh >= desc.meshes.size())
        {
            continue;
        }
        for (const gpu::Vertex& v : desc.meshes[instance.mesh].vertices)
        {
            const float3 p = TransformPoint(instance.transform, v.position);
            lo = hlsl::min(lo, p);
            hi = hlsl::max(hi, p);
        }
    }
}

std::string Validate(const SceneDesc& desc)
{
    for (const Mesh& mesh : desc.meshes)
    {
        if (mesh.vertices.empty() || mesh.indices.empty() || mesh.indices.size() % 3 != 0)
        {
            return std::format("mesh '{}' needs vertices and a multiple of 3 indices", mesh.name);
        }
        for (u32 index : mesh.indices)
        {
            if (index >= mesh.vertices.size())
            {
                return std::format("mesh '{}' index {} out of range", mesh.name, index);
            }
        }
    }
    for (const Material& m : desc.materials)
    {
        if (!(m.roughness >= 0.0f && m.roughness <= 1.0f) || !(m.metallic >= 0.0f && m.metallic <= 1.0f) ||
            !(m.emissionNits >= 0.0f))
        {
            return std::format("material '{}': roughness and metallic must be in [0, 1], emission >= 0", m.name);
        }
    }
    for (const Instance& instance : desc.instances)
    {
        if (instance.mesh >= desc.meshes.size() || instance.material >= desc.materials.size())
        {
            return std::format("instance '{}' references a missing mesh or material", instance.name);
        }
    }
    if (desc.instances.empty())
    {
        return "scene has no instances";
    }
    if (desc.volumes.size() > gpu::kMaxVolumes)
    {
        return std::format("{} volumes: at most {}", desc.volumes.size(), gpu::kMaxVolumes);
    }
    for (size_t i = 0; i < desc.volumes.size(); ++i)
    {
        const Volume& volume = desc.volumes[i];
        if (volume.enabled && (!(volume.extinction >= 0.0f) || !(volume.albedo >= 0.0f && volume.albedo <= 1.0f) ||
                               !(volume.anisotropy > -1.0f && volume.anisotropy < 1.0f) || !(volume.size >= 0.0f) ||
                               volume.maxResolution < 8 || volume.maxResolution > 2048))
        {
            return std::format("volume {}: needs extinction >= 0, albedo in [0, 1], anisotropy in (-1, 1), size >= 0 "
                               "and a maxResolution of 8 to 2048 voxels (the 3D texture limit)",
                               i);
        }
    }
    const Clouds& clouds = desc.clouds;
    if (clouds.enabled)
    {
        if (!(clouds.top >= clouds.bottom + 100.0f * clouds.scale) || !(clouds.scale > 0.0f) ||
            !(clouds.coverage >= 0.0f && clouds.coverage <= 1.0f) || !(clouds.extinction > 0.0f) ||
            !(clouds.albedo >= 0.0f && clouds.albedo <= 1.0f) || !(clouds.dropletDiameter > 0.0f))
        {
            return "clouds: need top >= bottom + 100 m * scale, coverage and albedo in [0, 1], extinction, "
                   "droplet size and scale > 0";
        }
        // Only camera rays look for clouds in front of a surface (Integrator.hlsli).
        const float highest = HighestPoint(desc);
        if (highest >= clouds.bottom)
        {
            return std::format("clouds: their base ({} m) must be above every instance (up to {} m)", clouds.bottom,
                               highest);
        }
    }
    return {};
}

} // namespace air::renderer
