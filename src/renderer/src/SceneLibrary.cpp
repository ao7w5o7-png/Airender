// Camera math, mesh builders and the built-in scenes.

#include "air/renderer/Scene.h"

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
    gpu::MaterialData out;
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

namespace
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

} // namespace

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

std::vector<std::string> BuiltinSceneNames()
{
    return {"cornell_box", "furnace"};
}

std::optional<SceneDesc> MakeBuiltinScene(std::string_view name)
{
    if (name == "cornell_box")
    {
        return MakeCornellBox();
    }
    if (name == "furnace")
    {
        return MakeFurnaceScene();
    }
    return std::nullopt;
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
    return {};
}

} // namespace air::renderer
