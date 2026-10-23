// Built-in test scenes beyond the Cornell box and the furnace. Each one stresses something specific
// (see Scene.h). They are generated procedurally with a fixed seed, so they need no assets and
// render the same everywhere. Units: metres, radiance in nits, camera exposure as EV100.

#include "air/renderer/Scene.h"

#include "SceneBuilders.h"

#include <cmath>
#include <format>
#include <numbers>
#include <vector>

namespace air::renderer::detail
{

namespace
{

using hlsl::cross;
using hlsl::dot;
using hlsl::normalize;

constexpr float kPi = std::numbers::pi_v<float>;

// Daylight: the physically based sky (Sky.h), the sun at the given position.
void AddSky(SceneDesc& scene, float elevationDegrees, float azimuthDegrees, float altitude = 0.0f)
{
    scene.sky = {
        .physical = true, .sunElevation = elevationDegrees, .sunAzimuth = azimuthDegrees, .altitude = altitude};
}

// Axis-aligned box centred at `center`.
Instance& AddBox(SceneDesc& scene, std::string name, float3 center, float3 halfExtent, u32 material, float yaw = 0.0f)
{
    Instance& instance = AddInstance(scene, name, AddMesh(scene, MakeBox(name, halfExtent)), material);
    SetTransform(instance, center, yaw);
    return instance;
}

// Value noise in [0, 1) on an integer lattice, smoothly interpolated; deterministic.
float LatticeValue(i32 x, i32 z, u32 octave)
{
    u32 h = u32(x) * 0x8da6b343u ^ u32(z) * 0xd8163841u ^ octave * 0xcb1ab31fu;
    h = (h ^ (h >> 16)) * 0x7feb352du;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    h ^= h >> 16;
    return float(h >> 8) * (1.0f / 16777216.0f);
}

float ValueNoise(float x, float z, u32 octave)
{
    const float fx = std::floor(x);
    const float fz = std::floor(z);
    const i32 ix = i32(fx);
    const i32 iz = i32(fz);
    auto smooth = [](float t) { return t * t * (3.0f - 2.0f * t); };
    const float tx = smooth(x - fx);
    const float tz = smooth(z - fz);
    const float a = LatticeValue(ix, iz, octave);
    const float b = LatticeValue(ix + 1, iz, octave);
    const float c = LatticeValue(ix, iz + 1, octave);
    const float d = LatticeValue(ix + 1, iz + 1, octave);
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
}

// Hills with ridges: fractal sum of value noise, metres.
float TerrainHeight(float x, float z)
{
    float height = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f / 160.0f;
    float total = 0.0f;
    for (u32 octave = 0; octave < 6; ++octave)
    {
        float n = ValueNoise(x * frequency, z * frequency, octave);
        if (octave >= 2)
        {
            n = 1.0f - std::abs(2.0f * n - 1.0f); // ridged detail
        }
        height += n * amplitude;
        total += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.03f;
    }
    const float h = height / total;
    return 70.0f * h * h; // flatter valleys, steeper peaks
}

} // namespace

// Veach's multiple importance sampling test: four glossy plates, smooth at the top to rough at the
// bottom, each tilted to mirror a row of four spherical lights into the camera. The lights have
// equal power and radii from 3 cm to 90 cm, so light sampling wins on rough plates and large
// lights while BSDF sampling wins on smooth plates and small lights; MIS must win everywhere.
SceneDesc MakeVeachMis()
{
    SceneDesc scene;
    scene.name = "veach_mis";
    const float3 eye(0.0f, 2.0f, 15.0f);
    const float radii[4] = {0.0333f, 0.1f, 0.3f, 0.9f};
    const float xs[4] = {-3.75f, -1.25f, 1.25f, 3.75f};
    for (u32 i = 0; i < 4; ++i)
    {
        const u32 light = AddMaterial(scene, {.name = std::format("Light{}", i),
                                              .baseColor = {0.0f, 0.0f, 0.0f},
                                              .roughness = 1.0f,
                                              .emissionNits = 800.0f * (0.1f / radii[i]) * (0.1f / radii[i])});
        Instance& sphere = AddInstance(scene, std::format("Light{}", i),
                                       AddMesh(scene, MakeSphere(std::format("Light{}", i), radii[i], 32)), light);
        SetTransform(sphere, {xs[i], 0.0f, 0.0f});
    }
    // Plate layout of the widely used version of this scene: 8 x 1 m plates in a stair below the
    // light row, the farthest the smoothest (GGX alpha 0.005, 0.02, 0.05, 0.1).
    const float3 plates[4][4] = {
        {{4.0f, -2.70651f, 0.25609f},
         {4.0f, -2.08375f, -0.526323f},
         {-4.0f, -2.08375f, -0.526323f},
         {-4.0f, -2.70651f, 0.25609f}},
        {{4.0f, -3.28825f, 1.36972f},
         {4.0f, -2.83856f, 0.476536f},
         {-4.0f, -2.83856f, 0.476536f},
         {-4.0f, -3.28825f, 1.36972f}},
        {{4.0f, -3.73096f, 2.70046f},
         {4.0f, -3.43378f, 1.74564f},
         {-4.0f, -3.43378f, 1.74564f},
         {-4.0f, -3.73096f, 2.70046f}},
        {{4.0f, -3.99615f, 4.0667f},
         {4.0f, -3.82069f, 3.08221f},
         {-4.0f, -3.82069f, 3.08221f},
         {-4.0f, -3.99615f, 4.0667f}},
    };
    const float alphas[4] = {0.005f, 0.02f, 0.05f, 0.1f};
    for (u32 k = 0; k < 4; ++k)
    {
        const float3* p = plates[k];
        const float3 center = (p[0] + p[1] + p[2] + p[3]) * 0.25f;
        float3 n = normalize(cross(p[1] - p[0], p[2] - p[0]));
        n = dot(n, eye - center) < 0.0f ? -n : n; // the reflecting side faces the camera
        const u32 plate = AddMaterial(scene, {.name = std::format("Plate{}", k),
                                              .baseColor = {0.9f, 0.9f, 0.9f},
                                              .roughness = std::sqrt(alphas[k]),
                                              .metallic = 1.0f});
        const std::string name = std::format("Plate{}", k);
        AddInstance(scene, name, AddMesh(scene, MakeQuad(name, p[0], p[1], p[2], p[3], n)), plate);
    }
    // The dim fill light of the original, up and to the right.
    const u32 fill = AddMaterial(
        scene, {.name = "Fill", .baseColor = {0.0f, 0.0f, 0.0f}, .roughness = 1.0f, .emissionNits = 6400.0f});
    SetTransform(AddInstance(scene, "Fill", AddMesh(scene, MakeSphere("Fill", 0.5f, 32)), fill), {10.0f, 10.0f, 4.0f});
    // A dark backdrop behind the lights, so their falloff has something to land on.
    const u32 backdrop =
        AddMaterial(scene, {.name = "Backdrop", .baseColor = {0.08f, 0.08f, 0.08f}, .roughness = 0.9f});
    AddInstance(scene, "Backdrop",
                AddMesh(scene, MakeQuad("Backdrop", {-14.0f, -8.0f, -3.0f}, {14.0f, -8.0f, -3.0f}, {14.0f, 9.0f, -3.0f},
                                        {-14.0f, 9.0f, -3.0f}, {0.0f, 0.0f, 1.0f})),
                backdrop);
    LookAt(scene.camera, eye, {0.0f, -2.0f, 2.5f});
    scene.camera.verticalFov = 28.0f * kPi / 180.0f;
    scene.camera.ev100 = 7.0f;
    return scene;
}

// Rows (back to front): white dielectric, red dielectric, gold; columns: roughness 0 to 1. One
// rectangular area light and a dim fill environment.
SceneDesc MakeMaterialGrid()
{
    SceneDesc scene;
    scene.name = "material_grid";
    const u32 grey = AddMaterial(scene, {.name = "GreyCard", .baseColor = {0.18f, 0.18f, 0.18f}, .roughness = 0.8f});
    AddInstance(scene, "Floor",
                AddMesh(scene, MakeQuad("Floor", {-10.0f, 0.0f, 10.0f}, {10.0f, 0.0f, 10.0f}, {10.0f, 0.0f, -10.0f},
                                        {-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f})),
                grey);
    AddInstance(scene, "Backdrop",
                AddMesh(scene, MakeQuad("Backdrop", {-10.0f, 0.0f, -3.0f}, {10.0f, 0.0f, -3.0f}, {10.0f, 8.0f, -3.0f},
                                        {-10.0f, 8.0f, -3.0f}, {0.0f, 0.0f, 1.0f})),
                grey);
    const u32 sphere = AddMesh(scene, MakeSphere("Sphere", 0.35f, 48));
    const float3 colors[3] = {{0.8f, 0.8f, 0.8f}, {0.7f, 0.05f, 0.04f}, {1.0f, 0.71f, 0.29f}};
    const char* names[3] = {"White", "Red", "Gold"};
    for (u32 row = 0; row < 3; ++row)
    {
        for (u32 column = 0; column < 7; ++column)
        {
            const u32 material = AddMaterial(scene, {.name = std::format("{}{}", names[row], column),
                                                     .baseColor = colors[row],
                                                     .roughness = float(column) / 6.0f,
                                                     .metallic = row == 2 ? 1.0f : 0.0f});
            Instance& instance = AddInstance(scene, scene.materials[material].name, sphere, material);
            SetTransform(instance, {(float(column) - 3.0f) * 0.9f, 0.35f, 0.5f - float(row) * 1.0f});
        }
    }
    const u32 light = AddMaterial(scene, {.name = "AreaLight",
                                          .baseColor = {0.0f, 0.0f, 0.0f},
                                          .roughness = 1.0f,
                                          .emissionColor = {1.0f, 0.92f, 0.82f},
                                          .emissionNits = 2500.0f});
    AddInstance(scene, "AreaLight",
                AddMesh(scene, MakeQuad("AreaLight", {-2.0f, 4.5f, 3.5f}, {2.0f, 4.5f, 3.5f}, {2.0f, 4.5f, 1.5f},
                                        {-2.0f, 4.5f, 1.5f}, {0.0f, -1.0f, 0.0f})),
                light);
    scene.environment = {30.0f, 30.0f, 30.0f};
    LookAt(scene.camera, {0.0f, 3.0f, 6.2f}, {0.0f, 0.3f, -0.6f});
    scene.camera.verticalFov = 0.6f;
    scene.camera.ev100 = 9.0f;
    return scene;
}

// About 2400 instances of two meshes (a sphere and a box) with 32 materials, a few of them
// emissive, on a large ground plane in daylight.
SceneDesc MakeSphereField()
{
    SceneDesc scene;
    scene.name = "sphere_field";
    Random rng(7);
    const u32 ground = AddMaterial(scene, {.name = "Ground", .baseColor = {0.30f, 0.32f, 0.27f}, .roughness = 0.95f});
    AddInstance(scene, "Ground",
                AddMesh(scene, MakeQuad("Ground", {-160.0f, 0.0f, 160.0f}, {160.0f, 0.0f, 160.0f},
                                        {160.0f, 0.0f, -160.0f}, {-160.0f, 0.0f, -160.0f}, {0.0f, 1.0f, 0.0f})),
                ground);
    std::vector<u32> surfaces;
    for (u32 i = 0; i < 28; ++i)
    {
        const float kind = rng.Next();
        const float3 tint(rng.Range(0.2f, 1.0f), rng.Range(0.2f, 1.0f), rng.Range(0.2f, 1.0f));
        Material m{.name = std::format("Surface{}", i)};
        if (kind < 0.45f)
        {
            m.baseColor = tint * 0.6f + float3(0.2f); // desaturated paint
            m.roughness = rng.Range(0.2f, 0.9f);
        }
        else if (kind < 0.8f)
        {
            m.baseColor = tint * 0.3f + float3(0.65f); // light metal tints
            m.roughness = rng.Range(0.05f, 0.5f);
            m.metallic = 1.0f;
        }
        else
        {
            m.baseColor = tint;
            m.roughness = rng.Range(0.0f, 0.15f); // glossy plastic
        }
        surfaces.push_back(AddMaterial(scene, m));
    }
    std::vector<u32> emitters;
    for (u32 i = 0; i < 4; ++i)
    {
        emitters.push_back(AddMaterial(scene, {.name = std::format("Glow{}", i),
                                               .baseColor = {0.0f, 0.0f, 0.0f},
                                               .roughness = 1.0f,
                                               .emissionColor = Hue(float(i) / 4.0f + 0.05f),
                                               .emissionNits = 5e4f * std::pow(10.0f, rng.Next())}));
    }
    const u32 sphere = AddMesh(scene, MakeSphere("Sphere", 1.0f, 24));
    const u32 box = AddMesh(scene, MakeBox("Box", {0.5f, 0.5f, 0.5f}));
    u32 count = 0;
    auto scatter = [&](float x0, float x1, float z0, float z1, float step)
    {
        for (float z = z0; z <= z1; z += step)
        {
            for (float x = x0; x <= x1; x += step)
            {
                const float px = x + rng.Range(-0.3f, 0.3f) * step;
                const float pz = z + rng.Range(-0.3f, 0.3f) * step;
                if (pz > 4.0f && std::abs(px) < 4.0f)
                {
                    continue; // keep the foreground open
                }
                const bool isSphere = rng.Next() < 0.8f;
                const bool emissive = isSphere && rng.Next() < 0.03f;
                const u32 material =
                    emissive ? emitters[rng.NextU32() % emitters.size()] : surfaces[rng.NextU32() % surfaces.size()];
                Instance& instance =
                    AddInstance(scene, std::format("Object{}", count++), isSphere ? sphere : box, material);
                if (isSphere)
                {
                    const float radius = rng.Range(0.2f, 0.6f);
                    SetTransform(instance, {px, radius, pz}, 0.0f, radius);
                }
                else
                {
                    const float size = rng.Range(0.3f, 1.0f);
                    SetTransform(instance, {px, 0.5f * size, pz}, rng.Range(0.0f, kPi), size);
                }
            }
        }
    };
    scatter(-30.0f, 30.0f, -45.0f, 8.0f, 1.5f);
    scatter(-70.0f, 70.0f, -110.0f, -46.0f, 3.0f);
    AddSky(scene, 32.0f, 140.0f);
    LookAt(scene.camera, {0.0f, 3.2f, 13.0f}, {0.0f, 0.6f, -6.0f});
    scene.camera.verticalFov = 0.75f;
    scene.camera.ev100 = 14.5f;
    return scene;
}

// A 44 m hall lit by 66 coloured ceiling panels and 300 tiny floor LEDs whose radiance spans two
// orders of magnitude: choosing which light to sample matters far more than how.
SceneDesc MakeManyLights()
{
    SceneDesc scene;
    scene.name = "many_lights";
    Random rng(11);
    const u32 floorMaterial =
        AddMaterial(scene, {.name = "Floor", .baseColor = {0.35f, 0.35f, 0.35f}, .roughness = 0.35f});
    const u32 wall = AddMaterial(scene, {.name = "Wall", .baseColor = {0.6f, 0.6f, 0.6f}, .roughness = 0.9f});
    const float x0 = -6.0f, x1 = 6.0f, y1 = 5.0f, z0 = -40.0f, z1 = 4.0f;
    AddInstance(scene, "Floor",
                AddMesh(scene, MakeQuad("Floor", {x0, 0.0f, z1}, {x1, 0.0f, z1}, {x1, 0.0f, z0}, {x0, 0.0f, z0},
                                        {0.0f, 1.0f, 0.0f})),
                floorMaterial);
    AddInstance(scene, "Ceiling",
                AddMesh(scene, MakeQuad("Ceiling", {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}, {x0, y1, z0},
                                        {0.0f, -1.0f, 0.0f})),
                wall);
    AddInstance(scene, "LeftWall",
                AddMesh(scene, MakeQuad("LeftWall", {x0, 0.0f, z1}, {x0, 0.0f, z0}, {x0, y1, z0}, {x0, y1, z1},
                                        {1.0f, 0.0f, 0.0f})),
                wall);
    AddInstance(scene, "RightWall",
                AddMesh(scene, MakeQuad("RightWall", {x1, 0.0f, z1}, {x1, 0.0f, z0}, {x1, y1, z0}, {x1, y1, z1},
                                        {-1.0f, 0.0f, 0.0f})),
                wall);
    AddInstance(scene, "BackWall",
                AddMesh(scene, MakeQuad("BackWall", {x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, y1, z0}, {x0, y1, z0},
                                        {0.0f, 0.0f, 1.0f})),
                wall);
    AddInstance(scene, "FrontWall",
                AddMesh(scene, MakeQuad("FrontWall", {x0, 0.0f, z1}, {x1, 0.0f, z1}, {x1, y1, z1}, {x0, y1, z1},
                                        {0.0f, 0.0f, -1.0f})),
                wall);
    const u32 column = AddMesh(scene, MakeBox("Column", {0.3f, 2.5f, 0.3f}));
    const u32 columnMaterial =
        AddMaterial(scene, {.name = "Column", .baseColor = {0.7f, 0.7f, 0.7f}, .roughness = 0.5f});
    for (u32 k = 0; k < 10; ++k)
    {
        for (float x : {-3.5f, 3.5f})
        {
            SetTransform(AddInstance(scene, std::format("Column{}", k), column, columnMaterial),
                         {x, 2.5f, -36.0f + 4.0f * float(k)});
        }
    }
    const u32 metal =
        AddMaterial(scene, {.name = "Chrome", .baseColor = {0.9f, 0.9f, 0.9f}, .roughness = 0.15f, .metallic = 1.0f});
    const u32 ball = AddMesh(scene, MakeSphere("Ball", 0.5f, 48));
    for (u32 k = 0; k < 8; ++k)
    {
        SetTransform(AddInstance(scene, std::format("Ball{}", k), ball, metal), {0.0f, 0.5f, -2.5f - 4.5f * float(k)});
    }
    // Ceiling panels: one 0.5 m quad mesh, instanced, each with its own colour and brightness.
    const u32 panel = AddMesh(scene, MakeQuad("Panel", {-0.25f, 0.0f, 0.25f}, {0.25f, 0.0f, 0.25f},
                                              {0.25f, 0.0f, -0.25f}, {-0.25f, 0.0f, -0.25f}, {0.0f, -1.0f, 0.0f}));
    u32 lightCount = 0;
    for (float x : {-4.0f, 0.0f, 4.0f})
    {
        for (u32 k = 0; k < 22; ++k)
        {
            const u32 material = AddMaterial(scene, {.name = std::format("Panel{}", lightCount),
                                                     .baseColor = {0.0f, 0.0f, 0.0f},
                                                     .roughness = 1.0f,
                                                     .emissionColor = Hue(rng.Next()) * 0.7f + float3(0.3f),
                                                     .emissionNits = 3e3f * std::pow(100.0f, rng.Next())});
            SetTransform(AddInstance(scene, std::format("Panel{}", lightCount++), panel, material),
                         {x, y1 - 0.01f, 2.0f - 1.9f * float(k)});
        }
    }
    const u32 led = AddMesh(scene, MakeSphere("Led", 0.04f, 12));
    std::vector<u32> ledMaterials;
    for (u32 i = 0; i < 12; ++i)
    {
        ledMaterials.push_back(AddMaterial(scene, {.name = std::format("Led{}", i),
                                                   .baseColor = {0.0f, 0.0f, 0.0f},
                                                   .roughness = 1.0f,
                                                   .emissionColor = Hue(float(i) / 12.0f),
                                                   .emissionNits = 2e5f * std::pow(10.0f, rng.Next())}));
    }
    for (u32 i = 0; i < 300; ++i)
    {
        SetTransform(AddInstance(scene, std::format("Led{}", i), led, ledMaterials[rng.NextU32() % 12]),
                     {rng.Range(x0 + 0.5f, x1 - 0.5f), 0.04f, rng.Range(z0 + 1.0f, z1 - 1.0f)});
    }
    LookAt(scene.camera, {0.0f, 1.7f, 2.5f}, {0.0f, 1.4f, -20.0f});
    scene.camera.verticalFov = 0.9f;
    scene.camera.ev100 = 12.0f;
    return scene;
}

// Two rooms joined by a doorway whose door is open 25 degrees. The only light hangs in the far
// room, so everything the camera sees in the near room is lit by light squeezing through the gap
// and bouncing: few paths find the light, which is hard for a unidirectional path tracer.
SceneDesc MakeDoor()
{
    SceneDesc scene;
    scene.name = "door";
    const u32 wall = AddMaterial(scene, {.name = "Wall", .baseColor = {0.75f, 0.74f, 0.7f}, .roughness = 0.9f});
    const u32 floorMaterial =
        AddMaterial(scene, {.name = "Floor", .baseColor = {0.45f, 0.33f, 0.24f}, .roughness = 0.6f});
    const float x0 = -3.0f, x1 = 3.0f, y1 = 3.0f, z0 = -6.0f, z1 = 6.0f;
    AddInstance(scene, "Floor",
                AddMesh(scene, MakeQuad("Floor", {x0, 0.0f, z1}, {x1, 0.0f, z1}, {x1, 0.0f, z0}, {x0, 0.0f, z0},
                                        {0.0f, 1.0f, 0.0f})),
                floorMaterial);
    AddInstance(scene, "Ceiling",
                AddMesh(scene, MakeQuad("Ceiling", {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}, {x0, y1, z0},
                                        {0.0f, -1.0f, 0.0f})),
                wall);
    AddInstance(scene, "LeftWall",
                AddMesh(scene, MakeQuad("LeftWall", {x0, 0.0f, z1}, {x0, 0.0f, z0}, {x0, y1, z0}, {x0, y1, z1},
                                        {1.0f, 0.0f, 0.0f})),
                wall);
    AddInstance(scene, "RightWall",
                AddMesh(scene, MakeQuad("RightWall", {x1, 0.0f, z1}, {x1, 0.0f, z0}, {x1, y1, z0}, {x1, y1, z1},
                                        {-1.0f, 0.0f, 0.0f})),
                wall);
    AddInstance(scene, "FarWall",
                AddMesh(scene, MakeQuad("FarWall", {x0, 0.0f, z0}, {x1, 0.0f, z0}, {x1, y1, z0}, {x0, y1, z0},
                                        {0.0f, 0.0f, 1.0f})),
                wall);
    AddInstance(scene, "NearWall",
                AddMesh(scene, MakeQuad("NearWall", {x0, 0.0f, z1}, {x1, 0.0f, z1}, {x1, y1, z1}, {x0, y1, z1},
                                        {0.0f, 0.0f, -1.0f})),
                wall);
    // The dividing wall at z = 0 around a 1 x 2.2 m doorway (x 0.5 to 1.5), 12 cm thick.
    AddBox(scene, "DividerLeft", {-1.25f, 1.5f, 0.0f}, {1.75f, 1.5f, 0.06f}, wall);
    AddBox(scene, "DividerRight", {2.25f, 1.5f, 0.0f}, {0.75f, 1.5f, 0.06f}, wall);
    AddBox(scene, "Lintel", {1.0f, 2.6f, 0.0f}, {0.5f, 0.4f, 0.06f}, wall);
    // The door, hinged at x = 1.5 and swung 25 degrees into the lit room.
    const u32 door = AddMaterial(scene, {.name = "Door", .baseColor = {0.55f, 0.45f, 0.35f}, .roughness = 0.5f});
    const float swing = -25.0f * kPi / 180.0f;
    const float3 hinge(1.5f, 0.0f, -0.06f);
    const float3 doorCenter = hinge + float3(-0.49f * std::cos(swing), 1.1f, 0.49f * std::sin(swing));
    AddBox(scene, "Door", doorCenter, {0.49f, 1.09f, 0.025f}, door, swing);
    // The light, in the far room.
    const u32 light = AddMaterial(scene, {.name = "Light",
                                          .baseColor = {0.0f, 0.0f, 0.0f},
                                          .roughness = 1.0f,
                                          .emissionColor = {1.0f, 0.85f, 0.7f},
                                          .emissionNits = 30000.0f});
    AddInstance(scene, "Light",
                AddMesh(scene, MakeQuad("Light", {-1.7f, y1 - 0.01f, -2.3f}, {-0.3f, y1 - 0.01f, -2.3f},
                                        {-0.3f, y1 - 0.01f, -3.7f}, {-1.7f, y1 - 0.01f, -3.7f}, {0.0f, -1.0f, 0.0f})),
                light);
    // Something to look at in the near room.
    const u32 gold =
        AddMaterial(scene, {.name = "Gold", .baseColor = {1.0f, 0.71f, 0.29f}, .roughness = 0.25f, .metallic = 1.0f});
    SetTransform(AddInstance(scene, "Sphere", AddMesh(scene, MakeSphere("Sphere", 0.45f, 48)), gold),
                 {-1.2f, 0.45f, 2.5f});
    const u32 plaster = AddMaterial(scene, {.name = "Plaster", .baseColor = {0.8f, 0.8f, 0.8f}, .roughness = 0.8f});
    AddBox(scene, "Box", {0.8f, 0.4f, 3.2f}, {0.4f, 0.4f, 0.4f}, plaster, 0.4f);
    LookAt(scene.camera, {-2.4f, 1.6f, 5.6f}, {0.9f, 1.0f, 0.0f});
    scene.camera.verticalFov = 0.9f;
    scene.camera.ev100 = 4.5f;
    return scene;
}

// 600 x 600 m of hills (1024 x 1024 vertices, 2.1M triangles) around a lake, in daylight.
SceneDesc MakeTerrain()
{
    SceneDesc scene;
    scene.name = "terrain";
    constexpr u32 kSide = 1024;
    constexpr float kSize = 600.0f;
    const float step = kSize / float(kSide - 1);
    Mesh terrain;
    terrain.name = "Terrain";
    // Heights on a grid one sample larger on each side, so every vertex has neighbours for its
    // central-difference normal.
    constexpr u32 kGrid = kSide + 2;
    std::vector<float> heights(size_t(kGrid) * kGrid);
    for (u32 j = 0; j < kGrid; ++j)
    {
        for (u32 i = 0; i < kGrid; ++i)
        {
            heights[size_t(j) * kGrid + i] =
                TerrainHeight(-0.5f * kSize + (float(i) - 1.0f) * step, -0.5f * kSize + (float(j) - 1.0f) * step);
        }
    }
    // Grid lookup by vertex index; -1 and kSide reach the border samples.
    auto height = [&](i32 i, i32 j) { return heights[size_t(j + 1) * kGrid + size_t(i + 1)]; };
    terrain.vertices.reserve(size_t(kSide) * kSide);
    for (u32 j = 0; j < kSide; ++j)
    {
        for (u32 i = 0; i < kSide; ++i)
        {
            const float x = -0.5f * kSize + float(i) * step;
            const float z = -0.5f * kSize + float(j) * step;
            const i32 vi = i32(i);
            const i32 vj = i32(j);
            const float dx = height(vi + 1, vj) - height(vi - 1, vj);
            const float dz = height(vi, vj + 1) - height(vi, vj - 1);
            const float3 normal = normalize(float3(-dx, 2.0f * step, -dz));
            terrain.vertices.push_back(
                {{x, height(vi, vj), z}, float(i) / float(kSide - 1), normal, float(j) / float(kSide - 1)});
        }
    }
    terrain.indices.reserve(size_t(kSide - 1) * (kSide - 1) * 6);
    for (u32 j = 0; j + 1 < kSide; ++j)
    {
        for (u32 i = 0; i + 1 < kSide; ++i)
        {
            const u32 a = j * kSide + i;
            const u32 b = a + 1;
            const u32 c = a + kSide;
            const u32 d = c + 1;
            // Counter-clockwise seen from above: cross(e1, e2) points up.
            terrain.indices.insert(terrain.indices.end(), {a, c, b, b, c, d});
        }
    }
    const u32 ground = AddMaterial(scene, {.name = "Ground", .baseColor = {0.13f, 0.17f, 0.07f}, .roughness = 0.9f});
    AddInstance(scene, "Terrain", AddMesh(scene, std::move(terrain)), ground);
    const u32 water = AddMaterial(scene, {.name = "Water", .baseColor = {0.02f, 0.05f, 0.06f}, .roughness = 0.06f});
    const float h = 0.5f * kSize;
    AddInstance(scene, "Lake",
                AddMesh(scene, MakeQuad("Lake", {-h, 6.0f, h}, {h, 6.0f, h}, {h, 6.0f, -h}, {-h, 6.0f, -h},
                                        {0.0f, 1.0f, 0.0f})),
                water);
    AddSky(scene, 18.0f, -60.0f, 300.0f);
    LookAt(scene.camera, {-60.0f, 65.0f, 260.0f}, {20.0f, 15.0f, -40.0f});
    scene.camera.verticalFov = 0.7f;
    scene.camera.ev100 = 14.0f;
    return scene;
}

// Late afternoon on an open plain: the sun 4 degrees above the horizon, long shadows across a
// 400 m ground, a few objects against the low sun. Shows the sky's colours, which come from the
// spectral atmosphere, and the planet's ground beyond the plane.
SceneDesc MakeSunset()
{
    SceneDesc scene;
    scene.name = "sunset";
    const u32 ground = AddMaterial(scene, {.name = "Grass", .baseColor = {0.16f, 0.2f, 0.09f}, .roughness = 0.9f});
    const u32 white = AddMaterial(scene, {.name = "White", .baseColor = {0.8f, 0.8f, 0.8f}, .roughness = 0.5f});
    const u32 metal =
        AddMaterial(scene, {.name = "Steel", .baseColor = {0.9f, 0.9f, 0.88f}, .roughness = 0.15f, .metallic = 1.0f});
    const u32 stone = AddMaterial(scene, {.name = "Stone", .baseColor = {0.45f, 0.4f, 0.35f}, .roughness = 0.8f});
    constexpr float kHalf = 200.0f;
    AddInstance(scene, "Ground",
                AddMesh(scene, MakeQuad("Ground", {-kHalf, 0.0f, kHalf}, {kHalf, 0.0f, kHalf}, {kHalf, 0.0f, -kHalf},
                                        {-kHalf, 0.0f, -kHalf}, {0.0f, 1.0f, 0.0f})),
                ground);
    const u32 sphere = AddMesh(scene, MakeSphere("Sphere", 1.0f, 48));
    SetTransform(AddInstance(scene, "WhiteSphere", sphere, white), {-1.6f, 1.0f, 0.0f});
    SetTransform(AddInstance(scene, "SteelSphere", sphere, metal), {1.4f, 0.8f, 1.2f}, 0.0f, 0.8f);
    const u32 pillar = AddMesh(scene, MakeBox("Pillar", {0.4f, 2.5f, 0.4f}));
    for (int i = 0; i < 5; ++i)
    {
        SetTransform(AddInstance(scene, std::format("Pillar{}", i), pillar, stone),
                     {-6.0f + 3.0f * float(i), 2.5f, -7.0f - 1.5f * float(i)}, 0.3f * float(i));
    }
    AddSky(scene, 4.0f, -35.0f);
    LookAt(scene.camera, {0.5f, 1.8f, 9.0f}, {-1.5f, 1.6f, -10.0f});
    scene.camera.verticalFov = 0.9f;
    scene.camera.ev100 = 11.5f;
    return scene;
}

// A meadow under fair-weather cumulus (Clouds.h) in the afternoon: the clouds' bright tops and grey
// bases, the sky between them, their shadows drifting over 4 km of grass, a farmstead for scale.
SceneDesc MakeCumulus()
{
    SceneDesc scene;
    scene.name = "cumulus";
    const u32 grass = AddMaterial(scene, {.name = "Grass", .baseColor = {0.14f, 0.19f, 0.07f}, .roughness = 0.9f});
    const u32 wall = AddMaterial(scene, {.name = "Whitewash", .baseColor = {0.75f, 0.73f, 0.68f}, .roughness = 0.7f});
    const u32 roof = AddMaterial(scene, {.name = "Roof", .baseColor = {0.35f, 0.12f, 0.08f}, .roughness = 0.6f});
    const u32 bark = AddMaterial(scene, {.name = "Bark", .baseColor = {0.2f, 0.15f, 0.1f}, .roughness = 0.9f});
    const u32 leaves = AddMaterial(scene, {.name = "Leaves", .baseColor = {0.08f, 0.15f, 0.05f}, .roughness = 0.8f});
    constexpr float kHalf = 2000.0f;
    AddInstance(scene, "Meadow",
                AddMesh(scene, MakeQuad("Meadow", {-kHalf, 0.0f, kHalf}, {kHalf, 0.0f, kHalf}, {kHalf, 0.0f, -kHalf},
                                        {-kHalf, 0.0f, -kHalf}, {0.0f, 1.0f, 0.0f})),
                grass);
    AddBox(scene, "House", {-14.0f, 3.0f, -40.0f}, {6.0f, 3.0f, 4.0f}, wall, 0.3f);
    AddBox(scene, "Roof", {-14.0f, 6.6f, -40.0f}, {6.4f, 0.6f, 4.4f}, roof, 0.3f);
    AddBox(scene, "Barn", {6.0f, 4.0f, -70.0f}, {8.0f, 4.0f, 5.0f}, roof, -0.2f);
    const u32 trunk = AddMesh(scene, MakeBox("Trunk", {0.25f, 2.0f, 0.25f}));
    const u32 crown = AddMesh(scene, MakeSphere("Crown", 2.6f, 32));
    const float3 trees[] = {{-30.0f, 0.0f, -25.0f},
                            {-24.0f, 0.0f, -55.0f},
                            {18.0f, 0.0f, -38.0f},
                            {30.0f, 0.0f, -95.0f},
                            {-60.0f, 0.0f, -120.0f}};
    for (size_t i = 0; i < std::size(trees); ++i)
    {
        SetTransform(AddInstance(scene, std::format("Trunk{}", i), trunk, bark), trees[i] + float3(0.0f, 2.0f, 0.0f));
        SetTransform(AddInstance(scene, std::format("Crown{}", i), crown, leaves), trees[i] + float3(0.0f, 5.6f, 0.0f));
    }
    AddSky(scene, 38.0f, 150.0f);
    scene.clouds = {.enabled = true, .bottom = 1200.0f, .top = 3200.0f, .coverage = 0.4f};
    LookAt(scene.camera, {0.0f, 1.7f, 0.0f}, {-10.0f, 12.0f, -60.0f});
    scene.camera.verticalFov = 1.0f;
    scene.camera.ev100 = 14.5f;
    return scene;
}

} // namespace air::renderer::detail
