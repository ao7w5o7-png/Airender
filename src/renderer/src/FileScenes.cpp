// The file scenes (SceneFiles.h): models from .deps/scenes composed under a sky, and the one list of
// every scene by name.

#include "air/renderer/SceneFiles.h"

#include "SceneBuilders.h"

#include <cmath>
#include <cstring>
#include <format>
#include <numbers>

namespace air::renderer
{
namespace
{

using detail::AddInstance;
using detail::AddMaterial;
using detail::AddMesh;
using detail::LookAt;

// The physical sky with its sun at elevation / azimuth (degrees).
void AddSun(SceneDesc& scene, float elevation, float azimuth)
{
    scene.sky.physical = true;
    scene.sky.sunElevation = elevation;
    scene.sky.sunAzimuth = azimuth;
}

float3 Srgb(float r, float g, float b)
{
    auto linear = [](float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); };
    return float3(linear(r), linear(g), linear(b));
}

bool Contains(std::string_view name, std::string_view part)
{
    return name.find(part) != std::string_view::npos;
}

// --- picapica --------------------------------------------------------------------------------------
// The PICA PICA files carry geometry and material names only: the names say what the demo's materials
// were (coloured plastics in matte, satin and glossy finishes, a few metals, rubber, glass), so this
// table makes physically based materials of them.
std::optional<Material> PicaMaterial(std::string_view name)
{
    Material m;
    m.roughness = Contains(name, "Matte")    ? 0.6f
                  : Contains(name, "Satin")  ? 0.38f
                  : Contains(name, "Glossy") ? 0.12f
                                             : 0.3f;
    struct Colour
    {
        std::string_view word;
        float3 srgb;
    };
    // The most specific word first.
    static const Colour kColours[] = {
        {"Blue_ExtraLight", {0.62f, 0.78f, 0.95f}},
        {"MediumGrey", {0.45f, 0.46f, 0.48f}},
        {"Grey_Dark", {0.16f, 0.16f, 0.17f}},
        {"Brown_Dark", {0.3f, 0.19f, 0.12f}},
        {"Wall", {0.85f, 0.85f, 0.83f}},
        {"Blue", {0.15f, 0.42f, 0.86f}},
        {"Red", {0.85f, 0.16f, 0.12f}},
        {"White", {0.9f, 0.9f, 0.88f}},
        {"Dark", {0.08f, 0.08f, 0.09f}},
        {"Brown", {0.58f, 0.38f, 0.22f}},
        {"Orange", {0.98f, 0.55f, 0.12f}},
        {"Pink", {0.98f, 0.55f, 0.68f}},
        {"Yellow", {0.98f, 0.8f, 0.2f}},
        {"Tape", {0.92f, 0.86f, 0.68f}},
        {"StickyNote", {0.98f, 0.86f, 0.3f}},
    };
    m.baseColor = Srgb(0.7f, 0.7f, 0.7f);
    for (const Colour& c : kColours)
    {
        if (Contains(name, c.word))
        {
            m.baseColor = Srgb(c.srgb.x, c.srgb.y, c.srgb.z);
            break;
        }
    }
    if (name.starts_with("Metal_"))
    {
        m.metallic = 1.0f;
        m.roughness = Contains(name, "HeatSink") ? 0.45f : 0.28f;
        m.baseColor = Contains(name, "Copper")       ? float3(0.95f, 0.64f, 0.54f)
                      : Contains(name, "CarbonCoat") ? float3(0.05f, 0.05f, 0.06f)
                      : Contains(name, "Blue")       ? float3(0.12f, 0.3f, 0.85f)
                                                     : float3(0.91f, 0.92f, 0.92f); // aluminium, the pen's tip
    }
    else if (name.starts_with("Rubber_"))
    {
        m.baseColor = float3(0.02f);
        m.roughness = 0.85f;
    }
    else if (name.starts_with("Glass_"))
    {
        m.baseColor = float3(0.005f);
        m.roughness = 0.03f;
    }
    else if (name.starts_with("Coated_"))
    {
        m.baseColor = float3(0.01f);
        m.roughness = 0.45f;
        m.coat = true;
        m.coatRoughness = 0.03f;
    }
    else if (name.starts_with("Paper_"))
    {
        m.roughness = 0.8f;
    }
    else if (Contains(name, "Pen_Ink") || Contains(name, "Pen_Plug"))
    {
        m.baseColor = Srgb(0.12f, 0.25f, 0.85f);
    }
    else if (Contains(name, "Transparent") || Contains(name, "InkTube"))
    {
        m.roughness = 0.1f; // not transmissive: a clear lacquered look instead
    }
    else if (name == "Lights")
    {
        m.baseColor = float3(0.9f, 0.74f, 0.35f);
        m.emissionColor = float3(1.0f, 0.75f, 0.35f);
        m.emissionNits = 2000.0f;
    }
    else if (name == "LED")
    {
        m.baseColor = float3(0.9f);
        m.emissionColor = float3(0.55f, 0.85f, 1.0f);
        m.emissionNits = 3000.0f;
    }
    return m;
}

SceneDesc MakePicaPica(const std::filesystem::path& root, std::string* error)
{
    SceneDesc scene;
    scene.name = "picapica";
    bool ok = true;
    auto fail = [&](const std::string& problem)
    {
        if (ok && error)
        {
            *error = problem;
        }
        ok = false;
    };
    // Texture sets the files do not name: the mug's, the books', the crayons' labels and the decals.
    auto textureSet = [&](const std::string& name, const char* color, const char* normal, const char* roughness)
    {
        std::string problem;
        std::optional<TextureImage> c = LoadTextureImage(root / color, &problem);
        std::optional<TextureImage> n = normal ? LoadTextureImage(root / normal, &problem) : std::nullopt;
        std::optional<TextureImage> r = roughness ? LoadTextureImage(root / roughness, &problem) : std::nullopt;
        if (!c || (normal && !n) || (roughness && !r))
        {
            fail(problem);
            return kNoTextures;
        }
        scene.textureSets.push_back(
            PackMaterialTextures(name, &*c, n ? &*n : nullptr, true, r ? &*r : nullptr, 0, nullptr, 0, 1024));
        return u32(scene.textureSets.size() - 1);
    };
    const u32 mug = textureSet("Mug", "pica-pica-seed-coffee-mug/Mug_01_D.tga",
                               "pica-pica-seed-coffee-mug/Mug_01_N.tga", "pica-pica-seed-coffee-mug/Mug_01_R.tga");
    const u32 pages = textureSet("Pages", "pica-pica-book-pile/Paper_Pages_01_D.tga", nullptr, nullptr);
    u32 covers[4];
    for (u32 i = 0; i < 4; ++i)
    {
        covers[i] =
            textureSet(std::format("Cover{}", i + 1),
                       std::format("pica-pica-book-pile/Paper_Book_Cover_0{}_D.tga", i + 1).c_str(), nullptr, nullptr);
    }
    const u32 labels[3] = {
        textureSet("LabelBlue", "pica-pica-floor-junk-clutter-02/Wax_Pastel_Label_Blue_01.png", nullptr, nullptr),
        textureSet("LabelRed", "pica-pica-floor-junk-clutter-02/Wax_Pastel_Label_Red_01.png", nullptr, nullptr),
        textureSet("LabelYellow", "pica-pica-floor-junk-clutter-02/Wax_Pastel_Label_Yellow_01.png", nullptr, nullptr)};
    const u32 caution = textureSet("Caution", "pica-pica-mini-diorama-01/Caution_Decal_01_D.tga", nullptr, nullptr);
    const u32 firstAid =
        textureSet("FirstAid", "pica-pica-mini-diorama-01/First_Aid_Wall_Kit_01_D.tga", nullptr, nullptr);
    const u32 note = textureSet("Note", "pica-pica-mini-diorama-01/StickyNote_02.tga", nullptr, nullptr);
    const u32 keyboard = textureSet("Keyboard", "pica-pica-mini-diorama-01/Decal_Keyboard_01_D.tga", nullptr, nullptr);
    if (!ok)
    {
        return scene;
    }
    auto textured = [](u32 set, float roughness, float cutoff)
    {
        Material m;
        m.baseColor = float3(1.0f);
        m.roughness = roughness;
        m.textures = set;
        m.alphaCutoff = cutoff;
        return m;
    };
    ImportOptions options;
    options.materialFor = [&](std::string_view name) -> std::optional<Material>
    {
        if (name == "Mug_01")
        {
            return textured(mug, 1.0f, 0.0f);
        }
        if (name == "lambert43SG")
        {
            return textured(pages, 0.8f, 0.0f);
        }
        if (name.starts_with("lambert4") && name.size() == 11 && name[8] >= '4' && name[8] <= '7')
        {
            return textured(covers[name[8] - '4'], 0.55f, 0.0f);
        }
        if (name.starts_with("Wax_Pastel_Label_0"))
        {
            return textured(labels[(name.back() - '1') % 3], 0.5f, 0.0f);
        }
        if (name == "Decal_Caution")
        {
            return textured(caution, 0.5f, 0.5f);
        }
        if (name == "Decal_First_Aid_Kit")
        {
            return textured(firstAid, 0.5f, 0.5f);
        }
        if (name == "Decal_Note_01")
        {
            return textured(note, 0.7f, 0.5f);
        }
        if (name == "Decal_Keyboard_01")
        {
            return textured(keyboard, 0.4f, 0.5f);
        }
        return PicaMaterial(name);
    };
    // The decals of the machines (Plastic_Decals) have no texture in the release.
    options.skipMaterial = [](std::string_view name) { return name == "Plastic_Decals"; };
    struct Asset
    {
        const char* file;
        float metresPerUnit; // 0: the file's
        float yaw;           // degrees
        float3 at;
        bool stand; // keep its display stand (the diorama's is its floor)
    };
    // A toy factory on a table top: the diorama in the middle, machines and parts around it, the books
    // (the processing unit on them) and the mug as the giants of this scale.
    const Asset assets[] = {
        {"pica-pica-mini-diorama-01/Mini_Diorama_01.fbx", 0.0f, 0.0f, {0.0f, 0.0f, 0.0f}, true},
        {"pica-pica-machines/Machines.fbx", 0.0f, -25.0f, {0.5f, 0.0f, 0.02f}, false},
        {"pica-pica-factory-central-processing-unit-01/Factory_Central_Processing_Unit_01.fbx",
         0.0f,
         15.0f,
         {-0.55f, 0.058f, 0.02f},
         false}, // on the books
        {"pica-pica-pallet-stack-01/Pallet_Stack_01.fbx", 0.0f, 10.0f, {0.36f, 0.0f, 0.36f}, false},
        {"pica-pica-floor-junk-clutter-01/Floor_Junk_Cluster_01.fbx", 0.0f, -20.0f, {-0.3f, 0.0f, 0.38f}, false},
        {"pica-pica-floor-junk-clutter-02/Floor_Junk_Cluster_02b.fbx", 0.0f, 40.0f, {0.02f, 0.0f, 0.48f}, false},
        {"pica-pica-book-pile/Book_Pile_04.obj", 0.01f, 105.0f, {-0.55f, 0.0f, 0.02f}, true},
        {"pica-pica-seed-coffee-mug/Mug_01.fbx", 0.01f, -60.0f, {0.58f, 0.0f, -0.36f}, true},
    };
    for (const Asset& a : assets)
    {
        options.metresPerUnit = a.metresPerUnit;
        // The stands are the plain blue plates a model was shown on (Mesh4 ... Mesh8).
        options.skipNode = [&a](std::string_view node)
        { return !a.stand && node.starts_with("Mesh") && node.size() <= 6; };
        std::string problem;
        const std::optional<ImportedModel> model = ImportModel(scene, root / a.file, options, &problem);
        if (!model)
        {
            fail(problem);
            return scene;
        }
        PlaceModel(scene, *model, 1.0f, a.yaw, a.at);
    }
    // The robots, several of one model, out and about.
    struct Robot
    {
        float3 at;
        float yaw;
    };
    const Robot robots[] = {{{0.16f, 0.0f, 0.27f}, 200.0f}, {{-0.1f, 0.0f, 0.24f}, 150.0f},
                            {{0.29f, 0.0f, 0.16f}, 250.0f}, {{-0.38f, 0.0f, 0.24f}, 120.0f},
                            {{0.06f, 0.0f, 0.36f}, 175.0f}, {{0.48f, 0.0f, 0.24f}, 230.0f},
                            {{-0.08f, 0.0f, 0.06f}, 90.0f}};
    options.metresPerUnit = 0.0f;
    options.skipNode = nullptr;
    std::string problem;
    const std::optional<ImportedModel> robot =
        ImportModel(scene, root / "pica-pica-robot-01/Robot_01.fbx", options, &problem);
    if (!robot)
    {
        fail(problem);
        return scene;
    }
    // One import, the others instances of its meshes.
    const std::vector<Instance> original(scene.instances.begin() + robot->firstInstance,
                                         scene.instances.begin() + robot->firstInstance + robot->instanceCount);
    for (size_t i = 0; i < std::size(robots); ++i)
    {
        ImportedModel copy = *robot;
        if (i > 0)
        {
            copy.firstInstance = u32(scene.instances.size());
            scene.instances.insert(scene.instances.end(), original.begin(), original.end());
        }
        PlaceModel(scene, copy, 1.0f, robots[i].yaw, robots[i].at);
    }
    // The table top.
    const u32 table = AddMaterial(scene, {.name = "Table", .baseColor = Srgb(0.82f, 0.8f, 0.76f), .roughness = 0.5f});
    AddInstance(scene, "Table",
                AddMesh(scene, MakeQuad("Table", {-20.0f, 0.0f, 20.0f}, {20.0f, 0.0f, 20.0f}, {20.0f, 0.0f, -20.0f},
                                        {-20.0f, 0.0f, -20.0f}, {0.0f, 1.0f, 0.0f})),
                table);
    AddSun(scene, 35.0f, 225.0f);
    LookAt(scene.camera, {0.32f, 0.2f, 0.95f}, {-0.02f, 0.05f, 0.0f});
    scene.camera.verticalFov = 0.8f;
    scene.camera.ev100 = 14.0f;
    return scene;
}

// --- sponza ----------------------------------------------------------------------------------------
SceneDesc MakeSponza(const std::filesystem::path& root, std::string* error)
{
    SceneDesc scene;
    scene.name = "sponza";
    ImportOptions options;
    options.maxTextureSize = 1024;
    if (!ImportModel(scene, root / "Sponza/Sponza.gltf", options, error))
    {
        return scene;
    }
    AddSun(scene, 62.0f, 110.0f);
    LookAt(scene.camera, {-10.5f, 1.7f, -0.4f}, {4.0f, 3.2f, 0.6f});
    scene.camera.verticalFov = 1.0f;
    scene.camera.ev100 = 12.0f;
    return scene;
}

// --- helmets ---------------------------------------------------------------------------------------
SceneDesc MakeHelmets(const std::filesystem::path& root, std::string* error)
{
    SceneDesc scene;
    scene.name = "helmets";
    struct Helmet
    {
        const char* file;
        float height; // metres
        float yaw;
        float3 at;
    };
    const Helmet helmets[] = {
        {"FlightHelmet/FlightHelmet.gltf", 0.62f, 20.0f, {-0.62f, 0.0f, 0.0f}},
        {"DamagedHelmet/DamagedHelmet.gltf", 0.36f, -10.0f, {0.0f, 0.25f, 0.1f}},
        {"SciFiHelmet/SciFiHelmet.gltf", 0.36f, -15.0f, {0.62f, 0.25f, 0.0f}},
    };
    ImportOptions options;
    options.maxTextureSize = 2048;
    for (const Helmet& h : helmets)
    {
        const std::optional<ImportedModel> model = ImportModel(scene, root / h.file, options, error);
        if (!model)
        {
            return scene;
        }
        const float size = model->boundsMax.y - model->boundsMin.y;
        PlaceModel(scene, *model, size > 0.0f ? h.height / size : 1.0f, h.yaw, h.at);
    }
    // Plinths under the two helmets without a stand, and the floor.
    const u32 stone = AddMaterial(scene, {.name = "Plinth", .baseColor = Srgb(0.55f, 0.55f, 0.56f), .roughness = 0.7f});
    for (float x : {0.0f, 0.62f})
    {
        Instance& plinth =
            AddInstance(scene, "Plinth", AddMesh(scene, MakeBox("Plinth", {0.16f, 0.125f, 0.16f})), stone);
        SetTransform(plinth, {x, 0.125f, x == 0.0f ? 0.1f : 0.0f});
    }
    const u32 floor = AddMaterial(scene, {.name = "Floor", .baseColor = Srgb(0.32f, 0.3f, 0.28f), .roughness = 0.5f});
    AddInstance(scene, "Floor",
                AddMesh(scene, MakeQuad("Floor", {-30.0f, 0.0f, 30.0f}, {30.0f, 0.0f, 30.0f}, {30.0f, 0.0f, -30.0f},
                                        {-30.0f, 0.0f, -30.0f}, {0.0f, 1.0f, 0.0f})),
                floor);
    AddSun(scene, 28.0f, 140.0f);
    LookAt(scene.camera, {0.15f, 0.62f, 1.75f}, {0.0f, 0.36f, 0.0f});
    scene.camera.verticalFov = 0.7f;
    scene.camera.ev100 = 14.0f;
    return scene;
}

struct FileScene
{
    std::string_view name;
    SceneDesc (*make)(const std::filesystem::path& root, std::string* error);
};

const FileScene kFileScenes[] = {
    {"picapica", MakePicaPica},
    {"sponza", MakeSponza},
    {"helmets", MakeHelmets},
};

} // namespace

void PlaceModel(SceneDesc& scene, const ImportedModel& model, float scale, float yawDegrees, float3 at)
{
    const float yaw = yawDegrees * std::numbers::pi_v<float> / 180.0f;
    const float c = std::cos(yaw) * scale;
    const float s = std::sin(yaw) * scale;
    const float turn[3][4] = {{c, 0.0f, s, 0.0f}, {0.0f, scale, 0.0f, 0.0f}, {-s, 0.0f, c, 0.0f}};
    float3 lo(1e30f);
    float3 hi(-1e30f);
    for (u32 i = model.firstInstance; i < model.firstInstance + model.instanceCount; ++i)
    {
        Instance& instance = scene.instances[i];
        float moved[3][4];
        for (int r = 0; r < 3; ++r)
        {
            for (int k = 0; k < 4; ++k)
            {
                moved[r][k] = turn[r][0] * instance.transform[0][k] + turn[r][1] * instance.transform[1][k] +
                              turn[r][2] * instance.transform[2][k];
            }
        }
        std::memcpy(instance.transform, moved, sizeof(moved));
        for (const gpu::Vertex& v : scene.meshes[instance.mesh].vertices)
        {
            const float3 p = TransformPoint(instance.transform, v.position);
            lo = hlsl::min(lo, p);
            hi = hlsl::max(hi, p);
        }
    }
    const float3 offset = at - float3((lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f);
    for (u32 i = model.firstInstance; i < model.firstInstance + model.instanceCount; ++i)
    {
        Instance& instance = scene.instances[i];
        instance.transform[0][3] += offset.x;
        instance.transform[1][3] += offset.y;
        instance.transform[2][3] += offset.z;
    }
}

std::vector<std::string> FileSceneNames()
{
    std::vector<std::string> names;
    for (const FileScene& scene : kFileScenes)
    {
        names.emplace_back(scene.name);
    }
    return names;
}

std::optional<SceneDesc> MakeFileScene(std::string_view name, std::string* error)
{
    for (const FileScene& scene : kFileScenes)
    {
        if (name != scene.name)
        {
            continue;
        }
        const std::filesystem::path root = SceneAssetsDirectory() / scene.name;
        std::error_code ec;
        if (!std::filesystem::exists(root / ".complete", ec))
        {
            if (error)
            {
                *error = std::format("scene '{}' needs its assets in {}: run `python3 tools/air.py fetch-scenes {}`",
                                     name, root.string(), name);
            }
            return std::nullopt;
        }
        std::string problem;
        SceneDesc desc = scene.make(root, &problem);
        if (!problem.empty())
        {
            if (error)
            {
                *error = std::format("scene '{}': {}", name, problem);
            }
            return std::nullopt;
        }
        return desc;
    }
    if (error)
    {
        *error = std::format("unknown scene '{}'", name);
    }
    return std::nullopt;
}

std::vector<std::string> SceneNames()
{
    std::vector<std::string> names = BuiltinSceneNames();
    for (std::string& name : FileSceneNames())
    {
        names.push_back(std::move(name));
    }
    return names;
}

std::optional<SceneDesc> MakeScene(std::string_view name, std::string* error)
{
    if (std::optional<SceneDesc> builtin = MakeBuiltinScene(name))
    {
        return builtin;
    }
    return MakeFileScene(name, error);
}

} // namespace air::renderer
