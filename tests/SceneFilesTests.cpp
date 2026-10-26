// Model files into scenes (SceneFiles.h): texture packing, glTF and OBJ imports of files written here,
// and the file scenes' names and missing assets. The downloaded scenes themselves are rendered by
// hand (docs/architecture.md §8.18): their assets are not in the repository.

#include "TestHelpers.h"

#include "air/renderer/SceneFiles.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace air;
using renderer::float3;

namespace
{

renderer::TextureImage Image(u32 w, u32 h, u32 texel)
{
    return {w, h, std::vector<u32>(size_t(w) * h, texel)};
}

u32 Channel(u32 texel, u32 channel)
{
    return (texel >> (8u * channel)) & 0xFFu;
}

std::string Base64(const std::vector<u8>& bytes)
{
    static constexpr char kDigits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3)
    {
        const u32 n = u32(bytes[i]) << 16 | (i + 1 < bytes.size() ? u32(bytes[i + 1]) << 8 : 0u) |
                      (i + 2 < bytes.size() ? u32(bytes[i + 2]) : 0u);
        out += kDigits[n >> 18];
        out += kDigits[(n >> 12) & 63];
        out += i + 1 < bytes.size() ? kDigits[(n >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? kDigits[n & 63] : '=';
    }
    return out;
}

std::filesystem::path TempDir(const char* name)
{
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "airender-tests" / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void SetEnv(const char* name, const std::string& value)
{
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty())
    {
        unsetenv(name);
    }
    else
    {
        setenv(name, value.c_str(), 1);
    }
#endif
}

} // namespace

// The surface texture takes the normal's x, its y flipped (glTF's +y is up the image, the renderer's
// towards increasing v), roughness and metallic from their channels; all textures share the largest
// size, at most maxSize.
TEST(SceneFiles_PackMaterialTextures)
{
    const renderer::TextureImage color = Image(2, 2, 0xFF336699u);
    const renderer::TextureImage normal = Image(4, 4, 200u | 60u << 8 | 255u << 16 | 255u << 24);
    const renderer::TextureImage mr = Image(4, 4, 0u | 100u << 8 | 30u << 16 | 255u << 24);
    const renderer::MaterialTextures t =
        renderer::PackMaterialTextures("Packed", &color, &normal, true, &mr, 1, &mr, 2, 1024);
    EXPECT(t.width == 4 && t.height == 4);
    EXPECT(t.color.size() == 16 && t.surface.size() == 16);
    EXPECT(t.color[5] == 0xFF336699u); // a uniform image stays uniform whatever the filter
    EXPECT(Channel(t.surface[5], 0) == 200u);
    EXPECT(Channel(t.surface[5], 1) == 255u - 60u);
    EXPECT(Channel(t.surface[5], 2) == 100u);
    EXPECT(Channel(t.surface[5], 3) == 30u);
    // Without a normal map and roughness: flat, and the material's constants as they are.
    const renderer::MaterialTextures plain =
        renderer::PackMaterialTextures("Plain", &color, nullptr, true, nullptr, 0, nullptr, 0, 1);
    EXPECT(plain.width == 1 && plain.height == 1);
    EXPECT(plain.surface[0] == (128u | 128u << 8 | 255u << 16 | 255u << 24));
    EXPECT(renderer::Validate(
               [&]
               {
                   renderer::SceneDesc scene = airtest::BuiltinScene("cornell_box");
                   scene.textureSets.push_back(t);
                   return scene;
               }())
               .empty());
}

// A quad in glTF (its buffer a data URI, no normals): its node's transform, its material's factors and
// emission, and normals from its faces.
TEST(SceneFiles_ImportsGltf)
{
    if (!renderer::ModelFilesAvailable())
    {
        return;
    }
    std::vector<u8> buffer;
    auto put = [&](const void* data, size_t size)
    {
        const u8* p = static_cast<const u8*>(data);
        buffer.insert(buffer.end(), p, p + size);
    };
    const float positions[12] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    const u16 indices[6] = {0, 1, 2, 0, 2, 3};
    put(positions, sizeof(positions));
    put(indices, sizeof(indices));
    const std::string json = R"({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0, "translation": [1, 2, 3], "scale": [2, 2, 2]}],
  "meshes": [{"name": "Quad", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
  "materials": [{"name": "Paint", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.25, 1, 1],
                 "roughnessFactor": 0.3, "metallicFactor": 0.7}, "emissiveFactor": [1, 0, 0]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
    {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}],
  "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 12}],
  "buffers": [{"byteLength": 60, "uri": "data:application/octet-stream;base64,)" +
                             Base64(buffer) + R"("}]
})";
    const std::filesystem::path dir = TempDir("gltf");
    std::ofstream(dir / "quad.gltf") << json;
    renderer::SceneDesc scene = airtest::BuiltinScene("cornell_box");
    const size_t meshes = scene.meshes.size();
    const size_t materials = scene.materials.size();
    std::string error;
    renderer::ImportOptions options;
    options.transform[0][3] = 10.0f; // moved 10 m along x after the file's own transform
    const std::optional<renderer::ImportedModel> model =
        renderer::ImportModel(scene, dir / "quad.gltf", options, &error);
    EXPECT(model.has_value());
    if (!model)
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
        return;
    }
    EXPECT(model->instanceCount == 1);
    EXPECT(scene.meshes.size() == meshes + 1 && scene.materials.size() == materials + 1);
    const renderer::Mesh& mesh = scene.meshes.back();
    EXPECT(mesh.vertices.size() == 4 && mesh.indices.size() == 6);
    EXPECT(std::abs(mesh.vertices[0].normal.z - 1.0f) < 1e-5f);
    const renderer::Instance& instance = scene.instances[model->firstInstance];
    EXPECT(std::abs(instance.transform[0][0] - 2.0f) < 1e-5f && std::abs(instance.transform[0][3] - 11.0f) < 1e-5f);
    EXPECT(std::abs(instance.transform[1][3] - 2.0f) < 1e-5f && std::abs(instance.transform[2][3] - 3.0f) < 1e-5f);
    EXPECT(std::abs(model->boundsMax.x - 13.0f) < 1e-4f && std::abs(model->boundsMax.y - 4.0f) < 1e-4f);
    const renderer::Material& m = scene.materials.back();
    EXPECT(m.name == "Paint" && std::abs(m.roughness - 0.3f) < 1e-6f && std::abs(m.metallic - 0.7f) < 1e-6f);
    EXPECT(std::abs(m.baseColor.y - 0.25f) < 1e-6f && m.emissionNits > 0.0f && m.textures == renderer::kNoTextures);
    EXPECT(renderer::Validate(scene).empty());
}

// An OBJ through ufbx: centimetres to metres by an override, texture coordinates counted from the
// top of the image, a material replaced by name and one left out with its geometry.
TEST(SceneFiles_ImportsObj)
{
    if (!renderer::ModelFilesAvailable())
    {
        return;
    }
    const std::filesystem::path dir = TempDir("obj");
    std::ofstream(dir / "tri.obj") << "v 0 0 0\nv 100 0 0\nv 0 100 0\nv 0 0 50\n"
                                      "vt 0 0\nvt 1 0\nvt 0 1\n"
                                      "usemtl Kept\nf 1/1 2/2 3/3\nusemtl Dropped\nf 1/1 3/3 4/1\n";
    renderer::SceneDesc scene = airtest::BuiltinScene("cornell_box");
    renderer::ImportOptions options;
    options.metresPerUnit = 0.01f;
    options.materialFor = [](std::string_view name) -> std::optional<renderer::Material>
    {
        if (name == "Kept")
        {
            return renderer::Material{.baseColor = {0.1f, 0.2f, 0.3f}, .roughness = 0.9f};
        }
        return std::nullopt;
    };
    options.skipMaterial = [](std::string_view name) { return name == "Dropped"; };
    std::string error;
    const std::optional<renderer::ImportedModel> model = renderer::ImportModel(scene, dir / "tri.obj", options, &error);
    EXPECT(model.has_value());
    if (!model)
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
        return;
    }
    EXPECT(model->instanceCount == 1);
    const renderer::Mesh& mesh = scene.meshes[scene.instances[model->firstInstance].mesh];
    EXPECT(mesh.indices.size() == 3);
    EXPECT(std::abs(model->boundsMax.x - 1.0f) < 1e-4f && std::abs(model->boundsMax.y - 1.0f) < 1e-4f);
    EXPECT(std::abs(model->boundsMax.z) < 1e-4f); // the dropped triangle's vertex is not there
    bool flipped = false;
    for (const gpu::Vertex& v : mesh.vertices)
    {
        flipped = flipped || (v.position.y > 0.5f && std::abs(v.v) < 1e-6f); // vt 0 1: the top row
    }
    EXPECT(flipped);
    const renderer::Material& m = scene.materials[scene.instances[model->firstInstance].material];
    EXPECT(m.name == "Kept" && std::abs(m.roughness - 0.9f) < 1e-6f);
}

// Every scene by one name list; a file scene without its assets says how to fetch them.
TEST(SceneFiles_NamesAndMissingAssets)
{
    const std::vector<std::string> names = renderer::SceneNames();
    for (const char* name : {"cornell_box", "coated", "picapica", "sponza", "helmets"})
    {
        EXPECT(std::find(names.begin(), names.end(), name) != names.end());
    }
    EXPECT(renderer::MakeScene("cornell_box").has_value());
    std::string error;
    EXPECT(!renderer::MakeScene("no_such_scene", &error) && error.find("unknown scene") != std::string::npos);
    const char* previous = std::getenv("AIR_SCENE_ASSETS");
    const std::string saved = previous ? previous : "";
    SetEnv("AIR_SCENE_ASSETS", TempDir("assets").string());
    EXPECT(!renderer::MakeScene("picapica", &error) && error.find("fetch-scenes picapica") != std::string::npos);
    SetEnv("AIR_SCENE_ASSETS", saved);
}
