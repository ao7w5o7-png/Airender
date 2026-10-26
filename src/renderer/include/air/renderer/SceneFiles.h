#pragma once

// Scenes made from model files (docs/architecture.md §8.18). `tools/air.py fetch-scenes` downloads
// their assets from pinned commits into .deps/scenes/<scene>/ (AIR_SCENE_ASSETS elsewhere); nothing
// of them is committed. ImportModel reads FBX and OBJ (ufbx) and glTF (cgltf) into a SceneDesc: a
// mesh per mesh and material, an instance per node, the materials' textures packed into
// MaterialTextures (stb_image). The file scenes compose such models under a sky:
//
//   picapica   the PICA PICA assets of SEED (Electronic Arts; CC BY-NC 4.0): a toy factory of plastic
//              machines, pallets, books and robots around the mini diorama, in the afternoon sun
//   sponza     the Sponza atrium (Crytek; Khronos glTF sample assets, under the license that names):
//              textures, normal maps and cutout foliage under the sky through the open roof
//   helmets    the flight (CC0), damaged (CC BY 4.0) and sci-fi (CC0) helmets of the glTF sample
//              assets: physically based texture sets side by side

#include "air/renderer/Scene.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace air::renderer
{

// Where the file scenes' assets are: AIR_SCENE_ASSETS, else <source>/.deps/scenes.
std::filesystem::path SceneAssetsDirectory();

// Whether the build has the model file libraries (AIR_WITH_MODEL_FILES): without them ImportModel and
// the file scenes report an error.
bool ModelFilesAvailable();

struct ImportOptions
{
    // Metres per unit of the file: 0 takes the file's own (FBX unit scale; glTF metres; OBJ none:
    // 1), a value overrides it (files whose metadata is wrong).
    float metresPerUnit = 0.0f;
    // Placed by this object-to-world transform after the conversion to metres and y up.
    float transform[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    // Textures larger than this are resampled down to it (both textures of a material share a size).
    u32 maxTextureSize = 1024;
    // A material by its name in the file, replacing what the file says (a file without its material
    // definitions, as the PICA PICA assets); nullopt keeps the file's. Its `textures` index the
    // scene's texture sets as they are before the import.
    std::function<std::optional<Material>(std::string_view name)> materialFor;
    // Materials whose name makes this return true are left out with their geometry (decals without
    // their alpha, helper geometry), and nodes likewise (FBX and OBJ; a model's display stand).
    std::function<bool(std::string_view name)> skipMaterial;
    std::function<bool(std::string_view name)> skipNode;
};

struct ImportedModel
{
    u32 firstInstance = 0;
    u32 instanceCount = 0;
    float3 boundsMin{};
    float3 boundsMax{};
};

// Appends the model's meshes, materials, texture sets and instances to `scene`. nullopt with `error`
// when it cannot be read (or the build has no model libraries: AIR_WITH_MODEL_FILES).
std::optional<ImportedModel> ImportModel(SceneDesc& scene, const std::filesystem::path& file,
                                         const ImportOptions& options, std::string* error);

// Packs a material's textures into a MaterialTextures (Scene.h's layout): base colour (sRGB, alpha in
// the colour texture's alpha), tangent-space normal (x, y as stored for increasing u and v, so a glTF
// normal map's green flips), roughness and metallic, all resampled to one size. Any may be absent.
struct TextureImage
{
    u32 width = 0;
    u32 height = 0;
    std::vector<u32> texels; // RGBA8, x fastest, red in the low byte
};
MaterialTextures PackMaterialTextures(std::string name, const TextureImage* color, const TextureImage* normal,
                                      bool normalGreenUp, const TextureImage* roughness, u32 roughnessChannel,
                                      const TextureImage* metallic, u32 metallicChannel, u32 maxSize);
std::optional<TextureImage> LoadTextureImage(const std::filesystem::path& file, std::string* error);

// Places a model's instances (just imported) by a uniform scale and a turn about +y (degrees), then
// moves them so that their footprint's centre is at `at` and their lowest point on it.
void PlaceModel(SceneDesc& scene, const ImportedModel& model, float scale, float yawDegrees, float3 at);

std::vector<std::string> FileSceneNames();
// A file scene; nullopt with `error` saying what is missing and how to fetch it.
std::optional<SceneDesc> MakeFileScene(std::string_view name, std::string* error);

// Every scene by name: the built-in ones (Scene.h) and the file scenes.
std::vector<std::string> SceneNames();
std::optional<SceneDesc> MakeScene(std::string_view name, std::string* error = nullptr);

} // namespace air::renderer
