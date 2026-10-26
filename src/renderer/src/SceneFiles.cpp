// Model files into scenes (SceneFiles.h): FBX and OBJ through ufbx, glTF through cgltf, textures through
// stb_image, packed into the renderer's MaterialTextures.

#include "air/renderer/SceneFiles.h"

#include "ParallelFor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>

#if AIR_WITH_MODEL_FILES
#include "cgltf.h"
#include "stb_image.h"
#include "ufbx.h"
#endif

namespace air::renderer
{

std::filesystem::path SceneAssetsDirectory()
{
    if (const char* dir = std::getenv("AIR_SCENE_ASSETS"); dir && *dir)
    {
        return dir;
    }
#ifdef AIR_SCENE_ASSETS_DIR
    return AIR_SCENE_ASSETS_DIR;
#else
    return ".deps/scenes";
#endif
}

bool ModelFilesAvailable()
{
#if AIR_WITH_MODEL_FILES
    return true;
#else
    return false;
#endif
}

namespace
{

u32 Channel(u32 texel, u32 channel)
{
    return (texel >> (8u * channel)) & 0xFFu;
}

// `image` at w x h: the mean of the texels each target texel covers when shrinking, bilinear (repeating)
// when growing, per channel in its stored encoding.
std::vector<u32> Resample(const TextureImage& image, u32 w, u32 h)
{
    if (image.width == w && image.height == h)
    {
        return image.texels;
    }
    std::vector<u32> out(size_t(w) * h);
    const float sx = float(image.width) / float(w);
    const float sy = float(image.height) / float(h);
    for (u32 y = 0; y < h; ++y)
    {
        for (u32 x = 0; x < w; ++x)
        {
            float sum[4] = {};
            float weight = 0.0f;
            auto add = [&](u32 u, u32 v, float wgt)
            {
                const u32 t = image.texels[size_t(v % image.height) * image.width + u % image.width];
                for (u32 c = 0; c < 4; ++c)
                {
                    sum[c] += wgt * float(Channel(t, c));
                }
                weight += wgt;
            };
            if (sx >= 1.0f && sy >= 1.0f)
            {
                const u32 x0 = u32(float(x) * sx);
                const u32 y0 = u32(float(y) * sy);
                const u32 x1 = std::max(x0 + 1, u32(float(x + 1) * sx));
                const u32 y1 = std::max(y0 + 1, u32(float(y + 1) * sy));
                for (u32 v = y0; v < y1; ++v)
                {
                    for (u32 u = x0; u < x1; ++u)
                    {
                        add(u, v, 1.0f);
                    }
                }
            }
            else
            {
                const float fx = (float(x) + 0.5f) * sx - 0.5f + float(image.width);
                const float fy = (float(y) + 0.5f) * sy - 0.5f + float(image.height);
                const u32 u0 = u32(fx);
                const u32 v0 = u32(fy);
                const float ax = fx - float(u0);
                const float ay = fy - float(v0);
                add(u0, v0, (1.0f - ax) * (1.0f - ay));
                add(u0 + 1, v0, ax * (1.0f - ay));
                add(u0, v0 + 1, (1.0f - ax) * ay);
                add(u0 + 1, v0 + 1, ax * ay);
            }
            u32 texel = 0;
            for (u32 c = 0; c < 4; ++c)
            {
                texel |= u32(std::clamp(std::lround(sum[c] / weight), 0l, 255l)) << (8u * c);
            }
            out[size_t(y) * w + x] = texel;
        }
    }
    return out;
}

#if AIR_WITH_MODEL_FILES

float Luminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// The transform `a` after `b` (both row-major 3x4 object-to-world).
void Compose(const float a[3][4], const float b[3][4], float out[3][4])
{
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] + (c == 3 ? a[r][3] : 0.0f);
        }
    }
}

// What an import collects before the textures are decoded: the materials, each with the images of
// its texture set (empty paths: none).
struct PendingMaterial
{
    Material material;
    std::filesystem::path color, normal, roughness, metallic;
    u32 roughnessChannel = 1; // glTF: green
    u32 metallicChannel = 2;  // glTF: blue
    bool normalGreenUp = true;
    bool HasTextures() const { return !color.empty() || !normal.empty() || !roughness.empty() || !metallic.empty(); }
};

// Decodes the materials' images (each once, in parallel), packs their texture sets and adds the
// materials to the scene; returns their indices.
std::optional<std::vector<u32>> AddMaterials(SceneDesc& scene, std::vector<PendingMaterial>& pending, u32 maxSize,
                                             std::string* error)
{
    std::map<std::filesystem::path, std::optional<TextureImage>> images;
    for (const PendingMaterial& p : pending)
    {
        for (const std::filesystem::path* file : {&p.color, &p.normal, &p.roughness, &p.metallic})
        {
            if (!file->empty())
            {
                images.emplace(*file, std::nullopt);
            }
        }
    }
    std::vector<std::pair<const std::filesystem::path, std::optional<TextureImage>>*> work;
    for (auto& entry : images)
    {
        work.push_back(&entry);
    }
    std::mutex problems;
    std::string firstProblem;
    detail::ParallelFor(int(work.size()),
                        [&](int i)
                        {
                            std::string problem;
                            work[size_t(i)]->second = LoadTextureImage(work[size_t(i)]->first, &problem);
                            if (!work[size_t(i)]->second)
                            {
                                const std::lock_guard lock(problems);
                                firstProblem = firstProblem.empty() ? problem : firstProblem;
                            }
                        });
    if (!firstProblem.empty())
    {
        if (error)
        {
            *error = firstProblem;
        }
        return std::nullopt;
    }
    auto image = [&](const std::filesystem::path& file) -> const TextureImage*
    { return file.empty() ? nullptr : &*images.at(file); };
    std::vector<MaterialTextures> sets(pending.size());
    detail::ParallelFor(int(pending.size()),
                        [&](int i)
                        {
                            const PendingMaterial& p = pending[size_t(i)];
                            if (p.HasTextures())
                            {
                                sets[size_t(i)] =
                                    PackMaterialTextures(p.material.name, image(p.color), image(p.normal),
                                                         p.normalGreenUp, image(p.roughness), p.roughnessChannel,
                                                         image(p.metallic), p.metallicChannel, maxSize);
                            }
                        });
    std::vector<u32> indices;
    for (size_t i = 0; i < pending.size(); ++i)
    {
        Material m = pending[i].material;
        if (pending[i].HasTextures())
        {
            m.textures = u32(scene.textureSets.size());
            scene.textureSets.push_back(std::move(sets[i]));
        }
        indices.push_back(u32(scene.materials.size()));
        scene.materials.push_back(std::move(m));
    }
    return indices;
}

// A texture the file names: as named, else by its file name next to the model (exports often keep the
// author's folders).
std::filesystem::path FindTexture(const std::filesystem::path& folder, const std::string& named)
{
    if (named.empty())
    {
        return {};
    }
    std::string clean = named;
    std::replace(clean.begin(), clean.end(), '\\', '/');
    const std::filesystem::path path(clean);
    std::error_code ec;
    if (std::filesystem::exists(folder / path, ec))
    {
        return folder / path;
    }
    if (std::filesystem::exists(path, ec))
    {
        return path;
    }
    if (std::filesystem::exists(folder / path.filename(), ec))
    {
        return folder / path.filename();
    }
    return {};
}

void GrowBounds(ImportedModel& model, float3 p)
{
    model.boundsMin = hlsl::min(model.boundsMin, p);
    model.boundsMax = hlsl::max(model.boundsMax, p);
}

// Smooth normals from the faces where a file has none.
void ComputeNormals(Mesh& mesh)
{
    for (gpu::Vertex& v : mesh.vertices)
    {
        v.normal = float3(0.0f);
    }
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
    {
        gpu::Vertex& a = mesh.vertices[mesh.indices[i]];
        gpu::Vertex& b = mesh.vertices[mesh.indices[i + 1]];
        gpu::Vertex& c = mesh.vertices[mesh.indices[i + 2]];
        const float3 n = hlsl::cross(b.position - a.position, c.position - a.position); // area weighted
        a.normal += n;
        b.normal += n;
        c.normal += n;
    }
    for (gpu::Vertex& v : mesh.vertices)
    {
        const float len = hlsl::length(v.normal);
        v.normal = len > 0.0f ? v.normal / len : float3(0.0f, 1.0f, 0.0f);
    }
}

std::string ToString(ufbx_string s)
{
    return std::string(s.data, s.length);
}

std::optional<ImportedModel> ImportUfbx(SceneDesc& scene, const std::filesystem::path& file,
                                        const ImportOptions& options, std::string* error)
{
    ufbx_load_opts opts = {};
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.target_unit_meters = 1.0f;
    opts.space_conversion = UFBX_SPACE_CONVERSION_MODIFY_GEOMETRY;
    opts.generate_missing_normals = true;
    ufbx_error problem;
    const std::string path = file.string();
    std::unique_ptr<ufbx_scene, void (*)(ufbx_scene*)> fbx(ufbx_load_file(path.c_str(), &opts, &problem),
                                                           ufbx_free_scene);
    if (!fbx)
    {
        if (error)
        {
            *error = std::format("{}: {}", path, std::string(problem.description.data, problem.description.length));
        }
        return std::nullopt;
    }
    // ufbx converted to metres by the file's unit; an override scales by what it should have been.
    const double fileUnit = fbx->settings.unit_meters > 0.0 ? fbx->settings.unit_meters : 1.0;
    const float scale = options.metresPerUnit > 0.0f ? float(options.metresPerUnit / fileUnit) : 1.0f;
    const std::filesystem::path folder = file.parent_path();

    // Materials: the file's (or the override), each once.
    std::vector<PendingMaterial> pending;
    std::vector<bool> skipped;
    std::map<const ufbx_material*, size_t> materialIndex;
    auto materialOf = [&](const ufbx_material* m) -> size_t
    {
        if (auto it = materialIndex.find(m); it != materialIndex.end())
        {
            return it->second;
        }
        const std::string name = m ? ToString(m->name) : "Default";
        PendingMaterial p;
        p.material.name = name;
        p.normalGreenUp = true;
        if (std::optional<Material> override = options.materialFor ? options.materialFor(name) : std::nullopt)
        {
            p.material = *override;
            p.material.name = name;
        }
        else if (m)
        {
            const ufbx_material_map& base = m->pbr.base_color;
            p.material.baseColor = float3(float(base.value_vec4.x), float(base.value_vec4.y), float(base.value_vec4.z));
            if (m->pbr.roughness.has_value)
            {
                p.material.roughness = std::clamp(float(m->pbr.roughness.value_real), 0.0f, 1.0f);
            }
            if (m->pbr.metalness.has_value)
            {
                p.material.metallic = std::clamp(float(m->pbr.metalness.value_real), 0.0f, 1.0f);
            }
            if (base.texture)
            {
                p.color = FindTexture(folder, ToString(base.texture->relative_filename));
                if (p.color.empty())
                {
                    p.color = FindTexture(folder, ToString(base.texture->filename));
                }
                if (!p.color.empty())
                {
                    p.material.baseColor = float3(1.0f);
                }
            }
            if (m->pbr.normal_map.texture)
            {
                p.normal = FindTexture(folder, ToString(m->pbr.normal_map.texture->relative_filename));
            }
        }
        const size_t index = pending.size();
        pending.push_back(std::move(p));
        skipped.push_back(options.skipMaterial && options.skipMaterial(name));
        materialIndex.emplace(m, index);
        return index;
    };

    // Meshes: one per mesh and material part, positions and normals as ufbx converted them, texture
    // coordinates with v flipped (FBX and OBJ count it upwards, the renderer from the top row).
    struct Part
    {
        u32 mesh = 0;
        size_t material = 0;
    };
    std::map<std::pair<const ufbx_mesh*, size_t>, Part> parts;
    std::vector<std::pair<std::pair<const ufbx_mesh*, size_t>, const ufbx_node*>> placements;
    std::vector<Mesh> meshes;
    for (const ufbx_node* node : fbx->nodes)
    {
        const ufbx_mesh* mesh = node->mesh;
        if (!mesh || (options.skipNode && options.skipNode(ToString(node->name))))
        {
            continue;
        }
        for (size_t pi = 0; pi < mesh->material_parts.count; ++pi)
        {
            const ufbx_mesh_part& part = mesh->material_parts.data[pi];
            if (part.num_triangles == 0)
            {
                continue;
            }
            const ufbx_material* material =
                part.index < mesh->materials.count ? mesh->materials.data[part.index] : nullptr;
            const size_t mi = materialOf(material);
            if (skipped[mi])
            {
                continue;
            }
            const auto key = std::make_pair(mesh, pi);
            if (!parts.contains(key))
            {
                std::vector<gpu::Vertex> corners;
                corners.reserve(part.num_triangles * 3);
                std::vector<uint32_t> triangle(mesh->max_face_triangles * 3);
                for (size_t f = 0; f < part.face_indices.count; ++f)
                {
                    const ufbx_face face = mesh->faces.data[part.face_indices.data[f]];
                    const u32 count = ufbx_triangulate_face(triangle.data(), triangle.size(), mesh, face);
                    for (u32 k = 0; k < count * 3; ++k)
                    {
                        const uint32_t i = triangle[k];
                        const ufbx_vec3 p = ufbx_get_vertex_vec3(&mesh->vertex_position, i);
                        const ufbx_vec3 n = mesh->vertex_normal.exists ? ufbx_get_vertex_vec3(&mesh->vertex_normal, i)
                                                                       : ufbx_vec3{0.0, 1.0, 0.0};
                        const ufbx_vec2 uv =
                            mesh->vertex_uv.exists ? ufbx_get_vertex_vec2(&mesh->vertex_uv, i) : ufbx_vec2{0.0, 0.0};
                        corners.push_back({float3(float(p.x), float(p.y), float(p.z)), float(uv.x),
                                           hlsl::normalize(float3(float(n.x), float(n.y), float(n.z))),
                                           1.0f - float(uv.y)});
                    }
                }
                Mesh out;
                out.name =
                    std::format("{}.{}", ToString(mesh->name).empty() ? ToString(node->name) : ToString(mesh->name),
                                pending[mi].material.name);
                out.indices.resize(corners.size());
                ufbx_vertex_stream stream = {corners.data(), corners.size(), sizeof(gpu::Vertex)};
                ufbx_error indexError;
                const size_t unique =
                    ufbx_generate_indices(&stream, 1, out.indices.data(), out.indices.size(), nullptr, &indexError);
                corners.resize(unique);
                out.vertices = std::move(corners);
                parts.emplace(key, Part{u32(meshes.size()), mi});
                meshes.push_back(std::move(out));
            }
            placements.emplace_back(key, node);
        }
    }
    if (placements.empty())
    {
        if (error)
        {
            *error = std::format("{}: no geometry", path);
        }
        return std::nullopt;
    }
    const std::optional<std::vector<u32>> materials = AddMaterials(scene, pending, options.maxTextureSize, error);
    if (!materials)
    {
        return std::nullopt;
    }
    const u32 meshBase = u32(scene.meshes.size());
    for (Mesh& mesh : meshes)
    {
        scene.meshes.push_back(std::move(mesh));
    }
    ImportedModel model;
    model.firstInstance = u32(scene.instances.size());
    model.boundsMin = float3(1e30f);
    model.boundsMax = float3(-1e30f);
    for (const auto& [key, node] : placements)
    {
        const Part& part = parts.at(key);
        float local[3][4];
        const ufbx_matrix& m = node->geometry_to_world;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                local[r][c] = float(m.cols[c].v[r]) * scale;
            }
        }
        Instance instance;
        instance.name = ToString(node->name);
        instance.mesh = meshBase + part.mesh;
        instance.material = (*materials)[part.material];
        Compose(options.transform, local, instance.transform);
        for (const gpu::Vertex& v : scene.meshes[instance.mesh].vertices)
        {
            GrowBounds(model, TransformPoint(instance.transform, v.position));
        }
        scene.instances.push_back(std::move(instance));
    }
    model.instanceCount = u32(scene.instances.size()) - model.firstInstance;
    return model;
}

std::optional<ImportedModel> ImportGltf(SceneDesc& scene, const std::filesystem::path& file,
                                        const ImportOptions& options, std::string* error)
{
    const std::string path = file.string();
    cgltf_options opts = {};
    cgltf_data* raw = nullptr;
    cgltf_result result = cgltf_parse_file(&opts, path.c_str(), &raw);
    std::unique_ptr<cgltf_data, void (*)(cgltf_data*)> data(raw, cgltf_free);
    if (result == cgltf_result_success)
    {
        result = cgltf_load_buffers(&opts, data.get(), path.c_str());
    }
    if (result == cgltf_result_success)
    {
        result = cgltf_validate(data.get());
    }
    if (result != cgltf_result_success)
    {
        if (error)
        {
            *error = std::format("{}: cannot read it as glTF (cgltf error {})", path, int(result));
        }
        return std::nullopt;
    }
    const float scale = options.metresPerUnit > 0.0f ? options.metresPerUnit : 1.0f;
    const std::filesystem::path folder = file.parent_path();
    auto imageFile = [&](const cgltf_texture_view& view) -> std::filesystem::path
    {
        if (!view.texture || !view.texture->image || !view.texture->image->uri ||
            std::strncmp(view.texture->image->uri, "data:", 5) == 0)
        {
            return {};
        }
        std::string uri = view.texture->image->uri;
        cgltf_decode_uri(uri.data());
        uri.resize(std::strlen(uri.c_str()));
        return FindTexture(folder, uri);
    };

    std::vector<PendingMaterial> pending;
    std::vector<bool> skipped;
    for (size_t i = 0; i < data->materials_count; ++i)
    {
        const cgltf_material& m = data->materials[i];
        PendingMaterial p;
        p.material.name = m.name ? m.name : std::format("Material{}", i);
        p.normalGreenUp = true; // glTF: +y up in the image, towards decreasing v
        if (std::optional<Material> override =
                options.materialFor ? options.materialFor(p.material.name) : std::nullopt)
        {
            const std::string name = p.material.name;
            p.material = *override;
            p.material.name = name;
        }
        else
        {
            if (m.has_pbr_metallic_roughness)
            {
                const cgltf_pbr_metallic_roughness& pbr = m.pbr_metallic_roughness;
                p.material.baseColor =
                    float3(pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2]);
                p.material.roughness = std::clamp(pbr.roughness_factor, 0.0f, 1.0f);
                p.material.metallic = std::clamp(pbr.metallic_factor, 0.0f, 1.0f);
                p.color = imageFile(pbr.base_color_texture);
                p.roughness = imageFile(pbr.metallic_roughness_texture);
                p.metallic = p.roughness;
            }
            p.normal = imageFile(m.normal_texture);
            if (m.alpha_mode != cgltf_alpha_mode_opaque && !p.color.empty())
            {
                p.material.alphaCutoff = m.alpha_mode == cgltf_alpha_mode_mask ? m.alpha_cutoff : 0.5f;
            }
            // Constant emission only: an emissive texture (lights painted on a surface) is left out.
            const float3 emissive(m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2]);
            if (!m.emissive_texture.texture && Luminance(emissive) > 0.0f)
            {
                const float strength = m.has_emissive_strength ? m.emissive_strength.emissive_strength : 1.0f;
                p.material.emissionColor = emissive;
                p.material.emissionNits = 100.0f * strength * Luminance(emissive);
            }
            if (m.has_clearcoat && m.clearcoat.clearcoat_factor > 0.5f)
            {
                p.material.coat = true;
                p.material.coatRoughness = std::clamp(m.clearcoat.clearcoat_roughness_factor, 0.0f, 1.0f);
            }
        }
        skipped.push_back(options.skipMaterial && options.skipMaterial(p.material.name));
        pending.push_back(std::move(p));
    }
    PendingMaterial fallback; // for primitives without a material, added only when one has none
    fallback.material.name = "Default";
    const size_t defaultMaterial = pending.size();
    pending.push_back(fallback);
    skipped.push_back(false);
    bool defaultUsed = false;

    std::map<std::pair<size_t, size_t>, std::pair<u32, size_t>> primitives; // mesh, primitive -> mesh, material
    std::vector<Mesh> meshes;
    auto meshOf = [&](size_t mi, size_t pi) -> std::optional<std::pair<u32, size_t>>
    {
        const auto key = std::make_pair(mi, pi);
        if (auto it = primitives.find(key); it != primitives.end())
        {
            return it->second;
        }
        const cgltf_primitive& prim = data->meshes[mi].primitives[pi];
        const size_t material = prim.material ? size_t(prim.material - data->materials) : defaultMaterial;
        if (prim.type != cgltf_primitive_type_triangles || skipped[material])
        {
            return std::nullopt;
        }
        defaultUsed = defaultUsed || material == defaultMaterial;
        const cgltf_accessor* positions = nullptr;
        const cgltf_accessor* normals = nullptr;
        const cgltf_accessor* uvs = nullptr;
        for (size_t a = 0; a < prim.attributes_count; ++a)
        {
            const cgltf_attribute& attribute = prim.attributes[a];
            if (attribute.type == cgltf_attribute_type_position)
            {
                positions = attribute.data;
            }
            else if (attribute.type == cgltf_attribute_type_normal)
            {
                normals = attribute.data;
            }
            else if (attribute.type == cgltf_attribute_type_texcoord && attribute.index == 0)
            {
                uvs = attribute.data;
            }
        }
        if (!positions || positions->count == 0)
        {
            return std::nullopt;
        }
        Mesh out;
        out.name = std::format("{}.{}", data->meshes[mi].name ? data->meshes[mi].name : "Mesh", pi);
        out.vertices.resize(positions->count);
        for (size_t v = 0; v < positions->count; ++v)
        {
            float p[3] = {};
            float n[3] = {0.0f, 1.0f, 0.0f};
            float uv[2] = {};
            cgltf_accessor_read_float(positions, v, p, 3);
            if (normals)
            {
                cgltf_accessor_read_float(normals, v, n, 3);
            }
            if (uvs)
            {
                cgltf_accessor_read_float(uvs, v, uv, 2);
            }
            out.vertices[v] = {float3(p[0], p[1], p[2]), uv[0], float3(n[0], n[1], n[2]), uv[1]};
        }
        if (prim.indices)
        {
            out.indices.resize(prim.indices->count);
            for (size_t i = 0; i < prim.indices->count; ++i)
            {
                out.indices[i] = u32(cgltf_accessor_read_index(prim.indices, i));
            }
        }
        else
        {
            out.indices.resize(positions->count);
            for (size_t i = 0; i < positions->count; ++i)
            {
                out.indices[i] = u32(i);
            }
        }
        out.indices.resize(out.indices.size() / 3 * 3);
        if (!normals)
        {
            ComputeNormals(out);
        }
        const std::pair<u32, size_t> made(u32(meshes.size()), material);
        meshes.push_back(std::move(out));
        primitives.emplace(key, made);
        return made;
    };

    const cgltf_scene* gltfScene = data->scene ? data->scene : (data->scenes_count > 0 ? &data->scenes[0] : nullptr);
    std::vector<std::pair<std::pair<u32, size_t>, std::array<float, 16>>> placements;
    std::vector<const cgltf_node*> stack;
    if (gltfScene)
    {
        stack.assign(gltfScene->nodes, gltfScene->nodes + gltfScene->nodes_count);
    }
    else
    {
        for (size_t i = 0; i < data->nodes_count; ++i)
        {
            if (!data->nodes[i].parent)
            {
                stack.push_back(&data->nodes[i]);
            }
        }
    }
    while (!stack.empty())
    {
        const cgltf_node* node = stack.back();
        stack.pop_back();
        for (size_t c = 0; c < node->children_count; ++c)
        {
            stack.push_back(node->children[c]);
        }
        if (!node->mesh)
        {
            continue;
        }
        std::array<float, 16> world{};
        cgltf_node_transform_world(node, world.data());
        const size_t mi = size_t(node->mesh - data->meshes);
        for (size_t pi = 0; pi < node->mesh->primitives_count; ++pi)
        {
            if (const std::optional<std::pair<u32, size_t>> made = meshOf(mi, pi))
            {
                placements.emplace_back(*made, world);
            }
        }
    }
    if (placements.empty())
    {
        if (error)
        {
            *error = std::format("{}: no geometry", path);
        }
        return std::nullopt;
    }
    if (!defaultUsed)
    {
        pending.pop_back();
    }
    const std::optional<std::vector<u32>> materials = AddMaterials(scene, pending, options.maxTextureSize, error);
    if (!materials)
    {
        return std::nullopt;
    }
    const u32 meshBase = u32(scene.meshes.size());
    for (Mesh& mesh : meshes)
    {
        scene.meshes.push_back(std::move(mesh));
    }
    ImportedModel model;
    model.firstInstance = u32(scene.instances.size());
    model.boundsMin = float3(1e30f);
    model.boundsMax = float3(-1e30f);
    for (const auto& [made, world] : placements)
    {
        float local[3][4];
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                local[r][c] = world[size_t(c) * 4 + size_t(r)] * scale; // column-major
            }
        }
        Instance instance;
        instance.name = scene.meshes[meshBase + made.first].name;
        instance.mesh = meshBase + made.first;
        instance.material = (*materials)[made.second];
        Compose(options.transform, local, instance.transform);
        for (const gpu::Vertex& v : scene.meshes[instance.mesh].vertices)
        {
            GrowBounds(model, TransformPoint(instance.transform, v.position));
        }
        scene.instances.push_back(std::move(instance));
    }
    model.instanceCount = u32(scene.instances.size()) - model.firstInstance;
    return model;
}

#endif

} // namespace

std::optional<TextureImage> LoadTextureImage(const std::filesystem::path& file, std::string* error)
{
#if AIR_WITH_MODEL_FILES
    std::ifstream in(file, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    int w = 0;
    int h = 0;
    int channels = 0;
    stbi_uc* pixels =
        bytes.empty() ? nullptr : stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &channels, 4);
    if (!pixels)
    {
        if (error)
        {
            *error = std::format("{}: {}", file.string(), bytes.empty() ? "cannot read it" : stbi_failure_reason());
        }
        return std::nullopt;
    }
    TextureImage image;
    image.width = u32(w);
    image.height = u32(h);
    image.texels.resize(size_t(w) * size_t(h));
    std::memcpy(image.texels.data(), pixels, image.texels.size() * 4);
    stbi_image_free(pixels);
    return image;
#else
    if (error)
    {
        *error = std::format("{}: built without the model file libraries (AIR_WITH_MODEL_FILES)", file.string());
    }
    return std::nullopt;
#endif
}

MaterialTextures PackMaterialTextures(std::string name, const TextureImage* color, const TextureImage* normal,
                                      bool normalGreenUp, const TextureImage* roughness, u32 roughnessChannel,
                                      const TextureImage* metallic, u32 metallicChannel, u32 maxSize)
{
    u32 w = 1;
    u32 h = 1;
    for (const TextureImage* image : {color, normal, roughness, metallic})
    {
        if (image)
        {
            w = std::max(w, image->width);
            h = std::max(h, image->height);
        }
    }
    const float shrink = std::min(1.0f, float(std::max(maxSize, 1u)) / float(std::max(w, h)));
    w = std::max(1u, u32(std::lround(float(w) * shrink)));
    h = std::max(1u, u32(std::lround(float(h) * shrink)));
    MaterialTextures t;
    t.name = std::move(name);
    t.width = w;
    t.height = h;
    const size_t texels = size_t(w) * h;
    t.color = color ? Resample(*color, w, h) : std::vector<u32>(texels, 0xFFFFFFFFu);
    const std::vector<u32> n = normal ? Resample(*normal, w, h) : std::vector<u32>();
    const std::vector<u32> r = roughness ? Resample(*roughness, w, h) : std::vector<u32>();
    const std::vector<u32> m = metallic == roughness && roughness ? r
                               : metallic                         ? Resample(*metallic, w, h)
                                                                  : std::vector<u32>();
    t.surface.resize(texels);
    for (size_t i = 0; i < texels; ++i)
    {
        const u32 x = n.empty() ? 128u : Channel(n[i], 0);
        const u32 y = n.empty() ? 128u : normalGreenUp ? 255u - Channel(n[i], 1) : Channel(n[i], 1);
        const u32 rough = r.empty() ? 255u : Channel(r[i], roughnessChannel);
        const u32 metal = m.empty() ? 255u : Channel(m[i], metallicChannel);
        t.surface[i] = x | y << 8 | rough << 16 | metal << 24;
    }
    return t;
}

std::optional<ImportedModel> ImportModel(SceneDesc& scene, const std::filesystem::path& file,
                                         const ImportOptions& options, std::string* error)
{
#if AIR_WITH_MODEL_FILES
    std::error_code ec;
    if (!std::filesystem::exists(file, ec))
    {
        if (error)
        {
            *error = std::format("{}: no such file", file.string());
        }
        return std::nullopt;
    }
    std::string extension = file.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (extension == ".gltf" || extension == ".glb")
    {
        return ImportGltf(scene, file, options, error);
    }
    if (extension == ".fbx" || extension == ".obj")
    {
        return ImportUfbx(scene, file, options, error);
    }
    if (error)
    {
        *error = std::format("{}: not a model file this reads (glTF, FBX, OBJ)", file.string());
    }
    return std::nullopt;
#else
    (void)scene;
    (void)options;
    if (error)
    {
        *error = std::format("{}: built without the model file libraries (AIR_WITH_MODEL_FILES)", file.string());
    }
    return std::nullopt;
#endif
}

} // namespace air::renderer
