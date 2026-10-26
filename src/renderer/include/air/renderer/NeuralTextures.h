#pragma once

// Neural texture compression of material texture sets (docs/architecture.md §8.16) with NVIDIA's
// LibNTC (the RTXNTC SDK's prebuilt libntc.dll, loaded at run time from next to the executable):
// a set's seven channels (colour, the normal's x and y, roughness, metallic) become a few small
// grids of latents and the weights of a small MLP that decodes them, at a few bits per texel instead
// of 64. Compression and decompression on load run on CUDA through the driver; for decompression on
// sample LibNTC hands out what a shader needs to decode a texel itself (libntc/shaders/Inference.hlsli,
// int8 dot products), and the path tracer does so at every hit. Windows x64 with an NVIDIA GPU only:
// elsewhere the codec is unavailable and says why.

#include "air/core/Json.h"
#include "air/core/Types.h"
#include "air/renderer/Scene.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace air::renderer
{

// A compressed set: a .ntc stream.
struct CompressedTextures
{
    std::vector<u8> file;
    float bitsPerPixel = 0.0f; // of the latent shape LibNTC chose
    float psnr = 0.0f;         // over all its channels decoded with the int8 weights, dB
    bool cached = false;       // found in the cache rather than compressed
};

// What a shader needs to decode one set's texels (Inference on Sample).
struct NeuralTextureGpuData
{
    // The latents: an RGBA8 2D array of latentLayers x latentMips subresources, layer by layer, each of
    // latentWidth >> mip x latentHeight >> mip texels (LibNTC's BGRA4 texels widened, which filter the
    // same: x / 15 is exactly representable in 8 bits).
    u32 latentWidth = 0;
    u32 latentHeight = 0;
    u32 latentLayers = 0;
    u32 latentMips = 0;
    std::vector<std::vector<u8>> latents;
    // A ByteAddressBuffer: NtcTextureSetConstants (gpu::kNeuralTextureConstantBytes), then the MLP's
    // int8 weights.
    std::vector<u8> data;
    // The decoded set at a quarter of its resolution (box filtered, the colour in linear light), which
    // the path tracer reads beyond the first surface instead of decoding (MaterialTextures' layout).
    u32 coarseWidth = 0;
    u32 coarseHeight = 0;
    std::vector<u32> coarseColor;
    std::vector<u32> coarseSurface;
};

inline constexpr u32 kNeuralTextureCoarseFactor = 4;

// Fills `out`'s coarse copy from a set's (decoded) texels.
void MakeNeuralTextureCoarseCopy(const MaterialTextures& textures, NeuralTextureGpuData& out);

class NeuralTextureCodec
{
public:
    // The codec, or nullptr (with the reason in *why) when LibNTC or a CUDA device is missing.
    static NeuralTextureCodec* Get(std::string* why = nullptr);

    // Compresses a set at about bitsPerPixel with `steps` training steps. With a cache directory the
    // result is kept there by the set's content and the settings, and found again instead of
    // compressing (seconds to tens of seconds a set).
    std::optional<CompressedTextures> Compress(const MaterialTextures& textures, float bitsPerPixel, u32 steps,
                                               const std::filesystem::path& cache, std::string* error = nullptr);
    // The set decoded on the GPU with the int8 weights (as a shader decodes it on sample), into RGBA8
    // texels like the source's.
    bool Decompress(const CompressedTextures& compressed, MaterialTextures& textures, std::string* error = nullptr);
    // The same in float32: gpu::kNeuralTextureChannels values per texel, x fastest, in the colour
    // spaces they are stored in (the colour in sRGB).
    std::optional<std::vector<float>> DecodeChannels(const CompressedTextures& compressed,
                                                     std::string* error = nullptr);
    // What a shader needs to decode its texels on sample.
    std::optional<NeuralTextureGpuData> PrepareInference(const CompressedTextures& compressed,
                                                         std::string* error = nullptr);
};

enum class NeuralTextureMode
{
    Off,
    OnLoad,   // decoded once, and rendered from ordinary textures
    OnSample, // decoded by the shader at every texture lookup, from latents and weights only
};

struct NeuralTextureOptions
{
    NeuralTextureMode mode = NeuralTextureMode::Off;
    float bitsPerPixel = 4.0f;
    // Training steps: 20000 take about 4 s a set on an RTX 5070 (the time does not depend on its size);
    // LibNTC's default of 100000 adds a few dB.
    u32 steps = 20000;
    std::filesystem::path cache = std::filesystem::temp_directory_path() / "airender-ntc";
};

struct NeuralTextureReport
{
    std::string name;
    float bitsPerPixel = 0.0f;
    float psnr = 0.0f;
    u64 sourceBytes = 0;     // the RGBA8 textures
    u64 compressedBytes = 0; // the .ntc stream
    bool cached = false;
};

// Puts every texture set of `scene` through neural texture compression: its texels become the
// decoded ones (what both modes render, so the CPU reference sees them too), and with OnSample the
// set also carries what the GPU decodes them from (MaterialTextures::neural). False, with the scene
// unchanged, when the codec is unavailable or a set fails.
bool ApplyNeuralTextures(SceneDesc& scene, const NeuralTextureOptions& options,
                         std::vector<NeuralTextureReport>* report = nullptr, std::string* error = nullptr);

// The shader's decoding (NtcCheck.hlsl: libntc/shaders/Inference.hlsli at every texel centre, the code
// the path tracer runs on sample) against LibNTC's on CUDA, both with the int8 weights, for the
// `textured` scene's sets compressed with `options`: per set and channel the mean difference, its root
// mean square and the largest (aircli ntc-check). "passed" when they agree.
json::Value CheckNeuralTextures(rhi::Device& device, const std::filesystem::path& rendererShaderDir,
                                const NeuralTextureOptions& options);

} // namespace air::renderer
