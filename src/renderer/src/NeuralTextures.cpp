#include "air/renderer/NeuralTextures.h"

#include "air/core/Log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <mutex>

#if AIR_WITH_NTC
#include <windows.h>

#include <libntc/ntc.h>
#endif

namespace air::renderer
{
namespace
{

using gpu::kNeuralTextureConstantBytes;

// The set's channels in the texture set LibNTC compresses.
constexpr int kColorChannels = 3;   // 0-2, sRGB
constexpr int kSurfaceChannels = 4; // 3-6: normal x, normal y, roughness, metallic
constexpr int kChannels = kColorChannels + kSurfaceChannels;

u64 Fnv1a(u64 hash, const void* data, size_t size)
{
    const auto* bytes = static_cast<const u8*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        hash = (hash ^ bytes[i]) * 0x100000001b3ull;
    }
    return hash;
}

// A cache entry: this magic, the rate and the PSNR, then the .ntc stream.
constexpr char kCacheMagic[8] = {'A', 'I', 'R', 'N', 'T', 'C', '0', '1'};

#if AIR_WITH_NTC

struct Library
{
    HMODULE module = nullptr;
    decltype(&ntc::GetInterfaceVersion) getInterfaceVersion = nullptr;
    decltype(&ntc::CreateContext) createContext = nullptr;
    decltype(&ntc::GetLastErrorMessage) getLastErrorMessage = nullptr;
    decltype(&ntc::StatusToString) statusToString = nullptr;
    decltype(&ntc::PickLatentShape) pickLatentShape = nullptr;
    decltype(&ntc::LossToPSNR) lossToPsnr = nullptr;
    ntc::IContext* context = nullptr; // never destroyed: CUDA may already be gone at exit
    std::string why;                  // set when the codec is unavailable
};

// LibNTC is not thread safe across one context: every call goes through this lock.
std::mutex gLock;

Library& Load()
{
    static Library library = []
    {
        Library l;
        wchar_t exe[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        const std::filesystem::path dll = std::filesystem::path(exe).parent_path() / "libntc.dll";
        l.module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!l.module)
        {
            l.why = std::format("{} could not be loaded (error {})", dll.string(), GetLastError());
            return l;
        }
        auto get = [&](auto& function, const char* name)
        {
            function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                reinterpret_cast<void*>(GetProcAddress(l.module, name)));
            if (!function && l.why.empty())
            {
                l.why = std::format("libntc.dll has no {}", name);
            }
        };
        get(l.getInterfaceVersion, "GetInterfaceVersion");
        get(l.createContext, "CreateContext");
        get(l.getLastErrorMessage, "GetLastErrorMessage");
        get(l.statusToString, "StatusToString");
        get(l.pickLatentShape, "PickLatentShape");
        get(l.lossToPsnr, "LossToPSNR");
        if (!l.why.empty())
        {
            return l;
        }
        if (l.getInterfaceVersion() != ntc::InterfaceVersion)
        {
            l.why = std::format("libntc.dll has interface {:#x}, the headers {:#x}", l.getInterfaceVersion(),
                                ntc::InterfaceVersion);
            return l;
        }
        ntc::ContextParameters parameters;
        parameters.cudaDevice = 0;
        parameters.graphicsApi = ntc::GraphicsAPI::None;
        parameters.enableCooperativeVector = false;
        const ntc::Status status = l.createContext(&l.context, parameters);
        if (status != ntc::Status::Ok)
        {
            l.why = std::format("LibNTC: {} ({})", l.statusToString(status), l.getLastErrorMessage());
            l.context = nullptr; // CudaUnavailable leaves a context that cannot compress
        }
        return l;
    }();
    return library;
}

bool Fail(Library& l, ntc::Status status, const char* what, std::string* error)
{
    if (status == ntc::Status::Ok)
    {
        return false;
    }
    if (error)
    {
        *error = std::format("LibNTC {}: {} ({})", what, l.statusToString(status), l.getLastErrorMessage());
    }
    return true;
}

// Owns a texture set or its metadata.
template <typename T, void (ntc::IContext::*Destroy)(T*) const> struct Owned
{
    ntc::IContext* context = nullptr;
    T* object = nullptr;
    ~Owned()
    {
        if (object)
        {
            (context->*Destroy)(object);
        }
    }
    T* operator->() const { return object; }
};
using OwnedSet = Owned<ntc::ITextureSet, &ntc::IContext::DestroyTextureSet>;
using OwnedMetadata = Owned<ntc::ITextureSetMetadata, &ntc::IContext::DestroyTextureSetMetadata>;

#endif

} // namespace

// NeuralTextureGpuData::coarseColor / coarseSurface: box filtered over kNeuralTextureCoarseFactor^2
// texels, the colour in linear light.
void MakeNeuralTextureCoarseCopy(const MaterialTextures& t, NeuralTextureGpuData& out)
{
    const u32 f = kNeuralTextureCoarseFactor;
    out.coarseWidth = std::max(t.width / f, 1u);
    out.coarseHeight = std::max(t.height / f, 1u);
    auto toLinear = [](u32 byte)
    {
        const float s = float(byte) / 255.0f;
        return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    };
    auto toSrgb = [](float l)
    {
        const float s = l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
        return u32(std::clamp(s, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    out.coarseColor.resize(size_t(out.coarseWidth) * out.coarseHeight);
    out.coarseSurface.resize(out.coarseColor.size());
    for (u32 y = 0; y < out.coarseHeight; ++y)
    {
        for (u32 x = 0; x < out.coarseWidth; ++x)
        {
            float color[3] = {};
            float surface[4] = {};
            u32 count = 0;
            for (u32 j = y * f; j < std::min((y + 1) * f, t.height); ++j)
            {
                for (u32 i = x * f; i < std::min((x + 1) * f, t.width); ++i)
                {
                    const u32 c = t.color[size_t(j) * t.width + i];
                    const u32 s = t.surface[size_t(j) * t.width + i];
                    for (u32 k = 0; k < 3; ++k)
                    {
                        color[k] += toLinear((c >> (8 * k)) & 0xFFu);
                    }
                    for (u32 k = 0; k < 4; ++k)
                    {
                        surface[k] += float((s >> (8 * k)) & 0xFFu);
                    }
                    ++count;
                }
            }
            u32 c = 0xFF000000u;
            u32 s = 0;
            for (u32 k = 0; k < 3; ++k)
            {
                c |= toSrgb(color[k] / float(count)) << (8 * k);
            }
            for (u32 k = 0; k < 4; ++k)
            {
                s |= u32(surface[k] / float(count) + 0.5f) << (8 * k);
            }
            out.coarseColor[size_t(y) * out.coarseWidth + x] = c;
            out.coarseSurface[size_t(y) * out.coarseWidth + x] = s;
        }
    }
}

NeuralTextureCodec* NeuralTextureCodec::Get(std::string* why)
{
#if AIR_WITH_NTC
    std::lock_guard lock(gLock);
    Library& l = Load();
    if (!l.context)
    {
        if (why)
        {
            *why = l.why;
        }
        return nullptr;
    }
    static NeuralTextureCodec codec;
    return &codec;
#else
    if (why)
    {
        *why = "built without LibNTC (AIR_WITH_NTC, Windows x64 only)";
    }
    return nullptr;
#endif
}

std::optional<CompressedTextures> NeuralTextureCodec::Compress(const MaterialTextures& textures, float bitsPerPixel,
                                                               u32 steps, const std::filesystem::path& cache,
                                                               std::string* error)
{
#if AIR_WITH_NTC
    // The cache entry: by the texels, their size and the settings.
    u64 key = 0xcbf29ce484222325ull;
    const u32 header[] = {textures.width, textures.height, steps, ntc::InterfaceVersion};
    key = Fnv1a(key, header, sizeof(header));
    key = Fnv1a(key, &bitsPerPixel, sizeof(bitsPerPixel));
    key = Fnv1a(key, textures.color.data(), textures.color.size() * sizeof(u32));
    key = Fnv1a(key, textures.surface.data(), textures.surface.size() * sizeof(u32));
    const std::filesystem::path entry =
        cache.empty() ? std::filesystem::path() : cache / std::format("{}-{:016x}.airntc", textures.name, key);
    if (!entry.empty())
    {
        std::ifstream in(entry, std::ios::binary);
        char magic[8] = {};
        CompressedTextures found;
        found.cached = true;
        if (in.read(magic, sizeof(magic)) && std::memcmp(magic, kCacheMagic, sizeof(magic)) == 0 &&
            in.read(reinterpret_cast<char*>(&found.bitsPerPixel), sizeof(float)) &&
            in.read(reinterpret_cast<char*>(&found.psnr), sizeof(float)))
        {
            found.file.assign(std::istreambuf_iterator<char>(in), {});
            if (!found.file.empty())
            {
                return found;
            }
        }
    }

    std::lock_guard lock(gLock);
    Library& l = Load();
    if (!l.context)
    {
        if (error)
        {
            *error = l.why;
        }
        return std::nullopt;
    }
    ntc::TextureSetDesc desc;
    desc.width = int(textures.width);
    desc.height = int(textures.height);
    desc.channels = kChannels;
    desc.mips = 1;
    ntc::TextureSetFeatures features;
    features.stagingBytesPerPixel = 16;
    features.separateRefOutData = true; // decoding after compression measures against the source
    OwnedSet set{l.context};
    if (Fail(l, l.context->CreateTextureSet(desc, features, &set.object), "CreateTextureSet", error))
    {
        return std::nullopt;
    }
    const ntc::ColorSpace srgb[kColorChannels] = {ntc::ColorSpace::sRGB, ntc::ColorSpace::sRGB, ntc::ColorSpace::sRGB};
    auto write = [&](const std::vector<u32>& texels, int first, int count, const ntc::ColorSpace* spaces)
    {
        ntc::WriteChannelsParameters p;
        p.mipLevel = 0;
        p.firstChannel = first;
        p.numChannels = count;
        p.pData = reinterpret_cast<const unsigned char*>(texels.data());
        p.addressSpace = ntc::AddressSpace::Host;
        p.width = desc.width;
        p.height = desc.height;
        p.pixelStride = 4;
        p.rowPitch = size_t(desc.width) * 4;
        p.channelFormat = ntc::ChannelFormat::UNORM8;
        p.srcColorSpaces = spaces;
        p.dstColorSpaces = spaces; // colour is learnt in sRGB, where its errors are perceived evenly
        return set->WriteChannels(p);
    };
    // The textures in the set: LibNTC learns only the channels they cover.
    ntc::ITextureMetadata* color = set->AddTexture();
    color->SetName("Color");
    color->SetRgbColorSpace(ntc::ColorSpace::sRGB);
    ntc::ITextureMetadata* surface = set->AddTexture();
    surface->SetName("Surface");
    if (Fail(l, color->SetChannels(0, kColorChannels), "SetChannels", error) ||
        Fail(l, surface->SetChannels(kColorChannels, kSurfaceChannels), "SetChannels", error) ||
        Fail(l, write(textures.color, 0, kColorChannels, srgb), "WriteChannels", error) ||
        Fail(l, write(textures.surface, kColorChannels, kSurfaceChannels, nullptr), "WriteChannels", error))
    {
        return std::nullopt;
    }
    float chosenBpp = 0.0f;
    ntc::LatentShape shape;
    if (Fail(l, l.pickLatentShape(bitsPerPixel, chosenBpp, shape), "PickLatentShape", error) ||
        Fail(l, set->SetLatentShape(shape), "SetLatentShape", error))
    {
        return std::nullopt;
    }
    ntc::CompressionSettings settings;
    settings.trainingSteps = int(steps);
    settings.stepsPerIteration = int(std::min<u32>(steps, 1000));
    settings.randomSeed = 1;
    for (int channel = kChannels; channel < NTC_MAX_CHANNELS; ++channel)
    {
        settings.lossFunctionScales[channel] = 0.0f; // the channels LibNTC pads the set with
    }
    if (Fail(l, set->BeginCompression(settings), "BeginCompression", error))
    {
        return std::nullopt;
    }
    ntc::Status status = ntc::Status::Incomplete;
    ntc::CompressionStats progress;
    while (status == ntc::Status::Incomplete)
    {
        status = set->RunCompressionSteps(&progress);
    }
    AIR_LOG(Info, "NeuralTextures", "'{}': {} steps, loss {}", textures.name, progress.currentStep, progress.loss);
    if (Fail(l, status, "RunCompressionSteps", error) ||
        Fail(l, set->FinalizeCompression(), "FinalizeCompression", error))
    {
        set->AbortCompression();
        return std::nullopt;
    }
    ntc::DecompressionStats stats;
    if (Fail(l, set->Decompress(&stats, true), "Decompress", error))
    {
        return std::nullopt;
    }
    CompressedTextures out;
    out.bitsPerPixel = chosenBpp;
    out.psnr = l.lossToPsnr(stats.overallLoss);
    size_t size = size_t(set->GetOutputStreamSize());
    out.file.resize(size);
    if (Fail(l, set->SaveToMemory(out.file.data(), &size), "SaveToMemory", error))
    {
        return std::nullopt;
    }
    out.file.resize(size);
    if (!entry.empty())
    {
        std::error_code ignored;
        std::filesystem::create_directories(cache, ignored);
        const std::filesystem::path partial = entry.string() + ".partial";
        {
            std::ofstream file(partial, std::ios::binary);
            file.write(kCacheMagic, sizeof(kCacheMagic));
            file.write(reinterpret_cast<const char*>(&out.bitsPerPixel), sizeof(float));
            file.write(reinterpret_cast<const char*>(&out.psnr), sizeof(float));
            file.write(reinterpret_cast<const char*>(out.file.data()), std::streamsize(out.file.size()));
        }
        std::filesystem::rename(partial, entry, ignored); // a cache that cannot be written only costs time
    }
    return out;
#else
    (void)textures, (void)bitsPerPixel, (void)steps, (void)cache;
    if (error)
    {
        *error = "built without LibNTC";
    }
    return std::nullopt;
#endif
}

bool NeuralTextureCodec::Decompress(const CompressedTextures& compressed, MaterialTextures& textures,
                                    std::string* error)
{
#if AIR_WITH_NTC
    std::lock_guard lock(gLock);
    Library& l = Load();
    ntc::TextureSetFeatures features;
    features.stagingBytesPerPixel = 16;
    features.enableCompression = false;
    OwnedSet set{l.context};
    if (Fail(l,
             l.context->CreateCompressedTextureSetFromMemory(compressed.file.data(), compressed.file.size(), features,
                                                             &set.object),
             "CreateCompressedTextureSetFromMemory", error) ||
        Fail(l, set->Decompress(nullptr, true), "Decompress", error))
    {
        return false;
    }
    const ntc::TextureSetDesc& desc = set->GetDesc();
    // (LibNTC rounds the channels up to a multiple of 4.)
    if (u32(desc.width) != textures.width || u32(desc.height) != textures.height || desc.channels < kChannels)
    {
        if (error)
        {
            *error = std::format("the compressed set is {} x {} x {}, the textures {} x {}", desc.width, desc.height,
                                 desc.channels, textures.width, textures.height);
        }
        return false;
    }
    const ntc::ColorSpace srgb[kColorChannels] = {ntc::ColorSpace::sRGB, ntc::ColorSpace::sRGB, ntc::ColorSpace::sRGB};
    auto read = [&](std::vector<u32>& texels, int first, int count, const ntc::ColorSpace* spaces)
    {
        ntc::ReadChannelsParameters p;
        p.page = ntc::TextureDataPage::Output;
        p.mipLevel = 0;
        p.firstChannel = first;
        p.numChannels = count;
        p.pOutData = reinterpret_cast<unsigned char*>(texels.data());
        p.addressSpace = ntc::AddressSpace::Host;
        p.width = desc.width;
        p.height = desc.height;
        p.pixelStride = 4;
        p.rowPitch = size_t(desc.width) * 4;
        p.channelFormat = ntc::ChannelFormat::UNORM8;
        p.dstColorSpaces = spaces;
        return set->ReadChannels(p);
    };
    std::vector<u32> color(size_t(desc.width) * desc.height, 0xFF000000u); // opaque: the colour has no alpha
    std::vector<u32> surface(color.size(), 0u);
    if (Fail(l, read(color, 0, kColorChannels, srgb), "ReadChannels", error) ||
        Fail(l, read(surface, kColorChannels, kSurfaceChannels, nullptr), "ReadChannels", error))
    {
        return false;
    }
    textures.color = std::move(color);
    textures.surface = std::move(surface);
    return true;
#else
    (void)compressed, (void)textures;
    if (error)
    {
        *error = "built without LibNTC";
    }
    return false;
#endif
}

std::optional<std::vector<float>> NeuralTextureCodec::DecodeChannels(const CompressedTextures& compressed,
                                                                     std::string* error)
{
#if AIR_WITH_NTC
    std::lock_guard lock(gLock);
    Library& l = Load();
    ntc::TextureSetFeatures features;
    features.stagingBytesPerPixel = int(kChannels * sizeof(float));
    features.enableCompression = false;
    OwnedSet set{l.context};
    if (Fail(l,
             l.context->CreateCompressedTextureSetFromMemory(compressed.file.data(), compressed.file.size(), features,
                                                             &set.object),
             "CreateCompressedTextureSetFromMemory", error) ||
        Fail(l, set->Decompress(nullptr, true), "Decompress", error))
    {
        return std::nullopt;
    }
    const ntc::TextureSetDesc& desc = set->GetDesc();
    // In the spaces the channels are stored in: colour in sRGB, the rest linear.
    const ntc::ColorSpace spaces[kChannels] = {
        ntc::ColorSpace::sRGB,   ntc::ColorSpace::sRGB,   ntc::ColorSpace::sRGB,  ntc::ColorSpace::Linear,
        ntc::ColorSpace::Linear, ntc::ColorSpace::Linear, ntc::ColorSpace::Linear};
    std::vector<float> values(size_t(desc.width) * desc.height * kChannels);
    ntc::ReadChannelsParameters p;
    p.page = ntc::TextureDataPage::Output;
    p.firstChannel = 0;
    p.numChannels = kChannels;
    p.pOutData = reinterpret_cast<unsigned char*>(values.data());
    p.width = desc.width;
    p.height = desc.height;
    p.pixelStride = kChannels * sizeof(float);
    p.rowPitch = size_t(desc.width) * p.pixelStride;
    p.channelFormat = ntc::ChannelFormat::FLOAT32;
    p.dstColorSpaces = spaces;
    if (Fail(l, set->ReadChannels(p), "ReadChannels", error))
    {
        return std::nullopt;
    }
    return values;
#else
    (void)compressed;
    if (error)
    {
        *error = "built without LibNTC";
    }
    return std::nullopt;
#endif
}

std::optional<NeuralTextureGpuData> NeuralTextureCodec::PrepareInference(const CompressedTextures& compressed,
                                                                         std::string* error)
{
#if AIR_WITH_NTC
    static_assert(sizeof(NtcTextureSetConstants) == kNeuralTextureConstantBytes);
    std::lock_guard lock(gLock);
    Library& l = Load();
    ntc::IStream* stream = nullptr;
    if (Fail(l, l.context->OpenReadOnlyMemory(compressed.file.data(), compressed.file.size(), &stream),
             "OpenReadOnlyMemory", error))
    {
        return std::nullopt;
    }
    OwnedMetadata metadata{l.context};
    const ntc::Status read = l.context->CreateTextureSetMetadataFromStream(stream, &metadata.object);
    l.context->CloseMemory(stream);
    if (Fail(l, read, "CreateTextureSetMetadataFromStream", error))
    {
        return std::nullopt;
    }
    const ntc::LatentTextureDesc latent = metadata->GetLatentTextureDesc();
    NeuralTextureGpuData out;
    out.latentWidth = u32(latent.width);
    out.latentHeight = u32(latent.height);
    out.latentLayers = u32(latent.arraySize);
    out.latentMips = u32(latent.mipLevels);
    for (int layer = 0; layer < latent.arraySize; ++layer)
    {
        for (int mip = 0; mip < latent.mipLevels; ++mip)
        {
            ntc::LatentTextureFootprint footprint;
            if (Fail(l, metadata->GetLatentTextureFootprint(mip, layer, footprint), "GetLatentTextureFootprint", error))
            {
                return std::nullopt;
            }
            const u32 width = std::max(out.latentWidth >> mip, 1u);
            const u32 height = std::max(out.latentHeight >> mip, 1u);
            const ntc::StreamRange range = footprint.buffer.rangeInStream;
            if (u32(footprint.width) != width || u32(footprint.height) != height ||
                range.offset + range.size > compressed.file.size() ||
                footprint.rowPitch * size_t(height) > footprint.buffer.uncompressedSize)
            {
                if (error)
                {
                    *error = std::format("latent mip {} layer {}: an unexpected footprint", mip, layer);
                }
                return std::nullopt;
            }
            std::vector<u8> bgra4(size_t(footprint.buffer.uncompressedSize));
            const u8* source = compressed.file.data() + range.offset;
            if (footprint.buffer.compressionType == ntc::CompressionType::None)
            {
                std::memcpy(bgra4.data(), source, std::min<size_t>(bgra4.size(), size_t(range.size)));
            }
            else if (Fail(l,
                          l.context->DecompressBuffer(footprint.buffer.compressionType, source, size_t(range.size),
                                                      bgra4.data(), bgra4.size(), footprint.buffer.uncompressedCrc32),
                          "DecompressBuffer", error))
            {
                return std::nullopt;
            }
            // B4G4R4A4: blue in bits 0-3, green 4-7, red 8-11, alpha 12-15; x / 15 = 17 x / 255.
            std::vector<u8> rgba8(size_t(width) * height * 4);
            for (u32 y = 0; y < height; ++y)
            {
                for (u32 x = 0; x < width; ++x)
                {
                    u16 t = 0;
                    std::memcpy(&t, bgra4.data() + y * footprint.rowPitch + x * 2, 2);
                    u8* p = rgba8.data() + (size_t(y) * width + x) * 4;
                    p[0] = u8(((t >> 8) & 0xF) * 17);
                    p[1] = u8(((t >> 4) & 0xF) * 17);
                    p[2] = u8((t & 0xF) * 17);
                    p[3] = u8(((t >> 12) & 0xF) * 17);
                }
            }
            out.latents.push_back(std::move(rgba8));
        }
    }
    const void* weights = nullptr;
    size_t weightBytes = 0;
    size_t convertedBytes = 0;
    ntc::InferenceData inference;
    if (Fail(l,
             metadata->GetInferenceWeights(ntc::InferenceWeightType::GenericInt8, &weights, &weightBytes,
                                           &convertedBytes),
             "GetInferenceWeights", error) ||
        Fail(l, l.context->MakeInferenceData(metadata.object, ntc::InferenceWeightType::GenericInt8, 0, &inference),
             "MakeInferenceData", error))
    {
        return std::nullopt;
    }
    out.data.resize(kNeuralTextureConstantBytes + (weightBytes + 3) / 4 * 4);
    std::memcpy(out.data.data(), &inference.constants, kNeuralTextureConstantBytes);
    std::memcpy(out.data.data() + kNeuralTextureConstantBytes, weights, weightBytes);
    return out;
#else
    (void)compressed;
    if (error)
    {
        *error = "built without LibNTC";
    }
    return std::nullopt;
#endif
}

bool ApplyNeuralTextures(SceneDesc& scene, const NeuralTextureOptions& options,
                         std::vector<NeuralTextureReport>* report, std::string* error)
{
    if (options.mode == NeuralTextureMode::Off || scene.textureSets.empty())
    {
        return true;
    }
    NeuralTextureCodec* codec = NeuralTextureCodec::Get(error);
    if (!codec)
    {
        return false;
    }
    std::vector<MaterialTextures> sets = scene.textureSets;
    std::vector<bool> cutout(sets.size(), false); // their colour's alpha is needed, which the codec drops
    for (const Material& m : scene.materials)
    {
        if (m.alphaCutoff > 0.0f && m.textures != kNoTextures && m.textures < sets.size())
        {
            cutout[m.textures] = true;
        }
    }
    for (size_t i = 0; i < sets.size(); ++i)
    {
        if (cutout[i])
        {
            continue;
        }
        MaterialTextures& set = sets[i];
        std::string problem;
        const std::optional<CompressedTextures> compressed =
            codec->Compress(set, options.bitsPerPixel, options.steps, options.cache, &problem);
        if (!compressed || !codec->Decompress(*compressed, set, &problem))
        {
            if (error)
            {
                *error = std::format("'{}': {}", set.name, problem);
            }
            return false;
        }
        if (options.mode == NeuralTextureMode::OnSample)
        {
            std::optional<NeuralTextureGpuData> gpu = codec->PrepareInference(*compressed, &problem);
            if (!gpu)
            {
                if (error)
                {
                    *error = std::format("'{}': {}", set.name, problem);
                }
                return false;
            }
            MakeNeuralTextureCoarseCopy(set, *gpu);
            set.neural = std::make_shared<const NeuralTextureGpuData>(std::move(*gpu));
        }
        if (report)
        {
            report->push_back({.name = set.name,
                               .bitsPerPixel = compressed->bitsPerPixel,
                               .psnr = compressed->psnr,
                               .sourceBytes = (set.color.size() + set.surface.size()) * sizeof(u32),
                               .compressedBytes = compressed->file.size(),
                               .cached = compressed->cached});
        }
    }
    scene.textureSets = std::move(sets);
    return true;
}

} // namespace air::renderer
