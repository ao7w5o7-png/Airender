#include "air/features/pathtracer/NeuralCoat.h"

#include "air/core/Log.h"
#include "air/features/pathtracer/Bsdf.h"
#include "air/renderer/shader/HlslCompat.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <functional>

namespace air::pt
{
namespace coat
{

using namespace air::hlsl;

// The hooks of the shader code compiled here: the BSDF's tables, and the network being checked.
thread_local const float* tLut = nullptr;
thread_local const float* tWeights = nullptr;

float BsdfLutLoad(uint index);
float4 NeuralCoatWeights4(uint index);

#define AIR_BSDF_COATS 1
#include "Bsdf.hlsli"

float BsdfLutLoad(uint index)
{
    return tLut[index];
}

float4 NeuralCoatWeights4(uint index)
{
    const float* p = tWeights;
    return p ? float4(p[index], p[index + 1], p[index + 2], p[index + 3]) : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

} // namespace coat

namespace
{

using coat::float3;
using coat::uint;

constexpr u32 kInputs = coat::kNeuralCoatInputs;
constexpr u32 kHidden = coat::kNeuralCoatHidden;
constexpr u32 kParameters = coat::kNeuralCoatParameters;
constexpr u32 kChunk = 64; // samples a task trains on
constexpr char kCacheMagic[8] = {'A', 'I', 'R', 'C', 'O', 'A', 'T', '1'};

static_assert(kInputs == 40 && kHidden == 16, "the shader code reads the layers as 10 and 4 float4s");

coat::BsdfMaterial ToShader(const CoatMaterialSample& s, uint model, uint seed)
{
    coat::BsdfMaterial m;
    m.baseColor = float3(s.baseColor[0], s.baseColor[1], s.baseColor[2]);
    m.roughness = s.roughness;
    m.metallic = s.metallic;
    m.coat = model;
    m.coatRoughness = s.coatRoughness;
    m.coatColor = float3(s.coatColor[0], s.coatColor[1], s.coatColor[2]);
    m.coatSeed = seed;
    m.flakes = s.flakes;
    m.flakeRoughness = s.flakeRoughness;
    m.flakeColor = float3(s.flakeColor[0], s.flakeColor[1], s.flakeColor[2]);
    m.subsurfaceExit = 0u;
    return m;
}

// --- The network on the CPU (the layout of NeuralCoat.hlsli) ---------------------------------------

struct Forward
{
    float x[kInputs];
    float a0[kHidden];
    float h0[kHidden];
    float a1[kHidden];
    float h1[kHidden];
    float z[3];
};

void Evaluate(const float* p, Forward& f)
{
    for (u32 r = 0; r < kHidden; ++r)
    {
        float s = 0.0f;
        for (u32 i = 0; i < kInputs; ++i)
        {
            s += p[coat::kNeuralCoatW0 + r * kInputs + i] * f.x[i];
        }
        f.a0[r] = s;
        f.h0[r] = std::max(s, 0.0f);
    }
    for (u32 r = 0; r < kHidden; ++r)
    {
        float s = p[coat::kNeuralCoatB1 + r];
        for (u32 k = 0; k < kHidden; ++k)
        {
            s += p[coat::kNeuralCoatW1 + r * kHidden + k] * f.h0[k];
        }
        f.a1[r] = s;
        f.h1[r] = std::max(s, 0.0f);
    }
    for (u32 j = 0; j < 3; ++j)
    {
        float s = p[coat::kNeuralCoatB2 + j];
        for (u32 k = 0; k < kHidden; ++k)
        {
            s += p[coat::kNeuralCoatW2 + j * kHidden + k] * f.h1[k];
        }
        f.z[j] = s;
    }
}

// Adds the gradient of a loss whose derivative by the outputs is dz to g.
void Backward(const float* p, const Forward& f, const float dz[3], float* g)
{
    float dh1[kHidden] = {};
    for (u32 j = 0; j < 3; ++j)
    {
        g[coat::kNeuralCoatB2 + j] += dz[j];
        for (u32 k = 0; k < kHidden; ++k)
        {
            g[coat::kNeuralCoatW2 + j * kHidden + k] += dz[j] * f.h1[k];
            dh1[k] += p[coat::kNeuralCoatW2 + j * kHidden + k] * dz[j];
        }
    }
    float dh0[kHidden] = {};
    for (u32 r = 0; r < kHidden; ++r)
    {
        const float da = f.a1[r] > 0.0f ? dh1[r] : 0.0f;
        if (da == 0.0f)
        {
            continue;
        }
        g[coat::kNeuralCoatB1 + r] += da;
        for (u32 k = 0; k < kHidden; ++k)
        {
            g[coat::kNeuralCoatW1 + r * kHidden + k] += da * f.h0[k];
            dh0[k] += p[coat::kNeuralCoatW1 + r * kHidden + k] * da;
        }
    }
    for (u32 r = 0; r < kHidden; ++r)
    {
        const float da = f.a0[r] > 0.0f ? dh0[r] : 0.0f;
        if (da == 0.0f)
        {
            continue;
        }
        for (u32 i = 0; i < kInputs; ++i)
        {
            g[coat::kNeuralCoatW0 + r * kInputs + i] += da * f.x[i];
        }
    }
}

// --- Samples -------------------------------------------------------------------------------------

struct Situation
{
    coat::BsdfMaterial material;
    float3 wo;
    float3 wi;
    bool valid = false;
};

// A material sample with wo from grazing to normal incidence and wi half from the coat's sampling
// (where rendering looks) and half uniform over the hemisphere.
Situation MakeSituation(std::span<const CoatMaterialSample> materials, uint seed)
{
    Situation s;
    uint rng = coat::Pcg(seed);
    const CoatMaterialSample& m =
        materials[std::min(u32(coat::Rand(rng) * float(materials.size())), u32(materials.size() - 1))];
    s.material = ToShader(m, coat::kCoatRandomWalk, coat::Pcg(seed ^ 0x68bc21ebu));
    const float muO = 0.02f + 0.98f * coat::Rand(rng);
    s.wo = float3(std::sqrt(1.0f - muO * muO), 0.0f, muO);
    const coat::BsdfContext c = coat::PrepareBsdf(s.material, s.wo);
    const float choice = coat::Rand(rng);
    const float uLobe = coat::Rand(rng);
    const coat::float2 u = coat::Rand2(rng);
    if (choice < 0.5f)
    {
        s.wi = coat::SampleBsdfDirection(c, uLobe, u); // the coat, the flakes, the base
    }
    else
    {
        const float r = std::sqrt(std::max(0.0f, 1.0f - u.x * u.x));
        s.wi = float3(r * std::cos(coat::kTwoPi * u.y), r * std::sin(coat::kTwoPi * u.y), u.x);
    }
    s.valid = s.wi.z > 1e-3f;
    return s;
}

// The mean of `walks` random walks' inner part.
float3 WalkMean(const Situation& s, u32 walks, uint seed)
{
    const coat::BsdfContext c = coat::PrepareBsdf(s.material, s.wo);
    float3 sum(0.0f);
    for (u32 i = 0; i < walks; ++i)
    {
        uint walk = coat::Pcg(seed + i * 0x9e3779b9u);
        sum = sum + coat::CoatInnerWalk(c, s.wi, walk);
    }
    return sum / float(walks);
}

void Encode(const Situation& s, Forward& f)
{
    const coat::NeuralCoatInput x = coat::NeuralCoatEncode(s.material, s.wo, s.wi);
    for (u32 i = 0; i < kInputs / 4; ++i)
    {
        f.x[i * 4] = x.v[i].x;
        f.x[i * 4 + 1] = x.v[i].y;
        f.x[i * 4 + 2] = x.v[i].z;
        f.x[i * 4 + 3] = x.v[i].w;
    }
}

float Channel(float3 v, u32 c)
{
    return c == 0 ? v.x : c == 1 ? v.y : v.z;
}

void ParallelFor(TaskSystem* tasks, u32 count, const std::function<void(u32)>& fn)
{
    if (tasks)
    {
        tasks->ParallelFor(count, fn);
        return;
    }
    for (u32 i = 0; i < count; ++i)
    {
        fn(i);
    }
}

u64 Fnv1a(u64 hash, const void* data, size_t size)
{
    const auto* bytes = static_cast<const u8*>(data);
    for (size_t i = 0; i < size; ++i)
    {
        hash = (hash ^ bytes[i]) * 0x100000001b3ull;
    }
    return hash;
}

} // namespace

std::vector<CoatMaterialSample> CollectCoatMaterials(const renderer::SceneDesc& scene, u32 count)
{
    std::vector<const renderer::Material*> coated;
    for (const renderer::Material& m : scene.materials)
    {
        if (m.coat)
        {
            coated.push_back(&m);
        }
    }
    std::vector<CoatMaterialSample> samples;
    if (coated.empty())
    {
        return samples;
    }
    const u32 per = std::max(count / u32(coated.size()), 1u);
    auto srgb = [](u32 byte)
    {
        const float s = float(byte) / 255.0f;
        return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    };
    for (const renderer::Material* m : coated)
    {
        const renderer::MaterialTextures* t =
            m->textures != renderer::kNoTextures ? &scene.textureSets[m->textures] : nullptr;
        uint rng = coat::Pcg(u32(samples.size()) + 0x2545f491u);
        for (u32 i = 0; i < per; ++i)
        {
            CoatMaterialSample s;
            s.baseColor[0] = m->baseColor.x;
            s.baseColor[1] = m->baseColor.y;
            s.baseColor[2] = m->baseColor.z;
            s.roughness = m->roughness;
            s.metallic = m->metallic;
            s.coatRoughness = m->coatRoughness;
            s.coatColor[0] = m->coatColor.x;
            s.coatColor[1] = m->coatColor.y;
            s.coatColor[2] = m->coatColor.z;
            s.flakes = m->flakes;
            s.flakeRoughness = m->flakeRoughness;
            s.flakeColor[0] = m->flakeColor.x;
            s.flakeColor[1] = m->flakeColor.y;
            s.flakeColor[2] = m->flakeColor.z;
            if (t)
            {
                const size_t texel = std::min(size_t(coat::Rand(rng) * float(t->color.size())), t->color.size() - 1);
                const u32 c = t->color[texel];
                const u32 surface = t->surface[texel];
                for (u32 k = 0; k < 3; ++k)
                {
                    s.baseColor[k] *= srgb((c >> (8 * k)) & 0xFFu);
                }
                s.roughness = std::clamp(s.roughness * float((surface >> 16) & 0xFFu) / 255.0f, 0.0f, 1.0f);
                s.metallic = std::clamp(s.metallic * float(surface >> 24) / 255.0f, 0.0f, 1.0f);
            }
            samples.push_back(s);
        }
    }
    return samples;
}

NeuralCoatResult TrainNeuralCoat(std::span<const CoatMaterialSample> materials, const NeuralCoatSettings& settings,
                                 TaskSystem* tasks)
{
    NeuralCoatResult result;
    if (materials.empty())
    {
        return result;
    }
    u64 key = 0xcbf29ce484222325ull;
    key = Fnv1a(key, materials.data(), materials.size_bytes());
    // The last value is the coat's version: raise it whenever Bsdf.hlsli's coat or NeuralCoat.hlsli changes.
    const u32 header[] = {settings.steps, settings.batch, settings.walks, settings.seed, kParameters, 5u};
    key = Fnv1a(key, header, sizeof(header));
    key = Fnv1a(key, &settings.learningRate, sizeof(float));
    const std::filesystem::path entry =
        settings.cache.empty() ? std::filesystem::path() : settings.cache / std::format("coat-{:016x}.bin", key);
    if (!entry.empty())
    {
        std::ifstream in(entry, std::ios::binary);
        char magic[8] = {};
        result.parameters.resize(kParameters);
        if (in.read(magic, 8) && std::memcmp(magic, kCacheMagic, 8) == 0 &&
            in.read(reinterpret_cast<char*>(result.parameters.data()), kParameters * sizeof(float)) &&
            in.read(reinterpret_cast<char*>(&result.loss), sizeof(float)))
        {
            result.cached = true;
            return result;
        }
    }

    // He-initialized hidden layers; the output layer starts at zero, the approximation unchanged.
    std::vector<float>& p = result.parameters;
    p.assign(kParameters, 0.0f);
    // The first layer's weights of the coat's own 32 inputs are drawn as before the flakes' 8 were added
    // (in that order, at that scale), theirs after: without flakes their inputs are 0, so a coat over a
    // plain base trains exactly the network it did.
    uint init = coat::Pcg(settings.seed * 0x85ebca6bu + 1u);
    constexpr u32 kCoatInputs = 32;
    const float scale = std::sqrt(6.0f / float(kCoatInputs));
    for (u32 i = 0; i < kHidden * kCoatInputs; ++i)
    {
        p[coat::kNeuralCoatW0 + (i / kCoatInputs) * kInputs + i % kCoatInputs] =
            (coat::Rand(init) * 2.0f - 1.0f) * scale;
    }
    uint flakeInit = coat::Pcg(settings.seed * 0x27d4eb2fu + 7u);
    for (u32 r = 0; r < kHidden; ++r)
    {
        for (u32 k = kCoatInputs; k < kInputs; ++k)
        {
            p[coat::kNeuralCoatW0 + r * kInputs + k] = (coat::Rand(flakeInit) * 2.0f - 1.0f) * scale;
        }
    }
    for (u32 i = 0; i < kHidden * kHidden; ++i)
    {
        p[coat::kNeuralCoatW1 + i] = (coat::Rand(init) * 2.0f - 1.0f) * std::sqrt(6.0f / float(kHidden));
    }
    std::vector<float> moment1(kParameters, 0.0f);
    std::vector<float> moment2(kParameters, 0.0f);
    const u32 chunks = std::max(settings.batch / kChunk, 1u);
    const u32 batch = chunks * kChunk;
    std::vector<float> gradients(size_t(chunks) * kParameters);
    std::vector<double> losses(chunks);
    std::vector<u32> counts(chunks);
    double recentLoss = 0.0;
    u32 recentSteps = 0;
    for (u32 step = 0; step < settings.steps; ++step)
    {
        std::fill(gradients.begin(), gradients.end(), 0.0f);
        ParallelFor(
            tasks, chunks,
            [&](u32 chunk)
            {
                coat::tLut = GetBsdfLut().data();
                float* g = &gradients[size_t(chunk) * kParameters];
                double loss = 0.0;
                u32 used = 0;
                for (u32 i = 0; i < kChunk; ++i)
                {
                    const uint seed = coat::Pcg(settings.seed + (step * batch + chunk * kChunk + i) * 2654435761u);
                    const Situation s = MakeSituation(materials, seed);
                    if (!s.valid)
                    {
                        continue;
                    }
                    const float3 target = WalkMean(s, settings.walks, coat::Pcg(seed ^ 0x27d4eb2fu));
                    const coat::BsdfContext c = coat::PrepareBsdf(s.material, s.wo);
                    const float3 approx = coat::CoatInnerApproximate(c, s.wi);
                    Forward f;
                    Encode(s, f);
                    Evaluate(p.data(), f);
                    // Relative L2 of the inner part, approx * exp(z), against the walks' mean
                    // (unbiased in their noise: the gradient is linear in the target).
                    float dz[3] = {};
                    for (u32 ch = 0; ch < 3; ++ch)
                    {
                        const float a = Channel(approx, ch);
                        const float z =
                            std::clamp(f.z[ch], -coat::kNeuralCoatMaxCorrection, coat::kNeuralCoatMaxCorrection);
                        const float y = a * std::exp(z);
                        const float scale = y + 0.05f * a + 1e-4f;
                        const float r = (y - Channel(target, ch)) / scale;
                        loss += double(r * r);
                        dz[ch] = std::abs(f.z[ch]) < coat::kNeuralCoatMaxCorrection ? 2.0f * r * y / scale : 0.0f;
                    }
                    Backward(p.data(), f, dz, g);
                    ++used;
                }
                losses[chunk] = loss;
                counts[chunk] = used;
            });
        // Adam over the batch's mean gradient, summed in chunk order (deterministic).
        u32 used = 0;
        double loss = 0.0;
        for (u32 chunk = 0; chunk < chunks; ++chunk)
        {
            used += counts[chunk];
            loss += losses[chunk];
        }
        const float invUsed = 1.0f / float(std::max(used, 1u));
        const float lr = settings.learningRate * std::pow(0.1f, float(step) / float(std::max(settings.steps, 1u)));
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float c1 = 1.0f / (1.0f - std::pow(beta1, float(step + 1)));
        const float c2 = 1.0f / (1.0f - std::pow(beta2, float(step + 1)));
        for (u32 i = 0; i < kParameters; ++i)
        {
            float g = 0.0f;
            for (u32 chunk = 0; chunk < chunks; ++chunk)
            {
                g += gradients[size_t(chunk) * kParameters + i];
            }
            g *= invUsed;
            moment1[i] = beta1 * moment1[i] + (1.0f - beta1) * g;
            moment2[i] = beta2 * moment2[i] + (1.0f - beta2) * g * g;
            p[i] -= lr * (moment1[i] * c1) / (std::sqrt(moment2[i] * c2) + 1e-8f);
        }
        if (step + 100 >= settings.steps)
        {
            recentLoss += loss * invUsed / 3.0;
            ++recentSteps;
        }
    }
    result.loss = float(recentLoss / std::max(recentSteps, 1u));
    if (!entry.empty())
    {
        std::error_code ignored;
        std::filesystem::create_directories(settings.cache, ignored);
        const std::filesystem::path partial = entry.string() + ".partial";
        {
            std::ofstream out(partial, std::ios::binary);
            out.write(kCacheMagic, 8);
            out.write(reinterpret_cast<const char*>(p.data()), std::streamsize(kParameters * sizeof(float)));
            out.write(reinterpret_cast<const char*>(&result.loss), sizeof(float));
        }
        std::filesystem::rename(partial, entry, ignored);
    }
    return result;
}

json::Value CheckNeuralCoat(std::span<const CoatMaterialSample> materials, std::span<const float> parameters,
                            u32 directions, TaskSystem* tasks)
{
    json::Value report = json::Value::MakeObject();
    json::Value list = json::Value::MakeArray();
    // Up to 12 of the samples, spread over the list (the materials come one after another).
    const u32 shown = std::min<u32>(u32(materials.size()), 12u);
    struct Errors
    {
        double neural = 0.0;
        double approximate = 0.0;
        double reference = 0.0;
        double albedo[3][3] = {}; // [angle][walk, neural, approximate]
    };
    std::vector<Errors> errors(shown);
    constexpr float kAngles[3] = {0.25f, 0.6f, 0.95f};
    ParallelFor(tasks, shown,
                [&](u32 index)
                {
                    coat::tLut = GetBsdfLut().data();
                    coat::tWeights = parameters.size() == kParameters ? parameters.data() : nullptr;
                    const CoatMaterialSample& m = materials[size_t(index) * materials.size() / shown];
                    const std::span<const CoatMaterialSample> one(&m, 1);
                    Errors& e = errors[index];
                    for (u32 d = 0; d < directions; ++d)
                    {
                        const Situation s = MakeSituation(one, coat::Pcg(0x1b873593u + d));
                        if (!s.valid)
                        {
                            continue;
                        }
                        const float3 reference = WalkMean(s, 256, coat::Pcg(d * 7919u + 17u));
                        coat::BsdfMaterial neural = s.material;
                        neural.coat = coat::kCoatNeural;
                        const coat::BsdfContext c = coat::PrepareBsdf(neural, s.wo);
                        const float y = coat::Luminance(reference);
                        e.neural += std::pow(coat::Luminance(coat::NeuralCoatInner(c, s.wi)) - y, 2.0f);
                        e.approximate += std::pow(coat::Luminance(coat::CoatInnerApproximate(c, s.wi)) - y, 2.0f);
                        e.reference += double(y) * y;
                    }
                    // Directional albedo with each model's own evaluations (the walk's: one per sample).
                    for (u32 a = 0; a < 3; ++a)
                    {
                        const float mu = kAngles[a];
                        const float3 wo(std::sqrt(1.0f - mu * mu), 0.0f, mu);
                        const uint models[3] = {coat::kCoatRandomWalk, coat::kCoatNeural, coat::kCoatApproximate};
                        for (u32 k = 0; k < 3; ++k)
                        {
                            uint rng = 0x9e3779b9u + index;
                            double sum = 0.0;
                            const u32 n = 8192;
                            for (u32 i = 0; i < n; ++i)
                            {
                                coat::BsdfMaterial mm = ToShader(m, models[k], coat::Pcg(i * 31u + a));
                                const float uLobe = coat::Rand(rng);
                                const coat::BsdfSample bs =
                                    coat::SampleBsdf(coat::PrepareBsdf(mm, wo), uLobe, coat::Rand2(rng));
                                if (bs.valid)
                                {
                                    sum += coat::Luminance(bs.weight);
                                }
                            }
                            e.albedo[a][k] = sum / n;
                        }
                    }
                });
    double neural = 0.0;
    double approximate = 0.0;
    double reference = 0.0;
    for (u32 i = 0; i < shown; ++i)
    {
        const Errors& e = errors[i];
        const CoatMaterialSample& m = materials[size_t(i) * materials.size() / shown];
        json::Value item = json::Value::MakeObject();
        json::Value base = json::Value::MakeArray();
        for (float v : m.baseColor)
        {
            base.Push(v);
        }
        item["baseColor"] = std::move(base);
        item["roughness"] = m.roughness;
        item["metallic"] = m.metallic;
        item["coatRoughness"] = m.coatRoughness;
        item["flakes"] = m.flakes;
        item["neuralError"] = std::sqrt(e.neural / std::max(e.reference, 1e-12));
        item["approximateError"] = std::sqrt(e.approximate / std::max(e.reference, 1e-12));
        json::Value albedo = json::Value::MakeArray();
        for (u32 a = 0; a < 3; ++a)
        {
            json::Value row = json::Value::MakeObject();
            row["cosTheta"] = kAngles[a];
            row["walk"] = e.albedo[a][0];
            row["neural"] = e.albedo[a][1];
            row["approximate"] = e.albedo[a][2];
            albedo.Push(std::move(row));
        }
        item["albedo"] = std::move(albedo);
        list.Push(std::move(item));
        neural += e.neural;
        approximate += e.approximate;
        reference += e.reference;
    }
    report["materials"] = std::move(list);
    report["neuralError"] = std::sqrt(neural / std::max(reference, 1e-12));
    report["approximateError"] = std::sqrt(approximate / std::max(reference, 1e-12));
    return report;
}

} // namespace air::pt
