#include "air/features/pathtracer/SubsurfaceFit.h"

#include "air/features/pathtracer/Bsdf.h"
#include "air/renderer/shader/HlslCompat.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>

namespace air::pt
{
namespace sss
{

using namespace air::hlsl;

// The hook of the shader code compiled here: the BSDF's tables.
thread_local const float* tLut = nullptr;

float BsdfLutLoad(uint index);

#include "Bsdf.hlsli"
#include "SubsurfaceProfile.hlsli"

float BsdfLutLoad(uint index)
{
    return tLut[index];
}

} // namespace sss

namespace
{

constexpr u32 kInputs = sss::kSubsurfaceProfileInputs;
constexpr u32 kHidden = sss::kSubsurfaceProfileHidden;
constexpr u32 kParameters = sss::kSubsurfaceProfileParameters;
static_assert(kParameters == kHidden * (kInputs + 1) + kHidden * (kHidden + 1) + kHidden + 1);
constexpr u32 kChunk = 256;   // samples a task fits on
constexpr u32 kMinWalks = 20; // walks a bin needs to be fit on
constexpr u32 kMaxEvents = 200000;

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

void BindLut()
{
    sss::tLut = GetBsdfLut().data();
}

// PCG, as the shaders' Rand.
struct Random
{
    u32 state;
    float Next()
    {
        state = state * 747796405u + 2891336453u;
        const u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return float((word >> 22u) ^ word) * (1.0f / 4294967296.0f);
    }
};

// The network's inputs (SubsurfaceProfileCorrection's).
void Encode(float logRadius, float color, float anisotropy, float x[kInputs])
{
    x[0] = std::max(logRadius, std::log(1e-4f)) * 0.25f;
    x[1] = color * 2.0f - 1.0f;
    x[2] = anisotropy * (2.0f / 0.9f) - 1.0f;
}

struct Forward
{
    float x[kInputs];
    float a1[kHidden];
    float h1[kHidden];
    float a2[kHidden];
    float h2[kHidden];
    float z;
};

// The layout of SubsurfaceProfileCorrection: per neuron its input weights, then its bias.
void Evaluate(const float* p, Forward& f)
{
    u32 w = 0;
    for (u32 i = 0; i < kHidden; ++i)
    {
        float a = p[w + kInputs];
        for (u32 k = 0; k < kInputs; ++k)
        {
            a += p[w + k] * f.x[k];
        }
        f.a1[i] = a;
        f.h1[i] = std::max(a, 0.0f);
        w += kInputs + 1;
    }
    for (u32 i = 0; i < kHidden; ++i)
    {
        float a = p[w + kHidden];
        for (u32 k = 0; k < kHidden; ++k)
        {
            a += p[w + k] * f.h1[k];
        }
        f.a2[i] = a;
        f.h2[i] = std::max(a, 0.0f);
        w += kHidden + 1;
    }
    float z = p[w + kHidden];
    for (u32 k = 0; k < kHidden; ++k)
    {
        z += p[w + k] * f.h2[k];
    }
    f.z = z;
}

// Adds dL/dp for dL/dz = dz to g.
void Backward(const float* p, const Forward& f, float dz, float* g)
{
    const u32 base1 = 0;
    const u32 base2 = kHidden * (kInputs + 1);
    const u32 base3 = base2 + kHidden * (kHidden + 1);
    float d2[kHidden];
    for (u32 k = 0; k < kHidden; ++k)
    {
        g[base3 + k] += dz * f.h2[k];
        d2[k] = f.a2[k] > 0.0f ? dz * p[base3 + k] : 0.0f;
    }
    g[base3 + kHidden] += dz;
    float d1[kHidden] = {};
    for (u32 i = 0; i < kHidden; ++i)
    {
        const u32 w = base2 + i * (kHidden + 1);
        for (u32 k = 0; k < kHidden; ++k)
        {
            g[w + k] += d2[i] * f.h1[k];
            d1[k] += d2[i] * p[w + k];
        }
        g[w + kHidden] += d2[i];
    }
    for (u32 i = 0; i < kHidden; ++i)
    {
        const float d = f.a1[i] > 0.0f ? d1[i] : 0.0f;
        const u32 w = base1 + i * (kInputs + 1);
        for (u32 k = 0; k < kInputs; ++k)
        {
            g[w + k] += d * f.x[k];
        }
        g[w + kInputs] += d;
    }
}

// Henyey-Greenstein's cosine for a uniform xi (CloudLookup.hlsli's SampleHgCosine).
float SampleHgCosine(float g, float xi)
{
    if (std::abs(g) < 1e-3f)
    {
        return 2.0f * xi - 1.0f;
    }
    const float t = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi);
    return std::clamp((1.0f + g * g - t * t) / (2.0f * g), -1.0f, 1.0f);
}

// The bins' edges in r.
float BinEdge(const SubsurfaceProfileTable& t, u32 edge)
{
    return std::exp(t.logMin + (t.logMax - t.logMin) * float(edge) / float(t.bins));
}

float BinArea(const SubsurfaceProfileTable& t, u32 bin)
{
    const float lo = bin == 0 ? 0.0f : BinEdge(t, bin);
    const float hi = BinEdge(t, bin + 1);
    return float(sss::kPi) * (hi * hi - lo * lo);
}

float BinCentre(const SubsurfaceProfileTable& t, u32 bin)
{
    return t.logMin + (t.logMax - t.logMin) * (float(bin) + 0.5f) / float(t.bins);
}

// Burley's profile (radius 1) averaged over the bin: its radial CDF is 1 - (e^-r + 3 e^(-r/3)) / 4.
float BurleyBinAverage(const SubsurfaceProfileTable& t, u32 bin)
{
    const auto cdf = [](float r) { return 1.0f - (std::exp(-r) + 3.0f * std::exp(-r / 3.0f)) * 0.25f; };
    const float lo = bin == 0 ? 0.0f : BinEdge(t, bin);
    return (cdf(BinEdge(t, bin + 1)) - cdf(lo)) / BinArea(t, bin);
}

// The fit's samples: the bins with enough walks, their correction target and weight.
struct Sample
{
    float x[kInputs];
    float target;
    float weight;
};

std::vector<Sample> MakeSamples(const SubsurfaceProfileTable& t)
{
    std::vector<Sample> samples;
    for (size_t c = 0; c < t.configs.size(); ++c)
    {
        const float color = t.configs[c].color;
        for (u32 b = 0; b < t.bins; ++b)
        {
            const size_t i = c * t.bins + b;
            if (t.counts[i] < kMinWalks)
            {
                continue;
            }
            Sample s;
            Encode(BinCentre(t, b), color, t.configs[c].anisotropy, s.x);
            s.target = std::log(t.profile[i] / (color * BurleyBinAverage(t, b)));
            // The bin's share of the light, and a floor so that the faint tail still counts.
            s.weight = t.profile[i] * BinArea(t, b) / std::max(color, 1e-3f) + 0.5f / float(t.bins);
            samples.push_back(s);
        }
    }
    return samples;
}

} // namespace

std::vector<SubsurfaceProfileTable::Config> SubsurfaceFitConfigs()
{
    std::vector<SubsurfaceProfileTable::Config> configs;
    const float colors[] = {0.05f, 0.1f,  0.15f, 0.2f,  0.25f, 0.3f,  0.35f, 0.4f,  0.45f, 0.5f,  0.55f,
                            0.6f,  0.65f, 0.7f,  0.75f, 0.8f,  0.85f, 0.9f,  0.93f, 0.96f, 0.98f, 0.99f};
    for (float g : {0.0f, 0.15f, 0.3f, 0.45f, 0.6f, 0.75f, 0.9f})
    {
        for (float color : colors)
        {
            configs.push_back({color, g});
        }
    }
    return configs;
}

SubsurfaceProfileTable SimulateSubsurfaceProfiles(std::span<const SubsurfaceProfileTable::Config> configs, u32 walks,
                                                  u32 seed, TaskSystem* tasks)
{
    SubsurfaceProfileTable t;
    t.configs.assign(configs.begin(), configs.end());
    t.walks = walks;
    t.profile.assign(t.configs.size() * t.bins, 0.0f);
    t.counts.assign(t.configs.size() * t.bins, 0u);
    t.total.assign(t.configs.size(), 0.0f);
    // Tasks of whole chunks of walks, summed in order: the same result for any task count.
    constexpr u32 kWalksPerTask = 4096;
    const u32 tasksPerConfig = (walks + kWalksPerTask - 1) / kWalksPerTask;
    const u32 taskCount = u32(t.configs.size()) * tasksPerConfig;
    std::vector<double> sums(size_t(taskCount) * (t.bins + 1), 0.0);
    std::vector<u32> hits(size_t(taskCount) * t.bins, 0u);
    ParallelFor(tasks, taskCount,
                [&](u32 task)
                {
                    BindLut();
                    const u32 c = task / tasksPerConfig;
                    const u32 first = (task % tasksPerConfig) * kWalksPerTask;
                    const u32 last = std::min(first + kWalksPerTask, walks);
                    const float color = t.configs[c].color;
                    const float g = t.configs[c].anisotropy;
                    const float sigmaT = sss::SubsurfaceExtinction(color, g);
                    const float albedo = sss::SubsurfaceSingleScattering(color, g);
                    double* sum = &sums[size_t(task) * (t.bins + 1)];
                    u32* hit = &hits[size_t(task) * t.bins];
                    for (u32 w = first; w < last; ++w)
                    {
                        Random random{sss::Pcg(seed ^ sss::Pcg(c * 0x9e3779b9u ^ sss::Pcg(w)))};
                        // Depth below the boundary along z; the walk starts at the origin.
                        float px = 0.0f;
                        float py = 0.0f;
                        float pz = 0.0f;
                        const float u0 = random.Next();
                        const float phi0 = 2.0f * float(sss::kPi) * random.Next();
                        float dz = std::sqrt(1.0f - u0);
                        float dx = std::sqrt(u0) * std::cos(phi0);
                        float dy = std::sqrt(u0) * std::sin(phi0);
                        float weight = 1.0f;
                        for (u32 event = 0; event < kMaxEvents; ++event)
                        {
                            const float s = -std::log(1.0f - random.Next()) / sigmaT;
                            if (pz + s * dz <= 0.0f)
                            {
                                const float reach = dz < 0.0f ? -pz / dz : 0.0f;
                                const float r = std::hypot(px + dx * reach, py + dy * reach);
                                const float lr = std::log(std::max(r, 1e-30f));
                                const int bin =
                                    int(std::floor((lr - t.logMin) / (t.logMax - t.logMin) * float(t.bins)));
                                if (bin < int(t.bins))
                                {
                                    const u32 b = u32(std::max(bin, 0));
                                    sum[b] += weight;
                                    ++hit[b];
                                }
                                sum[t.bins] += weight;
                                break;
                            }
                            px += dx * s;
                            py += dy * s;
                            pz += dz * s;
                            weight *= albedo;
                            // Henyey-Greenstein about the direction.
                            const float cosTheta = SampleHgCosine(g, random.Next());
                            const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
                            const float phi = 2.0f * float(sss::kPi) * random.Next();
                            const sss::Frame f = sss::MakeFrame(sss::float3(dx, dy, dz));
                            const sss::float3 d = sss::ToWorld(
                                f, sss::float3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta));
                            dx = d.x;
                            dy = d.y;
                            dz = d.z;
                            if (weight < 0.05f)
                            {
                                if (random.Next() >= 0.25f)
                                {
                                    break;
                                }
                                weight *= 4.0f;
                            }
                        }
                    }
                });
    for (u32 c = 0; c < u32(t.configs.size()); ++c)
    {
        for (u32 k = 0; k < tasksPerConfig; ++k)
        {
            const u32 task = c * tasksPerConfig + k;
            for (u32 b = 0; b < t.bins; ++b)
            {
                t.profile[size_t(c) * t.bins + b] += float(sums[size_t(task) * (t.bins + 1) + b]);
                t.counts[size_t(c) * t.bins + b] += hits[size_t(task) * t.bins + b];
            }
            t.total[c] += float(sums[size_t(task) * (t.bins + 1) + t.bins]);
        }
        t.total[c] /= float(walks);
        for (u32 b = 0; b < t.bins; ++b)
        {
            t.profile[size_t(c) * t.bins + b] /= float(walks) * BinArea(t, b);
        }
    }
    return t;
}

SubsurfaceFitResult FitSubsurfaceProfile(const SubsurfaceProfileTable& table, const SubsurfaceFitSettings& settings,
                                         TaskSystem* tasks)
{
    const std::vector<Sample> samples = MakeSamples(table);
    SubsurfaceFitResult result;
    std::vector<float>& p = result.parameters;
    p.assign(kParameters, 0.0f);
    // He initialization of the hidden layers, the output at 0: Burley's profile to begin with.
    Random random{sss::Pcg(settings.seed)};
    const auto normal = [&]
    {
        const float u1 = std::max(random.Next(), 1e-7f);
        const float u2 = random.Next();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * float(sss::kPi) * u2);
    };
    u32 w = 0;
    for (u32 i = 0; i < kHidden; ++i, w += kInputs + 1)
    {
        for (u32 k = 0; k < kInputs; ++k)
        {
            p[w + k] = normal() * std::sqrt(2.0f / float(kInputs));
        }
        p[w + kInputs] = 0.1f;
    }
    for (u32 i = 0; i < kHidden; ++i, w += kHidden + 1)
    {
        for (u32 k = 0; k < kHidden; ++k)
        {
            p[w + k] = normal() * std::sqrt(2.0f / float(kHidden));
        }
        p[w + kHidden] = 0.1f;
    }
    double weightSum = 0.0;
    for (const Sample& s : samples)
    {
        weightSum += s.weight;
    }
    const float invWeight = float(1.0 / std::max(weightSum, 1e-30));
    const u32 chunks = (u32(samples.size()) + kChunk - 1) / kChunk;
    std::vector<float> gradients(size_t(chunks) * kParameters);
    std::vector<double> losses(chunks);
    std::vector<float> moment1(kParameters, 0.0f);
    std::vector<float> moment2(kParameters, 0.0f);
    for (u32 step = 0; step < settings.steps; ++step)
    {
        std::fill(gradients.begin(), gradients.end(), 0.0f);
        ParallelFor(tasks, chunks,
                    [&](u32 chunk)
                    {
                        float* g = &gradients[size_t(chunk) * kParameters];
                        double loss = 0.0;
                        const u32 end = std::min<u32>((chunk + 1) * kChunk, u32(samples.size()));
                        for (u32 i = chunk * kChunk; i < end; ++i)
                        {
                            Forward f;
                            std::copy(std::begin(samples[i].x), std::end(samples[i].x), f.x);
                            Evaluate(p.data(), f);
                            const float error = f.z - samples[i].target;
                            loss += samples[i].weight * error * error;
                            Backward(p.data(), f, 2.0f * samples[i].weight * error * invWeight, g);
                        }
                        losses[chunk] = loss;
                    });
        double loss = 0.0;
        for (double l : losses)
        {
            loss += l;
        }
        result.loss = float(loss * invWeight);
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
            moment1[i] = beta1 * moment1[i] + (1.0f - beta1) * g;
            moment2[i] = beta2 * moment2[i] + (1.0f - beta2) * g * g;
            p[i] -= lr * (moment1[i] * c1) / (std::sqrt(moment2[i] * c2) + 1e-8f);
        }
    }
    return result;
}

float EvaluateSubsurfaceProfileCorrection(std::span<const float> parameters, float logRadius, float color,
                                          float anisotropy)
{
    Forward f;
    Encode(logRadius, color, anisotropy, f.x);
    Evaluate(parameters.data(), f);
    return std::clamp(f.z, -4.0f, 4.0f);
}

std::span<const float> CompiledSubsurfaceProfile()
{
    return {sss::kSubsurfaceProfileWeights, kParameters};
}

float ShaderSubsurfaceProfileCorrection(float logRadius, float color, float anisotropy)
{
    return sss::SubsurfaceProfileCorrection(std::exp(logRadius), 1.0f, color, anisotropy);
}

json::Value CheckSubsurfaceProfile(const SubsurfaceProfileTable& table, std::span<const float> parameters)
{
    json::Value report = json::Value::MakeObject();
    json::Value configs = json::Value::MakeArray();
    double burleySum = 0.0;
    double networkSum = 0.0;
    double energyBurley = 0.0;
    double energyNetwork = 0.0;
    for (size_t c = 0; c < table.configs.size(); ++c)
    {
        const float color = table.configs[c].color;
        const float g = table.configs[c].anisotropy;
        double light = 0.0;
        double burleyError = 0.0;
        double networkError = 0.0;
        double networkTotal = 0.0;
        double burleyTotal = 0.0;
        for (u32 b = 0; b < table.bins; ++b)
        {
            const float walk = table.profile[c * table.bins + b];
            const float burley = color * BurleyBinAverage(table, b);
            const float network =
                burley * std::exp(EvaluateSubsurfaceProfileCorrection(parameters, BinCentre(table, b), color, g));
            const float area = BinArea(table, b);
            burleyTotal += burley * area;
            networkTotal += network * area;
            if (table.counts[c * table.bins + b] < kMinWalks)
            {
                continue;
            }
            // Relative errors weighed by the bin's share of the light.
            const double share = walk * area;
            light += share;
            burleyError += share * std::pow((burley - walk) / walk, 2.0);
            networkError += share * std::pow((network - walk) / walk, 2.0);
        }
        const double burleyRms = std::sqrt(burleyError / std::max(light, 1e-30));
        const double networkRms = std::sqrt(networkError / std::max(light, 1e-30));
        burleySum += burleyRms;
        networkSum += networkRms;
        energyBurley += std::abs(burleyTotal / table.total[c] - 1.0);
        energyNetwork += std::abs(networkTotal / table.total[c] - 1.0);
        json::Value entry = json::Value::MakeObject();
        entry["color"] = color;
        entry["anisotropy"] = g;
        entry["walkTotal"] = table.total[c];
        entry["burleyTotal"] = burleyTotal;
        entry["networkTotal"] = networkTotal;
        entry["burleyRms"] = burleyRms;
        entry["networkRms"] = networkRms;
        configs.Push(std::move(entry));
    }
    const double n = double(std::max<size_t>(table.configs.size(), 1));
    report["configs"] = std::move(configs);
    report["burleyRms"] = burleySum / n;
    report["networkRms"] = networkSum / n;
    report["burleyEnergyError"] = energyBurley / n;
    report["networkEnergyError"] = energyNetwork / n;
    return report;
}

std::string SubsurfaceProfileHeader(std::span<const float> parameters)
{
    std::string out =
        "// The neural subsurface profile's MLP (Subsurface.hlsli): 3 inputs (log(r / d) / 4, the colour and g,\n"
        "// mapped to about [-1, 1]), two hidden layers of 16 ReLUs, one output z, which scales Burley's profile\n"
        "// by exp(z). Per neuron its input weights, then its bias. Trained by `aircli subsurface-fit` on the\n"
        "// half-space walk (SubsurfaceFit.cpp), which writes this file. Shared by HLSL and C++.\n"
        "#ifndef AIR_PT_SUBSURFACE_PROFILE_NETWORK_H\n"
        "#define AIR_PT_SUBSURFACE_PROFILE_NETWORK_H\n\n"
        "AIR_CONST uint kSubsurfaceProfileInputs = 3u;\n"
        "AIR_CONST uint kSubsurfaceProfileHidden = 16u;\n"
        "AIR_CONST uint kSubsurfaceProfileParameters = 353u;\n\n"
        "AIR_CONST float kSubsurfaceProfileWeights[353] = {\n";
    std::string line = "   ";
    for (float v : parameters)
    {
        std::string number = std::format("{:.8g}", v);
        if (number.find_first_of(".e") == std::string::npos)
        {
            number += ".0";
        }
        const std::string piece = " " + number + "f,";
        if (line.size() + piece.size() > 118)
        {
            out += line + "\n";
            line = "   ";
        }
        line += piece;
    }
    out += line + "\n};\n\n#endif\n";
    return out;
}

} // namespace air::pt
