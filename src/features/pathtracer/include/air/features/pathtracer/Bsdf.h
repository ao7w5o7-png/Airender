#pragma once

// CPU access to the path tracer's BSDF (shaders/Bsdf.hlsli). The GPU and CPU run the same source,
// so the lookup table built here is the one uploaded to the GPU, and tests can check energy
// conservation (white furnace), reciprocity and sampling of the exact shader code.

#include "air/core/Types.h"

#include <array>
#include <span>
#include <vector>

namespace air::pt
{

using Rgb = std::array<float, 3>;
using Direction = std::array<float, 3>; // local shading frame, z = normal, pointing away from the surface

// How a clear coat's inner part is evaluated (Bsdf.hlsli kCoat*).
enum class CoatModel : u32
{
    None,
    Neural,     // the approximation corrected by the MLP bound with BindNeuralCoat (else approximate)
    RandomWalk, // the reference: one random walk per evaluation
    Approximate,
};

struct Material
{
    Rgb baseColor = {0.8f, 0.8f, 0.8f};
    float roughness = 0.5f; // perceptual; alpha = roughness^2
    float metallic = 0.0f;
    CoatModel coat = CoatModel::None; // a clear coat over it, evaluated so
    float coatRoughness = 0.05f;
    Rgb coatColor = {1.0f, 1.0f, 1.0f}; // let through at normal incidence, one way
    u32 coatSeed = 0;                   // CoatModel::RandomWalk: the walks' random numbers
};

struct BsdfValue
{
    Rgb fCos{};       // f(wo, wi) * cos(theta_i)
    float pdf = 0.0f; // solid angle pdf of SampleBsdf producing wi
};

struct BsdfDraw
{
    Direction wi{};
    Rgb weight{}; // f * cos / pdf
    float pdf = 0.0f;
    bool valid = false;
};

// Directional albedo and Kulla-Conty tables (layout in Bsdf.hlsli). Built once on first use, thread safe.
std::span<const float> GetBsdfLut();

BsdfValue EvaluateBsdf(const Material& material, const Direction& wo, const Direction& wi);
BsdfDraw SampleBsdf(const Material& material, const Direction& wo, float uLobe, float u0, float u1);

// The parameters of the neural coat's MLP (NeuralCoat.hlsli) that CoatModel::Neural evaluates on this
// thread; empty: none.
void BindNeuralCoat(std::span<const float> parameters);

// Monte Carlo estimate of the directional albedo towards wo = (sin, 0, cos) using SampleBsdf (a
// random walk coat draws new walks for every sample).
Rgb EstimateAlbedo(const Material& material, float cosThetaO, u32 sampleCount);
// The same integral with uniform hemisphere sampling of EvaluateBsdf: independent of the sampler.
Rgb EstimateAlbedoUniform(const Material& material, float cosThetaO, u32 sampleCount);

} // namespace air::pt
