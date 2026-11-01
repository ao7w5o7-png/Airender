#pragma once

// The neural subsurface profile (shaders/SubsurfaceProfile.hlsli, docs/architecture.md §8.20): the
// light that leaves a flat half-space of a subsurface material at a distance r from where it entered,
// learnt from random walks in it. The profile depends only on the material's colour, its anisotropy
// and r over its radius, so one network serves every material: `aircli subsurface-fit` simulates the
// walks, fits the MLP and writes shaders/SubsurfaceProfileNetwork.h, which the GPU and the CPU compile.

#include "air/core/Json.h"
#include "air/core/TaskSystem.h"
#include "air/core/Types.h"

#include <span>
#include <string>
#include <vector>

namespace air::pt
{

// The half-space walk's profile for a set of (colour, anisotropy), with the radius 1: R(r) per unit
// area in bins of log(r), and what leaves in all.
struct SubsurfaceProfileTable
{
    struct Config
    {
        float color = 0.5f;
        float anisotropy = 0.0f;
    };
    std::vector<Config> configs;
    u32 bins = 50;
    float logMin = -6.0f; // bin b spans log(r) in logMin + [b, b + 1] (logMax - logMin) / bins
    float logMax = 4.0f;
    std::vector<float> profile; // [config][bin], the bin's average
    std::vector<u32> counts;    // walks that left in the bin
    std::vector<float> total;   // [config]: the light that left in all per unit that entered
    u32 walks = 0;              // per config
};

// The colours and anisotropies the network is fit on: colours from 0.05 to 0.99, g from 0 to 0.9.
std::vector<SubsurfaceProfileTable::Config> SubsurfaceFitConfigs();

// Simulates `walks` walks per config (cosine-weighted entry through an index-matched boundary, the
// medium's single-scattering albedo from the BSDF LUT's inversion and its extinction from the colour,
// Subsurface.hlsli's walk in a flat half-space). Deterministic for a seed, whatever the task count.
SubsurfaceProfileTable SimulateSubsurfaceProfiles(std::span<const SubsurfaceProfileTable::Config> configs, u32 walks,
                                                  u32 seed, TaskSystem* tasks);

struct SubsurfaceFitSettings
{
    u32 steps = 6000;           // Adam steps over the whole table
    float learningRate = 3e-3f; // decays to a tenth over the steps
    u32 seed = 1;
};

struct SubsurfaceFitResult
{
    std::vector<float> parameters; // kSubsurfaceProfileParameters
    float loss = 0.0f;             // the weighted squared error of z at the end
};

// Fits the MLP's correction z = log(R / (A Burley)) to the table, weighing every bin by its share of
// the light (bins with few walks left out).
SubsurfaceFitResult FitSubsurfaceProfile(const SubsurfaceProfileTable& table, const SubsurfaceFitSettings& settings,
                                         TaskSystem* tasks);

// The network's z(log(r / d), colour, g) with these parameters (the shader's code, on the CPU).
float EvaluateSubsurfaceProfileCorrection(std::span<const float> parameters, float logRadius, float color,
                                          float anisotropy);
// The parameters compiled into the shaders (SubsurfaceProfileNetwork.h).
std::span<const float> CompiledSubsurfaceProfile();
// The compiled network as the shader code evaluates it (SubsurfaceProfile.hlsli).
float ShaderSubsurfaceProfileCorrection(float logRadius, float color, float anisotropy);

// Burley's profile and the network's against the table: per config the light that leaves in all
// (walk, Burley, network) and the profile's error relative to the walk's, weighed by the light (RMS),
// and their averages over the table.
json::Value CheckSubsurfaceProfile(const SubsurfaceProfileTable& table, std::span<const float> parameters);

// SubsurfaceProfileNetwork.h with these parameters.
std::string SubsurfaceProfileHeader(std::span<const float> parameters);

} // namespace air::pt
