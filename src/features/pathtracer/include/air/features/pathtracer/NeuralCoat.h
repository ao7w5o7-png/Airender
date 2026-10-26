#pragma once

// The clear coat's neural inner part (shaders/NeuralCoat.hlsli, docs/architecture.md §8.17): an MLP
// trained on the CPU, from random walks through the coat (Bsdf.hlsli's reference), to correct the
// approximation of the light the coat lets in and out again, for the coated materials of a scene.
// The GPU and the CPU reference evaluate the same parameters.

#include "air/core/Json.h"
#include "air/core/TaskSystem.h"
#include "air/core/Types.h"
#include "air/renderer/Scene.h"

#include <filesystem>
#include <span>
#include <vector>

namespace air::pt
{

// A coated material as the BSDF sees it at one shading point (after its textures).
struct CoatMaterialSample
{
    float baseColor[3] = {0.8f, 0.8f, 0.8f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    float coatRoughness = 0.05f;
    float coatColor[3] = {1.0f, 1.0f, 1.0f};
};

// The scene's coated materials, up to `count` samples spread over them: a textured one by texels of
// its textures (scaling its constants as the integrator does), the others as they are. Empty: none.
std::vector<CoatMaterialSample> CollectCoatMaterials(const renderer::SceneDesc& scene, u32 count = 4096);

struct NeuralCoatSettings
{
    u32 steps = 1500;           // Adam steps
    u32 batch = 4096;           // samples per step, each with fresh directions
    u32 walks = 4;              // random walks averaged into a sample's target
    float learningRate = 4e-3f; // decays to a tenth over the steps
    u32 seed = 1;
    std::filesystem::path cache = std::filesystem::temp_directory_path() / "airender-neural-coat";
};

struct NeuralCoatResult
{
    std::vector<float> parameters; // kNeuralCoatParameters
    float loss = 0.0f;             // relative L2 of the last steps
    bool cached = false;
};

// Trains the MLP for these materials (or finds it in the cache: the same materials and settings give
// the same parameters, also across runs). Deterministic for a given task count.
NeuralCoatResult TrainNeuralCoat(std::span<const CoatMaterialSample> materials, const NeuralCoatSettings& settings,
                                 TaskSystem* tasks);

// The network against the random walk for these materials: per material sample, the inner part's
// error relative to its size over random directions (the network's and the approximation's), and
// the directional albedo towards a few angles by the three models (aircli coat-check).
json::Value CheckNeuralCoat(std::span<const CoatMaterialSample> materials, std::span<const float> parameters,
                            u32 directions, TaskSystem* tasks);

} // namespace air::pt
