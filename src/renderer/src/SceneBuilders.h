#pragma once

// Helpers for the built-in scene definitions (SceneLibrary.cpp, TestScenes.cpp). Internal to the
// renderer module.

#include "air/renderer/Scene.h"

#include <string>

namespace air::renderer::detail
{

u32 AddMaterial(SceneDesc& scene, Material material);
u32 AddMesh(SceneDesc& scene, Mesh mesh);
Instance& AddInstance(SceneDesc& scene, std::string name, u32 mesh, u32 material);

// Object-to-world transform whose columns are the world-space object axes and the translation.
void SetBasis(Instance& instance, float3 x, float3 y, float3 z, float3 translation);
// Points the camera from `eye` at `target` (camera yaw 0 / pitch 0 looks down -z).
void LookAt(Camera& camera, float3 eye, float3 target);

// Disc in the plane through `center` facing `normal`, as a triangle fan.
Mesh MakeDisc(std::string name, float3 center, float3 normal, float radius, u32 segments);

// Deterministic random numbers (PCG32): the same sequence on every compiler and standard library,
// so generated scenes, and the images rendered from them, match across platforms.
class Random
{
public:
    explicit Random(u64 seed) : mState(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    u32 NextU32()
    {
        const u64 old = mState;
        mState = old * 6364136223846793005ull + 1442695040888963407ull;
        const u32 shifted = u32(((old >> 18u) ^ old) >> 27u);
        const u32 rotation = u32(old >> 59u);
        return (shifted >> rotation) | (shifted << ((32u - rotation) & 31u));
    }
    float Next() { return float(NextU32() >> 8) * (1.0f / 16777216.0f); } // [0, 1)
    float Range(float lo, float hi) { return lo + (hi - lo) * Next(); }

private:
    u64 mState;
};

// Linear RGB for a hue in [0, 1) at full saturation and value.
float3 Hue(float h);

// Test scenes (TestScenes.cpp), registered in SceneLibrary.cpp; see Scene.h for what each one tests.
SceneDesc MakeVeachMis();
SceneDesc MakeMaterialGrid();
SceneDesc MakeSphereField();
SceneDesc MakeManyLights();
SceneDesc MakeDoor();
SceneDesc MakeTerrain();
SceneDesc MakeSunset();

} // namespace air::renderer::detail
