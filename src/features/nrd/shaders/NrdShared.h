// Shared between NrdFeature.cpp and the nrd feature's shaders (Prepare.hlsl, Composite.hlsl).
#ifndef AIR_NRD_SHARED_H
#define AIR_NRD_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::nrdf::gpu
{
using namespace air::hlsl;
#endif

AIR_CONST uint kNrdRelax = 0u;  // NRD's RELAX_DIFFUSE_SPECULAR
AIR_CONST uint kNrdReblur = 1u; // NRD's REBLUR_DIFFUSE_SPECULAR
// The albedos the signals are divided by (and multiplied by again) are at least this.
AIR_CONST float kNrdAlbedoFloor = 0.01f;

// The path tracer's signals and guides -> NRD's inputs.
struct NrdPrepareConstants
{
    uint diffuseSignal;     // Texture2D<float4>: renderer::slots::kDiffuseSignal
    uint specularSignal;    // Texture2D<float4>
    uint diffuseAlbedo;     // Texture2D<float4>
    uint specularAlbedo;    // Texture2D<float4>
    uint normalRoughness;   // Texture2D<float4>: world normal, a = perceptual roughness
    uint viewDepth;         // Texture2D<float>
    uint inDiffuse;         // RWTexture2D<float4>: IN_DIFF_RADIANCE_HITDIST
    uint inSpecular;        // RWTexture2D<float4>: IN_SPEC_RADIANCE_HITDIST
    uint inNormalRoughness; // RWTexture2D<float4> (R10G10B10A2): IN_NORMAL_ROUGHNESS
    uint width;
    uint height;
    uint denoiser;      // kNrd*
    float hitDistanceA; // REBLUR's hit distance normalization (ReblurHitDistanceParameters)
    float hitDistanceB;
    float hitDistanceC;
    float denoisingRange; // view depths from here on are not surfaces (the sky)
};

// NRD's outputs -> the frame: the noisy light of the surfaces replaced by the denoised.
struct NrdCompositeConstants
{
    uint color;                // RWTexture2D<float4>: sceneColor
    uint diffuseSignal;        // Texture2D<float4>
    uint specularSignal;       // Texture2D<float4>
    uint diffuseAlbedo;        // Texture2D<float4>
    uint specularAlbedo;       // Texture2D<float4>
    uint viewDepth;            // Texture2D<float>
    uint outDiffuse;           // Texture2D<float4>: OUT_DIFF_RADIANCE_HITDIST
    uint outSpecular;          // Texture2D<float4>: OUT_SPEC_RADIANCE_HITDIST
    uint surfaceTransmittance; // Texture2D<float>: renderer::slots::kSurfaceTransmittance (kInvalidIndex: 1)
    uint width;
    uint height;
    uint denoiser;
    float denoisingRange;
    uint view; // 0: the frame; 1: the denoised signals alone (the indirect light: the GI)
};

#ifdef __cplusplus
static_assert(sizeof(NrdPrepareConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(NrdCompositeConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::nrdf::gpu
#endif

#endif
