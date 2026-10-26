// Neural texture sets on the GPU (renderer/NeuralTextures.h), shared by C++ and HLSL. The path tracer
// decodes them on sample through TextureSetData (SceneShared.h).
#ifndef AIR_RENDERER_NEURAL_TEXTURE_SHARED_H
#define AIR_RENDERER_NEURAL_TEXTURE_SHARED_H

#include "air/renderer/shader/HlslCompat.h"

#ifdef __cplusplus
namespace air::gpu
{
using namespace air::hlsl;
#endif

// A set's channels: colour 0-2 (stored in sRGB), the normal's x and y 3-4, roughness 5, metallic 6.
AIR_CONST uint kNeuralTextureChannels = 7u;

// NtcCheck.hlsl (renderer::CheckNeuralTextures, aircli ntc-check): a set decoded at every texel centre
// by LibNTC's shader code, for comparison with its decoding on CUDA.
struct NeuralTextureCheckConstants
{
    uint latents; // Texture2DArray<float4>: the set's latents
    uint data;    // ByteAddressBuffer: NtcTextureSetConstants, then the int8 weights
    uint outputs; // RWByteAddressBuffer: kNeuralTextureChannels float32 per texel, in their stored spaces
    uint width;
    uint height;
    uint pad0;
    uint pad1;
    uint pad2;
};

#ifdef __cplusplus
static_assert(sizeof(NeuralTextureCheckConstants) == 32);
} // namespace air::gpu
#endif

#endif
