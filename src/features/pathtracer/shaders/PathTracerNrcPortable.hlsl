// air-shader-target: cs_6_6
// The path tracer whose paths end in the neural radiance cache (kFlagNrc, NrcShared.h), predicted in
// float32 (renderer/shader/Mlp.hlsli) on devices without Shader Model 6.10 linear algebra. A compute
// shader like PathTracerNrc.hlsl, so the two compare like for like.

#define AIR_PT_NRC 2
#define AIR_PT_COMPUTE 1
#include "PathTracerKernel.hlsli"
