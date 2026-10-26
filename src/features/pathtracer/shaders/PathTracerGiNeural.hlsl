// air-shader-target: lib_6_6
// PathTracerGi.hlsl with the neural parts (AIR_PT_NEURAL, see PathTracerNeural.hlsl).

#define AIR_PT_RESTIR_GI 1
#define AIR_PT_NEURAL 1
#include "PathTracerKernel.hlsli"
