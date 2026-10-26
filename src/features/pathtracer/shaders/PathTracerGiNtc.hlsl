// air-shader-target: lib_6_6
// PathTracerGi.hlsl that decodes neural texture sets on sample at the first surface (AIR_PT_NTC, see
// PathTracerNtc.hlsl).

#define AIR_PT_RESTIR_GI 1
#define AIR_PT_NTC 1
#include "PathTracerKernel.hlsli"
