// air-shader-target: lib_6_6
// PathTracer.hlsl that decodes neural texture sets on sample at the first surface (AIR_PT_NTC,
// docs/architecture.md §8.16). A variant of its own because the decoder's registers cost the kernel
// about 8% where no set needs it; PathTracer.cpp uses it while the scene has such sets (also in place
// of the SER kernel).

#define AIR_PT_NTC 1
#include "PathTracerKernel.hlsli"
