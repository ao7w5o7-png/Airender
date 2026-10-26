// air-shader-target: lib_6_6
// PathTracer.hlsl with the neural parts (AIR_PT_NEURAL): neural texture sets decoded on sample at the
// first surface (docs/architecture.md §8.16) and the clear coats' random walk and network (§8.17). A
// variant of its own because their code costs the kernel 7-18% where no scene needs it; PathTracer.cpp
// uses it while the scene has such sets or coated materials (also in place of the SER kernel).

#define AIR_PT_NEURAL 1
#include "PathTracerKernel.hlsli"
