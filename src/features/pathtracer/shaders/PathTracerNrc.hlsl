// air-shader-target: cs_6_10
// air-shader-dxil-only
// The path tracer whose paths end in the neural radiance cache (kFlagNrc, NrcShared.h,
// docs/architecture.md §8.11), predicted on the matrix units with Shader Model 6.10 linear algebra. A
// compute shader (its rays are inline queries): the driver runs linear algebra there, not in ray
// tracing pipelines. PathTracer.cpp uses it while the cache runs on a device with
// rhi::DeviceCaps::linearAlgebra.

#define AIR_PT_NRC 1
#define AIR_PT_COMPUTE 1
#include "PathTracerKernel.hlsli"
