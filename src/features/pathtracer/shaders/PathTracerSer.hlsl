// air-shader-target: lib_6_9
// air-shader-dxil-only
// The path tracer with DXR 1.2 shader execution reordering (SM 6.9). Used when the device actually
// reorders (rhi::DeviceCaps::shaderExecutionReordering) and pathtracer.ser is on (off by default:
// measured slower with the current cheap shading, see docs/architecture.md §8.2). Every thread runs
// the same estimator as PathTracer.hlsl; the two compilations round a few paths differently (about
// 1% of pixels differ at 256 spp, the converged means are equal), so compare images statistically.

#define AIR_PT_SER 1
#include "PathTracerKernel.hlsli"
