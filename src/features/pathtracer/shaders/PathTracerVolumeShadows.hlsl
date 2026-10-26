// air-shader-target: lib_6_6
// The path tracer under real-time volumes (docs/architecture.md §8.13): the volumes are not tracked
// but ray marched over the surfaces afterwards (VolumeMarch.hlsl), which this kernel's output feeds:
// with kFlagVolumeMarch it writes the frame's samples and their first hit's distance instead of adding
// them to the accumulation. The light that reaches surfaces through volumes comes from their moment
// shadow maps. PathTracer.cpp uses it while the march renders the scene's volumes.

#define AIR_PT_VOLUME_SHADOWS 1
#include "PathTracerKernel.hlsli"
