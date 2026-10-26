// air-shader-target: lib_6_6
// The path tracer with the scene's volume (VolumeLookup.hlsli, docs/architecture.md §8.12). A variant
// of its own because tracking a volume costs the plain kernel about 5% even where there is none;
// PathTracer.cpp uses it whenever the scene has one.

#define AIR_PT_VOLUME 1
#include "PathTracerKernel.hlsli"
