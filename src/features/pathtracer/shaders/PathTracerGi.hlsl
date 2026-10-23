// air-shader-target: lib_6_6
// The path tracer recording ReSTIR GI's candidates (kFlagReSTIRGI, docs/architecture.md §8.8). A
// variant of its own because that state costs the plain kernel half its speed; PathTracer.cpp uses
// it whenever ReSTIR GI runs.

#define AIR_PT_RESTIR_GI 1
#include "PathTracerKernel.hlsli"
