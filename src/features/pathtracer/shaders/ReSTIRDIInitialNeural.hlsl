// ReSTIR DI's first pass for scenes with coated materials: their coats evaluated as the trace's neural kernels
// evaluate them (AIR_PT_NEURAL, see PathTracerNeural.hlsl).

#define AIR_PT_NEURAL 1
#include "ReSTIRDI.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    InitialPass(pixel);
}
