// ReSTIR GI's spatial pass for scenes with coated materials: their coats evaluated as the trace's neural kernels
// evaluate them (AIR_PT_NEURAL, see PathTracerNeural.hlsl).

#define AIR_PT_NEURAL 1
#include "ReSTIRGI.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    GiSpatialPass(pixel);
}
