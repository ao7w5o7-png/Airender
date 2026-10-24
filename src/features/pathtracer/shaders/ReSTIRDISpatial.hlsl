// ReSTIR DI, last pass: reuse from neighbouring pixels, shading and the next frame's history
// (ReSTIRDI.hlsli).

#include "ReSTIRDI.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    SpatialPass(pixel);
}
