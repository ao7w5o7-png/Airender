// ReSTIR DI, last pass: reuse from neighbouring pixels, shading and the next frame's history
// (ReSTIR.hlsli).

#include "ReSTIR.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    SpatialPass(pixel);
}
