// ReSTIR GI, last pass: reuse from neighbouring pixels, shading and the next frame's history
// (ReSTIRGI.hlsli).

#include "ReSTIRGI.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    GiSpatialPass(pixel);
}
