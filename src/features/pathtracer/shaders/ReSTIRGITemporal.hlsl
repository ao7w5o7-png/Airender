// ReSTIR GI, first pass: reuse of last frame's reservoir at the reprojected pixel (ReSTIRGI.hlsli).

#include "ReSTIRGI.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    GiTemporalPass(pixel);
}
