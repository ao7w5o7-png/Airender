// ReSTIR DI, second pass: reuse of last frame's reservoir at the reprojected pixel (ReSTIRDI.hlsli).

#include "ReSTIRDI.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    TemporalPass(pixel);
}
