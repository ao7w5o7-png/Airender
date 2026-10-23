// ReSTIR DI, second pass: reuse of last frame's reservoir at the reprojected pixel (ReSTIR.hlsli).

#include "ReSTIR.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    TemporalPass(pixel);
}
