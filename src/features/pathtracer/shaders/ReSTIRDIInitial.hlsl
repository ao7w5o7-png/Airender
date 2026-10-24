// ReSTIR DI, first pass: each pixel's initial reservoir from light and BSDF candidates (ReSTIRDI.hlsli).

#include "ReSTIRDI.hlsli"

[numthreads(8, 8, 1)]
void Main(uint2 pixel : SV_DispatchThreadID)
{
    InitialPass(pixel);
}
