// Probe GI voxelization (Voxelize.hlsli): the Chebyshev distance in bricks to the nearest brick a surface
// crosses, one axis per pass (the max-norm distance transform is separable): x from the occupancy, then
// y and z from the previous pass, min over the line of max(offset, input). Logical coordinates are relative
// to the cascade's first brick; storage is toroidal. Outside the cascade nothing is occupied.

#include "Voxelize.hlsli"

[numthreads(4, 4, 4)]
void Main(uint3 id : SV_DispatchThreadID)
{
    CascadeData c = LoadCascade();
    int3 firstBrick = Origin(c) / int(kVoxelBrick);
    Texture3D<uint> input = ResourceDescriptorHeap[g.bricksIn];
    RWTexture3D<uint> output = ResourceDescriptorHeap[g.bricks];
    int3 axis = int3(g.axis == 0u ? 1 : 0, g.axis == 1u ? 1 : 0, g.axis == 2u ? 1 : 0);
    int position = int(dot(int3(id), axis));
    uint best = kBrickDistanceMax;
    for (int offset = -int(kBrickDistanceMax); offset <= int(kBrickDistanceMax); ++offset)
    {
        int at = position + offset;
        if (at < 0 || at >= int(kBrickResolution))
        {
            continue;
        }
        int3 cell = int3(id) + axis * offset;
        uint3 s = uint3((firstBrick + cell) & int(kBrickResolution - 1u));
        uint value = input.Load(int4(s, 0));
        uint candidate;
        if (g.axis == 0u)
        {
            candidate = value != 0u ? uint(abs(offset)) : kBrickDistanceMax; // occupancy
        }
        else
        {
            candidate = max(uint(abs(offset)), value);
        }
        best = min(best, candidate);
    }
    output[uint3((firstBrick + int3(id)) & int(kBrickResolution - 1u))] = best;
}
