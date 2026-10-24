// The neural radiance cache's inputs and outputs (NrcShared.h): what the network sees of a query and
// what its outputs mean. Shared by the trace kernel's cache lookups, NrcTrain*.hlsl and
// NrcResolve*.hlsl. Include after Mlp.hlsli (whose hooks read the hash grid).
#ifndef AIR_PT_NRC_HLSLI
#define AIR_PT_NRC_HLSLI

#include "air/renderer/shader/Mlp.hlsli"
#include "NrcShared.h"

// The network's 64 inputs: 32 from the hash grid at the position, 16 spherical harmonics of the
// direction the light leaves in, the normal, the roughness, both albedos and a 1 (the first layer's
// bias).
MlpVector NrcEncode(HashGridData grid, NrcQuery q)
{
    MlpVector x = MlpZero();
    MlpEncodeHashGrid(grid, q.position, x);
    float4 sh0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sh1 = sh0;
    float4 sh2 = sh0;
    float4 sh3 = sh0;
    MlpSphericalHarmonics(NrcUnpackDirection(q.direction), sh0, sh1, sh2, sh3);
    x.v[8] = sh0;
    x.v[9] = sh1;
    x.v[10] = sh2;
    x.v[11] = sh3;
    float3 n = NrcUnpackDirection(q.normal);
    x.v[12] = float4(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, NrcUnpackUnorm16(q.roughness, 0u));
    x.v[13] = float4(NrcUnpackColor(q.diffuse), 1.0f);
    x.v[14] = float4(NrcUnpackColor(q.specular), 0.0f);
    return x;
}

// Reflected radiance (nits) from the network's first three outputs.
float3 NrcRadiance(NrcQuery q, float3 output, float normalization)
{
    float3 positive = float3(max(output.x, 0.0f), max(output.y, 0.0f), max(output.z, 0.0f));
    return positive * NrcDemodulation(q) / normalization;
}

// What the network should output for a record (its units: normalized and demodulated).
float3 NrcTarget(NrcTrainingRecord r, float normalization)
{
    float3 t = r.value * normalization / NrcDemodulation(r.query);
    return float3(min(t.x, 1e4f), min(t.y, 1e4f), min(t.z, 1e4f));
}

#endif
