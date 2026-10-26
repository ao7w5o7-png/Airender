// The path tracer's signals and guides in NRD's encodings (NRD.hlsli's front end): each lobe's light
// divided by its albedo (at least kNrdAlbedoFloor) with its hit distance (normalized for REBLUR), and
// the normal with the roughness in the octahedral R10G10B10A2 layout. Sky pixels are left zero:
// NRD skips everything beyond the denoising range.

#include "air/renderer/shader/SceneShared.h"
#include "NrdShared.h"
#include "NRD.hlsli"

[[vk::push_constant]] ConstantBuffer<NrdPrepareConstants> g : register(b0);

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    Texture2D<float4> diffuseSignal = ResourceDescriptorHeap[g.diffuseSignal];
    Texture2D<float4> specularSignal = ResourceDescriptorHeap[g.specularSignal];
    Texture2D<float4> diffuseAlbedo = ResourceDescriptorHeap[g.diffuseAlbedo];
    Texture2D<float4> specularAlbedo = ResourceDescriptorHeap[g.specularAlbedo];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.normalRoughness];
    Texture2D<float> viewDepth = ResourceDescriptorHeap[g.viewDepth];
    RWTexture2D<float4> inDiffuse = ResourceDescriptorHeap[g.inDiffuse];
    RWTexture2D<float4> inSpecular = ResourceDescriptorHeap[g.inSpecular];
    RWTexture2D<float4> inNormalRoughness = ResourceDescriptorHeap[g.inNormalRoughness];

    float viewZ = viewDepth[id.xy];
    float4 nr = normalRoughness[id.xy];
    if (!(viewZ < g.denoisingRange) || dot(nr.xyz, nr.xyz) == 0.0f)
    {
        inDiffuse[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        inSpecular[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        inNormalRoughness[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    float4 d = diffuseSignal[id.xy];
    float4 s = specularSignal[id.xy];
    float3 diffuse = d.rgb / max(diffuseAlbedo[id.xy].rgb, kNrdAlbedoFloor);
    float3 specular = s.rgb / max(specularAlbedo[id.xy].rgb, kNrdAlbedoFloor);
    float roughness = saturate(nr.w);
    if (g.denoiser == kNrdReblur)
    {
        float3 parameters = float3(g.hitDistanceA, g.hitDistanceB, g.hitDistanceC);
        inDiffuse[id.xy] = REBLUR_FrontEnd_PackRadianceAndNormHitDist(
            diffuse, REBLUR_FrontEnd_GetNormHitDist(d.a, viewZ, parameters, 1.0f), true);
        inSpecular[id.xy] = REBLUR_FrontEnd_PackRadianceAndNormHitDist(
            specular, REBLUR_FrontEnd_GetNormHitDist(s.a, viewZ, parameters, roughness), true);
    }
    else
    {
        inDiffuse[id.xy] = RELAX_FrontEnd_PackRadianceAndHitDist(diffuse, d.a, true);
        inSpecular[id.xy] = RELAX_FrontEnd_PackRadianceAndHitDist(specular, s.a, true);
    }
    inNormalRoughness[id.xy] = NRD_FrontEnd_PackNormalAndRoughness(normalize(nr.xyz), roughness, 0.0f);
}
