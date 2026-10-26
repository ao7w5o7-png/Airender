// The denoised light of the surfaces in place of the noisy: the frame is what lies in front of them
// plus their transmittance times (emission + diffuse + specular), so adding the transmittance times
// each albedo times (denoised - noisy) swaps exactly the signals NRD was given and leaves everything
// else (emission, the sky, volumes, light the signals do not carry) as it was.

#include "air/renderer/shader/SceneShared.h"
#include "NrdShared.h"
#include "NRD.hlsli"

[[vk::push_constant]] ConstantBuffer<NrdCompositeConstants> g : register(b0);

// What Prepare.hlsl gave NRD, before its encoding (RELAX_FrontEnd_PackRadianceAndHitDist's sanitizing).
float3 Noisy(float3 light, float3 albedo)
{
    float3 x = light / max(albedo, kNrdAlbedoFloor);
    return any(isnan(x)) || any(isinf(x)) ? float3(0.0f, 0.0f, 0.0f) : clamp(x, 0.0f, NRD_FP16_MAX);
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    Texture2D<float> viewDepth = ResourceDescriptorHeap[g.viewDepth];
    if (!(viewDepth[id.xy] < g.denoisingRange))
    {
        if (g.view == 1u)
        {
            RWTexture2D<float4> sky = ResourceDescriptorHeap[g.color];
            sky[id.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        return; // NRD does not write beyond the denoising range
    }
    Texture2D<float4> diffuseSignal = ResourceDescriptorHeap[g.diffuseSignal];
    Texture2D<float4> specularSignal = ResourceDescriptorHeap[g.specularSignal];
    Texture2D<float4> diffuseAlbedo = ResourceDescriptorHeap[g.diffuseAlbedo];
    Texture2D<float4> specularAlbedo = ResourceDescriptorHeap[g.specularAlbedo];
    Texture2D<float4> outDiffuse = ResourceDescriptorHeap[g.outDiffuse];
    Texture2D<float4> outSpecular = ResourceDescriptorHeap[g.outSpecular];
    RWTexture2D<float4> color = ResourceDescriptorHeap[g.color];

    float3 da = max(diffuseAlbedo[id.xy].rgb, kNrdAlbedoFloor);
    float3 sa = max(specularAlbedo[id.xy].rgb, kNrdAlbedoFloor);
    float3 diffuse;
    float3 specular;
    if (g.denoiser == kNrdReblur)
    {
        diffuse = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(outDiffuse[id.xy]).rgb;
        specular = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(outSpecular[id.xy]).rgb;
    }
    else
    {
        diffuse = RELAX_BackEnd_UnpackRadiance(outDiffuse[id.xy]).rgb;
        specular = RELAX_BackEnd_UnpackRadiance(outSpecular[id.xy]).rgb;
    }
    float transmittance = 1.0f;
    if (g.surfaceTransmittance != kInvalidIndex)
    {
        Texture2D<float> t = ResourceDescriptorHeap[g.surfaceTransmittance];
        transmittance = t[id.xy];
    }
    float3 change = da * (max(diffuse, 0.0f) - Noisy(diffuseSignal[id.xy].rgb, da)) +
                    sa * (max(specular, 0.0f) - Noisy(specularSignal[id.xy].rgb, sa));
    float4 c = color[id.xy];
    if (g.view == 1u)
    {
        // The global illumination alone (the signals carry the indirect light), denoised.
        color[id.xy] = float4(transmittance * (da * max(diffuse, 0.0f) + sa * max(specular, 0.0f)), c.a);
        return;
    }
    color[id.xy] = float4(max(c.rgb + transmittance * change, 0.0f), c.a);
}
