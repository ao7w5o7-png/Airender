// Probe GI: adds the GI to the frame (ProbeGiShared.h). The pixel's diffuse albedo times its irradiance / pi
// plus its specular albedo times its specular light (the split sum's albedo), exposed and dimmed by the air
// between it and the camera like the rest of what the path tracer resolved there; or a view of the GI alone.
// Also keeps what each pixel's surface reflects (the lit frame without the air, without its emission), for
// next frame's screen traces.
//   GiConstants: output sceneColor, input0 view depth, input1 diffuse albedo, input2 specular albedo,
//   input3 irradiance, input4 specular, input5 emission (kInvalidIndex: none), input6 the surfaces' light
//   (RWTexture2D<float4>, nits times kHalfLightScale), input7 surface transmittance (Texture2D<float>, kInvalidIndex: 1),
//   rays this frame's irradiance before its history (Texture2D<float4>: a 1 where a probe fitted; view 8),
//   diffuseSignal / specularSignal the denoiser's signals (kInvalidIndex: none): the GI is added to them per
//   lobe, exposed and through the air but not the surface's transmittance, as the path tracer writes them.

#include "GiCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kSkyDepth = 65000.0f;

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.width || id.y >= f.height)
    {
        return;
    }
    gSkyTable = f.skyTable;
    RWTexture2D<float4> color = ResourceDescriptorHeap[g.output];
    RWTexture2D<float4> reflected = ResourceDescriptorHeap[g.input6];
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    float z = depth.Load(int3(id.xy, 0));
    float4 c = color[id.xy];
    if (!(z < kSkyDepth) || z <= 0.0f)
    {
        reflected[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (f.view != kGiViewLit)
        {
            color[id.xy] = float4(0.0f, 0.0f, 0.0f, c.w);
        }
        return;
    }
    Texture2D<float4> diffuseAlbedo = ResourceDescriptorHeap[g.input1];
    Texture2D<float4> specularAlbedo = ResourceDescriptorHeap[g.input2];
    Texture2D<float4> irradiance = ResourceDescriptorHeap[g.input3];
    Texture2D<float4> specular = ResourceDescriptorHeap[g.input4];
    float3 view;
    float3 p = PixelPosition(f, id.xy, z, view);
    float distance = z / max(dot(view, f.camera.forward.xyz), 1e-6f);
    Aerial air = AerialPerspective(f.sky, view, distance);
    float transmittance = 1.0f;
    if (g.input7 != kInvalidIndex)
    {
        Texture2D<float> surface = ResourceDescriptorHeap[g.input7];
        transmittance = surface.Load(int3(id.xy, 0));
    }
    float3 e = irradiance.Load(int3(id.xy, 0)).xyz / kHalfLightScale;
    float3 s = specular.Load(int3(id.xy, 0)).xyz / kHalfLightScale;
    float3 diffuseGi = diffuseAlbedo.Load(int3(id.xy, 0)).xyz * e * kInvPi;
    float3 specularGi = specularAlbedo.Load(int3(id.xy, 0)).xyz * s;
    float3 added = (diffuseGi + specularGi) * air.transmittance * (transmittance * f.exposure);
    if (g.diffuseSignal != kInvalidIndex)
    {
        RWTexture2D<float4> diffuseSignal = ResourceDescriptorHeap[g.diffuseSignal];
        RWTexture2D<float4> specularSignal = ResourceDescriptorHeap[g.specularSignal];
        diffuseSignal[id.xy] += float4(diffuseGi * air.transmittance * f.exposure, 0.0f);
        specularSignal[id.xy] += float4(specularGi * air.transmittance * f.exposure, 0.0f);
    }
    float3 direct = c.xyz;
    float3 lit = direct + added;
    switch (f.view)
    {
    case kGiViewGi: c.xyz = added; break;
    case kGiViewIrradiance: c.xyz = e * kInvPi * f.exposure; break;
    case kGiViewSpecular: c.xyz = s * f.exposure; break;
    case kGiViewProbes:
    {
        // Over the GI the pixels none of the 4 probes around fitted: yellow, the 4 x 4 search's; red, the cache's.
        Texture2D<float4> now = ResourceDescriptorHeap[g.rays];
        float fit = now.Load(int3(id.xy, 0)).w;
        c.xyz = fit > 0.9f ? added : (fit > 0.5f ? float3(1.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f));
        break;
    }
    default: c.xyz = lit; break;
    }
    color[id.xy] = c;
    // What the surface reflects towards the camera: the lit frame before the air, without its emission (with
    // kGiDirectVoxels, one bounce to check it, its direct light alone).
    float3 t = max(air.transmittance * transmittance, 1e-4f);
    float3 shown = (f.flags & kGiDirectVoxels) != 0u ? direct : lit;
    float3 surfaceLight = (shown / f.exposure - air.inscatter) / t;
    if (g.input5 != kInvalidIndex)
    {
        Texture2D<float4> emission = ResourceDescriptorHeap[g.input5];
        surfaceLight -= emission.Load(int3(id.xy, 0)).xyz / (f.exposure * t);
    }
    reflected[id.xy] = float4(max(surfaceLight, 0.0f) * kHalfLightScale, 1.0f);
}
