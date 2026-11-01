// Probe GI debug views of the voxel scene (probe_gi.view 4..7, 9): every pixel's camera ray traced through
// the voxels as the GI's rays are, showing what they hit: its albedo, the light it reflects (exposed), the
// steps taken (Turbo), the cascade (colours) or the radiance cache's irradiance there.
//   GiConstants: output = sceneColor (RWTexture2D<float4>), width, height.

#include "VoxelTrace.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

// Turbo colormap (Mikhailov 2019), polynomial fit, sRGB encoded.
float3 Turbo(float x)
{
    x = saturate(x);
    const float4 kr = float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
    const float4 kg = float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
    const float4 kb = float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
    const float2 kr2 = float2(-152.94239396f, 59.28637943f);
    const float2 kg2 = float2(4.27729857f, 2.82956604f);
    const float2 kb2 = float2(-89.90310912f, 27.34824973f);
    float4 v4 = float4(1.0f, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;
    return float3(dot(v4, kr) + dot(v2, kr2), dot(v4, kg) + dot(v2, kg2), dot(v4, kb) + dot(v2, kb2));
}

float3 SrgbToLinear(float3 c)
{
    c = saturate(c);
    return select(c <= 0.04045f, c / 12.92f, pow((c + 0.055f) / 1.055f, 2.4f));
}

// The distance field's gradient: the surface's normal on the side the ray came from.
float3 FieldNormal(CascadeData c, float3 v)
{
    float3 grad = float3(SampleDistance(c, v + float3(0.5f, 0, 0)) - SampleDistance(c, v - float3(0.5f, 0, 0)),
                         SampleDistance(c, v + float3(0, 0.5f, 0)) - SampleDistance(c, v - float3(0, 0.5f, 0)),
                         SampleDistance(c, v + float3(0, 0, 0.5f)) - SampleDistance(c, v - float3(0, 0, 0.5f)));
    return dot(grad, grad) > 1e-12f ? normalize(grad) : float3(0.0f, 1.0f, 0.0f);
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g.width || id.y >= g.height)
    {
        return;
    }
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    float2 size = float2(g.width, g.height);
    float3 direction = CameraRay(f.camera, float2(id.xy) + 0.5f + f.jitter, size);
    float3 origin = f.camera.position.xyz;
    VoxelHit hit = TraceVoxels(f, origin, direction, 0.0f, 1e30f);
    float3 value = float3(0.0f, 0.0f, 0.0f);
    if (f.view == kGiViewSteps)
    {
        value = SrgbToLinear(Turbo(float(hit.steps) / max(f.debugScale, 1.0f)));
    }
    else if (!hit.hit)
    {
        value = f.view == kGiViewVoxelLight ? SkyLight(f, direction) * f.exposure : float3(0.0f, 0.0f, 0.0f);
    }
    else
    {
        CascadeData c = f.cascades[hit.cascade];
        bool found;
        bool backface;
        if (f.view == kGiViewVoxels)
        {
            value = VoxelAlbedo(c, hit.voxel, direction, found);
            if (!found)
            {
                value = float3(1.0f, 0.0f, 1.0f); // a hit without a surface voxel
            }
        }
        else if (f.view == kGiViewVoxelLight)
        {
            value = VoxelRadiance(f, hit.cascade, hit.voxel, direction, found, backface) * f.exposure;
            if (backface)
            {
                value = float3(0.0f, 0.0f, 0.05f);
            }
        }
        else if (f.view == kGiViewCascades)
        {
            const float3 colours[8] = {float3(1, 0.2, 0.2), float3(1, 0.6, 0.1), float3(0.9, 0.9, 0.1), float3(0.2, 0.9, 0.2),
                                       float3(0.1, 0.8, 0.9), float3(0.2, 0.3, 1), float3(0.7, 0.2, 1), float3(1, 1, 1)};
            value = colours[hit.cascade] * (0.4f + 0.6f * frac(hit.t * 0.25f));
        }
        else if (f.view == kGiViewCache)
        {
            float3 n = FieldNormal(c, hit.voxel);
            float3 p = origin + direction * hit.t + n * (1.5f * c.voxelSize);
            float3 irradiance;
            if (CacheIrradiance(f, p, n, irradiance))
            {
                value = irradiance * kInvPi * f.exposure;
            }
            else
            {
                value = float3(0.3f, 0.0f, 0.3f);
            }
        }
    }
    RWTexture2D<float4> output = ResourceDescriptorHeap[g.output];
    output[id.xy] = float4(value, 1.0f);
}
