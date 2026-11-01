// Probe GI: every pixel's light from the screen probes (ProbeGiShared.h). The four probes of the tiles
// around the pixel, bilinearly, weighed by how close each lies to the pixel's plane (the depth buffer's, not a
// normal map's: ProbeFit.hlsli) and how alike their
// normals are, give the irradiance for the pixel's normal and the specular light along its reflection (the
// probes' radiance convolved with a lobe as wide as its roughness: a rough lobe only; harmonics up to band 2
// hold no sharp reflections). Where none fits (an edge, a small object between probes) the nearest probes
// that fit in the 3 x 3 tiles around, then the radiance cache's irradiance at the pixel.
//   GiConstants: input0 view depth, input1 normal + roughness, input2 the probes' rays (filtered, ProbeRayFilter.hlsl),
//   input3 probes, input4 harmonics, input6 the traced reflections (ReflectionResolve.hlsl; kInvalidIndex: none), input7
//   the tiles' adaptive probes (StructuredBuffer<uint4>, ProbeAdapt.hlsl; kInvalidIndex: none),
//   output irradiance (RWTexture2D<float4>, rgb: irradiance in nits * sr, a: 1 where one of the 4 probes around fitted,
//   0.75 where the search found one, 0 the cache),
//   input5 specular (RWTexture2D<float4>, rgb: radiance in nits).

#include "VoxelTrace.hlsli"
#include "ProbeFit.hlsli"
#include "ProbeRays.hlsli"
#include "ReflectionCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kSkyDepth = 65000.0f;
// Lobe samples per probe (4 measured alike, 0.13 ms more on sponza at 1920 x 1080). Rough lobes too: the harmonics'
// lobe was off there as well (cornell_box's specular light +14.5% against -5.9%).
static const uint kSpecularSamples = 2u;


// The irradiance for normal n from a probe of normal probeNormal: exact at the probe's normal (its rays'
// quadrature), plus the harmonics' change from there to n.
float3 ProbeIrradiance(ProbeSh sh, float3 n, float3 probeNormal)
{
    return max(sh.normalIrradiance.xyz + ShEvaluate(sh, n) - ShEvaluate(sh, probeNormal), 0.0f);
}

// The probe's radiance harmonics (its irradiance over the clamped cosine's bands) convolved with a lobe of
// zonal factors exp(-l^2 / 2s), s the Phong exponent of GGX roughness, evaluated along r.
float3 ShSpecular(ProbeSh sh, float3 r, float roughness)
{
    float alpha = max(roughness * roughness, 0.02f);
    float s = max(2.0f / (alpha * alpha) - 2.0f, 1.0f);
    float l1 = exp(-0.5f / s) / 2.0943951f;
    float l2 = exp(-2.0f / s) / 0.7853982f;
    float b[9];
    ShBasis(r, b);
    float3 c0 = sh.c0.xyz;
    float3 c1 = float3(sh.c0.w, sh.c1.x, sh.c1.y);
    float3 c2 = float3(sh.c1.z, sh.c1.w, sh.c2.x);
    float3 c3 = float3(sh.c2.y, sh.c2.z, sh.c2.w);
    float3 c4 = sh.c3.xyz;
    float3 c5 = float3(sh.c3.w, sh.c4.x, sh.c4.y);
    float3 c6 = float3(sh.c4.z, sh.c4.w, sh.c5.x);
    float3 c7 = float3(sh.c5.y, sh.c5.z, sh.c5.w);
    float3 c8 = sh.c6;
    float3 v = c0 * (b[0] / kPi) + (c1 * b[1] + c2 * b[2] + c3 * b[3]) * l1 +
               (c4 * b[4] + c5 * b[5] + c6 * b[6] + c7 * b[7] + c8 * b[8]) * l2;
    return max(v, 0.0f);
}


float2 HemiOctEncode(float3 v)
{
    v /= abs(v.x) + abs(v.y) + abs(v.z);
    return float2(v.x + v.y, v.x - v.y);
}

// The light a probe's rays (this frame's, filtered) saw arriving from `direction`, bilinear over its cells; false
// below its hemisphere.
bool ProbeRadiance(Texture2D<float4> rays, uint2 tile, Basis basis, float3 direction, out float3 radiance)
{
    radiance = float3(0.0f, 0.0f, 0.0f);
    float3 local = ToLocal(basis, direction);
    if (local.z <= 0.0f)
    {
        return false;
    }
    float2 uv = (HemiOctEncode(local) * 0.5f + 0.5f) * float(kScreenProbeTexels) - 0.5f;
    float2 base = floor(uv);
    float2 w1 = uv - base;
    for (uint k = 0u; k < 4u; ++k)
    {
        int2 corner = int2(k & 1u, k >> 1);
        int2 cell = clamp(int2(base) + corner, 0, int(kScreenProbeTexels) - 1);
        float2 wc = select(corner != 0, w1, 1.0f - w1);
        radiance += rays.Load(int3(tile * kScreenProbeTexels + uint2(cell), 0)).xyz * (wc.x * wc.y);
    }
    radiance /= kHalfLightScale;
    return true;
}

// The specular light along the pixel's GGX lobe from a probe's rays: kSpecularSamples directions drawn from the lobe
// (its visible normals), each the probe's light that way; where the lobe leaves the probe's hemisphere, the lobe of
// its harmonics. Rough reflections of the light around (the harmonics, up to band 2, blur everything narrower than
// about a radian).
float3 ProbeSpecular(Texture2D<float4> rays, uint2 tile, ScreenProbe probe, ProbeSh sh, Basis pixel, float3 wo,
                     float3 r, float roughness, float2 u)
{

    float alpha = max(roughness * roughness, 0.02f);
    Basis basis = MakeBasis(UnpackNormal(probe.normal));
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float found = 0.0f;
    for (uint k = 0u; k < kSpecularSamples; ++k)
    {
        float2 uk = frac(u + float2(0.7548776662f, 0.5698402910f) * float(k));
        float3 h = SampleGgxVndf(wo, alpha, uk);
        float3 l = reflect(-wo, h);
        float3 radiance;
        if (l.z > 0.0f && ProbeRadiance(rays, tile, basis, ToWorld(pixel, l), radiance))
        {
            sum += radiance;
            found += 1.0f;
        }
    }
    return found > 0.0f ? sum / found : ShSpecular(sh, r, roughness);
}

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    if (id.x >= f.width || id.y >= f.height)
    {
        return;
    }
    RWTexture2D<float4> irradianceOut = ResourceDescriptorHeap[g.output];
    RWTexture2D<float4> specularOut = ResourceDescriptorHeap[g.input5];
    Texture2D<float> depth = ResourceDescriptorHeap[g.input0];
    Texture2D<float4> normalRoughness = ResourceDescriptorHeap[g.input1];
    float z = depth.Load(int3(id.xy, 0));
    if (!(z < kSkyDepth) || z <= 0.0f)
    {
        irradianceOut[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        specularOut[id.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    float3 view;
    float3 p = PixelPosition(f, id.xy, z, view);
    float4 nr = normalRoughness.Load(int3(id.xy, 0));
    float3 n = dot(nr.xyz, nr.xyz) > 0.0f ? normalize(nr.xyz) : -view;
    float3 r = reflect(view, n);
    float footprint = PixelFootprint(f, z);
    float3 plane = DepthNormal(f, depth, id.xy, p, z, view, n);
    StructuredBuffer<ScreenProbe> probes = ResourceDescriptorHeap[g.input3];
    StructuredBuffer<ProbeSh> harmonics = ResourceDescriptorHeap[g.input4];
    Texture2D<float4> rays = ResourceDescriptorHeap[g.input2];
    Basis pixelBasis = MakeBasis(n);
    float3 wo = ToLocal(pixelBasis, -view);
    wo.z = max(wo.z, 1e-4f);
    uint seed = Pcg(id.x + id.y * f.width ^ Pcg(f.frameIndex));
    float2 u = Rand2(seed);
    float2 tile = (float2(id.xy) + 0.5f) / float(f.probeTile) - 0.5f;
    int2 base = int2(floor(tile));
    float2 w1 = tile - float2(base);
    float3 irradiance = float3(0.0f, 0.0f, 0.0f);
    float3 specular = float3(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
    for (uint k = 0u; k < 4u; ++k)
    {
        int2 corner = int2(k & 1u, k >> 1);
        int2 q = clamp(base + corner, int2(0, 0), int2(f.probesX - 1u, f.probesY - 1u));
        uint index = uint(q.y) * f.probesX + uint(q.x);
        float2 wc = select(corner != 0, w1, 1.0f - w1);
        ScreenProbe probe = probes[index];
        float w = wc.x * wc.y * FitWeight(p, n, plane, footprint, probe);
        if (w <= 1e-4f)
        {
            continue;
        }
        ProbeSh sh = harmonics[index];
        irradiance += ProbeIrradiance(sh, n, UnpackNormal(probe.normal)) * w;
        specular += ProbeSpecular(rays, uint2(q), probe, sh, pixelBasis, wo, r, nr.w, u) * w;
        weight += w;
    }
    float fitted = 1.0f;
    if (weight <= 1e-3f && g.input7 != kInvalidIndex)
    {
        // The adaptive probes placed where the tiles' probes fit nothing (ProbeAdapt.hlsl), of this tile and the 8
        // around, by fit and distance in tiles.
        StructuredBuffer<uint4> headers = ResourceDescriptorHeap[g.input7];
        int2 own = int2(id.xy / f.probeTile);
        irradiance = float3(0.0f, 0.0f, 0.0f);
        specular = float3(0.0f, 0.0f, 0.0f);
        weight = 0.0f;
        for (int k = 0; k < 9; ++k)
        {
            uint4 header = AdaptiveProbes(headers, f, own + int2(k % 3 - 1, k / 3 - 1));
            for (uint j = 0u; j < 4u; ++j)
            {
                if (header[j] == 0u)
                {
                    continue;
                }
                uint index = header[j] - 1u;
                ScreenProbe probe = probes[index];
                float2 at = float2(probe.pixel & 0xFFFFu, probe.pixel >> 16);
                float w = FitWeight(p, n, plane, footprint, probe) / (1.0f + length(at - float2(id.xy)) / float(f.probeTile));
                if (w <= 1e-3f)
                {
                    continue;
                }
                ProbeSh sh = harmonics[index];
                irradiance += ProbeIrradiance(sh, n, UnpackNormal(probe.normal)) * w;
                specular += ProbeSpecular(rays, uint2(index % f.probesX, index / f.probesX), probe, sh, pixelBasis, wo,
                                          r, nr.w, u) * w;
                weight += w;
            }
        }
    }
    if (weight <= 1e-3f)
    {
        // The probes around do not see this pixel's surface: the best fits nearby (a: 0.75).
        fitted = 0.75f;
        irradiance = float3(0.0f, 0.0f, 0.0f);
        specular = float3(0.0f, 0.0f, 0.0f);
        weight = 0.0f;
        for (int y = -1; y <= 2; ++y)
        {
            for (int x = -1; x <= 2; ++x)
            {
                int2 q = base + int2(x, y);
                if (any(q < 0) || q.x >= int(f.probesX) || q.y >= int(f.probesY))
                {
                    continue;
                }
                uint index = uint(q.y) * f.probesX + uint(q.x);
                ScreenProbe probe = probes[index];
                float w = FitWeight(p, n, plane, footprint, probe);
                if (w <= 1e-3f)
                {
                    continue;
                }
                ProbeSh sh = harmonics[index];
                irradiance += ProbeIrradiance(sh, n, UnpackNormal(probe.normal)) * w;
                specular += ShSpecular(sh, r, nr.w) * w;
                weight += w;
            }
        }
    }
    if (weight <= 1e-3f)
    {
        fitted = 0.0f;
        float3 cache;
        uint first = FindCascade(f, p, kCascadeMargin);
        float voxel = first < f.cascadeCount ? f.cascades[first].voxelSize : 0.1f;
        if (CacheIrradiance(f, p + n * voxel, n, cache))
        {
            irradiance = cache;
            specular = cache * kInvPi;
        }
        weight = 1.0f;
    }
    specular /= weight;
    // Smooth surfaces: their own reflection rays (ReflectionResolve.hlsl), blended in from kMaxRoughnessToTrace down.
    float traced = ReflectionWeight(nr.w);
    if (traced > 0.0f && g.input6 != kInvalidIndex)
    {
        Texture2D<float4> reflections = ResourceDescriptorHeap[g.input6];
        float4 reflected = reflections.Load(int3(id.xy, 0));
        if (reflected.w > 0.0f)
        {
            specular = lerp(specular, reflected.xyz / kHalfLightScale, traced);
        }
    }
    irradianceOut[id.xy] = float4(irradiance / weight * kHalfLightScale, fitted);
    specularOut[id.xy] = float4(specular * kHalfLightScale, fitted);
}
