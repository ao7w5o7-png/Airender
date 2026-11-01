// Probe GI: helpers every pass uses (camera rays, packing, random numbers, the sky). HLSL only.
#ifndef AIR_PROBE_GI_COMMON_HLSLI
#define AIR_PROBE_GI_COMMON_HLSLI

#include "ProbeGiShared.h"
#include "air/renderer/shader/SceneShared.h"

SamplerState gLinearClamp : register(s0);
SamplerState gPointClamp : register(s1);
SamplerState gLinearWrap : register(s2);

static const float kPi = 3.14159265358979f;
static const float kInvPi = 0.318309886183791f;

// --- Frame -----------------------------------------------------------------------------------------

// The frame's constants, in the frame's upload memory (RenderGraph::AllocateUpload).
GiFrame LoadFrame(uint frames, uint offset)
{
    ByteAddressBuffer buffer = ResourceDescriptorHeap[frames];
    return buffer.Load<GiFrame>(offset);
}

// --- Random numbers ------------------------------------------------------------------------------

// PCG hash (Jarzynski & Olano 2020).
uint Pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Rand(inout uint seed)
{
    seed = Pcg(seed);
    return float(seed >> 8) * (1.0f / 16777216.0f);
}

float2 Rand2(inout uint seed)
{
    float x = Rand(seed);
    float y = Rand(seed);
    return float2(x, y);
}

// Additive recurrence with the plastic constant (R2): well spread offsets for frame n.
float2 R2(uint n)
{
    return frac(float2(0.7548776662f, 0.5698402910f) * float(n) + 0.5f);
}

float Luminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

float MaxComponent(float3 c)
{
    return max(c.x, max(c.y, c.z));
}

// --- Frames and sampling --------------------------------------------------------------------------

struct Basis
{
    float3 t;
    float3 b;
    float3 n;
};

// Orthonormal basis around n (Duff et al. 2017).
Basis MakeBasis(float3 n)
{
    float s = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + n.z);
    float b = n.x * n.y * a;
    Basis f;
    f.t = float3(1.0f + s * n.x * n.x * a, s * b, -s * n.x);
    f.b = float3(b, s + n.y * n.y * a, -n.y);
    f.n = n;
    return f;
}

float3 ToWorld(Basis f, float3 v)
{
    return f.t * v.x + f.b * v.y + f.n * v.z;
}

float3 ToLocal(Basis f, float3 v)
{
    return float3(dot(v, f.t), dot(v, f.b), dot(v, f.n));
}

// Cosine-weighted hemisphere around +z from the unit square by the concentric map (Shirley and Chiu
// 1997): neighbouring cells of a stratified grid give neighbouring directions. pdf = z / pi.
float3 SampleCosineConcentric(float2 u)
{
    float2 o = u * 2.0f - 1.0f;
    float r;
    float phi;
    if (o.x == 0.0f && o.y == 0.0f)
    {
        return float3(0.0f, 0.0f, 1.0f);
    }
    if (abs(o.x) > abs(o.y))
    {
        r = o.x;
        phi = (kPi / 4.0f) * (o.y / o.x);
    }
    else
    {
        r = o.y;
        phi = (kPi / 2.0f) - (kPi / 4.0f) * (o.x / o.y);
    }
    float2 d = r * float2(cos(phi), sin(phi));
    return float3(d.x, d.y, sqrt(max(0.0f, 1.0f - dot(d, d))));
}

// --- Packing ---------------------------------------------------------------------------------------

float2 OctWrap(float2 v)
{
    return (1.0f - abs(v.yx)) * select(v.xy >= 0.0f, 1.0f, -1.0f);
}

// Unit vector <-> [-1, 1]^2 (octahedral map).
float2 OctEncode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.z >= 0.0f ? n.xy : OctWrap(n.xy);
    return p;
}

float3 OctDecode(float2 p)
{
    float3 n = float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0f, -t, t);
    return normalize(n);
}

uint PackNormal(float3 n)
{
    float2 p = OctEncode(n) * 0.5f + 0.5f;
    uint2 q = uint2(saturate(p) * 65535.0f + 0.5f);
    return q.x | (q.y << 16);
}

float3 UnpackNormal(uint v)
{
    float2 p = float2(v & 0xFFFFu, v >> 16) / 65535.0f;
    return OctDecode(p * 2.0f - 1.0f);
}

// A surface voxel face's normal (12:12 octahedral) and its surface's offset from the voxel's centre along it
// (-1..1 voxels, 8 bits).
uint PackNormalOffset(float3 n, float offset)
{
    float2 p = OctEncode(n) * 0.5f + 0.5f;
    uint2 q = uint2(saturate(p) * 4095.0f + 0.5f);
    uint o = uint(saturate(offset * 0.5f + 0.5f) * 255.0f + 0.5f);
    return q.x | (q.y << 12) | (o << 24);
}

float3 UnpackNormalOffset(uint v, out float offset)
{
    float2 p = float2(v & 0xFFFu, (v >> 12) & 0xFFFu) / 4095.0f;
    offset = float(v >> 24) / 255.0f * 2.0f - 1.0f;
    return OctDecode(p * 2.0f - 1.0f);
}

uint PackUnorm4(float4 v)
{
    uint4 q = uint4(saturate(v) * 255.0f + 0.5f);
    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
}

float4 UnpackUnorm4(uint v)
{
    return float4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, v >> 24) / 255.0f;
}

// Radiance (nits) as RGB9E5 of nits / 256: 1.5e-5 .. 1.7e7 nits with 9-bit mantissas (single estimates of
// next event estimation near bright lights reach millions of nits before they are averaged).
static const float kRadiancePackScale = 1.0f / 256.0f;

// `dither` in [0, 1) rounds the mantissas: 0.5 to the nearest, a uniform random number stochastically (the stored
// value is then right on average, which a value blended over frames needs: rounded to the nearest, a blend whose
// step is below half a unit of the mantissa never moves it, and it stops short of the mean it converges to).
uint PackRadiance(float3 rgb, float dither)
{
    float3 c = clamp(rgb * kRadiancePackScale, 0.0f, 65408.0f);
    float maxc = max(c.x, max(c.y, c.z));
    if (!(maxc > 1.0e-9f))
    {
        return 0u;
    }
    int e = max(-16, int(floor(log2(maxc)))) + 16; // biased by 15, plus 1
    float scale = exp2(float(e - 24));
    uint maxm = uint(floor(maxc / scale + dither));
    if (maxm >= 512u)
    {
        scale *= 2.0f;
        e += 1;
    }
    uint3 m = min(uint3(floor(c / scale + dither)), 511u);
    return m.x | (m.y << 9) | (m.z << 18) | (uint(clamp(e, 0, 31)) << 27);
}

// `v` rounded to a half float's precision by `dither` in [0, 1), as PackRadiance rounds (stochastically: a history
// blended at a few percent and stored in half floats otherwise stops up to 0.8% short of its mean); the store then
// converts it exactly.
float3 RoundToHalf(float3 v, float dither)
{
    float3 a = abs(v);
    float3 ulp = exp2(floor(log2(max(a, 6.103515625e-5f))) - 10.0f); // below the smallest normal: subnormal steps
    return sign(v) * floor(a / ulp + dither) * ulp;
}

uint PackRadiance(float3 rgb)
{
    return PackRadiance(rgb, 0.5f);
}

float3 UnpackRadiance(uint v)
{
    float scale = exp2(float(int(v >> 27) - 24)) / kRadiancePackScale;
    return float3(v & 0x1FFu, (v >> 9) & 0x1FFu, (v >> 18) & 0x1FFu) * scale;
}

// --- Camera ----------------------------------------------------------------------------------------

// Direction through pixel position `pixel` (0 .. size, y down): the path tracer's camera rays.
float3 CameraRay(GiCamera c, float2 pixel, float2 size)
{
    float ndcX = pixel.x / size.x * 2.0f - 1.0f;
    float ndcY = 1.0f - pixel.y / size.y * 2.0f;
    float tanHalf = c.position.w;
    float aspect = c.forward.w;
    return normalize(c.forward.xyz + c.right.xyz * (ndcX * tanHalf * aspect) + c.up.xyz * (ndcY * tanHalf));
}

// Pixel position of a world point, and its view depth in .z (<= 0: behind the camera).
float3 CameraProject(GiCamera c, float3 p, float2 size)
{
    float3 d = p - c.position.xyz;
    float z = dot(d, c.forward.xyz);
    float zc = max(z, 1e-6f);
    float tanHalf = c.position.w;
    float aspect = c.forward.w;
    float ndcX = dot(d, c.right.xyz) / (zc * tanHalf * aspect);
    float ndcY = dot(d, c.up.xyz) / (zc * tanHalf);
    return float3((ndcX + 1.0f) * 0.5f * size.x, (1.0f - ndcY) * 0.5f * size.y, z);
}

// The world position of a pixel from its view depth (slots::kViewDepth), with the frame's jitter.
float3 PixelPosition(GiFrame f, uint2 pixel, float viewDepth, out float3 direction)
{
    float2 size = float2(f.width, f.height);
    direction = CameraRay(f.camera, float2(pixel) + 0.5f + f.jitter, size);
    return f.camera.position.xyz + direction * (viewDepth / max(dot(direction, f.camera.forward.xyz), 1e-6f));
}

// Metres a pixel spans at view depth z.
float PixelFootprint(GiFrame f, float z)
{
    return 2.0f * f.camera.position.w * z / float(f.height);
}

// --- Sky -------------------------------------------------------------------------------------------

static uint gSkyTable;

float3 SkyTableLoad(uint texel)
{
    StructuredBuffer<float> table = ResourceDescriptorHeap[gSkyTable];
    return float3(table[texel * 3u], table[texel * 3u + 1u], table[texel * 3u + 2u]);
}

float4 AerialInscatterSample(SkyData sky, float3 uvw)
{
    Texture3D<float4> table = ResourceDescriptorHeap[sky.aerialInscatter];
    return table.SampleLevel(gLinearClamp, uvw, 0.0f);
}

float4 AerialTransmittanceSample(SkyData sky, float3 uvw)
{
    Texture3D<float4> table = ResourceDescriptorHeap[sky.aerialTransmittance];
    return table.SampleLevel(gLinearClamp, uvw, 0.0f);
}

#include "air/renderer/shader/SkyLookup.hlsli"

// The sky's light along a ray that leaves the scene, without the sun's disc (direct light).
float3 SkyLight(GiFrame f, float3 direction)
{
    gSkyTable = f.skyTable;
    return SkyRadiance(f.sky, direction);
}

#endif
