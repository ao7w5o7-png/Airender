// The integrator's scene hooks (Integrator.hlsli) over the bindless scene buffers, and inline shadow
// rays: shared by the trace kernel and the ReSTIR passes. The including shader declares `g`, its push
// constants, with the fields tlas, instances, materials, lights, instanceLights, lightTree, bsdfLut and
// skyTable, and supplies TraceClosest itself (closest-hit shaders or an inline query). The clouds'
// and the aerial perspective's resources come with their CloudData and SkyData, the volumes' with
// their table (IntegratorParams::volumeTable).
#ifndef AIR_PT_SCENE_ACCESS_HLSLI
#define AIR_PT_SCENE_ACCESS_HLSLI

// Static samplers of the bindless root signature.
SamplerState gLinearClamp : register(s0);
SamplerState gLinearWrap : register(s2);

float BsdfLutLoad(uint index)
{
    StructuredBuffer<float> lut = ResourceDescriptorHeap[g.bsdfLut];
    return lut[index];
}

InstanceData LoadInstance(uint instance)
{
    StructuredBuffer<InstanceData> instances = ResourceDescriptorHeap[g.instances];
    return instances[instance];
}

uint LoadInstanceLight(uint instance)
{
    StructuredBuffer<uint> instanceLights = ResourceDescriptorHeap[g.instanceLights];
    return instanceLights[instance];
}

uint LoadIndex(InstanceData instance, uint index)
{
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(instance.indexBuffer)];
    return indices[index];
}

Vertex LoadVertex(InstanceData instance, uint index)
{
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(instance.vertexBuffer)];
    return vertices[index];
}

MaterialData LoadMaterial(uint material)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[g.materials];
    return materials[material];
}

TextureSetData LoadTextureSet(uint table, uint set)
{
    StructuredBuffer<TextureSetData> sets = ResourceDescriptorHeap[table];
    return sets[set];
}

float4 MaterialTextureSample(uint texture, float2 uv)
{
    Texture2D<float4> t = ResourceDescriptorHeap[NonUniformResourceIndex(texture)];
    return t.SampleLevel(gLinearWrap, uv, 0.0f);
}

// The kernels with the neural parts (AIR_PT_NEURAL: PathTracerNeural.hlsl, PathTracerGiNeural.hlsl,
// ReSTIR*Neural.hlsl) evaluate the clear coats (Bsdf.hlsli AIR_BSDF_COATS: their network, random walk
// or approximation) and decode neural texture sets on sample; the others see coated materials as
// their bases and read the sets' coarse copies.
#ifndef AIR_PT_NEURAL
#define AIR_PT_NEURAL 0
#endif
#define AIR_BSDF_COATS AIR_PT_NEURAL

// The neural coat's parameters (FrameData::neuralCoat, a constant buffer: every thread reads the
// same weights), set by the kernel.
static uint gNeuralCoat;

struct NeuralCoatWeightTable
{
    float4 w[kNeuralCoatParameterVectors];
};

float4 NeuralCoatWeights4(uint index)
{
    ConstantBuffer<NeuralCoatWeightTable> table = ResourceDescriptorHeap[gNeuralCoat];
    return table.w[index / 4u];
}

// Neural texture sets decoded on sample (TextureSetData::latents, NeuralTextures.h) with LibNTC's int8
// inference, where configuring found its shaders (AIR_WITH_NTC).
#if AIR_PT_NEURAL && __has_include("libntc/shaders/Inference.hlsli")
#define AIR_PT_NTC 1
#include "libntc/shaders/Inference.hlsli"

// A neural set's colour (linear) and surface at uv: the texel that stochastic texture filtering picks
// with u (uniform in [0, 1)^2), whose expectation is the bilinear filter's result (Pharr et al. 2024),
// decoded by the set's MLP.
void NeuralTextureSample(TextureSetData t, float2 uv, float2 u, out float4 color, out float4 surface)
{
    ByteAddressBuffer data = ResourceDescriptorHeap[NonUniformResourceIndex(t.neural)];
    Texture2DArray<float4> latents = ResourceDescriptorHeap[NonUniformResourceIndex(t.latents)];
    NtcTextureSetConstants c = data.Load<NtcTextureSetConstants>(0);
    int2 size = int2(c.imageWidth, c.imageHeight);
    int2 texel = int2(floor(uv * float2(size) - 0.5f + u));
    texel = ((texel % size) + size) % size;
    float outputs[NTC_MLP_OUTPUT_CHANNELS];
    [unroll]
    for (int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; ++i)
    {
        outputs[i] = 0.0f;
    }
    NtcSampleTextureSet(c, latents, gLinearWrap, data, kNeuralTextureConstantBytes, texel, 0, true, outputs);
    color = saturate(float4(outputs[0], outputs[1], outputs[2], 1.0f));
    surface = saturate(float4(outputs[3], outputs[4], outputs[5], outputs[6]));
}
#endif

LightTriangle LoadLight(uint light)
{
    StructuredBuffer<LightTriangle> lights = ResourceDescriptorHeap[g.lights];
    return lights[light];
}

float3 SkyTableLoad(uint texel)
{
    StructuredBuffer<float> table = ResourceDescriptorHeap[g.skyTable];
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

float CloudBaseSample(CloudData c, float3 uvw)
{
    Texture3D<float> base = ResourceDescriptorHeap[c.baseTexture];
    return base.SampleLevel(gLinearWrap, uvw, 0.0f);
}

float CloudDetailSample(CloudData c, float3 uvw)
{
    Texture3D<float> detail = ResourceDescriptorHeap[c.detailTexture];
    return detail.SampleLevel(gLinearWrap, uvw, 0.0f);
}

float CloudMajorantLoad(CloudData c, uint cell)
{
    StructuredBuffer<float> majorants = ResourceDescriptorHeap[c.majorants];
    return majorants[cell];
}

float CloudShadowSample(CloudData c, float2 uv)
{
    Texture2D<float> map = ResourceDescriptorHeap[c.shadowMap];
    return map.SampleLevel(gLinearWrap, uv, 0.0f);
}

float4 CloudSkySample(CloudData c, float2 uv)
{
    Texture2D<float4> map = ResourceDescriptorHeap[c.skyMap];
    return map.SampleLevel(gLinearClamp, uv, 0.0f);
}

VolumeData LoadVolume(uint table, uint index)
{
    StructuredBuffer<VolumeData> volumes = ResourceDescriptorHeap[table];
    return volumes[index];
}

float VolumeDensitySample(VolumeData v, float3 uvw)
{
    Texture3D<float> density = ResourceDescriptorHeap[v.density];
    return density.SampleLevel(gLinearClamp, uvw, 0.0f);
}

float VolumeMajorantLoad(VolumeData v, uint cell)
{
    StructuredBuffer<float> majorants = ResourceDescriptorHeap[v.majorants];
    return majorants[cell];
}

LightTreeNode LoadLightNode(uint node)
{
    StructuredBuffer<LightTreeNode> tree = ResourceDescriptorHeap[g.lightTree];
    return tree[node];
}

// The materials' textures (FrameData::textureSets) for the alpha test of cutouts, set by the kernel.
static uint gTextureSets;

#include "Cutout.hlsli"

// Whether a ray query's candidate is solid: the instances of cutout materials are non-opaque
// (Cutout.hlsli), so their triangles come as candidates the query commits unless the alpha test cuts
// them out; every other instance is forced opaque and never yields one. Every query over the scene
// runs `while (query.Proceed()) { if (CandidateIsSolid(...)) query.CommitNonOpaqueTriangleHit(); }`.
bool CandidateIsSolid(uint candidateType, uint instance, uint primitive, float2 barycentrics)
{
    return candidateType == CANDIDATE_NON_OPAQUE_TRIANGLE &&
           !AlphaCutsOut(gTextureSets, instance, primitive, barycentrics);
}

bool TraceVisible(float3 origin, float3 direction, float tMax)
{
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    while (query.Proceed())
    {
        if (CandidateIsSolid(query.CandidateType(), query.CandidateInstanceID(), query.CandidatePrimitiveIndex(),
                             query.CandidateTriangleBarycentrics()))
        {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return query.CommittedStatus() == COMMITTED_NOTHING;
}

#endif
