// The integrator's AIR_PT_VOLUME_SHADOWS hooks on the GPU (Integrator.hlsli): what the volumes let
// through of a light, from their moment shadow maps (VolumeShadowsData, MomentShadow.hlsli). For the
// kernels that light surfaces through volumes the ray march renders (PathTracerVolumeShadows.hlsl,
// VolumeProbes.hlsl), which set gVolumeShadows from their FrameData before tracing. Included after
// SceneAccess.hlsli (its static samplers).
#ifndef AIR_PT_VOLUME_SHADOW_HOOKS_HLSLI
#define AIR_PT_VOLUME_SHADOW_HOOKS_HLSLI

static VolumeShadowsData gVolumeShadows;

float MomentMapB0Sample(MomentShadowMap m, float2 uv)
{
    Texture2DArray<float> b0 = ResourceDescriptorHeap[gVolumeShadows.b0];
    return b0.SampleLevel(gLinearClamp, float3(uv, float(m.slice)), 0.0f);
}

float4 MomentMapBSample(MomentShadowMap m, float2 uv)
{
    Texture2DArray<float4> moments = ResourceDescriptorHeap[gVolumeShadows.moments];
    return moments.SampleLevel(gLinearClamp, float3(uv, float(m.slice)), 0.0f);
}

float MomentMapOccluderLoad(MomentShadowMap m, uint x, uint y)
{
    Texture2DArray<float> occluders = ResourceDescriptorHeap[gVolumeShadows.occluders];
    return occluders.Load(int4(int(x), int(y), int(m.slice), 0));
}

#include "air/renderer/shader/MomentShadow.hlsli"

MomentShadowMap LoadMomentMap(uint index)
{
    StructuredBuffer<MomentShadowMap> maps = ResourceDescriptorHeap[gVolumeShadows.maps];
    return maps[index];
}

uint VolumeGroupMap(uint group)
{
    StructuredBuffer<uint> groupMaps = ResourceDescriptorHeap[gVolumeShadows.groupMaps];
    return groupMaps[group];
}

float VolumeSunShadow(float3 p)
{
    return gVolumeShadows.sunMap == kInvalidIndex ? 1.0f : MomentShadow(LoadMomentMap(gVolumeShadows.sunMap), p);
}

// From the map of the group whose origin is nearest to lightPoint (a large emitter has several).
float VolumeGroupShadow(uint group, float3 p, float3 lightPoint)
{
    uint first = VolumeGroupMap(group);
    if (first == kInvalidIndex)
    {
        return 1.0f;
    }
    MomentShadowMap m = LoadMomentMap(first);
    float nearest = dot(m.origin - lightPoint, m.origin - lightPoint);
    for (uint k = 1u; k < m.count; ++k)
    {
        MomentShadowMap other = LoadMomentMap(first + k);
        float d = dot(other.origin - lightPoint, other.origin - lightPoint);
        if (d < nearest)
        {
            nearest = d;
            m = other;
        }
    }
    return MomentShadow(m, p);
}

bool VolumeExplicitGroup(uint group)
{
    return VolumeGroupMap(group) != kInvalidIndex;
}

#endif
