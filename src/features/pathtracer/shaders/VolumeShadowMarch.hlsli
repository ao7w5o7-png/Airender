// A moment shadow map's texel (MomentShadow.hlsli), shared by the GPU pass that renders the maps
// (VolumeShadows.hlsl) and the CPU twin: its ray from the light through every volume, in steps of a
// fraction of their voxels, adding up the moments of the optical depth it crosses. The steps are not
// jittered (the maps stay put from frame to frame); their midpoints are sampled.
//
// The including code defines VolumeBricks.hlsli's hooks and
//   VolumeMarchData  LoadMarchVolume(uint index), VolumeBrickLevel LoadVolumeLevel(uint index)
#ifndef AIR_PT_VOLUME_SHADOW_MARCH_HLSLI
#define AIR_PT_VOLUME_SHADOW_MARCH_HLSLI

AIR_CONST uint kShadowMaxSteps = 4096u; // per volume and texel

void MarchShadowTexel(MomentShadowMap m, float2 uv, uint volumeCount, uint level, float stepScale, AIR_OUT(float) b0,
                      AIR_OUT(float4) b)
{
    b0 = 0.0f;
    b = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 o;
    float3 d;
    MomentMapRay(m, uv, o, d);
    for (uint v = 0u; v < volumeCount; ++v)
    {
        VolumeMarchData volume = LoadMarchVolume(v);
        float t0;
        float t1;
        if (!BoxSegment(o, d, volume.boundsMin, volume.boundsMax, m.depthNear, m.depthFar, t0, t1))
        {
            continue;
        }
        VolumeBrickLevel l = LoadVolumeLevel(volume.levels + level);
        uint steps = min(uint(ceil((t1 - t0) / (stepScale * l.voxelSize))), kShadowMaxSteps);
        float dt = (t1 - t0) / float(max(steps, 1u));
        for (uint i = 0u; i < steps; ++i)
        {
            float t = t0 + (float(i) + 0.5f) * dt;
            float density = BrickDensity(l, o + d * t);
            if (density > 0.0f)
            {
                AddMoments(b0, b, volume.extinction * density * dt, MomentDepth(m, t));
            }
        }
    }
}

#endif
