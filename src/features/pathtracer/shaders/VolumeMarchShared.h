// The real-time volume renderer's data shared by C++ and HLSL (docs/architecture.md §8.13; see
// air/renderer/shader/HlslCompat.h): what the trace kernel and the ray march read to light through
// volumes (moment shadow maps, VolumeShadowsData in FrameData), the probes of the light around each
// volume, and the push constants of its passes.
#ifndef AIR_PT_VOLUME_MARCH_SHARED_H
#define AIR_PT_VOLUME_MARCH_SHARED_H

#include "air/renderer/shader/HlslCompat.h"
#include "air/renderer/shader/MomentShadowShared.h"
#include "air/renderer/shader/VolumeBricksShared.h"

#ifdef __cplusplus
namespace air::pt::gpu
{
using namespace air::hlsl;
using air::gpu::MomentShadowMap;
using air::gpu::VolumeBrickLevel;
using air::gpu::VolumeMarchData;
#endif

AIR_CONST uint kMaxVolumeLights = 8;     // light groups with moment shadow maps (besides the sun)
AIR_CONST uint kMaxLightOrigins = 4;     // maps of a large emitter close to the volumes, over its extent
AIR_CONST uint kExactGroupTriangles = 8; // groups of up to this many triangles light volumes exactly
AIR_CONST uint kVolumeProbeAxis = 8;     // probes per volume and axis, spanning its box
AIR_CONST uint kVolumeProbes = 512;      // kVolumeProbeAxis^3
AIR_CONST uint kVolumeProbeRays = 16;    // traced per probe and frame
AIR_CONST uint kProbesPerGroup = 4;      // in a thread group of 64, a thread per ray (kVolumeProbes / 4 groups)
// The probes live in one Texture3D<float4> (RGBA32F) that the march samples with hardware trilinear
// filtering: volume v's probe (i, j, k) at texel (n v + i, j, n b + k) of block b, n = kVolumeProbeAxis,
// the blocks holding L1 spherical harmonics of the red, green and blue radiance arriving at the probe
// and (in x) the volume's occlusion there. Lookups stay inside one volume's block (VolumeProbes.hlsli).
AIR_CONST uint kProbeBlocks = 4;
AIR_CONST uint kMarchTile = 8;        // the march's thread groups cover 8 x 8 pixels
AIR_CONST uint kMarchMaxVolumes = 4;  // volumes a ray marches through at once (its segments' masks)
AIR_CONST uint kMarchMaxSegments = 8; // 2 kMarchMaxVolumes boundaries

// VolumeMarchConstants::flags
AIR_CONST uint kMarchReset = 1u;       // discard the accumulation (as kFlagReset)
AIR_CONST uint kMarchAdaptive = 2u;    // adaptive sampling groups: lighting shared by 2x2 / 4x4 pixels
AIR_CONST uint kMarchAir = 4u;         // the aerial perspective dims the volumes (fused at their depth)
AIR_CONST uint kMarchNoDirect = 8u;    // leave the sun's and the lights' light out (indirect view)
AIR_CONST uint kMarchNoIndirect = 16u; // leave the probes' light out (direct view)
AIR_CONST uint kMarchFurnace = 32u;    // the white furnace (kFlagFurnace): albedo 1, the light groups off

// How the trace kernel and the march read the volumes' shadows.
struct VolumeShadowsData
{
    uint maps;        // StructuredBuffer<MomentShadowMap>
    uint b0;          // Texture2DArray<float>, a slice per map
    uint moments;     // Texture2DArray<float4>
    uint sunMap;      // the sun's entry in maps; kInvalidIndex: none
    uint groupMaps;   // StructuredBuffer<uint>: per light group its map, kInvalidIndex: none (no shadow)
    uint lightGroups; // StructuredBuffer<LightGroup>
    uint mapCount;
    uint occluders; // Texture2DArray<float>: the depth of each texel ray's first opaque hit
};

// Root constants of VolumeMarch.Shadows: one thread per texel of every map.
struct VolumeShadowConstants
{
    uint maps;    // StructuredBuffer<MomentShadowMap>
    uint volumes; // StructuredBuffer<VolumeMarchData>
    uint levels;  // StructuredBuffer<VolumeBrickLevel>
    uint volumeCount;
    uint b0;         // RWTexture2DArray<float>
    uint moments;    // RWTexture2DArray<float4>
    uint size;       // texels per side
    uint level;      // the bricks' level the march reads
    float stepScale; // its step, in voxels of that level
    uint tlas;       // for the opaque slice
    uint occluders;  // RWTexture2DArray<float>
    uint pad0;
};

// Root constants of VolumeMarch.Probes: a thread group per probe, a thread per ray.
struct VolumeProbeConstants
{
    uint frameData; // StructuredBuffer<FrameData>, for the integrator
    uint frameSlot;
    uint tlas;
    uint instances;
    uint materials;
    uint lights;
    uint instanceLights;
    uint lightTree;
    uint bsdfLut;
    uint skyTable;
    uint volumes; // StructuredBuffer<VolumeMarchData>
    uint probes;  // RWTexture3D<float4>: the probe field (kProbeBlocks)
    uint frame;   // seeds the rays
    float blend;  // weight of this frame's estimate
    uint pad0;
    uint pad1;
};

// Root constants of VolumeMarch.March.
struct VolumeMarchConstants
{
    uint frameData; // StructuredBuffer<FrameData>
    uint frameSlot;
    uint surface;      // Texture2D<float4>: this frame's samples (rgb sum) and their first hit's distance (a)
    uint accumulation; // RWTexture2D<float4>: rgb sum, a = sample count
    uint volumes;      // StructuredBuffer<VolumeMarchData>
    uint levels;       // StructuredBuffer<VolumeBrickLevel>
    uint probes;       // Texture3D<float4>: the probe field (kProbeBlocks), probeVolumes volumes wide
    uint blueNoise;    // StructuredBuffer<uint>: 64 x 64 ranks
    uint lights;       // StructuredBuffer<LightTriangle>
    uint marchLights;  // StructuredBuffer<uint>: the light groups that light the volumes (each with a map)
    uint marchLightCount;
    uint level;        // the bricks' level the march reads
    float stepScale;   // its step, in voxels of that level
    uint flags;        // kMarch*
    float shareBelow2; // adaptive sampling: a wave whose transmittance is below this shares lighting over 2 x 2
    float shareBelow4; // ... over 4 x 4
    uint view;         // the path tracer's debug view (kDebugVolume*: the march's own)
    float viewScale;
    uint samplesPerPixel; // of the frame's trace, which the march's light is counted for
    uint probeVolumes;
};

#ifdef __cplusplus
static_assert(sizeof(VolumeShadowsData) == 32);
static_assert(kVolumeProbes % kProbesPerGroup == 0 && kProbesPerGroup * kVolumeProbeRays == 64);
static_assert(sizeof(VolumeShadowConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(VolumeProbeConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(VolumeMarchConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::pt::gpu
#endif

#endif
