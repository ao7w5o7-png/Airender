// Path tracer data shared by C++ and HLSL (see air/renderer/shader/HlslCompat.h).
#ifndef AIR_PATH_TRACER_SHARED_H
#define AIR_PATH_TRACER_SHARED_H

#include "NrcShared.h"
#include "VolumeMarchShared.h"
#include "air/renderer/shader/CloudShared.h"
#include "air/renderer/shader/HlslCompat.h"
#include "air/renderer/shader/SkyShared.h"
#include "air/renderer/shader/VolumeShared.h"

#ifdef __cplusplus
namespace air::pt::gpu
{
using namespace air::hlsl;
using air::gpu::CloudData;
using air::gpu::SkyData;
using air::gpu::VolumeData;
#endif

// Debug views (pathtracer.debugView). Everything but kDebugNone replaces the radiance.
AIR_CONST uint kDebugNone = 0;
AIR_CONST uint kDebugNormals = 1;   // first-hit shading normal, n * 0.5 + 0.5
AIR_CONST uint kDebugBaseColor = 2; // first-hit base color
AIR_CONST uint kDebugRoughness = 3;
AIR_CONST uint kDebugMetallic = 4;
AIR_CONST uint kDebugDepth = 5;         // first-hit distance / debugScale
AIR_CONST uint kDebugIntersections = 6; // heatmap: triangles intersected by the camera ray / debugScale
AIR_CONST uint kDebugBounces = 7;       // heatmap: path length / maxBounces (Russian roulette shows here)
AIR_CONST uint kDebugDirect = 8;        // radiance of paths with at most one bounce (emission + direct light)
AIR_CONST uint kDebugIndirect = 9;      // radiance of the remaining paths: the global illumination
AIR_CONST uint kDebugNrcCache = 10;     // the neural radiance cache's prediction at the first hit (pathtracer.nrc)
// The real-time volumes' ray march (VolumeMarch.hlsl, pathtracer.volumeMarch), over the surfaces:
AIR_CONST uint kDebugVolumeSteps = 11;    // heatmap: lit steps / debugScale
AIR_CONST uint kDebugVolumeRate = 12;     // lighting rate: blue every pixel, green shared by 2 x 2, red by 4 x 4
AIR_CONST uint kDebugVolumeSegments = 13; // heatmap: volumes the camera ray crosses (of kMarchMaxVolumes)
// Where the clear coats' network shades the first surface (blue) and how much it corrects the analytic
// approximation towards the mirror direction (green brighter, red darker; full at 2x / 0.5x); other
// coat models purple, the rest grey.
AIR_CONST uint kDebugNeuralCoat = 14;
AIR_CONST uint kDebugViewCount = 15;

AIR_CONST uint kFlagFurnace = 1u;      // every material white, emitters off: an energy-conserving image is uniform
AIR_CONST uint kFlagReset = 2u;        // discard the accumulated samples
AIR_CONST uint kFlagGlobalJitter = 4u; // every sample uses FrameData.jitter (for temporal denoisers)
AIR_CONST uint kFlagGuides = 8u;       // write the denoiser guide textures
AIR_CONST uint kFlagNoNee = 16u;       // BSDF sampling only (reference for the MIS tests)
// The direct light at the first surface comes from ReSTIR DI (ReSTIRDI.hlsli): no NEE there, no
// emission found by its BSDF sample; the first sample's hit goes to TraceConstants::primaryHits.
AIR_CONST uint kFlagReSTIR = 32u;
// The light of paths with two or more bounces comes from ReSTIR GI (ReSTIRGI.hlsli); the first
// sample's candidate goes to TraceConstants::giReservoirs.
AIR_CONST uint kFlagReSTIRGI = 64u;
AIR_CONST uint kFlagRecordGi = 128u; // set by the kernel on the sample that records the GI candidate
// Paths end in the neural radiance cache (NrcShared.h) where their footprint has spread enough.
AIR_CONST uint kFlagNrc = 256u;
AIR_CONST uint kFlagNrcTrain = 512u;     // set by the kernel on its pixel's training path
AIR_CONST uint kFlagNrcUnbiased = 1024u; // that training path is never ended by the cache
// The volumes are rendered by the ray march (VolumeMarch.hlsli): the trace kernel's paths light the
// surfaces through their moment shadow maps, and the kernel writes this frame's samples (rgb) and
// their first hit's distance (a) instead of adding to the accumulation.
AIR_CONST uint kFlagVolumeMarch = 2048u;
AIR_CONST uint kFlagProbe = 4096u; // a probe's path for the march's ambient light (VolumeProbes.hlsl)
// Write the first surface's reflected indirect light split by lobe for a denoiser
// (TraceConstants::diffuseSignal, specularSignal; ReSTIR GI adds its own). Its direct light passes the
// denoiser: filtered by it, the shadows spread and the frame lost 2-10% of its light.
AIR_CONST uint kFlagSignals = 8192u;
// The clouds come from the real-time clouds' maps (CloudMaps.hlsli, CloudMarcher.h): the camera's
// rays from FrameData::cloudLayer, other rays that leave the scene from CloudData::skyMap, the sun's
// light through them from CloudData::shadowMap. No walks through them.
AIR_CONST uint kFlagCloudMaps = 16384u;
AIR_CONST float kMissReflectionDistance = 1000.0f; // a reflected ray that leaves the scene, metres

// The neural coat's parameters as float4s (NeuralCoat.hlsli kNeuralCoatParameters / 4), the size of its
// constant buffer before padding to 256 bytes (FrameData::neuralCoat).
AIR_CONST uint kNeuralCoatParameterVectors = 245u;

// Result of TraceClosest() in the integrator (not stored in buffers).
struct HitInfo
{
    bool hit;
    float t;
    uint instance;  // InstanceID()
    uint primitive; // PrimitiveIndex()
    float2 barycentrics;
};

struct CameraData
{
    float4 position; // xyz, w = tan(verticalFov / 2)
    float4 forward;  // xyz, w = aspect (width / height)
    float4 right;
    float4 up;
};

// Everything a frame needs that does not fit in push constants. One element per frame in an
// upload ring; TraceConstants::frameSlot selects it.
struct FrameData
{
    CameraData camera;
    CameraData previousCamera; // for motion vectors
    SkyData sky;               // sky radiance (a table or uniform) and the sun; uniform in the furnace
    CloudData clouds;          // the cloud layer and its resources; enabled == 0: none
    float2 jitter;             // sub-pixel offset in [-0.5, 0.5) for kFlagGlobalJitter
    uint frameIndex;
    uint flags;
    uint width;
    uint height;
    uint samplesPerPixel;
    uint maxBounces;
    uint lightCount;
    uint debugView;
    float debugScale;                // normalizes depth / intersection debug views
    uint lightTreeBounces;           // NEE uses the light tree at path vertices below this, power beyond
    float exposure;                  // radiance -> scene-linear, applied by Resolve
    uint cloudEvents;                // scattering events a walk through the clouds may take
    uint volumeEvents;               // scattering events a path may take in the volumes, besides its bounces
    uint coatModel;                  // kCoat*: how coated materials evaluate their coat (IntegratorParams::coatModel)
    NrcData nrc;                     // the neural radiance cache, with kFlagNrc
    uint volumeCount;                // the scene's volumes (GpuScene::volumes)
    uint volumeTable;                // StructuredBuffer<VolumeData> (GpuScene::volumeTable)
    uint textureSets;                // StructuredBuffer<TextureSetData> (GpuScene::textureSetTable)
    uint neuralCoat;                 // ByteAddressBuffer: the neural coat's parameters (NeuralCoat.hlsli)
    VolumeShadowsData volumeShadows; // their moment shadow maps, with kFlagVolumeMarch
    // With kFlagCloudMaps: the clouds over the camera's rays at half the render size (Texture2D<float4>:
    // rgb what they add, nits, a what they leave of what lies behind) and the distance to their front
    // (Texture2D<float>), CloudMarcher's reconstruction.
    uint cloudLayer;
    uint cloudDepth;
    uint pad0;
    uint pad1;
};

// Root constants of PathTracer.Trace: bindless indices only.
struct TraceConstants
{
    uint frameData; // StructuredBuffer<FrameData>
    uint frameSlot;
    uint tlas;
    uint instances;       // StructuredBuffer<InstanceData>
    uint materials;       // StructuredBuffer<MaterialData>
    uint lights;          // StructuredBuffer<LightTriangle>
    uint bsdfLut;         // StructuredBuffer<float>
    uint accumulation;    // RWTexture2D<float4>: rgb = sum, a = sample count
    uint diffuseAlbedo;   // RWTexture2D<float4>, written with kFlagGuides
    uint specularAlbedo;  // RWTexture2D<float4>
    uint normalRoughness; // RWTexture2D<float4>
    uint viewDepth;       // RWTexture2D<float>
    uint motionVectors;   // RWTexture2D<float2>
    uint instanceLights;  // StructuredBuffer<uint>: first light-list entry per instance
    uint lightTree;       // StructuredBuffer<LightTreeNode>
    uint primaryHits;     // RWStructuredBuffer<PrimaryHit>, written with kFlagReSTIR or kFlagReSTIRGI
    uint giReservoirs;    // RWStructuredBuffer<GiReservoir>, written with kFlagReSTIRGI
    uint skyTable;        // StructuredBuffer<float>: the sky radiance table (SkyShared.h)
    // With kFlagSignals (RWTexture2D<float4>): the first surface's reflected indirect light per sample
    // through its diffuse base and its specular lobes, a = the reflected ray's length.
    uint diffuseSignal;
    uint specularSignal;
};

struct ResolveConstants
{
    uint frameData;
    uint frameSlot;
    uint accumulation; // Texture2D<float4>
    uint output;       // RWTexture2D<float4>
    // Under real-time volumes, for a denoiser (kInvalidIndex: none): the frame's surfaces alone
    // (Texture2D<float4>, the march's input) resolved into `before` (RWTexture2D<float4>, like output).
    uint surface;
    uint before;
};

#ifdef __cplusplus
static_assert(sizeof(CameraData) == 64);
static_assert(sizeof(FrameData) == 640);
static_assert(sizeof(TraceConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(ResolveConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::pt::gpu
#endif

#endif
