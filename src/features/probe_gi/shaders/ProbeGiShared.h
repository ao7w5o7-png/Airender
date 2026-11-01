// Probe GI data shared by C++ and HLSL (see air/renderer/shader/HlslCompat.h). docs/architecture.md §8.21.
//
// The scene as the GI traces it: cascades of voxels around the camera (kVoxelResolution^3 each, the
// voxel doubling from one to the next), stored toroidally (world voxel v at v mod kVoxelResolution, so a
// cascade that moves keeps what it still covers), each with
//   * a distance field: the distance to the nearest triangle, exact up to kVoxelBand voxels (R8_UNORM of
//     distance / kVoxelBand), sampled with trilinear filtering (rays read it only in bricks a surface crosses);
//   * per brick of kVoxelBrick^3 voxels the Chebyshev distance, in bricks, to the nearest brick a surface
//     crosses (R8_UINT), so that rays leap over empty space;
//   * surface voxels (those a triangle crosses), sparse: an index texture (0 or 1 + slot) into a pool of
//     SurfaceVoxel, each with up to six faces: what the voxel shows towards +x, -x, +y, -y, +z and -z (the
//     averaged albedo and normal of the triangles' sides that face that way) and the radiance it reflects
//     that way. Both sides of a thin wall in one voxel keep their own light. Slots freed where a cascade
//     moved on are reused (a ring of free slots).
// A cascade that moves voxelizes only the bricks it did not cover before; the rest keeps its voxels and light.
// Its light: next event estimation of the sun and the emissive triangles with shadow rays through the
// distance field, plus the indirect light from the radiance cache (VoxelLight.hlsl).
// The radiance cache: per cascade a lattice of kCacheResolution^3 probes (one per 8^3 voxels), each an
// octahedral map of kCacheTexels^2 directions of the light arriving there (traced through the voxels) with
// its rays' distances, and that light's irradiance as spherical harmonics.
// Screen probes: one per tile of pixels, kScreenProbeTexels^2 rays over the hemisphere of its surface
// (cosine-weighted), traced on screen first, then through the voxels, then the cache.
//
// Light here is the light reflected towards the GI's rays: never the emission of what they hit (that is
// direct light, the path tracer's) nor the sun's disc.
#ifndef AIR_PROBE_GI_SHARED_H
#define AIR_PROBE_GI_SHARED_H

#include "air/renderer/shader/HlslCompat.h"
#include "air/renderer/shader/SkyShared.h"

#ifdef __cplusplus
#include <cstddef> // offsetof

namespace air::pgi::gpu
{
using namespace air::hlsl;
using air::gpu::SkyData;
#endif

AIR_CONST uint kVoxelResolution = 128u; // voxels per side of a cascade
AIR_CONST uint kVoxelBrick = 4u;        // voxels per side of a brick
AIR_CONST uint kBrickResolution = 32u;  // bricks per side of a cascade
AIR_CONST float kVoxelBand = 2.0f;      // distances are exact up to this many voxels
AIR_CONST uint kBrickDistanceMax = 15u; // brick distances are counted up to this
AIR_CONST uint kMaxCascades = 8u;
AIR_CONST uint kGiFrameCascades = 192u; // byte offset of GiFrame::cascades
AIR_CONST uint kVoxelFaces = 6u;        // +x, -x, +y, -y, +z, -z
AIR_CONST uint kCacheResolution = 16u;  // radiance cache probes per side of a level (one per 8^3 voxels)
AIR_CONST uint kCacheTexels = 8u;       // octahedral texels per side of a cache probe
AIR_CONST uint kCacheProbes = kCacheResolution * kCacheResolution * kCacheResolution;
AIR_CONST uint kScreenProbeTexels = 8u; // rays per side of a screen probe (kScreenProbeTexels^2 a frame)
AIR_CONST uint kShCoefficients = 9u;    // spherical harmonics up to band 2
AIR_CONST float kGiPi = 3.14159265358979f;
// Light kept in half floats (rays, the cache's probes, the pixels' histories) is stored as nits times this:
// half floats end at 65504, and the light next to a small bright lamp reaches millions of nits (the GPU
// truncates larger values on store, which loses that energy).
AIR_CONST float kHalfLightScale = 1.0f / 256.0f;

// GiFrame::flags
AIR_CONST uint kGiFurnace = 1u;       // white furnace (slots::kFurnace)
AIR_CONST uint kGiHistory = 2u;       // last frame's results are valid (no reset)
AIR_CONST uint kGiScreenTraces = 4u;  // probe rays march the depth buffer before the voxels
AIR_CONST uint kGiCache = 8u;         // probe rays end in the radiance cache past GiFrame::cacheDistance
AIR_CONST uint kGiDirectVoxels = 16u; // the voxels reflect their direct light only (one bounce; checks)

// Views (probe_gi.view): what the composite writes in place of the frame.
AIR_CONST uint kGiViewLit = 0u;        // the frame with the GI added
AIR_CONST uint kGiViewGi = 1u;         // the light the GI adds alone (pathtracer.debugView=15 is its reference)
AIR_CONST uint kGiViewIrradiance = 2u; // the diffuse irradiance / pi the pixels receive (white surfaces)
AIR_CONST uint kGiViewSpecular = 3u;   // the specular light the pixels receive (before the specular albedo)
AIR_CONST uint kGiViewVoxels = 4u;     // camera rays through the voxels: the albedo they hit
AIR_CONST uint kGiViewVoxelLight = 5u; // camera rays through the voxels: the light the voxels reflect
AIR_CONST uint kGiViewSteps = 6u;      // camera rays through the voxels: steps (heatmap)
AIR_CONST uint kGiViewCascades = 7u;   // camera rays through the voxels: the cascade they hit in
AIR_CONST uint kGiViewProbes = 8u;     // the screen probes and the pixels that found none
AIR_CONST uint kGiViewCache = 9u;      // camera rays through the voxels: the radiance cache's irradiance where they hit
AIR_CONST uint kGiViewCount = 10u;

struct GiCamera
{
    float4 position; // xyz, w = tan(verticalFov / 2)
    float4 forward;  // xyz, w = aspect (width / height)
    float4 right;
    float4 up;
};

// One cascade of the voxel scene and its level of the radiance cache.
struct CascadeData
{
    int originX; // world voxel coordinates of the box's min corner (multiples of kCascadeSnap)
    int originY;
    int originZ;
    float voxelSize; // metres
    uint sdf;        // Texture3D<float> (R8_UNORM): distance / kVoxelBand, toroidal
    uint bricks;     // Texture3D<uint> (R8_UINT): Chebyshev distance in bricks to a brick with a surface
    uint voxelIndex; // Texture3D<uint> (R32_UINT): 0, or 1 + the surface voxel's slot
    uint voxels;     // StructuredBuffer<SurfaceVoxel>
    uint valid;      // 0: not built yet (traces pass through it to the next)
    // The radiance cache level over the same box: probe (i, j, k) of world probe coordinates sits at the
    // centre of voxels 8i .. 8i + 7 (and j, k), stored at (i, j, k) mod kCacheResolution.
    uint cacheRadiance;   // Texture2D<float4>: rgb radiance arriving at the probe, a: validity of the probe
    uint cacheDistance;   // Texture2D<float2>: mean distance and squared distance of the rays per texel
    uint cacheIrradiance; // StructuredBuffer<ProbeSh>: the probes' light as spherical harmonics
    // For the passes that update them (VoxelLight, CacheTrace, CacheIntegrate: every cascade in one dispatch):
    uint voxelsWritable; // RWStructuredBuffer<SurfaceVoxel>, the same buffer as `voxels`
    uint counters;       // ByteAddressBuffer (kCounter*)
    uint lightBudget;    // surface voxels lit this frame, cycling (0: all, the cascade was just built)
    float lightBlend;    // weight of this frame's estimate in a voxel's light
    uint cacheProbes;    // cache probes traced this frame, cycling (0: all)
    uint cacheRadianceWritable;
    uint cacheDistanceWritable;
    uint cacheIrradianceWritable;
};

// What a surface voxel shows towards each axis direction, packed. A face is present when some triangle's
// side (front or back: surfaces reflect on both) faces that way.
struct SurfaceVoxel
{
    uint coord;     // storage coordinates x | y << 8 | z << 16, face mask << 24
    uint albedo[6]; // RGBA8: linear albedo (the BSDF's average reflectance), a = how much of the voxel it covers
    // The averaged normal of the face's sides (two 12-bit octahedral components) and where their surface lies
    // along it from the voxel's centre (8-bit snorm of -1..1 voxels): PackNormalOffset.
    uint normal[6];
    uint radiance[6]; // RGB9E5 (PackRadiance): the light it reflects that way, nits
    // The part of radiance that is its direct light (the sun's and the emitters'), averaged over more lightings than
    // the rest: one shadow ray per lighting makes it the noisier part, while the indirect light, every further bounce
    // over frames, needs the faster blend to converge.
    uint direct[6];
    uint age; // 4 bits per face: lightings it has had (saturating; 0: radiance is a first guess)
};

// Irradiance of a probe as spherical harmonics up to band 2, per colour channel: the light convolved with
// the clamped cosine, so evaluating it for a normal gives the irradiance (W/m^2-like, nits * sr).
struct ProbeSh
{
    float4 c0; // (r, g, b) of coefficient 0 and the r of coefficient 1, then on in that order
    float4 c1;
    float4 c2;
    float4 c3;
    float4 c4;
    float4 c5;
    float3 c6;
    float valid; // the probe sees the outside of the scene's surfaces (not inside a mesh)
    // Screen probes: the irradiance at the probe's own normal by quadrature of its rays (exact where harmonics
    // up to band 2 are not: they are off by -7% to +13% for light from a single direction), w the frames of
    // history it accumulates (ProbeTemporal.hlsl).
    float4 normalIrradiance;
};

// A screen probe: where its rays start.
struct ScreenProbe
{
    float3 position; // world, on the surface
    uint normal;     // octahedral 16:16
    float depth;     // view depth of its pixel (0: no probe, the tile shows the sky)
    uint pixel;      // x | y << 16
    float footprint; // metres a pixel covers at the probe
    uint pad0;
};

// Everything a frame needs that does not fit in push constants: an element of an upload ring.
struct GiFrame
{
    GiCamera camera;
    GiCamera previousCamera;
    SkyData sky; // uniform and white in the furnace
    CascadeData cascades[8];
    float2 jitter; // sub-pixel offset of the frame's samples, pixels
    float2 previousJitter;
    uint width; // render size
    uint height;
    uint frameIndex;
    uint cascadeCount;
    float exposure; // radiance -> sceneColor
    uint flags;     // kGi*
    uint view;      // kGiView*
    uint lightCount;
    uint lights;    // StructuredBuffer<LightTriangle>, the scene's light list (alias table)
    uint skyTable;  // StructuredBuffer<float>: the sky radiance table
    uint probeTile; // pixels per side of a screen probe's tile
    uint probesX;   // screen probes per row
    uint probesY;
    float cacheDistance;   // metres (times the cache spacing where the ray is) before a probe ray ends in the cache
    float screenThickness; // relative depth a screen trace accepts as a hit
    uint screenSteps;      // samples of a screen trace
    float voxelBlend;      // weight of a new lighting estimate in a surface voxel's radiance
    float cacheBlend;      // weight of a new trace in a cache probe's radiance
    float temporalBlend;   // weight of a frame in the pixels' history (at least)
    uint debugCascade;
    float debugScale;
    float probeBlend; // weight of a frame in the screen probes' history (at least)
    // Rows of adaptive probes after the probesY rows of the tiles' (probesX each): probes placed where the 4 probes
    // around a pixel do not fit it (ProbeAdapt.hlsl).
    uint adaptiveRows;
    // Weight of a frame in the history of a pixel that did not move (at least; temporalBlend from a pixel of motion
    // on, and for a while after the lights changed).
    float staticBlend;
};

// The instances a cascade's voxelization reads: their triangles numbered one after the other.
struct VoxelizeRange
{
    uint instance;      // GpuScene::instances
    uint firstTriangle; // the number of triangles of the ranges before
    uint triangleCount;
    uint pad0;
};

// VoxelizeCounters (RWByteAddressBuffer, uints)
AIR_CONST uint kCounterWorkItems = 0u;      // voxelization work items (triangle, brick) of the last update
AIR_CONST uint kCounterVoxels = 4u;         // slots ever allocated (freed ones are reused first)
AIR_CONST uint kCounterWorkOverflow = 8u;   // work items that did not fit
AIR_CONST uint kCounterVoxelOverflow = 12u; // surface voxels that did not fit
AIR_CONST uint kCounterFreeHead = 16u;      // the ring of free slots: next to take
AIR_CONST uint kCounterFreeTail = 20u;      // next to put (tail - head slots are free)
AIR_CONST uint kCounterFreshHead = 24u;     // surface voxels ever put in the ring of new ones (kFresh*)
AIR_CONST uint kCounterBytes = 32u;
// New surface voxels of a cascade that moved (VoxelizeResolve.hlsl): a ring of (slot, frame) that VoxelLight.hlsl
// lights, besides its budget, every frame for their first kFreshFrames frames (the newest kFreshWindow of them): a
// voxel that comes into view converges in a few frames instead of waiting for the budget to come round.
AIR_CONST uint kFreshCapacity = 65536u;
AIR_CONST uint kFreshWindow = 16384u;
AIR_CONST uint kFreshFrames = 8u;
AIR_CONST uint kVoxelMarked = 0xFFFFFFFFu; // voxelIndex of a surface voxel before it has a slot
// A work item's brick, flagged when the triangle may cross the brick's voxels (not only reach them with its
// band): those also take its albedo and normal.
AIR_CONST uint kWorkNear = 0x80000000u;

// Push constants of the voxelization passes (Voxelize*.hlsl).
struct VoxelizeConstants
{
    uint frameData; // ByteAddressBuffer (the frame's upload memory): the GiFrame at byte frameOffset
    uint frameOffset;
    uint cascade;
    uint instances;  // StructuredBuffer<InstanceData>
    uint materials;  // StructuredBuffer<MaterialData>
    uint giTextures; // StructuredBuffer<uint>: per texture set, a Texture2D (RGBA8 sRGB) of its colour and
                     // metallic (or alpha, for cutouts) with mips; kInvalidIndex: none
    uint ranges;     // ByteAddressBuffer (the frame's upload memory): rangeCount VoxelizeRange at byte rangeOffset
    uint rangeCount;
    uint triangleCount;
    uint counters;  // RWByteAddressBuffer (kCounter*)
    uint workItems; // RWStructuredBuffer<uint2>: (triangle, brick storage index)
    uint workCapacity;
    uint distances;  // RWTexture3D<uint>: asuint of the distance in voxels (atomic minimum)
    uint voxelIndex; // RWTexture3D<uint>
    uint voxels;     // RWStructuredBuffer<SurfaceVoxel>
    uint voxelCapacity;
    uint accumulators; // RWStructuredBuffer<uint>: kAccumulatorStride per slot
    uint sdf;          // RWTexture3D<float>
    uint occupancy;    // RWTexture3D<uint> (bricks: 1 where a surface voxel is)
    uint bricks;       // RWTexture3D<uint> (Chebyshev distance), and the axis pass's input
    uint axis;         // VoxelizeBricks: 0, 1, 2
    uint bricksIn;     // Texture3D<uint>: the previous axis pass's result
    uint furnace;
    uint rangeOffset;
    // A cascade that moved keeps the voxels of its old box: [keep, keep + kVoxelResolution) relative to its
    // new origin (whole bricks); only the others are voxelized again. incremental = 0: everything is.
    int keepX;
    int keepY;
    int keepZ;
    uint incremental;
    uint freeSlots; // RWStructuredBuffer<uint>: the ring of free slots (voxelCapacity)
    uint fresh;     // RWStructuredBuffer<uint2>: the ring of new voxels (kFreshCapacity)
};

// Per slot and face: red, green, blue, weight, normal x, y, z, the surface's offset from the voxel's centre
// along the normal (fixed point, kAccumulatorScale).
AIR_CONST uint kAccumulatorFace = 8u;
AIR_CONST uint kAccumulatorStride = 48u;
AIR_CONST float kAccumulatorScale = 4096.0f;

// Push constants of the passes that read the frame and one cascade (lighting, the cache, the views).
struct GiConstants
{
    uint frameData; // ByteAddressBuffer (the frame's upload memory): the GiFrame at byte frameOffset
    uint frameOffset;
    uint cascade;
    uint counters; // ByteAddressBuffer: the cascade's (kCounterVoxels)
    uint lights;   // StructuredBuffer<LightTriangle>
    uint period;   // VoxelLight: each surface voxel is lit once per `period` frames
    uint phase;
    float blend; // VoxelLight: weight of this frame's estimate (1: replace)
    uint probes; // CacheTrace / CacheIntegrate: StructuredBuffer<uint> of the probes to update (world probe coordinates
                 // packed 10:10:10, biased)
    uint probeCount;
    uint rays;   // RWTexture2D<float4> (cache rays: radiance, distance)
    uint output; // views: RWTexture2D<float4>
    uint input0; // pass specific
    uint input1;
    uint input2;
    uint input3;
    uint input4;
    uint input5;
    uint input6;
    uint input7;
    uint width;
    uint height;
    uint diffuseSignal;  // Composite: RWTexture2D<float4>, renderer::slots::kDiffuseSignal (kInvalidIndex: none)
    uint specularSignal; // Composite: RWTexture2D<float4>, renderer::slots::kSpecularSignal (kInvalidIndex: none)
};

#ifdef __cplusplus
static_assert(sizeof(GiCamera) == 64);
static_assert(sizeof(CascadeData) == 80);
static_assert(sizeof(SurfaceVoxel) == 104);
static_assert(sizeof(ProbeSh) == 128);
static_assert(sizeof(ScreenProbe) == 32);
static_assert(sizeof(GiFrame) == 128 + 64 + 640 + 16 + 96);
static_assert(offsetof(GiFrame, cascades) == kGiFrameCascades);
static_assert(sizeof(VoxelizeRange) == 16);
static_assert(sizeof(VoxelizeConstants) <= 128, "push constants are limited to 128 bytes");
static_assert(sizeof(GiConstants) <= 128, "push constants are limited to 128 bytes");
} // namespace air::pgi::gpu
#endif

#endif
