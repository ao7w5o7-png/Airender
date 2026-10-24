// Neural radiance cache (Mueller et al. 2021, "Real-time Neural Radiance Caching for Path
// Tracing"): data shared by C++ and HLSL. docs/architecture.md §8.11.
//
// Paths end where their footprint has spread enough (usually at the first diffuse bounce) and take
// the light that vertex reflects from a small network (renderer/shader/MlpShared.h). One pixel per
// tile x tile traces a training path: its render query is deferred (NrcResolve.hlsl), the path goes
// on for a suffix that ends in the cache in turn (self-training, its tail predicted inline), and
// every vertex it saw becomes a training record: the cache's inputs there and the light the path
// found it reflect. 1 in unbiasedPeriod training paths is never ended by the cache, which keeps
// self-training from drifting.
//
// The network predicts reflected radiance (nits) times `normalization` (the camera's exposure, so
// values are O(1) for float16) divided by the vertex's albedo (diffuse + specular, demodulation: the
// network learns the light, not the textures).
#ifndef AIR_PT_NRC_SHARED_H
#define AIR_PT_NRC_SHARED_H

#include "air/renderer/shader/HlslCompat.h"
#include "air/renderer/shader/MlpShared.h"

#ifdef __cplusplus
namespace air::pt::gpu
{
using namespace air::hlsl;
using air::gpu::HashGridData;
using air::gpu::MlpBuffers;
#endif

// A path vertex as the cache sees it.
struct NrcQuery
{
    float3 position;
    uint normal;    // shading normal, octahedral, 2 x unorm16
    uint direction; // towards the previous vertex, octahedral
    uint roughness; // unorm16
    uint diffuse;   // diffuse albedo, unorm 10:10:10
    uint specular;  // specular albedo (directional, at this outgoing direction), unorm 10:10:10
};

// A training sample at a vertex of a training path. While the path is traced `value` holds the light
// gathered before the vertex reflects any and `throughput` the path's throughput up to it; when the
// path ends, `value` becomes the light the vertex reflected towards the previous one (nits).
struct NrcTrainingRecord
{
    NrcQuery query;
    float3 value;
    uint valid; // 1: `value` is complete
    float3 throughput;
    uint pad;
};

// A training path's own render query, for its pixel (radiance += throughput * cache).
struct NrcPixelQuery
{
    NrcQuery query;
    float3 throughput;
    uint pixel; // y * width + x; kNrcNoPixel: the path ended without one
};

AIR_CONST uint kNrcNoPixel = 0xFFFFFFFFu;

// Everything the trace and the cache's passes share, in FrameData.
struct NrcData
{
    MlpBuffers network;
    HashGridData grid;
    uint trainingRecords; // RWStructuredBuffer<NrcTrainingRecord>: recordsPerPath per training path
    uint pixelQueries;    // RWStructuredBuffer<NrcPixelQuery>: one per training path
    uint tile;            // a training path every tile x tile pixels
    uint tilesX;          // whole tiles across and down the image (training paths: tilesX * tilesY)
    uint tilesY;
    uint offsetX; // this frame's training pixel within each tile
    uint offsetY;
    uint recordsPerPath;
    uint unbiasedPeriod; // 1 in this many training paths is never ended by the cache
    float normalization; // radiance -> the network's units (the exposure)
    float spread;        // a path ends where its spread exceeds this times the primary's (paper: 0.01)
    uint frame;
};

// Root constants of NrcTrain*.hlsl and NrcResolve*.hlsl.
struct NrcConstants
{
    uint frameData; // StructuredBuffer<FrameData>
    uint frameSlot;
    uint paths;        // training paths this frame
    uint step;         // NrcTrain: this optimizer step trains the paths NrcStepPath() gives
    uint steps;        // of the frame
    uint accumulation; // NrcResolve: RWTexture2D<float4>
    uint permutation;  // NrcTrain: coprime with `paths`
    uint pad0;
};

#ifdef __cplusplus
static_assert(sizeof(NrcQuery) == 32);
static_assert(sizeof(NrcTrainingRecord) == 64);
static_assert(sizeof(NrcPixelQuery) == 48);
static_assert(sizeof(NrcData) == 192);
static_assert(sizeof(NrcConstants) == 32);
#endif

// --- Packing ---------------------------------------------------------------------------------------

AIR_INLINE uint NrcPackUnorm16x2(float a, float b)
{
    return uint(floor(saturate(a) * 65535.0f + 0.5f)) | (uint(floor(saturate(b) * 65535.0f + 0.5f)) << 16u);
}

AIR_INLINE float NrcUnpackUnorm16(uint packed, uint shift)
{
    return float((packed >> shift) & 0xFFFFu) * (1.0f / 65535.0f);
}

// Octahedral map of a unit vector (Cigolle et al. 2014), as two unorm16.
AIR_INLINE uint NrcPackDirection(float3 d)
{
    float l1 = abs(d.x) + abs(d.y) + abs(d.z);
    float x = d.x / l1;
    float y = d.y / l1;
    if (d.z < 0.0f)
    {
        float ox = (1.0f - abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        float oy = (1.0f - abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = ox;
        y = oy;
    }
    return NrcPackUnorm16x2(x * 0.5f + 0.5f, y * 0.5f + 0.5f);
}

AIR_INLINE float3 NrcUnpackDirection(uint packed)
{
    float x = NrcUnpackUnorm16(packed, 0u) * 2.0f - 1.0f;
    float y = NrcUnpackUnorm16(packed, 16u) * 2.0f - 1.0f;
    float z = 1.0f - abs(x) - abs(y);
    if (z < 0.0f)
    {
        float ox = (1.0f - abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        float oy = (1.0f - abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = ox;
        y = oy;
    }
    return normalize(float3(x, y, z));
}

AIR_INLINE uint NrcPackColor(float3 c)
{
    return uint(floor(saturate(c.x) * 1023.0f + 0.5f)) | (uint(floor(saturate(c.y) * 1023.0f + 0.5f)) << 10u) |
           (uint(floor(saturate(c.z) * 1023.0f + 0.5f)) << 20u);
}

AIR_INLINE float3 NrcUnpackColor(uint packed)
{
    return float3(float(packed & 1023u), float((packed >> 10u) & 1023u), float((packed >> 20u) & 1023u)) *
           (1.0f / 1023.0f);
}

// The factor the network's output is divided by: the vertex's albedo (never quite 0).
AIR_INLINE float3 NrcDemodulation(NrcQuery q)
{
    float3 albedo = NrcUnpackColor(q.diffuse) + NrcUnpackColor(q.specular);
    return float3(max(albedo.x, 1e-3f), max(albedo.y, 1e-3f), max(albedo.z, 1e-3f));
}

// The path whose records a step's i-th group of threads trains (i < the step's share of the paths):
// every steps-th path of the frame's paths in a scattered order (a multiplicative permutation), so
// that every step sees all of the image.
AIR_INLINE uint NrcStepPath(uint i, uint step, uint steps, uint paths, uint permutation)
{
    return uint((uint64_t(i * steps + step) * uint64_t(permutation)) % uint64_t(paths));
}

#ifdef __cplusplus
} // namespace air::pt::gpu
#endif

#endif
