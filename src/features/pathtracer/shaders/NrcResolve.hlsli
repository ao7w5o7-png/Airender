// The training paths' own render queries (NrcShared.h): a training path deferred its pixel's cache
// lookup so that the trace kernel's lookups (each thread's last) stay one per thread; this pass adds
// it to the accumulation, with the same network as the trace. Compiled twice: NrcResolve.hlsl on the
// matrix units (AIR_NRC_LINALG 1) and NrcResolvePortable.hlsl in float32.

#include "PathTracerShared.h"

[[vk::push_constant]] ConstantBuffer<NrcConstants> g : register(b0);

static NrcData gNrc;

float4 MlpLoadWeights4(uint index)
{
    ByteAddressBuffer parameters = ResourceDescriptorHeap[gNrc.network.parameters];
    return asfloat(parameters.Load4(index * 4u));
}

float2 MlpLoadFeatures(uint index)
{
    ByteAddressBuffer parameters = ResourceDescriptorHeap[gNrc.network.parameters];
    return asfloat(parameters.Load2(index * 4u));
}

#include "Nrc.hlsli"
#if AIR_NRC_LINALG
#include "air/renderer/shader/MlpLinAlg.hlsli"
#endif

// A thread per training path; every thread runs the network (see NrcTrain.hlsli).
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    FrameData frame = frames[g.frameSlot];
    gNrc = frame.nrc;
    RWStructuredBuffer<NrcPixelQuery> queries = ResourceDescriptorHeap[gNrc.pixelQueries];
    NrcPixelQuery p = queries[min(id.x, g.paths - 1u)];
    bool valid = id.x < g.paths && p.pixel != kNrcNoPixel;
    MlpVector x = NrcEncode(gNrc.grid, p.query);
#if AIR_NRC_LINALG
    MlpLinAlgForward f = MlpLinAlgEvaluate(gNrc.network, MlpToHalf(x));
    float3 output = float3(f.output[0], f.output[1], f.output[2]);
#else
    MlpForward f = MlpEvaluate(x);
    float3 output = f.output.v[0].xyz;
#endif
    if (valid)
    {
        RWTexture2D<float4> accumulation = ResourceDescriptorHeap[g.accumulation];
        uint2 pixel = uint2(p.pixel % frame.width, p.pixel / frame.width);
        accumulation[pixel] += float4(p.throughput * NrcRadiance(p.query, output, gNrc.normalization), 0.0f);
    }
}
