// Training the neural radiance cache (NrcShared.h): one optimizer step's records. A thread per record
// slot evaluates the network on the record's inputs, takes the loss gradient against its target and
// backpropagates it into the gradients, which renderer::NeuralNetwork then applies. Compiled twice:
// NrcTrain.hlsl on the matrix units (AIR_NRC_LINALG 1, Shader Model 6.10) and NrcTrainPortable.hlsl
// in float32.

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
#include "air/renderer/shader/MlpGradients.hlsli"
#if AIR_NRC_LINALG
#include "air/renderer/shader/MlpLinAlg.hlsli"
#endif

// A thread per record slot of the step's paths. Every thread runs the network, those without a record
// on a dummy input and with a zero loss gradient: the matrix units see whole waves (see
// Integrator.hlsli) and the float32 path's wave sums need every lane.
[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    StructuredBuffer<FrameData> frames = ResourceDescriptorHeap[g.frameData];
    gNrc = frames[g.frameSlot].nrc;
    uint i = id.x / gNrc.recordsPerPath;
    bool valid = i * g.steps + g.step < g.paths;
    uint path = valid ? NrcStepPath(i, g.step, g.steps, g.paths, g.permutation) : 0u;
    NrcTrainingRecord record;
    record.query.position = float3(0.0f, 0.0f, 0.0f);
    record.query.normal = 0u;
    record.query.direction = 0u;
    record.query.roughness = 0u;
    record.query.diffuse = 0u;
    record.query.specular = 0u;
    record.value = float3(0.0f, 0.0f, 0.0f);
    record.valid = 0u;
    record.throughput = float3(0.0f, 0.0f, 0.0f);
    record.pad = 0u;
    if (valid)
    {
        RWStructuredBuffer<NrcTrainingRecord> records = ResourceDescriptorHeap[gNrc.trainingRecords];
        record = records[path * gNrc.recordsPerPath + id.x % gNrc.recordsPerPath];
        valid = record.valid != 0u;
    }
    MlpVector x = MlpZero();
    if (valid)
    {
        x = NrcEncode(gNrc.grid, record.query);
    }
#if AIR_NRC_LINALG
    MlpLinAlgForward f = MlpLinAlgEvaluate(gNrc.network, MlpToHalf(x));
    float3 output = float3(f.output[0], f.output[1], f.output[2]);
    float3 gradient = valid ? MlpLossGradient(output, NrcTarget(record, gNrc.normalization)) * kMlpLossScale
                            : float3(0.0f, 0.0f, 0.0f);
    MlpHalfOutput dy = (MlpHalfOutput)0.0h;
    dy[0] = (half)gradient.x;
    dy[1] = (half)gradient.y;
    dy[2] = (half)gradient.z;
    MlpVector dx = MlpFromHalf(MlpLinAlgBackpropagate(gNrc.network, f, dy));
#else
    MlpForward f = MlpEvaluate(x);
    float3 gradient = valid ? MlpLossGradient(f.output.v[0].xyz, NrcTarget(record, gNrc.normalization)) * kMlpLossScale
                            : float3(0.0f, 0.0f, 0.0f);
    MlpVector dy = MlpZero();
    dy.v[0] = float4(gradient, 0.0f);
    MlpBackward b = MlpDifferentiate(f, dy);
    MlpAccumulateWeightGradients(gNrc.network, f, b);
    MlpVector dx = b.inputGradient;
#endif
    if (valid)
    {
        MlpScatterHashGrid(gNrc.network, gNrc.grid, record.query.position, dx);
    }
}
