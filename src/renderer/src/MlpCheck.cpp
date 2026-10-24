// renderer::CheckNeuralNetwork: the GPU's network kernels (MlpCheck*.hlsl, the code training kernels
// use) against the CPU twin (Mlp.h), on random inputs and loss gradients.

#include "air/core/Log.h"
#include "air/renderer/Mlp.h"
#include "air/renderer/NeuralNetwork.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/Transfer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace air::renderer
{
namespace
{

using rhi::Access;

struct GpuResults
{
    bool ran = false;
    std::vector<float> outputs;         // kMlpOutputs per sample
    std::vector<float> inputGradients;  // kMlpWidth per sample
    std::vector<float> weightGradients; // summed, without the kernels' loss scale
};

std::vector<float> ToFloats(const std::vector<u8>& bytes, size_t count)
{
    std::vector<float> values(count, 0.0f);
    std::memcpy(values.data(), bytes.data(), std::min(bytes.size(), count * sizeof(float)));
    return values;
}

GpuResults RunOnGpu(rhi::Device& device, const std::filesystem::path& dir, bool linearAlgebra,
                    const std::vector<float>& parameters, const std::vector<float>& inputs,
                    const std::vector<float>& outputGradients, u32 samples)
{
    GpuResults results;
    NeuralNetwork network(device, {.encodingParameters = 0, .outputs = gpu::kMlpOutputs, .name = "MlpCheck"});
    if (!network.Initialize(dir, linearAlgebra, 1))
    {
        return results;
    }
    network.Upload(parameters);
    rhi::ComputePipelineDesc desc;
    desc.shader = LoadShader(device, dir, linearAlgebra ? "MlpCheck.cs" : "MlpCheckPortable.cs", "Main");
    desc.pushConstantBytes = sizeof(gpu::MlpCheckConstants);
    desc.debugName = linearAlgebra ? "MlpCheck" : "MlpCheckPortable";
    if (desc.shader.bytecode.empty())
    {
        AIR_LOG(Warning, "Renderer", "{}.cs is missing: not checked", desc.debugName);
        return results;
    }
    const rhi::PipelineHandle pipeline = device.CreateComputePipeline(desc);

    const rhi::BufferUsage usage = rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::ShaderResource;
    auto upload = [&](const std::vector<float>& values, const char* name)
    {
        return CreateBufferWithData(device, {.size = values.size() * 4, .usage = usage, .debugName = name},
                                    values.data(), values.size() * 4);
    };
    const rhi::BufferHandle inputBuffer = upload(inputs, "MlpCheck.Inputs");
    const rhi::BufferHandle gradientBuffer = upload(outputGradients, "MlpCheck.OutputGradients");
    const rhi::BufferHandle outputBuffer =
        upload(std::vector<float>(size_t(samples) * gpu::kMlpOutputs, 0.0f), "MlpCheck.Outputs");
    const rhi::BufferHandle inputGradientBuffer =
        upload(std::vector<float>(size_t(samples) * gpu::kMlpWidth, 0.0f), "MlpCheck.InputGradients");

    rdg::RenderGraph graph(device);
    network.BeginFrame(graph);
    const rdg::BufferRef parametersRef = network.Parameters();
    const rdg::BufferRef matricesRef = network.Matrices();
    const rdg::BufferRef gradientsRef = network.Gradients();
    const rdg::BufferRef inputsRef = graph.ImportBuffer(inputBuffer, Access::SRVCompute, Access::SRVCompute);
    const rdg::BufferRef lossRef = graph.ImportBuffer(gradientBuffer, Access::SRVCompute, Access::SRVCompute);
    const rdg::BufferRef outputsRef = graph.ImportBuffer(outputBuffer, Access::SRVCompute, Access::SRVCompute);
    const rdg::BufferRef inputGradientsRef =
        graph.ImportBuffer(inputGradientBuffer, Access::SRVCompute, Access::SRVCompute);
    gpu::MlpCheckConstants constants{};
    constants.network = network.GetBuffers();
    constants.samples = samples;
    graph.AddPass(
        "MlpCheck", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(parametersRef, Access::SRVCompute);
            if (matricesRef.IsValid())
            {
                b.Read(matricesRef, Access::SRVCompute);
            }
            b.Write(gradientsRef, Access::UAVCompute);
            b.Read(inputsRef, Access::SRVCompute);
            b.Read(lossRef, Access::SRVCompute);
            b.Write(outputsRef, Access::UAVCompute);
            b.Write(inputGradientsRef, Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            constants.inputs = ctx.SRV(inputsRef);
            constants.outputGradients = ctx.SRV(lossRef);
            constants.outputs = ctx.UAV(outputsRef);
            constants.inputGradients = ctx.UAV(inputGradientsRef);
            ctx.cmd.SetPipeline(pipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch((samples + 63) / 64, 1, 1);
        });
    device.WaitForFence(rhi::QueueType::Graphics, graph.Execute());

    results.ran = true;
    results.outputs =
        ToFloats(ReadbackBuffer(device, outputBuffer, Access::SRVCompute), size_t(samples) * gpu::kMlpOutputs);
    results.inputGradients =
        ToFloats(ReadbackBuffer(device, inputGradientBuffer, Access::SRVCompute), size_t(samples) * gpu::kMlpWidth);
    results.weightGradients = network.ReadWeightGradients();
    for (float& g : results.weightGradients)
    {
        g /= gpu::kMlpLossScale;
    }
    for (rhi::BufferHandle buffer : {inputBuffer, gradientBuffer, outputBuffer, inputGradientBuffer})
    {
        device.DestroyBuffer(buffer);
    }
    device.DestroyPipeline(pipeline);
    return results;
}

// The error of gpu against cpu relative to cpu's size: root mean square, and largest element (one
// ReLU switched by rounding changes its elements of a gradient entirely: the largest error is large
// where the RMS one is small).
struct Error
{
    double rms = 0.0;
    double largest = 0.0;
};

Error RelativeError(const std::vector<float>& gpu, const std::vector<double>& cpu)
{
    double squares = 0.0;
    double errorSquares = 0.0;
    double largest = 0.0;
    double largestError = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i)
    {
        const double difference = i < gpu.size() ? double(gpu[i]) - cpu[i] : cpu[i];
        squares += cpu[i] * cpu[i];
        errorSquares += std::isfinite(difference) ? difference * difference : 1e60;
        largest = std::max(largest, std::abs(cpu[i]));
        largestError = std::max(largestError, std::isfinite(difference) ? std::abs(difference) : 1e30);
    }
    Error e;
    e.rms = squares > 0.0 ? std::sqrt(errorSquares / squares) : std::sqrt(errorSquares);
    e.largest = largest > 0.0 ? largestError / largest : largestError;
    return e;
}

json::Value ToJson(const Error& e)
{
    json::Value v = json::Value::MakeObject();
    v["rms"] = e.rms;
    v["largest"] = e.largest;
    return v;
}

struct Answers
{
    std::vector<double> outputs;         // kMlpOutputs per sample
    std::vector<double> inputGradients;  // kMlpWidth per sample
    std::vector<double> weightGradients; // summed
};

// Relative RMS errors of outputs, input gradients and weight gradients, each within its tolerance.
json::Value CompareAll(const GpuResults& gpu, const Answers& cpu, double outputTolerance, double gradientTolerance,
                       bool& passed)
{
    json::Value result = json::Value::MakeObject();
    const Error outputError = RelativeError(gpu.outputs, cpu.outputs);
    const Error inputError = RelativeError(gpu.inputGradients, cpu.inputGradients);
    const Error weightError = RelativeError(gpu.weightGradients, cpu.weightGradients);
    result["outputs"] = ToJson(outputError);
    result["inputGradients"] = ToJson(inputError);
    result["weightGradients"] = ToJson(weightError);
    result["outputTolerance"] = outputTolerance;
    result["gradientTolerance"] = gradientTolerance;
    passed = outputError.rms <= outputTolerance && inputError.rms <= gradientTolerance &&
             weightError.rms <= gradientTolerance;
    result["passed"] = passed;
    return result;
}

} // namespace

json::Value CheckNeuralNetwork(rhi::Device& device, const std::filesystem::path& rendererShaderDir, u32 samples,
                               u32 seed)
{
    samples = std::max(samples, 1u);
    // Every output row in use, so the output layer is checked whole (transposed too).
    const std::vector<float> parameters = InitializeMlpParameters(0, gpu::kMlpOutputs, seed);
    std::mt19937 rng(seed + 1);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> symmetric(-1.0f, 1.0f);
    std::vector<float> inputs(size_t(samples) * gpu::kMlpWidth);
    std::vector<float> outputGradients(size_t(samples) * gpu::kMlpOutputs);
    for (u32 s = 0; s < samples; ++s)
    {
        for (u32 i = 0; i < gpu::kMlpWidth; ++i)
        {
            inputs[size_t(s) * gpu::kMlpWidth + i] = i + 1 == gpu::kMlpWidth ? 1.0f : unit(rng);
        }
        for (u32 o = 0; o < gpu::kMlpOutputs; ++o)
        {
            outputGradients[size_t(s) * gpu::kMlpOutputs + o] = symmetric(rng);
        }
    }

    // The CPU's answers in float32 (renderer/Mlp.h), in double where they are sums.
    Answers float32;
    float32.outputs.resize(size_t(samples) * gpu::kMlpOutputs);
    float32.inputGradients.resize(size_t(samples) * gpu::kMlpWidth);
    float32.weightGradients.assign(gpu::kMlpWeights, 0.0);
    std::vector<double>& outputs = float32.outputs;
    std::vector<double>& inputGradients = float32.inputGradients;
    std::vector<double>& weightGradients = float32.weightGradients;
    for (u32 s = 0; s < samples; ++s)
    {
        const std::span<const float> input(&inputs[size_t(s) * gpu::kMlpWidth], gpu::kMlpWidth);
        const std::span<const float> loss(&outputGradients[size_t(s) * gpu::kMlpOutputs], gpu::kMlpOutputs);
        float output[gpu::kMlpOutputs] = {};
        EvaluateMlp(parameters, input, output);
        float inputGradient[gpu::kMlpWidth] = {};
        BackpropagateMlp(parameters, input, loss, weightGradients, inputGradient);
        for (u32 o = 0; o < gpu::kMlpOutputs; ++o)
        {
            outputs[size_t(s) * gpu::kMlpOutputs + o] = output[o];
        }
        for (u32 i = 0; i < gpu::kMlpWidth; ++i)
        {
            inputGradients[size_t(s) * gpu::kMlpWidth + i] = inputGradient[i];
        }
    }

    json::Value report = json::Value::MakeObject();
    report["samples"] = samples;
    bool passed = true;
    for (bool linearAlgebra : {false, true})
    {
        const char* name = linearAlgebra ? "linearAlgebra" : "float32";
        if (linearAlgebra && !device.GetCapabilities().linearAlgebra)
        {
            report[name] = "unavailable";
            continue;
        }
        const GpuResults gpu =
            RunOnGpu(device, rendererShaderDir, linearAlgebra, parameters, inputs, outputGradients, samples);
        if (!gpu.ran)
        {
            report[name] = "not run (shaders missing)";
            passed = false;
            continue;
        }
        // The float32 kernels round like the CPU. The matrix units multiply float16 and sum in float16
        // blocks (about 1e-3 per product, measured), which backpropagation compounds and which also
        // switches the ReLUs whose input is near zero (a few percent of a gradient's RMS); a wrong
        // layout, transpose or offset shows as errors near 1.
        bool ok = false;
        json::Value result =
            linearAlgebra ? CompareAll(gpu, float32, 5e-3, 5e-2, ok) : CompareAll(gpu, float32, 1e-5, 1e-5, ok);
        passed = passed && ok;
        report[name] = std::move(result);
    }
    report["passed"] = passed;
    return report;
}

} // namespace air::renderer
