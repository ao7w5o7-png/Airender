// The renderer's neural networks on the CPU (renderer/Mlp.h, shader/Mlp.hlsli, the same float32 code
// GPUs without linear algebra run): backpropagation agrees with finite differences through the MLP
// and the hash grid, the direction encoding is orthonormal, and Adam fits a function. Then the GPU
// network's passes (renderer/NeuralNetwork.h) on the Null device.

#include "TestFramework.h"

#include "air/rdg/RenderGraph.h"
#include "air/renderer/Mlp.h"
#include "air/renderer/NeuralNetwork.h"
#include "air/rhi/RHI.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

using namespace air;
using namespace air::hlsl;
using namespace air::gpu;

namespace
{

// A loss linear in the outputs: its gradient with respect to them is `weights`.
constexpr float kLossWeights[3] = {0.3f, -0.7f, 0.5f};

double LinearLoss(std::span<const float> parameters, std::span<const float> input)
{
    float outputs[kMlpOutputs] = {};
    renderer::EvaluateMlp(parameters, input, outputs);
    return double(kLossWeights[0]) * outputs[0] + double(kLossWeights[1]) * outputs[1] +
           double(kLossWeights[2]) * outputs[2];
}

std::vector<float> LossGradient()
{
    std::vector<float> g(kMlpOutputs, 0.0f);
    for (u32 c = 0; c < 3; ++c)
    {
        g[c] = kLossWeights[c];
    }
    return g;
}

// Fraction of (analytic, numeric) pairs that agree. ReLU kinks crossed by the finite difference step
// make a few disagree legitimately.
struct Agreement
{
    u32 checked = 0;
    u32 agreed = 0;
    void Add(double analytic, double numeric)
    {
        ++checked;
        if (std::abs(analytic - numeric) <= 2e-3 + 1e-2 * std::abs(analytic))
        {
            ++agreed;
        }
        else if (checked - agreed <= 3)
        {
            std::fprintf(stderr, "  analytic %.6g numeric %.6g\n", analytic, numeric);
        }
    }
    double Fraction() const { return checked > 0 ? double(agreed) / checked : 0.0; }
};

} // namespace

TEST(Mlp_BackpropagationMatchesFiniteDifferences)
{
    std::vector<float> parameters = renderer::InitializeMlpParameters(0, 3, 1);
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> input(kMlpWidth);
    for (float& x : input)
    {
        x = unit(rng);
    }
    std::vector<double> weightGradients(kMlpWeights, 0.0);
    std::vector<float> inputGradient(kMlpWidth, 0.0f);
    const std::vector<float> lossGradient = LossGradient();
    renderer::BackpropagateMlp(parameters, input, lossGradient, weightGradients, inputGradient);

    const float h = 1e-3f;
    Agreement weights;
    std::uniform_int_distribution<u32> pick(0, kMlpWeights - 1);
    for (u32 n = 0; n < 400; ++n)
    {
        const u32 i = pick(rng);
        const float saved = parameters[i];
        parameters[i] = saved + h;
        const double plus = LinearLoss(parameters, input);
        parameters[i] = saved - h;
        const double minus = LinearLoss(parameters, input);
        parameters[i] = saved;
        weights.Add(weightGradients[i], (plus - minus) / (2.0 * h));
    }
    Agreement inputs;
    for (u32 i = 0; i < kMlpWidth; ++i)
    {
        const float saved = input[i];
        input[i] = saved + h;
        const double plus = LinearLoss(parameters, input);
        input[i] = saved - h;
        const double minus = LinearLoss(parameters, input);
        input[i] = saved;
        inputs.Add(inputGradient[i], (plus - minus) / (2.0 * h));
    }
    std::printf("  weights %u/%u agree, inputs %u/%u\n", weights.agreed, weights.checked, inputs.agreed,
                inputs.checked);
    EXPECT(weights.Fraction() >= 0.99);
    EXPECT(inputs.Fraction() >= 0.97);
    // The output rows beyond the 3 in use start at zero and get no gradient.
    double unused = 0.0;
    for (u32 i = MlpMatrixOffset(kMlpMatrices - 1) + 3 * kMlpWidth; i < kMlpWeights; ++i)
    {
        unused += std::abs(parameters[i]) + std::abs(weightGradients[i]);
    }
    EXPECT(unused == 0.0);
}

TEST(Mlp_HashGridGradientMatchesFiniteDifferences)
{
    const u32 tableSize = 1u << 10;
    const HashGridData grid = renderer::MakeHashGrid(float3(-1.0f, 0.0f, 2.0f), float3(3.0f, 2.0f, 4.0f), tableSize,
                                                     2.0f, 64.0f, kMlpWeights);
    std::vector<float> parameters = renderer::InitializeMlpParameters(kHashLevels * tableSize * kHashFeatures, 3, 3);
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> features(-0.5f, 0.5f);
    for (u32 i = kMlpWeights; i < parameters.size(); ++i)
    {
        parameters[i] = features(rng); // large enough for the hash grid to matter
    }
    // The coarse levels map vertices one to one, the fine ones hash: both kinds are checked.
    EXPECT(renderer::MakeHashGrid(float3(0.0f), float3(1.0f), tableSize, 2.0f, 64.0f, 0).resolution[0].x == 2.0f);
    const float3 p(0.7f, 1.3f, 3.1f);
    auto loss = [&]
    {
        std::vector<float> input(kMlpWidth, 0.0f);
        renderer::EncodeHashGrid(parameters, grid, p, input);
        input[kHashInputs] = 1.0f;
        return LinearLoss(parameters, input);
    };
    std::vector<float> input(kMlpWidth, 0.0f);
    renderer::EncodeHashGrid(parameters, grid, p, input);
    input[kHashInputs] = 1.0f;
    std::vector<double> gradients(parameters.size(), 0.0);
    std::vector<float> inputGradient(kMlpWidth, 0.0f);
    renderer::BackpropagateMlp(parameters, input, LossGradient(), gradients, inputGradient);
    renderer::BackpropagateHashGrid(grid, p, inputGradient, gradients);

    const float h = 1e-3f;
    Agreement entries;
    for (u32 level = 0; level < kHashLevels; ++level)
    {
        for (u32 corner = 0; corner < 8; corner += 3)
        {
            uint entry = 0;
            float weight = 0.0f;
            MlpHashCorner(grid, p, level, corner, entry, weight);
            for (u32 f = 0; f < kHashFeatures; ++f)
            {
                const u32 i = MlpHashParameter(grid, level, entry) + f;
                const float saved = parameters[i];
                parameters[i] = saved + h;
                const double plus = loss();
                parameters[i] = saved - h;
                const double minus = loss();
                parameters[i] = saved;
                entries.Add(gradients[i], (plus - minus) / (2.0 * h));
            }
        }
    }
    std::printf("  hash grid entries %u/%u agree\n", entries.agreed, entries.checked);
    EXPECT(entries.Fraction() >= 0.97);
    // The 8 trilinear weights of a level sum to 1.
    for (u32 level = 0; level < kHashLevels; ++level)
    {
        double sum = 0.0;
        for (u32 corner = 0; corner < 8; ++corner)
        {
            uint entry = 0;
            float weight = 0.0f;
            MlpHashCorner(grid, p, level, corner, entry, weight);
            sum += weight;
            EXPECT(entry < tableSize);
        }
        EXPECT(std::abs(sum - 1.0) < 1e-5);
    }
}

TEST(Mlp_SphericalHarmonicsAreOrthonormal)
{
    // Midpoint quadrature over the sphere.
    const u32 rows = 256;
    const u32 columns = 512;
    double gram[16][16] = {};
    for (u32 i = 0; i < rows; ++i)
    {
        const double theta = (i + 0.5) * std::numbers::pi / rows;
        for (u32 j = 0; j < columns; ++j)
        {
            const double phi = (j + 0.5) * 2.0 * std::numbers::pi / columns;
            const float3 d(float(std::sin(theta) * std::cos(phi)), float(std::sin(theta) * std::sin(phi)),
                           float(std::cos(theta)));
            float4 sh[4];
            MlpSphericalHarmonics(d, sh[0], sh[1], sh[2], sh[3]);
            const double area = std::sin(theta) * (std::numbers::pi / rows) * (2.0 * std::numbers::pi / columns);
            for (u32 a = 0; a < 16; ++a)
            {
                for (u32 b = 0; b < 16; ++b)
                {
                    gram[a][b] += double(MlpComponent(sh[a / 4], a % 4)) * MlpComponent(sh[b / 4], b % 4) * area;
                }
            }
        }
    }
    double worst = 0.0;
    for (u32 a = 0; a < 16; ++a)
    {
        for (u32 b = 0; b < 16; ++b)
        {
            worst = std::max(worst, std::abs(gram[a][b] - (a == b ? 1.0 : 0.0)));
        }
    }
    std::printf("  largest deviation from the identity: %.2e\n", worst);
    EXPECT(worst < 1e-3);
}

TEST(Mlp_AdamFitsAFunction)
{
    // A colourful function of position and direction, as a radiance cache would learn.
    auto target = [](float3 p, float3 d)
    {
        return float3(0.5f + 0.4f * std::sin(6.0f * p.x) * std::cos(4.0f * p.y), 0.3f + 0.2f * d.y + 0.2f * p.z,
                      0.8f * p.x * p.y + 0.1f + 0.1f * d.x * d.x);
    };
    const u32 tableSize = 1u << 12;
    const HashGridData grid = renderer::MakeHashGrid(float3(0.0f), float3(1.0f), tableSize, 4.0f, 128.0f, kMlpWeights);
    std::vector<float> parameters = renderer::InitializeMlpParameters(kHashLevels * tableSize * kHashFeatures, 3, 5);
    std::vector<float> m(parameters.size(), 0.0f);
    std::vector<float> v(parameters.size(), 0.0f);
    std::mt19937 rng(6);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    struct Sample
    {
        float3 p, d;
    };
    auto draw = [&]
    {
        const float z = 2.0f * unit(rng) - 1.0f;
        const float phi = 2.0f * std::numbers::pi_v<float> * unit(rng);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        return Sample{float3(unit(rng), unit(rng), unit(rng)), float3(r * std::cos(phi), r * std::sin(phi), z)};
    };
    auto encode = [&](const Sample& s, std::vector<float>& input)
    {
        std::fill(input.begin(), input.end(), 0.0f);
        renderer::EncodeHashGrid(parameters, grid, s.p, input);
        float4 sh[4];
        MlpSphericalHarmonics(s.d, sh[0], sh[1], sh[2], sh[3]);
        for (u32 i = 0; i < 16; ++i)
        {
            input[kHashInputs + i] = MlpComponent(sh[i / 4], i % 4);
        }
        input[kHashInputs + 16] = 1.0f;
    };
    std::vector<Sample> test(256);
    for (Sample& s : test)
    {
        s = draw();
    }
    auto relativeError = [&]
    {
        std::vector<float> input(kMlpWidth);
        double sum = 0.0;
        for (const Sample& s : test)
        {
            encode(s, input);
            float y[kMlpOutputs] = {};
            renderer::EvaluateMlp(parameters, input, y);
            const float3 t = target(s.p, s.d);
            sum += (std::abs(y[0] - t.x) + std::abs(y[1] - t.y) + std::abs(y[2] - t.z)) / (t.x + t.y + t.z);
        }
        return sum / double(test.size());
    };
    const double before = relativeError();
    const u32 batch = 128;
    std::vector<float> input(kMlpWidth);
    std::vector<double> gradients(parameters.size());
    std::vector<float> inputGradient(kMlpWidth);
    std::vector<float> lossGradient(kMlpOutputs, 0.0f);
    for (u32 step = 1; step <= 400; ++step)
    {
        std::fill(gradients.begin(), gradients.end(), 0.0);
        for (u32 b = 0; b < batch; ++b)
        {
            const Sample s = draw();
            encode(s, input);
            float y[kMlpOutputs] = {};
            renderer::EvaluateMlp(parameters, input, y);
            const float3 g = MlpLossGradient(float3(y[0], y[1], y[2]), target(s.p, s.d));
            lossGradient[0] = g.x;
            lossGradient[1] = g.y;
            lossGradient[2] = g.z;
            renderer::BackpropagateMlp(parameters, input, lossGradient, gradients, inputGradient);
            renderer::BackpropagateHashGrid(grid, s.p, inputGradient, gradients);
        }
        renderer::AdamStep(renderer::MakeAdam(1e-2f, step, 1.0f / float(batch)), parameters, m, v, gradients);
    }
    const double after = relativeError();
    std::printf("  mean relative error %.3f before training, %.4f after 400 steps of %u samples\n", before, after,
                batch);
    EXPECT(after < 0.06);
    EXPECT(after < before * 0.1);
}

// renderer::NeuralNetwork's passes on the Null device: with linear algebra (the Null device pretends)
// the uploaded weights are converted once, then every step converts the gradients and the new
// weights around the optimizer; without, only the optimizer runs.
TEST(Mlp_NetworkStepsAroundTheOptimizer)
{
    std::unique_ptr<rhi::Device> device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    device->SetCommandCapture(true);
    const rhi::PipelineHandle kernel = device->CreateComputePipeline({.debugName = "Train"});
    auto count = [](const std::vector<std::string>& lines, std::string_view needle)
    {
        return std::count_if(lines.begin(), lines.end(),
                             [&](const std::string& line) { return line.find(needle) != std::string::npos; });
    };
    for (bool linearAlgebra : {true, false})
    {
        renderer::NeuralNetwork network(*device, {.encodingParameters = 64, .outputs = 3, .name = "Test"});
        EXPECT(network.Initialize(AIR_BUILD_SHADER_DIR "/renderer", linearAlgebra, 7));
        EXPECT(network.UsesLinearAlgebra() == linearAlgebra);
        for (int frame = 0; frame < 2; ++frame)
        {
            rdg::RenderGraph graph(*device);
            network.BeginFrame(graph);
            const rdg::BufferRef parameters = network.Parameters();
            const rdg::BufferRef matrices = network.Matrices();
            const rdg::BufferRef gradients = network.Gradients();
            EXPECT(matrices.IsValid() == linearAlgebra);
            graph.AddPass(
                "Train", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    b.Read(parameters, rhi::Access::SRVCompute);
                    if (matrices.IsValid())
                    {
                        b.Read(matrices, rhi::Access::SRVCompute);
                    }
                    b.Write(gradients, rhi::Access::UAVCompute);
                },
                [=](rdg::PassContext& ctx)
                {
                    ctx.cmd.SetPipeline(kernel);
                    ctx.cmd.Dispatch(1, 1, 1);
                });
            network.AddOptimizerPasses(graph, {}, 128.0f);
            graph.Execute();
            const std::vector<std::string> commands = device->TakeCapturedCommands();
            EXPECT(count(commands, "VALIDATION") == 0);
            EXPECT(count(commands, "BeginEvent 'Test.Optimize'") == 1);
            const auto conversions = count(commands, "ConvertMatrix ");
            EXPECT(conversions == (linearAlgebra ? (frame == 0 ? 9 : 6) : 0));
        }
        EXPECT(network.GetStep() == 2);
        EXPECT(network.GetParameterCount() == kMlpWeights + 64);
    }
    device->DestroyPipeline(kernel);
    device->WaitIdle();
}
