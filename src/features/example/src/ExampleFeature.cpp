// Example feature: the template to copy when adding a new rendering algorithm.
//
// It shows the full shape of a feature in ~100 lines:
//   * parameters exposed to the CLI / reports (scale, opacity, animate)
//   * pipelines created once in Initialize()
//   * a transient texture created inside the render graph
//   * two passes declaring their reads/writes, bindless indices passed through push constants
//   * registration under a name (Register_example)

#include "air/renderer/RenderFeature.h"
#include "air/renderer/ShaderLoader.h"

#include "ExampleShared.h"

namespace air::example
{
namespace
{

using rhi::Access;

class ExampleFeature final : public renderer::RenderFeature
{
public:
    ExampleFeature()
    {
        mParameters.Add("scale", &mScale, "Spatial frequency of the pattern", 0.0f, 1000.0f);
        mParameters.Add("opacity", &mOpacity, "Blend factor over the existing output", 0.0f, 1.0f);
        mParameters.Add("animate", &mAnimate, "Advance the pattern every frame");
    }

    ~ExampleFeature() override
    {
        if (mDevice)
        {
            mDevice->DestroyPipeline(mGenerate);
            mDevice->DestroyPipeline(mComposite);
        }
    }

    bool Initialize(const renderer::FeatureInitContext& context) override
    {
        mDevice = &context.device;
        mGenerate = mDevice->CreateComputePipeline(
            {.shader = renderer::LoadShader(*mDevice, context.shaderDir, "Generate.cs", "Main"),
             .pushConstantBytes = sizeof(GenerateConstants),
             .debugName = "Example.Generate"});
        mComposite = mDevice->CreateComputePipeline(
            {.shader = renderer::LoadShader(*mDevice, context.shaderDir, "Composite.cs", "Main"),
             .pushConstantBytes = sizeof(CompositeConstants),
             .debugName = "Example.Composite"});
        return mGenerate.IsValid() && mComposite.IsValid();
    }

    void AddPasses(renderer::FrameContext& context) override
    {
        rdg::RenderGraph& graph = context.graph;
        const u32 width = context.width;
        const u32 height = context.height;

        // Transient: lives only inside this frame's graph, its memory may be shared with other passes.
        rdg::TextureRef pattern =
            graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                 .width = width,
                                 .height = height,
                                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                 .debugName = "Example.Pattern"});

        // Everything the execute lambdas need is captured by value: they may run on worker threads.
        GenerateConstants generate{};
        generate.width = width;
        generate.height = height;
        generate.time = mAnimate ? float(context.frameIndex) * 0.1f : 0.0f;
        generate.scale = mScale;
        const rhi::PipelineHandle generatePipeline = mGenerate;

        graph.AddPass(
            "Example.Generate", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b) { b.Write(pattern, Access::UAVCompute); },
            [=](rdg::PassContext& ctx) mutable
            {
                generate.outputUav = ctx.UAV(pattern);
                ctx.cmd.SetPipeline(generatePipeline);
                ctx.cmd.PushConstants(&generate, sizeof(generate));
                ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
            });

        const float opacity = mOpacity;
        const rhi::PipelineHandle compositePipeline = mComposite;
        const rdg::TextureRef output = context.output;
        graph.AddPass(
            "Example.Composite", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(pattern, Access::SRVCompute);
                b.Write(output, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx)
            {
                CompositeConstants composite{};
                composite.patternSrv = ctx.SRV(pattern);
                composite.outputUav = ctx.UAV(output);
                composite.width = width;
                composite.height = height;
                composite.opacity = opacity;
                ctx.cmd.SetPipeline(compositePipeline);
                ctx.cmd.PushConstants(&composite, sizeof(composite));
                ctx.cmd.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
            });
    }

private:
    rhi::Device* mDevice = nullptr;
    rhi::PipelineHandle mGenerate;
    rhi::PipelineHandle mComposite;
    float mScale = 20.0f;
    float mOpacity = 1.0f;
    bool mAnimate = true;
};

} // namespace
} // namespace air::example

namespace air::features
{

void Register_example(renderer::FeatureRegistry& registry)
{
    registry.Register({"example", "Template feature: animated pattern composited over the output",
                       [] { return std::make_unique<example::ExampleFeature>(); }});
}

} // namespace air::features
