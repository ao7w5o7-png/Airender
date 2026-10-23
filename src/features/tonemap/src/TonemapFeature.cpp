// Tonemap feature: turns the exposed scene-linear frame output into a display image, in place.
// Add it last. It highlights NaN / Inf (magenta) and negative values (cyan) so numeric bugs in
// earlier features show up as soon as they happen.

#include "air/renderer/RenderFeature.h"
#include "air/renderer/ShaderLoader.h"

#include "TonemapShared.h"

namespace air::tonemap {
namespace {

using rhi::Access;

class TonemapFeature final : public renderer::RenderFeature {
public:
    TonemapFeature() {
        mParameters.Add("operator", &mOperator, "0 clamp, 1 ACES, 2 Khronos PBR Neutral", 0, gpu::kOperatorCount - 1);
        mParameters.Add("whitePoint", &mWhitePoint, "Input value shown as white by the clamp operator", 1e-3f, 1e6f);
        mParameters.Add("highlightInvalid", &mHighlightInvalid, "Show NaN/Inf as magenta and negative values as cyan");
        mParameters.Add("srgb", &mEncodeSrgb, "Apply the sRGB transfer function (off for linear captures)");
    }

    ~TonemapFeature() override {
        if (mDevice) {
            mDevice->DestroyPipeline(mPipeline);
        }
    }

    bool Initialize(const renderer::FeatureInitContext& context) override {
        mDevice = &context.device;
        mPipeline = mDevice->CreateComputePipeline(
            {.shader = renderer::LoadShader(*mDevice, context.shaderDir, "Tonemap.cs", "Main"),
             .pushConstantBytes = sizeof(gpu::TonemapConstants),
             .debugName = "Tonemap"});
        return mPipeline.IsValid();
    }

    void AddPasses(renderer::FrameContext& context) override {
        gpu::TonemapConstants constants{};
        constants.width = context.width;
        constants.height = context.height;
        constants.op = mOperator;
        constants.whitePoint = mWhitePoint;
        constants.flags =
            (mHighlightInvalid ? gpu::kFlagHighlightInvalid : 0u) | (mEncodeSrgb ? gpu::kFlagEncodeSrgb : 0u) |
            (context.blackboard.GetValue(renderer::slots::kOutputIsDisplay) != 0.0 ? gpu::kFlagDisplayInput : 0u);
        const rdg::TextureRef output = context.output;
        const rhi::PipelineHandle pipeline = mPipeline;
        context.graph.AddPass(
            "Tonemap", rdg::PassFlags::Compute, [&](rdg::PassBuilder& b) { b.Write(output, Access::UAVCompute); },
            [=](rdg::PassContext& ctx) mutable {
                constants.image = ctx.UAV(output);
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&constants, sizeof(constants));
                ctx.cmd.Dispatch((constants.width + 7) / 8, (constants.height + 7) / 8, 1);
            });
    }

private:
    rhi::Device* mDevice = nullptr;
    rhi::PipelineHandle mPipeline;
    u32 mOperator = gpu::kOperatorAces;
    float mWhitePoint = 1.0f;
    bool mHighlightInvalid = true;
    bool mEncodeSrgb = true;
};

} // namespace
} // namespace air::tonemap

namespace air::features {

void Register_tonemap(renderer::FeatureRegistry& registry) {
    registry.Register({"tonemap", "Display transform (ACES / PBR Neutral / clamp + sRGB) with NaN highlighting",
                       [] { return std::make_unique<tonemap::TonemapFeature>(); }});
}

} // namespace air::features
