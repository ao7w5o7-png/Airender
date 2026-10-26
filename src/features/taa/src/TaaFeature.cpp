// Temporal anti-aliasing (docs/architecture.md §8.14), the Upscale stage behind a denoiser of the
// surfaces (nrd): the path tracer jitters its camera rays every frame in real time, which only a
// temporal filter turns into smooth edges instead of crawling ones. It reconstructs the output from the
// frame's jittered samples (and so also upsamples a frame rendered below the output size), reprojects
// its last result with the motion vectors and clips it to the neighbourhood of a fast history that
// follows the frames over a few of them (Taa.hlsli: clipped to the frame's own noisy neighbourhood,
// the history lost the mean of skewed noise).
//
// It stands aside for frames that average themselves (slots::kRealTime 0: accumulation), for debug
// views, and when a feature denoises the whole frame (one that consumes
// slots::kColorBeforeTransparency, DLSS Ray Reconstruction, which also anti-aliases and upscales).

#include "air/core/Log.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/Scene.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/shader/SceneShared.h"

#include "TaaShared.h"

#include <array>
#include <utility>

namespace air::taa
{
namespace
{

using rhi::Access;
namespace slots = renderer::slots;

class TaaFeature final : public renderer::RenderFeature
{
public:
    TaaFeature()
    {
        mParameters.Add("enabled", &mEnabled, "Temporal anti-aliasing of real-time frames");
        mParameters.Add("blend", &mBlend, "Weight of the current frame", 0.02f, 1.0f);
        mParameters.Add("clip", &mClip, "History clipped to the fast history's mean +- this many deviations", 0.5f,
                        4.0f);
        mParameters.Add("fastBlend", &mFastBlend, "Weight of the current frame in the fast history", 0.05f, 1.0f);
    }

    ~TaaFeature() override
    {
        if (mDevice)
        {
            mDevice->WaitIdle();
            for (rhi::TextureHandle& t : mHistory)
            {
                if (t.IsValid())
                {
                    mDevice->DestroyTexture(t);
                }
            }
            for (rhi::TextureHandle& t : mFastHistory)
            {
                if (t.IsValid())
                {
                    mDevice->DestroyTexture(t);
                }
            }
            mPipeline.Destroy(*mDevice);
            mFastPipeline.Destroy(*mDevice);
        }
    }

    bool Initialize(const renderer::FeatureInitContext& context) override
    {
        mDevice = &context.device;
        mWidth = context.width;
        mHeight = context.height;
        mRenderWidth = context.renderWidth;
        mRenderHeight = context.renderHeight;
        mPipeline.Load(context.device, context.shaderDir, "Taa.cs", "TAA", sizeof(gpu::TaaConstants));
        mFastPipeline.Load(context.device, context.shaderDir, "TaaFast.cs", "TAA.Fast", sizeof(gpu::TaaConstants));
        if (!mPipeline.IsAvailable() || !mFastPipeline.IsAvailable())
        {
            AIR_LOG(Error, "TAA", "Taa.cs or TaaFast.cs is missing from {}", context.shaderDir.string());
            return false;
        }
        for (u32 i = 0; i < 2; ++i)
        {
            mHistory[i] =
                context.device.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                              .width = mWidth,
                                              .height = mHeight,
                                              .usage = rhi::TextureUsage::UnorderedAccess |
                                                       rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySrc,
                                              .debugName = i == 0 ? "TAA.History[0]" : "TAA.History[1]"});
            mFastHistory[i] = context.device.CreateTexture(
                {.format = rhi::Format::RGBA16_FLOAT,
                 .width = mWidth,
                 .height = mHeight,
                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                 .debugName = i == 0 ? "TAA.FastHistory[0]" : "TAA.FastHistory[1]"});
        }
        return true;
    }

    renderer::FeatureStage GetStage() const override { return renderer::FeatureStage::Upscale; }

    std::vector<std::string_view> ConsumedSlots() const override
    {
        return mEnabled ? std::vector<std::string_view>{slots::kMotionVectors} : std::vector<std::string_view>{};
    }

    json::Value GetStatus() const override
    {
        json::Value status = json::Value::MakeObject();
        status["filtered"] = mFiltered;
        status["cameraMoved"] = mCameraMoved;
        return status;
    }

    void AddPasses(renderer::FrameContext& context) override
    {
        mFiltered = false;
        const auto motion = context.blackboard.FindTexture(slots::kMotionVectors);
        if (!mEnabled || context.outputWritten || !motion || context.blackboard.GetValue(slots::kRealTime) == 0.0 ||
            context.blackboard.GetValue(slots::kOutputIsDisplay) != 0.0 ||
            context.Consumes(slots::kColorBeforeTransparency))
        {
            mValid = false; // the next filtered frame starts over
            return;
        }
        mFiltered = true;
        rdg::RenderGraph& graph = context.graph;
        const bool history = mValid && context.blackboard.GetValue(slots::kHistoryReset) == 0.0;
        const u32 current = mCurrent;
        const rdg::TextureRef result = graph.ImportTexture(mHistory[current], mAccess[current], Access::CopySrc);
        mAccess[current] = Access::CopySrc;
        const rdg::TextureRef fastResult =
            graph.ImportTexture(mFastHistory[current], mFastAccess[current], Access::SRVCompute);
        mFastAccess[current] = Access::SRVCompute;
        rdg::TextureRef past;
        rdg::TextureRef fastPast;
        if (history)
        {
            past = graph.ImportTexture(mHistory[current ^ 1u], mAccess[current ^ 1u], Access::CopySrc);
            mAccess[current ^ 1u] = Access::CopySrc;
            fastPast = graph.ImportTexture(mFastHistory[current ^ 1u], mFastAccess[current ^ 1u], Access::SRVCompute);
            mFastAccess[current ^ 1u] = Access::SRVCompute;
        }
        mCurrent ^= 1u;
        mValid = true;

        gpu::TaaConstants c{};
        c.renderWidth = mRenderWidth;
        c.renderHeight = mRenderHeight;
        c.width = mWidth;
        c.height = mHeight;
        c.jitterX = float(context.blackboard.GetValue(slots::kJitterX));
        c.jitterY = float(context.blackboard.GetValue(slots::kJitterY));
        c.blend = mBlend;
        c.fastBlend = mFastBlend;
        c.clip = mClip;
        c.frame = mFrame++;
        mCameraMoved = history && !(context.camera == mPreviousCamera);
        c.cameraMoved = mCameraMoved ? 1.0f : 0.0f;
        mPreviousCamera = context.camera;
        c.history = air::gpu::kInvalidIndex;
        c.fastHistory = air::gpu::kInvalidIndex;
        const rdg::TextureRef color = context.sceneColor;
        const rdg::TextureRef motionVectors = *motion;
        const rhi::PipelineHandle fastPipeline = mFastPipeline.Get(*mDevice);
        graph.AddPass(
            "TAA.Fast", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(color, Access::SRVCompute);
                b.Read(motionVectors, Access::SRVCompute);
                if (fastPast.IsValid())
                {
                    b.Read(fastPast, Access::SRVCompute);
                }
                b.Write(fastResult, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                c.color = ctx.SRV(color);
                c.motion = ctx.SRV(motionVectors);
                if (fastPast.IsValid())
                {
                    c.fastHistory = ctx.SRV(fastPast);
                }
                c.fastResult = ctx.UAV(fastResult);
                ctx.cmd.SetPipeline(fastPipeline);
                ctx.cmd.PushConstants(&c, sizeof(c));
                ctx.cmd.Dispatch((c.width + 7) / 8, (c.height + 7) / 8, 1);
            });
        const rhi::PipelineHandle pipeline = mPipeline.Get(*mDevice);
        graph.AddPass(
            "TAA", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                b.Read(color, Access::SRVCompute);
                b.Read(motionVectors, Access::SRVCompute);
                b.Read(fastResult, Access::SRVCompute);
                if (past.IsValid())
                {
                    b.Read(past, Access::SRVCompute);
                }
                b.Write(result, Access::UAVCompute);
            },
            [=](rdg::PassContext& ctx) mutable
            {
                c.color = ctx.SRV(color);
                c.motion = ctx.SRV(motionVectors);
                c.fastResult = ctx.SRV(fastResult);
                if (past.IsValid())
                {
                    c.history = ctx.SRV(past);
                }
                c.result = ctx.UAV(result);
                ctx.cmd.SetPipeline(pipeline);
                ctx.cmd.PushConstants(&c, sizeof(c));
                ctx.cmd.Dispatch((c.width + 7) / 8, (c.height + 7) / 8, 1);
            });
        // The result is next frame's history; the output gets a copy (at the render resolution it is
        // the frame this pass reads).
        const rdg::TextureRef output = context.output;
        graph.AddPass(
            "TAA.Output", rdg::PassFlags::Copy,
            [&](rdg::PassBuilder& b)
            {
                b.Read(result, Access::CopySrc);
                b.Write(output, Access::CopyDst);
            },
            [=](rdg::PassContext& ctx) { ctx.cmd.CopyTexture(ctx.Get(output), ctx.Get(result)); });
        context.outputWritten = true;
    }

private:
    bool mEnabled = true;
    float mBlend = 0.1f;
    float mClip = 1.25f;
    float mFastBlend = 0.25f;

    rhi::Device* mDevice = nullptr;
    u32 mWidth = 0;
    u32 mHeight = 0;
    u32 mRenderWidth = 0;
    u32 mRenderHeight = 0;
    renderer::LazyComputePipeline mPipeline;
    renderer::LazyComputePipeline mFastPipeline;
    std::array<rhi::TextureHandle, 2> mHistory;
    std::array<Access, 2> mAccess{Access::None, Access::None};
    std::array<rhi::TextureHandle, 2> mFastHistory;
    std::array<Access, 2> mFastAccess{Access::None, Access::None};
    u32 mCurrent = 0;
    u32 mFrame = 0;
    renderer::Camera mPreviousCamera;
    bool mValid = false;
    bool mFiltered = false;
    bool mCameraMoved = false;
};

} // namespace
} // namespace air::taa

namespace air::features
{
void Register_taa(renderer::FeatureRegistry& registry)
{
    registry.Register({"taa", "Temporal anti-aliasing and upsampling of real-time frames",
                       [] { return std::make_unique<taa::TaaFeature>(); }});
}
} // namespace air::features
