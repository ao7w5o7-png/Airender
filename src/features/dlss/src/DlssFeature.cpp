// DLSS Ray Reconstruction (NGX "DLSSD") as the Upscale stage: denoises the path tracer's noisy frame
// (FrameContext::sceneColor, at the render resolution) into the output. At native resolution this
// is DLAA; with FrameRunnerDesc::renderScale below 1 it also upscales, which is how RR is meant to
// be used for path tracing: tracing and RR both get cheaper (renderScale 0.667 = Quality,
// 0.58 = Balanced, 0.5 = Performance, 0.333 = Ultra Performance; the NGX mode follows the ratio).
//
// Inputs come from the blackboard (renderer::slots): diffuse/specular albedo, normals with roughness
// in w, linear view depth, motion vectors (render pixels) and the camera jitter. Run the path
// tracer with accumulate=false and spp=1..4. When it passes the frame through (disabled, debug
// view, NGX failure) FrameRunner's bilinear upscale fills the output instead.
//
// D3D12 only, NVIDIA RTX only: Initialize() refuses (with the reason in the log) anywhere else.
// The NGX calls record into the RHI's command list through GetNativeHandle().

#include "air/core/Log.h"
#include "air/renderer/RenderFeature.h"

#include <d3d12.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd_d3d.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <mutex>
#include <string>

namespace air::dlss
{
namespace
{

using rhi::Access;
namespace slots = renderer::slots;

// Identifies this application to NGX (a custom-engine project id).
constexpr const char* kProjectId = "6c1f3a59-2d0e-4c8b-9a7e-3f5b1d2c4e80";
constexpr float kNear = 0.01f;
constexpr float kFar = 10000.0f;

std::string ResultText(NVSDK_NGX_Result result)
{
    return std::format("0x{:08X}", u32(result));
}

// NGX quality mode for an output / render ratio per axis (NGX's own ratios: 1.5, 1.724, 2, 3).
NVSDK_NGX_PerfQuality_Value QualityForRatio(float ratio)
{
    if (ratio < 1.01f)
    {
        return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    if (ratio < 1.6f)
    {
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    }
    if (ratio < 1.85f)
    {
        return NVSDK_NGX_PerfQuality_Value_Balanced;
    }
    if (ratio < 2.5f)
    {
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    }
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

const char* QualityName(NVSDK_NGX_PerfQuality_Value quality)
{
    switch (quality)
    {
    case NVSDK_NGX_PerfQuality_Value_DLAA: return "DLAA";
    case NVSDK_NGX_PerfQuality_Value_MaxQuality: return "Quality";
    case NVSDK_NGX_PerfQuality_Value_Balanced: return "Balanced";
    case NVSDK_NGX_PerfQuality_Value_MaxPerf: return "Performance";
    case NVSDK_NGX_PerfQuality_Value_UltraPerformance: return "Ultra Performance";
    default: return "?";
    }
}

// Row-major matrices for row vectors (v * M), left-handed view space (x right, y up, z forward)
// and D3D clip space with depth in [0, 1].
using Matrix = std::array<float, 16>;

void ViewMatrices(const renderer::Camera& camera, float aspect, Matrix& worldToView, Matrix& viewToClip)
{
    const renderer::float3 r = camera.Right();
    const renderer::float3 u = camera.Up();
    const renderer::float3 f = camera.Forward();
    const renderer::float3 p = camera.position;
    // clang-format off
    worldToView = {r.x, u.x, f.x, 0.0f,
                   r.y, u.y, f.y, 0.0f,
                   r.z, u.z, f.z, 0.0f,
                   -hlsl::dot(r, p), -hlsl::dot(u, p), -hlsl::dot(f, p), 1.0f};
    const float ys = 1.0f / std::tan(camera.verticalFov * 0.5f);
    const float xs = ys / aspect;
    const float zs = kFar / (kFar - kNear);
    viewToClip = {xs,   0.0f, 0.0f,        0.0f,
                  0.0f, ys,   0.0f,        0.0f,
                  0.0f, 0.0f, zs,          1.0f,
                  0.0f, 0.0f, -kNear * zs, 0.0f};
    // clang-format on
}

class DlssFeature final : public renderer::RenderFeature
{
public:
    DlssFeature()
    {
        mParameters.Add("enabled", &mEnabled, "Run Ray Reconstruction (off: pass the noisy frame through)");
        mParameters.Add("preset", &mPreset,
                        "0 default (RR2 = F in SDK 310.9), 4 D (transformer), 5 E (latest transformer), 6 F (RR2)", 0,
                        6);
        mParameters.Add("flipJitter", &mFlipJitter,
                        "Pass the jitter with the wrong sign (diagnostic: a static image then shakes)");
    }

    ~DlssFeature() override
    {
        if (mDevice)
        {
            mDevice->WaitIdle();
        }
        if (mHandle)
        {
            NVSDK_NGX_D3D12_ReleaseFeature(mHandle);
        }
        if (mNgxParameters)
        {
            NVSDK_NGX_D3D12_DestroyParameters(mNgxParameters);
        }
        if (mNgxDevice)
        {
            NVSDK_NGX_D3D12_Shutdown1(mNgxDevice);
        }
    }

    bool Initialize(const renderer::FeatureInitContext& context) override
    {
        if (context.device.GetBackend() != rhi::Backend::D3D12)
        {
            AIR_LOG(Error, "DLSS", "Ray Reconstruction needs the D3D12 backend (got {})",
                    rhi::ToString(context.device.GetBackend()));
            return false;
        }
        mDevice = &context.device;
        mWidth = context.width;
        mHeight = context.height;
        mRenderWidth = context.renderWidth;
        mRenderHeight = context.renderHeight;
        mQuality = QualityForRatio(float(mWidth) / float(std::max(mRenderWidth, 1u)));
        auto* device = static_cast<ID3D12Device*>(context.device.GetNativeObject(rhi::NativeObject::Device));
        NVSDK_NGX_Result result =
            NVSDK_NGX_D3D12_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", L".", device);
        if (NVSDK_NGX_FAILED(result))
        {
            AIR_LOG(Error, "DLSS", "NGX init failed ({}): needs an NVIDIA RTX GPU and a recent driver",
                    ResultText(result));
            return false;
        }
        mNgxDevice = device;
        result = NVSDK_NGX_D3D12_GetCapabilityParameters(&mNgxParameters);
        int available = 0;
        if (NVSDK_NGX_FAILED(result) ||
            NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetI(
                mNgxParameters, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &available)) ||
            !available)
        {
            AIR_LOG(Error, "DLSS", "Ray Reconstruction is not available on '{}'",
                    context.device.GetCapabilities().adapterName);
            return false;
        }
        // Created now rather than on the first frame, so a size NGX rejects is known before any pass
        // claims to write the output.
        auto cmd = mDevice->CreateCommandList(rhi::QueueType::Graphics);
        cmd->Begin();
        bool created = false;
        {
            std::lock_guard lock(mHandleMutex);
            created = CreateHandle(static_cast<ID3D12GraphicsCommandList*>(cmd->GetNativeHandle()), mPreset);
        }
        cmd->End();
        rhi::CommandList* lists[] = {cmd.get()};
        mDevice->WaitForFence(rhi::QueueType::Graphics, mDevice->Submit(rhi::QueueType::Graphics, lists));
        if (!created)
        {
            return false;
        }
        AIR_LOG(Info, "DLSS", "Ray Reconstruction {}: {}x{} -> {}x{}", QualityName(mQuality), mRenderWidth,
                mRenderHeight, mWidth, mHeight);
        return true;
    }

    renderer::FeatureStage GetStage() const override { return renderer::FeatureStage::Upscale; }

    std::vector<std::string_view> ConsumedSlots() const override
    {
        return {slots::kDiffuseAlbedo, slots::kSpecularAlbedo, slots::kNormalRoughness,
                slots::kViewDepth,     slots::kMotionVectors,  slots::kColorBeforeTransparency};
    }

    void AddPasses(renderer::FrameContext& context) override
    {
        {
            std::lock_guard lock(mHandleMutex);
            if (!mEnabled || mCreateFailed)
            {
                return; // pass through: FrameRunner upscales sceneColor if the sizes differ
            }
        }
        const auto diffuse = context.blackboard.FindTexture(slots::kDiffuseAlbedo);
        const auto specular = context.blackboard.FindTexture(slots::kSpecularAlbedo);
        const auto normals = context.blackboard.FindTexture(slots::kNormalRoughness);
        const auto depth = context.blackboard.FindTexture(slots::kViewDepth);
        const auto motion = context.blackboard.FindTexture(slots::kMotionVectors);
        if (!diffuse || !specular || !normals || !depth || !motion)
        {
            if (!mWarnedMissingGuides)
            {
                AIR_LOG(Warning, "DLSS", "no guide textures this frame (enable pathtracer.guides); passing through");
                mWarnedMissingGuides = true;
            }
            return;
        }
        if (context.blackboard.GetValue(slots::kOutputIsDisplay) != 0.0)
        {
            return; // debug views are not radiance
        }

        rdg::RenderGraph& graph = context.graph;
        const rdg::TextureRef color = context.sceneColor;
        // At native resolution sceneColor is the output itself and NGX cannot read and write one
        // resource: it writes a temporary that is copied back. Upscaling writes the output directly.
        const bool upscaling = mRenderWidth != mWidth || mRenderHeight != mHeight;
        const rdg::TextureRef denoised =
            upscaling ? context.output
                      : graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                             .width = mWidth,
                                             .height = mHeight,
                                             .usage = rhi::TextureUsage::UnorderedAccess |
                                                      rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySrc,
                                             .debugName = "DLSS.Denoised"});

        // Optional: the frame before transparent media were laid over it (the path tracer's real-time
        // volumes), which keeps them from ghosting.
        const auto before = context.blackboard.FindTexture(slots::kColorBeforeTransparency);
        struct Inputs
        {
            rdg::TextureRef color, denoised, diffuse, specular, normals, depth, motion, before;
        } in{color, denoised, *diffuse, *specular, *normals, *depth, *motion, before ? *before : rdg::TextureRef{}};
        // The path tracer samples at pixel + 0.5 + jitter; NGX takes the jitter as the shift of the
        // rendered image, which is the opposite. Measured on a static camera: with this sign the
        // output stays within 0.02 px from frame to frame, with the other it moves by up to 0.6 px.
        const float sign = mFlipJitter ? 1.0f : -1.0f;
        const float jitterX = float(context.blackboard.GetValue(slots::kJitterX)) * sign;
        const float jitterY = float(context.blackboard.GetValue(slots::kJitterY)) * sign;
        const bool reset = mFirstFrame || context.blackboard.GetValue(slots::kHistoryReset) != 0.0;
        mFirstFrame = false;
        Matrix worldToView{};
        Matrix viewToClip{};
        ViewMatrices(context.camera, float(mWidth) / float(mHeight), worldToView, viewToClip);
        const u32 preset = mPreset;

        graph.AddPass(
            "DLSS.RayReconstruction", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                for (rdg::TextureRef t : {in.color, in.diffuse, in.specular, in.normals, in.depth, in.motion})
                {
                    b.Read(t, Access::SRVCompute);
                }
                if (in.before.IsValid())
                {
                    b.Read(in.before, Access::SRVCompute);
                }
                b.Write(in.denoised, Access::UAVCompute);
            },
            [this, in, jitterX, jitterY, reset, preset, worldToView, viewToClip](rdg::PassContext& ctx) mutable
            {
                auto* list = static_cast<ID3D12GraphicsCommandList*>(ctx.cmd.GetNativeHandle());
                auto resource = [&](rdg::TextureRef t)
                { return static_cast<ID3D12Resource*>(ctx.device.GetNativeResource(ctx.Get(t))); };
                if (!EnsureHandle(list, preset))
                {
                    return;
                }
                NVSDK_NGX_D3D12_DLSSD_Eval_Params eval = {};
                eval.pInColor = resource(in.color);
                eval.pInOutput = resource(in.denoised);
                eval.pInDiffuseAlbedo = resource(in.diffuse);
                eval.pInSpecularAlbedo = resource(in.specular);
                eval.pInNormals = resource(in.normals);
                eval.pInRoughness = nullptr; // packed in normals.w
                eval.pInDepth = resource(in.depth);
                eval.pInMotionVectors = resource(in.motion);
                eval.pInColorBeforeTransparency = in.before.IsValid() ? resource(in.before) : nullptr;
                eval.InJitterOffsetX = jitterX;
                eval.InJitterOffsetY = jitterY;
                eval.InRenderSubrectDimensions = {mRenderWidth, mRenderHeight};
                eval.InReset = reset ? 1 : 0;
                eval.InMVScaleX = 1.0f;
                eval.InMVScaleY = 1.0f;
                eval.pInWorldToViewMatrix = worldToView.data();
                eval.pInViewToClipMatrix = viewToClip.data();
                eval.InPreExposure = 1.0f; // the path tracer already applied the camera exposure
                eval.InExposureScale = 1.0f;
                const NVSDK_NGX_Result result = NGX_D3D12_EVALUATE_DLSSD_EXT(list, mHandle, mNgxParameters, &eval);
                if (NVSDK_NGX_FAILED(result))
                {
                    AIR_LOG(Error, "DLSS", "evaluate failed ({})", ResultText(result));
                }
            });

        if (upscaling)
        {
            context.outputWritten = true;
            return;
        }
        graph.AddPass(
            "DLSS.Output", rdg::PassFlags::Copy,
            [&](rdg::PassBuilder& b)
            {
                b.Read(in.denoised, Access::CopySrc);
                b.Write(in.color, Access::CopyDst);
            },
            [in](rdg::PassContext& ctx) { ctx.cmd.CopyTexture(ctx.Get(in.color), ctx.Get(in.denoised)); });
        context.outputWritten = true;
    }

    json::Value GetStatus() const override
    {
        std::lock_guard lock(mHandleMutex);
        json::Value status = json::Value::MakeObject();
        status["created"] = mHandle != nullptr;
        status["mode"] = QualityName(mQuality);
        status["render"] = std::format("{}x{}", mRenderWidth, mRenderHeight);
        status["output"] = std::format("{}x{}", mWidth, mHeight);
        if (mCreateFailed)
        {
            status["error"] = mCreateError;
        }
        return status;
    }

private:
    // Recreates the NGX feature inside the frame's command list when the preset changed.
    bool EnsureHandle(ID3D12GraphicsCommandList* list, u32 preset)
    {
        std::lock_guard lock(mHandleMutex);
        if (mHandle && preset == mHandlePreset)
        {
            return true;
        }
        if (mHandle)
        {
            mDevice->WaitIdle();
            NVSDK_NGX_D3D12_ReleaseFeature(mHandle);
            mHandle = nullptr;
        }
        return CreateHandle(list, preset);
    }

    // Records the NGX feature creation into `list`. The caller holds mHandleMutex.
    bool CreateHandle(ID3D12GraphicsCommandList* list, u32 preset)
    {
        if (mCreateFailed)
        {
            return false; // already reported; a resize creates a new feature instance that tries again
        }
        for (const char* hint : {NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
                                 NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
                                 NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
                                 NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
                                 NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
                                 NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality})
        {
            NVSDK_NGX_Parameter_SetUI(mNgxParameters, hint, preset);
        }
        const bool upscaling = mRenderWidth != mWidth || mRenderHeight != mHeight;
        NVSDK_NGX_DLSSD_Create_Params create = {};
        create.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        create.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
        create.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
        create.InWidth = mRenderWidth;
        create.InHeight = mRenderHeight;
        create.InTargetWidth = mWidth;
        create.InTargetHeight = mHeight;
        create.InPerfQualityValue = mQuality;
        // Motion vectors are in render pixels, which NGX calls low resolution when upscaling.
        create.InFeatureCreateFlags =
            NVSDK_NGX_DLSS_Feature_Flags_IsHDR | (upscaling ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0);
        const NVSDK_NGX_Result result = NGX_D3D12_CREATE_DLSSD_EXT(list, 1, 1, &mHandle, mNgxParameters, &create);
        if (NVSDK_NGX_FAILED(result))
        {
            AIR_LOG(Error, "DLSS", "creating Ray Reconstruction {}x{} -> {}x{} failed ({}); passing the frame through",
                    mRenderWidth, mRenderHeight, mWidth, mHeight, ResultText(result));
            mHandle = nullptr;
            // Not mEnabled = false: that is the user's setting, and the editor carries it over to the
            // next feature instance (e.g. after a resize to a size NGX accepts).
            mCreateFailed = true;
            mCreateError = std::format("creating Ray Reconstruction {}x{} -> {}x{} failed ({})", mRenderWidth,
                                       mRenderHeight, mWidth, mHeight, ResultText(result));
            return false;
        }
        mHandlePreset = preset;
        return true;
    }

    rhi::Device* mDevice = nullptr;
    ID3D12Device* mNgxDevice = nullptr;
    NVSDK_NGX_Parameter* mNgxParameters = nullptr;
    NVSDK_NGX_Handle* mHandle = nullptr;
    u32 mHandlePreset = 0;
    mutable std::mutex mHandleMutex;
    u32 mWidth = 0; // output resolution
    u32 mHeight = 0;
    u32 mRenderWidth = 0; // input (render) resolution
    u32 mRenderHeight = 0;
    NVSDK_NGX_PerfQuality_Value mQuality = NVSDK_NGX_PerfQuality_Value_DLAA;
    bool mEnabled = true;
    u32 mPreset = 0;
    bool mFlipJitter = false;
    bool mFirstFrame = true;
    bool mWarnedMissingGuides = false;
    bool mCreateFailed = false; // guarded by mHandleMutex
    std::string mCreateError;
};

} // namespace
} // namespace air::dlss

namespace air::features
{

void Register_dlss(renderer::FeatureRegistry& registry)
{
    registry.Register({"dlss", "NVIDIA DLSS Ray Reconstruction denoiser (D3D12, RTX GPUs)",
                       [] { return std::make_unique<dlss::DlssFeature>(); }});
}

} // namespace air::features
