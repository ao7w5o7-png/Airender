// DLSS Ray Reconstruction (NGX "DLSSD") as a feature: replaces the path tracer's noisy frame with
// the denoised one, at the same resolution (DLAA-style, no upscaling).
//
// Inputs come from the blackboard (renderer::slots): the frame output as noisy color (scene-linear,
// already exposed), diffuse/specular albedo, normals with roughness in w, linear view depth,
// motion vectors and the camera jitter. Run the path tracer with accumulate=false and spp=1..4.
// Order: pathtracer, dlss, tonemap.
//
// D3D12 only, NVIDIA RTX only: Initialize() refuses (with the reason in the log) anywhere else.
// The NGX calls record into the RHI's command list through GetNativeHandle().

#include "air/core/Log.h"
#include "air/renderer/RenderFeature.h"

#include <d3d12.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_d3d.h>

#include <cmath>
#include <cstring>
#include <mutex>

namespace air::dlss {
namespace {

using rhi::Access;
namespace slots = renderer::slots;

// Identifies this application to NGX (a custom-engine project id).
constexpr const char* kProjectId = "6c1f3a59-2d0e-4c8b-9a7e-3f5b1d2c4e80";
constexpr float kNear = 0.01f;
constexpr float kFar = 10000.0f;

std::string ResultText(NVSDK_NGX_Result result) {
    return std::format("0x{:08X}", u32(result));
}

// Row-major matrices for row vectors (v * M), left-handed view space (x right, y up, z forward)
// and D3D clip space with depth in [0, 1].
void ViewMatrices(const renderer::Camera& camera, float aspect, float worldToView[16], float viewToClip[16]) {
    const renderer::float3 r = camera.Right();
    const renderer::float3 u = camera.Up();
    const renderer::float3 f = camera.Forward();
    const renderer::float3 p = camera.position;
    const float view[16] = {r.x,
                            u.x,
                            f.x,
                            0.0f,
                            r.y,
                            u.y,
                            f.y,
                            0.0f,
                            r.z,
                            u.z,
                            f.z,
                            0.0f,
                            -hlsl::dot(r, p),
                            -hlsl::dot(u, p),
                            -hlsl::dot(f, p),
                            1.0f};
    const float ys = 1.0f / std::tan(camera.verticalFov * 0.5f);
    const float xs = ys / aspect;
    const float zs = kFar / (kFar - kNear);
    const float proj[16] = {xs,   0.0f, 0.0f, 0.0f, 0.0f, ys,   0.0f,        0.0f,
                            0.0f, 0.0f, zs,   1.0f, 0.0f, 0.0f, -kNear * zs, 0.0f};
    std::memcpy(worldToView, view, sizeof(view));
    std::memcpy(viewToClip, proj, sizeof(proj));
}

class DlssFeature final : public renderer::RenderFeature {
public:
    DlssFeature() {
        mParameters.Add("enabled", &mEnabled, "Run Ray Reconstruction (off: pass the noisy frame through)");
        mParameters.Add("preset", &mPreset, "0 default, 4 D (transformer), 5 E (latest transformer)", 0, 5);
        mParameters.Add("flipJitter", &mFlipJitter, "Negate the jitter passed to NGX (sign convention check)");
    }

    ~DlssFeature() override {
        if (mDevice) {
            mDevice->WaitIdle();
        }
        if (mHandle) {
            NVSDK_NGX_D3D12_ReleaseFeature(mHandle);
        }
        if (mNgxParameters) {
            NVSDK_NGX_D3D12_DestroyParameters(mNgxParameters);
        }
        if (mNgxDevice) {
            NVSDK_NGX_D3D12_Shutdown1(mNgxDevice);
        }
    }

    bool Initialize(const renderer::FeatureInitContext& context) override {
        if (context.device.GetBackend() != rhi::Backend::D3D12) {
            AIR_LOG(Error, "DLSS", "Ray Reconstruction needs the D3D12 backend (got {})",
                    rhi::ToString(context.device.GetBackend()));
            return false;
        }
        mDevice = &context.device;
        mWidth = context.width;
        mHeight = context.height;
        auto* device = static_cast<ID3D12Device*>(context.device.GetNativeObject(rhi::NativeObject::Device));
        NVSDK_NGX_Result result =
            NVSDK_NGX_D3D12_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", L".", device);
        if (NVSDK_NGX_FAILED(result)) {
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
            !available) {
            AIR_LOG(Error, "DLSS", "Ray Reconstruction is not available on '{}'",
                    context.device.GetCapabilities().adapterName);
            return false;
        }
        AIR_LOG(Info, "DLSS", "Ray Reconstruction available, {}x{}", mWidth, mHeight);
        return true;
    }

    void AddPasses(renderer::FrameContext& context) override {
        if (!mEnabled) {
            return;
        }
        const auto diffuse = context.blackboard.FindTexture(slots::kDiffuseAlbedo);
        const auto specular = context.blackboard.FindTexture(slots::kSpecularAlbedo);
        const auto normals = context.blackboard.FindTexture(slots::kNormalRoughness);
        const auto depth = context.blackboard.FindTexture(slots::kViewDepth);
        const auto motion = context.blackboard.FindTexture(slots::kMotionVectors);
        if (!diffuse || !specular || !normals || !depth || !motion) {
            if (!mWarnedMissingGuides) {
                AIR_LOG(Warning, "DLSS", "no guide textures this frame (enable pathtracer.guides); passing through");
                mWarnedMissingGuides = true;
            }
            return;
        }
        if (context.blackboard.GetValue(slots::kOutputIsDisplay) != 0.0) {
            return; // debug views are not radiance
        }

        rdg::RenderGraph& graph = context.graph;
        const rdg::TextureRef color = context.output;
        const rdg::TextureRef denoised =
            graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                 .width = mWidth,
                                 .height = mHeight,
                                 .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource |
                                          rhi::TextureUsage::CopySrc,
                                 .debugName = "DLSS.Denoised"});

        struct Inputs {
            rdg::TextureRef color, denoised, diffuse, specular, normals, depth, motion;
        } in{color, denoised, *diffuse, *specular, *normals, *depth, *motion};
        const float sign = mFlipJitter ? -1.0f : 1.0f;
        const float jitterX = float(context.blackboard.GetValue(slots::kJitterX)) * sign;
        const float jitterY = float(context.blackboard.GetValue(slots::kJitterY)) * sign;
        const bool reset = mFirstFrame || context.blackboard.GetValue(slots::kHistoryReset) != 0.0;
        mFirstFrame = false;
        float worldToView[16], viewToClip[16];
        ViewMatrices(context.camera, float(mWidth) / float(mHeight), worldToView, viewToClip);
        const u32 preset = mPreset;

        graph.AddPass(
            "DLSS.RayReconstruction", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b) {
                for (rdg::TextureRef t : {in.color, in.diffuse, in.specular, in.normals, in.depth, in.motion}) {
                    b.Read(t, Access::SRVCompute);
                }
                b.Write(in.denoised, Access::UAVCompute);
            },
            [this, in, jitterX, jitterY, reset, preset, worldToView = std::to_array(worldToView),
             viewToClip = std::to_array(viewToClip)](rdg::PassContext& ctx) mutable {
                auto* list = static_cast<ID3D12GraphicsCommandList*>(ctx.cmd.GetNativeHandle());
                auto resource = [&](rdg::TextureRef t) {
                    return static_cast<ID3D12Resource*>(ctx.device.GetNativeResource(ctx.Get(t)));
                };
                if (!EnsureHandle(list, preset)) {
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
                eval.InJitterOffsetX = jitterX;
                eval.InJitterOffsetY = jitterY;
                eval.InRenderSubrectDimensions = {mWidth, mHeight};
                eval.InReset = reset ? 1 : 0;
                eval.InMVScaleX = 1.0f;
                eval.InMVScaleY = 1.0f;
                eval.pInWorldToViewMatrix = worldToView.data();
                eval.pInViewToClipMatrix = viewToClip.data();
                eval.InPreExposure = 1.0f; // the path tracer already applied the camera exposure
                eval.InExposureScale = 1.0f;
                const NVSDK_NGX_Result result = NGX_D3D12_EVALUATE_DLSSD_EXT(list, mHandle, mNgxParameters, &eval);
                if (NVSDK_NGX_FAILED(result)) {
                    AIR_LOG(Error, "DLSS", "evaluate failed ({})", ResultText(result));
                }
            });

        graph.AddPass(
            "DLSS.Output", rdg::PassFlags::Copy,
            [&](rdg::PassBuilder& b) {
                b.Read(in.denoised, Access::CopySrc);
                b.Write(in.color, Access::CopyDst);
            },
            [in](rdg::PassContext& ctx) { ctx.cmd.CopyTexture(ctx.Get(in.color), ctx.Get(in.denoised)); });
    }

    json::Value GetStatus() const override {
        json::Value status = json::Value::MakeObject();
        status["created"] = mHandle != nullptr;
        return status;
    }

private:
    // The NGX feature is created inside a command list on first use (and again when the preset changes).
    bool EnsureHandle(ID3D12GraphicsCommandList* list, u32 preset) {
        std::lock_guard lock(mHandleMutex);
        if (mHandle && preset == mHandlePreset) {
            return true;
        }
        if (mHandle) {
            mDevice->WaitIdle();
            NVSDK_NGX_D3D12_ReleaseFeature(mHandle);
            mHandle = nullptr;
        }
        NVSDK_NGX_Parameter_SetUI(mNgxParameters, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
                                  preset);
        NVSDK_NGX_DLSSD_Create_Params create = {};
        create.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        create.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
        create.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
        create.InWidth = mWidth;
        create.InHeight = mHeight;
        create.InTargetWidth = mWidth;
        create.InTargetHeight = mHeight;
        create.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
        create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
        const NVSDK_NGX_Result result = NGX_D3D12_CREATE_DLSSD_EXT(list, 1, 1, &mHandle, mNgxParameters, &create);
        if (NVSDK_NGX_FAILED(result)) {
            AIR_LOG(Error, "DLSS", "creating Ray Reconstruction failed ({})", ResultText(result));
            mHandle = nullptr;
            mEnabled = false;
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
    std::mutex mHandleMutex;
    u32 mWidth = 0;
    u32 mHeight = 0;
    bool mEnabled = true;
    u32 mPreset = 0;
    bool mFlipJitter = false;
    bool mFirstFrame = true;
    bool mWarnedMissingGuides = false;
};

} // namespace
} // namespace air::dlss

namespace air::features {

void Register_dlss(renderer::FeatureRegistry& registry) {
    registry.Register({"dlss", "NVIDIA DLSS Ray Reconstruction denoiser (D3D12, RTX GPUs)",
                       [] { return std::make_unique<dlss::DlssFeature>(); }});
}

} // namespace air::features
