// NVIDIA Real-time Denoisers (NRD, docs/architecture.md §8.14) as the path tracer's default denoiser
// in real time. A Scene stage feature after the path tracer, at the render resolution:
//   NRD.Prepare    the path tracer's signals (renderer::slots::kDiffuseSignal, kSpecularSignal: the
//                  light the first surface reflects per lobe) divided by the albedos, and its guides,
//                  in NRD's encodings (Prepare.hlsl, with NRD.hlsli)
//   NRD.Denoise    NRD's own dispatches (RELAX or REBLUR, diffuse and specular), recorded here with
//                  D3D12 into the RHI's command list: a root signature and pipeline per NRD pipeline,
//                  NRD's textures, a shader-visible descriptor heap and an upload buffer for its
//                  constants, both rings over the frames in flight
//   NRD.Composite  the frame with the denoised light in place of the noisy (Composite.hlsl)
// It steps aside when a feature that denoises the whole frame runs (one that consumes
// slots::kColorBeforeTransparency, DLSS Ray Reconstruction).
//
// D3D12 only: Initialize() refuses elsewhere.

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/RenderFeature.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/shader/SceneShared.h"

#include "NrdShared.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <NRD.h>

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace air::nrdf
{
namespace
{

using Microsoft::WRL::ComPtr;
using rhi::Access;
namespace slots = renderer::slots;

constexpr nrd::Identifier kIdentifier = 1;
// The path tracer writes 65504 for the sky's view depth; surfaces are nearer.
constexpr float kDenoisingRange = 60000.0f;
constexpr u32 kMaxDispatches = 64; // per frame
constexpr float kNear = 0.01f;
constexpr float kFar = 10000.0f;

// Row-major matrices for row vectors (v * M), left-handed view space (x right, y up, z forward along
// the view depth the path tracer writes), D3D clip space: the same memory as NRD's column-major
// matrices for column vectors.
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

DXGI_FORMAT ToDxgi(nrd::Format format)
{
    switch (format)
    {
    case nrd::Format::R8_UNORM: return DXGI_FORMAT_R8_UNORM;
    case nrd::Format::R8_SNORM: return DXGI_FORMAT_R8_SNORM;
    case nrd::Format::R8_UINT: return DXGI_FORMAT_R8_UINT;
    case nrd::Format::R8_SINT: return DXGI_FORMAT_R8_SINT;
    case nrd::Format::RG8_UNORM: return DXGI_FORMAT_R8G8_UNORM;
    case nrd::Format::RG8_SNORM: return DXGI_FORMAT_R8G8_SNORM;
    case nrd::Format::RG8_UINT: return DXGI_FORMAT_R8G8_UINT;
    case nrd::Format::RG8_SINT: return DXGI_FORMAT_R8G8_SINT;
    case nrd::Format::RGBA8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case nrd::Format::RGBA8_SNORM: return DXGI_FORMAT_R8G8B8A8_SNORM;
    case nrd::Format::RGBA8_UINT: return DXGI_FORMAT_R8G8B8A8_UINT;
    case nrd::Format::RGBA8_SINT: return DXGI_FORMAT_R8G8B8A8_SINT;
    case nrd::Format::RGBA8_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case nrd::Format::R16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case nrd::Format::R16_SNORM: return DXGI_FORMAT_R16_SNORM;
    case nrd::Format::R16_UINT: return DXGI_FORMAT_R16_UINT;
    case nrd::Format::R16_SINT: return DXGI_FORMAT_R16_SINT;
    case nrd::Format::R16_SFLOAT: return DXGI_FORMAT_R16_FLOAT;
    case nrd::Format::RG16_UNORM: return DXGI_FORMAT_R16G16_UNORM;
    case nrd::Format::RG16_SNORM: return DXGI_FORMAT_R16G16_SNORM;
    case nrd::Format::RG16_UINT: return DXGI_FORMAT_R16G16_UINT;
    case nrd::Format::RG16_SINT: return DXGI_FORMAT_R16G16_SINT;
    case nrd::Format::RG16_SFLOAT: return DXGI_FORMAT_R16G16_FLOAT;
    case nrd::Format::RGBA16_UNORM: return DXGI_FORMAT_R16G16B16A16_UNORM;
    case nrd::Format::RGBA16_SNORM: return DXGI_FORMAT_R16G16B16A16_SNORM;
    case nrd::Format::RGBA16_UINT: return DXGI_FORMAT_R16G16B16A16_UINT;
    case nrd::Format::RGBA16_SINT: return DXGI_FORMAT_R16G16B16A16_SINT;
    case nrd::Format::RGBA16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case nrd::Format::R32_UINT: return DXGI_FORMAT_R32_UINT;
    case nrd::Format::R32_SINT: return DXGI_FORMAT_R32_SINT;
    case nrd::Format::R32_SFLOAT: return DXGI_FORMAT_R32_FLOAT;
    case nrd::Format::RG32_UINT: return DXGI_FORMAT_R32G32_UINT;
    case nrd::Format::RG32_SINT: return DXGI_FORMAT_R32G32_SINT;
    case nrd::Format::RG32_SFLOAT: return DXGI_FORMAT_R32G32_FLOAT;
    case nrd::Format::RGB32_UINT: return DXGI_FORMAT_R32G32B32_UINT;
    case nrd::Format::RGB32_SINT: return DXGI_FORMAT_R32G32B32_SINT;
    case nrd::Format::RGB32_SFLOAT: return DXGI_FORMAT_R32G32B32_FLOAT;
    case nrd::Format::RGBA32_UINT: return DXGI_FORMAT_R32G32B32A32_UINT;
    case nrd::Format::RGBA32_SINT: return DXGI_FORMAT_R32G32B32A32_SINT;
    case nrd::Format::RGBA32_SFLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case nrd::Format::R10_G10_B10_A2_UNORM: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case nrd::Format::R10_G10_B10_A2_UINT: return DXGI_FORMAT_R10G10B10A2_UINT;
    case nrd::Format::R11_G11_B10_UFLOAT: return DXGI_FORMAT_R11G11B10_FLOAT;
    case nrd::Format::R9_G9_B9_E5_UFLOAT: return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

u32 AlignUp(u32 value, u32 alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// The textures of one dispatch the RHI's render graph owns, in the states it declared for the pass.
struct Inputs
{
    rdg::TextureRef motion, normalRoughness, viewDepth, diffuse, specular, outDiffuse, outSpecular;
};

class NrdFeature final : public renderer::RenderFeature
{
public:
    NrdFeature()
    {
        mParameters.Add("enabled", &mEnabled, "Denoise the path tracer's frame (in real time: accumulate=false)");
        mParameters.Add("denoiser", &mDenoiser, "0 RELAX, 1 REBLUR (diffuse and specular)", 0, 1);
        mParameters.Add("historyFrames", &mHistoryFrames, "Frames NRD accumulates at most (30: half a second)", 1, 63);
        mParameters.Add("antiFirefly", &mAntiFirefly,
                        "NRD's firefly suppression (clamps bright samples: 4% of the sphere field's light)");
        mParameters.Add("view", &mView, "0 the frame, 1 the global illumination alone (the indirect light, denoised)",
                        0, 1);
    }

    ~NrdFeature() override
    {
        if (mDevice)
        {
            mDevice->WaitIdle();
            mPrepare.Destroy(*mDevice);
            mComposite.Destroy(*mDevice);
        }
        DestroyInstance();
    }

    bool Initialize(const renderer::FeatureInitContext& context) override
    {
        if (context.device.GetBackend() != rhi::Backend::D3D12)
        {
            AIR_LOG(Warning, "NRD", "the nrd feature needs the D3D12 backend (got {})",
                    rhi::ToString(context.device.GetBackend()));
            return false;
        }
        mDevice = &context.device;
        mD3d = static_cast<ID3D12Device*>(context.device.GetNativeObject(rhi::NativeObject::Device));
        mWidth = context.renderWidth;
        mHeight = context.renderHeight;
        mSlots = context.device.GetFramesInFlight() + 1;
        mPrepare.Load(context.device, context.shaderDir, "Prepare.cs", "NRD.Prepare", sizeof(gpu::NrdPrepareConstants));
        mComposite.Load(context.device, context.shaderDir, "Composite.cs", "NRD.Composite",
                        sizeof(gpu::NrdCompositeConstants));
        if (!mPrepare.IsAvailable() || !mComposite.IsAvailable())
        {
            AIR_LOG(Error, "NRD", "its shaders are missing from {}", context.shaderDir.string());
            return false;
        }
        return CreateInstance(mDenoiser);
    }

    renderer::FeatureStage GetStage() const override { return renderer::FeatureStage::Scene; }

    std::vector<std::string_view> ConsumedSlots() const override
    {
        if (!mEnabled)
        {
            return {};
        }
        return {slots::kDiffuseAlbedo, slots::kSpecularAlbedo, slots::kNormalRoughness, slots::kViewDepth,
                slots::kMotionVectors, slots::kDiffuseSignal,  slots::kSpecularSignal,  slots::kSurfaceTransmittance};
    }

    json::Value GetStatus() const override
    {
        json::Value status = json::Value::MakeObject();
        status["denoised"] = mDenoised;
        status["denoiser"] = mInstanceKind == gpu::kNrdReblur ? "REBLUR" : "RELAX";
        return status;
    }

    void AddPasses(renderer::FrameContext& context) override
    {
        mDenoised = false;
        if (!mEnabled || !mInstance || context.blackboard.GetValue(slots::kOutputIsDisplay) != 0.0)
        {
            return;
        }
        if (context.Consumes(slots::kColorBeforeTransparency))
        {
            if (!std::exchange(mWarnedOtherDenoiser, true))
            {
                AIR_LOG(Info, "NRD", "another feature denoises the whole frame: NRD steps aside");
            }
            return;
        }
        const auto diffuseSignal = context.blackboard.FindTexture(slots::kDiffuseSignal);
        const auto specularSignal = context.blackboard.FindTexture(slots::kSpecularSignal);
        const auto diffuseAlbedo = context.blackboard.FindTexture(slots::kDiffuseAlbedo);
        const auto specularAlbedo = context.blackboard.FindTexture(slots::kSpecularAlbedo);
        const auto normalRoughness = context.blackboard.FindTexture(slots::kNormalRoughness);
        const auto viewDepth = context.blackboard.FindTexture(slots::kViewDepth);
        const auto motion = context.blackboard.FindTexture(slots::kMotionVectors);
        if (!diffuseSignal || !specularSignal || !diffuseAlbedo || !specularAlbedo || !normalRoughness || !viewDepth ||
            !motion)
        {
            return; // nothing to denoise this frame (accumulating, or a debug view)
        }
        const auto surfaceTransmittance = context.blackboard.FindTexture(slots::kSurfaceTransmittance);
        if (mDenoiser != mInstanceKind && !CreateInstance(mDenoiser))
        {
            return;
        }
        mDenoised = true;

        rdg::RenderGraph& graph = context.graph;
        auto make = [&](rhi::Format format, const char* name)
        {
            return graph.CreateTexture({.format = format,
                                        .width = mWidth,
                                        .height = mHeight,
                                        .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                        .debugName = name});
        };
        Inputs in;
        in.motion = *motion;
        in.viewDepth = *viewDepth;
        in.normalRoughness = make(rhi::Format::R10G10B10A2_UNORM, "NRD.NormalRoughness");
        in.diffuse = make(rhi::Format::RGBA16_FLOAT, "NRD.InDiffuse");
        in.specular = make(rhi::Format::RGBA16_FLOAT, "NRD.InSpecular");
        in.outDiffuse = make(rhi::Format::RGBA16_FLOAT, "NRD.OutDiffuse");
        in.outSpecular = make(rhi::Format::RGBA16_FLOAT, "NRD.OutSpecular");
        const u32 groupsX = (mWidth + 7) / 8;
        const u32 groupsY = (mHeight + 7) / 8;
        const nrd::ReblurHitDistanceParameters hitDistance;

        {
            gpu::NrdPrepareConstants c{};
            c.width = mWidth;
            c.height = mHeight;
            c.denoiser = mInstanceKind;
            c.hitDistanceA = hitDistance.A;
            c.hitDistanceB = hitDistance.B;
            c.hitDistanceC = hitDistance.C;
            c.denoisingRange = kDenoisingRange;
            const rhi::PipelineHandle pipeline = mPrepare.Get(*mDevice);
            const rdg::TextureRef signals[] = {*diffuseSignal,  *specularSignal,  *diffuseAlbedo,
                                               *specularAlbedo, *normalRoughness, *viewDepth};
            graph.AddPass(
                "NRD.Prepare", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    for (rdg::TextureRef t : signals)
                    {
                        b.Read(t, Access::SRVCompute);
                    }
                    for (rdg::TextureRef t : {in.diffuse, in.specular, in.normalRoughness})
                    {
                        b.Write(t, Access::UAVCompute);
                    }
                },
                [=](rdg::PassContext& ctx) mutable
                {
                    c.diffuseSignal = ctx.SRV(signals[0]);
                    c.specularSignal = ctx.SRV(signals[1]);
                    c.diffuseAlbedo = ctx.SRV(signals[2]);
                    c.specularAlbedo = ctx.SRV(signals[3]);
                    c.normalRoughness = ctx.SRV(signals[4]);
                    c.viewDepth = ctx.SRV(signals[5]);
                    c.inDiffuse = ctx.UAV(in.diffuse);
                    c.inSpecular = ctx.UAV(in.specular);
                    c.inNormalRoughness = ctx.UAV(in.normalRoughness);
                    ctx.cmd.SetPipeline(pipeline);
                    ctx.cmd.PushConstants(&c, sizeof(c));
                    ctx.cmd.Dispatch(groupsX, groupsY, 1);
                });
        }

        // This frame's settings, taken now (the denoise pass records later, maybe on another thread).
        nrd::CommonSettings common{};
        Matrix worldToView{};
        Matrix viewToClip{};
        ViewMatrices(context.camera, float(mWidth) / float(mHeight), worldToView, viewToClip);
        const bool reset = mFirstFrame || context.blackboard.GetValue(slots::kHistoryReset) != 0.0;
        mFirstFrame = false;
        if (reset)
        {
            mPreviousWorldToView = worldToView;
            mPreviousViewToClip = viewToClip;
        }
        std::memcpy(common.viewToClipMatrix, viewToClip.data(), sizeof(viewToClip));
        std::memcpy(common.viewToClipMatrixPrev, mPreviousViewToClip.data(), sizeof(viewToClip));
        std::memcpy(common.worldToViewMatrix, worldToView.data(), sizeof(worldToView));
        std::memcpy(common.worldToViewMatrixPrev, mPreviousWorldToView.data(), sizeof(worldToView));
        mPreviousWorldToView = worldToView;
        mPreviousViewToClip = viewToClip;
        // The path tracer's motion: previous minus current position in pixels.
        common.motionVectorScale[0] = 1.0f / float(mWidth);
        common.motionVectorScale[1] = 1.0f / float(mHeight);
        common.motionVectorScale[2] = 0.0f;
        // It samples pixel + 0.5 + jitter.
        const float jitter[2] = {float(context.blackboard.GetValue(slots::kJitterX)),
                                 float(context.blackboard.GetValue(slots::kJitterY))};
        common.cameraJitter[0] = jitter[0];
        common.cameraJitter[1] = jitter[1];
        common.cameraJitterPrev[0] = reset ? jitter[0] : mPreviousJitter[0];
        common.cameraJitterPrev[1] = reset ? jitter[1] : mPreviousJitter[1];
        mPreviousJitter[0] = jitter[0];
        mPreviousJitter[1] = jitter[1];
        common.resourceSize[0] = common.resourceSizePrev[0] = common.rectSize[0] = common.rectSizePrev[0] = u16(mWidth);
        common.resourceSize[1] = common.resourceSizePrev[1] = common.rectSize[1] = common.rectSizePrev[1] =
            u16(mHeight);
        common.denoisingRange = kDenoisingRange;
        common.frameIndex = mFrameIndex++;
        common.accumulationMode = reset ? nrd::AccumulationMode::RESTART : nrd::AccumulationMode::CONTINUE;
        nrd::RelaxSettings relax;
        relax.diffuseMaxAccumulatedFrameNum = mHistoryFrames;
        relax.specularMaxAccumulatedFrameNum = mHistoryFrames;
        relax.diffuseMaxFastAccumulatedFrameNum = std::max(mHistoryFrames / 5, 1u);
        relax.specularMaxFastAccumulatedFrameNum = std::max(mHistoryFrames / 5, 1u);
        relax.enableAntiFirefly = mAntiFirefly;
        nrd::ReblurSettings reblur;
        reblur.maxAccumulatedFrameNum = mHistoryFrames;
        reblur.maxFastAccumulatedFrameNum = std::max(mHistoryFrames / 5, 1u);
        reblur.enableAntiFirefly = mAntiFirefly;
        reblur.hitDistanceParameters = hitDistance;

        graph.AddPass(
            "NRD.Denoise", rdg::PassFlags::Compute,
            [&](rdg::PassBuilder& b)
            {
                for (rdg::TextureRef t : {in.motion, in.viewDepth, in.normalRoughness, in.diffuse, in.specular})
                {
                    b.Read(t, Access::SRVCompute);
                }
                b.Write(in.outDiffuse, Access::UAVCompute);
                b.Write(in.outSpecular, Access::UAVCompute);
            },
            [this, in, common, relax, reblur](rdg::PassContext& ctx)
            {
                const void* settings = mInstanceKind == gpu::kNrdReblur ? static_cast<const void*>(&reblur) : &relax;
                Denoise(ctx, in, common, settings);
            });

        {
            gpu::NrdCompositeConstants c{};
            c.width = mWidth;
            c.height = mHeight;
            c.denoiser = mInstanceKind;
            c.denoisingRange = kDenoisingRange;
            c.view = mView;
            c.surfaceTransmittance = air::gpu::kInvalidIndex;
            const rhi::PipelineHandle pipeline = mComposite.Get(*mDevice);
            const rdg::TextureRef reads[] = {
                *diffuseSignal, *specularSignal,
                *diffuseAlbedo, *specularAlbedo,
                *viewDepth,     in.outDiffuse,
                in.outSpecular, surfaceTransmittance ? *surfaceTransmittance : rdg::TextureRef{}};
            const rdg::TextureRef color = context.sceneColor;
            graph.AddPass(
                "NRD.Composite", rdg::PassFlags::Compute,
                [&](rdg::PassBuilder& b)
                {
                    for (rdg::TextureRef t : reads)
                    {
                        if (t.IsValid())
                        {
                            b.Read(t, Access::SRVCompute);
                        }
                    }
                    b.Write(color, Access::UAVCompute);
                },
                [=](rdg::PassContext& ctx) mutable
                {
                    c.diffuseSignal = ctx.SRV(reads[0]);
                    c.specularSignal = ctx.SRV(reads[1]);
                    c.diffuseAlbedo = ctx.SRV(reads[2]);
                    c.specularAlbedo = ctx.SRV(reads[3]);
                    c.viewDepth = ctx.SRV(reads[4]);
                    c.outDiffuse = ctx.SRV(reads[5]);
                    c.outSpecular = ctx.SRV(reads[6]);
                    if (reads[7].IsValid())
                    {
                        c.surfaceTransmittance = ctx.SRV(reads[7]);
                    }
                    c.color = ctx.UAV(color);
                    ctx.cmd.SetPipeline(pipeline);
                    ctx.cmd.PushConstants(&c, sizeof(c));
                    ctx.cmd.Dispatch(groupsX, groupsY, 1);
                });
        }
    }

private:
    struct Pipeline
    {
        ComPtr<ID3D12RootSignature> rootSignature;
        ComPtr<ID3D12PipelineState> state;
        bool table = false; // has a descriptor table (root parameter 1)
    };

    struct PoolTexture
    {
        ComPtr<ID3D12Resource> resource;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    };

    // NRD's instance for one denoiser kind, its pipelines, textures, descriptor heap and constant ring.
    bool CreateInstance(u32 kind)
    {
        if (mInstance)
        {
            mDevice->WaitIdle(); // the GPU may still read what is released
        }
        DestroyInstance();
        const nrd::DenoiserDesc denoiser{kIdentifier, kind == gpu::kNrdReblur ? nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR
                                                                              : nrd::Denoiser::RELAX_DIFFUSE_SPECULAR};
        nrd::InstanceCreationDesc creation{};
        creation.denoisers = &denoiser;
        creation.denoisersNum = 1;
        if (nrd::CreateInstance(creation, mInstance) != nrd::Result::SUCCESS)
        {
            AIR_LOG(Error, "NRD", "CreateInstance failed");
            mInstance = nullptr;
            return false;
        }
        mInstanceKind = kind;
        mFirstFrame = true;
        const nrd::InstanceDesc& desc = *nrd::GetInstanceDesc(*mInstance);

        // Static samplers in the constant buffer's space.
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers(desc.samplersNum);
        for (u32 i = 0; i < desc.samplersNum; ++i)
        {
            D3D12_STATIC_SAMPLER_DESC& s = samplers[i];
            s = {};
            s.Filter = desc.samplers[i] == nrd::Sampler::LINEAR_CLAMP ? D3D12_FILTER_MIN_MAG_MIP_LINEAR
                                                                      : D3D12_FILTER_MIN_MAG_MIP_POINT;
            s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            s.MaxLOD = D3D12_FLOAT32_MAX;
            s.ShaderRegister = desc.samplersBaseRegisterIndex + i;
            s.RegisterSpace = desc.constantBufferAndSamplersSpaceIndex;
            s.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        u32 maxDescriptors = 0;
        for (u32 p = 0; p < desc.pipelinesNum; ++p)
        {
            const nrd::PipelineDesc& pipeline = desc.pipelines[p];
            // Root parameter 0: the constants (a root CBV), 1: the textures in NRD's range order, read-only
            // then read-write, each range from the resources' base register.
            std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
            u32 offset = 0;
            for (u32 r = 0; r < pipeline.resourceRangesNum; ++r)
            {
                const nrd::ResourceRangeDesc& range = pipeline.resourceRanges[r];
                D3D12_DESCRIPTOR_RANGE d{};
                d.RangeType = range.descriptorType == nrd::DescriptorType::TEXTURE ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                                                                                   : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                d.NumDescriptors = range.descriptorsNum;
                d.BaseShaderRegister = desc.resourcesBaseRegisterIndex;
                d.RegisterSpace = desc.resourcesSpaceIndex;
                d.OffsetInDescriptorsFromTableStart = offset;
                offset += range.descriptorsNum;
                ranges.push_back(d);
            }
            maxDescriptors = std::max(maxDescriptors, offset);
            D3D12_ROOT_PARAMETER parameters[2] = {};
            parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[0].Descriptor.ShaderRegister = desc.constantBufferRegisterIndex;
            parameters[0].Descriptor.RegisterSpace = desc.constantBufferAndSamplersSpaceIndex;
            parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[1].DescriptorTable.NumDescriptorRanges = u32(ranges.size());
            parameters[1].DescriptorTable.pDescriptorRanges = ranges.data();
            parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_ROOT_SIGNATURE_DESC rootDesc{};
            rootDesc.NumParameters = ranges.empty() ? 1u : 2u;
            rootDesc.pParameters = parameters;
            rootDesc.NumStaticSamplers = u32(samplers.size());
            rootDesc.pStaticSamplers = samplers.data();
            ComPtr<ID3DBlob> blob;
            ComPtr<ID3DBlob> errors;
            Pipeline out;
            out.table = !ranges.empty();
            if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors)) ||
                FAILED(mD3d->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&out.rootSignature))))
            {
                AIR_LOG(Error, "NRD", "root signature of pipeline {} ({}) failed: {}", p, pipeline.shaderIdentifier,
                        errors ? static_cast<const char*>(errors->GetBufferPointer()) : "");
                return false;
            }
            D3D12_COMPUTE_PIPELINE_STATE_DESC state{};
            state.pRootSignature = out.rootSignature.Get();
            state.CS = {pipeline.computeShaderDXIL.bytecode, size_t(pipeline.computeShaderDXIL.size)};
            if (FAILED(mD3d->CreateComputePipelineState(&state, IID_PPV_ARGS(&out.state))))
            {
                AIR_LOG(Error, "NRD", "pipeline {} ({}) failed to compile", p, pipeline.shaderIdentifier);
                return false;
            }
            std::wstring name(pipeline.shaderIdentifier,
                              pipeline.shaderIdentifier + std::strlen(pipeline.shaderIdentifier));
            out.state->SetName(name.c_str());
            mPipelines.push_back(std::move(out));
        }

        // NRD's own textures, at the render resolution divided by their downsample factors.
        auto createPool = [&](const nrd::TextureDesc* textures, u32 count, std::vector<PoolTexture>& pool)
        {
            for (u32 i = 0; i < count; ++i)
            {
                const u32 factor = std::max<u32>(textures[i].downsampleFactor, 1u);
                D3D12_RESOURCE_DESC r{};
                r.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                r.Width = (mWidth + factor - 1) / factor;
                r.Height = (mHeight + factor - 1) / factor;
                r.DepthOrArraySize = 1;
                r.MipLevels = 1;
                r.Format = ToDxgi(textures[i].format);
                r.SampleDesc.Count = 1;
                r.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
                PoolTexture t;
                if (r.Format == DXGI_FORMAT_UNKNOWN ||
                    FAILED(mD3d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &r, D3D12_RESOURCE_STATE_COMMON,
                                                         nullptr, IID_PPV_ARGS(&t.resource))))
                {
                    AIR_LOG(Error, "NRD", "texture {} of its pool (format {}) failed", i, u32(textures[i].format));
                    return false;
                }
                t.resource->SetName(L"NRD.Pool");
                pool.push_back(std::move(t));
            }
            return true;
        };
        if (!createPool(desc.permanentPool, desc.permanentPoolSize, mPermanent) ||
            !createPool(desc.transientPool, desc.transientPoolSize, mTransient))
        {
            return false;
        }

        // A frame's descriptors and constants, one ring slot per frame in flight (and one more).
        mDescriptorsPerFrame = maxDescriptors * kMaxDispatches;
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = std::max(mDescriptorsPerFrame * mSlots, 1u);
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(mD3d->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mHeap))))
        {
            AIR_LOG(Error, "NRD", "descriptor heap failed");
            return false;
        }
        mDescriptorSize = mD3d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        mConstantStride =
            AlignUp(std::max(desc.constantBufferMaxDataSize, 1u), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = u64(mConstantStride) * kMaxDispatches * mSlots;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
        if (FAILED(mD3d->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &buffer,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                 IID_PPV_ARGS(&mConstants))) ||
            FAILED(mConstants->Map(0, nullptr, reinterpret_cast<void**>(&mConstantsMapped))))
        {
            AIR_LOG(Error, "NRD", "constant buffer failed");
            return false;
        }
        AIR_LOG(Info, "NRD", "{} at {}x{}: {} pipelines, {} + {} textures of its own",
                nrd::GetDenoiserString(denoiser.denoiser), mWidth, mHeight, desc.pipelinesNum, desc.permanentPoolSize,
                desc.transientPoolSize);
        return true;
    }

    void DestroyInstance()
    {
        if (mConstants && mConstantsMapped)
        {
            mConstants->Unmap(0, nullptr);
        }
        mConstantsMapped = nullptr;
        mConstants.Reset();
        mHeap.Reset();
        mPipelines.clear();
        mPermanent.clear();
        mTransient.clear();
        if (mInstance)
        {
            nrd::DestroyInstance(*mInstance);
            mInstance = nullptr;
        }
    }

    // Records NRD's dispatches for this frame into the pass's command list.
    void Denoise(rdg::PassContext& ctx, const Inputs& in, const nrd::CommonSettings& common, const void* settings)
    {
        auto* list = static_cast<ID3D12GraphicsCommandList*>(ctx.cmd.GetNativeHandle());
        auto native = [&](rdg::TextureRef t)
        { return static_cast<ID3D12Resource*>(ctx.device.GetNativeResource(ctx.Get(t))); };
        nrd::SetCommonSettings(*mInstance, common);
        nrd::SetDenoiserSettings(*mInstance, kIdentifier, settings);
        const nrd::DispatchDesc* dispatches = nullptr;
        u32 count = 0;
        nrd::GetComputeDispatches(*mInstance, &kIdentifier, 1, dispatches, count);
        AIR_CHECK(count <= kMaxDispatches, "NRD: {} dispatches, more than {}", count, kMaxDispatches);

        // The render graph's textures start (and must end) in the states the pass declared.
        struct Tracked
        {
            ID3D12Resource* resource;
            D3D12_RESOURCE_STATES state;
        };
        constexpr D3D12_RESOURCE_STATES kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        constexpr D3D12_RESOURCE_STATES kWrite = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        Tracked graphTextures[] = {{native(in.motion), kRead},      {native(in.normalRoughness), kRead},
                                   {native(in.viewDepth), kRead},   {native(in.diffuse), kRead},
                                   {native(in.specular), kRead},    {native(in.outDiffuse), kWrite},
                                   {native(in.outSpecular), kWrite}};
        const Tracked declared[std::size(graphTextures)] = {graphTextures[0], graphTextures[1], graphTextures[2],
                                                            graphTextures[3], graphTextures[4], graphTextures[5],
                                                            graphTextures[6]};
        auto lookup = [&](const nrd::ResourceDesc& r) -> std::pair<ID3D12Resource*, D3D12_RESOURCE_STATES*>
        {
            switch (r.type)
            {
            case nrd::ResourceType::IN_MV: return {graphTextures[0].resource, &graphTextures[0].state};
            case nrd::ResourceType::IN_NORMAL_ROUGHNESS: return {graphTextures[1].resource, &graphTextures[1].state};
            case nrd::ResourceType::IN_VIEWZ: return {graphTextures[2].resource, &graphTextures[2].state};
            case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST:
                return {graphTextures[3].resource, &graphTextures[3].state};
            case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST:
                return {graphTextures[4].resource, &graphTextures[4].state};
            case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST:
                return {graphTextures[5].resource, &graphTextures[5].state};
            case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST:
                return {graphTextures[6].resource, &graphTextures[6].state};
            case nrd::ResourceType::TRANSIENT_POOL:
                return {mTransient[r.indexInPool].resource.Get(), &mTransient[r.indexInPool].state};
            case nrd::ResourceType::PERMANENT_POOL:
                return {mPermanent[r.indexInPool].resource.Get(), &mPermanent[r.indexInPool].state};
            default: AIR_CHECK(false, "NRD asks for {}", nrd::GetResourceTypeString(r.type)); return {nullptr, nullptr};
            }
        };

        const u32 slot = mFrameCounter++ % mSlots;
        u32 descriptor = slot * mDescriptorsPerFrame;
        u32 constantOffset = slot * mConstantStride * kMaxDispatches;
        D3D12_GPU_VIRTUAL_ADDRESS constants = 0;
        const D3D12_CPU_DESCRIPTOR_HANDLE cpuStart = mHeap->GetCPUDescriptorHandleForHeapStart();
        const D3D12_GPU_DESCRIPTOR_HANDLE gpuStart = mHeap->GetGPUDescriptorHandleForHeapStart();
        ID3D12DescriptorHeap* heaps[] = {mHeap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        for (u32 i = 0; i < count; ++i)
        {
            const nrd::DispatchDesc& d = dispatches[i];
            const Pipeline& pipeline = mPipelines[d.pipelineIndex];
            barriers.clear();
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; // every earlier write lands before this dispatch reads
            barriers.push_back(uav);
            const u32 first = descriptor;
            for (u32 r = 0; r < d.resourcesNum; ++r)
            {
                const nrd::ResourceDesc& resource = d.resources[r];
                auto [texture, state] = lookup(resource);
                const bool read = resource.descriptorType == nrd::DescriptorType::TEXTURE;
                const D3D12_RESOURCE_STATES wanted = read ? kRead : kWrite;
                if (*state != wanted)
                {
                    D3D12_RESOURCE_BARRIER t{};
                    t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    t.Transition.pResource = texture;
                    t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    t.Transition.StateBefore = *state;
                    t.Transition.StateAfter = wanted;
                    barriers.push_back(t);
                    *state = wanted;
                }
                AIR_CHECK(descriptor < (slot + 1) * mDescriptorsPerFrame, "NRD: out of descriptors");
                const D3D12_CPU_DESCRIPTOR_HANDLE handle{cpuStart.ptr + SIZE_T(descriptor) * mDescriptorSize};
                if (read)
                {
                    mD3d->CreateShaderResourceView(texture, nullptr, handle);
                }
                else
                {
                    mD3d->CreateUnorderedAccessView(texture, nullptr, nullptr, handle);
                }
                ++descriptor;
            }
            list->ResourceBarrier(u32(barriers.size()), barriers.data());
            if (d.constantBufferDataSize > 0 && (!d.constantBufferDataMatchesPreviousDispatch || constants == 0))
            {
                std::memcpy(mConstantsMapped + constantOffset, d.constantBufferData, d.constantBufferDataSize);
                constants = mConstants->GetGPUVirtualAddress() + constantOffset;
                constantOffset += mConstantStride;
            }
            list->SetComputeRootSignature(pipeline.rootSignature.Get());
            list->SetPipelineState(pipeline.state.Get());
            if (constants != 0)
            {
                list->SetComputeRootConstantBufferView(0, constants);
            }
            if (pipeline.table)
            {
                list->SetComputeRootDescriptorTable(1, {gpuStart.ptr + u64(first) * mDescriptorSize});
            }
            list->Dispatch(d.gridWidth, d.gridHeight, 1);
        }
        // The graph's textures back in the states it expects.
        barriers.clear();
        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barriers.push_back(uav);
        for (size_t i = 0; i < std::size(graphTextures); ++i)
        {
            if (graphTextures[i].state != declared[i].state)
            {
                D3D12_RESOURCE_BARRIER t{};
                t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                t.Transition.pResource = graphTextures[i].resource;
                t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                t.Transition.StateBefore = graphTextures[i].state;
                t.Transition.StateAfter = declared[i].state;
                barriers.push_back(t);
            }
        }
        list->ResourceBarrier(u32(barriers.size()), barriers.data());
    }

    bool mEnabled = true;
    u32 mDenoiser = gpu::kNrdRelax;
    u32 mHistoryFrames = 30;
    bool mAntiFirefly = false;
    u32 mView = 0;

    rhi::Device* mDevice = nullptr;
    ID3D12Device* mD3d = nullptr;
    u32 mWidth = 0;
    u32 mHeight = 0;
    u32 mSlots = 1;
    renderer::LazyComputePipeline mPrepare;
    renderer::LazyComputePipeline mComposite;

    nrd::Instance* mInstance = nullptr;
    u32 mInstanceKind = ~0u;
    std::vector<Pipeline> mPipelines;
    std::vector<PoolTexture> mPermanent;
    std::vector<PoolTexture> mTransient;
    ComPtr<ID3D12DescriptorHeap> mHeap;
    u32 mDescriptorSize = 0;
    u32 mDescriptorsPerFrame = 0;
    ComPtr<ID3D12Resource> mConstants;
    u8* mConstantsMapped = nullptr;
    u32 mConstantStride = 0;
    u64 mFrameCounter = 0;

    u32 mFrameIndex = 0;
    bool mFirstFrame = true;
    Matrix mPreviousWorldToView{};
    Matrix mPreviousViewToClip{};
    float mPreviousJitter[2] = {};
    bool mDenoised = false;
    bool mWarnedOtherDenoiser = false;
};

} // namespace
} // namespace air::nrdf

namespace air::features
{
void Register_nrd(renderer::FeatureRegistry& registry)
{
    registry.Register({"nrd", "NVIDIA Real-time Denoisers (RELAX / REBLUR) on the path tracer's surfaces (D3D12)",
                       [] { return std::make_unique<nrdf::NrdFeature>(); }});
}
} // namespace air::features
