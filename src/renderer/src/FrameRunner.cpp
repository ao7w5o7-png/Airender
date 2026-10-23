#include "air/renderer/FrameRunner.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/Diagnostics.h"
#include "air/renderer/ShaderLoader.h"
#include "air/renderer/shader/UpscaleShared.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>

namespace air::renderer
{

namespace
{

u32 ScaledSize(u32 size, float scale)
{
    return std::clamp(u32(std::lround(double(size) * double(scale))), 1u, size);
}

} // namespace

json::Value FrameResult::ToJson() const
{
    json::Value out = json::Value::MakeObject();
    out["frame"] = frameIndex;
    out["cpuMs"] = cpuMilliseconds;
    out["passes"] = graphStats.passCount;
    out["culledPasses"] = graphStats.culledPassCount;
    out["barriers"] = graphStats.barrierCount;
    out["commandLists"] = graphStats.commandListCount;
    out["asyncComputePasses"] = graphStats.asyncComputePassCount;
    out["submissions"] = graphStats.submissionCount;
    out["queueWaits"] = graphStats.queueWaitCount;
    out["transientResources"] = graphStats.transientResourceCount;
    out["transientHeapBytes"] = graphStats.transientHeapSize;
    out["transientBytesUnaliased"] = graphStats.transientBytesUnaliased;
    out["transientHeapsCreated"] = graphStats.transientHeapsCreated;
    out["transientResourcesCreated"] = graphStats.transientResourcesCreated;
    out["transientResourcesReused"] = graphStats.transientResourcesReused;
    json::Value& pool = out["transientPool"];
    pool["heaps"] = transientPool.heaps;
    pool["heapBytes"] = transientPool.heapBytes;
    pool["resources"] = transientPool.resources;
    pool["heapsReleased"] = transientPool.heapsReleased;
    pool["resourcesReleased"] = transientPool.resourcesReleased;
    json::Value errors = json::Value::MakeArray();
    for (const std::string& e : validationErrors)
    {
        errors.Push(e);
    }
    out["validationErrors"] = std::move(errors);
    if (!passCpu.empty() || gpu.frameIndex != ~0ull)
    {
        json::Value passes = json::Value::MakeArray();
        for (const auto& pass : passCpu)
        {
            json::Value p = json::Value::MakeObject();
            p["name"] = pass.name;
            p["cpuMs"] = pass.cpuMs;
            passes.Push(std::move(p));
        }
        out["passCpu"] = std::move(passes);
        out["gpu"] = renderer::ToJson(gpu);
        out["memory"] = renderer::ToJson(memory);
        out["processMemory"] = renderer::ToJson(processMemory);
    }
    return out;
}

namespace
{
constexpr u64 kBudgetCheckFrames = 120; // frames between VRAM budget checks (QueryVideoMemoryInfo)
}

FrameRunner::FrameRunner(rhi::Device& device, TaskSystem& tasks, FrameRunnerDesc desc)
    : mDevice(device), mTasks(tasks), mDesc(std::move(desc))
{
    if (mDesc.poolTransients)
    {
        mTransientPool = std::make_unique<rdg::TransientResourcePool>(mDevice);
    }
    mOutput = mDevice.CreateTexture(
        {.format = mDesc.outputFormat,
         .width = mDesc.width,
         .height = mDesc.height,
         // CopyDst: features that produce the frame elsewhere (dlss) copy their result back into it.
         .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySrc |
                  rhi::TextureUsage::CopyDst,
         .debugName = "FrameOutput"});
    AIR_CHECK(mOutput.IsValid(), "failed to create the frame output texture");
    AIR_CHECK(mDesc.renderScale > 0.0f && mDesc.renderScale <= 1.0f, "renderScale {} must be in (0, 1]",
              mDesc.renderScale);
    mRenderWidth = ScaledSize(mDesc.width, mDesc.renderScale);
    mRenderHeight = ScaledSize(mDesc.height, mDesc.renderScale);
    if (mRenderWidth != mDesc.width || mRenderHeight != mDesc.height)
    {
        mUpscalePipeline = mDevice.CreateComputePipeline(
            {.shader = LoadShader(mDevice, mDesc.shaderRoot / "renderer", "Upscale.cs", "Main"),
             .pushConstantBytes = sizeof(gpu::UpscaleConstants),
             .debugName = "FrameRunner.Upscale"});
        AIR_CHECK(mUpscalePipeline.IsValid(), "failed to create the FrameRunner.Upscale pipeline");
    }
    mDevice.SetCommandCapture(true); // always on: validation lines are collected every frame
    if (mDesc.profile)
    {
        mDevice.SetGpuProfiling(true);
    }
}

FrameRunner::~FrameRunner()
{
    mFeatures.clear(); // features release their resources before the output goes away
    mDevice.DestroyTexture(mOutput);
    if (mUpscalePipeline.IsValid())
    {
        mDevice.DestroyPipeline(mUpscalePipeline);
    }
}

bool FrameRunner::AddFeature(const std::string& name, std::unique_ptr<RenderFeature> feature)
{
    AIR_CHECK(feature, "AddFeature('{}') with a null feature", name);
    AIR_CHECK(!FindFeature(name), "feature '{}' added twice", name);
    FeatureInitContext context{mDevice,      mTasks,       mDesc.shaderRoot / name, mDesc.width, mDesc.height,
                               mRenderWidth, mRenderHeight};
    if (!feature->Initialize(context))
    {
        AIR_LOG(Error, "FrameRunner", "feature '{}' failed to initialize", name);
        return false;
    }
    for (std::string_view slot : feature->ConsumedSlots())
    {
        mConsumedSlots.emplace(slot);
    }
    // Keep the list ordered by stage; within a stage, in the order features were added.
    const FeatureStage stage = feature->GetStage();
    const auto position =
        std::find_if(mFeatures.begin(), mFeatures.end(), [&](const Entry& e) { return e.feature->GetStage() > stage; });
    mFeatures.insert(position, {name, std::move(feature)});
    return true;
}

RenderFeature* FrameRunner::FindFeature(std::string_view name) const
{
    for (const Entry& e : mFeatures)
    {
        if (e.name == name)
        {
            return e.feature.get();
        }
    }
    return nullptr;
}

bool FrameRunner::SetParameter(std::string_view qualifiedName, std::string_view value, std::string* error)
{
    const size_t dot = qualifiedName.find('.');
    if (dot == std::string_view::npos)
    {
        if (error)
        {
            *error = std::format("parameter '{}' must be written as <feature>.<parameter>", qualifiedName);
        }
        return false;
    }
    RenderFeature* feature = FindFeature(qualifiedName.substr(0, dot));
    if (!feature)
    {
        if (error)
        {
            *error = std::format("no active feature named '{}'", qualifiedName.substr(0, dot));
        }
        return false;
    }
    return feature->Parameters().Set(qualifiedName.substr(dot + 1), value, error);
}

FrameResult FrameRunner::RenderFrame(const GpuScene& scene, const Camera& camera,
                                     const std::function<void(FrameContext&)>& extraPasses)
{
    AIR_PROFILE_SCOPE("FrameRunner.RenderFrame");
    const auto start = std::chrono::steady_clock::now();
    FrameResult result;
    result.frameIndex = mFrameIndex;
    std::shared_ptr<FrameCapture> capture;
    if (std::exchange(mCaptureRequested, false))
    {
        capture = std::make_shared<FrameCapture>();
        capture->frameIndex = mFrameIndex;
        capture->width = mDesc.width;
        capture->height = mDesc.height;
        capture->renderWidth = mRenderWidth;
        capture->renderHeight = mRenderHeight;
    }

    mDevice.BeginFrame();
    {
        rdg::RenderGraph graph(mDevice, &mTasks, mDesc.graphConfig, mTransientPool.get());
        const rhi::Access finalAccess = rhi::Access::SRVCompute;
        rdg::TextureRef output = graph.ImportTexture(mOutput, mOutputAccess, finalAccess);
        mOutputAccess = finalAccess;

        Blackboard blackboard;
        FrameContext context{mDevice,    mTasks, graph,       scene,        camera,
                             blackboard, output, mDesc.width, mDesc.height, mFrameIndex};
        context.renderWidth = mRenderWidth;
        context.renderHeight = mRenderHeight;
        context.sceneColor = output;
        context.consumedSlots = &mConsumedSlots;
        const bool upscaling = mUpscalePipeline.IsValid();
        if (upscaling)
        {
            context.sceneColor =
                graph.CreateTexture({.format = rhi::Format::RGBA16_FLOAT,
                                     .width = mRenderWidth,
                                     .height = mRenderHeight,
                                     .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource,
                                     .debugName = "FrameRunner.SceneColor"});
        }
        {
            AIR_PROFILE_SCOPE("FrameRunner.AddPasses");
            bool outputReady = !upscaling;
            for (Entry& e : mFeatures)
            {
                if (!outputReady && e.feature->GetStage() == FeatureStage::Post)
                {
                    AddFallbackUpscale(context);
                    outputReady = true;
                }
                AIR_PROFILE_SCOPE(e.name);
                e.feature->AddPasses(context);
            }
            if (!outputReady)
            {
                AddFallbackUpscale(context);
            }
            if (extraPasses)
            {
                extraPasses(context);
            }
        }
        graph.Execute();

        result.graphStats = graph.GetStats();
        if (mTransientPool)
        {
            result.transientPool = mTransientPool->GetStats();
        }
        if (mDesc.profile)
        {
            result.passCpu = graph.GetPassTimings();
        }
        if (mDesc.captureGraph)
        {
            result.graphText = graph.DumpText();
            result.graphviz = graph.DumpGraphviz();
        }
        if (capture)
        {
            capture->graph = graph.DumpJson();
        }
    }
    for (std::string& line : mDevice.TakeCapturedCommands())
    {
        if (line.find("VALIDATION") != std::string::npos)
        {
            result.validationErrors.push_back(line);
        }
        if (capture)
        {
            capture->commands.push_back(line);
        }
        if (mDesc.captureCommands)
        {
            result.commands.push_back(std::move(line));
        }
    }
    if (capture)
    {
        capture->deviceFrame = mDevice.GetFrameIndex();
        capture->resources = mDevice.GetResourceList();
        capture->memory = mDevice.GetMemoryStats();
        result.capture = std::move(capture);
    }
    mDevice.EndFrame();
    ++mFrameIndex;
    // Near the VRAM budget, idle transient memory goes now rather than after the pool's retire delay.
    if (mTransientPool && mFrameIndex % kBudgetCheckFrames == 0)
    {
        const rhi::MemoryBudget vram = mDevice.GetMemoryStats().local;
        if (vram.budgetBytes > 0 && vram.usageBytes > vram.budgetBytes / 100 * 95)
        {
            mTransientPool->Trim(mDevice.GetFramesInFlight() + 1);
            result.transientPool = mTransientPool->GetStats();
        }
    }
    if (mDesc.profile)
    {
        result.gpu = mDevice.GetGpuTimings();
        result.memory = mDevice.GetMemoryStats();
        result.processMemory = GetProcessMemory();
    }

    result.cpuMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

void FrameRunner::AddFallbackUpscale(FrameContext& context)
{
    if (context.outputWritten)
    {
        return; // an Upscale feature reconstructed the output
    }
    const rdg::TextureRef source = context.sceneColor;
    const rdg::TextureRef destination = context.output;
    gpu::UpscaleConstants constants{};
    constants.sourceWidth = context.renderWidth;
    constants.sourceHeight = context.renderHeight;
    constants.destinationWidth = context.width;
    constants.destinationHeight = context.height;
    const rhi::PipelineHandle pipeline = mUpscalePipeline;
    context.graph.AddPass(
        "FrameRunner.Upscale", rdg::PassFlags::Compute,
        [&](rdg::PassBuilder& b)
        {
            b.Read(source, rhi::Access::SRVCompute);
            b.Write(destination, rhi::Access::UAVCompute);
        },
        [=](rdg::PassContext& ctx) mutable
        {
            constants.source = ctx.SRV(source);
            constants.destination = ctx.UAV(destination);
            ctx.cmd.SetPipeline(pipeline);
            ctx.cmd.PushConstants(&constants, sizeof(constants));
            ctx.cmd.Dispatch((constants.destinationWidth + 7) / 8, (constants.destinationHeight + 7) / 8, 1);
        });
    context.outputWritten = true;
}

json::Value FrameRunner::DescribeFeatures() const
{
    json::Value out = json::Value::MakeArray();
    for (const Entry& e : mFeatures)
    {
        json::Value item = json::Value::MakeObject();
        item["name"] = e.name;
        item["parameters"] = e.feature->Parameters().ToJson();
        item["status"] = e.feature->GetStatus();
        out.Push(std::move(item));
    }
    return out;
}

} // namespace air::renderer
