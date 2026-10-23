#include "air/renderer/FrameRunner.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/renderer/Diagnostics.h"

#include <chrono>
#include <format>

namespace air::renderer
{

json::Value FrameResult::ToJson() const
{
    json::Value out = json::Value::MakeObject();
    out["frame"] = frameIndex;
    out["cpuMs"] = cpuMilliseconds;
    out["passes"] = graphStats.passCount;
    out["culledPasses"] = graphStats.culledPassCount;
    out["barriers"] = graphStats.barrierCount;
    out["commandLists"] = graphStats.commandListCount;
    out["transientResources"] = graphStats.transientResourceCount;
    out["transientHeapBytes"] = graphStats.transientHeapSize;
    out["transientBytesUnaliased"] = graphStats.transientBytesUnaliased;
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

FrameRunner::FrameRunner(rhi::Device& device, TaskSystem& tasks, FrameRunnerDesc desc)
    : mDevice(device), mTasks(tasks), mDesc(std::move(desc))
{
    mOutput = mDevice.CreateTexture(
        {.format = mDesc.outputFormat,
         .width = mDesc.width,
         .height = mDesc.height,
         .usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySrc,
         .debugName = "FrameOutput"});
    AIR_CHECK(mOutput.IsValid(), "failed to create the frame output texture");
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
}

bool FrameRunner::AddFeature(const std::string& name, std::unique_ptr<RenderFeature> feature)
{
    AIR_CHECK(feature, "AddFeature('{}') with a null feature", name);
    AIR_CHECK(!FindFeature(name), "feature '{}' added twice", name);
    FeatureInitContext context{mDevice, mTasks, mDesc.shaderRoot / name, mDesc.width, mDesc.height};
    if (!feature->Initialize(context))
    {
        AIR_LOG(Error, "FrameRunner", "feature '{}' failed to initialize", name);
        return false;
    }
    mFeatures.push_back({name, std::move(feature)});
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

    mDevice.BeginFrame();
    {
        rdg::RenderGraph graph(mDevice, &mTasks, mDesc.graphConfig);
        const rhi::Access finalAccess = rhi::Access::SRVCompute;
        rdg::TextureRef output = graph.ImportTexture(mOutput, mOutputAccess, finalAccess);
        mOutputAccess = finalAccess;

        Blackboard blackboard;
        FrameContext context{mDevice,    mTasks, graph,       scene,        camera,
                             blackboard, output, mDesc.width, mDesc.height, mFrameIndex};
        {
            AIR_PROFILE_SCOPE("FrameRunner.AddPasses");
            for (Entry& e : mFeatures)
            {
                AIR_PROFILE_SCOPE(e.name);
                e.feature->AddPasses(context);
            }
            if (extraPasses)
            {
                extraPasses(context);
            }
        }
        graph.Execute();

        result.graphStats = graph.GetStats();
        if (mDesc.profile)
        {
            result.passCpu = graph.GetPassTimings();
        }
        if (mDesc.captureGraph)
        {
            result.graphText = graph.DumpText();
            result.graphviz = graph.DumpGraphviz();
        }
    }
    for (std::string& line : mDevice.TakeCapturedCommands())
    {
        if (line.find("VALIDATION") != std::string::npos)
        {
            result.validationErrors.push_back(line);
        }
        if (mDesc.captureCommands)
        {
            result.commands.push_back(std::move(line));
        }
    }
    mDevice.EndFrame();
    ++mFrameIndex;
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
