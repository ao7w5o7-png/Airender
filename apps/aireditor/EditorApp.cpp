#include "EditorApp.h"

#include "Widgets.h"

#include "air/core/Log.h"
#include "air/renderer/Diagnostics.h"
#include "air/renderer/ProfileReport.h"
#include "air/renderer/Transfer.h"

#include <imgui.h>
#include <imgui_internal.h> // DockBuilder

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <numbers>
#include <utility>

namespace air::editor
{

namespace
{

constexpr size_t kMaxLogLines = 2000;
constexpr float kResizeDelaySeconds = 0.25f;
constexpr float kMaxResizeTimerStep = 1.0f / 30.0f; // a slow frame counts as at most this
constexpr u32 kResizeSettleFrames = 5;
constexpr u32 kMaxShaderLogLines = 40;

std::string ParameterText(const renderer::ParameterSet::Entry& entry)
{
    if (const auto* b = std::get_if<bool*>(&entry.value))
    {
        return **b ? "true" : "false";
    }
    if (const auto* u = std::get_if<u32*>(&entry.value))
    {
        return std::to_string(**u);
    }
    return std::format("{}", **std::get_if<float*>(&entry.value));
}

ImTextureID TextureId(rhi::Device& device, rhi::TextureHandle texture)
{
    // ImGui's DX12 backend binds a GPU descriptor handle; the bindless heap is the one it draws with.
    return ImTextureID(device.GetBindlessDescriptor(device.GetBindlessSRV(texture)).gpu);
}

float Degrees(float radians)
{
    return radians * 180.0f / std::numbers::pi_v<float>;
}

float Radians(float degrees)
{
    return degrees * std::numbers::pi_v<float> / 180.0f;
}

} // namespace

EditorApp::EditorApp(rhi::Device& device, TaskSystem& tasks, EditorOptions options)
    : mDevice(device), mTasks(tasks), mOptions(std::move(options))
{
    mFocusWindow = mOptions.focusWindow;
    SetLogSink(
        [this](LogLevel level, std::string_view category, std::string_view message)
        {
            std::lock_guard lock(mLogMutex);
            mLog.push_back({level, std::format("[{}] {}", category, message)});
            if (mLog.size() > kMaxLogLines)
            {
                mLog.pop_front();
            }
        });
    Profiler::Get().SetEnabled(true);
    Profiler::Get().SetThreadName("Main");
    if (!LoadScene(mOptions.scene))
    {
        LoadScene("cornell_box");
    }
    CreateRunner(mOptions.width, mOptions.height);
    mShaderReload = std::make_unique<renderer::ShaderHotReload>(mOptions.shaderRoot);
    if (mShaderReload->IsAvailable())
    {
        AIR_LOG(Info, "Shaders", "hot reload: watching {} files of {} shaders (Ctrl+R recompiles all)",
                mShaderReload->GetWatchedFileCount(), mShaderReload->GetShaderCount());
    }
    else
    {
        AIR_LOG(Info, "Shaders", "hot reload unavailable: {}", mShaderReload->GetUnavailableReason());
    }
}

EditorApp::~EditorApp()
{
    mDevice.WaitIdle();
    mRunner.reset();
    mScene.reset();
    SetLogSink({});
}

bool EditorApp::LoadScene(const std::string& name)
{
    std::optional<renderer::SceneDesc> desc = renderer::MakeBuiltinScene(name);
    if (!desc)
    {
        AIR_LOG(Error, "Editor", "unknown scene '{}'", name);
        return false;
    }
    mDevice.WaitIdle(); // the GPU may still read the old scene
    mScene.reset();
    mScene = std::make_unique<renderer::Scene>(mDevice, std::move(*desc));
    mSceneName = name;
    mCamera = mScene->Desc().camera;
    AIR_LOG(Info, "Editor", "scene '{}': {} triangles, {} instances, {} materials", name, mScene->GetTriangleCount(),
            mScene->Desc().instances.size(), mScene->Desc().materials.size());
    return true;
}

void EditorApp::CreateRunner(u32 width, u32 height)
{
    width = std::max(width, 8u);
    height = std::max(height, 8u);
    // Keep parameter values and render graph settings across the rebuild.
    std::vector<std::pair<std::string, std::string>> saved;
    rdg::RenderGraph::Config graphConfig;
    if (mRunner)
    {
        graphConfig = mRunner->GetDesc().graphConfig;
        for (const std::string& name : mActiveFeatures)
        {
            if (renderer::RenderFeature* feature = mRunner->FindFeature(name))
            {
                for (const auto& entry : feature->Parameters().Entries())
                {
                    saved.emplace_back(name + "." + entry.name, ParameterText(entry));
                }
            }
        }
    }
    mDevice.WaitIdle();
    mRunner.reset();

    renderer::FrameRunnerDesc desc;
    desc.width = width;
    desc.height = height;
    // With DLSS the path tracer renders below the output size and Ray Reconstruction upscales.
    desc.renderScale = IsDlssEnabled() ? kDlssModes[mOptions.dlssMode].renderScale : 1.0f;
    desc.shaderRoot = mOptions.shaderRoot;
    desc.graphConfig = graphConfig;
    desc.profile = true;
    mRunner = std::make_unique<renderer::FrameRunner>(mDevice, mTasks, desc);
    renderer::FeatureRegistry registry;
    renderer::RegisterBuiltinFeatures(registry);
    mActiveFeatures.clear();
    for (const std::string& name : mOptions.features)
    {
        auto feature = registry.Create(name);
        if (!feature)
        {
            AIR_LOG(Error, "Editor", "unknown feature '{}'", name);
            continue;
        }
        if (mRunner->AddFeature(name, std::move(feature)))
        {
            mActiveFeatures.push_back(name);
        }
    }
    for (const auto& [name, value] : saved)
    {
        SetParameter(name, value);
    }
    mPendingWidth = width;
    mPendingHeight = height;
    AIR_LOG(Info, "Editor", "rendering at {}x{} (scene {}x{}) with {}", width, height, mRunner->GetRenderWidth(),
            mRunner->GetRenderHeight(),
            [&]
            {
                std::string list;
                for (const std::string& f : mActiveFeatures)
                {
                    list += (list.empty() ? "" : ", ") + f;
                }
                return list;
            }());
}

void EditorApp::Resize(u32 width, u32 height)
{
    const auto& desc = mRunner->GetDesc();
    if (width != desc.width || height != desc.height)
    {
        CreateRunner(width, height);
    }
}

std::string EditorApp::SetParameter(const std::string& qualified, const std::string& value)
{
    std::string error;
    if (!mRunner->SetParameter(qualified, value, &error))
    {
        return error;
    }
    return {};
}

std::string EditorApp::GetParameter(std::string_view feature, std::string_view name) const
{
    if (renderer::RenderFeature* f = mRunner->FindFeature(std::string(feature)))
    {
        for (const auto& entry : f->Parameters().Entries())
        {
            if (entry.name == name)
            {
                return ParameterText(entry);
            }
        }
    }
    return {};
}

void EditorApp::SetDlssMode(u32 mode)
{
    mode = std::min<u32>(mode, u32(std::size(kDlssModes)) - 1);
    if (mode == mOptions.dlssMode)
    {
        return;
    }
    mOptions.dlssMode = mode;
    if (IsDlssEnabled())
    {
        const auto& desc = mRunner->GetDesc();
        CreateRunner(desc.width, desc.height);
    }
}

bool EditorApp::IsDlssEnabled() const
{
    return std::find(mOptions.features.begin(), mOptions.features.end(), "dlss") != mOptions.features.end();
}

void EditorApp::SetDlssEnabled(bool enabled)
{
    if (enabled == IsDlssEnabled())
    {
        return;
    }
    std::erase(mOptions.features, std::string("dlss"));
    if (enabled)
    {
        // RR denoises single jittered frames. Remember the accumulation settings to restore them:
        // without a denoiser those frames would shimmer.
        mPreDlssAccumulate = GetParameter("pathtracer", "accumulate");
        mPreDlssSpp = GetParameter("pathtracer", "spp");
        const auto pt = std::find(mOptions.features.begin(), mOptions.features.end(), "pathtracer");
        mOptions.features.insert(pt == mOptions.features.end() ? mOptions.features.begin() : pt + 1, "dlss");
    }
    const auto& desc = mRunner->GetDesc();
    CreateRunner(desc.width, desc.height);
    if (enabled)
    {
        SetParameter("pathtracer.accumulate", "false");
        SetParameter("pathtracer.spp", "1");
    }
    else
    {
        SetParameter("pathtracer.accumulate", mPreDlssAccumulate.empty() ? "true" : mPreDlssAccumulate);
        if (!mPreDlssSpp.empty())
        {
            SetParameter("pathtracer.spp", mPreDlssSpp);
        }
    }
}

// --- Frame ---------------------------------------------------------------------------------------

void EditorApp::RenderFrame(const std::function<void(renderer::FrameContext&)>& uiPass)
{
    const auto now = std::chrono::steady_clock::now();
    const double wallMs =
        mLastFrameTime ? std::chrono::duration<double, std::milli>(now - *mLastFrameTime).count() : 0.0;
    mLastFrameTime = now;

    mScene->Update();
    // A report requested while a capture still waits for its GPU timings uses that capture.
    u32 captureTargets = 0;
    if (mReportRequest != 0 && mPendingCapture)
    {
        mPendingTargets |= std::exchange(mReportRequest, 0u);
    }
    else if (mReportRequest != 0)
    {
        mRunner->RequestCapture();
        captureTargets = std::exchange(mReportRequest, 0u);
    }
    Profiler::Get().BeginFrame();
    mLastResult = mRunner->RenderFrame(mScene->Get(), mCamera, uiPass);
    Profiler::Get().EndFrame();
    for (const std::string& error : mLastResult.validationErrors)
    {
        AIR_LOG(Error, "Validation", "{}", error);
    }
    if (mLastResult.capture)
    {
        mPendingCapture = mLastResult.capture;
        mPendingTargets = captureTargets;
        mPendingCaptureFrames = 0;
    }
    if (mPendingCapture)
    {
        FinishProfileReport();
    }
    // Samples traced this frame: spp when the trace pass ran (it stops once converged).
    u32 samples = 0;
    const bool traced = std::any_of(mLastResult.passCpu.begin(), mLastResult.passCpu.end(),
                                    [](const auto& pass) { return pass.name == "PathTracer.Trace"; });
    if (renderer::RenderFeature* pt = mRunner->FindFeature("pathtracer"); traced && pt)
    {
        for (const auto& entry : pt->Parameters().Entries())
        {
            if (entry.name == "spp")
            {
                samples = **std::get_if<u32*>(&entry.value);
            }
        }
    }
    const auto& desc = mRunner->GetDesc();
    mDiagnostics.Update(mLastResult, wallMs, samples * desc.width * desc.height);
}

void EditorApp::ExportProfileReport()
{
    mReportRequest |= kReportFile;
    mDiagnostics.SetReportStatus("capturing the next frame...");
}

void EditorApp::FinishProfileReport()
{
    // The captured frame's GPU timings resolve a few frames after it; report without them if they
    // never come (GPU profiling off, device lost).
    const rhi::GpuFrameTimings& gpu = mLastResult.gpu;
    const bool arrived = gpu.frameIndex != ~0ull && gpu.frameIndex >= mPendingCapture->deviceFrame;
    if (!arrived && ++mPendingCaptureFrames < 16)
    {
        return;
    }
    const json::Value report = renderer::MakeProfileReport({.capture = mPendingCapture.get(),
                                                            .gpu = arrived ? gpu : rhi::GpuFrameTimings{},
                                                            .backend = rhi::ToString(mDevice.GetBackend()),
                                                            .adapter = mDevice.GetCapabilities().adapterName,
                                                            .scene = mSceneName,
                                                            .features = mRunner->DescribeFeatures()});
    const u64 frame = mPendingCapture->frameIndex;
    const u32 targets = std::exchange(mPendingTargets, 0u);
    mPendingCapture.reset();
    if ((targets & kReportPanel) && mHtmlView)
    {
        mHtmlView->PostJson(report.Dump(0));
    }
    if (!(targets & kReportFile))
    {
        return;
    }
    const std::filesystem::path path = mOptions.profileDirectory / std::format("{}-frame{}.html", mSceneName, frame);
    if (!renderer::WriteProfileHtml(path, report))
    {
        AIR_LOG(Error, "Editor", "could not write the profile report {}", path.string());
        mDiagnostics.SetReportStatus("could not write " + path.string());
        return;
    }
    mLastProfileReport = std::filesystem::absolute(path);
    AIR_LOG(Info, "Editor", "profile report: {}", mLastProfileReport.string());
    mDiagnostics.SetReportStatus("wrote " + path.generic_string());
    if (mOptions.openFile)
    {
        mOptions.openFile(mLastProfileReport);
    }
}

bool EditorApp::SaveScreenshot(const std::filesystem::path& path)
{
    mDevice.WaitIdle();
    const std::optional<Image> image =
        renderer::ReadbackTexture(mDevice, mRunner->GetOutput(), rhi::Access::SRVCompute);
    if (!image)
    {
        return false;
    }
    std::filesystem::path ppm = path, pfm = path;
    ppm.replace_extension(".ppm");
    pfm.replace_extension(".pfm");
    // The output is display-encoded when the tonemap feature ran last.
    const bool displayEncoded = !mActiveFeatures.empty() && mActiveFeatures.back() == "tonemap";
    const bool ok = WritePpm(ppm, *image, !displayEncoded) && WritePfm(pfm, *image);
    AIR_LOG(Info, "Editor", "{} {} and {}", ok ? "saved" : "could not save", ppm.string(), pfm.string());
    return ok;
}

// --- UI ------------------------------------------------------------------------------------------

void EditorApp::PollShaderReload()
{
    if (!mShaderReload->IsAvailable() ||
        !(mShaderHotReload || mShaderRecompileRequested || mShaderReload->IsCompiling()))
    {
        return;
    }
    std::optional<renderer::ShaderCompileResult> result = mShaderReload->Poll();
    if (!result)
    {
        return;
    }
    mShaderRecompileRequested = false;
    if (!result->success)
    {
        // One log line per compiler line, so the Log panel shows them in red.
        mShaderFailed = true;
        mShaderStatus.clear();
        std::string_view log = result->log;
        u32 lines = 0;
        while (!log.empty() && lines < kMaxShaderLogLines)
        {
            const size_t end = log.find('\n');
            std::string_view line = log.substr(0, end);
            log = end == std::string_view::npos ? std::string_view() : log.substr(end + 1);
            if (!line.empty() && line.back() == '\r')
            {
                line.remove_suffix(1);
            }
            if (line.empty())
            {
                continue;
            }
            AIR_LOG(Error, "Shaders", "{}", line);
            if (mShaderStatus.empty() && line.find("error") != std::string_view::npos)
            {
                mShaderStatus = line;
            }
            ++lines;
        }
        AIR_LOG(Error, "Shaders", "compile failed; the previous shaders keep running");
        if (mShaderStatus.empty())
        {
            mShaderStatus = "compile failed";
        }
        return;
    }
    if (result->shaders.empty())
    {
        return;
    }
    mShaderFailed = false;
    std::string names;
    for (const std::string& name : result->shaders)
    {
        names += (names.empty() ? "" : ", ") + name;
    }
    mShaderStatus = std::format("{} shader{} reloaded in {:.0f} ms", result->shaders.size(),
                                result->shaders.size() == 1 ? "" : "s", result->milliseconds);
    AIR_LOG(Info, "Shaders", "{}: {}", mShaderStatus, names);
    if (result->sharedHeaderChanged)
    {
        AIR_LOG(Warning, "Shaders",
                "a *Shared.h header changed: rebuild the editor if a struct layout changed (C++ and HLSL must agree)");
    }
    const auto& desc = mRunner->GetDesc();
    CreateRunner(desc.width, desc.height); // every feature reloads its shaders; parameters are kept
}

void EditorApp::BuildUi(float deltaSeconds)
{
    // Before any window is built, like the resize below: a reload recreates the runner.
    PollShaderReload();
    // Apply a viewport resize once the size stopped changing, for both a while and a few frames: one
    // slow frame (pipelines compiling at startup) must not commit a transient layout size such as
    // 1328x20. Done before any window is built so this frame's draw data only references the new
    // output texture.
    const auto& desc = mRunner->GetDesc();
    if (mPendingWidth != mSettlingWidth || mPendingHeight != mSettlingHeight)
    {
        mSettlingWidth = mPendingWidth;
        mSettlingHeight = mPendingHeight;
        mResizeTimer = 0.0f;
        mSettledFrames = 0;
    }
    if (!mLockResolution && (mPendingWidth != desc.width || mPendingHeight != desc.height))
    {
        mResizeTimer += std::min(deltaSeconds, kMaxResizeTimerStep);
        ++mSettledFrames;
        if (mResizeTimer >= kResizeDelaySeconds && mSettledFrames >= kResizeSettleFrames &&
            !ImGui::IsMouseDown(ImGuiMouseButton_Left))
        {
            CreateRunner(mPendingWidth, mPendingHeight);
            mResizeTimer = 0.0f;
            mSettledFrames = 0;
        }
    }
    else
    {
        mResizeTimer = 0.0f;
        mSettledFrames = 0;
    }
    BuildMenuBar();
    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
    if (!mDockLayoutBuilt)
    {
        mDockLayoutBuilt = true;
        // Default layout on first run (imgui.ini overrides it afterwards).
        if (ImGui::DockBuilderGetNode(dockspace) && ImGui::DockBuilderGetNode(dockspace)->IsLeafNode() &&
            ImGui::FindWindowSettingsByID(ImHashStr("Viewport")) == nullptr)
        {
            ImGuiID left = 0, right = 0, bottom = 0, center = dockspace;
            ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.28f, &right, &center);
            ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.30f, &bottom, &center);
            ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.22f, &left, &center);
            ImGui::DockBuilderDockWindow("Viewport", center);
            ImGui::DockBuilderDockWindow("Scene", left);
            ImGui::DockBuilderDockWindow("Render", right);
            ImGui::DockBuilderDockWindow("Debug", right);
            ImGui::DockBuilderDockWindow("Profiler", bottom);
            ImGui::DockBuilderDockWindow(kHtmlProfilerWindow, bottom);
            ImGui::DockBuilderDockWindow("Memory", bottom);
            ImGui::DockBuilderDockWindow("Log", bottom);
            ImGui::DockBuilderFinish(dockspace);
        }
    }
    BuildViewport(deltaSeconds);
    if (mShowScene)
    {
        BuildScenePanel();
    }
    if (mShowRender)
    {
        BuildRenderPanel();
    }
    if (mShowDebug)
    {
        BuildDebugPanel();
    }
    if (mShowProfiler)
    {
        mDiagnostics.BuildProfiler(&mShowProfiler);
        if (mDiagnostics.TakeReportRequest())
        {
            ExportProfileReport();
        }
    }
    if (mShowHtmlProfiler)
    {
        BuildHtmlProfiler();
    }
    else if (mHtmlView)
    {
        mHtmlView->SetBounds(0, 0, 0, 0, false);
    }
    if (mShowMemory)
    {
        mDiagnostics.BuildMemory(mDevice, &mShowMemory);
    }
    if (mShowLog)
    {
        BuildLogPanel();
    }
    if (mShowImGuiDemo)
    {
        ImGui::ShowDemoWindow(&mShowImGuiDemo);
    }
    // Once the window exists (it may take a frame), focus it: a docked window's tab gets selected.
    if (!mFocusWindow.empty() && ImGui::FindWindowByName(mFocusWindow.c_str()))
    {
        ImGui::SetWindowFocus(mFocusWindow.c_str());
        mFocusWindow.clear();
    }
}

// The profile report of §9.2 in a web view over this panel, refreshed while visible.
void EditorApp::BuildHtmlProfiler()
{
    // A layout saved before this panel existed: dock it next to the ImGui profiler.
    if (const ImGuiWindow* profiler = ImGui::FindWindowByName("Profiler"); profiler && profiler->DockId != 0)
    {
        ImGui::SetNextWindowDockID(profiler->DockId, ImGuiCond_FirstUseEver);
    }
    bool visible = false;
    ImVec2 origin(0.0f, 0.0f), size(0.0f, 0.0f);
    if (ImGui::Begin(kHtmlProfilerWindow, &mShowHtmlProfiler))
    {
        ImGui::Checkbox("Live", &mHtmlLive);
        ImGui::SetItemTooltip("Capture a frame for the page every %.1f s while this panel is visible; the "
                              "profiler's Pause (P) holds it.",
                              mOptions.htmlProfilerInterval);
        ImGui::SameLine();
        if (ImGui::Button("Capture"))
        {
            mReportRequest |= kReportPanel;
        }
        ImGui::SameLine();
        if (ImGui::Button("Open in browser"))
        {
            ExportProfileReport();
        }
        if (mHtmlView)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", mHtmlView->GetStatus().c_str());
        }
        origin = ImGui::GetCursorScreenPos();
        size = ImGui::GetContentRegionAvail();
        if (!mOptions.createHtmlView)
        {
            ImGui::TextWrapped("No web view on this platform: Open in browser shows the same report.");
        }
        else
        {
            if (!mHtmlView)
            {
                mHtmlView = mOptions.createHtmlView();
                mHtmlView->LoadPage(renderer::MakeProfileHtml(json::Value())); // waits for the first capture
            }
            ImGui::Dummy(size); // the page covers this area
            // A popup (a menu, a combo) under the web view would be hidden by it.
            visible = size.x >= 16.0f && size.y >= 16.0f &&
                      !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
            const auto now = std::chrono::steady_clock::now();
            if (visible && mHtmlLive && !mDiagnostics.IsPaused() && !mPendingCapture && mReportRequest == 0 &&
                now - mLastHtmlCapture >= std::chrono::duration<float>(mOptions.htmlProfilerInterval))
            {
                mReportRequest |= kReportPanel;
                mLastHtmlCapture = now;
            }
        }
    }
    ImGui::End();
    if (mHtmlView)
    {
        mHtmlView->SetBounds(int(origin.x), int(origin.y), int(size.x), int(size.y), visible);
    }
}

void EditorApp::BuildMenuBar()
{
    if (!ImGui::BeginMainMenuBar())
    {
        return;
    }
    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("Save screenshot", "F12"))
        {
            SaveScreenshot(std::format("airender_{:03}", mScreenshotCounter++));
        }
        if (ImGui::MenuItem("Export frame profile (HTML)"))
        {
            ExportProfileReport();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Quit", "Alt+F4"))
        {
            mQuit = true;
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Scene"))
    {
        for (const std::string& name : renderer::BuiltinSceneNames())
        {
            if (ImGui::MenuItem(name.c_str(), nullptr, name == mSceneName))
            {
                LoadScene(name);
            }
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Window"))
    {
        ImGui::MenuItem("Scene", nullptr, &mShowScene);
        ImGui::MenuItem("Render", nullptr, &mShowRender);
        ImGui::MenuItem("Debug", nullptr, &mShowDebug);
        ImGui::MenuItem("Profiler", nullptr, &mShowProfiler);
        ImGui::MenuItem(kHtmlProfilerWindow, nullptr, &mShowHtmlProfiler);
        ImGui::MenuItem("Memory", nullptr, &mShowMemory);
        ImGui::MenuItem("Log", nullptr, &mShowLog);
        ImGui::Separator();
        ImGui::MenuItem("ImGui demo", nullptr, &mShowImGuiDemo);
        ImGui::EndMenu();
    }
    const std::string stats =
        std::format("{}{} | {:.1f} fps{} | GPU {} ", mDiagnostics.IsPaused() ? "PROFILER PAUSED | " : "",
                    mDevice.GetCapabilities().adapterName, mDiagnostics.GetFps(), mOptions.vsync ? "" : " (uncapped)",
                    FormatMs(mDiagnostics.GetGpuBusyMs()));
    ImGui::SameLine(ImGui::GetWindowWidth() - ImGui::CalcTextSize(stats.c_str()).x - 12.0f);
    ImGui::TextDisabled("%s", stats.c_str());
    ImGui::EndMainMenuBar();
    if (ImGui::IsKeyPressed(ImGuiKey_F12, false))
    {
        SaveScreenshot(std::format("airender_{:03}", mScreenshotCounter++));
    }
    // P is a plain letter: ignore it while a text field has focus.
    if (ImGui::IsKeyPressed(ImGuiKey_Pause, false) ||
        (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_P, false)))
    {
        mDiagnostics.SetPaused(!mDiagnostics.IsPaused());
    }
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false) && mShaderReload->IsAvailable())
    {
        mShaderReload->RecompileAll();
        mShaderRecompileRequested = true;
    }
}

void EditorApp::BuildViewport(float deltaSeconds)
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    const bool visible = ImGui::Begin("Viewport");
    ImGui::PopStyleVar();
    if (!visible)
    {
        mViewportHovered = false;
        ImGui::End();
        return;
    }
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    mPendingWidth = u32(std::max(avail.x * mResolutionScale, 8.0f));
    mPendingHeight = u32(std::max(avail.y * mResolutionScale, 8.0f));

    // Fit the rendered image into the window, keeping its aspect ratio.
    const auto& desc = mRunner->GetDesc();
    const float aspect = float(desc.width) / float(desc.height);
    ImVec2 size = avail;
    if (size.x / std::max(size.y, 1.0f) > aspect)
    {
        size.x = size.y * aspect;
    }
    else
    {
        size.y = size.x / aspect;
    }
    const ImVec2 cursor = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(cursor.x + (avail.x - size.x) * 0.5f, cursor.y + (avail.y - size.y) * 0.5f));
    const ImVec2 imageMin = ImGui::GetCursorScreenPos();
    ImGui::Image(ImTextureRef(TextureId(mDevice, mRunner->GetOutput())), size);
    mViewportHovered = ImGui::IsItemHovered();

    // Overlay: progress and the controls hint.
    std::string overlay = std::format("{}x{}", desc.width, desc.height);
    if (mRunner->GetRenderWidth() != desc.width || mRunner->GetRenderHeight() != desc.height)
    {
        overlay = std::format("scene {}x{} -> {}", mRunner->GetRenderWidth(), mRunner->GetRenderHeight(), overlay);
    }
    if (renderer::RenderFeature* pt = mRunner->FindFeature("pathtracer"))
    {
        overlay += "  " + pt->GetStatus().Dump(0);
    }
    overlay += std::format("  |  {:.1f} Mspp/s", mDiagnostics.GetSamplesPerSecond() / 1e6f);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(ImVec2(imageMin.x, imageMin.y),
                        ImVec2(imageMin.x + ImGui::CalcTextSize(overlay.c_str()).x + 12.0f, imageMin.y + 22.0f),
                        IM_COL32(0, 0, 0, 140));
    draw->AddText(ImVec2(imageMin.x + 6.0f, imageMin.y + 4.0f), IM_COL32(230, 230, 230, 255), overlay.c_str());
    if (mShaderFailed)
    {
        const std::string banner =
            std::format("shader compile failed: {}  (see Log; the previous shaders keep running)", mShaderStatus);
        draw->AddRectFilled(ImVec2(imageMin.x, imageMin.y + 24.0f),
                            ImVec2(imageMin.x + ImGui::CalcTextSize(banner.c_str()).x + 12.0f, imageMin.y + 46.0f),
                            IM_COL32(150, 20, 20, 220));
        draw->AddText(ImVec2(imageMin.x + 6.0f, imageMin.y + 28.0f), IM_COL32(255, 235, 235, 255), banner.c_str());
    }
    if (mViewportHovered)
    {
        ImGui::SetItemTooltip("Hold right mouse: look around, WASD/QE: move, wheel: speed, Shift: fast");
    }
    UpdateCamera(deltaSeconds);
    ImGui::End();
}

void EditorApp::UpdateCamera(float deltaSeconds)
{
    ImGuiIO& io = ImGui::GetIO();
    if (mViewportHovered && io.MouseWheel != 0.0f)
    {
        mMoveSpeed = std::clamp(mMoveSpeed * std::pow(1.2f, io.MouseWheel), 0.01f, 1000.0f);
    }
    const bool looking = ImGui::IsMouseDown(ImGuiMouseButton_Right) && (mViewportHovered || io.WantCaptureMouse);
    if (!looking || !(mViewportHovered || ImGui::IsMouseDragging(ImGuiMouseButton_Right)))
    {
        return;
    }
    renderer::Camera camera = mCamera;
    const float sensitivity = 0.0035f;
    camera.yaw -= io.MouseDelta.x * sensitivity;
    camera.pitch = std::clamp(camera.pitch - io.MouseDelta.y * sensitivity, -1.55f, 1.55f);
    renderer::float3 move(0.0f);
    if (ImGui::IsKeyDown(ImGuiKey_W))
    {
        move += camera.Forward();
    }
    if (ImGui::IsKeyDown(ImGuiKey_S))
    {
        move -= camera.Forward();
    }
    if (ImGui::IsKeyDown(ImGuiKey_D))
    {
        move += camera.Right();
    }
    if (ImGui::IsKeyDown(ImGuiKey_A))
    {
        move -= camera.Right();
    }
    if (ImGui::IsKeyDown(ImGuiKey_E))
    {
        move += renderer::float3(0.0f, 1.0f, 0.0f);
    }
    if (ImGui::IsKeyDown(ImGuiKey_Q))
    {
        move -= renderer::float3(0.0f, 1.0f, 0.0f);
    }
    const float speed = mMoveSpeed * (io.KeyShift ? 4.0f : 1.0f) * deltaSeconds;
    camera.position += move * speed;
    mCamera = camera; // the path tracer restarts accumulation when the camera changes
}

void EditorApp::BuildScenePanel()
{
    if (!ImGui::Begin("Scene", &mShowScene))
    {
        ImGui::End();
        return;
    }
    if (ImGui::BeginCombo("Scene", mSceneName.c_str()))
    {
        for (const std::string& name : renderer::BuiltinSceneNames())
        {
            if (ImGui::Selectable(name.c_str(), name == mSceneName))
            {
                LoadScene(name);
            }
        }
        ImGui::EndCombo();
    }
    ImGui::Text("%u triangles, %zu instances", mScene->GetTriangleCount(), mScene->Desc().instances.size());
    ImGui::Text("Geometry %s, %u emissive triangles, light power %.1f",
                renderer::FormatBytes(mScene->GetGeometryBytes()).c_str(), mScene->Get().lightCount,
                mScene->Get().lightPower);

    ImGui::SeparatorText("Camera");
    ImGui::DragFloat3("Position", &mCamera.position.x, 0.01f);
    float yaw = Degrees(mCamera.yaw), pitch = Degrees(mCamera.pitch), fov = Degrees(mCamera.verticalFov);
    if (ImGui::DragFloat("Yaw", &yaw, 0.2f))
    {
        mCamera.yaw = Radians(yaw);
    }
    if (ImGui::DragFloat("Pitch", &pitch, 0.2f, -89.0f, 89.0f))
    {
        mCamera.pitch = Radians(pitch);
    }
    if (ImGui::SliderFloat("Vertical FOV", &fov, 5.0f, 120.0f, "%.1f deg"))
    {
        mCamera.verticalFov = Radians(fov);
    }
    ImGui::SliderFloat("EV100", &mCamera.ev100, -6.0f, 20.0f, "%.2f");
    ImGui::SetItemTooltip("Exposure value at ISO 100. Exposure scale = 1 / (1.2 * 2^EV100).\n"
                          "Sunny day ~15, overcast ~12, office ~7, living room ~5.");
    ImGui::DragFloat("Move speed", &mMoveSpeed, 0.05f, 0.01f, 1000.0f, "%.2f m/s", ImGuiSliderFlags_Logarithmic);
    if (ImGui::Button("Reset camera"))
    {
        mCamera = mScene->Desc().camera;
    }

    ImGui::SeparatorText("Sky");
    renderer::Sky sky = mSkyEdit.value_or(mScene->Desc().sky);
    bool apply = ImGui::Checkbox("Physical sky and sun", &sky.physical);
    ImGui::SetItemTooltip("An Earth-like atmosphere lit by the sun, computed spectrally (docs/architecture.md, "
                          "section 8.9).\nOff: a uniform environment and no sun.");
    bool edited = apply;
    if (sky.physical)
    {
        auto slider = [&](const char* label, float* value, float lo, float hi, const char* format, int flags = 0)
        {
            edited |= ImGui::SliderFloat(label, value, lo, hi, format, flags);
            apply |= ImGui::IsItemDeactivatedAfterEdit();
        };
        slider("Sun elevation", &sky.sunElevation, -10.0f, 90.0f, "%.1f deg");
        slider("Sun azimuth", &sky.sunAzimuth, -180.0f, 180.0f, "%.1f deg");
        slider("Aerosol depth", &sky.aerosolDepth, 0.001f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Haze optical depth at 550 nm: 0.005 pristine, 0.05 clear, 0.3 and up hazy.");
        slider("Sun radius", &sky.sunAngularRadius, 0.05f, 5.0f, "%.3f deg", ImGuiSliderFlags_Logarithmic);
        slider("Ground albedo", &sky.groundAlbedo, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Of the planet seen below the horizon, beyond the scene.");
    }
    else
    {
        renderer::float3 env = mScene->Desc().environment;
        float envNits = std::max({env.x, env.y, env.z});
        renderer::float3 envColor = envNits > 0.0f ? env / envNits : renderer::float3(1.0f);
        bool envChanged = ImGui::ColorEdit3("Color##env", &envColor.x, ImGuiColorEditFlags_Float);
        envChanged |=
            ImGui::DragFloat("Radiance (nits)##env", &envNits, 1.0f, 0.0f, 1e6f, "%.1f", ImGuiSliderFlags_Logarithmic);
        if (envChanged)
        {
            mScene->SetEnvironment(envColor * envNits);
        }
    }
    if (apply)
    {
        mScene->SetSky(sky);
        mSkyEdit.reset();
    }
    else if (edited)
    {
        mSkyEdit = sky;
    }

    ImGui::SeparatorText("Materials");
    const auto& materials = mScene->Desc().materials;
    for (u32 i = 0; i < materials.size(); ++i)
    {
        renderer::Material m = materials[i];
        if (!ImGui::TreeNode(&materials[i], "%s", m.name.c_str()))
        {
            continue;
        }
        bool changed = ImGui::ColorEdit3("Base color", &m.baseColor.x, ImGuiColorEditFlags_Float);
        changed |= ImGui::SliderFloat("Roughness", &m.roughness, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Metallic", &m.metallic, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Emission color", &m.emissionColor.x);
        changed |= ImGui::DragFloat("Emission (nits)", &m.emissionNits, 5.0f, 0.0f, 1e7f, "%.1f",
                                    ImGuiSliderFlags_Logarithmic);
        if (changed)
        {
            mScene->SetMaterial(i, m);
        }
        ImGui::TreePop();
    }
    ImGui::End();
}

void EditorApp::BuildRenderPanel()
{
    if (!ImGui::Begin("Render", &mShowRender))
    {
        ImGui::End();
        return;
    }
    const auto& desc = mRunner->GetDesc();
    ImGui::Text("Backend %s on %s", rhi::ToString(mDevice.GetBackend()), mDevice.GetCapabilities().adapterName.c_str());
    ImGui::Text("Resolution %ux%u", desc.width, desc.height);
    ImGui::Checkbox("VSync", &mOptions.vsync);
    ImGui::SetItemTooltip("Off: frames are presented as fast as the GPU renders them (uncapped frame rate).");
    ImGui::Checkbox("Lock resolution", &mLockResolution);
    ImGui::SetItemTooltip("Unlocked, the render resolution follows the viewport size.");
    ImGui::SliderFloat("Resolution scale", &mResolutionScale, 0.25f, 2.0f, "%.2f");

    renderer::FeatureRegistry registry;
    renderer::RegisterBuiltinFeatures(registry);
    if (registry.Find("dlss"))
    {
        bool dlss = std::find(mOptions.features.begin(), mOptions.features.end(), "dlss") != mOptions.features.end();
        if (ImGui::Checkbox("DLSS Ray Reconstruction", &dlss))
        {
            SetDlssEnabled(dlss);
        }
        if (dlss)
        {
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10.0f);
            if (ImGui::BeginCombo("DLSS mode", kDlssModes[mOptions.dlssMode].name))
            {
                for (u32 i = 0; i < std::size(kDlssModes); ++i)
                {
                    if (ImGui::Selectable(kDlssModes[i].name, i == mOptions.dlssMode))
                    {
                        SetDlssMode(i);
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SetItemTooltip("The path tracer renders at the mode's fraction of the viewport and Ray\n"
                                  "Reconstruction denoises and upscales. DLAA renders at full size: best quality,\n"
                                  "highest cost. Quality traces 44%% of the pixels with nearly the same image.");
        }
    }
    BuildShaderSection();

    for (const std::string& name : mActiveFeatures)
    {
        renderer::RenderFeature* feature = mRunner->FindFeature(name);
        if (!feature || !ImGui::CollapsingHeader(name.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
        {
            continue;
        }
        ImGui::PushID(name.c_str());
        EditParameters(feature->Parameters());
        const std::string status = feature->GetStatus().Dump(0);
        if (status != "null")
        {
            ImGui::TextDisabled("%s", status.c_str());
        }
        ImGui::PopID();
    }
    ImGui::End();
}

void EditorApp::BuildShaderSection()
{
    ImGui::SeparatorText("Shaders");
    if (!mShaderReload->IsAvailable())
    {
        ImGui::TextDisabled("Hot reload unavailable: %s", mShaderReload->GetUnavailableReason().c_str());
        return;
    }
    ImGui::Checkbox("Hot reload", &mShaderHotReload);
    ImGui::SetItemTooltip("Watches %zu HLSL files of %zu shaders. Saving one recompiles the shaders that use it\n"
                          "with the build's dxc command line and reloads them. A compile error keeps the\n"
                          "previous shaders running and is listed in the Log.",
                          mShaderReload->GetWatchedFileCount(), mShaderReload->GetShaderCount());
    ImGui::SameLine();
    if (ImGui::Button("Recompile all"))
    {
        mShaderReload->RecompileAll();
        mShaderRecompileRequested = true;
    }
    ImGui::SetItemTooltip("Ctrl+R");
    if (mShaderReload->IsCompiling())
    {
        ImGui::SameLine();
        ImGui::TextDisabled("compiling...");
    }
    if (!mShaderStatus.empty())
    {
        ImGui::TextColored(mShaderFailed ? ImVec4(1.0f, 0.4f, 0.35f, 1.0f) : ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%s",
                           mShaderStatus.c_str());
    }
}

void EditorApp::BuildDebugPanel()
{
    if (!ImGui::Begin("Debug", &mShowDebug))
    {
        ImGui::End();
        return;
    }
    renderer::RenderFeature* pt = mRunner->FindFeature("pathtracer");
    if (pt)
    {
        ImGui::SeparatorText("Path tracer view");
        static const char* const kViews[] = {"Lit",   "Normals",       "Base color", "Roughness",    "Metallic",
                                             "Depth", "Intersections", "Bounces",    "Direct light", "Indirect light"};
        u32* view = nullptr;
        bool* furnace = nullptr;
        bool* nee = nullptr;
        for (const auto& entry : pt->Parameters().Entries())
        {
            if (entry.name == "debugView")
            {
                view = *std::get_if<u32*>(&entry.value);
            }
            else if (entry.name == "furnace")
            {
                furnace = *std::get_if<bool*>(&entry.value);
            }
            else if (entry.name == "nee")
            {
                nee = *std::get_if<bool*>(&entry.value);
            }
        }
        bool changed = false;
        // Two columns, the second placed after the widest label so nothing is clipped at any DPI.
        float widest = 0.0f;
        for (const char* name : kViews)
        {
            widest = std::max(widest, ImGui::CalcTextSize(name).x);
        }
        const ImGuiStyle& style = ImGui::GetStyle();
        const float secondColumn = style.WindowPadding.x + ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + widest +
                                   style.ItemSpacing.x * 2.0f;
        for (u32 i = 0; view && i < std::size(kViews); ++i)
        {
            if (i % 2 == 1)
            {
                ImGui::SameLine(secondColumn);
            }
            if (ImGui::RadioButton(kViews[i], *view == i))
            {
                *view = i;
                changed = true;
            }
        }
        ImGui::SeparatorText("Energy and sampling checks");
        if (furnace && ImGui::Checkbox("White furnace", furnace))
        {
            changed = true;
        }
        ImGui::SetItemTooltip("Every material white, lights off, environment mapped to 18%% grey.\n"
                              "An energy-conserving renderer shows a flat grey image: darker objects lose\n"
                              "energy, brighter ones create it. Best with Scene > furnace.");
        if (nee && ImGui::Checkbox("Next event estimation + MIS", nee))
        {
            changed = true;
        }
        ImGui::SetItemTooltip("Off: BSDF sampling only. Both converge to the same image; NEE is less noisy.");
        if (changed)
        {
            pt->Parameters().Touch();
        }
    }
    ImGui::SeparatorText("Frame");
    const auto& stats = mLastResult.graphStats;
    ImGui::Text("Render graph: %u passes (%u culled), %u barriers, %u command lists", stats.passCount,
                stats.culledPassCount, stats.barrierCount, stats.commandListCount);
    ImGui::Text("Transient memory: %s (%s without aliasing)", renderer::FormatBytes(stats.transientHeapSize).c_str(),
                renderer::FormatBytes(stats.transientBytesUnaliased).c_str());
    if (mRunner)
    {
        rdg::RenderGraph::Config config = mRunner->GetDesc().graphConfig;
        if (ImGui::Checkbox("Async compute", &config.enableAsyncCompute))
        {
            mRunner->SetGraphConfig(config);
        }
        ImGui::SetItemTooltip("Run passes marked AsyncCompute on the compute queue, overlapping graphics work.\n"
                              "Off: they run on the graphics queue in declaration order (for A/B timing).");
        ImGui::SameLine();
        ImGui::TextDisabled("%u async passes, %u submissions, %u queue waits", stats.asyncComputePassCount,
                            stats.submissionCount, stats.queueWaitCount);
        const auto& pool = mLastResult.transientPool;
        ImGui::Text("Transients: %u in %s (%s unaliased); pool %u heaps, %s, %u resources",
                    stats.transientResourceCount, renderer::FormatBytes(stats.transientHeapSize).c_str(),
                    renderer::FormatBytes(stats.transientBytesUnaliased).c_str(), pool.heaps,
                    renderer::FormatBytes(pool.heapBytes).c_str(), pool.resources);
        ImGui::SetItemTooltip("Transient memory comes from a pool kept across frames: this frame created %u heaps and "
                              "%u resources and reused %u resources of earlier frames.",
                              stats.transientHeapsCreated, stats.transientResourcesCreated,
                              stats.transientResourcesReused);
    }
    ImGui::Text("Validation errors this frame: %zu", mLastResult.validationErrors.size());
    ImGui::End();
}

void EditorApp::BuildLogPanel()
{
    if (!ImGui::Begin("Log", &mShowLog))
    {
        ImGui::End();
        return;
    }
    if (ImGui::Button("Clear"))
    {
        std::lock_guard lock(mLogMutex);
        mLog.clear();
    }
    ImGui::SameLine();
    ImGui::Checkbox("Auto-scroll", &mLogAutoScroll);
    ImGui::BeginChild("lines", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
    {
        std::lock_guard lock(mLogMutex);
        for (const LogLine& line : mLog)
        {
            const ImVec4 color = line.level >= LogLevel::Error     ? ImVec4(1.0f, 0.4f, 0.35f, 1.0f)
                                 : line.level == LogLevel::Warning ? ImVec4(1.0f, 0.8f, 0.3f, 1.0f)
                                                                   : ImGui::GetStyleColorVec4(ImGuiCol_Text);
            ImGui::TextColored(color, "%s", line.text.c_str());
        }
    }
    if (mLogAutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
    {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    ImGui::End();
}

} // namespace air::editor
