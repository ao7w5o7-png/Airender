#pragma once

// The editor's platform-independent core: owns the scene, the camera and the FrameRunner, builds
// every ImGui window and drives one frame at a time. The platform layer (MainWin32.cpp) only
// provides the window, the swapchain and the ImGui backends; tests drive it headless on the Null
// device.
//
//   EditorApp app(device, tasks, options);
//   per frame: ImGui::NewFrame(); app.BuildUi(dt); ImGui::Render();
//              app.RenderFrame([&](FrameContext& c) { /* add the swapchain UI pass */ });

#include "DiagnosticsPanels.h"
#include "HtmlView.h"

#include "air/core/Image.h"
#include "air/core/TaskSystem.h"
#include "air/renderer/FrameRunner.h"
#include "air/renderer/Scene.h"
#include "air/renderer/ShaderHotReload.h"

#include <chrono>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace air::editor
{

// DLSS Ray Reconstruction modes: output / render ratio as in NGX (DLAA renders at the output size).
struct DlssMode
{
    const char* name;
    float renderScale;
};
inline constexpr DlssMode kDlssModes[] = {
    {"DLAA", 1.0f},        {"Quality", 1.0f / 1.5f},           {"Balanced", 0.58f},
    {"Performance", 0.5f}, {"Ultra Performance", 1.0f / 3.0f},
};

struct EditorOptions
{
    std::filesystem::path shaderRoot = AIR_DEFAULT_SHADER_DIR;
    std::string scene = "cornell_box";
    std::vector<std::string> features = {"pathtracer", "tonemap"}; // "dlss" is inserted when available
    u32 width = 1280; // initial render resolution; follows the viewport afterwards
    u32 height = 720;
    bool vsync = true; // off: present as fast as possible (uncapped frame rate, tearing when supported)
    u32 dlssMode = 1;  // index into kDlssModes, used while dlss is enabled (Quality)
    // Opens a written file (the profile report) in the system's default application. Unset in tests.
    std::function<void(const std::filesystem::path&)> openFile;
    std::filesystem::path profileDirectory = "out/profiles"; // where ExportProfileReport() writes
    // The web view of the HTML profiler panel (WebView2 on Windows); unset, the panel offers the browser.
    std::function<std::unique_ptr<HtmlView>()> createHtmlView;
    float htmlProfilerInterval = 1.0f; // seconds between the panel's live captures
    std::string focusWindow;           // a window brought to front at startup (selects its dock tab)
};

inline constexpr const char* kHtmlProfilerWindow = "Profiler (HTML)";

struct LogLine
{
    LogLevel level = LogLevel::Info;
    std::string text;
};

class EditorApp
{
public:
    EditorApp(rhi::Device& device, TaskSystem& tasks, EditorOptions options);
    ~EditorApp();
    EditorApp(const EditorApp&) = delete;
    EditorApp& operator=(const EditorApp&) = delete;

    // Builds all windows. Call between ImGui::NewFrame() and ImGui::Render().
    void BuildUi(float deltaSeconds);
    // Updates the scene, renders a frame and lets `uiPass` add passes (the swapchain UI) to the graph.
    void RenderFrame(const std::function<void(renderer::FrameContext&)>& uiPass = {});

    bool WantsQuit() const { return mQuit; }
    // Presentation sync chosen in the UI; the platform layer applies it to its swapchain.
    bool GetVsync() const { return mOptions.vsync; }
    void SetVsync(bool vsync) { mOptions.vsync = vsync; }
    DiagnosticsPanels& Diagnostics() { return mDiagnostics; }
    rhi::TextureHandle GetOutput() const { return mRunner ? mRunner->GetOutput() : rhi::TextureHandle{}; }
    renderer::FrameRunner* GetRunner() const { return mRunner.get(); }
    renderer::Camera& MutableCamera() { return mCamera; }
    const renderer::FrameResult& GetLastResult() const { return mLastResult; }
    const std::vector<std::string>& GetActiveFeatures() const { return mActiveFeatures; }

    // Saves the displayed frame as <path>.ppm (display-encoded) and <path>.pfm.
    bool SaveScreenshot(const std::filesystem::path& path);
    // Captures the next frame and, once its GPU timings arrive a few frames later, writes its HTML
    // profile report (renderer/ProfileReport.h) to <profileDirectory>/<scene>-frame<N>.html and opens it.
    void ExportProfileReport();
    const std::filesystem::path& GetLastProfileReport() const { return mLastProfileReport; } // empty until written
    // The HTML profiler panel shows the same report in a web view, refreshed every
    // htmlProfilerInterval seconds while it is visible ("Live") and the profiler is not paused.
    void SetHtmlProfilerLive(bool live) { mHtmlLive = live; }
    HtmlView* GetHtmlView() const { return mHtmlView.get(); }
    // Brings a window to front on the next frame; for a docked one this selects its tab.
    void FocusWindow(std::string name) { mFocusWindow = std::move(name); }
    // Applies a resolution change now (the UI otherwise waits until the viewport stops resizing).
    void Resize(u32 width, u32 height);
    bool LoadScene(const std::string& name);
    // Inserts or removes the dlss feature after pathtracer. Enabling switches the path tracer to
    // jittered single frames (accumulate=false, spp=1); disabling restores what was set before.
    void SetDlssEnabled(bool enabled);
    bool IsDlssEnabled() const;
    // Index into kDlssModes; recreates the runner at the mode's render scale while dlss is enabled.
    void SetDlssMode(u32 mode);
    u32 GetDlssMode() const { return mOptions.dlssMode; }
    // A feature parameter as text ("" if the feature or parameter is missing).
    std::string GetParameter(std::string_view feature, std::string_view name) const;
    // Shader hot reload: polled every BuildUi(); a successful compile recreates the runner.
    renderer::ShaderHotReload& GetShaderReload() { return *mShaderReload; }
    bool HasShaderError() const { return mShaderFailed; }

private:
    void CreateRunner(u32 width, u32 height);
    void BuildMenuBar();
    void BuildViewport(float deltaSeconds);
    void BuildScenePanel();
    void BuildRenderPanel();
    void BuildDebugPanel();
    void BuildLogPanel();
    void UpdateCamera(float deltaSeconds);
    void PollShaderReload();
    void BuildShaderSection();
    void BuildHtmlProfiler();
    void FinishProfileReport();
    std::string SetParameter(const std::string& qualified, const std::string& value);

    rhi::Device& mDevice;
    TaskSystem& mTasks;
    EditorOptions mOptions;
    std::unique_ptr<renderer::Scene> mScene;
    std::string mSceneName;
    renderer::Camera mCamera;
    std::unique_ptr<renderer::FrameRunner> mRunner;
    std::vector<std::string> mActiveFeatures;
    renderer::FrameResult mLastResult;
    std::optional<std::chrono::steady_clock::time_point> mLastFrameTime;
    DiagnosticsPanels mDiagnostics;

    // Viewport and camera control.
    u32 mPendingWidth = 0;
    u32 mPendingHeight = 0;
    float mResizeTimer = 0.0f;
    u32 mSettledFrames = 0; // frames the pending size has been unchanged
    u32 mSettlingWidth = 0; // pending size the timer is running for
    u32 mSettlingHeight = 0;
    bool mLockResolution = false;
    float mResolutionScale = 1.0f;
    float mMoveSpeed = 1.5f; // metres per second
    bool mViewportHovered = false;
    // The sky as being edited: applied to the scene when a control is released, since every change
    // rebuilds the sky table (Scene::SetSky).
    std::optional<renderer::Sky> mSkyEdit;
    // Path tracer settings from before DLSS was enabled, restored when it is disabled.
    std::string mPreDlssAccumulate;
    std::string mPreDlssSpp;

    // Shader hot reload of the build's HLSL sources (see renderer::ShaderHotReload).
    std::unique_ptr<renderer::ShaderHotReload> mShaderReload;
    bool mShaderHotReload = true;           // watch the sources
    bool mShaderRecompileRequested = false; // Recompile all / Ctrl+R, also while not watching
    bool mShaderFailed = false;             // the last compile failed; the previous shaders still run
    std::string mShaderStatus;

    // Profile reports: where the next capture goes, and the captured frame waiting for its GPU timings.
    static constexpr u32 kReportFile = 1;  // written and opened (ExportProfileReport)
    static constexpr u32 kReportPanel = 2; // sent to the HTML profiler panel
    u32 mReportRequest = 0;
    u32 mPendingTargets = 0;
    std::shared_ptr<const renderer::FrameCapture> mPendingCapture;
    u32 mPendingCaptureFrames = 0;
    std::filesystem::path mLastProfileReport;
    std::unique_ptr<HtmlView> mHtmlView;
    bool mHtmlLive = true;
    std::chrono::steady_clock::time_point mLastHtmlCapture;
    std::string mFocusWindow;

    // Windows.
    bool mShowScene = true;
    bool mShowRender = true;
    bool mShowDebug = true;
    bool mShowProfiler = true;
    bool mShowHtmlProfiler = true;
    bool mShowMemory = true;
    bool mShowLog = true;
    bool mShowImGuiDemo = false;
    bool mQuit = false;
    bool mDockLayoutBuilt = false;
    u32 mScreenshotCounter = 0;

    // Log lines come from any thread through the log sink.
    std::mutex mLogMutex;
    std::deque<LogLine> mLog;
    bool mLogAutoScroll = true;
};

} // namespace air::editor
