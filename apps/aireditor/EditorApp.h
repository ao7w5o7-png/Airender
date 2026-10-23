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

#include "air/core/Image.h"
#include "air/core/TaskSystem.h"
#include "air/renderer/FrameRunner.h"
#include "air/renderer/Scene.h"

#include <chrono>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace air::editor
{

struct EditorOptions
{
    std::filesystem::path shaderRoot = AIR_DEFAULT_SHADER_DIR;
    std::string scene = "cornell_box";
    std::vector<std::string> features = {"pathtracer", "tonemap"}; // "dlss" is inserted when available
    u32 width = 1280; // initial render resolution; follows the viewport afterwards
    u32 height = 720;
};

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
    rhi::TextureHandle GetOutput() const { return mRunner ? mRunner->GetOutput() : rhi::TextureHandle{}; }
    renderer::FrameRunner* GetRunner() const { return mRunner.get(); }
    renderer::Camera& MutableCamera() { return mCamera; }
    const renderer::FrameResult& GetLastResult() const { return mLastResult; }
    const std::vector<std::string>& GetActiveFeatures() const { return mActiveFeatures; }

    // Saves the displayed frame as <path>.ppm (display-encoded) and <path>.pfm.
    bool SaveScreenshot(const std::filesystem::path& path);
    // Applies a resolution change now (the UI otherwise waits until the viewport stops resizing).
    void Resize(u32 width, u32 height);
    bool LoadScene(const std::string& name);

private:
    void CreateRunner(u32 width, u32 height);
    void BuildMenuBar();
    void BuildViewport(float deltaSeconds);
    void BuildScenePanel();
    void BuildRenderPanel();
    void BuildDebugPanel();
    void BuildLogPanel();
    void UpdateCamera(float deltaSeconds);
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
    bool mLockResolution = false;
    float mResolutionScale = 1.0f;
    float mMoveSpeed = 1.5f; // metres per second
    bool mViewportHovered = false;

    // Windows.
    bool mShowScene = true;
    bool mShowRender = true;
    bool mShowDebug = true;
    bool mShowProfiler = true;
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
