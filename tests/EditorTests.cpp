// Drives the editor UI headless: a Dear ImGui context without platform/renderer backends, the
// Null device, and the real EditorApp. Catches ImGui asserts (ID clashes, unbalanced Begin/End),
// broken panels and render graph validation errors in editor-driven frames.

#include "TestHelpers.h"

#include "EditorApp.h"

#include <imgui.h>

using namespace air;
using namespace airtest;

namespace
{

class HeadlessUi
{
public:
    HeadlessUi()
    {
        IMGUI_CHECKVERSION();
        mContext = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.DisplaySize = ImVec2(1600.0f, 900.0f);
        io.DeltaTime = 1.0f / 60.0f;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures; // texture requests are simply not uploaded
    }
    ~HeadlessUi() { ImGui::DestroyContext(mContext); }
    HeadlessUi(const HeadlessUi&) = delete;
    HeadlessUi& operator=(const HeadlessUi&) = delete;

    void Frame(editor::EditorApp& app, float deltaSeconds = 1.0f / 60.0f)
    {
        ImGui::GetIO().DeltaTime = deltaSeconds;
        ImGui::NewFrame();
        app.BuildUi(deltaSeconds);
        ImGui::Render();
        // Pretend the backend uploaded every requested texture.
        for (ImTextureData* texture : ImGui::GetPlatformIO().Textures)
        {
            if (texture->Status == ImTextureStatus_WantCreate || texture->Status == ImTextureStatus_WantUpdates)
            {
                texture->SetTexID(ImTextureID(1));
                texture->SetStatus(ImTextureStatus_OK);
            }
            else if (texture->Status == ImTextureStatus_WantDestroy)
            {
                texture->SetTexID(ImTextureID_Invalid);
                texture->SetStatus(ImTextureStatus_Destroyed);
            }
        }
        app.RenderFrame();
    }

private:
    ImGuiContext* mContext = nullptr;
};

editor::EditorOptions SmallOptions()
{
    editor::EditorOptions options;
    options.width = 96;
    options.height = 64;
    return options;
}

} // namespace

TEST(Editor_RunsFramesHeadless)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    {
        HeadlessUi ui;
        editor::EditorApp app(*device, tasks, SmallOptions());
        EXPECT(app.GetActiveFeatures().size() == 2);
        for (int i = 0; i < 4; ++i)
        {
            ui.Frame(app);
            EXPECT(app.GetLastResult().validationErrors.empty());
        }
        EXPECT(app.GetLastResult().graphText.empty()); // the editor does not capture graph dumps
        EXPECT(!app.GetLastResult().passCpu.empty());
    }
    device->WaitIdle();
    EXPECT(device->GetMemoryStats().liveTextures == 0);
}

TEST(Editor_ResizeKeepsParameters)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    HeadlessUi ui;
    editor::EditorApp app(*device, tasks, SmallOptions());
    EXPECT(app.GetRunner()->SetParameter("pathtracer.maxBounces", "3"));
    EXPECT(app.GetRunner()->SetParameter("tonemap.operator", "2"));
    app.Resize(128, 72);
    EXPECT(app.GetRunner()->GetDesc().width == 128 && app.GetRunner()->GetDesc().height == 72);
    for (const auto& entry : app.GetRunner()->FindFeature("pathtracer")->Parameters().Entries())
    {
        if (entry.name == "maxBounces")
        {
            EXPECT(**std::get_if<u32*>(&entry.value) == 3);
        }
    }
    ui.Frame(app);
    EXPECT(app.GetLastResult().validationErrors.empty());
}

TEST(Editor_SwitchesScenesAndDebugViews)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    HeadlessUi ui;
    editor::EditorApp app(*device, tasks, SmallOptions());
    EXPECT(app.LoadScene("furnace"));
    EXPECT(!app.LoadScene("missing"));
    EXPECT(app.GetRunner()->SetParameter("pathtracer.furnace", "true"));
    for (int view = 0; view < 10; ++view)
    {
        EXPECT(app.GetRunner()->SetParameter("pathtracer.debugView", std::to_string(view)));
        ui.Frame(app);
        EXPECT(app.GetLastResult().validationErrors.empty());
    }
    app.MutableCamera().yaw += 0.5f;
    ui.Frame(app);
    const auto path = std::filesystem::temp_directory_path() / "airender_editor_test";
    EXPECT(app.SaveScreenshot(path));
    EXPECT(std::filesystem::exists(path.string() + ".ppm") && std::filesystem::exists(path.string() + ".pfm"));
}

// Paused, the profiler keeps the captured frame while frames keep rendering; Step captures one more.
TEST(Editor_ProfilerPauseFreezesCapturedFrame)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    HeadlessUi ui;
    editor::EditorApp app(*device, tasks, SmallOptions());
    for (int i = 0; i < 3; ++i)
    {
        ui.Frame(app);
    }
    editor::DiagnosticsPanels& diagnostics = app.Diagnostics();
    diagnostics.SetPaused(true);
    const u64 captured = diagnostics.GetCapturedFrame();
    for (int i = 0; i < 3; ++i)
    {
        ui.Frame(app); // the panels are built paused: no ImGui asserts, no validation errors
        EXPECT(app.GetLastResult().validationErrors.empty());
    }
    EXPECT(diagnostics.GetCapturedFrame() == captured);
    diagnostics.StepFrame();
    ui.Frame(app);
    EXPECT(diagnostics.GetCapturedFrame() == captured + 4);
    ui.Frame(app);
    EXPECT(diagnostics.GetCapturedFrame() == captured + 4); // still paused after the step
    EXPECT(diagnostics.GetFps() > 0.0f);                    // the live frame rate keeps updating
    diagnostics.SetPaused(false);
    ui.Frame(app);
    EXPECT(diagnostics.GetCapturedFrame() == captured + 6);
}

TEST(Editor_VsyncOption)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    HeadlessUi ui;
    editor::EditorOptions options = SmallOptions();
    options.vsync = false;
    editor::EditorApp app(*device, tasks, options);
    EXPECT(!app.GetVsync());
    ui.Frame(app);
    app.SetVsync(true);
    EXPECT(app.GetVsync());
    ui.Frame(app);
}

// Turning DLSS off must restore progressive accumulation: jittered 1 spp frames shimmer without it.
TEST(Editor_DlssToggleRestoresAccumulation)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    HeadlessUi ui;
    editor::EditorApp app(*device, tasks, SmallOptions());
    EXPECT(app.GetRunner()->SetParameter("pathtracer.spp", "8"));
    app.SetDlssEnabled(true); // on the Null device the dlss feature declines to initialize
    EXPECT(app.IsDlssEnabled());
    EXPECT(app.GetParameter("pathtracer", "accumulate") == "false" && app.GetParameter("pathtracer", "spp") == "1");
    ui.Frame(app);
    app.SetDlssEnabled(false);
    EXPECT(!app.IsDlssEnabled());
    EXPECT(app.GetParameter("pathtracer", "accumulate") == "true" && app.GetParameter("pathtracer", "spp") == "8");
    ui.Frame(app);
    EXPECT(app.GetLastResult().validationErrors.empty());
}

// One slow frame (pipelines compiling) must not commit the viewport size of that frame: the render
// resolution only follows the viewport once it has been stable for a while and a few frames.
TEST(Editor_ResizeWaitsForAStableViewport)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    HeadlessUi ui;
    editor::EditorApp app(*device, tasks, SmallOptions());
    ui.Frame(app, 2.0f); // a 2 s frame: measures the viewport, which is not 96x64
    ui.Frame(app, 2.0f);
    EXPECT(app.GetRunner()->GetDesc().width == 96 && app.GetRunner()->GetDesc().height == 64);
    for (int i = 0; i < 30; ++i)
    {
        ui.Frame(app);
    }
    EXPECT(app.GetRunner()->GetDesc().width != 96); // followed the viewport after it settled
    EXPECT(app.GetLastResult().validationErrors.empty());
}

// DLSS modes set the runner's render scale; on the Null device dlss declines to initialize, so the
// frame goes through FrameRunner's bilinear upscale and must still be valid.
TEST(Editor_DlssModeSetsRenderScale)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    TaskSystem tasks(2);
    HeadlessUi ui;
    editor::EditorApp app(*device, tasks, SmallOptions());
    app.SetDlssEnabled(true);
    EXPECT(app.GetDlssMode() == 1); // Quality by default
    EXPECT(app.GetRunner()->GetRenderWidth() == 64 && app.GetRunner()->GetRenderHeight() == 43);
    app.SetDlssMode(3); // Performance
    EXPECT(app.GetRunner()->GetRenderWidth() == 48 && app.GetRunner()->GetRenderHeight() == 32);
    ui.Frame(app);
    EXPECT(app.GetLastResult().validationErrors.empty());
    app.SetDlssEnabled(false);
    EXPECT(app.GetRunner()->GetRenderWidth() == 96 && app.GetRunner()->GetRenderHeight() == 64);
}
