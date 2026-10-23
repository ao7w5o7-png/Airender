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

    void Frame(editor::EditorApp& app)
    {
        ImGui::NewFrame();
        app.BuildUi(1.0f / 60.0f);
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
