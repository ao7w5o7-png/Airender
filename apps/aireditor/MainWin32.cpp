// aireditor: Win32 window + DXGI swapchain + Dear ImGui (Win32 / DX12 backends) around EditorApp.
//
//   aireditor [--scene NAME] [--shaders DIR] [--features a,b,...] [--no-vsync] [--dlss]
//             [--dlss-mode dlaa|quality|balanced|performance|ultra] [--gpu-validation] [--no-validation]
//
// --features replaces the feature chain (default pathtracer,tonemap); dlss is inserted after the
// path tracer when enabled.
//
// The UI is drawn by one render graph pass at the end of every frame, into the swapchain back
// buffer, reading the frame output as a shader resource. ImGui's textures (font atlas) and the
// viewport image live in the RHI's bindless descriptor heap, so ImGui and the renderer share one
// shader-visible heap.

#include "EditorApp.h"

#include "air/core/Log.h"
#include "air/rhi/RHIDevice.h"

#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_win32.h>

#include <windows.h>

#include <d3d12.h>
#include <shellapi.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace
{

using namespace air;

struct WindowState
{
    u32 width = 1600;
    u32 height = 900;
    bool resized = false;
    bool minimized = false;
};
WindowState gWindow;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam))
    {
        return 1;
    }
    switch (msg)
    {
    case WM_SIZE:
        gWindow.minimized = wParam == SIZE_MINIMIZED;
        if (!gWindow.minimized)
        {
            gWindow.width = LOWORD(lParam);
            gWindow.height = HIWORD(lParam);
            gWindow.resized = true;
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xFFF0) == SC_KEYMENU)
        {
            return 0; // no ALT application menu
        }
        break;
    case WM_DPICHANGED:
    {
        const RECT* suggested = reinterpret_cast<const RECT*>(lParam);
        SetWindowPos(hwnd, nullptr, suggested->left, suggested->top, suggested->right - suggested->left,
                     suggested->bottom - suggested->top, SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_DESTROY: PostQuitMessage(0); return 0;
    default: break;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// ImGui's SRV descriptors are slots of the bindless heap.
rhi::Device* gDevice = nullptr;
std::unordered_map<u64, rhi::BindlessIndex> gImGuiSlots;

void AllocateImGuiSrv(ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* cpu, D3D12_GPU_DESCRIPTOR_HANDLE* gpu)
{
    const rhi::BindlessIndex slot = gDevice->AllocateBindlessSlot();
    const rhi::DescriptorHandles handles = gDevice->GetBindlessDescriptor(slot);
    cpu->ptr = SIZE_T(handles.cpu);
    gpu->ptr = handles.gpu;
    gImGuiSlots[handles.gpu] = slot;
}

void FreeImGuiSrv(ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE gpu)
{
    const auto it = gImGuiSlots.find(gpu.ptr);
    if (it != gImGuiSlots.end())
    {
        gDevice->FreeBindlessSlot(it->second); // released once the GPU is done with it
        gImGuiSlots.erase(it);
    }
}

std::string Narrow(const wchar_t* text)
{
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string out(size > 0 ? size - 1 : 0, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), size, nullptr, nullptr);
    return out;
}

void ApplyStyle(float dpiScale)
{
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.ScaleAllSizes(dpiScale);
    style.FontScaleDpi = dpiScale;
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand)
{
    editor::EditorOptions options;
    rhi::DeviceDesc deviceDesc{.backend = rhi::Backend::D3D12};
    bool dlss = false;
#ifdef NDEBUG
    deviceDesc.enableValidation = false;
#endif
    {
        int argc = 0;
        LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = Narrow(argv[i]);
            if (arg == "--scene" && i + 1 < argc)
            {
                options.scene = Narrow(argv[++i]);
            }
            else if (arg == "--shaders" && i + 1 < argc)
            {
                options.shaderRoot = Narrow(argv[++i]);
            }
            else if (arg == "--features" && i + 1 < argc)
            {
                options.features.clear();
                const std::string list = Narrow(argv[++i]);
                for (size_t begin = 0; begin <= list.size();)
                {
                    const size_t end = std::min(list.find(',', begin), list.size());
                    if (end > begin)
                    {
                        options.features.push_back(list.substr(begin, end - begin));
                    }
                    begin = end + 1;
                }
            }
            else if (arg == "--dlss")
            {
                dlss = true;
            }
            else if (arg == "--dlss-mode" && i + 1 < argc)
            {
                const std::string mode = Narrow(argv[++i]);
                for (u32 m = 0; m < std::size(editor::kDlssModes); ++m)
                {
                    std::string name = editor::kDlssModes[m].name;
                    std::erase(name, ' ');
                    if (_stricmp(name.c_str(), mode.c_str()) == 0 ||
                        (mode == "ultra" && m + 1 == std::size(editor::kDlssModes)))
                    {
                        options.dlssMode = m;
                    }
                }
            }
            else if (arg == "--no-vsync")
            {
                options.vsync = false;
            }
            else if (arg == "--gpu-validation")
            {
                deviceDesc.enableValidation = true;
                deviceDesc.enableGpuValidation = true;
            }
            else if (arg == "--no-validation")
            {
                deviceDesc.enableValidation = false;
            }
        }
        LocalFree(argv);
    }
    // Next to the executable first (installed layout), then the build tree.
    wchar_t exePath[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    const std::filesystem::path besideExe = std::filesystem::path(exePath).parent_path() / "shaders";
    if (std::filesystem::exists(besideExe))
    {
        options.shaderRoot = besideExe;
    }

    ImGui_ImplWin32_EnableDpiAwareness();
    const float dpiScale =
        ImGui_ImplWin32_GetDpiScaleForMonitor(MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));

    WNDCLASSEXW windowClass = {sizeof(windowClass)};
    windowClass.style = CS_CLASSDC;
    windowClass.lpfnWndProc = WndProc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.lpszClassName = L"AirenderEditor";
    RegisterClassExW(&windowClass);
    gWindow.width = u32(1600 * dpiScale);
    gWindow.height = u32(900 * dpiScale);
    HWND hwnd = CreateWindowW(windowClass.lpszClassName, L"Airender Editor", WS_OVERLAPPEDWINDOW, 100, 100,
                              int(gWindow.width), int(gWindow.height), nullptr, nullptr, instance, nullptr);

    auto device = rhi::CreateDevice(deviceDesc);
    if (!device)
    {
        MessageBoxW(hwnd, L"Could not create a D3D12 device with DXR 1.1 and shader model 6.6.", L"Airender", MB_OK);
        return 1;
    }
    gDevice = device.get();
    RECT client = {};
    GetClientRect(hwnd, &client);
    gWindow.width = u32(std::max<LONG>(client.right - client.left, 8));
    gWindow.height = u32(std::max<LONG>(client.bottom - client.top, 8));
    auto swapchain = device->CreateSwapchain(
        {.window = hwnd, .width = gWindow.width, .height = gWindow.height, .vsync = options.vsync});
    if (!swapchain)
    {
        return 1;
    }
    ShowWindow(hwnd, showCommand);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    io.ConfigDpiScaleFonts = true;
    ApplyStyle(dpiScale);
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX12_InitInfo init;
    init.Device = static_cast<ID3D12Device*>(device->GetNativeObject(rhi::NativeObject::Device));
    init.CommandQueue = static_cast<ID3D12CommandQueue*>(device->GetNativeObject(rhi::NativeObject::GraphicsQueue));
    init.NumFramesInFlight = int(device->GetFramesInFlight());
    init.RTVFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
    init.DSVFormat = DXGI_FORMAT_UNKNOWN;
    init.SrvDescriptorHeap =
        static_cast<ID3D12DescriptorHeap*>(device->GetNativeObject(rhi::NativeObject::ResourceDescriptorHeap));
    init.SrvDescriptorAllocFn = AllocateImGuiSrv;
    init.SrvDescriptorFreeFn = FreeImGuiSrv;
    ImGui_ImplDX12_Init(&init);

    TaskSystem tasks;
    {
        editor::EditorApp app(*device, tasks, options);
        if (dlss)
        {
            app.SetDlssEnabled(true);
        }
        auto last = std::chrono::steady_clock::now();
        bool running = true;
        while (running && !app.WantsQuit())
        {
            MSG msg;
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
                running &= msg.message != WM_QUIT;
            }
            if (!running)
            {
                break;
            }
            if (gWindow.minimized)
            {
                Sleep(16);
                continue;
            }
            if (swapchain->GetDesc().vsync != app.GetVsync())
            {
                swapchain->SetVsync(app.GetVsync());
            }
            if (gWindow.resized)
            {
                gWindow.resized = false;
                swapchain->Resize(gWindow.width, gWindow.height);
            }
            const auto now = std::chrono::steady_clock::now();
            const float dt = std::chrono::duration<float>(now - last).count();
            last = now;

            ImGui_ImplDX12_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
            app.BuildUi(dt);
            ImGui::Render();

            const rhi::TextureHandle backBuffer = swapchain->GetCurrentTexture();
            app.RenderFrame(
                [&](renderer::FrameContext& context)
                {
                    // Back buffers rest in Present between frames.
                    const rdg::TextureRef target =
                        context.graph.ImportTexture(backBuffer, rhi::Access::Present, rhi::Access::Present);
                    const rdg::TextureRef image = context.output;
                    context.graph.AddPass(
                        "Editor.UI", rdg::PassFlags::Raster | rdg::PassFlags::NeverCull,
                        [&](rdg::PassBuilder& b)
                        {
                            b.Read(image, rhi::Access::SRVGraphics);
                            b.Write(target, rhi::Access::RenderTarget);
                        },
                        [=](rdg::PassContext& ctx)
                        {
                            const float clear[4] = {0.08f, 0.08f, 0.09f, 1.0f};
                            ctx.cmd.BeginRenderPass(ctx.Get(target), clear);
                            auto* list = static_cast<ID3D12GraphicsCommandList*>(ctx.cmd.GetNativeHandle());
                            ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), list);
                            ctx.cmd.EndRenderPass();
                        });
                });
            if (!swapchain->Present())
            {
                AIR_LOG(Fatal, "Editor", "present failed (device removed?), see the log above");
                break;
            }
        }
        device->WaitIdle();
    }
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    swapchain.reset();
    device->WaitIdle();
    DestroyWindow(hwnd);
    UnregisterClassW(windowClass.lpszClassName, instance);
    return 0;
}
