#include "WebViewHost.h"

#include "air/core/Log.h"
#include "air/core/Types.h"

#include <objbase.h> // COM's `interface`, which WebView2.h uses (WIN32_LEAN_AND_MEAN leaves it out)

#include <WebView2.h>
#include <wrl.h>

#include <algorithm>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace air::editor
{

namespace
{

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;

std::wstring Widen(std::string_view text)
{
    if (text.empty())
    {
        return {};
    }
    const int length = MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), nullptr, 0);
    std::wstring out(size_t(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), out.data(), length);
    return out;
}

// What the asynchronous WebView2 callbacks share with their host, which they may outlive.
struct State
{
    HWND parent = nullptr;
    ComPtr<ICoreWebView2Controller> controller;
    ComPtr<ICoreWebView2> webView;
    bool loaded = false;      // the page has loaded: posted messages reach it
    std::wstring pendingPage; // to load once the web view exists
    std::wstring latestJson;  // the latest message, sent again when a page has loaded
    RECT bounds = {};         // the last visible bounds
    bool visible = false;
    std::string status = "starting the web view";
};

std::vector<State*> gStates; // live hosts, for NotifyWebViewParentMoved (UI thread only)

void Fail(State& state, std::string_view what, HRESULT hr)
{
    state.status = std::format("{} failed (0x{:08x})", what, static_cast<u32>(hr));
    AIR_LOG(Error, "Editor", "WebView2: {}", state.status);
}

void Attach(const std::shared_ptr<State>& state, ICoreWebView2Controller* controller)
{
    state->controller = controller;
    controller->get_CoreWebView2(&state->webView);
    ComPtr<ICoreWebView2Settings> settings;
    if (SUCCEEDED(state->webView->get_Settings(&settings)))
    {
        settings->put_IsStatusBarEnabled(FALSE);
        settings->put_AreDevToolsEnabled(TRUE); // F12 inspects the page
    }
    // The editor is dark: no white flash while loading, and the page's dark theme by default.
    ComPtr<ICoreWebView2Controller2> controller2;
    if (SUCCEEDED(controller->QueryInterface(IID_PPV_ARGS(&controller2))))
    {
        controller2->put_DefaultBackgroundColor({255, 13, 13, 13});
    }
    ComPtr<ICoreWebView2_13> webView13;
    ComPtr<ICoreWebView2Profile> profile;
    if (SUCCEEDED(state->webView.As(&webView13)) && SUCCEEDED(webView13->get_Profile(&profile)))
    {
        profile->put_PreferredColorScheme(COREWEBVIEW2_PREFERRED_COLOR_SCHEME_DARK);
    }
    std::weak_ptr<State> weak = state;
    EventRegistrationToken token = {};
    state->webView->add_NavigationCompleted(
        Callback<ICoreWebView2NavigationCompletedEventHandler>(
            [weak](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs* args) -> HRESULT
            {
                const std::shared_ptr<State> s = weak.lock();
                if (!s)
                {
                    return S_OK;
                }
                BOOL success = FALSE;
                args->get_IsSuccess(&success);
                s->loaded = success;
                s->status = success ? "ready" : "the page did not load";
                if (success && !s->latestJson.empty())
                {
                    s->webView->PostWebMessageAsJson(s->latestJson.c_str());
                }
                return S_OK;
            })
            .Get(),
        &token);
    controller->put_Bounds(state->bounds);
    controller->put_IsVisible(state->visible);
    state->status = "loading the page";
    if (!state->pendingPage.empty())
    {
        state->webView->NavigateToString(state->pendingPage.c_str());
        state->pendingPage.clear();
    }
}

class WebViewHost final : public HtmlView
{
public:
    WebViewHost(HWND parent, const std::filesystem::path& userData) : mState(std::make_shared<State>())
    {
        mState->parent = parent;
        gStates.push_back(mState.get());
        std::error_code error;
        std::filesystem::create_directories(userData, error);
        std::weak_ptr<State> weak = mState;
        const HRESULT hr = CreateCoreWebView2EnvironmentWithOptions(
            nullptr, userData.c_str(), nullptr,
            Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
                [weak](HRESULT result, ICoreWebView2Environment* environment) -> HRESULT
                {
                    const std::shared_ptr<State> state = weak.lock();
                    if (!state)
                    {
                        return S_OK;
                    }
                    if (FAILED(result) || !environment)
                    {
                        Fail(*state, "creating the WebView2 environment", result);
                        return S_OK;
                    }
                    const HRESULT created = environment->CreateCoreWebView2Controller(
                        state->parent, Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                                           [weak](HRESULT result, ICoreWebView2Controller* controller) -> HRESULT
                                           {
                                               const std::shared_ptr<State> state = weak.lock();
                                               if (!state)
                                               {
                                                   if (controller)
                                                   {
                                                       controller->Close(); // the host is gone
                                                   }
                                                   return S_OK;
                                               }
                                               if (FAILED(result) || !controller)
                                               {
                                                   Fail(*state, "creating the WebView2 control", result);
                                                   return S_OK;
                                               }
                                               Attach(state, controller);
                                               return S_OK;
                                           })
                                           .Get());
                    if (FAILED(created))
                    {
                        Fail(*state, "creating the WebView2 control", created);
                    }
                    return S_OK;
                })
                .Get());
        if (FAILED(hr))
        {
            Fail(*mState, "starting WebView2 (is the WebView2 runtime installed?)", hr);
        }
    }

    ~WebViewHost() override
    {
        std::erase(gStates, mState.get());
        if (mState->controller)
        {
            mState->controller->Close();
        }
        mState->webView.Reset();
        mState->controller.Reset();
    }

    WebViewHost(const WebViewHost&) = delete;
    WebViewHost& operator=(const WebViewHost&) = delete;

    void SetBounds(int x, int y, int width, int height, bool visible) override
    {
        State& s = *mState;
        if (visible)
        {
            const RECT bounds = {x, y, x + std::max(width, 0), y + std::max(height, 0)};
            if (s.controller && !EqualRect(&bounds, &s.bounds))
            {
                s.controller->put_Bounds(bounds);
            }
            s.bounds = bounds;
        }
        if (s.controller && visible != s.visible)
        {
            s.controller->put_IsVisible(visible);
        }
        s.visible = visible;
    }

    void LoadPage(const std::string& html) override
    {
        State& s = *mState;
        s.loaded = false;
        if (s.webView)
        {
            s.webView->NavigateToString(Widen(html).c_str());
        }
        else
        {
            s.pendingPage = Widen(html);
        }
    }

    void PostJson(const std::string& json) override
    {
        State& s = *mState;
        s.latestJson = Widen(json);
        if (s.webView && s.loaded)
        {
            s.webView->PostWebMessageAsJson(s.latestJson.c_str());
        }
    }

    std::string GetStatus() const override { return mState->status; }

private:
    std::shared_ptr<State> mState;
};

} // namespace

std::unique_ptr<HtmlView> CreateWebViewHost(HWND parent, const std::filesystem::path& userData)
{
    return std::make_unique<WebViewHost>(parent, userData);
}

void NotifyWebViewParentMoved()
{
    for (State* state : gStates)
    {
        if (state->controller)
        {
            state->controller->NotifyParentWindowPositionChanged();
        }
    }
}

} // namespace air::editor
