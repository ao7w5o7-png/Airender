#pragma once

// A web page shown over an ImGui panel, for the editor's HTML profiler. The platform layer
// implements it (WebViewHost.cpp: WebView2 on Windows) and hands it to EditorApp through
// EditorOptions::createHtmlView; tests use a fake. Without one the panel offers the browser.

#include <string>

namespace air::editor
{

class HtmlView
{
public:
    virtual ~HtmlView() = default;

    // The panel's content rectangle in client-area pixels; hidden while `visible` is false (its tab
    // is not shown, it is closed, or an ImGui popup would be covered by it).
    virtual void SetBounds(int x, int y, int width, int height, bool visible) = 0;
    // Loads a page; messages posted before it has loaded are kept (the latest one) until it has.
    virtual void LoadPage(const std::string& html) = 0;
    // Sends JSON to the page's window.chrome.webview "message" listeners.
    virtual void PostJson(const std::string& json) = 0;
    // A short state for the panel's toolbar: starting, loading, ready, or what failed.
    virtual std::string GetStatus() const = 0;
};

} // namespace air::editor
