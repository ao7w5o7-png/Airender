#pragma once

// The HTML profiler panel's web view on Windows: a WebView2 control in a child window of the
// editor's main window, kept over the panel's content rectangle (HtmlView.h). WebView2 needs a
// single-threaded COM apartment on the window's thread and a running message loop.

#include "HtmlView.h"

#include <windows.h>

#include <filesystem>
#include <memory>

namespace air::editor
{

// `userData` is WebView2's profile folder (cache, settings), created when missing.
std::unique_ptr<HtmlView> CreateWebViewHost(HWND parent, const std::filesystem::path& userData);
// The parent window moved: lets the web views place their popups (context menus) correctly.
void NotifyWebViewParentMoved();

} // namespace air::editor
