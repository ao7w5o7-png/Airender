#pragma once

// Shader hot reload. Watches the HLSL sources (and the headers they include) listed in
// <shader root>/manifest.json, which CMake writes at configure time (cmake/AirShaders.cmake), and
// recompiles the shaders whose files changed with exactly the build's compiler and arguments.
//
// Compilation runs on a background thread into temporary files; the outputs are replaced only
// when every changed shader compiled, so a typo never leaves a half-updated set and the previous
// shaders keep running. The caller recreates its pipelines when a compile succeeded (the editor
// recreates its FrameRunner, which reloads every feature's shaders).
//
//   ShaderHotReload reload(shaderRoot);
//   per frame: if (auto result = reload.Poll(); result && result->success) RecreatePipelines();
//
// Headers named *Shared.h are also compiled into C++: a struct layout change needs a rebuild of the
// application, which the result reports (sharedHeaderChanged).

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace air::renderer
{

struct ShaderCompileResult
{
    bool success = false;
    std::vector<std::string> shaders; // "<feature>/<output file>" of each shader compiled
    std::string log;                  // compiler output (errors and warnings), empty when clean
    double milliseconds = 0.0;
    bool sharedHeaderChanged = false; // a *Shared.h changed: rebuild C++ if a struct layout changed
};

class ShaderHotReload
{
public:
    explicit ShaderHotReload(const std::filesystem::path& shaderRoot);
    ~ShaderHotReload(); // waits for a running compile

    ShaderHotReload(const ShaderHotReload&) = delete;
    ShaderHotReload& operator=(const ShaderHotReload&) = delete;

    // False when there is no manifest (e.g. no dxc at configure time) or its compiler is missing.
    bool IsAvailable() const { return mUnavailableReason.empty(); }
    const std::string& GetUnavailableReason() const { return mUnavailableReason; }
    size_t GetShaderCount() const { return mShaders.size(); }
    size_t GetWatchedFileCount() const { return mFiles.size(); }

    // Call every frame. Checks file times at most every `kCheckInterval`, starts compiling what
    // changed once the files have been quiet for `kSettleTime` (editors save in several steps), and
    // returns the result of a compile that finished since the last call.
    std::optional<ShaderCompileResult> Poll();
    // Queues every shader for compilation (picked up by the next Poll()).
    void RecompileAll();
    bool IsCompiling() const { return mCompiling.load(); }

    static constexpr std::chrono::milliseconds kCheckInterval{250};
    static constexpr std::chrono::milliseconds kSettleTime{150};

private:
    struct Shader
    {
        std::string name; // "<feature>/<output file>"
        std::filesystem::path compiler;
        std::filesystem::path source;
        std::filesystem::path output;
        std::vector<std::string> arguments;
        std::vector<size_t> files; // indices into mFiles
    };
    struct WatchedFile
    {
        std::filesystem::path path;
        std::filesystem::file_time_type time;
        bool shared = false; // *Shared.h: compiled into C++ as well
    };

    void StartCompile();
    void Compile(std::vector<size_t> shaders, bool sharedChanged);

    std::vector<Shader> mShaders;
    std::vector<WatchedFile> mFiles;
    std::string mUnavailableReason;

    std::chrono::steady_clock::time_point mLastCheck{};
    std::chrono::steady_clock::time_point mLastChange{};
    std::vector<bool> mDirty; // per shader, waiting to be compiled
    bool mDirtyShared = false;

    std::thread mWorker;
    std::atomic<bool> mCompiling{false};
    std::mutex mResultMutex;
    std::optional<ShaderCompileResult> mResult; // finished, not yet returned by Poll()
};

} // namespace air::renderer
