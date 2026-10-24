#include "air/renderer/ShaderHotReload.h"

#include "air/core/Json.h"
#include "air/core/Process.h"

#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <utility>

namespace air::renderer
{

namespace fs = std::filesystem;

namespace
{

fs::file_time_type WriteTime(const fs::path& path)
{
    std::error_code error;
    const fs::file_time_type time = fs::last_write_time(path, error);
    return error ? fs::file_time_type::min() : time; // a file being saved may briefly be missing
}

bool EndsWith(const std::string& text, std::string_view suffix)
{
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

ShaderHotReload::ShaderHotReload(const fs::path& shaderRoot)
{
    const fs::path manifestPath = shaderRoot / "manifest.json";
    std::ifstream file(manifestPath, std::ios::binary);
    if (!file)
    {
        mUnavailableReason = std::format("no {} (configure with dxc available)", manifestPath.generic_string());
        return;
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::string error;
    const std::optional<json::Value> manifest = json::Parse(text, &error);
    if (!manifest)
    {
        mUnavailableReason = std::format("{}: {}", manifestPath.generic_string(), error);
        return;
    }
    const json::Value* compiler = manifest->Find("compiler");
    mCompiler = compiler ? fs::path(compiler->AsString()) : fs::path();
    if (mCompiler.empty() || !fs::exists(mCompiler))
    {
        mUnavailableReason = std::format("shader compiler '{}' not found", mCompiler.generic_string());
        return;
    }

    std::map<std::string, size_t> fileIndex;
    const json::Value* shaders = manifest->Find("shaders");
    for (const json::Value& entry : shaders ? shaders->AsArray() : json::Value::Array{})
    {
        auto field = [&](const char* name) -> const json::Value&
        {
            static const json::Value kNull;
            const json::Value* value = entry.Find(name);
            return value ? *value : kNull;
        };
        Shader shader;
        shader.source = field("source").AsString();
        shader.output = field("output").AsString();
        shader.name = std::format("{}/{}", field("name").AsString(), shader.output.filename().string());
        for (const json::Value& argument : field("args").AsArray())
        {
            shader.arguments.push_back(argument.AsString());
        }
        for (const json::Value& watched : field("watch").AsArray())
        {
            const std::string& path = watched.AsString();
            auto [it, inserted] = fileIndex.try_emplace(path, mFiles.size());
            if (inserted)
            {
                mFiles.push_back({path, WriteTime(path), EndsWith(path, "Shared.h")});
            }
            shader.files.push_back(it->second);
        }
        if (!shader.source.empty() && !shader.output.empty())
        {
            mShaders.push_back(std::move(shader));
        }
    }
    if (mShaders.empty())
    {
        mUnavailableReason = std::format("{} lists no shaders", manifestPath.generic_string());
    }
    mDirty.assign(mShaders.size(), false);
}

ShaderHotReload::~ShaderHotReload()
{
    if (mWorker.joinable())
    {
        mWorker.join();
    }
}

void ShaderHotReload::RecompileAll()
{
    mDirty.assign(mShaders.size(), true);
    mLastChange = std::chrono::steady_clock::now() - kSettleTime; // no need to wait for quiet files
}

std::optional<ShaderCompileResult> ShaderHotReload::Poll()
{
    std::optional<ShaderCompileResult> finished;
    if (!mCompiling.load() && mWorker.joinable())
    {
        mWorker.join();
        std::lock_guard lock(mResultMutex);
        finished = std::exchange(mResult, std::nullopt);
    }
    if (!IsAvailable())
    {
        return finished;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - mLastCheck >= kCheckInterval)
    {
        mLastCheck = now;
        for (size_t f = 0; f < mFiles.size(); ++f)
        {
            const fs::file_time_type time = WriteTime(mFiles[f].path);
            if (time == mFiles[f].time)
            {
                continue;
            }
            mFiles[f].time = time;
            mLastChange = now;
            mDirtyShared = mDirtyShared || mFiles[f].shared;
            for (size_t s = 0; s < mShaders.size(); ++s)
            {
                for (size_t watched : mShaders[s].files)
                {
                    mDirty[s] = mDirty[s] || watched == f;
                }
            }
        }
    }
    const bool anyDirty = std::find(mDirty.begin(), mDirty.end(), true) != mDirty.end();
    if (anyDirty && !mCompiling.load() && now - mLastChange >= kSettleTime)
    {
        StartCompile();
    }
    return finished;
}

void ShaderHotReload::StartCompile()
{
    std::vector<size_t> shaders;
    for (size_t s = 0; s < mShaders.size(); ++s)
    {
        if (mDirty[s])
        {
            shaders.push_back(s);
            mDirty[s] = false;
        }
    }
    const bool shared = std::exchange(mDirtyShared, false);
    mCompiling.store(true);
    mWorker =
        std::thread([this, shaders = std::move(shaders), shared]() mutable { Compile(std::move(shaders), shared); });
}

// Runs on the worker thread; reads only data that is immutable after construction.
void ShaderHotReload::Compile(std::vector<size_t> shaders, bool sharedChanged)
{
    const auto start = std::chrono::steady_clock::now();
    ShaderCompileResult result;
    result.sharedHeaderChanged = sharedChanged;
    bool failed = false;
    std::vector<fs::path> temporaries;
    for (size_t index : shaders)
    {
        const Shader& shader = mShaders[index];
        result.shaders.push_back(shader.name);
        fs::path temporary = shader.output;
        temporary += ".tmp";
        std::error_code error;
        fs::create_directories(shader.output.parent_path(), error);
        std::vector<std::string> arguments = shader.arguments;
        arguments.insert(arguments.end(), {shader.source.string(), "-Fo", temporary.string()});
        const ProcessResult process = RunProcess(mCompiler, arguments);
        if (!process.output.empty())
        {
            result.log += process.output;
            if (result.log.back() != '\n')
            {
                result.log += '\n';
            }
        }
        if (process.exitCode != 0)
        {
            failed = true;
            if (process.exitCode == -1)
            {
                result.log += std::format("{}: could not start {}\n", shader.name, mCompiler.string());
            }
        }
        temporaries.push_back(std::move(temporary));
    }
    // All or nothing: a failed compile keeps every previous output.
    for (size_t i = 0; i < temporaries.size(); ++i)
    {
        std::error_code error;
        if (!failed)
        {
            fs::rename(temporaries[i], mShaders[shaders[i]].output, error);
            if (error)
            {
                failed = true;
                result.log += std::format("{}: cannot replace {}: {}\n", mShaders[shaders[i]].name,
                                          mShaders[shaders[i]].output.string(), error.message());
            }
        }
        fs::remove(temporaries[i], error);
    }
    result.success = !failed;
    result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::lock_guard lock(mResultMutex);
    mResult = std::move(result);
    mCompiling.store(false);
}

} // namespace air::renderer
