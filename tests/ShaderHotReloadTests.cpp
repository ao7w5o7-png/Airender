// Shader hot reload with the build's real compiler (skipped when the build has no dxc), plus the
// JSON parser and process runner it is built on.

#include "TestHelpers.h"

#include "air/core/Json.h"
#include "air/core/Process.h"
#include "air/renderer/ShaderHotReload.h"

#include <fstream>
#include <iterator>
#include <thread>

using namespace air;
namespace fs = std::filesystem;

namespace
{

std::string ReadText(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Writes `text` and moves the file time forward, so a change is seen even within the file system's
// timestamp resolution.
void WriteSource(const fs::path& path, const std::string& text)
{
    const bool existed = fs::exists(path);
    const fs::file_time_type before = existed ? fs::last_write_time(path) : fs::file_time_type::min();
    std::ofstream(path, std::ios::binary) << text;
    if (existed && fs::last_write_time(path) <= before)
    {
        fs::last_write_time(path, before + std::chrono::seconds(1));
    }
}

// The compiler the build itself uses, from its manifest; empty when shaders are not compiled.
fs::path BuildCompiler()
{
    const std::optional<json::Value> manifest = json::Parse(ReadText(fs::path(AIR_BUILD_SHADER_DIR) / "manifest.json"));
    const json::Value* compiler = manifest ? manifest->Find("compiler") : nullptr;
    return compiler && fs::exists(compiler->AsString()) ? fs::path(compiler->AsString()) : fs::path();
}

std::optional<renderer::ShaderCompileResult> WaitForCompile(renderer::ShaderHotReload& reload)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (std::optional<renderer::ShaderCompileResult> result = reload.Poll())
        {
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return std::nullopt;
}

const char* const kGoodShader = "#include \"FillShared.h\"\n"
                                "RWTexture2D<float4> image : register(u0);\n"
                                "[numthreads(8, 8, 1)]\n"
                                "void Main(uint3 id : SV_DispatchThreadID)\n"
                                "{\n"
                                "    image[id.xy] = kFill;\n"
                                "}\n";

} // namespace

TEST(Json_ParseRoundTripAndErrors)
{
    json::Value v;
    v["name"] = "a \"quoted\" \\ path\n";
    v["number"] = 1.5;
    v["list"].Push(true);
    v["list"].Push(nullptr);
    v["nested"]["k"] = -2;
    const std::optional<json::Value> parsed = json::Parse(v.Dump());
    EXPECT(parsed && parsed->Dump(0) == v.Dump(0));
    EXPECT(parsed->Find("name")->AsString() == "a \"quoted\" \\ path\n");
    EXPECT(parsed->Find("nested")->Find("k")->AsNumber() == -2.0);
    EXPECT(parsed->Find("list")->AsArray().size() == 2 && parsed->Find("list")->AsArray()[0].AsBool());
    EXPECT(parsed->Find("missing") == nullptr && json::Value(3).AsString().empty());
    // \u escapes, including a surrogate pair, decode to UTF-8.
    EXPECT(json::Parse("\"\\u00e9\\ud83d\\ude00\"")->AsString() == "\xc3\xa9\xf0\x9f\x98\x80");
    std::string error;
    EXPECT(!json::Parse("{\"a\": }", &error) && error.find("byte") != std::string::npos);
    EXPECT(!json::Parse("[1, 2", &error));
    EXPECT(!json::Parse("1 2", &error));
    EXPECT(!json::Parse("\"\\ud83d\"", &error)); // unpaired surrogate
}

TEST(Process_RunsProgramsAndCapturesOutput)
{
#if defined(_WIN32)
    const ProcessResult result = RunProcess("C:/Windows/System32/cmd.exe", {"/c", "echo", "hello world"});
#else
    const ProcessResult result = RunProcess("/bin/sh", {"-c", "echo hello world"});
#endif
    EXPECT(result.exitCode == 0 && result.output.find("hello world") != std::string::npos);
    EXPECT(RunProcess("/definitely/not/a/program", {}).exitCode == -1);
}

TEST(ShaderHotReload_UnavailableWithoutManifest)
{
    renderer::ShaderHotReload reload(fs::temp_directory_path() / "air_no_such_shader_root");
    EXPECT(!reload.IsAvailable() && !reload.GetUnavailableReason().empty());
    EXPECT(!reload.Poll());
}

TEST(ShaderHotReload_RecompilesChangesAndKeepsOutputsOnError)
{
    const fs::path compiler = BuildCompiler();
    if (compiler.empty())
    {
        return; // this build compiles no shaders (no dxc): nothing to reload with
    }
    const fs::path root = fs::temp_directory_path() / "air_hot_reload_test";
    fs::remove_all(root);
    fs::create_directories(root / "src");
    const fs::path source = root / "src" / "Fill.hlsl";
    const fs::path header = root / "src" / "FillShared.h";
    const fs::path output = root / "shaders" / "test" / "Fill.cs.dxil";
    WriteSource(header, "static const float4 kFill = float4(1, 1, 1, 1);\n");
    WriteSource(source, kGoodShader);

    json::Value entry;
    entry["name"] = "test";
    entry["format"] = "dxil";
    entry["source"] = source.generic_string();
    entry["output"] = output.generic_string();
    for (const char* argument : {"-T", "cs_6_6", "-E", "Main", "-HV", "2021", "-WX"})
    {
        entry["args"].Push(argument);
    }
    entry["args"].Push("-I");
    entry["args"].Push((root / "src").generic_string());
    entry["watch"].Push(source.generic_string());
    entry["watch"].Push(header.generic_string());
    json::Value manifest;
    manifest["compiler"] = compiler.generic_string();
    manifest["shaders"].Push(std::move(entry));
    fs::create_directories(root / "shaders");
    std::ofstream(root / "shaders" / "manifest.json") << manifest.Dump();

    renderer::ShaderHotReload reload(root / "shaders");
    EXPECT(reload.IsAvailable() && reload.GetShaderCount() == 1 && reload.GetWatchedFileCount() == 2);
    EXPECT(!reload.Poll()); // nothing changed yet

    // Editing an included header recompiles the shader that includes it.
    WriteSource(header, "static const float4 kFill = float4(0.5, 0.5, 0.5, 1);\n");
    std::optional<renderer::ShaderCompileResult> result = WaitForCompile(reload);
    EXPECT(result && result->success && result->shaders.size() == 1 && result->shaders[0] == "test/Fill.cs.dxil");
    EXPECT(result && result->sharedHeaderChanged); // *Shared.h is also compiled into C++ by convention
    EXPECT(fs::exists(output));
    const std::string good = ReadText(output);

    // A syntax error fails the compile, reports the compiler's message and keeps the previous output.
    WriteSource(source, "void Main( {\n");
    result = WaitForCompile(reload);
    EXPECT(result && !result->success && result->log.find("error") != std::string::npos);
    EXPECT(ReadText(output) == good);
    EXPECT(!fs::exists(fs::path(output.string() + ".tmp")));

    // Fixed again, and a forced recompile of everything.
    WriteSource(source, kGoodShader);
    result = WaitForCompile(reload);
    EXPECT(result && result->success);
    reload.RecompileAll();
    result = WaitForCompile(reload);
    EXPECT(result && result->success && !result->sharedHeaderChanged);
    fs::remove_all(root);
}
