#include "air/core/Process.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ; // the parent's environment, passed on to the child
#endif

namespace air
{

#if defined(_WIN32)

namespace
{

std::wstring Widen(const std::string& text)
{
    if (text.empty())
    {
        return {};
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), nullptr, 0);
    std::wstring out(size_t(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), out.data(), size);
    return out;
}

// Quotes one argument so the C runtime's command line parser gives it back unchanged: backslashes
// are literal except before a quote, where they and the quote are escaped.
void AppendQuoted(std::wstring& commandLine, const std::wstring& argument)
{
    if (!commandLine.empty())
    {
        commandLine += L' ';
    }
    if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == std::wstring::npos)
    {
        commandLine += argument;
        return;
    }
    commandLine += L'"';
    size_t backslashes = 0;
    for (wchar_t c : argument)
    {
        if (c == L'\\')
        {
            ++backslashes;
            continue;
        }
        if (c == L'"')
        {
            commandLine.append(backslashes * 2 + 1, L'\\');
        }
        else
        {
            commandLine.append(backslashes, L'\\');
        }
        backslashes = 0;
        commandLine += c;
    }
    commandLine.append(backslashes * 2, L'\\');
    commandLine += L'"';
}

} // namespace

ProcessResult RunProcess(const std::filesystem::path& program, const std::vector<std::string>& arguments)
{
    ProcessResult result;
    SECURITY_ATTRIBUTES inherit = {sizeof(inherit), nullptr, TRUE};
    HANDLE readPipe = nullptr;
    HANDLE writePipe = nullptr;
    if (!CreatePipe(&readPipe, &writePipe, &inherit, 0))
    {
        return result;
    }
    SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0); // only the child's end is inherited

    // Native separators: some programs parse their own argv[0], and cmd.exe reads the forward
    // slashes of "C:/Windows/System32/cmd.exe" as switches.
    const std::wstring programPath = std::filesystem::path(program).make_preferred().wstring();
    std::wstring commandLine;
    AppendQuoted(commandLine, programPath);
    for (const std::string& argument : arguments)
    {
        AppendQuoted(commandLine, Widen(argument));
    }

    STARTUPINFOW startup = {sizeof(startup)};
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = writePipe;
    startup.hStdError = writePipe;
    PROCESS_INFORMATION process = {};
    const BOOL started = CreateProcessW(programPath.c_str(), commandLine.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(writePipe); // the pipe reports EOF once the child closed its end too
    if (!started)
    {
        CloseHandle(readPipe);
        return result;
    }

    char buffer[4096];
    DWORD read = 0;
    while (ReadFile(readPipe, buffer, sizeof(buffer), &read, nullptr) && read > 0)
    {
        result.output.append(buffer, read);
    }
    CloseHandle(readPipe);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(process.hProcess, &exitCode);
    result.exitCode = int(exitCode);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return result;
}

#else

ProcessResult RunProcess(const std::filesystem::path& program, const std::vector<std::string>& arguments)
{
    ProcessResult result;
    int pipeFds[2];
    if (pipe(pipeFds) != 0)
    {
        return result;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipeFds[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, pipeFds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipeFds[0]);

    const std::string path = program.string();
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(path.c_str()));
    for (const std::string& argument : arguments)
    {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = 0;
    const int spawned = posix_spawn(&pid, path.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipeFds[1]);
    if (spawned != 0)
    {
        close(pipeFds[0]);
        return result;
    }
    char buffer[4096];
    ssize_t count = 0;
    while ((count = read(pipeFds[0], buffer, sizeof(buffer))) > 0)
    {
        result.output.append(buffer, size_t(count));
    }
    close(pipeFds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    result.exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return result;
}

#endif

} // namespace air
