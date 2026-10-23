#pragma once

// Runs an external program to completion and captures what it prints. No shell is involved: the
// program and each argument are passed as given, so paths with spaces need no quoting. Used by the
// shader hot reload to invoke the same compiler as the build.

#include <filesystem>
#include <string>
#include <vector>

namespace air
{

struct ProcessResult
{
    int exitCode = -1;  // -1 when the program could not be started
    std::string output; // stdout and stderr, in the order the program wrote them
};

ProcessResult RunProcess(const std::filesystem::path& program, const std::vector<std::string>& arguments);

} // namespace air
