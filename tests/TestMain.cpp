#include "TestFramework.h"

#include "air/core/Log.h"

#include <cstring>

int main(int argc, char** argv)
{
    air::SetMinLogLevel(air::LogLevel::Warning);
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int ran = 0, failedTests = 0;
    for (const auto& test : airtest::Registry())
    {
        if (filter && !std::strstr(test.name, filter))
        {
            continue;
        }
        const int before = airtest::FailureCount();
        std::printf("[ RUN  ] %s\n", test.name);
        test.fn();
        const bool ok = airtest::FailureCount() == before;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", test.name);
        ++ran;
        failedTests += ok ? 0 : 1;
    }
    std::printf("%d tests, %d failed\n", ran, failedTests);
    return failedTests == 0 && ran > 0 ? 0 : 1;
}
