#pragma once

#include "air/core/Log.h"

#include <cstdlib>

// AIR_CHECK is always on: use it for invariants whose violation would corrupt GPU state.
#define AIR_CHECK(cond, ...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond)) [[unlikely]]                                                                                      \
        {                                                                                                              \
            ::air::LogMessage(                                                                                         \
                ::air::LogLevel::Fatal, "Check",                                                                       \
                std::format("{}:{}: check failed: {} {}", __FILE__, __LINE__, #cond, std::format("" __VA_ARGS__)));    \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#ifndef NDEBUG
#define AIR_DCHECK(cond, ...) AIR_CHECK(cond, __VA_ARGS__)
#else
#define AIR_DCHECK(cond, ...) ((void)0)
#endif
