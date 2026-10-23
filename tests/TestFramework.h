#pragma once

// Minimal self-registering test framework (no external dependencies, works offline).
//   TEST(Name) { EXPECT(a == b); }
// Run all: air_tests        Run matching: air_tests <substring>

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace airtest {

struct TestCase {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& Registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& FailureCount() {
    static int failures = 0;
    return failures;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { Registry().push_back({name, std::move(fn)}); }
};

} // namespace airtest

#define AIR_TEST_CONCAT2(a, b) a##b
#define AIR_TEST_CONCAT(a, b) AIR_TEST_CONCAT2(a, b)

#define TEST(name)                                                                       \
    static void AIR_TEST_CONCAT(Test_, name)();                                          \
    static ::airtest::Registrar AIR_TEST_CONCAT(Registrar_, name)(#name, &AIR_TEST_CONCAT(Test_, name)); \
    static void AIR_TEST_CONCAT(Test_, name)()

#define EXPECT(cond)                                                                     \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "  %s:%d: EXPECT failed: %s\n", __FILE__, __LINE__, #cond); \
            ++::airtest::FailureCount();                                                 \
        }                                                                                \
    } while (0)
