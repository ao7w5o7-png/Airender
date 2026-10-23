#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace air {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

constexpr u64 AlignUp(u64 value, u64 alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}
constexpr bool IsPow2(u64 value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// Enables bitwise operators on an enum class used as a flag set.
#define AIR_ENUM_FLAGS(E)                                                                                              \
    constexpr E operator|(E a, E b) {                                                                                  \
        return E(std::underlying_type_t<E>(a) | std::underlying_type_t<E>(b));                                         \
    }                                                                                                                  \
    constexpr E operator&(E a, E b) {                                                                                  \
        return E(std::underlying_type_t<E>(a) & std::underlying_type_t<E>(b));                                         \
    }                                                                                                                  \
    constexpr E& operator|=(E& a, E b) {                                                                               \
        return a = a | b;                                                                                              \
    }                                                                                                                  \
    constexpr bool Any(E a) {                                                                                          \
        return std::underlying_type_t<E>(a) != 0;                                                                      \
    }

} // namespace air
