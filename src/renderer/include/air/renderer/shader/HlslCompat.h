// HLSL <-> C++ compatibility layer for code shared between shaders and the CPU (shared structs,
// BSDFs, samplers, the reference integrator used by tests).
//
// Shared code is written in the common subset of HLSL 2021 and C++20:
//   * vector types float2/float3/float4 and uint, component access through .x/.y/.z/.w (no swizzles)
//   * free functions dot, cross, normalize, length, lerp, saturate, clamp, min, max, abs, sqrt, ...
//   * AIR_INOUT(T) / AIR_OUT(T) for reference parameters, AIR_CONST for file-scope constants
//   * no pointers, references, templates, recursion or matrices
// In HLSL this header only defines the macros; in C++ it provides the types and functions in
// namespace air::hlsl.
#ifndef AIR_SHADER_HLSL_COMPAT_H
#define AIR_SHADER_HLSL_COMPAT_H

#ifdef __cplusplus

#include "air/core/Types.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#define AIR_INOUT(T) T&
#define AIR_OUT(T) T&
#define AIR_CONST static constexpr

// HLSL spells its intrinsics in lower case; the C++ mirror keeps the same names.
// NOLINTBEGIN(readability-identifier-naming)
namespace air::hlsl {

using uint = u32;

struct float2 {
    float x = 0, y = 0;
    constexpr float2() = default;
    constexpr float2(float s) : x(s), y(s) {} // NOLINT(google-explicit-constructor): HLSL scalar promotion
    constexpr float2(float x_, float y_) : x(x_), y(y_) {}
    bool operator==(const float2&) const = default; // C++ only: whole-vector equality
};

struct float3 {
    float x = 0, y = 0, z = 0;
    constexpr float3() = default;
    constexpr float3(float s) : x(s), y(s), z(s) {} // NOLINT(google-explicit-constructor): HLSL scalar promotion
    constexpr float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    bool operator==(const float3&) const = default;
};

struct float4 {
    float x = 0, y = 0, z = 0, w = 0;
    constexpr float4() = default;
    constexpr float4(float s) : x(s), y(s), z(s), w(s) {} // NOLINT(google-explicit-constructor)
    constexpr float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    constexpr float4(float3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    bool operator==(const float4&) const = default;
};

static_assert(sizeof(float2) == 8 && sizeof(float3) == 12 && sizeof(float4) == 16);

// Component-wise arithmetic.
#define AIR_HLSL_VEC_OPS(T, APPLY)                                                                                     \
    constexpr T operator+(T a, T b) {                                                                                  \
        return APPLY(a, b, +);                                                                                         \
    }                                                                                                                  \
    constexpr T operator-(T a, T b) {                                                                                  \
        return APPLY(a, b, -);                                                                                         \
    }                                                                                                                  \
    constexpr T operator*(T a, T b) {                                                                                  \
        return APPLY(a, b, *);                                                                                         \
    }                                                                                                                  \
    constexpr T operator/(T a, T b) {                                                                                  \
        return APPLY(a, b, /);                                                                                         \
    }                                                                                                                  \
    constexpr T operator*(T a, float s) {                                                                              \
        return a * T(s);                                                                                               \
    }                                                                                                                  \
    constexpr T operator*(float s, T a) {                                                                              \
        return T(s) * a;                                                                                               \
    }                                                                                                                  \
    constexpr T operator/(T a, float s) {                                                                              \
        return a / T(s);                                                                                               \
    }                                                                                                                  \
    constexpr T operator+(T a, float s) {                                                                              \
        return a + T(s);                                                                                               \
    }                                                                                                                  \
    constexpr T operator-(T a, float s) {                                                                              \
        return a - T(s);                                                                                               \
    }                                                                                                                  \
    constexpr T operator-(float s, T a) {                                                                              \
        return T(s) - a;                                                                                               \
    }                                                                                                                  \
    constexpr T operator-(T a) {                                                                                       \
        return T(0.0f) - a;                                                                                            \
    }                                                                                                                  \
    constexpr T& operator+=(T& a, T b) {                                                                               \
        return a = a + b;                                                                                              \
    }                                                                                                                  \
    constexpr T& operator-=(T& a, T b) {                                                                               \
        return a = a - b;                                                                                              \
    }                                                                                                                  \
    constexpr T& operator*=(T& a, T b) {                                                                               \
        return a = a * b;                                                                                              \
    }                                                                                                                  \
    constexpr T& operator/=(T& a, T b) {                                                                               \
        return a = a / b;                                                                                              \
    }

#define AIR_HLSL_APPLY2(a, b, op) float2(a.x op b.x, a.y op b.y)
#define AIR_HLSL_APPLY3(a, b, op) float3(a.x op b.x, a.y op b.y, a.z op b.z)
#define AIR_HLSL_APPLY4(a, b, op) float4(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w)
AIR_HLSL_VEC_OPS(float2, AIR_HLSL_APPLY2)
AIR_HLSL_VEC_OPS(float3, AIR_HLSL_APPLY3)
AIR_HLSL_VEC_OPS(float4, AIR_HLSL_APPLY4)
#undef AIR_HLSL_VEC_OPS
#undef AIR_HLSL_APPLY2
#undef AIR_HLSL_APPLY3
#undef AIR_HLSL_APPLY4

// Scalar intrinsics.
inline float saturate(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}
inline float clamp(float v, float lo, float hi) {
    return std::clamp(v, lo, hi);
}
inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}
inline float rsqrt(float v) {
    return 1.0f / std::sqrt(v);
}
inline float rcp(float v) {
    return 1.0f / v;
}
inline float frac(float v) {
    return v - std::floor(v);
}
inline bool isfinite(float v) {
    return std::isfinite(v);
}
using std::abs;
using std::acos;
using std::cos;
using std::exp;
using std::floor;
using std::log;
using std::log2;
using std::max;
using std::min;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;
inline uint asuint(float v) {
    uint u;
    std::memcpy(&u, &v, 4);
    return u;
}
inline float asfloat(uint u) {
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// Vector intrinsics.
constexpr float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}
constexpr float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
constexpr float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
constexpr float3 cross(float3 a, float3 b) {
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline float length(float3 v) {
    return std::sqrt(dot(v, v));
}
inline float length(float2 v) {
    return std::sqrt(dot(v, v));
}
inline float3 normalize(float3 v) {
    return v * (1.0f / length(v));
}
inline float3 lerp(float3 a, float3 b, float t) {
    return a + (b - a) * t;
}
inline float3 lerp(float3 a, float3 b, float3 t) {
    return a + (b - a) * t;
}
inline float3 saturate(float3 v) {
    return float3(saturate(v.x), saturate(v.y), saturate(v.z));
}
inline float3 min(float3 a, float3 b) {
    return float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}
inline float3 max(float3 a, float3 b) {
    return float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}
inline float3 abs(float3 v) {
    return float3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}
inline float3 sqrt(float3 v) {
    return float3(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z));
}
inline float3 exp(float3 v) {
    return float3(std::exp(v.x), std::exp(v.y), std::exp(v.z));
}
inline float3 pow(float3 v, float e) {
    return float3(std::pow(v.x, e), std::pow(v.y, e), std::pow(v.z, e));
}
inline float3 reflect(float3 i, float3 n) {
    return i - 2.0f * dot(n, i) * n;
}
inline bool any(float3 v) {
    return v.x != 0.0f || v.y != 0.0f || v.z != 0.0f;
}

} // namespace air::hlsl
// NOLINTEND(readability-identifier-naming)

#else // HLSL

#define AIR_INOUT(T) inout T
#define AIR_OUT(T) out T
#define AIR_CONST static const

#endif

#endif // AIR_SHADER_HLSL_COMPAT_H
