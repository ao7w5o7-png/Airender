// Random numbers, orthonormal frames and hemisphere / triangle sampling.
// Shared by HLSL and C++ (see air/renderer/shader/HlslCompat.h for the rules).
#ifndef AIR_PT_SAMPLING_HLSLI
#define AIR_PT_SAMPLING_HLSLI

AIR_CONST float kPi = 3.14159265358979f;
AIR_CONST float kTwoPi = 6.28318530717959f;
AIR_CONST float kInvPi = 0.318309886183791f;

// PCG hash (Jarzynski & Olano 2020).
uint Pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform float in [0, 1).
float Rand(AIR_INOUT(uint) seed) {
    seed = Pcg(seed);
    return float(seed >> 8) * (1.0f / 16777216.0f);
}

float Luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

float MaxComponent(float3 c) {
    return max(c.x, max(c.y, c.z));
}

// Orthonormal basis around n (Duff et al. 2017). Local space: z = normal.
struct Frame {
    float3 t;
    float3 b;
    float3 n;
};

Frame MakeFrame(float3 n) {
    float s = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + n.z);
    float b = n.x * n.y * a;
    Frame f;
    f.t = float3(1.0f + s * n.x * n.x * a, s * b, -s * n.x);
    f.b = float3(b, s + n.y * n.y * a, -n.y);
    f.n = n;
    return f;
}

float3 ToLocal(Frame f, float3 v) {
    return float3(dot(v, f.t), dot(v, f.b), dot(v, f.n));
}

float3 ToWorld(Frame f, float3 v) {
    return f.t * v.x + f.b * v.y + f.n * v.z;
}

// Cosine weighted hemisphere around +z. pdf = z / pi.
float3 SampleCosineHemisphere(float2 u) {
    float r = sqrt(u.x);
    float phi = kTwoPi * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1.0f - u.x)));
}

// Uniform point on a triangle, returned as barycentrics (b1, b2) of vertices 1 and 2.
float2 SampleTriangle(float2 u) {
    float su = sqrt(u.x);
    return float2(1.0f - su, u.y * su);
}

// Power heuristic (beta = 2) for multiple importance sampling.
float PowerHeuristic(float pdfA, float pdfB) {
    float a = pdfA * pdfA;
    float b = pdfB * pdfB;
    return a + b > 0.0f ? a / (a + b) : 0.0f;
}

#endif
