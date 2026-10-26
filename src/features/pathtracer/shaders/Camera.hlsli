// The camera's rays (CameraData, PathTracerShared.h), shared by C++ and HLSL: the path tracer's and the
// volumes' ray march (VolumeMarch.hlsl) trace the same ones.
#ifndef AIR_PT_CAMERA_HLSLI
#define AIR_PT_CAMERA_HLSLI

float3 Xyz(float4 v)
{
    return float3(v.x, v.y, v.z);
}

// Direction through pixel position `pixel` (0 .. size, y down).
float3 CameraRayDirection(CameraData c, float2 pixel, float2 size)
{
    float ndcX = pixel.x / size.x * 2.0f - 1.0f;
    float ndcY = 1.0f - pixel.y / size.y * 2.0f;
    float tanHalf = c.position.w;
    float aspect = c.forward.w;
    return normalize(Xyz(c.forward) + Xyz(c.right) * (ndcX * tanHalf * aspect) + Xyz(c.up) * (ndcY * tanHalf));
}

// Pixel position of a world point (or of a direction when w = 0), inverse of CameraRayDirection.
float2 CameraProject(CameraData c, float3 p, float w, float2 size)
{
    float3 d = p - Xyz(c.position) * w;
    float z = max(dot(d, Xyz(c.forward)), 1e-6f);
    float tanHalf = c.position.w;
    float aspect = c.forward.w;
    float ndcX = dot(d, Xyz(c.right)) / (z * tanHalf * aspect);
    float ndcY = dot(d, Xyz(c.up)) / (z * tanHalf);
    return float2((ndcX + 1.0f) * 0.5f * size.x, (1.0f - ndcY) * 0.5f * size.y);
}

#endif
