#include "VolumeShadowLayout.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace air::pt
{

using hlsl::float3;

namespace
{

float BoxDistance(float3 p, float3 lo, float3 hi)
{
    const float3 d = hlsl::max(hlsl::max(lo - p, p - hi), float3(0.0f));
    return hlsl::length(d);
}

float BoxFarthest(float3 p, float3 lo, float3 hi)
{
    float farthest = 0.0f;
    for (int c = 0; c < 8; ++c)
    {
        const float3 corner((c & 1) != 0 ? hi.x : lo.x, (c & 2) != 0 ? hi.y : lo.y, (c & 4) != 0 ? hi.z : lo.z);
        farthest = std::max(farthest, hlsl::length(corner - p));
    }
    return farthest;
}

// Two unit vectors that make an orthonormal basis with unit w (Duff et al. 2017).
void Perpendicular(float3 w, float3& u, float3& v)
{
    const float s = w.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (s + w.z);
    const float b = w.x * w.y * a;
    u = float3(1.0f + s * w.x * w.x * a, s * b, -s * w.x);
    v = float3(b, s + w.y * w.y * a, -w.y);
}

// Where a group's maps look from. A point is enough for an emitter that is small, or far from the
// volumes, or closed (a sphere lights them from its side): the light reaches them along much the same
// rays from all over it. A large open emitter close to them is not: a column of smoke over a bed of
// embers hides their centre from the smoke above it, which the edges of the embers light past it. Its
// area in each quadrant of its plane (around its centre, along the principal axes of the area) gets
// a map at its centre, those with the most area first.
std::vector<float3> GroupOrigins(const air::gpu::LightGroup& g, std::span<const air::gpu::LightTriangle> lights,
                                 float distance, u32 maxOrigins)
{
    const float open = hlsl::length(g.normalArea);
    if (maxOrigins < 2 || open < 0.5f * g.area || g.radius < 0.25f * (distance + g.radius))
    {
        return {g.centre};
    }
    // The emitter's triangles as pieces (each cut into n^2) in the plane through the centre.
    float3 u;
    float3 v;
    Perpendicular(g.normalArea / open, u, v);
    struct Piece
    {
        float3 position;
        float area;
        float x;
        float y;
    };
    std::vector<Piece> pieces;
    constexpr u32 kCuts = 8;
    double xx = 0.0;
    double xy = 0.0;
    double yy = 0.0;
    for (u32 i = g.firstLight; i < g.firstLight + g.lightCount && i < lights.size(); ++i)
    {
        const air::gpu::LightTriangle& t = lights[i];
        const float area = t.area / float(kCuts * kCuts);
        for (u32 a = 0; a < kCuts; ++a)
        {
            for (u32 b = 0; a + b < kCuts; ++b)
            {
                for (u32 flip = 0; flip < 2; ++flip)
                {
                    if (flip == 1 && a + b + 1 >= kCuts)
                    {
                        continue;
                    }
                    const float offset = flip == 0 ? 1.0f / 3.0f : 2.0f / 3.0f;
                    const float3 p = t.p0 + t.e1 * ((float(a) + offset) / float(kCuts)) +
                                     t.e2 * ((float(b) + offset) / float(kCuts));
                    const float x = hlsl::dot(p - g.centre, u);
                    const float y = hlsl::dot(p - g.centre, v);
                    pieces.push_back({p, area, x, y});
                    xx += double(area) * x * x;
                    xy += double(area) * x * y;
                    yy += double(area) * y * y;
                }
            }
        }
    }
    // The principal axes of the area: the eigenvectors of its second moments.
    const double angle = 0.5 * std::atan2(2.0 * xy, xx - yy);
    const float c = float(std::cos(angle));
    const float s = float(std::sin(angle));
    double weight[4] = {};
    double sum[4][3] = {};
    for (const Piece& p : pieces)
    {
        const float major = c * p.x + s * p.y;
        const float minor = -s * p.x + c * p.y;
        const int q = (major >= 0.0f ? 1 : 0) + (minor >= 0.0f ? 2 : 0);
        weight[q] += p.area;
        sum[q][0] += double(p.area) * p.position.x;
        sum[q][1] += double(p.area) * p.position.y;
        sum[q][2] += double(p.area) * p.position.z;
    }
    std::vector<std::pair<double, float3>> quadrants;
    for (int q = 0; q < 4; ++q)
    {
        if (weight[q] > 0.02 * g.area)
        {
            quadrants.emplace_back(weight[q], float3(float(sum[q][0] / weight[q]), float(sum[q][1] / weight[q]),
                                                     float(sum[q][2] / weight[q])));
        }
    }
    std::stable_sort(quadrants.begin(), quadrants.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<float3> origins;
    for (size_t i = 0; i < quadrants.size() && i < maxOrigins; ++i)
    {
        origins.push_back(quadrants[i].second);
    }
    return origins.empty() ? std::vector<float3>{g.centre} : origins;
}

} // namespace

VolumeMarchSettings ClampVolumeMarchSettings(const VolumeMarchSettings& settings)
{
    VolumeMarchSettings out = settings;
    out.shadowSize = std::clamp(settings.shadowSize, 16u, 2048u);
    out.lights = std::min(settings.lights, gpu::kMaxVolumeLights);
    out.lightOrigins = std::clamp(settings.lightOrigins, 1u, gpu::kMaxLightOrigins);
    out.level = std::min(settings.level, air::gpu::kVolumeLevels - 1);
    out.shadowLevel = std::min(settings.shadowLevel, air::gpu::kVolumeLevels - 1);
    out.step = std::max(settings.step, 0.05f);
    out.shadowStep = std::max(settings.shadowStep, 0.05f);
    out.temporalBlend = std::clamp(settings.temporalBlend, 0.01f, 1.0f);
    return out;
}

VolumeShadowLayout LayoutVolumeShadows(std::span<const air::gpu::VolumeMarchData> volumes,
                                       std::span<const air::gpu::LightGroup> groups,
                                       std::span<const air::gpu::LightTriangle> lights, const air::gpu::SkyData& sky,
                                       const VolumeMarchSettings& settings)
{
    VolumeShadowLayout layout;
    layout.groupMaps.assign(std::max<size_t>(groups.size(), 1), air::gpu::kInvalidIndex);
    if (volumes.empty())
    {
        return layout;
    }
    float3 lo(1e30f);
    float3 hi(-1e30f);
    for (const air::gpu::VolumeMarchData& v : volumes)
    {
        lo = hlsl::min(lo, v.boundsMin);
        hi = hlsl::max(hi, v.boundsMax);
    }

    // The light groups that matter most to the volumes: power over the squared distance to their box.
    std::vector<std::pair<float, u32>> ranked;
    for (u32 i = 0; i < groups.size(); ++i)
    {
        const air::gpu::LightGroup& g = groups[i];
        if (g.power > 0.0f)
        {
            const float d = BoxDistance(g.centre, lo, hi);
            ranked.emplace_back(g.power / (d * d + g.radius * g.radius + 1e-4f), i);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < ranked.size() && i < settings.lights; ++i)
    {
        layout.lights.push_back(ranked[i].second);
    }

    // A map for the sun over the volumes' box (orthographic), around points of each group (octahedral),
    // their depths spanning the box as seen from the light.
    const float margin = 0.02f * std::max(hlsl::length(hi - lo), 1e-3f);
    if (hlsl::dot(sky.sunRadiance, float3(1.0f)) > 0.0f)
    {
        air::gpu::MomentShadowMap m{};
        m.kind = air::gpu::kMomentMapSun;
        m.axisW = -sky.sunDirection;
        float3 u;
        float3 v;
        Perpendicular(m.axisW, u, v);
        float3 low(1e30f);
        float3 high(-1e30f);
        for (int c = 0; c < 8; ++c)
        {
            const float3 corner((c & 1) != 0 ? hi.x : lo.x, (c & 2) != 0 ? hi.y : lo.y, (c & 4) != 0 ? hi.z : lo.z);
            const float3 projected(hlsl::dot(corner, u), hlsl::dot(corner, v), hlsl::dot(corner, m.axisW));
            low = hlsl::min(low, projected);
            high = hlsl::max(high, projected);
        }
        low = low - float3(margin);
        high = high + float3(margin);
        m.origin = u * low.x + v * low.y + m.axisW * low.z;
        m.axisU = u * (high.x - low.x);
        m.axisV = v * (high.y - low.y);
        m.depthNear = 0.0f;
        m.depthFar = high.z - low.z;
        m.slice = 0;
        m.size = settings.shadowSize;
        m.skipInstance = air::gpu::kInvalidIndex;
        m.count = 1;
        layout.sunMap = 0;
        layout.maps.push_back(m);
    }
    for (u32 group : layout.lights)
    {
        const air::gpu::LightGroup& g = groups[group];
        const std::vector<float3> origins =
            GroupOrigins(g, lights, BoxDistance(g.centre, lo, hi), settings.lightOrigins);
        layout.groupMaps[group] = u32(layout.maps.size());
        for (const float3 origin : origins)
        {
            air::gpu::MomentShadowMap m{};
            m.kind = air::gpu::kMomentMapPoint;
            m.origin = origin;
            m.depthNear = std::max(BoxDistance(origin, lo, hi) - margin, 0.0f);
            m.depthFar = BoxFarthest(origin, lo, hi) + margin;
            m.slice = u32(layout.maps.size());
            m.size = settings.shadowSize;
            m.skipInstance = g.instance;
            m.count = u32(origins.size());
            layout.maps.push_back(m);
        }
    }
    return layout;
}

} // namespace air::pt
