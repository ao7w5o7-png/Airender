// Light tree construction (gpu::LightTreeNode): a binary tree over the emissive triangles, split by
// a surface area orientation heuristic (Conty Estevez and Kulla, "Importance Sampling of Many Lights
// with Adaptive Tree Splitting", 2018, in the form of Physically Based Rendering, 4th edition): a
// split is cheap when each side has little power, a small box and a narrow cone of normals.

#include "air/renderer/Scene.h"

#include "air/core/Assert.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace air::renderer
{

namespace
{

constexpr float kPi = std::numbers::pi_v<float>;
constexpr u32 kBuckets = 12;
constexpr u32 kMaxDepth = 64;      // trails are 64 bits
constexpr u32 kBalancedDepth = 40; // past this, split in the middle: depth stays below 64

float SafeSqrt(float v)
{
    return std::sqrt(std::max(v, 0.0f));
}

float SafeAcos(float v)
{
    return std::acos(std::clamp(v, -1.0f, 1.0f));
}

// Angle between unit vectors, accurate for nearly parallel ones too.
float AngleBetween(float3 a, float3 b)
{
    if (dot(a, b) < 0.0f)
    {
        return kPi - 2.0f * std::asin(std::min(length(a + b) * 0.5f, 1.0f));
    }
    return 2.0f * std::asin(std::min(length(b - a) * 0.5f, 1.0f));
}

// Rotates v by `angle` around the unit `axis` (Rodrigues).
float3 Rotate(float3 v, float3 axis, float angle)
{
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return v * c + cross(axis, v) * s + axis * (dot(axis, v) * (1.0f - c));
}

struct Box
{
    float3 lo = float3(std::numeric_limits<float>::max());
    float3 hi = float3(-std::numeric_limits<float>::max());

    void Grow(float3 p)
    {
        lo = min(lo, p);
        hi = max(hi, p);
    }
    void Grow(const Box& b)
    {
        lo = min(lo, b.lo);
        hi = max(hi, b.hi);
    }
    float3 Diagonal() const { return max(hi - lo, float3(0.0f)); }
    float SurfaceArea() const
    {
        const float3 d = Diagonal();
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

float Axis(float3 v, u32 dim)
{
    return dim == 0 ? v.x : dim == 1 ? v.y : v.z;
}

// What the tree knows about a set of emitters.
struct Bounds
{
    Box box;
    float3 axis = float3(0.0f, 0.0f, 1.0f);
    float cosThetaO = 1.0f; // normals within this angle of `axis`
    float cosThetaE = 1.0f; // each emits within this angle of its normal
    float power = 0.0f;

    float3 Centroid() const { return (box.lo + box.hi) * 0.5f; }
};

Bounds Union(const Bounds& a, const Bounds& b)
{
    if (a.power <= 0.0f)
    {
        return b;
    }
    if (b.power <= 0.0f)
    {
        return a;
    }
    Bounds out;
    out.box = a.box;
    out.box.Grow(b.box);
    out.power = a.power + b.power;
    out.cosThetaE = std::min(a.cosThetaE, b.cosThetaE);
    // The smallest cone holding both normal cones.
    const float thetaA = SafeAcos(a.cosThetaO);
    const float thetaB = SafeAcos(b.cosThetaO);
    const float thetaD = AngleBetween(a.axis, b.axis);
    if (std::min(thetaD + thetaB, kPi) <= thetaA)
    {
        out.axis = a.axis;
        out.cosThetaO = a.cosThetaO;
        return out;
    }
    if (std::min(thetaD + thetaA, kPi) <= thetaB)
    {
        out.axis = b.axis;
        out.cosThetaO = b.cosThetaO;
        return out;
    }
    const float thetaO = (thetaA + thetaD + thetaB) * 0.5f;
    const float3 rotationAxis = cross(a.axis, b.axis);
    if (thetaO >= kPi || dot(rotationAxis, rotationAxis) == 0.0f)
    {
        out.axis = a.axis;
        out.cosThetaO = -1.0f; // every direction
        return out;
    }
    out.axis = normalize(Rotate(a.axis, normalize(rotationAxis), thetaO - thetaA));
    out.cosThetaO = std::cos(thetaO);
    return out;
}

// Surface area orientation heuristic of one side of a split along `dim`.
float Cost(const Bounds& b, const Box& parent, u32 dim)
{
    const float thetaO = SafeAcos(b.cosThetaO);
    const float thetaE = SafeAcos(b.cosThetaE);
    const float thetaW = std::min(thetaO + thetaE, kPi);
    const float sinThetaO = SafeSqrt(1.0f - b.cosThetaO * b.cosThetaO);
    const float orientation =
        2.0f * kPi * (1.0f - b.cosThetaO) +
        kPi / 2.0f *
            (2.0f * thetaW * sinThetaO - std::cos(thetaO - 2.0f * thetaW) - 2.0f * thetaO * sinThetaO + b.cosThetaO);
    const float3 diagonal = parent.Diagonal();
    const float extent = Axis(diagonal, dim);
    const float regularity = extent > 0.0f ? std::max({diagonal.x, diagonal.y, diagonal.z}) / extent : 1.0f;
    return b.power * orientation * regularity * b.box.SurfaceArea();
}

struct Item
{
    u32 light;
    Bounds bounds;
};

struct Builder
{
    std::vector<gpu::LightTriangle>& lights;
    std::vector<gpu::LightTreeNode> nodes;

    u32 Bucket(const Box& centroids, const Bounds& b, u32 dim) const
    {
        const float lo = Axis(centroids.lo, dim);
        const float extent = Axis(centroids.hi, dim) - lo;
        const float offset = extent > 0.0f ? (Axis(b.Centroid(), dim) - lo) / extent : 0.0f;
        return std::min(u32(offset * float(kBuckets)), kBuckets - 1);
    }

    Bounds Build(std::span<Item> items, u64 trail, u32 depth)
    {
        const u32 index = u32(nodes.size());
        nodes.emplace_back();
        if (items.size() == 1)
        {
            const Item& item = items[0];
            gpu::LightTriangle& light = lights[item.light];
            light.treeTrailLow = u32(trail);
            light.treeTrailHigh = u32(trail >> 32);
            light.treeDepth = depth;
            Store(index, item.bounds, item.light, true);
            return item.bounds;
        }

        Box box;
        Box centroids;
        for (const Item& item : items)
        {
            box.Grow(item.bounds.box);
            centroids.Grow(item.bounds.Centroid());
        }
        // Cheapest split between buckets of centroids, over the three axes.
        float bestCost = std::numeric_limits<float>::max();
        u32 bestDim = ~0u;
        u32 bestBucket = 0;
        for (u32 dim = 0; dim < 3 && depth < kBalancedDepth; ++dim)
        {
            if (Axis(centroids.hi, dim) <= Axis(centroids.lo, dim))
            {
                continue;
            }
            Bounds buckets[kBuckets];
            for (const Item& item : items)
            {
                Bounds& bucket = buckets[Bucket(centroids, item.bounds, dim)];
                bucket = Union(bucket, item.bounds);
            }
            for (u32 split = 0; split + 1 < kBuckets; ++split)
            {
                Bounds below;
                Bounds above;
                for (u32 b = 0; b <= split; ++b)
                {
                    below = Union(below, buckets[b]);
                }
                for (u32 b = split + 1; b < kBuckets; ++b)
                {
                    above = Union(above, buckets[b]);
                }
                const float cost = Cost(below, box, dim) + Cost(above, box, dim);
                if (below.power > 0.0f && above.power > 0.0f && cost < bestCost)
                {
                    bestCost = cost;
                    bestDim = dim;
                    bestBucket = split;
                }
            }
        }
        size_t mid = items.size() / 2;
        if (bestDim != ~0u)
        {
            const auto it = std::partition(items.begin(), items.end(), [&](const Item& item)
                                           { return Bucket(centroids, item.bounds, bestDim) <= bestBucket; });
            mid = size_t(it - items.begin());
        }
        if (mid == 0 || mid == items.size())
        {
            mid = items.size() / 2; // no useful split (one bucket, coincident lights): halve
        }
        AIR_CHECK(depth < kMaxDepth, "light tree: {} lights are too many for its 64-bit trails", lights.size());
        const Bounds first = Build(items.subspan(0, mid), trail, depth + 1);
        const u32 second = u32(nodes.size());
        const Bounds last = Build(items.subspan(mid), trail | (u64(1) << depth), depth + 1);
        const Bounds bounds = Union(first, last);
        Store(index, bounds, second, false);
        return bounds;
    }

    void Store(u32 index, const Bounds& b, u32 child, bool leaf)
    {
        gpu::LightTreeNode& node = nodes[index];
        node.boundsMin = b.box.lo;
        node.boundsMax = b.box.hi;
        node.power = b.power;
        node.axis = b.axis;
        node.cosThetaO = b.cosThetaO;
        node.cosThetaE = b.cosThetaE;
        node.child = child;
        node.isLeaf = leaf ? 1u : 0u;
    }
};

} // namespace

std::vector<gpu::LightTreeNode> BuildLightTree(std::vector<gpu::LightTriangle>& lights)
{
    std::vector<Item> items;
    for (u32 i = 0; i < lights.size(); ++i)
    {
        gpu::LightTriangle& light = lights[i];
        light.treeTrailLow = 0;
        light.treeTrailHigh = 0;
        light.treeDepth = gpu::kInvalidIndex;
        const float3 normal = cross(light.e1, light.e2);
        if (light.probability <= 0.0f || dot(normal, normal) <= 0.0f)
        {
            continue; // no power: never sampled, probability 0 for MIS
        }
        Item item;
        item.light = i;
        item.bounds.box.Grow(light.p0);
        item.bounds.box.Grow(light.p0 + light.e1);
        item.bounds.box.Grow(light.p0 + light.e2);
        item.bounds.axis = normalize(normal);
        item.bounds.cosThetaO = 1.0f; // one normal
        item.bounds.cosThetaE = 0.0f; // a one-sided Lambertian emitter: the whole front hemisphere
        item.bounds.power = light.probability;
        items.push_back(item);
    }
    Builder builder{lights, {}};
    if (items.empty())
    {
        builder.nodes.push_back({}); // a leaf without power: sampling finds nothing
        builder.nodes[0].isLeaf = 1u;
        return builder.nodes;
    }
    builder.Build(items, 0, 0);
    return builder.nodes;
}

} // namespace air::renderer
