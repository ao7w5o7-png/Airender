// Probe GI: the light the surface voxels reflect (ProbeGiShared.h), every cascade in one dispatch (y: the
// cascade), one thread per face of a voxel (x: voxel * 6 + face) so that no thread runs the rays of six faces
// one after the other. Per face that shows a surface, at the surface it averages (not the voxel's centre): next
// event estimation of the sun (a direction in its disc) and of one emissive triangle, each with a shadow ray
// through the distance field, plus the indirect light: radiosity rays from the face through the voxels (the
// light they hit, from the frames before: every further bounce over frames; past radiosityDistance voxels the
// radiance cache's), or the cache's irradiance; times albedo / pi. The triangle is resampled (Talbot et al.
// 2005) from kLightCandidates drawn by power from the light list's alias table, in proportion to their
// unshadowed light here: one draw alone mostly picks a light far away. Each face averages its first estimates
// and then blends new ones in (CascadeData::lightBlend; its age counts them). A frame lights
// CascadeData::lightBudget voxels of each cascade, cycling through them (all when 0); and, in a second dispatch per
// cascade, the voxels new where it moved, every frame while they are young (kFreshFrames, ProbeGiShared.h).
//   GiConstants: lights, input1 radiosity rays per face (0: the cache), input2 radiosity distance (voxels); for the
//   new voxels cascade and input3 their ring (StructuredBuffer<uint2>), else input3 kInvalidIndex.

#include "VoxelTrace.hlsli"

[[vk::push_constant]] ConstantBuffer<GiConstants> g : register(b0);

static const float kShadowOffset = 1.0f;  // voxels along the face's normal where its shadow rays start
static const float kSurfaceJitter = 0.8f; // voxels: the lighting point's range over the face (+-0.4)
// Voxels: a light lying within the surface's band closer than this is taken as visible (further away it
// lights the surface at a grazing angle, little, and a wall may stand between: a light on the ceiling of the
// next room is level with this room's ceiling).
static const float kLevelLightReach = 4.0f;
static const float kInteriorMargin = 4.0f; // voxels of the finer cascade
static const uint kLightCandidates = 8u;
static const uint kAgeMax = 15u; // 4 bits
static const float kDirectBlendScale = 0.25f;
// Faces of a wave that share their estimates (Main).
static const float kShareRadius = 1.75f; // voxels between the points they were lit at
static const float kShareNormal = 0.9f;
static const float kSharePlane = 0.5f; // voxels off the face's plane

float2 SampleTriangle(float2 u)
{
    float s = sqrt(u.x);
    return float2(1.0f - s, u.y * s);
}

// One face's lighting: its irradiance this lighting, direct and indirect.
struct FaceLight
{
    uint slot;
    uint face;
    float3 surface; // where it was lit
    float3 n;
    float3 albedo;
    float3 direct;
    float3 indirect;
    uint seed;
};
// The estimate of a face's light this lighting, unshared (false: thread id lights no face this frame).
bool LightFace(GiFrame f, uint3 id, out FaceLight r)
{
    r = (FaceLight)0;
    bool fresh = g.input3 != kInvalidIndex; // the new voxels of cascade g.cascade, else the budgets (y: the cascade)
    uint cascade = fresh ? g.cascade : id.y;
    if (cascade >= f.cascadeCount || f.cascades[cascade].valid == 0u)
    {
        return false;
    }
    CascadeData c = f.cascades[cascade];
    ByteAddressBuffer counters = ResourceDescriptorHeap[c.counters];
    RWStructuredBuffer<SurfaceVoxel> voxels = ResourceDescriptorHeap[c.voxelsWritable];
    uint capacity;
    uint stride;
    voxels.GetDimensions(capacity, stride);
    uint limit = min(counters.Load(kCounterVoxels), capacity);
    uint index = id.x / 6u;
    uint face = id.x % 6u;
    uint slot;
    if (fresh)
    {
        // The newest first, while they are kFreshFrames frames young.
        uint head = counters.Load(kCounterFreshHead);
        if (index >= min(head, kFreshWindow))
        {
            return false;
        }
        StructuredBuffer<uint2> ring = ResourceDescriptorHeap[g.input3];
        uint2 entry = ring[(head - 1u - index) % kFreshCapacity];
        if (f.frameIndex - entry.y >= kFreshFrames || entry.x >= limit)
        {
            return false;
        }
        slot = entry.x;
    }
    else
    {
        if (index >= (c.lightBudget == 0u ? limit : min(c.lightBudget, limit)))
        {
            return false;
        }
        slot = c.lightBudget == 0u ? index : (f.frameIndex * c.lightBudget + index) % limit;
    }
    uint coord = voxels[slot].coord;
    if (((coord >> 24) & (1u << face)) == 0u)
    {
        return false;
    }
    gVoxelsWritable = true;
    gSkyTable = f.skyTable;
    int3 origin = CascadeOrigin(c);
    uint3 storage = uint3(coord & 0xFFu, (coord >> 8) & 0xFFu, (coord >> 16) & 0xFFu);
    int3 local = int3((storage - uint3(origin & int(kVoxelResolution - 1u))) & (kVoxelResolution - 1u));
    float3 centre = (float3(origin + local) + 0.5f) * c.voxelSize;
    // A voxel inside the next finer cascade is rarely seen (rays that start there trace the finer one): it is lit
    // eight times less often.
    if (!fresh && c.lightBudget != 0u && cascade > 0u && f.cascades[cascade - 1u].valid != 0u &&
        InCascade(f.cascades[cascade - 1u], centre, kInteriorMargin) && ((f.frameIndex + slot) & 7u) != 0u)
    {
        return false;
    }
    uint seed = Pcg((slot * 6u + face) ^ Pcg(f.frameIndex * 8u + cascade));
    float offset;
    float3 n = UnpackNormalOffset(voxels[slot].normal[face], offset);
    float3 albedo = UnpackUnorm4(voxels[slot].albedo[face]).xyz;
    // The surface the face shows, at a point that moves over it from one lighting to the next, so that over
    // frames the face takes the average of its light (a light close to the surface lights it by 1 / d^2: its
    // value at the centre alone is far from the average); and where its shadow rays start: a voxel off it, out of
    // its own band of the distance field. A point moved into other geometry (a wall next to a floor) or past a
    // wall that crosses the voxel (a thin wall between two rooms) stays at the centre.
    float3 surface = centre + n * (offset * c.voxelSize);
    Basis tangent = MakeBasis(n);
    float2 shift = (Rand2(seed) - 0.5f) * kSurfaceJitter;
    float3 moved = surface + (tangent.t * shift.x + tangent.b * shift.y) * c.voxelSize;
    float3 p = surface + n * (kShadowOffset * c.voxelSize);
    float3 movedP = moved + n * (kShadowOffset * c.voxelSize);
    float3 step = movedP - p;
    float stepLength = length(step);
    if (SampleDistance(c, movedP / c.voxelSize) >= kHitDistance &&
        (stepLength < 1e-6f || VoxelsVisibleFrom(f, p, step / stepLength, 0.0f, stepLength, cascade)))
    {
        p = movedP;
        surface = moved;
    }
    float3 irradiance = float3(0.0f, 0.0f, 0.0f);
    if (HasSun(f.sky) && (f.flags & kGiFurnace) == 0u)
    {
        float3 wi = SampleSunDirection(f.sky, Rand2(seed));
        float cosine = dot(n, wi);
        if (cosine > 0.0f && VoxelsVisibleFrom(f, p, wi, 0.0f, 1e30f, cascade))
        {
            irradiance += f.sky.sunRadiance * (SunSolidAngle(f.sky) * cosine);
        }
    }
    if (f.lightCount > 0u && (f.flags & kGiFurnace) == 0u)
    {
        StructuredBuffer<LightTriangle> lights = ResourceDescriptorHeap[g.lights];
        // The candidates' weights: unshadowed luminance here per their density (area measure).
        float3 lightPoint = float3(0.0f, 0.0f, 0.0f);
        float3 lightRadiance = float3(0.0f, 0.0f, 0.0f);
        float3 lightNormal = float3(0.0f, 0.0f, 1.0f);
        float chosenTarget = 0.0f;
        float weightSum = 0.0f;
        for (uint k = 0u; k < kLightCandidates; ++k)
        {
            float uSelect = Rand(seed);
            float uAlias = Rand(seed);
            float2 uPoint = Rand2(seed);
            float uKeep = Rand(seed);
            LightTriangle light = lights[min(uint(uSelect * float(f.lightCount)), f.lightCount - 1u)];
            if (uAlias >= light.aliasThreshold)
            {
                light = lights[light.alias];
            }
            float2 b = SampleTriangle(uPoint);
            float3 candidate = light.p0 + light.e1 * b.x + light.e2 * b.y;
            float3 toLight = candidate - surface;
            float dist2 = max(dot(toLight, toLight), 1e-8f);
            float3 wi = toLight * rsqrt(dist2);
            float3 normal = normalize(cross(light.e1, light.e2));
            float target = Luminance(light.radiance) * max(dot(n, wi), 0.0f) * max(dot(-wi, normal), 0.0f) / dist2;
            if (target <= 0.0f || light.probability <= 0.0f)
            {
                continue;
            }
            float w = target * light.area / light.probability;
            weightSum += w;
            if (uKeep * weightSum < w)
            {
                lightPoint = candidate;
                lightRadiance = light.radiance;
                lightNormal = normal;
                chosenTarget = target;
            }
        }
        float3 toLight = lightPoint - surface;
        float dist = length(toLight);
        float3 wi = toLight / max(dist, 1e-6f);
        float cosine = dot(n, wi);
        float cosLight = dot(-wi, lightNormal);
        if (chosenTarget > 0.0f && cosine > 0.0f && cosLight > 0.0f)
        {
            // The shadow ray goes from a voxel off the surface to the light and stops before the emitter's own
            // band of the distance field, which it enters early when it arrives at a grazing angle (lights on a
            // ceiling). A light lying within the surface's own band (lamps standing on a floor, for the floor's
            // voxels) is below what the voxels resolve: taken as visible, since any ray to it would run through
            // the surface's band.
            // The margin is a voxel of the cascade the ray reaches the light in (it only moves to coarser ones),
            // whose band around the emitter (and a ceiling it hangs from) is the thicker one.
            float3 toShadowEnd = lightPoint - p;
            float shadowDist = length(toShadowEnd);
            uint end = max(FindCascade(f, lightPoint, kCascadeMargin), cascade);
            float endVoxel = end < f.cascadeCount ? f.cascades[end].voxelSize : c.voxelSize;
            float tMax = shadowDist - kShadowOffset * endVoxel / max(cosLight, 0.1f);
            bool level = dot(lightPoint - surface, n) < kShadowOffset * c.voxelSize &&
                         dist < kLevelLightReach * c.voxelSize;
            if (level || tMax <= 0.0f ||
                VoxelsVisibleFrom(f, p, toShadowEnd / max(shadowDist, 1e-6f), 0.0f, tMax, cascade))
            {
                // f / target times the candidates' mean weight (f: the unshadowed irradiance, target its luminance).
                irradiance += lightRadiance / max(Luminance(lightRadiance), 1e-12f) * (weightSum / float(kLightCandidates));
            }
        }
    }
    // The indirect light (the sky, and what other voxels reflect: all further bounces over frames).
    float3 indirect = float3(0.0f, 0.0f, 0.0f);
    if ((f.flags & kGiDirectVoxels) == 0u)
    {
        uint rays = g.input1;
        if (rays > 0u)
        {
            float tCache = g.input2 > 0u ? float(g.input2) * c.voxelSize : 1e30f;
            float3 sum = float3(0.0f, 0.0f, 0.0f);
            for (uint r = 0u; r < rays; ++r)
            {
                float3 wi = ToWorld(tangent, SampleCosineConcentric(Rand2(seed)));
                float distance;
                uint source;
                sum += TraceLight(f, p, wi, 0.0f, tCache, cascade, distance, source);
            }
            indirect += sum * (kPi / float(rays));
        }
        else
        {
            float3 cached;
            if (CacheIrradiance(f, surface + n * (0.5f * c.voxelSize), n, cached))
            {
                indirect += cached;
            }
        }
    }
    r.slot = slot;
    r.face = face;
    r.surface = surface;
    r.n = n;
    r.albedo = albedo;
    r.direct = irradiance;
    r.indirect = indirect;
    r.seed = seed;
    return true;
}

[numthreads(64, 1, 1)]
void Main(uint3 id : SV_DispatchThreadID)
{
    GiFrame f = LoadFrame(g.frameData, g.frameOffset);
    uint cascade = g.input3 != kInvalidIndex ? g.cascade : id.y;
    if (cascade >= f.cascadeCount || f.cascades[cascade].valid == 0u)
    {
        return;
    }
    CascadeData c = f.cascades[cascade];
    FaceLight r;
    bool active = LightFace(f, id, r);
    // The faces of a wave are those of a few neighbouring voxels (slots are handed out in the order a
    // voxelization visits the voxels): each takes the mean of the estimates of the faces among them that show the
    // same surface (the same axis, a like normal, the same plane, within kShareRadius voxels), several rays for
    // the price of one, over a voxel or two. Light from one shadow ray and one radiosity ray per face otherwise
    // stays noisy for many lightings, and the noise of a coarse voxel, which the rays of many probes hit, moves the
    // whole image's light slowly up and down (many_lights: by 3% over a hundred frames).
    float3 direct = float3(0.0f, 0.0f, 0.0f);
    float3 indirect = float3(0.0f, 0.0f, 0.0f);
    float count = 0.0f;
    for (uint lane = 0u; lane < WaveGetLaneCount(); ++lane)
    {
        bool other = WaveReadLaneAt(active, lane);
        uint face = WaveReadLaneAt(r.face, lane);
        float3 surface = WaveReadLaneAt(r.surface, lane);
        float3 n = WaveReadLaneAt(r.n, lane);
        float3 d = WaveReadLaneAt(r.direct, lane);
        float3 i = WaveReadLaneAt(r.indirect, lane);
        float3 offset = (surface - r.surface) / c.voxelSize;
        if (active && other && face == r.face && dot(n, r.n) > kShareNormal &&
            dot(offset, offset) <= kShareRadius * kShareRadius && abs(dot(offset, r.n)) < kSharePlane)
        {
            direct += d;
            indirect += i;
            count += 1.0f;
        }
    }
    if (!active)
    {
        return;
    }
    direct /= count;
    indirect /= count;
    // The first estimates are averaged (a new voxel's light starts as a guess: age 0 replaces it), then blended: the
    // direct light at a quarter of the indirect light's rate (SurfaceVoxel::direct).
    RWStructuredBuffer<SurfaceVoxel> voxels = ResourceDescriptorHeap[c.voxelsWritable];
    uint slot = r.slot;
    uint face = r.face;
    uint seed = r.seed;
    // CascadeData::lightBlend 1: lit whole (built now, or the lights changed): the estimates start over.
    bool restart = c.lightBlend >= 1.0f;
    uint age = restart ? 0u : (voxels[slot].age >> (face * 4u)) & kAgeMax;
    float average = 1.0f / float(age + 1u);
    float3 oldDirect = UnpackRadiance(voxels[slot].direct[face]);
    float3 oldIndirect = max(UnpackRadiance(voxels[slot].radiance[face]) - oldDirect, 0.0f);
    float3 newDirect = lerp(oldDirect, r.albedo * kInvPi * direct, max(c.lightBlend * kDirectBlendScale, average));
    float3 bounced = lerp(oldIndirect, r.albedo * kInvPi * indirect, max(c.lightBlend, average));
    uint packedDirect = PackRadiance(newDirect, Rand(seed));
    voxels[slot].direct[face] = packedDirect;
    voxels[slot].radiance[face] = PackRadiance(UnpackRadiance(packedDirect) + bounced, Rand(seed));
    if (restart)
    {
        InterlockedAnd(voxels[slot].age, ~(kAgeMax << (face * 4u)));
        InterlockedOr(voxels[slot].age, 1u << (face * 4u));
    }
    else if (age < kAgeMax)
    {
        InterlockedAdd(voxels[slot].age, 1u << (face * 4u));
    }
}