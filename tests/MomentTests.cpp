// Moment shadow maps (shader/MomentShadow.hlsli): the octahedral map, the moments a march adds up, and
// the reconstruction: within the Markov-Krein bounds of the true distribution of optical depth, close
// to it for the smooth profiles of smoke, and well behaved with the moments rounded to half floats and
// mixed by bilinear filtering.

#include "TestFramework.h"

#include "air/renderer/Transfer.h"
#include "air/renderer/shader/MomentShadowShared.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace air;
using namespace air::hlsl;
using namespace air::gpu;

namespace
{

// MomentShadow.hlsli on the CPU (the map lookups are not exercised here).
struct MomentReader
{
    float MomentMapB0Sample(MomentShadowMap /*m*/, float2 /*uv*/) const { return 0.0f; }
    float4 MomentMapBSample(MomentShadowMap /*m*/, float2 /*uv*/) const { return float4(0.0f); }

// NOLINTBEGIN(readability-identifier-naming)
#include "air/renderer/shader/MomentShadow.hlsli"
    // NOLINTEND(readability-identifier-naming)
};

u32 Hash(u32 v)
{
    u32 state = v * 747796405u + 2891336453u;
    u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float Uniform(u32& seed)
{
    seed = Hash(seed);
    return float(seed >> 8) * (1.0f / 16777216.0f);
}

// The optical depth of a ray as kSteps steps over z in [-1, 1]: one to three soft blobs (smoke), or a
// slab of even density, scaled to a total depth from 0.05 to 50.
constexpr int kSteps = 512;

struct Profile
{
    std::vector<double> tau; // per step
    double total = 0.0;
};

double StepDepth(int k)
{
    return -1.0 + (double(k) + 0.5) * (2.0 / kSteps);
}

Profile RandomProfile(u32& seed)
{
    Profile p;
    p.tau.assign(kSteps, 0.0);
    const int kind = int(Uniform(seed) * 4.0f);
    if (kind == 3)
    {
        const double a = -1.0 + 1.6 * Uniform(seed);
        const double b = std::min(a + 0.05 + 0.8 * Uniform(seed), 1.0);
        for (int k = 0; k < kSteps; ++k)
        {
            p.tau[k] = StepDepth(k) >= a && StepDepth(k) < b ? 1.0 : 0.0;
        }
    }
    else
    {
        for (int blob = 0; blob <= kind; ++blob)
        {
            const double centre = -0.9 + 1.8 * Uniform(seed);
            const double width = 0.02 + 0.3 * Uniform(seed);
            const double amplitude = 0.2 + Uniform(seed);
            for (int k = 0; k < kSteps; ++k)
            {
                const double x = (StepDepth(k) - centre) / width;
                p.tau[k] += amplitude * std::exp(-0.5 * x * x);
            }
        }
    }
    double sum = 0.0;
    for (double t : p.tau)
    {
        sum += t;
    }
    p.total = 0.05 * std::pow(1000.0, double(Uniform(seed))); // 0.05 .. 50
    for (double& t : p.tau)
    {
        t *= p.total / sum;
    }
    return p;
}

void Moments(const Profile& p, double& b0, double b[4])
{
    b0 = 0.0;
    b[0] = b[1] = b[2] = b[3] = 0.0;
    for (int k = 0; k < kSteps; ++k)
    {
        const double z = StepDepth(k);
        b0 += p.tau[k];
        b[0] += p.tau[k] * z;
        b[1] += p.tau[k] * z * z;
        b[2] += p.tau[k] * z * z * z;
        b[3] += p.tau[k] * z * z * z * z;
    }
}

// The share of the optical depth in front of z (z between steps).
double TrueShare(const Profile& p, double z)
{
    double front = 0.0;
    for (int k = 0; k < kSteps && StepDepth(k) < z; ++k)
    {
        front += p.tau[k];
    }
    return front / p.total;
}

float Half(float v)
{
    return renderer::HalfToFloat(renderer::FloatToHalf(v));
}

// A ray's moments as bilinear filtering mixes them: one profile, or four with 8-bit weights.
Profile RandomTexelMix(u32& seed, bool single)
{
    Profile mix;
    mix.tau.assign(kSteps, 0.0);
    const float wx = std::round(Uniform(seed) * 256.0f) / 256.0f;
    const float wy = std::round(Uniform(seed) * 256.0f) / 256.0f;
    const double weights[4] = {(1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy};
    for (int t = 0; t < (single ? 1 : 4); ++t)
    {
        const Profile p = RandomProfile(seed);
        for (int k = 0; k < kSteps; ++k)
        {
            mix.tau[k] += p.tau[k] * (single ? 1.0 : weights[t]);
        }
    }
    for (double v : mix.tau)
    {
        mix.total += v;
    }
    return mix;
}

struct ReconstructionStats
{
    double outside = 0.0; // how far the truth lies outside the bounds the moments give
    double meanError = 0.0;
    double worstRise = 0.0; // transmittance growing with depth
    int nans = 0;
};

// 2000 rays (half of them mixes of four), each queried at 64 depths.
ReconstructionStats Reconstruct(bool halves, float bias)
{
    MomentReader reader;
    ReconstructionStats stats;
    u32 seed = 17;
    int count = 0;
    for (int trial = 0; trial < 2000; ++trial)
    {
        const Profile mix = RandomTexelMix(seed, trial % 2 == 0);
        double b0;
        double b[4];
        Moments(mix, b0, b);
        const float fb0 = halves ? Half(float(b0)) : float(b0);
        const float4 fb = halves ? float4(Half(float(b[0])), Half(float(b[1])), Half(float(b[2])), Half(float(b[3])))
                                 : float4(float(b[0]), float(b[1]), float(b[2]), float(b[3]));
        float previous = 1.0f;
        for (int q = 0; q < 64; ++q)
        {
            const float z = -1.0f + (float(q) + 0.37f) / 32.0f;
            const float share = reader.MomentAbsorbanceShare(fb / fb0, z, bias, MomentReader::kMomentOverestimation);
            const float lower = reader.MomentAbsorbanceShare(fb / fb0, z, bias, 0.0f);
            const float upper = reader.MomentAbsorbanceShare(fb / fb0, z, bias, 1.0f);
            if (!std::isfinite(share) || !std::isfinite(lower) || !std::isfinite(upper))
            {
                ++stats.nans;
                continue;
            }
            const double truth = TrueShare(mix, z);
            stats.outside = std::max(stats.outside, std::max(double(lower) - truth, truth - double(upper)));
            const float transmittance = std::exp(-fb0 * share);
            stats.meanError += std::abs(double(transmittance) - std::exp(-mix.total * truth));
            stats.worstRise = std::max(stats.worstRise, double(transmittance - previous));
            previous = transmittance;
            ++count;
        }
    }
    stats.meanError /= std::max(count, 1);
    return stats;
}

} // namespace

TEST(Moments_OctahedralMapRoundTrips)
{
    MomentReader reader;
    u32 seed = 3;
    double worst = 0.0;
    for (int i = 0; i < 20000; ++i)
    {
        const float z = Uniform(seed) * 2.0f - 1.0f;
        const float phi = Uniform(seed) * 6.2831853f;
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float3 d(r * std::cos(phi), r * std::sin(phi), z);
        const float2 uv = reader.OctahedralUv(d);
        EXPECT(uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f);
        const float3 back = reader.OctahedralDirection(uv);
        worst = std::max(worst, double(length(back - d)));
    }
    EXPECT(worst < 1e-5);
}

// From exact moments the reconstruction is a pair of sharp bounds: whatever distribution of optical
// depth has these moments, the share in front of a depth lies between the three-point distributions
// that count the point at that depth not at all and wholly. The 4-moment estimate between them
// (a quarter of that point) misses the true transmittance by 0.034 on average over random smoke,
// slabs and their mixes (2000 rays, total depths 0.05-50): the price of four numbers per texel.
TEST(Moments_ReconstructionStaysWithinTheBounds)
{
    const ReconstructionStats exact = Reconstruct(false, 5e-7f);
    std::printf("  single precision: outside the bounds by %.5f, mean |T error| %.4f\n", exact.outside,
                exact.meanError);
    EXPECT(exact.nans == 0);
    EXPECT(exact.outside < 1e-3);
    EXPECT(exact.meanError < 0.04);
}

// Stored as half floats (11 significant bits, the maps' RGBA16_FLOAT) the moments are pulled harder
// into the valid set: below a bias of 1e-3 rounding made reconstructions fail outright (bounds missed
// by up to 1, transmittance growing with depth); at kMomentBias they hold and cost 0.003 of mean
// error over exact moments.
TEST(Moments_HalfFloatMomentsStayValid)
{
    const ReconstructionStats exact = Reconstruct(false, 5e-7f);
    const ReconstructionStats halves = Reconstruct(true, MomentReader::kMomentBias);
    std::printf("  half floats, bias %.0e: outside the bounds by %.5f, mean |T error| %.4f, rises %.5f\n",
                MomentReader::kMomentBias, halves.outside, halves.meanError, halves.worstRise);
    EXPECT(halves.nans == 0);
    EXPECT(halves.outside < 2e-3);
    EXPECT(halves.worstRise < 1e-3);
    EXPECT(halves.meanError < exact.meanError + 0.005);
    const ReconstructionStats weak = Reconstruct(true, 1e-4f);
    EXPECT(weak.outside > 0.1); // the bias is needed
}

// The transmittance lookup: no volume in front, all of it behind, and in between what the share gives.
TEST(Moments_TransmittanceAtTheEnds)
{
    MomentReader reader;
    float b0 = 0.0f;
    float4 b(0.0f);
    for (int k = 0; k < 64; ++k)
    {
        const float z = -0.5f + (float(k) + 0.5f) / 64.0f; // even density over [-0.5, 0.5]
        reader.AddMoments(b0, b, 2.0f / 64.0f, z);
    }
    EXPECT(std::abs(b0 - 2.0f) < 1e-5f);
    EXPECT(reader.MomentTransmittance(b0, b, -1.0f) == 1.0f);
    EXPECT(reader.MomentTransmittance(b0, b, -0.9f) > 0.97f);
    EXPECT(std::abs(reader.MomentTransmittance(b0, b, 1.0f) - std::exp(-2.0f)) < 1e-6f);
    EXPECT(std::abs(reader.MomentTransmittance(b0, b, 0.9f) - std::exp(-2.0f)) < 0.03f);
    // Half way through the slab (truly exp(-1) = 0.37) the estimate errs on the bright side: the
    // three-point distribution through z = 0 puts 0.44 of the depth there, of which it counts a quarter.
    const float middle = reader.MomentTransmittance(b0, b, 0.0f);
    EXPECT(middle > std::exp(-1.0f) && middle < 0.47f);
    EXPECT(reader.MomentTransmittance(0.0f, float4(0.0f), 0.5f) == 1.0f);
}
