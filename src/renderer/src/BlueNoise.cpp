#include "air/renderer/BlueNoise.h"

#include "air/core/Assert.h"

#include <algorithm>
#include <cmath>

namespace air::renderer
{
namespace
{

// The energy of points under a Gaussian of sigma 1.5 texels on a size x size torus, updated as
// points come and go (the filter is cut where it falls below 1e-4).
class Energy
{
public:
    explicit Energy(u32 size) : mSize(size), mValues(size_t(size) * size, 0.0)
    {
        constexpr double kSigma = 1.5;
        mRadius = int(std::min<u32>(size / 2, 7));
        for (int dy = -mRadius; dy <= mRadius; ++dy)
        {
            for (int dx = -mRadius; dx <= mRadius; ++dx)
            {
                mKernel.push_back(std::exp(-double(dx * dx + dy * dy) / (2.0 * kSigma * kSigma)));
            }
        }
    }

    void Add(u32 index, double sign)
    {
        const int x0 = int(index % mSize);
        const int y0 = int(index / mSize);
        const int n = int(mSize);
        size_t k = 0;
        for (int dy = -mRadius; dy <= mRadius; ++dy)
        {
            const int y = ((y0 + dy) % n + n) % n;
            for (int dx = -mRadius; dx <= mRadius; ++dx, ++k)
            {
                const int x = ((x0 + dx) % n + n) % n;
                mValues[size_t(y) * mSize + size_t(x)] += sign * mKernel[k];
            }
        }
    }

    // The point with the highest energy among those that are set (the tightest cluster) or the
    // lowest among those that are not (the largest void); ties go to the lowest index.
    u32 Tightest(const std::vector<u8>& points) const { return Find(points, 1, true); }
    u32 Largest(const std::vector<u8>& points) const { return Find(points, 0, false); }

private:
    u32 Find(const std::vector<u8>& points, u8 state, bool highest) const
    {
        u32 best = ~0u;
        for (u32 i = 0; i < u32(mValues.size()); ++i)
        {
            if (points[i] != state)
            {
                continue;
            }
            if (best == ~0u || (highest ? mValues[i] > mValues[best] : mValues[i] < mValues[best]))
            {
                best = i;
            }
        }
        return best;
    }

    u32 mSize;
    int mRadius = 0;
    std::vector<double> mKernel;
    std::vector<double> mValues;
};

u32 Next(u32& state)
{
    state = state * 747796405u + 2891336453u;
    u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

} // namespace

std::vector<u16> MakeBlueNoise(u32 size, u32 seed)
{
    AIR_CHECK(size >= 4 && size <= 256, "MakeBlueNoise: size {} is not in 4..256", size);
    const u32 count = size * size;
    // An initial pattern of a tenth of the points, at random, relaxed until removing its tightest
    // cluster and filling its largest void are the same move.
    std::vector<u8> initial(count, 0);
    const u32 ones = std::max(count / 10, 1u);
    u32 state = seed * 0x9E3779B9u + 1u;
    for (u32 placed = 0; placed < ones;)
    {
        const u32 i = Next(state) % count;
        if (initial[i] == 0)
        {
            initial[i] = 1;
            ++placed;
        }
    }
    Energy energy(size);
    for (u32 i = 0; i < count; ++i)
    {
        if (initial[i] != 0)
        {
            energy.Add(i, 1.0);
        }
    }
    for (u32 iteration = 0; iteration < 4 * count; ++iteration)
    {
        const u32 cluster = energy.Tightest(initial);
        initial[cluster] = 0;
        energy.Add(cluster, -1.0);
        const u32 hole = energy.Largest(initial);
        initial[hole] = 1;
        energy.Add(hole, 1.0);
        if (hole == cluster)
        {
            break;
        }
    }

    std::vector<u16> ranks(count, 0);
    // The initial points get the low ranks, the last one removed from the tightest cluster rank 0.
    std::vector<u8> points = initial;
    Energy removing = energy;
    for (u32 rank = ones; rank-- > 0;)
    {
        const u32 cluster = removing.Tightest(points);
        points[cluster] = 0;
        removing.Add(cluster, -1.0);
        ranks[cluster] = u16(rank);
    }
    // The others, each in the largest void left. (Past half of the points, the void of the points is
    // the tightest cluster of the gaps, so one rule serves to the end.)
    points = initial;
    for (u32 rank = ones; rank < count; ++rank)
    {
        const u32 hole = energy.Largest(points);
        points[hole] = 1;
        energy.Add(hole, 1.0);
        ranks[hole] = u16(rank);
    }
    return ranks;
}

const std::vector<u16>& BlueNoise()
{
    static const std::vector<u16> noise = MakeBlueNoise(kBlueNoiseSize, 1);
    return noise;
}

} // namespace air::renderer
