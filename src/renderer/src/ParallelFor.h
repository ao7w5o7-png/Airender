#pragma once

// A parallel loop over all hardware threads for the renderer's CPU precomputation (sky tables, cloud
// volumes), which runs before any TaskSystem exists. Internal to the renderer module.

#include <algorithm>
#include <thread>
#include <vector>

namespace air::renderer::detail
{

// Runs body(i) for i in [0, count), rows interleaved across the threads.
template <typename F> void ParallelFor(int count, F&& body)
{
    const int threads = std::max(1, std::min(int(std::thread::hardware_concurrency()), count));
    std::vector<std::thread> pool;
    pool.reserve(size_t(threads));
    for (int t = 0; t < threads; ++t)
    {
        pool.emplace_back(
            [&, t]
            {
                for (int i = t; i < count; i += threads)
                {
                    body(i);
                }
            });
    }
    for (std::thread& thread : pool)
    {
        thread.join();
    }
}

} // namespace air::renderer::detail
