#include "TestFramework.h"

#include "air/core/TaskSystem.h"

#include <atomic>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace air;

TEST(TaskSystem_ChainRunsInDependencyOrder)
{
    TaskSystem tasks(4);
    std::mutex mutex;
    std::vector<int> order;
    auto push = [&](int v)
    {
        std::lock_guard lock(mutex);
        order.push_back(v);
    };
    TaskHandle a = tasks.Launch("A", [&] { push(1); });
    TaskHandle b = tasks.Launch("B", [&] { push(2); }, {&a, 1});
    TaskHandle c = tasks.Launch("C", [&] { push(3); }, {&b, 1});
    c.Wait();
    EXPECT((order == std::vector<int>{1, 2, 3}));
}

TEST(TaskSystem_DiamondWaitsForAllPrerequisites)
{
    TaskSystem tasks(4);
    std::atomic<int> done{0};
    std::atomic<bool> sawBoth{false};
    TaskHandle root = tasks.Launch("root", [] {});
    TaskHandle left = tasks.Launch("left", [&] { ++done; }, {&root, 1});
    TaskHandle right = tasks.Launch("right", [&] { ++done; }, {&root, 1});
    TaskHandle deps[] = {left, right};
    TaskHandle join = tasks.Launch("join", [&] { sawBoth = done.load() == 2; }, deps);
    join.Wait();
    EXPECT(sawBoth.load());
}

TEST(TaskSystem_ManyTasksAndWhenAll)
{
    TaskSystem tasks(0);
    std::atomic<int> counter{0};
    std::vector<TaskHandle> handles;
    for (int i = 0; i < 5000; ++i)
    {
        handles.push_back(tasks.Launch("inc", [&] { counter.fetch_add(1, std::memory_order_relaxed); }));
    }
    tasks.WhenAll(handles).Wait();
    EXPECT(counter.load() == 5000);
}

TEST(TaskSystem_ParallelForCoversEveryIndexOnce)
{
    TaskSystem tasks(4);
    std::vector<std::atomic<int>> hits(10007);
    tasks.ParallelFor(u32(hits.size()), [&](u32 i) { hits[i].fetch_add(1); }, 64);
    bool allOnce = true;
    for (auto& h : hits)
    {
        allOnce = allOnce && h.load() == 1;
    }
    EXPECT(allOnce);
}

TEST(TaskSystem_PipeSerialisesWithoutLocks)
{
    TaskSystem tasks(4);
    Pipe pipe(tasks, "Ordered");
    std::vector<int> values; // deliberately unsynchronised: the pipe is the synchronisation
    for (int i = 0; i < 2000; ++i)
    {
        pipe.Launch("push", [&values, i] { values.push_back(i); });
    }
    pipe.WaitIdle();
    std::vector<int> expected(2000);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT(values == expected);
}

TEST(TaskSystem_NestedWaitOnSingleWorkerDoesNotDeadlock)
{
    TaskSystem tasks(1);
    std::atomic<int> result{0};
    TaskHandle outer =
        tasks.Launch("outer",
                     [&]
                     {
                         TaskHandle inner = tasks.Launch("inner", [&] { result = 42; });
                         inner.Wait(); // the only worker is busy here; Wait() must execute 'inner' itself
                     });
    outer.Wait();
    EXPECT(result.load() == 42);
}

TEST(TaskSystem_ExceptionDoesNotBreakDependents)
{
    TaskSystem tasks(2);
    std::atomic<bool> ran{false};
    TaskHandle bad = tasks.Launch("throws", [] { throw std::runtime_error("boom"); });
    TaskHandle after = tasks.Launch("after", [&] { ran = true; }, {&bad, 1});
    after.Wait();
    EXPECT(ran.load());
}
