#include "air/core/TaskSystem.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>

namespace air {

struct TaskState {
    std::string name;
    std::function<void()> fn;
    TaskPriority priority = TaskPriority::Normal;
    TaskSystem* system = nullptr;

    // 1 (held by Launch until all prerequisites are registered) + number of incomplete prerequisites.
    std::atomic<i32> pendingDependencies{1};
    std::atomic<bool> done{false};

    std::mutex mutex; // guards completed + subsequents
    bool completed = false;
    std::vector<std::shared_ptr<TaskState>> subsequents;
};

struct TaskSystem::Impl {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::shared_ptr<TaskState>> queues[size_t(TaskPriority::Count)];
    bool stop = false;
    std::atomic<u32> waiters{0};

    std::shared_ptr<TaskState> PopLocked() {
        for (auto& queue : queues) {
            if (!queue.empty()) {
                auto task = std::move(queue.front());
                queue.pop_front();
                return task;
            }
        }
        return nullptr;
    }
    bool HasWorkLocked() const {
        return std::any_of(std::begin(queues), std::end(queues), [](const auto& q) { return !q.empty(); });
    }
};

namespace {
thread_local const TaskSystem* tlsWorkerOwner = nullptr;
}

// ---------------------------------------------------------------------------
// TaskHandle
// ---------------------------------------------------------------------------

bool TaskHandle::IsCompleted() const { return !mState || mState->done.load(std::memory_order_acquire); }

void TaskHandle::Wait() const {
    if (!mState || mState->done.load(std::memory_order_acquire)) {
        return;
    }
    mState->system->HelpUntil(*mState);
}

// ---------------------------------------------------------------------------
// TaskSystem
// ---------------------------------------------------------------------------

TaskSystem::TaskSystem(u32 workerCount) : mImpl(std::make_unique<Impl>()) {
    if (workerCount == 0) {
        workerCount = std::max(1u, std::thread::hardware_concurrency() - 1);
    }
    mWorkers.reserve(workerCount);
    for (u32 i = 0; i < workerCount; ++i) {
        mWorkers.emplace_back([this, i] { WorkerMain(i); });
    }
}

TaskSystem::~TaskSystem() {
    {
        std::lock_guard lock(mImpl->mutex);
        mImpl->stop = true;
    }
    mImpl->cv.notify_all();
    for (auto& worker : mWorkers) {
        worker.join();
    }
}

bool TaskSystem::IsWorkerThread() const { return tlsWorkerOwner == this; }

TaskHandle TaskSystem::Launch(std::string name, std::function<void()> fn, std::span<const TaskHandle> prerequisites,
                              TaskPriority priority) {
    auto task = std::make_shared<TaskState>();
    task->name = std::move(name);
    task->fn = std::move(fn);
    task->priority = priority;
    task->system = this;

    for (const TaskHandle& prerequisite : prerequisites) {
        if (!prerequisite.mState) {
            continue;
        }
        AIR_CHECK(prerequisite.mState->system == this, "prerequisite belongs to another TaskSystem");
        std::lock_guard lock(prerequisite.mState->mutex);
        if (!prerequisite.mState->completed) {
            task->pendingDependencies.fetch_add(1, std::memory_order_relaxed);
            prerequisite.mState->subsequents.push_back(task);
        }
    }

    if (task->pendingDependencies.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        Schedule(task);
    }
    return TaskHandle(std::move(task));
}

TaskHandle TaskSystem::WhenAll(std::span<const TaskHandle> prerequisites) {
    return Launch("WhenAll", [] {}, prerequisites, TaskPriority::High);
}

void TaskSystem::Schedule(std::shared_ptr<TaskState> task) {
    {
        std::lock_guard lock(mImpl->mutex);
        mImpl->queues[size_t(task->priority)].push_back(std::move(task));
    }
    // notify_all: helping waiters share the condition variable with idle workers.
    mImpl->cv.notify_all();
}

void TaskSystem::Execute(const std::shared_ptr<TaskState>& task) {
    try {
        if (task->fn) {
            task->fn();
        }
    } catch (const std::exception& e) {
        AIR_LOG(Error, "Task", "task '{}' threw: {}", task->name, e.what());
    } catch (...) {
        AIR_LOG(Error, "Task", "task '{}' threw an unknown exception", task->name);
    }
    task->fn = nullptr; // release captures as soon as possible

    std::vector<std::shared_ptr<TaskState>> subsequents;
    {
        std::lock_guard lock(task->mutex);
        task->completed = true;
        subsequents.swap(task->subsequents);
    }
    task->done.store(true, std::memory_order_seq_cst);

    if (mImpl->waiters.load(std::memory_order_seq_cst) > 0) {
        // Taking the lock orders this notify after any waiter's predicate check.
        std::lock_guard lock(mImpl->mutex);
        mImpl->cv.notify_all();
    }

    for (auto& subsequent : subsequents) {
        if (subsequent->pendingDependencies.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            Schedule(std::move(subsequent));
        }
    }
}

bool TaskSystem::TryExecuteOne() {
    std::shared_ptr<TaskState> task;
    {
        std::lock_guard lock(mImpl->mutex);
        task = mImpl->PopLocked();
    }
    if (!task) {
        return false;
    }
    Execute(task);
    return true;
}

void TaskSystem::HelpUntil(const TaskState& target) {
    while (!target.done.load(std::memory_order_acquire)) {
        if (TryExecuteOne()) {
            continue;
        }
        mImpl->waiters.fetch_add(1, std::memory_order_seq_cst);
        {
            std::unique_lock lock(mImpl->mutex);
            mImpl->cv.wait(lock, [&] { return target.done.load(std::memory_order_seq_cst) || mImpl->HasWorkLocked(); });
        }
        mImpl->waiters.fetch_sub(1, std::memory_order_seq_cst);
    }
}

void TaskSystem::WorkerMain(u32 index) {
    tlsWorkerOwner = this;
    for (;;) {
        std::shared_ptr<TaskState> task;
        {
            std::unique_lock lock(mImpl->mutex);
            mImpl->cv.wait(lock, [&] { return mImpl->stop || mImpl->HasWorkLocked(); });
            task = mImpl->PopLocked();
            if (!task && mImpl->stop) {
                break; // drain the queues before exiting
            }
        }
        if (task) {
            Execute(task);
        }
    }
    tlsWorkerOwner = nullptr;
}

void TaskSystem::ParallelFor(u32 count, const std::function<void(u32)>& fn, u32 minBatch) {
    if (count == 0) {
        return;
    }
    minBatch = std::max(1u, minBatch);
    const u32 batchCount = (count + minBatch - 1) / minBatch;
    const u32 helperCount = std::min(batchCount, GetWorkerCount() + 1) - 1;

    std::atomic<u32> next{0};
    auto body = [&] {
        for (;;) {
            const u32 begin = next.fetch_add(minBatch, std::memory_order_relaxed);
            if (begin >= count) {
                return;
            }
            const u32 end = std::min(count, begin + minBatch);
            for (u32 i = begin; i < end; ++i) {
                fn(i);
            }
        }
    };

    std::vector<TaskHandle> helpers;
    helpers.reserve(helperCount);
    for (u32 i = 0; i < helperCount; ++i) {
        helpers.push_back(Launch("ParallelFor", body, {}, TaskPriority::High));
    }
    body();
    for (const TaskHandle& helper : helpers) {
        helper.Wait();
    }
}

// ---------------------------------------------------------------------------
// Pipe
// ---------------------------------------------------------------------------

TaskHandle Pipe::Launch(std::string name, std::function<void()> fn, std::span<const TaskHandle> prerequisites,
                        TaskPriority priority) {
    std::lock_guard lock(mMutex);
    std::vector<TaskHandle> deps(prerequisites.begin(), prerequisites.end());
    deps.push_back(mLast);
    mLast = mSystem.Launch(mName + "/" + name, std::move(fn), deps, priority);
    return mLast;
}

void Pipe::WaitIdle() {
    TaskHandle last;
    {
        std::lock_guard lock(mMutex);
        last = mLast;
    }
    last.Wait();
}

} // namespace air
