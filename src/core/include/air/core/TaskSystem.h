#pragma once

// Task system:
//   - Launch() schedules a task that runs once all of its prerequisites have completed.
//   - Tasks form a DAG; waiting on a task from any thread (including a worker) helps execute
//     other tasks instead of blocking, so nested waits cannot deadlock the pool.
//   - Pipe serialises the tasks launched through it. Use a pipe to own
//     non-thread-safe state instead of a mutex: everything that touches the state runs in the pipe.
//
// Thread-safety: every public function of TaskSystem, TaskHandle and Pipe may be called from any thread.

#include "air/core/Types.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace air
{

enum class TaskPriority : u8
{
    High = 0,
    Normal = 1,
    Background = 2,
    Count = 3
};

class TaskSystem;
struct TaskState;

class TaskHandle
{
public:
    TaskHandle() = default;

    bool IsValid() const { return mState != nullptr; }
    bool IsCompleted() const;
    // Blocks until completion while executing other ready tasks on this thread.
    void Wait() const;

private:
    friend class TaskSystem;
    explicit TaskHandle(std::shared_ptr<TaskState> state) : mState(std::move(state)) {}
    std::shared_ptr<TaskState> mState;
};

class TaskSystem
{
public:
    // workerCount == 0 picks hardware_concurrency() - 1 (at least 1).
    explicit TaskSystem(u32 workerCount = 0);
    ~TaskSystem();

    TaskSystem(const TaskSystem&) = delete;
    TaskSystem& operator=(const TaskSystem&) = delete;

    TaskHandle Launch(std::string name, std::function<void()> fn, std::span<const TaskHandle> prerequisites = {},
                      TaskPriority priority = TaskPriority::Normal);

    // A task that completes when all prerequisites complete (a join).
    TaskHandle WhenAll(std::span<const TaskHandle> prerequisites);

    // Runs fn(index) for index in [0, count), splitting into batches of at least minBatch.
    // The calling thread participates. Returns after every index has been processed.
    void ParallelFor(u32 count, const std::function<void(u32)>& fn, u32 minBatch = 1);

    u32 GetWorkerCount() const { return u32(mWorkers.size()); }
    // True when called from one of this system's worker threads.
    bool IsWorkerThread() const;

private:
    friend class TaskHandle;
    friend class Pipe;

    void Schedule(std::shared_ptr<TaskState> task);
    void Execute(const std::shared_ptr<TaskState>& task);
    bool TryExecuteOne();
    void WorkerMain(u32 index);
    void HelpUntil(const TaskState& task);

    struct Impl;
    std::unique_ptr<Impl> mImpl;
    std::vector<std::thread> mWorkers;
};

// Executes tasks launched through it one at a time, in launch order.
class Pipe
{
public:
    Pipe(TaskSystem& system, std::string name) : mSystem(system), mName(std::move(name)) {}

    TaskHandle Launch(const std::string& name, std::function<void()> fn, std::span<const TaskHandle> prerequisites = {},
                      TaskPriority priority = TaskPriority::Normal);
    // Waits for every task launched so far.
    void WaitIdle();

private:
    TaskSystem& mSystem;
    std::string mName;
    std::mutex mMutex;
    TaskHandle mLast;
};

} // namespace air
