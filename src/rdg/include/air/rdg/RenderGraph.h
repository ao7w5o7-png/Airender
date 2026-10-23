#pragma once

// Render graph (frame graph): passes declare resource usage, the graph schedules the rest.
//
//   RenderGraph graph(device, &tasks);
//   auto color = graph.CreateTexture({.format = Format::RGBA16_FLOAT, .width = w, .height = h,
//                                     .usage = TextureUsage::UnorderedAccess | TextureUsage::ShaderResource,
//                                     .debugName = "SceneColor"});
//   graph.AddPass("Shade", PassFlags::Compute,
//       [&](PassBuilder& b) { b.Write(color, Access::UAVCompute); },
//       [=](PassContext& ctx) { ctx.cmd.SetPipeline(pso); ...; ctx.cmd.Dispatch(...); });
//   graph.Execute();
//
// Passes declare every resource they touch and how; the graph then
//   1. culls passes whose results nobody consumes,
//   2. computes transient resource lifetimes and aliases them inside one placed heap,
//   3. derives every barrier (passes never write barriers by hand),
//   4. records passes in parallel on the task system and submits them in declaration order,
//   5. runs AsyncCompute passes on the compute queue, concurrently with graphics work they do not
//      depend on, and inserts the cross-queue waits and hand-over barriers they need.
//
// Threading: building the graph (Create*/Import*/AddPass/Execute) is single threaded and checked.
// Execute lambdas may run concurrently with each other on worker threads, so they must only use
// their PassContext and data captured by value (or immutable data).

#include "air/core/TaskSystem.h"
#include "air/rhi/RHIDevice.h"

#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace air::rdg
{

using rhi::Access;

enum class PassFlags : u32
{
    None = 0,
    Raster = 1 << 0,
    Compute = 1 << 1,
    RayTracing = 1 << 2,
    Copy = 1 << 3,
    NeverCull = 1 << 4, // keep even if no output is consumed (e.g. readback, debug capture)
    // Run on the compute queue, overlapping graphics passes it does not depend on. For Compute,
    // RayTracing or Copy passes whose accesses the compute queue supports (rhi::IsAccessLegalOnQueue).
    AsyncCompute = 1 << 5,
};
AIR_ENUM_FLAGS(PassFlags)

struct TextureRef
{
    u32 id = ~0u;
    bool IsValid() const { return id != ~0u; }
};
struct BufferRef
{
    u32 id = ~0u;
    bool IsValid() const { return id != ~0u; }
};

class RenderGraph;

class PassBuilder
{
public:
    // Read accesses must be read-only (SRV, CopySrc, ...). Write accesses must be writable (UAV, RT, ...).
    // Declaring a UAV write on a resource also covers reading it in the same pass.
    void Read(TextureRef texture, Access access);
    void Write(TextureRef texture, Access access);
    void Read(BufferRef buffer, Access access);
    void Write(BufferRef buffer, Access access);

private:
    friend class RenderGraph;
    PassBuilder(RenderGraph& graph, u32 pass) : mGraph(graph), mPass(pass) {}
    RenderGraph& mGraph;
    u32 mPass;
};

class PassContext
{
public:
    rhi::CommandList& cmd;
    rhi::Device& device;

    // Only resources declared in the pass setup are accessible.
    rhi::TextureHandle Get(TextureRef texture) const;
    rhi::BufferHandle Get(BufferRef buffer) const;
    rhi::BindlessIndex SRV(TextureRef texture) const { return device.GetBindlessSRV(Get(texture)); }
    rhi::BindlessIndex UAV(TextureRef texture) const { return device.GetBindlessUAV(Get(texture)); }
    rhi::BindlessIndex SRV(BufferRef buffer) const { return device.GetBindlessSRV(Get(buffer)); }
    rhi::BindlessIndex UAV(BufferRef buffer) const { return device.GetBindlessUAV(Get(buffer)); }

private:
    friend class RenderGraph;
    PassContext(rhi::CommandList& c, rhi::Device& d, const RenderGraph& g, u32 pass)
        : cmd(c), device(d), mGraph(g), mPass(pass)
    {
    }
    const RenderGraph& mGraph;
    u32 mPass;
};

class RenderGraph
{
public:
    struct Config
    {
        bool enableCulling = true;
        bool enableAliasing = true;
        bool parallelRecording = true;
        u32 passesPerCommandList = 4;
        bool enableAsyncCompute = true; // off: AsyncCompute passes run on the graphics queue
    };

    struct Stats
    {
        u32 passCount = 0;
        u32 culledPassCount = 0;
        u32 barrierCount = 0;
        u32 transientResourceCount = 0;
        u64 transientHeapSize = 0;       // with aliasing
        u64 transientBytesUnaliased = 0; // what it would cost without aliasing
        u32 commandListCount = 0;
        u32 asyncComputePassCount = 0; // live passes on the compute queue
        u32 submissionCount = 0;       // batches submitted (one per run of passes on a queue)
        u32 queueWaitCount = 0;        // GPU-side cross-queue waits
    };

    RenderGraph(rhi::Device& device, TaskSystem* tasks, Config config);
    RenderGraph(rhi::Device& device, TaskSystem* tasks = nullptr) : RenderGraph(device, tasks, Config{}) {}
    ~RenderGraph();

    RenderGraph(const RenderGraph&) = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;

    // Transient resources: created on first use, memory aliased, released after Execute().
    TextureRef CreateTexture(const rhi::TextureDesc& desc);
    BufferRef CreateBuffer(const rhi::BufferDesc& desc);

    // External resources: `current` is their state now, they are left in `final` after the graph.
    TextureRef ImportTexture(rhi::TextureHandle texture, Access current, Access final);
    BufferRef ImportBuffer(rhi::BufferHandle buffer, Access current, Access final);

    // Keeps a transient resource alive past Execute(). It gets its own
    // committed memory, ends in `final`, and *out receives the handle. The caller destroys it.
    void ExtractTexture(TextureRef texture, Access final, rhi::TextureHandle* out);
    void ExtractBuffer(BufferRef buffer, Access final, rhi::BufferHandle* out);

    template <typename SetupFn, typename ExecuteFn>
    void AddPass(std::string name, PassFlags flags, SetupFn&& setup, ExecuteFn&& execute)
    {
        const u32 pass = AddPassInternal(std::move(name), flags, std::function<void(PassContext&)>(execute));
        PassBuilder builder(*this, pass);
        setup(builder);
    }

    void Compile();
    // Compiles if needed, allocates transient memory, records and submits. Returns the fence value.
    u64 Execute();

    const Stats& GetStats() const { return mStats; }

    // CPU time each live pass spent recording (its execute lambda), valid after Execute().
    struct PassTiming
    {
        std::string name;
        double cpuMs = 0.0;
    };
    std::vector<PassTiming> GetPassTimings() const;
    // Human readable dump of the compiled graph: passes, accesses, barriers, lifetimes, heap offsets.
    std::string DumpText() const;
    // Graphviz dot of passes and resources (culled passes are grey).
    std::string DumpGraphviz() const;

private:
    friend class PassBuilder;
    friend class PassContext;

    enum class ResourceKind : u8
    {
        Texture,
        Buffer
    };

    struct Resource
    {
        ResourceKind kind = ResourceKind::Texture;
        std::string name;
        rhi::TextureDesc textureDesc;
        rhi::BufferDesc bufferDesc;
        bool imported = false;
        bool extracted = false;
        Access initialAccess = Access::None;
        Access finalAccess = Access::None;
        rhi::TextureHandle* extractTextureOut = nullptr;
        rhi::BufferHandle* extractBufferOut = nullptr;

        // Compile results
        bool used = false;
        u32 firstPass = ~0u;
        u32 lastPass = 0;
        rhi::AllocationInfo allocation;
        u64 heapOffset = ~0ull;
        // Used by a compute-queue pass: it may run alongside any graphics pass, so its memory is
        // never shared with another transient.
        bool asyncUse = false;

        // Execute results
        rhi::TextureHandle texture;
        rhi::BufferHandle buffer;

        bool IsTransient() const { return !imported && !extracted; }
    };

    struct ResourceAccess
    {
        u32 resource;
        Access access;
    };

    struct Pass
    {
        std::string name;
        PassFlags flags = PassFlags::None;
        std::function<void(PassContext&)> execute;
        std::vector<ResourceAccess> accesses; // merged: one entry per resource
        bool culled = false;
        std::vector<rhi::Barrier> barriersBefore;
        std::vector<u32> barrierResources; // resource id of each barrier
        double cpuMs = 0.0;                // recording time, written by the recording thread
        rhi::QueueType queue = rhi::QueueType::Graphics;
        u32 batch = 0;
    };

    // A run of consecutive live passes on one queue: one submission. Batch 0 is an (often empty)
    // graphics prologue: the compute queue starts the frame there (the fork), and it holds transitions
    // the compute queue cannot do before its first batch.
    struct Batch
    {
        rhi::QueueType queue = rhi::QueueType::Graphics;
        u32 begin = 0; // range in mLivePasses
        u32 end = 0;
        // The latest batch of each other queue this one waits for, ~0u for none.
        u32 waitFor[size_t(rhi::QueueType::Count)] = {~0u, ~0u, ~0u};
        // Transitions made at the end of this batch on behalf of a later batch on another queue.
        std::vector<rhi::Barrier> barriersAfter;
        std::vector<u32> barriersAfterResources;
        bool finalBarriers = false; // the graph's final barriers are recorded at the end of this batch
        bool fork = false;          // the prologue is submitted, even empty: the compute queue waits for it
    };

    u32 AddPassInternal(std::string name, PassFlags flags, std::function<void(PassContext&)> execute);
    void AddAccess(u32 pass, u32 resource, ResourceKind kind, Access access, bool write);
    void CheckThread() const;
    rhi::Barrier MakeBarrier(const Resource& resource, Access before, Access after, bool discard) const;

    void CullPasses();
    void BuildBatches();
    void ComputeLifetimes();
    void AllocateTransientMemory();
    void BuildBarriers();
    void RecordRange(rhi::CommandList& cmd, u32 begin, u32 end);

    const Resource& ResourceFor(u32 pass, u32 id, ResourceKind kind) const;

    rhi::Device& mDevice;
    TaskSystem* mTasks;
    Config mConfig;
    std::thread::id mOwnerThread;

    std::vector<Resource> mResources;
    std::vector<Pass> mPasses;
    std::vector<u32> mLivePasses; // non-culled pass indices in order
    std::vector<Batch> mBatches;
    std::vector<rhi::Barrier> mFinalBarriers;
    std::vector<u32> mFinalBarrierResources;
    rhi::HeapHandle mTransientHeap;
    bool mCompiled = false;
    bool mExecuted = false;
    Stats mStats;
};

} // namespace air::rdg
