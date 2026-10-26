#include "air/rdg/RenderGraph.h"

#include "air/rdg/TransientResourcePool.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/core/Profiler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <memory>
#include <sstream>

namespace air::rdg
{

using rhi::Barrier;
using rhi::BufferUsage;
using rhi::TextureUsage;

namespace
{

// Which usage flag an access needs. Catches "forgot UnorderedAccess usage" at AddPass time with the
// pass name in the message, instead of as a device removal later.
bool TextureSupports(const rhi::TextureDesc& desc, Access access)
{
    auto need = [&](Access mask, TextureUsage usage) { return !Any(access & mask) || Any(desc.usage & usage); };
    return need(Access::SRVGraphics | Access::SRVCompute, TextureUsage::ShaderResource) &&
           need(Access::UAVMask, TextureUsage::UnorderedAccess) &&
           need(Access::RenderTarget, TextureUsage::RenderTarget) &&
           need(Access::DepthRead | Access::DepthWrite, TextureUsage::DepthStencil) &&
           need(Access::CopySrc, TextureUsage::CopySrc) && need(Access::CopyDst, TextureUsage::CopyDst);
}

bool BufferSupports(const rhi::BufferDesc& desc, Access access)
{
    auto need = [&](Access mask, BufferUsage usage) { return !Any(access & mask) || Any(desc.usage & usage); };
    return need(Access::SRVGraphics | Access::SRVCompute, BufferUsage::ShaderResource) &&
           need(Access::UAVMask, BufferUsage::UnorderedAccess) && need(Access::Constant, BufferUsage::Constant) &&
           need(Access::IndirectArgs, BufferUsage::Indirect) && need(Access::CopySrc, BufferUsage::CopySrc) &&
           need(Access::CopyDst, BufferUsage::CopyDst);
}

std::string FlagsToString(PassFlags flags)
{
    std::string s;
    auto add = [&](PassFlags f, const char* name)
    {
        if (Any(flags & f))
        {
            s += s.empty() ? name : std::string("|") + name;
        }
    };
    add(PassFlags::Raster, "Raster");
    add(PassFlags::Compute, "Compute");
    add(PassFlags::RayTracing, "RayTracing");
    add(PassFlags::Copy, "Copy");
    add(PassFlags::NeverCull, "NeverCull");
    add(PassFlags::AsyncCompute, "AsyncCompute");
    return s.empty() ? "None" : s;
}

} // namespace

// ===========================================================================
// PassBuilder / PassContext
// ===========================================================================

void PassBuilder::Read(TextureRef t, Access a)
{
    mGraph.AddAccess(mPass, t.id, RenderGraph::ResourceKind::Texture, a, false);
}
void PassBuilder::Write(TextureRef t, Access a)
{
    mGraph.AddAccess(mPass, t.id, RenderGraph::ResourceKind::Texture, a, true);
}
void PassBuilder::Read(BufferRef b, Access a)
{
    mGraph.AddAccess(mPass, b.id, RenderGraph::ResourceKind::Buffer, a, false);
}
void PassBuilder::Write(BufferRef b, Access a)
{
    mGraph.AddAccess(mPass, b.id, RenderGraph::ResourceKind::Buffer, a, true);
}

rhi::TextureHandle PassContext::Get(TextureRef texture) const
{
    return mGraph.ResourceFor(mPass, texture.id, RenderGraph::ResourceKind::Texture).texture;
}
rhi::BufferHandle PassContext::Get(BufferRef buffer) const
{
    return mGraph.ResourceFor(mPass, buffer.id, RenderGraph::ResourceKind::Buffer).buffer;
}

// ===========================================================================
// RenderGraph: building
// ===========================================================================

RenderGraph::RenderGraph(rhi::Device& device, TaskSystem* tasks, Config config, TransientResourcePool* pool)
    : mDevice(device), mTasks(tasks), mConfig(config), mOwnerThread(std::this_thread::get_id()), mPool(pool)
{
    mConfig.passesPerCommandList = std::max(1u, mConfig.passesPerCommandList);
}

RenderGraph::~RenderGraph()
{
    if (mCompiled && !mExecuted)
    {
        AIR_LOG(Warning, "RDG", "render graph compiled but never executed ({} passes)", mPasses.size());
    }
}

void RenderGraph::CheckThread() const
{
    AIR_CHECK(std::this_thread::get_id() == mOwnerThread,
              "RenderGraph is built on one thread; record parallel work inside pass lambdas instead");
}

TextureRef RenderGraph::CreateTexture(const rhi::TextureDesc& desc)
{
    CheckThread();
    AIR_CHECK(!mCompiled, "CreateTexture after Compile()");
    AIR_CHECK(desc.format != rhi::Format::Unknown, "texture '{}' has no format", desc.debugName);
    Resource r;
    r.kind = ResourceKind::Texture;
    r.name = desc.debugName.empty() ? std::format("Texture{}", mResources.size()) : desc.debugName;
    r.textureDesc = desc;
    r.textureDesc.debugName = r.name;
    mResources.push_back(std::move(r));
    return {u32(mResources.size() - 1)};
}

BufferRef RenderGraph::CreateBuffer(const rhi::BufferDesc& desc)
{
    CheckThread();
    AIR_CHECK(!mCompiled, "CreateBuffer after Compile()");
    AIR_CHECK(desc.size > 0, "buffer '{}' has zero size", desc.debugName);
    AIR_CHECK(desc.memory == rhi::MemoryLocation::DeviceLocal, "transient buffer '{}' must be DeviceLocal",
              desc.debugName);
    Resource r;
    r.kind = ResourceKind::Buffer;
    r.name = desc.debugName.empty() ? std::format("Buffer{}", mResources.size()) : desc.debugName;
    r.bufferDesc = desc;
    r.bufferDesc.debugName = r.name;
    mResources.push_back(std::move(r));
    return {u32(mResources.size() - 1)};
}

TextureRef RenderGraph::ImportTexture(rhi::TextureHandle texture, Access current, Access final)
{
    CheckThread();
    const rhi::TextureDesc* desc = mDevice.GetTextureDesc(texture);
    AIR_CHECK(desc, "ImportTexture: invalid handle");
    Resource r;
    r.kind = ResourceKind::Texture;
    r.name = desc->debugName;
    r.textureDesc = *desc;
    r.imported = true;
    r.initialAccess = current;
    r.finalAccess = final;
    r.texture = texture;
    mResources.push_back(std::move(r));
    return {u32(mResources.size() - 1)};
}

BufferRef RenderGraph::ImportBuffer(rhi::BufferHandle buffer, Access current, Access final)
{
    CheckThread();
    const rhi::BufferDesc* desc = mDevice.GetBufferDesc(buffer);
    AIR_CHECK(desc, "ImportBuffer: invalid handle");
    Resource r;
    r.kind = ResourceKind::Buffer;
    r.name = desc->debugName;
    r.bufferDesc = *desc;
    r.imported = true;
    r.initialAccess = current;
    r.finalAccess = final;
    r.buffer = buffer;
    mResources.push_back(std::move(r));
    return {u32(mResources.size() - 1)};
}

void RenderGraph::ExtractTexture(TextureRef texture, Access final, rhi::TextureHandle* out)
{
    CheckThread();
    AIR_CHECK(texture.id < mResources.size() && mResources[texture.id].kind == ResourceKind::Texture,
              "ExtractTexture: invalid ref");
    Resource& r = mResources[texture.id];
    AIR_CHECK(!r.imported, "ExtractTexture on imported texture '{}'", r.name);
    r.extracted = true;
    r.finalAccess = final;
    r.extractTextureOut = out;
}

void RenderGraph::ExtractBuffer(BufferRef buffer, Access final, rhi::BufferHandle* out)
{
    CheckThread();
    AIR_CHECK(buffer.id < mResources.size() && mResources[buffer.id].kind == ResourceKind::Buffer,
              "ExtractBuffer: invalid ref");
    Resource& r = mResources[buffer.id];
    AIR_CHECK(!r.imported, "ExtractBuffer on imported buffer '{}'", r.name);
    r.extracted = true;
    r.finalAccess = final;
    r.extractBufferOut = out;
}

u32 RenderGraph::AddPassInternal(std::string name, PassFlags flags, std::function<void(PassContext&)> execute)
{
    CheckThread();
    AIR_CHECK(!mCompiled, "AddPass('{}') after Compile()", name);
    AIR_CHECK(!Any(flags & PassFlags::AsyncCompute) || !Any(flags & PassFlags::Raster),
              "pass '{}': AsyncCompute cannot be a raster pass (the compute queue does not rasterize)", name);
    Pass pass;
    pass.name = std::move(name);
    pass.flags = flags;
    pass.execute = std::move(execute);
    mPasses.push_back(std::move(pass));
    return u32(mPasses.size() - 1);
}

void RenderGraph::AddAccess(u32 passIndex, u32 id, ResourceKind kind, Access access, bool write)
{
    Pass& pass = mPasses[passIndex];
    AIR_CHECK(id < mResources.size() && mResources[id].kind == kind, "pass '{}' uses an invalid resource ref",
              pass.name);
    const Resource& r = mResources[id];
    AIR_CHECK(access != Access::None, "pass '{}' declares Access::None on '{}'", pass.name, r.name);
    if (write)
    {
        AIR_CHECK(IsWriteAccess(access), "pass '{}' declares Write on '{}' with read-only access {}", pass.name, r.name,
                  rhi::ToString(access));
    }
    else
    {
        AIR_CHECK(!IsWriteAccess(access), "pass '{}' declares Read on '{}' with writable access {}", pass.name, r.name,
                  rhi::ToString(access));
    }
    AIR_CHECK(!Any(pass.flags & PassFlags::AsyncCompute) || IsAccessLegalOnQueue(access, rhi::QueueType::Compute),
              "pass '{}' is AsyncCompute but uses '{}' as {}, which the compute queue does not support", pass.name,
              r.name, rhi::ToString(access));
    const bool supported =
        kind == ResourceKind::Texture ? TextureSupports(r.textureDesc, access) : BufferSupports(r.bufferDesc, access);
    AIR_CHECK(supported, "pass '{}': '{}' is not created with the usage flags needed for {}", pass.name, r.name,
              rhi::ToString(access));

    for (ResourceAccess& existing : pass.accesses)
    {
        if (existing.resource != id)
        {
            continue;
        }
        if (existing.access == access)
        {
            return;
        }
        // Different read-only states combine (e.g. SRVCompute|CopySrc); a write state cannot be mixed.
        AIR_CHECK(!IsWriteAccess(existing.access) && !write,
                  "pass '{}' accesses '{}' as both {} and {}; split the pass", pass.name, r.name,
                  rhi::ToString(existing.access), rhi::ToString(access));
        existing.access = existing.access | access;
        return;
    }
    pass.accesses.push_back({id, access});
}

const RenderGraph::Resource& RenderGraph::ResourceFor(u32 passIndex, u32 id, ResourceKind kind) const
{
    const Pass& pass = mPasses[passIndex];
    const bool declared = std::any_of(pass.accesses.begin(), pass.accesses.end(),
                                      [&](const ResourceAccess& a) { return a.resource == id; });
    AIR_CHECK(declared && mResources[id].kind == kind, "pass '{}' uses '{}' without declaring it in setup", pass.name,
              id < mResources.size() ? mResources[id].name : "<invalid>");
    return mResources[id];
}

// ===========================================================================
// RenderGraph: compile
// ===========================================================================

void RenderGraph::Compile()
{
    AIR_PROFILE_SCOPE("RDG.Compile");
    CheckThread();
    if (mCompiled)
    {
        return;
    }
    mCompiled = true;
    mStats = {};
    mStats.passCount = u32(mPasses.size());

    CullPasses();
    BuildBatches();
    ComputeLifetimes();
    BuildBarriers(); // the cross-queue waits order uses, which decides what may share memory
    ComputeReach();
    AllocateTransientMemory();
}

void RenderGraph::CullPasses()
{
    std::vector<bool> needed(mResources.size(), false);
    for (u32 i = 0; i < mResources.size(); ++i)
    {
        needed[i] = mResources[i].imported || mResources[i].extracted;
    }
    // Walk backwards: a pass survives if it is NeverCull or writes something a surviving pass (or
    // the outside world) needs. Its inputs then become needed too.
    for (u32 p = u32(mPasses.size()); p-- > 0;)
    {
        Pass& pass = mPasses[p];
        bool keep = !mConfig.enableCulling || Any(pass.flags & PassFlags::NeverCull);
        for (const ResourceAccess& a : pass.accesses)
        {
            keep = keep || (IsWriteAccess(a.access) && needed[a.resource]);
        }
        pass.culled = !keep;
        if (keep)
        {
            for (const ResourceAccess& a : pass.accesses)
            {
                needed[a.resource] = true;
            }
        }
    }
    mLivePasses.clear();
    for (u32 p = 0; p < mPasses.size(); ++p)
    {
        if (!mPasses[p].culled)
        {
            mLivePasses.push_back(p);
        }
        else
        {
            ++mStats.culledPassCount;
        }
    }
}

void RenderGraph::BuildBatches()
{
    mBatches.clear();
    mBatches.push_back({}); // the graphics prologue
    for (u32 live = 0; live < mLivePasses.size(); ++live)
    {
        Pass& pass = mPasses[mLivePasses[live]];
        const bool async = mConfig.enableAsyncCompute && Any(pass.flags & PassFlags::AsyncCompute);
        pass.queue = async ? rhi::QueueType::Compute : rhi::QueueType::Graphics;
        mStats.asyncComputePassCount += async ? 1u : 0u;
        if (mBatches.size() == 1 || mBatches.back().queue != pass.queue)
        {
            Batch batch;
            batch.queue = pass.queue;
            batch.begin = live;
            batch.end = live;
            mBatches.push_back(std::move(batch));
        }
        pass.batch = u32(mBatches.size() - 1);
        mBatches.back().end = live + 1;
    }
}

void RenderGraph::ComputeLifetimes()
{
    for (u32 p : mLivePasses)
    {
        const size_t queue = size_t(mPasses[p].queue);
        for (const ResourceAccess& a : mPasses[p].accesses)
        {
            Resource& r = mResources[a.resource];
            r.used = true;
            r.firstPass = std::min(r.firstPass, p);
            r.lastPass = std::max(r.lastPass, p);
            r.firstUse[queue] = std::min(r.firstUse[queue], p);
            r.lastUse[queue] = r.lastUse[queue] == ~0u ? p : std::max(r.lastUse[queue], p);
            r.asyncUse = r.asyncUse || mPasses[p].queue != rhi::QueueType::Graphics;
        }
    }
}

void RenderGraph::ComputeReach()
{
    constexpr u32 kNone = ~0u;
    auto merge = [](std::array<u32, size_t(rhi::QueueType::Count)>& into,
                    const std::array<u32, size_t(rhi::QueueType::Count)>& from)
    {
        for (size_t q = 0; q < into.size(); ++q)
        {
            if (from[q] != kNone && (into[q] == kNone || from[q] > into[q]))
            {
                into[q] = from[q];
            }
        }
    };
    mReach.assign(mBatches.size(), {kNone, kNone, kNone});
    std::array<u32, size_t(rhi::QueueType::Count)> previous = {kNone, kNone, kNone}; // last batch of each queue
    for (u32 b = 0; b < mBatches.size(); ++b)
    {
        const Batch& batch = mBatches[b];
        const size_t queue = size_t(batch.queue);
        if (previous[queue] != kNone)
        {
            merge(mReach[b], mReach[previous[queue]]); // a queue runs its batches in order
        }
        for (const u32 wait : batch.waitFor)
        {
            if (wait != kNone && mBatches[wait].IsSubmitted())
            {
                merge(mReach[b], mReach[wait]);
            }
        }
        mReach[b][queue] = b;
        previous[queue] = b;
    }
}

bool RenderGraph::Precedes(u32 a, u32 b) const
{
    const Resource& x = mResources[a];
    const Resource& y = mResources[b];
    for (size_t qx = 0; qx < size_t(rhi::QueueType::Count); ++qx)
    {
        if (x.lastUse[qx] == ~0u)
        {
            continue;
        }
        const u32 lastBatch = mPasses[x.lastUse[qx]].batch;
        for (size_t qy = 0; qy < size_t(rhi::QueueType::Count); ++qy)
        {
            if (y.firstUse[qy] == ~0u)
            {
                continue;
            }
            // One queue runs its passes in order; across queues, y's first pass must start after a
            // wait that covers x's last batch on that queue.
            const u32 reached = mReach[mPasses[y.firstUse[qy]].batch][qx];
            const bool ordered = qx == qy ? x.lastUse[qx] < y.firstUse[qy] : reached != ~0u && reached >= lastBatch;
            if (!ordered)
            {
                return false;
            }
        }
    }
    return true;
}

void RenderGraph::AllocateTransientMemory()
{
    struct Placed
    {
        u32 resource;
        u64 begin, end;
    };
    std::vector<u32> order;
    for (u32 i = 0; i < mResources.size(); ++i)
    {
        Resource& r = mResources[i];
        if (!r.used || !r.IsTransient())
        {
            continue;
        }
        r.allocation = r.kind == ResourceKind::Texture ? mDevice.GetTextureAllocationInfo(r.textureDesc)
                                                       : mDevice.GetBufferAllocationInfo(r.bufferDesc);
        mStats.transientBytesUnaliased += r.allocation.size;
        ++mStats.transientResourceCount;
        order.push_back(i);
    }
    // Big resources first packs better; ties broken by first use for determinism.
    std::stable_sort(order.begin(), order.end(),
                     [&](u32 a, u32 b) { return mResources[a].allocation.size > mResources[b].allocation.size; });

    std::vector<Placed> placed;
    u64 heapSize = 0;
    for (u32 index : order)
    {
        Resource& r = mResources[index];
        u64 offset = 0;
        if (mConfig.enableAliasing)
        {
            // First fit: bump past every placed resource that is alive at the same time and overlaps.
            bool moved = true;
            while (moved)
            {
                moved = false;
                for (const Placed& p : placed)
                {
                    // Sharing memory needs every use of one to end before any use of the other
                    // starts: pass order on one queue, the cross-queue waits between queues.
                    const bool timeOverlap = !Precedes(p.resource, index) && !Precedes(index, p.resource);
                    const bool memOverlap = p.begin < offset + r.allocation.size && offset < p.end;
                    if (timeOverlap && memOverlap)
                    {
                        offset = AlignUp(p.end, r.allocation.alignment);
                        moved = true;
                    }
                }
            }
        }
        else
        {
            offset = AlignUp(heapSize, r.allocation.alignment);
        }
        r.heapOffset = offset;
        placed.push_back({index, offset, offset + r.allocation.size});
        heapSize = std::max(heapSize, offset + r.allocation.size);
    }
    mStats.transientHeapSize = heapSize;
}

Barrier RenderGraph::MakeBarrier(const Resource& r, Access before, Access after, bool discard) const
{
    Barrier b;
    b.type = r.kind == ResourceKind::Texture ? Barrier::Type::Texture : Barrier::Type::Buffer;
    b.before = before;
    b.after = after;
    b.discard = discard;
    return b;
}

void RenderGraph::BuildBarriers()
{
    std::vector<Access> state(mResources.size());
    std::vector<bool> touched(mResources.size(), false);
    for (u32 i = 0; i < mResources.size(); ++i)
    {
        state[i] = mResources[i].imported ? mResources[i].initialAccess : Access::None;
    }

    // Handles are only known at Execute(); remember which resource each barrier is for.
    auto add = [](std::vector<Barrier>& barriers, std::vector<u32>& resources, Barrier b, u32 resource)
    {
        barriers.push_back(b);
        resources.push_back(resource);
    };

    // Per resource, the batch of its last write or change of state (any queue; 0 is the prologue, ~0u
    // none yet), and the last batch of every queue that read it since then.
    constexpr size_t kQueues = size_t(rhi::QueueType::Count);
    std::vector<u32> lastChange(mResources.size(), ~0u);
    std::vector<std::array<u32, kQueues>> readsSince(mResources.size());
    for (std::array<u32, kQueues>& reads : readsSince)
    {
        reads.fill(~0u);
    }
    auto waitFor = [&](u32 batch, u32 other)
    {
        u32& wait = mBatches[batch].waitFor[size_t(mBatches[other].queue)];
        wait = wait == ~0u ? other : std::max(wait, other);
    };

    for (u32 p : mLivePasses)
    {
        Pass& pass = mPasses[p];
        pass.barriersBefore.clear();
        pass.barrierResources.clear();
        for (const ResourceAccess& a : pass.accesses)
        {
            const Resource& r = mResources[a.resource];
            const Access before = state[a.resource];
            // A use waits for the last write or change of state on another queue (read after write);
            // a write or a change of state also for every read since then on the other queues (write
            // after read). Reads that leave the state as it is run side by side on both queues.
            // (An imported resource already in the state needs no barrier at its first use either.)
            const bool read = (touched[a.resource] || r.imported) && before == a.access && !IsWriteAccess(a.access);
            const u32 changed = lastChange[a.resource];
            if (changed != ~0u && mBatches[changed].queue != pass.queue)
            {
                waitFor(pass.batch, changed);
            }
            if (!read)
            {
                for (size_t q = 0; q < kQueues; ++q)
                {
                    const u32 reader = readsSince[a.resource][q];
                    if (reader != ~0u && q != size_t(pass.queue))
                    {
                        waitFor(pass.batch, reader);
                    }
                }
            }
            // The batch that hands a graphics-only state over: the last use there.
            u32 previous = changed;
            for (size_t q = 0; q < kQueues; ++q)
            {
                const u32 reader = readsSince[a.resource][q];
                previous = reader != ~0u && (previous == ~0u || reader > previous) ? reader : previous;
            }
            if (!touched[a.resource] && !r.imported)
            {
                // First use of a graph-created resource: contents are undefined (and possibly aliased).
                add(pass.barriersBefore, pass.barrierResources, MakeBarrier(r, Access::None, a.access, true),
                    a.resource);
            }
            else if (before != a.access || Any(a.access & Access::UAVMask))
            {
                const Barrier barrier = MakeBarrier(r, before, a.access, false);
                if (IsAccessLegalOnQueue(before, pass.queue))
                {
                    // A state change, or UAV -> UAV to order the writes of consecutive passes.
                    add(pass.barriersBefore, pass.barrierResources, barrier, a.resource);
                }
                else
                {
                    // This queue cannot leave a graphics-only state: the graphics queue transitions the
                    // resource after its last use there (or in the prologue), and this batch waits.
                    const u32 owner = previous != ~0u ? previous : 0u;
                    AIR_CHECK(mBatches[owner].queue == rhi::QueueType::Graphics,
                              "'{}' in {} on the {} queue cannot be handed to pass '{}'", r.name, rhi::ToString(before),
                              rhi::ToString(mBatches[owner].queue), pass.name);
                    add(mBatches[owner].barriersAfter, mBatches[owner].barriersAfterResources, barrier, a.resource);
                    waitFor(pass.batch, owner);
                    ++mStats.barrierCount;
                }
            }
            touched[a.resource] = true;
            state[a.resource] = a.access;
            if (read)
            {
                u32& reader = readsSince[a.resource][size_t(pass.queue)];
                reader = reader == ~0u ? pass.batch : std::max(reader, pass.batch);
            }
            else
            {
                lastChange[a.resource] = pass.batch;
                readsSince[a.resource].fill(~0u);
            }
        }
        mStats.barrierCount += u32(pass.barriersBefore.size());
    }
    for (u32 i = 0; i < mResources.size(); ++i)
    {
        mResources[i].lastAccess = state[i]; // the state a pooled transient's resource is left in
    }

    // The compute queue starts the frame at the graphics prologue (the fork). Without it, a compute
    // batch with no graphics input runs as soon as it is submitted, while the graphics queue may still
    // be on the previous frame and using the imported resources that batch writes.
    for (u32 b = 1; b < mBatches.size(); ++b)
    {
        if (mBatches[b].queue == rhi::QueueType::Compute)
        {
            waitFor(b, 0);
            mBatches[0].fork = mBatches[b].waitFor[size_t(rhi::QueueType::Graphics)] == 0;
            break;
        }
    }

    // The frame ends on the graphics queue, after all compute work: exported resources leave in their
    // final states (some graphics-only) and the next frame starts from a joined state.
    u32 lastCompute = ~0u;
    for (u32 b = 0; b < mBatches.size(); ++b)
    {
        lastCompute = mBatches[b].queue == rhi::QueueType::Compute ? b : lastCompute;
    }
    Batch* holder = &mBatches.back();
    if (lastCompute != ~0u &&
        (holder->queue != rhi::QueueType::Graphics || holder->waitFor[size_t(rhi::QueueType::Compute)] != lastCompute))
    {
        Batch epilogue;
        epilogue.begin = epilogue.end = u32(mLivePasses.size());
        epilogue.waitFor[size_t(rhi::QueueType::Compute)] = lastCompute;
        mBatches.push_back(std::move(epilogue));
        holder = &mBatches.back();
    }
    holder->finalBarriers = true;

    mFinalBarriers.clear();
    mFinalBarrierResources.clear();
    for (u32 i = 0; i < mResources.size(); ++i)
    {
        const Resource& r = mResources[i];
        const bool exported = r.imported || (r.extracted && r.used);
        if (exported && state[i] != r.finalAccess)
        {
            add(mFinalBarriers, mFinalBarrierResources, MakeBarrier(r, state[i], r.finalAccess, false), i);
        }
    }
    // A pooled transient comes back next frame, first used on the same queue. When that queue cannot
    // leave the state this frame ends in (compute after a pixel-shader read), end in common instead,
    // or the pool could never hand the resource back.
    if (mPool)
    {
        for (u32 i = 0; i < mResources.size(); ++i)
        {
            Resource& r = mResources[i];
            if (r.used && r.IsTransient() && !IsAccessLegalOnQueue(state[i], mPasses[r.firstPass].queue))
            {
                add(mFinalBarriers, mFinalBarrierResources, MakeBarrier(r, state[i], Access::None, false), i);
                r.lastAccess = Access::None;
            }
        }
    }
    mStats.barrierCount += u32(mFinalBarriers.size());
}

// ===========================================================================
// RenderGraph: execute
// ===========================================================================

void RenderGraph::RecordRange(rhi::CommandList& cmd, u32 begin, u32 end)
{
    for (u32 i = begin; i < end; ++i)
    {
        const u32 passIndex = mLivePasses[i];
        Pass& pass = mPasses[passIndex];
        cmd.Barriers(pass.barriersBefore);
        cmd.BeginEvent(pass.name);
        PassContext context(cmd, mDevice, *this, passIndex);
        if (pass.execute)
        {
            AIR_PROFILE_SCOPE(pass.name);
            const auto start = std::chrono::steady_clock::now();
            pass.execute(context);
            pass.cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        cmd.EndEvent();
    }
}

std::vector<RenderGraph::PassTiming> RenderGraph::GetPassTimings() const
{
    std::vector<PassTiming> timings;
    timings.reserve(mLivePasses.size());
    for (u32 p : mLivePasses)
    {
        timings.push_back({mPasses[p].name, mPasses[p].cpuMs});
    }
    return timings;
}

u64 RenderGraph::Execute()
{
    AIR_PROFILE_SCOPE("RDG.Execute");
    CheckThread();
    AIR_CHECK(!mExecuted, "RenderGraph::Execute called twice");
    Compile();
    mExecuted = true;

    // 1. Memory: one placed heap for all transient resources (pooled across frames with a
    //    TransientResourcePool), committed memory for extracted ones.
    if (mStats.transientHeapSize > 0)
    {
        if (mPool)
        {
            bool created = false;
            mPoolSlot = mPool->AcquireHeap(mStats.transientHeapSize, &created);
            mTransientHeap = mPool->GetHeap(mPoolSlot);
            mStats.transientHeapsCreated += created ? 1u : 0u;
        }
        else
        {
            mTransientHeap =
                mDevice.CreateHeap(mStats.transientHeapSize, rhi::MemoryLocation::DeviceLocal, "RDG Transient");
            AIR_CHECK(mTransientHeap.IsValid(), "failed to allocate {} byte transient heap", mStats.transientHeapSize);
            ++mStats.transientHeapsCreated;
        }
    }
    for (Resource& r : mResources)
    {
        if (!r.used || r.imported)
        {
            continue;
        }
        if (r.extracted)
        {
            if (r.kind == ResourceKind::Texture)
            {
                r.texture = mDevice.CreateTexture(r.textureDesc);
            }
            else
            {
                r.buffer = mDevice.CreateBuffer(r.bufferDesc);
            }
        }
        else if (mPool)
        {
            const rhi::QueueType firstQueue = mPasses[r.firstPass].queue;
            const TransientResourcePool::Placement placement =
                r.kind == ResourceKind::Texture
                    ? mPool->AcquireTexture(mPoolSlot, r.heapOffset, r.textureDesc, firstQueue)
                    : mPool->AcquireBuffer(mPoolSlot, r.heapOffset, r.bufferDesc, firstQueue);
            r.texture = placement.texture;
            r.buffer = placement.buffer;
            r.currentAccess = placement.state;
            r.poolEntry = placement.entry;
            ++(placement.reused ? mStats.transientResourcesReused : mStats.transientResourcesCreated);
        }
        else
        {
            if (r.kind == ResourceKind::Texture)
            {
                r.texture = mDevice.CreatePlacedTexture(r.textureDesc, mTransientHeap, r.heapOffset);
            }
            else
            {
                r.buffer = mDevice.CreatePlacedBuffer(r.bufferDesc, mTransientHeap, r.heapOffset);
            }
            ++mStats.transientResourcesCreated;
        }
        AIR_CHECK(r.kind == ResourceKind::Texture ? r.texture.IsValid() : r.buffer.IsValid(),
                  "failed to create resource '{}'", r.name);
    }

    // 2. Resolve barrier resource ids to RHI handles. A transient's first barrier (discard) starts
    //    from the state its resource is really in: none for a new one, where an earlier frame left
    //    a pooled one.
    auto resolve = [&](std::vector<Barrier>& barriers, const std::vector<u32>& resources)
    {
        for (size_t i = 0; i < barriers.size(); ++i)
        {
            const Resource& r = mResources[resources[i]];
            barriers[i].texture = r.texture;
            barriers[i].buffer = r.buffer;
            if (barriers[i].discard)
            {
                barriers[i].before = r.currentAccess;
            }
        }
    };
    for (u32 p : mLivePasses)
    {
        resolve(mPasses[p].barriersBefore, mPasses[p].barrierResources);
    }
    for (Batch& batch : mBatches)
    {
        resolve(batch.barriersAfter, batch.barriersAfterResources);
    }
    resolve(mFinalBarriers, mFinalBarrierResources);

    // 3. Record. Each batch splits into chunks of passes, one command list each; every chunk of every
    //    batch records in parallel. A batch's hand-over (and the final) barriers end its last chunk.
    struct Chunk
    {
        u32 batch;
        u32 begin, end; // range in mLivePasses
        bool last;
    };
    std::vector<Chunk> chunks;
    const u32 chunkSize = mConfig.passesPerCommandList;
    for (u32 b = 0; b < mBatches.size(); ++b)
    {
        const Batch& batch = mBatches[b];
        if (batch.begin == batch.end)
        {
            if (batch.IsSubmitted())
            {
                chunks.push_back({b, batch.begin, batch.end, true}); // barriers only (or nothing: the fork)
            }
            continue;
        }
        for (u32 begin = batch.begin; begin < batch.end; begin += chunkSize)
        {
            const u32 end = std::min(batch.end, begin + chunkSize);
            chunks.push_back({b, begin, end, end == batch.end});
        }
    }
    std::vector<std::unique_ptr<rhi::CommandList>> lists(chunks.size());
    for (size_t c = 0; c < chunks.size(); ++c)
    {
        lists[c] = mDevice.CreateCommandList(mBatches[chunks[c].batch].queue);
    }
    auto recordChunk = [&](u32 c)
    {
        const Chunk& chunk = chunks[c];
        const Batch& batch = mBatches[chunk.batch];
        rhi::CommandList& cmd = *lists[c];
        cmd.Begin();
        RecordRange(cmd, chunk.begin, chunk.end);
        if (chunk.last)
        {
            cmd.Barriers(batch.barriersAfter);
            if (batch.finalBarriers)
            {
                cmd.Barriers(mFinalBarriers);
            }
        }
        cmd.End();
    };
    if (mConfig.parallelRecording && mTasks && chunks.size() > 1)
    {
        mTasks->ParallelFor(u32(chunks.size()), recordChunk);
    }
    else
    {
        for (u32 c = 0; c < chunks.size(); ++c)
        {
            recordChunk(c);
        }
    }
    mStats.commandListCount = u32(chunks.size());

    // 4. Submit batch by batch in declaration order; a batch first waits (on the GPU) for the batches
    //    of the other queue it depends on, which were all submitted before it.
    std::vector<u64> fences(mBatches.size(), 0);
    u64 fence = 0;
    size_t next = 0;
    for (u32 b = 0; b < mBatches.size(); ++b)
    {
        std::vector<rhi::CommandList*> raw;
        while (next < chunks.size() && chunks[next].batch == b)
        {
            raw.push_back(lists[next++].get());
        }
        if (raw.empty())
        {
            continue;
        }
        const Batch& batch = mBatches[b];
        for (size_t q = 0; q < size_t(rhi::QueueType::Count); ++q)
        {
            const u32 wait = batch.waitFor[q];
            if (wait != ~0u && fences[wait] != 0)
            {
                mDevice.QueueWait(batch.queue, rhi::QueueType(q), fences[wait]);
                ++mStats.queueWaitCount;
            }
        }
        fences[b] = mDevice.Submit(batch.queue, raw);
        ++mStats.submissionCount;
        fence = batch.queue == rhi::QueueType::Graphics ? fences[b] : fence;
    }

    // 5. Release transient resources (deferred by the device until the GPU is done) and hand out extractions.
    for (Resource& r : mResources)
    {
        if (r.extracted)
        {
            if (r.extractTextureOut)
            {
                *r.extractTextureOut = r.texture;
            }
            if (r.extractBufferOut)
            {
                *r.extractBufferOut = r.buffer;
            }
            if (!r.used)
            {
                AIR_LOG(Warning, "RDG", "extracted resource '{}' was never written by a pass", r.name);
            }
            continue;
        }
        if (!r.used || r.imported)
        {
            continue;
        }
        if (mPool)
        {
            mPool->SetFinalState(mPoolSlot, r.poolEntry, r.lastAccess); // the pool keeps it
        }
        else if (r.kind == ResourceKind::Texture)
        {
            mDevice.DestroyTexture(r.texture);
        }
        else
        {
            mDevice.DestroyBuffer(r.buffer);
        }
    }
    if (mPool)
    {
        mPool->EndFrame(mPoolSlot, fence); // the frame's GPU work ends with its last graphics submission
    }
    else if (mTransientHeap.IsValid())
    {
        mDevice.DestroyHeap(mTransientHeap);
    }
    return fence;
}

// ===========================================================================
// Debug output
// ===========================================================================

std::string RenderGraph::DumpText() const
{
    std::ostringstream out;
    out << std::format("RenderGraph: {} passes ({} culled), {} barriers, transient heap {} KiB (unaliased {} KiB)\n",
                       mStats.passCount, mStats.culledPassCount, mStats.barrierCount, mStats.transientHeapSize / 1024,
                       mStats.transientBytesUnaliased / 1024);
    out << "Passes:\n";
    for (u32 p = 0; p < mPasses.size(); ++p)
    {
        const Pass& pass = mPasses[p];
        out << std::format("  [{}] {} ({}){}\n", p, pass.name, FlagsToString(pass.flags), pass.culled ? " CULLED" : "");
        for (size_t i = 0; i < pass.barriersBefore.size(); ++i)
        {
            const Barrier& b = pass.barriersBefore[i];
            out << std::format("      barrier {}: {} -> {}{}\n", mResources[pass.barrierResources[i]].name,
                               rhi::ToString(b.before), rhi::ToString(b.after), b.discard ? " (discard)" : "");
        }
        for (const ResourceAccess& a : pass.accesses)
        {
            out << std::format("      {} {} as {}\n", IsWriteAccess(a.access) ? "write" : "read ",
                               mResources[a.resource].name, rhi::ToString(a.access));
        }
    }
    // The queue schedule, only when it is more than one graphics submission.
    if (mStats.asyncComputePassCount > 0)
    {
        out << "Submissions:\n";
        for (u32 b = 0; b < mBatches.size(); ++b)
        {
            const Batch& batch = mBatches[b];
            if (!batch.IsSubmitted())
            {
                continue;
            }
            std::string line = std::format("  ({}) {}:", b, rhi::ToString(batch.queue));
            line += batch.fork ? " fork" : "";
            for (u32 i = batch.begin; i < batch.end; ++i)
            {
                line += std::format(" [{}]", mLivePasses[i]);
            }
            for (size_t q = 0; q < size_t(rhi::QueueType::Count); ++q)
            {
                const u32 wait = batch.waitFor[q];
                if (wait != ~0u && mBatches[wait].IsSubmitted())
                {
                    line += std::format(", waits for ({})", wait);
                }
            }
            out << line << (batch.finalBarriers ? ", final barriers\n" : "\n");
            for (size_t i = 0; i < batch.barriersAfter.size(); ++i)
            {
                const Barrier& barrier = batch.barriersAfter[i];
                out << std::format("      then barrier {}: {} -> {} (for another queue)\n",
                                   mResources[batch.barriersAfterResources[i]].name, rhi::ToString(barrier.before),
                                   rhi::ToString(barrier.after));
            }
        }
    }
    out << "Resources:\n";
    for (const Resource& r : mResources)
    {
        const char* kind = r.kind == ResourceKind::Texture ? "tex" : "buf";
        const char* origin = r.imported ? "imported" : r.extracted ? "extracted" : "transient";
        if (!r.used)
        {
            out << std::format("  {} {} ({}) unused\n", kind, r.name, origin);
        }
        else if (r.IsTransient())
        {
            out << std::format("  {} {} ({}) passes [{}, {}] heap offset {} size {}\n", kind, r.name, origin,
                               r.firstPass, r.lastPass, r.heapOffset, r.allocation.size);
        }
        else
        {
            out << std::format("  {} {} ({}) passes [{}, {}] final {}\n", kind, r.name, origin, r.firstPass, r.lastPass,
                               rhi::ToString(r.finalAccess));
        }
    }
    return out.str();
}

json::Value RenderGraph::DumpJson() const
{
    auto barrierJson = [](const Barrier& b, u32 resource)
    {
        json::Value out = json::Value::MakeObject();
        out["resource"] = resource;
        out["before"] = rhi::ToString(b.before);
        out["after"] = rhi::ToString(b.after);
        if (b.discard)
        {
            out["discard"] = true;
        }
        return out;
    };
    json::Value out = json::Value::MakeObject();
    json::Value& stats = out["stats"];
    stats["passes"] = mStats.passCount;
    stats["culledPasses"] = mStats.culledPassCount;
    stats["barriers"] = mStats.barrierCount;
    stats["transientResources"] = mStats.transientResourceCount;
    stats["transientHeapBytes"] = mStats.transientHeapSize;
    stats["transientBytesUnaliased"] = mStats.transientBytesUnaliased;
    stats["commandLists"] = mStats.commandListCount;
    stats["asyncComputePasses"] = mStats.asyncComputePassCount;
    stats["submissions"] = mStats.submissionCount;
    stats["queueWaits"] = mStats.queueWaitCount;
    stats["transientHeapsCreated"] = mStats.transientHeapsCreated;
    stats["transientResourcesCreated"] = mStats.transientResourcesCreated;
    stats["transientResourcesReused"] = mStats.transientResourcesReused;

    json::Value passes = json::Value::MakeArray();
    for (u32 p = 0; p < mPasses.size(); ++p)
    {
        const Pass& pass = mPasses[p];
        json::Value item = json::Value::MakeObject();
        item["index"] = p;
        item["name"] = pass.name;
        item["flags"] = FlagsToString(pass.flags);
        item["culled"] = pass.culled;
        if (!pass.culled)
        {
            item["queue"] = rhi::ToString(pass.queue);
            item["batch"] = pass.batch;
            item["cpuMs"] = pass.cpuMs;
        }
        json::Value accesses = json::Value::MakeArray();
        for (const ResourceAccess& a : pass.accesses)
        {
            json::Value access = json::Value::MakeObject();
            access["resource"] = a.resource;
            access["access"] = rhi::ToString(a.access);
            access["write"] = IsWriteAccess(a.access);
            accesses.Push(std::move(access));
        }
        item["accesses"] = std::move(accesses);
        json::Value barriers = json::Value::MakeArray();
        for (size_t i = 0; i < pass.barriersBefore.size(); ++i)
        {
            barriers.Push(barrierJson(pass.barriersBefore[i], pass.barrierResources[i]));
        }
        item["barriers"] = std::move(barriers);
        passes.Push(std::move(item));
    }
    out["passes"] = std::move(passes);

    json::Value submissions = json::Value::MakeArray();
    for (u32 b = 0; b < mBatches.size(); ++b)
    {
        const Batch& batch = mBatches[b];
        if (!batch.IsSubmitted())
        {
            continue;
        }
        json::Value item = json::Value::MakeObject();
        item["batch"] = b;
        item["queue"] = rhi::ToString(batch.queue);
        json::Value batchPasses = json::Value::MakeArray();
        for (u32 i = batch.begin; i < batch.end; ++i)
        {
            batchPasses.Push(mLivePasses[i]);
        }
        item["passes"] = std::move(batchPasses);
        json::Value waits = json::Value::MakeArray();
        for (size_t q = 0; q < size_t(rhi::QueueType::Count); ++q)
        {
            const u32 wait = batch.waitFor[q];
            if (wait != ~0u && mBatches[wait].IsSubmitted())
            {
                json::Value w = json::Value::MakeObject();
                w["queue"] = rhi::ToString(rhi::QueueType(q));
                w["batch"] = wait;
                waits.Push(std::move(w));
            }
        }
        item["waits"] = std::move(waits);
        item["fork"] = batch.fork;
        item["finalBarriers"] = batch.finalBarriers;
        json::Value after = json::Value::MakeArray();
        for (size_t i = 0; i < batch.barriersAfter.size(); ++i)
        {
            after.Push(barrierJson(batch.barriersAfter[i], batch.barriersAfterResources[i]));
        }
        item["barriersAfter"] = std::move(after);
        submissions.Push(std::move(item));
    }
    out["submissions"] = std::move(submissions);
    json::Value finalBarriers = json::Value::MakeArray();
    for (size_t i = 0; i < mFinalBarriers.size(); ++i)
    {
        finalBarriers.Push(barrierJson(mFinalBarriers[i], mFinalBarrierResources[i]));
    }
    out["finalBarriers"] = std::move(finalBarriers);

    json::Value resources = json::Value::MakeArray();
    for (u32 i = 0; i < mResources.size(); ++i)
    {
        const Resource& r = mResources[i];
        json::Value item = json::Value::MakeObject();
        item["index"] = i;
        item["name"] = r.name;
        item["origin"] = r.imported ? "imported" : r.extracted ? "extracted" : "transient";
        item["used"] = r.used;
        if (r.kind == ResourceKind::Texture)
        {
            item["kind"] = "texture";
            item["format"] = rhi::ToString(r.textureDesc.format);
            item["width"] = r.textureDesc.width;
            item["height"] = r.textureDesc.height;
            item["depthOrArraySize"] = r.textureDesc.depthOrArraySize;
            item["mips"] = r.textureDesc.mipLevels;
        }
        else
        {
            item["kind"] = "buffer";
            item["bytes"] = r.bufferDesc.size;
        }
        if (r.used)
        {
            item["firstPass"] = r.firstPass;
            item["lastPass"] = r.lastPass;
        }
        if (r.used && r.IsTransient())
        {
            item["heapOffset"] = r.heapOffset;
            item["size"] = r.allocation.size;
            item["alignment"] = r.allocation.alignment;
            item["async"] = r.asyncUse;
        }
        if (r.imported)
        {
            item["initialAccess"] = rhi::ToString(r.initialAccess);
        }
        if (!r.IsTransient())
        {
            item["finalAccess"] = rhi::ToString(r.finalAccess);
        }
        resources.Push(std::move(item));
    }
    out["resources"] = std::move(resources);
    return out;
}

std::string RenderGraph::DumpGraphviz() const
{
    std::ostringstream out;
    out << "digraph RenderGraph {\n  rankdir=LR;\n  node [fontname=\"Helvetica\"];\n";
    for (u32 p = 0; p < mPasses.size(); ++p)
    {
        const Pass& pass = mPasses[p];
        out << std::format("  p{} [shape=box, style=filled, fillcolor=\"{}\", label=\"{}\"];\n", p,
                           pass.culled ? "#dddddd" : "#ffd27f", pass.name);
    }
    for (u32 r = 0; r < mResources.size(); ++r)
    {
        const Resource& res = mResources[r];
        const char* color = res.imported ? "#9fd3ff" : res.extracted ? "#b6f2a5" : "#ffffff";
        out << std::format("  r{} [shape=ellipse, style=filled, fillcolor=\"{}\", label=\"{}\"];\n", r, color,
                           res.name);
    }
    for (u32 p = 0; p < mPasses.size(); ++p)
    {
        for (const ResourceAccess& a : mPasses[p].accesses)
        {
            if (IsWriteAccess(a.access))
            {
                out << std::format("  p{} -> r{} [label=\"{}\", color=red];\n", p, a.resource, rhi::ToString(a.access));
            }
            else
            {
                out << std::format("  r{} -> p{} [label=\"{}\"];\n", a.resource, p, rhi::ToString(a.access));
            }
        }
    }
    out << "}\n";
    return out.str();
}

} // namespace air::rdg
