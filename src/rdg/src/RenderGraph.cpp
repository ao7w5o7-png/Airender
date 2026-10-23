#include "air/rdg/RenderGraph.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"
#include "air/core/Profiler.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <memory>
#include <sstream>

namespace air::rdg {

using rhi::Barrier;
using rhi::BufferUsage;
using rhi::TextureUsage;

namespace {

// Which usage flag an access needs. Catches "forgot UnorderedAccess usage" at AddPass time with the
// pass name in the message, instead of as a device removal later.
bool TextureSupports(const rhi::TextureDesc& desc, Access access) {
    auto need = [&](Access mask, TextureUsage usage) { return !Any(access & mask) || Any(desc.usage & usage); };
    return need(Access::SRVGraphics | Access::SRVCompute, TextureUsage::ShaderResource) &&
           need(Access::UAVMask, TextureUsage::UnorderedAccess) &&
           need(Access::RenderTarget, TextureUsage::RenderTarget) &&
           need(Access::DepthRead | Access::DepthWrite, TextureUsage::DepthStencil) &&
           need(Access::CopySrc, TextureUsage::CopySrc) && need(Access::CopyDst, TextureUsage::CopyDst);
}

bool BufferSupports(const rhi::BufferDesc& desc, Access access) {
    auto need = [&](Access mask, BufferUsage usage) { return !Any(access & mask) || Any(desc.usage & usage); };
    return need(Access::SRVGraphics | Access::SRVCompute, BufferUsage::ShaderResource) &&
           need(Access::UAVMask, BufferUsage::UnorderedAccess) && need(Access::Constant, BufferUsage::Constant) &&
           need(Access::IndirectArgs, BufferUsage::Indirect) && need(Access::CopySrc, BufferUsage::CopySrc) &&
           need(Access::CopyDst, BufferUsage::CopyDst);
}

std::string FlagsToString(PassFlags flags) {
    std::string s;
    auto add = [&](PassFlags f, const char* name) {
        if (Any(flags & f)) {
            s += s.empty() ? name : std::string("|") + name;
        }
    };
    add(PassFlags::Raster, "Raster");
    add(PassFlags::Compute, "Compute");
    add(PassFlags::RayTracing, "RayTracing");
    add(PassFlags::Copy, "Copy");
    add(PassFlags::NeverCull, "NeverCull");
    return s.empty() ? "None" : s;
}

} // namespace

// ===========================================================================
// PassBuilder / PassContext
// ===========================================================================

void PassBuilder::Read(TextureRef t, Access a) {
    mGraph.AddAccess(mPass, t.id, RenderGraph::ResourceKind::Texture, a, false);
}
void PassBuilder::Write(TextureRef t, Access a) {
    mGraph.AddAccess(mPass, t.id, RenderGraph::ResourceKind::Texture, a, true);
}
void PassBuilder::Read(BufferRef b, Access a) {
    mGraph.AddAccess(mPass, b.id, RenderGraph::ResourceKind::Buffer, a, false);
}
void PassBuilder::Write(BufferRef b, Access a) {
    mGraph.AddAccess(mPass, b.id, RenderGraph::ResourceKind::Buffer, a, true);
}

rhi::TextureHandle PassContext::Get(TextureRef texture) const {
    return mGraph.ResourceFor(mPass, texture.id, RenderGraph::ResourceKind::Texture).texture;
}
rhi::BufferHandle PassContext::Get(BufferRef buffer) const {
    return mGraph.ResourceFor(mPass, buffer.id, RenderGraph::ResourceKind::Buffer).buffer;
}

// ===========================================================================
// RenderGraph: building
// ===========================================================================

RenderGraph::RenderGraph(rhi::Device& device, TaskSystem* tasks, Config config)
    : mDevice(device), mTasks(tasks), mConfig(config), mOwnerThread(std::this_thread::get_id()) {
    mConfig.passesPerCommandList = std::max(1u, mConfig.passesPerCommandList);
}

RenderGraph::~RenderGraph() {
    if (mCompiled && !mExecuted) {
        AIR_LOG(Warning, "RDG", "render graph compiled but never executed ({} passes)", mPasses.size());
    }
}

void RenderGraph::CheckThread() const {
    AIR_CHECK(std::this_thread::get_id() == mOwnerThread,
              "RenderGraph is built on one thread; record parallel work inside pass lambdas instead");
}

TextureRef RenderGraph::CreateTexture(const rhi::TextureDesc& desc) {
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

BufferRef RenderGraph::CreateBuffer(const rhi::BufferDesc& desc) {
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

TextureRef RenderGraph::ImportTexture(rhi::TextureHandle texture, Access current, Access final) {
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

BufferRef RenderGraph::ImportBuffer(rhi::BufferHandle buffer, Access current, Access final) {
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

void RenderGraph::ExtractTexture(TextureRef texture, Access final, rhi::TextureHandle* out) {
    CheckThread();
    AIR_CHECK(texture.id < mResources.size() && mResources[texture.id].kind == ResourceKind::Texture,
              "ExtractTexture: invalid ref");
    Resource& r = mResources[texture.id];
    AIR_CHECK(!r.imported, "ExtractTexture on imported texture '{}'", r.name);
    r.extracted = true;
    r.finalAccess = final;
    r.extractTextureOut = out;
}

void RenderGraph::ExtractBuffer(BufferRef buffer, Access final, rhi::BufferHandle* out) {
    CheckThread();
    AIR_CHECK(buffer.id < mResources.size() && mResources[buffer.id].kind == ResourceKind::Buffer,
              "ExtractBuffer: invalid ref");
    Resource& r = mResources[buffer.id];
    AIR_CHECK(!r.imported, "ExtractBuffer on imported buffer '{}'", r.name);
    r.extracted = true;
    r.finalAccess = final;
    r.extractBufferOut = out;
}

u32 RenderGraph::AddPassInternal(std::string name, PassFlags flags, std::function<void(PassContext&)> execute) {
    CheckThread();
    AIR_CHECK(!mCompiled, "AddPass('{}') after Compile()", name);
    Pass pass;
    pass.name = std::move(name);
    pass.flags = flags;
    pass.execute = std::move(execute);
    mPasses.push_back(std::move(pass));
    return u32(mPasses.size() - 1);
}

void RenderGraph::AddAccess(u32 passIndex, u32 id, ResourceKind kind, Access access, bool write) {
    Pass& pass = mPasses[passIndex];
    AIR_CHECK(id < mResources.size() && mResources[id].kind == kind, "pass '{}' uses an invalid resource ref",
              pass.name);
    const Resource& r = mResources[id];
    AIR_CHECK(access != Access::None, "pass '{}' declares Access::None on '{}'", pass.name, r.name);
    if (write) {
        AIR_CHECK(IsWriteAccess(access), "pass '{}' declares Write on '{}' with read-only access {}", pass.name, r.name,
                  rhi::ToString(access));
    } else {
        AIR_CHECK(!IsWriteAccess(access), "pass '{}' declares Read on '{}' with writable access {}", pass.name, r.name,
                  rhi::ToString(access));
    }
    const bool supported =
        kind == ResourceKind::Texture ? TextureSupports(r.textureDesc, access) : BufferSupports(r.bufferDesc, access);
    AIR_CHECK(supported, "pass '{}': '{}' is not created with the usage flags needed for {}", pass.name, r.name,
              rhi::ToString(access));

    for (ResourceAccess& existing : pass.accesses) {
        if (existing.resource != id) {
            continue;
        }
        if (existing.access == access) {
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

const RenderGraph::Resource& RenderGraph::ResourceFor(u32 passIndex, u32 id, ResourceKind kind) const {
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

void RenderGraph::Compile() {
    AIR_PROFILE_SCOPE("RDG.Compile");
    CheckThread();
    if (mCompiled) {
        return;
    }
    mCompiled = true;
    mStats = {};
    mStats.passCount = u32(mPasses.size());

    CullPasses();
    ComputeLifetimes();
    AllocateTransientMemory();
    BuildBarriers();
}

void RenderGraph::CullPasses() {
    std::vector<bool> needed(mResources.size(), false);
    for (u32 i = 0; i < mResources.size(); ++i) {
        needed[i] = mResources[i].imported || mResources[i].extracted;
    }
    // Walk backwards: a pass survives if it is NeverCull or writes something a surviving pass (or
    // the outside world) needs. Its inputs then become needed too.
    for (u32 p = u32(mPasses.size()); p-- > 0;) {
        Pass& pass = mPasses[p];
        bool keep = !mConfig.enableCulling || Any(pass.flags & PassFlags::NeverCull);
        for (const ResourceAccess& a : pass.accesses) {
            keep = keep || (IsWriteAccess(a.access) && needed[a.resource]);
        }
        pass.culled = !keep;
        if (keep) {
            for (const ResourceAccess& a : pass.accesses) {
                needed[a.resource] = true;
            }
        }
    }
    mLivePasses.clear();
    for (u32 p = 0; p < mPasses.size(); ++p) {
        if (!mPasses[p].culled) {
            mLivePasses.push_back(p);
        } else {
            ++mStats.culledPassCount;
        }
    }
}

void RenderGraph::ComputeLifetimes() {
    for (u32 p : mLivePasses) {
        for (const ResourceAccess& a : mPasses[p].accesses) {
            Resource& r = mResources[a.resource];
            r.used = true;
            r.firstPass = std::min(r.firstPass, p);
            r.lastPass = std::max(r.lastPass, p);
        }
    }
}

void RenderGraph::AllocateTransientMemory() {
    struct Placed {
        u32 first, last;
        u64 begin, end;
    };
    std::vector<u32> order;
    for (u32 i = 0; i < mResources.size(); ++i) {
        Resource& r = mResources[i];
        if (!r.used || !r.IsTransient()) {
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
    for (u32 index : order) {
        Resource& r = mResources[index];
        u64 offset = 0;
        if (mConfig.enableAliasing) {
            // First fit: bump past every placed resource that is alive at the same time and overlaps.
            bool moved = true;
            while (moved) {
                moved = false;
                for (const Placed& p : placed) {
                    const bool timeOverlap = p.first <= r.lastPass && r.firstPass <= p.last;
                    const bool memOverlap = p.begin < offset + r.allocation.size && offset < p.end;
                    if (timeOverlap && memOverlap) {
                        offset = AlignUp(p.end, r.allocation.alignment);
                        moved = true;
                    }
                }
            }
        } else {
            offset = AlignUp(heapSize, r.allocation.alignment);
        }
        r.heapOffset = offset;
        placed.push_back({r.firstPass, r.lastPass, offset, offset + r.allocation.size});
        heapSize = std::max(heapSize, offset + r.allocation.size);
    }
    mStats.transientHeapSize = heapSize;
}

Barrier RenderGraph::MakeBarrier(const Resource& r, Access before, Access after, bool discard) const {
    Barrier b;
    b.type = r.kind == ResourceKind::Texture ? Barrier::Type::Texture : Barrier::Type::Buffer;
    b.before = before;
    b.after = after;
    b.discard = discard;
    return b;
}

void RenderGraph::BuildBarriers() {
    std::vector<Access> state(mResources.size());
    std::vector<bool> touched(mResources.size(), false);
    for (u32 i = 0; i < mResources.size(); ++i) {
        state[i] = mResources[i].imported ? mResources[i].initialAccess : Access::None;
    }

    // Handles are only known at Execute(); remember which resource each barrier is for.
    auto add = [](std::vector<Barrier>& barriers, std::vector<u32>& resources, Barrier b, u32 resource) {
        barriers.push_back(b);
        resources.push_back(resource);
    };

    for (u32 p : mLivePasses) {
        Pass& pass = mPasses[p];
        pass.barriersBefore.clear();
        pass.barrierResources.clear();
        for (const ResourceAccess& a : pass.accesses) {
            const Resource& r = mResources[a.resource];
            const Access before = state[a.resource];
            if (!touched[a.resource] && !r.imported) {
                // First use of a graph-created resource: contents are undefined (and possibly aliased).
                add(pass.barriersBefore, pass.barrierResources, MakeBarrier(r, Access::None, a.access, true),
                    a.resource);
            } else if (before != a.access || Any(a.access & Access::UAVMask)) {
                // A state change, or UAV -> UAV to order the writes of consecutive passes.
                add(pass.barriersBefore, pass.barrierResources, MakeBarrier(r, before, a.access, false), a.resource);
            }
            touched[a.resource] = true;
            state[a.resource] = a.access;
        }
        mStats.barrierCount += u32(pass.barriersBefore.size());
    }

    mFinalBarriers.clear();
    mFinalBarrierResources.clear();
    for (u32 i = 0; i < mResources.size(); ++i) {
        const Resource& r = mResources[i];
        const bool exported = r.imported || (r.extracted && r.used);
        if (exported && state[i] != r.finalAccess) {
            add(mFinalBarriers, mFinalBarrierResources, MakeBarrier(r, state[i], r.finalAccess, false), i);
        }
    }
    mStats.barrierCount += u32(mFinalBarriers.size());
}

// ===========================================================================
// RenderGraph: execute
// ===========================================================================

void RenderGraph::RecordRange(rhi::CommandList& cmd, u32 begin, u32 end, bool last) {
    for (u32 i = begin; i < end; ++i) {
        const u32 passIndex = mLivePasses[i];
        Pass& pass = mPasses[passIndex];
        cmd.Barriers(pass.barriersBefore);
        cmd.BeginEvent(pass.name);
        PassContext context(cmd, mDevice, *this, passIndex);
        if (pass.execute) {
            AIR_PROFILE_SCOPE(pass.name);
            const auto start = std::chrono::steady_clock::now();
            pass.execute(context);
            pass.cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        cmd.EndEvent();
    }
    if (last) {
        cmd.Barriers(mFinalBarriers);
    }
}

std::vector<RenderGraph::PassTiming> RenderGraph::GetPassTimings() const {
    std::vector<PassTiming> timings;
    timings.reserve(mLivePasses.size());
    for (u32 p : mLivePasses) {
        timings.push_back({mPasses[p].name, mPasses[p].cpuMs});
    }
    return timings;
}

u64 RenderGraph::Execute() {
    AIR_PROFILE_SCOPE("RDG.Execute");
    CheckThread();
    AIR_CHECK(!mExecuted, "RenderGraph::Execute called twice");
    Compile();
    mExecuted = true;

    // 1. Memory: one placed heap for all transient resources, committed memory for extracted ones.
    if (mStats.transientHeapSize > 0) {
        mTransientHeap =
            mDevice.CreateHeap(mStats.transientHeapSize, rhi::MemoryLocation::DeviceLocal, "RDG Transient");
        AIR_CHECK(mTransientHeap.IsValid(), "failed to allocate {} byte transient heap", mStats.transientHeapSize);
    }
    for (Resource& r : mResources) {
        if (!r.used || r.imported) {
            continue;
        }
        if (r.kind == ResourceKind::Texture) {
            r.texture = r.extracted ? mDevice.CreateTexture(r.textureDesc)
                                    : mDevice.CreatePlacedTexture(r.textureDesc, mTransientHeap, r.heapOffset);
            AIR_CHECK(r.texture.IsValid(), "failed to create texture '{}'", r.name);
        } else {
            r.buffer = r.extracted ? mDevice.CreateBuffer(r.bufferDesc)
                                   : mDevice.CreatePlacedBuffer(r.bufferDesc, mTransientHeap, r.heapOffset);
            AIR_CHECK(r.buffer.IsValid(), "failed to create buffer '{}'", r.name);
        }
    }

    // 2. Resolve barrier resource ids to RHI handles.
    auto resolve = [&](std::vector<Barrier>& barriers, const std::vector<u32>& resources) {
        for (size_t i = 0; i < barriers.size(); ++i) {
            barriers[i].texture = mResources[resources[i]].texture;
            barriers[i].buffer = mResources[resources[i]].buffer;
        }
    };
    for (u32 p : mLivePasses) {
        resolve(mPasses[p].barriersBefore, mPasses[p].barrierResources);
    }
    resolve(mFinalBarriers, mFinalBarrierResources);

    // 3. Record. Each chunk of passes goes into its own command list; chunks record in parallel and
    //    are submitted in declaration order, so GPU execution order matches the graph.
    const u32 liveCount = u32(mLivePasses.size());
    const u32 chunkSize = mConfig.passesPerCommandList;
    const u32 chunkCount = std::max(1u, (liveCount + chunkSize - 1) / chunkSize);
    std::vector<std::unique_ptr<rhi::CommandList>> lists(chunkCount);
    for (auto& list : lists) {
        list = mDevice.CreateCommandList(rhi::QueueType::Graphics);
    }
    auto recordChunk = [&](u32 chunk) {
        rhi::CommandList& cmd = *lists[chunk];
        cmd.Begin();
        const u32 begin = chunk * chunkSize;
        RecordRange(cmd, begin, std::min(liveCount, begin + chunkSize), chunk == chunkCount - 1);
        cmd.End();
    };
    if (mConfig.parallelRecording && mTasks && chunkCount > 1) {
        mTasks->ParallelFor(chunkCount, recordChunk);
    } else {
        for (u32 c = 0; c < chunkCount; ++c) {
            recordChunk(c);
        }
    }
    mStats.commandListCount = chunkCount;

    std::vector<rhi::CommandList*> raw;
    raw.reserve(lists.size());
    for (auto& list : lists) {
        raw.push_back(list.get());
    }
    const u64 fence = mDevice.Submit(rhi::QueueType::Graphics, raw);

    // 4. Release transient resources (deferred by the device until the GPU is done) and hand out extractions.
    for (Resource& r : mResources) {
        if (r.extracted) {
            if (r.extractTextureOut) {
                *r.extractTextureOut = r.texture;
            }
            if (r.extractBufferOut) {
                *r.extractBufferOut = r.buffer;
            }
            if (!r.used) {
                AIR_LOG(Warning, "RDG", "extracted resource '{}' was never written by a pass", r.name);
            }
            continue;
        }
        if (!r.used || r.imported) {
            continue;
        }
        if (r.kind == ResourceKind::Texture) {
            mDevice.DestroyTexture(r.texture);
        } else {
            mDevice.DestroyBuffer(r.buffer);
        }
    }
    if (mTransientHeap.IsValid()) {
        mDevice.DestroyHeap(mTransientHeap);
    }
    return fence;
}

// ===========================================================================
// Debug output
// ===========================================================================

std::string RenderGraph::DumpText() const {
    std::ostringstream out;
    out << std::format("RenderGraph: {} passes ({} culled), {} barriers, transient heap {} KiB (unaliased {} KiB)\n",
                       mStats.passCount, mStats.culledPassCount, mStats.barrierCount, mStats.transientHeapSize / 1024,
                       mStats.transientBytesUnaliased / 1024);
    out << "Passes:\n";
    for (u32 p = 0; p < mPasses.size(); ++p) {
        const Pass& pass = mPasses[p];
        out << std::format("  [{}] {} ({}){}\n", p, pass.name, FlagsToString(pass.flags), pass.culled ? " CULLED" : "");
        for (size_t i = 0; i < pass.barriersBefore.size(); ++i) {
            const Barrier& b = pass.barriersBefore[i];
            out << std::format("      barrier {}: {} -> {}{}\n", mResources[pass.barrierResources[i]].name,
                               rhi::ToString(b.before), rhi::ToString(b.after), b.discard ? " (discard)" : "");
        }
        for (const ResourceAccess& a : pass.accesses) {
            out << std::format("      {} {} as {}\n", IsWriteAccess(a.access) ? "write" : "read ",
                               mResources[a.resource].name, rhi::ToString(a.access));
        }
    }
    out << "Resources:\n";
    for (const Resource& r : mResources) {
        const char* kind = r.kind == ResourceKind::Texture ? "tex" : "buf";
        const char* origin = r.imported ? "imported" : r.extracted ? "extracted" : "transient";
        if (!r.used) {
            out << std::format("  {} {} ({}) unused\n", kind, r.name, origin);
        } else if (r.IsTransient()) {
            out << std::format("  {} {} ({}) passes [{}, {}] heap offset {} size {}\n", kind, r.name, origin,
                               r.firstPass, r.lastPass, r.heapOffset, r.allocation.size);
        } else {
            out << std::format("  {} {} ({}) passes [{}, {}] final {}\n", kind, r.name, origin, r.firstPass, r.lastPass,
                               rhi::ToString(r.finalAccess));
        }
    }
    return out.str();
}

std::string RenderGraph::DumpGraphviz() const {
    std::ostringstream out;
    out << "digraph RenderGraph {\n  rankdir=LR;\n  node [fontname=\"Helvetica\"];\n";
    for (u32 p = 0; p < mPasses.size(); ++p) {
        const Pass& pass = mPasses[p];
        out << std::format("  p{} [shape=box, style=filled, fillcolor=\"{}\", label=\"{}\"];\n", p,
                           pass.culled ? "#dddddd" : "#ffd27f", pass.name);
    }
    for (u32 r = 0; r < mResources.size(); ++r) {
        const Resource& res = mResources[r];
        const char* color = res.imported ? "#9fd3ff" : res.extracted ? "#b6f2a5" : "#ffffff";
        out << std::format("  r{} [shape=ellipse, style=filled, fillcolor=\"{}\", label=\"{}\"];\n", r, color,
                           res.name);
    }
    for (u32 p = 0; p < mPasses.size(); ++p) {
        for (const ResourceAccess& a : mPasses[p].accesses) {
            if (IsWriteAccess(a.access)) {
                out << std::format("  p{} -> r{} [label=\"{}\", color=red];\n", p, a.resource, rhi::ToString(a.access));
            } else {
                out << std::format("  r{} -> p{} [label=\"{}\"];\n", a.resource, p, rhi::ToString(a.access));
            }
        }
    }
    out << "}\n";
    return out.str();
}

} // namespace air::rdg
