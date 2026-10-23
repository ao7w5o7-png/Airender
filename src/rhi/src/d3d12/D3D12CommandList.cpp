// Direct3D 12 command recording and barrier translation.

#include "D3D12Backend.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace air::rhi::d3d12 {

D3D12CommandList::~D3D12CommandList() {
    if (mContext) {
        if (mRecording) {
            mList->Close();
        }
        mDevice.ReleaseContext(mQueue, std::move(mContext)); // never submitted: reusable right away
    }
}

void D3D12CommandList::Begin() {
    AIR_CHECK(!mRecording, "Begin() on a command list that is already recording");
    if (!mContext) {
        mContext = mDevice.AcquireContext(mQueue);
    }
    AIR_D3D12_CHECK(mContext->allocator->Reset());
    AIR_D3D12_CHECK(mContext->list->Reset(mContext->allocator.Get(), nullptr));
    mList = mContext->list.Get();
    mRecording = true;
    mRootStateBound = false;
    mEventDepth = 0;
    mRayTracingPipeline = nullptr;
    mCapture = mDevice.CaptureEnabled();
    mTimestamps = mQueue == QueueType::Copy ? nullptr : mDevice.CurrentTimestampFrame();
    mTimestampBlocks.clear();
    mOpenScopes.clear();
    mClosedScopes.clear();
}

void D3D12CommandList::End() {
    AIR_CHECK(mRecording, "End() without Begin()");
    AIR_CHECK(mEventDepth == 0, "{} unbalanced BeginEvent()", mEventDepth);
    if (mTimestamps) {
        for (const TimestampBlock& block : mTimestampBlocks) {
            if (block.used > block.begin) {
                mList->ResolveQueryData(mTimestamps->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, block.begin,
                                        block.used - block.begin, mTimestamps->readback.Get(), u64(block.begin) * 8);
            }
        }
    }
    AIR_D3D12_CHECK(mList->Close());
    mRecording = false;
}

std::unique_ptr<CommandContext> D3D12CommandList::DetachContext() {
    AIR_CHECK(!mRecording, "Submit() of a command list that is still recording (missing End())");
    mList = nullptr;
    return std::move(mContext);
}

void D3D12CommandList::BindRootState() {
    if (mRootStateBound || mQueue == QueueType::Copy) {
        return;
    }
    ID3D12DescriptorHeap* heap = mDevice.BindlessHeap();
    mList->SetDescriptorHeaps(1, &heap);
    mList->SetComputeRootSignature(mDevice.RootSignature());
    if (mQueue == QueueType::Graphics) {
        mList->SetGraphicsRootSignature(mDevice.RootSignature());
    }
    mRootStateBound = true;
}

void D3D12CommandList::Capture(std::string text) {
    if (mCapture) {
        mCaptured.push_back(std::move(text));
    }
}

void D3D12CommandList::Barriers(std::span<const Barrier> barriers) {
    std::vector<D3D12_RESOURCE_BARRIER> native;
    native.reserve(barriers.size() * 2);
    auto uavBarrier = [&](ID3D12Resource* resource) {
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = resource;
        native.push_back(b);
    };
    auto transition = [&](ID3D12Resource* resource, Access before, Access after) {
        const D3D12_RESOURCE_STATES from = ToStates(before);
        const D3D12_RESOURCE_STATES to = ToStates(after);
        if (from == to) {
            if (to == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                uavBarrier(resource);
            }
            return;
        }
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = resource;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        native.push_back(b);
    };

    for (const Barrier& barrier : barriers) {
        switch (barrier.type) {
        case Barrier::Type::Texture: {
            D3D12Texture* texture = mDevice.Texture(barrier.texture);
            AIR_CHECK(texture, "barrier on invalid texture");
            if (barrier.discard) {
                // First use of a (possibly aliased) transient: activate it in the heap. Fresh
                // placed resources start in COMMON, which is what Access::None maps to.
                D3D12_RESOURCE_BARRIER aliasing = {};
                aliasing.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                aliasing.Aliasing.pResourceAfter = texture->resource.Get();
                native.push_back(aliasing);
            }
            transition(texture->resource.Get(), barrier.discard ? Access::None : barrier.before, barrier.after);
            Capture(std::format("Barrier tex '{}' {} -> {}{}", texture->desc.debugName, ToString(barrier.before),
                                ToString(barrier.after), barrier.discard ? " discard" : ""));
            break;
        }
        case Barrier::Type::Buffer: {
            D3D12Buffer* buffer = mDevice.Buffer(barrier.buffer);
            AIR_CHECK(buffer, "barrier on invalid buffer");
            if (buffer->hostVisible) {
                break; // upload / readback heaps have a fixed state
            }
            if (barrier.discard && !buffer->ownsAllocation) {
                D3D12_RESOURCE_BARRIER aliasing = {};
                aliasing.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                aliasing.Aliasing.pResourceAfter = buffer->resource.Get();
                native.push_back(aliasing);
            }
            transition(buffer->resource.Get(), barrier.discard ? Access::None : barrier.before, barrier.after);
            Capture(std::format("Barrier buf '{}' {} -> {}", buffer->desc.debugName, ToString(barrier.before),
                                ToString(barrier.after)));
            break;
        }
        case Barrier::Type::AccelStruct: {
            D3D12AccelStruct* as = mDevice.AccelStruct(barrier.accelStruct);
            AIR_CHECK(as, "barrier on invalid acceleration structure");
            uavBarrier(as->resource.Get());
            break;
        }
        case Barrier::Type::Aliasing: {
            D3D12_RESOURCE_BARRIER aliasing = {};
            aliasing.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
            native.push_back(aliasing);
            break;
        }
        }
    }
    if (!native.empty()) {
        mList->ResourceBarrier(UINT(native.size()), native.data());
    }
}

void D3D12CommandList::CopyBuffer(BufferHandle dst, u64 dstOffset, BufferHandle src, u64 srcOffset, u64 size) {
    D3D12Buffer* d = mDevice.Buffer(dst);
    D3D12Buffer* s = mDevice.Buffer(src);
    AIR_CHECK(d && s, "CopyBuffer: invalid buffer");
    AIR_CHECK(dstOffset + size <= d->desc.size && srcOffset + size <= s->desc.size,
              "CopyBuffer '{}' <- '{}': {} bytes out of range", d->desc.debugName, s->desc.debugName, size);
    mList->CopyBufferRegion(d->resource.Get(), dstOffset, s->resource.Get(), srcOffset, size);
    Capture(std::format("CopyBuffer '{}' <- '{}' ({} bytes)", d->desc.debugName, s->desc.debugName, size));
}

namespace {

D3D12_PLACED_SUBRESOURCE_FOOTPRINT Footprint(const TextureDesc& desc, u32 mip, u64 offset) {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    footprint.Offset = offset;
    footprint.Footprint.Format = ToDxgi(desc.format);
    footprint.Footprint.Width = std::max(1u, desc.width >> mip);
    footprint.Footprint.Height = std::max(1u, desc.height >> mip);
    footprint.Footprint.Depth = 1;
    footprint.Footprint.RowPitch = TextureCopyRowPitch(desc.format, footprint.Footprint.Width);
    return footprint;
}

} // namespace

void D3D12CommandList::CopyBufferToTexture(TextureHandle dst, u32 mip, BufferHandle src, u64 srcOffset) {
    D3D12Texture* t = mDevice.Texture(dst);
    D3D12Buffer* b = mDevice.Buffer(src);
    AIR_CHECK(t && b, "CopyBufferToTexture: invalid resource");
    AIR_CHECK(srcOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT == 0, "CopyBufferToTexture: offset {} not 512 aligned",
              srcOffset);
    D3D12_TEXTURE_COPY_LOCATION to = {};
    to.pResource = t->resource.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.SubresourceIndex = mip;
    D3D12_TEXTURE_COPY_LOCATION from = {};
    from.pResource = b->resource.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = Footprint(t->desc, mip, srcOffset);
    mList->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    Capture(std::format("CopyBufferToTexture '{}' mip {} <- '{}'", t->desc.debugName, mip, b->desc.debugName));
}

void D3D12CommandList::CopyTextureToBuffer(BufferHandle dst, u64 dstOffset, TextureHandle src, u32 mip) {
    D3D12Texture* t = mDevice.Texture(src);
    D3D12Buffer* b = mDevice.Buffer(dst);
    AIR_CHECK(t && b, "CopyTextureToBuffer: invalid resource");
    AIR_CHECK(dstOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT == 0, "CopyTextureToBuffer: offset {} not 512 aligned",
              dstOffset);
    D3D12_TEXTURE_COPY_LOCATION to = {};
    to.pResource = b->resource.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = Footprint(t->desc, mip, dstOffset);
    D3D12_TEXTURE_COPY_LOCATION from = {};
    from.pResource = t->resource.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.SubresourceIndex = mip;
    mList->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    Capture(std::format("CopyTextureToBuffer '{}' <- '{}' mip {}", b->desc.debugName, t->desc.debugName, mip));
}

void D3D12CommandList::CopyTexture(TextureHandle dst, TextureHandle src) {
    D3D12Texture* d = mDevice.Texture(dst);
    D3D12Texture* s = mDevice.Texture(src);
    AIR_CHECK(d && s, "CopyTexture: invalid texture");
    mList->CopyResource(d->resource.Get(), s->resource.Get());
    Capture(std::format("CopyTexture '{}' <- '{}'", d->desc.debugName, s->desc.debugName));
}

void D3D12CommandList::ClearTexture(TextureHandle handle, const float color[4]) {
    D3D12Texture* t = mDevice.Texture(handle);
    AIR_CHECK(t, "ClearTexture: invalid texture");
    if (Any(t->desc.usage & TextureUsage::RenderTarget)) {
        mList->ClearRenderTargetView(mDevice.GetRtv(handle), color, 0, nullptr);
    } else {
        AIR_CHECK(Any(t->desc.usage & TextureUsage::UnorderedAccess),
                  "ClearTexture '{}' needs RenderTarget or UnorderedAccess usage", t->desc.debugName);
        BindRootState();
        const BindlessIndex uav = mDevice.GetBindlessUAV(handle, 0);
        const DescriptorHandles handles = mDevice.GetBindlessDescriptor(uav);
        mList->ClearUnorderedAccessViewFloat(D3D12_GPU_DESCRIPTOR_HANDLE{handles.gpu}, mDevice.GetUavCpuMirror(uav),
                                             t->resource.Get(), color, 0, nullptr);
    }
    Capture(std::format("ClearTexture '{}'", t->desc.debugName));
}

void D3D12CommandList::SetPipeline(PipelineHandle handle) {
    D3D12Pipeline* pipeline = mDevice.Pipeline(handle);
    AIR_CHECK(pipeline, "SetPipeline: invalid pipeline");
    BindRootState();
    if (pipeline->rayTracing) {
        mList->SetPipelineState1(pipeline->stateObject.Get());
        mRayTracingPipeline = pipeline;
    } else {
        mList->SetPipelineState(pipeline->pso.Get());
        mRayTracingPipeline = nullptr;
    }
    Capture(std::format("SetPipeline '{}'", pipeline->name));
}

void D3D12CommandList::PushConstants(const void* data, u32 size) {
    AIR_CHECK(size <= kRootConstantCount * 4, "push constants limited to {} bytes, got {}", kRootConstantCount * 4,
              size);
    BindRootState();
    u32 words[kRootConstantCount] = {};
    std::memcpy(words, data, size);
    mList->SetComputeRoot32BitConstants(0, (size + 3) / 4, words, 0);
}

void D3D12CommandList::Dispatch(u32 x, u32 y, u32 z) {
    mList->Dispatch(x, y, z);
    Capture(std::format("Dispatch {}x{}x{}", x, y, z));
}

void D3D12CommandList::DispatchRays(u32 width, u32 height, u32 depth) {
    AIR_CHECK(mRayTracingPipeline, "DispatchRays without a ray tracing pipeline");
    D3D12_DISPATCH_RAYS_DESC desc = {};
    desc.RayGenerationShaderRecord = mRayTracingPipeline->rayGen;
    desc.MissShaderTable = mRayTracingPipeline->miss;
    desc.HitGroupTable = mRayTracingPipeline->hitGroups;
    desc.Width = width;
    desc.Height = height;
    desc.Depth = depth;
    mList->DispatchRays(&desc);
    Capture(std::format("DispatchRays {}x{}x{}", width, height, depth));
}

void D3D12CommandList::BeginRenderPass(TextureHandle target, const float* clearColor) {
    D3D12Texture* t = mDevice.Texture(target);
    AIR_CHECK(t, "BeginRenderPass: invalid texture");
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = mDevice.GetRtv(target);
    BindRootState();
    mList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT viewport = {0, 0, float(t->desc.width), float(t->desc.height), 0, 1};
    const D3D12_RECT scissor = {0, 0, LONG(t->desc.width), LONG(t->desc.height)};
    mList->RSSetViewports(1, &viewport);
    mList->RSSetScissorRects(1, &scissor);
    if (clearColor) {
        mList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
    }
    Capture(std::format("BeginRenderPass '{}'", t->desc.debugName));
}

void D3D12CommandList::EndRenderPass() {
    Capture("EndRenderPass");
}

u32 D3D12CommandList::AllocateTimestamp() {
    if (mTimestampBlocks.empty() || mTimestampBlocks.back().used == mTimestampBlocks.back().begin + kTimestampBlock) {
        const u32 begin = mTimestamps->next.fetch_add(kTimestampBlock);
        if (begin + kTimestampBlock > kMaxTimestampsPerFrame) {
            return ~0u; // out of queries this frame: the scope is not timed
        }
        mTimestampBlocks.push_back({begin, begin});
    }
    return mTimestampBlocks.back().used++;
}

void D3D12CommandList::BeginEvent(std::string_view name) {
    ++mEventDepth;
    // PIX convention: metadata 1 = ANSI string (also shown by RenderDoc and Nsight).
    std::string text(name);
    mList->BeginEvent(1, text.c_str(), UINT(text.size() + 1));
    if (mTimestamps) {
        const u32 index = AllocateTimestamp();
        if (index != ~0u) {
            mList->EndQuery(mTimestamps->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
        }
        mOpenScopes.push_back({std::move(text), index});
    }
    Capture(std::format("BeginEvent '{}'", name));
}

void D3D12CommandList::EndEvent() {
    AIR_CHECK(mEventDepth > 0, "EndEvent without BeginEvent");
    --mEventDepth;
    if (mTimestamps && !mOpenScopes.empty()) {
        OpenScope scope = std::move(mOpenScopes.back());
        mOpenScopes.pop_back();
        const u32 index = scope.begin == ~0u ? ~0u : AllocateTimestamp();
        if (index != ~0u) {
            mList->EndQuery(mTimestamps->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
            mClosedScopes.push_back({std::move(scope.name), u32(mOpenScopes.size()), scope.begin, index});
        }
    }
    mList->EndEvent();
    Capture("EndEvent");
}

void* D3D12CommandList::GetNativeHandle() {
    mRootStateBound = false; // native code may change heaps and root signatures
    mRayTracingPipeline = nullptr;
    return mList;
}

} // namespace air::rhi::d3d12
