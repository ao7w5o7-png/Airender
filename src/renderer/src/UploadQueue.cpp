#include "air/renderer/UploadQueue.h"

#include "air/core/Assert.h"
#include "air/core/Profiler.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace air::renderer
{

using rhi::Access;
using rhi::Barrier;
using rhi::QueueType;

namespace
{

// Every distinct handle once.
template <typename H> void SortUnique(std::vector<H>& handles)
{
    const auto key = [](H h) { return (u64(h.index) << 32) | h.generation; };
    std::sort(handles.begin(), handles.end(), [&](H a, H b) { return key(a) < key(b); });
    handles.erase(std::unique(handles.begin(), handles.end()), handles.end());
}

} // namespace

UploadQueue::UploadQueue(rhi::Device& device, Config config) : mDevice(device), mConfig(config)
{
    AIR_CHECK(mConfig.stagingBytes >= 1 << 20, "UploadQueue: a staging ring of {} bytes is too small",
              mConfig.stagingBytes);
    mStaging = mDevice.CreateBuffer({.size = mConfig.stagingBytes,
                                     .usage = rhi::BufferUsage::CopySrc,
                                     .memory = rhi::MemoryLocation::Upload,
                                     .debugName = "UploadQueue.Staging"});
    AIR_CHECK(mStaging.IsValid(), "UploadQueue: out of memory creating {} bytes of staging", mConfig.stagingBytes);
    mStagingCpu = static_cast<u8*>(mDevice.GetMappedPointer(mStaging));
    AIR_CHECK(mStagingCpu, "UploadQueue: the staging ring is not mapped");
}

UploadQueue::~UploadQueue()
{
    WaitOnCpu(Submit());
    std::lock_guard lock(mMutex);
    RetireLocked();
    for (const Dedicated& d : mDedicated)
    {
        mDevice.DestroyBuffer(d.buffer);
    }
    mDevice.DestroyBuffer(mStaging);
}

bool UploadQueue::PlaceLocked(u64 size, u64 alignment, u64* offset)
{
    const u64 capacity = mConfig.stagingBytes;
    if (mRegions.empty())
    {
        mHead = 0;
        *offset = 0;
        return size <= capacity;
    }
    const u64 tail = mRegions.front().begin; // the oldest byte still in use
    const u64 at = AlignUp(mHead, alignment);
    if (mHead > tail)
    {
        // In use: [tail, head). Free: [head, capacity) and [0, tail).
        if (at + size <= capacity)
        {
            *offset = at;
            return true;
        }
        *offset = 0;
        return size <= tail;
    }
    // Wrapped, in use: [tail, capacity) and [0, head). Free: [head, tail).
    *offset = at;
    return at + size <= tail;
}

void UploadQueue::RetireLocked()
{
    const auto complete = [&](Ticket ticket)
    { return ticket <= mFences.size() && mDevice.IsFenceComplete(QueueType::Copy, mFences[ticket - 1]); };
    while (!mRegions.empty() && complete(mRegions.front().ticket))
    {
        mRegions.pop_front();
    }
    std::erase_if(mDedicated,
                  [&](const Dedicated& d)
                  {
                      if (!complete(d.ticket))
                      {
                          return false;
                      }
                      mDevice.DestroyBuffer(d.buffer);
                      return true;
                  });
}

u8* UploadQueue::StageLocked(u64 size, u64 alignment, rhi::BufferHandle* buffer, u64* offset)
{
    const Ticket open = mFences.size() + 1;
    if (size > mConfig.stagingBytes)
    {
        const rhi::BufferHandle dedicated = mDevice.CreateBuffer({.size = size,
                                                                  .usage = rhi::BufferUsage::CopySrc,
                                                                  .memory = rhi::MemoryLocation::Upload,
                                                                  .debugName = "UploadQueue.Dedicated"});
        AIR_CHECK(dedicated.IsValid(), "UploadQueue: out of memory staging {} bytes", size);
        mDedicated.push_back({dedicated, open});
        ++mStats.dedicated;
        *buffer = dedicated;
        *offset = 0;
        return static_cast<u8*>(mDevice.GetMappedPointer(dedicated));
    }
    for (;;)
    {
        RetireLocked();
        if (PlaceLocked(size, alignment, offset))
        {
            const Ticket ticket = mFences.size() + 1;
            if (!mRegions.empty() && mRegions.back().ticket == ticket && mRegions.back().end <= *offset)
            {
                mRegions.back().end = *offset + size; // the open batch's region grows
            }
            else
            {
                mRegions.push_back({*offset, *offset + size, ticket});
            }
            mHead = *offset + size;
            *buffer = mStaging;
            return mStagingCpu + *offset;
        }
        // The ring is full: wait for its oldest batch (submitting it first when it is the open one).
        AIR_PROFILE_SCOPE("UploadQueue.Wait");
        const auto start = std::chrono::steady_clock::now();
        const u64 fence = FenceLocked(mRegions.front().ticket);
        mDevice.WaitForFence(QueueType::Copy, fence);
        ++mStats.cpuWaits;
        mStats.waitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
}

UploadQueue::Ticket UploadQueue::AddLocked(const Copy& copy)
{
    if (copy.texture.IsValid())
    {
        mWrittenTextures.push_back(copy.texture);
    }
    else
    {
        mWrittenBuffers.push_back(copy.buffer);
    }
    mPending.push_back(copy);
    mPendingBytes += copy.size;
    mStats.bytes += copy.size;
    ++mStats.copies;
    const Ticket ticket = mFences.size() + 1;
    if (mPendingBytes >= mConfig.submitBytes)
    {
        SubmitLocked();
    }
    return ticket;
}

UploadQueue::Ticket UploadQueue::WriteBuffer(rhi::BufferHandle dst, u64 dstOffset, std::span<const u8> data)
{
    const rhi::BufferDesc* desc = mDevice.GetBufferDesc(dst);
    AIR_CHECK(desc, "UploadQueue::WriteBuffer: invalid buffer");
    AIR_CHECK(Any(desc->usage & rhi::BufferUsage::CopyDst), "UploadQueue: '{}' has no CopyDst usage", desc->debugName);
    AIR_CHECK(!data.empty() && dstOffset + data.size() <= desc->size, "UploadQueue: {} bytes at {} out of '{}'",
              data.size(), dstOffset, desc->debugName);
    std::lock_guard lock(mMutex); // held until the staged bytes are written: their batch cannot go before
    Copy copy{.buffer = dst, .dstOffset = dstOffset, .size = data.size()};
    u8* staged = StageLocked(data.size(), 16, &copy.source, &copy.sourceOffset);
    std::memcpy(staged, data.data(), data.size());
    return AddLocked(copy);
}

UploadQueue::Ticket UploadQueue::WriteTexture(rhi::TextureHandle dst, u32 mip, u32 layer, std::span<const u8> texels)
{
    const rhi::TextureDesc* desc = mDevice.GetTextureDesc(dst);
    AIR_CHECK(desc, "UploadQueue::WriteTexture: invalid texture");
    AIR_CHECK(Any(desc->usage & rhi::TextureUsage::CopyDst), "UploadQueue: '{}' has no CopyDst usage", desc->debugName);
    AIR_CHECK(mip < desc->mipLevels, "UploadQueue: '{}' has no mip {}", desc->debugName, mip);
    const u32 width = std::max(desc->width >> mip, 1u);
    const u32 height = std::max(desc->height >> mip, 1u);
    const u32 slices =
        desc->dimension == rhi::TextureDimension::Tex3D ? std::max(desc->depthOrArraySize >> mip, 1u) : 1u;
    const u64 rowBytes = u64(width) * rhi::FormatBytesPerPixel(desc->format);
    const u64 pitch = rhi::TextureCopyRowPitch(desc->format, width);
    const u64 rows = u64(height) * slices;
    AIR_CHECK(texels.size() == rowBytes * rows, "UploadQueue: '{}' mip {} layer {} takes {} bytes, got {}",
              desc->debugName, mip, layer, rowBytes * rows, texels.size());
    std::lock_guard lock(mMutex);
    Copy copy{.texture = dst, .mip = mip, .layer = layer, .size = pitch * rows};
    // Texture copies read from 512-byte aligned offsets (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT).
    u8* staged = StageLocked(copy.size, 512, &copy.source, &copy.sourceOffset);
    for (u64 row = 0; row < rows; ++row)
    {
        std::memcpy(staged + row * pitch, texels.data() + row * rowBytes, rowBytes);
    }
    return AddLocked(copy);
}

UploadQueue::Ticket UploadQueue::SubmitLocked()
{
    if (mPending.empty())
    {
        return mFences.size();
    }
    AIR_PROFILE_SCOPE("UploadQueue.Submit");
    std::vector<rhi::BufferHandle> buffers;
    std::vector<rhi::TextureHandle> textures;
    for (const Copy& c : mPending)
    {
        if (c.texture.IsValid())
        {
            textures.push_back(c.texture);
        }
        else
        {
            buffers.push_back(c.buffer);
        }
    }
    SortUnique(buffers);
    SortUnique(textures);
    std::vector<Barrier> acquire;
    std::vector<Barrier> release;
    for (const rhi::BufferHandle b : buffers)
    {
        acquire.push_back(Barrier::Buf(b, Access::None, Access::CopyDst));
        release.push_back(Barrier::Buf(b, Access::CopyDst, Access::None));
    }
    for (const rhi::TextureHandle t : textures)
    {
        acquire.push_back(Barrier::Tex(t, Access::None, Access::CopyDst));
        release.push_back(Barrier::Tex(t, Access::CopyDst, Access::None));
    }

    auto cmd = mDevice.CreateCommandList(QueueType::Copy);
    cmd->Begin();
    cmd->BeginEvent("UploadQueue");
    cmd->Barriers(acquire);
    for (const Copy& c : mPending)
    {
        if (c.texture.IsValid())
        {
            cmd->CopyBufferToTexture(c.texture, c.mip, c.layer, c.source, c.sourceOffset);
        }
        else
        {
            cmd->CopyBuffer(c.buffer, c.dstOffset, c.source, c.sourceOffset, c.size);
        }
    }
    cmd->Barriers(release);
    cmd->EndEvent();
    cmd->End();
    rhi::CommandList* lists[] = {cmd.get()};
    mFences.push_back(mDevice.Submit(QueueType::Copy, lists));
    ++mStats.batches;
    mPending.clear();
    mPendingBytes = 0;
    return mFences.size();
}

u64 UploadQueue::FenceLocked(Ticket ticket)
{
    if (ticket == 0)
    {
        return 0;
    }
    if (ticket > mFences.size())
    {
        SubmitLocked();
    }
    AIR_CHECK(ticket <= mFences.size(), "UploadQueue: ticket {} was never handed out", ticket);
    return mFences[ticket - 1];
}

UploadQueue::Ticket UploadQueue::Submit()
{
    std::lock_guard lock(mMutex);
    return SubmitLocked();
}

UploadQueue::Ticket UploadQueue::MakeAvailable(QueueType queue, Access access)
{
    AIR_CHECK(queue != QueueType::Copy && IsAccessLegalOnQueue(access, queue),
              "UploadQueue::MakeAvailable: {} on the {} queue", rhi::ToString(access), rhi::ToString(queue));
    std::lock_guard lock(mMutex);
    const Ticket ticket = SubmitLocked();
    SortUnique(mWrittenBuffers);
    SortUnique(mWrittenTextures);
    std::vector<Barrier> barriers;
    for (const rhi::BufferHandle b : mWrittenBuffers)
    {
        barriers.push_back(Barrier::Buf(b, Access::None, access));
    }
    for (const rhi::TextureHandle t : mWrittenTextures)
    {
        barriers.push_back(Barrier::Tex(t, Access::None, access));
    }
    mWrittenBuffers.clear();
    mWrittenTextures.clear();
    if (barriers.empty())
    {
        return ticket;
    }
    mDevice.QueueWait(queue, QueueType::Copy, mFences[ticket - 1]);
    auto cmd = mDevice.CreateCommandList(queue);
    cmd->Begin();
    cmd->Barriers(barriers);
    cmd->End();
    rhi::CommandList* lists[] = {cmd.get()};
    mDevice.Submit(queue, lists);
    return ticket;
}

bool UploadQueue::IsComplete(Ticket ticket)
{
    std::lock_guard lock(mMutex);
    return ticket == 0 || (ticket <= mFences.size() && mDevice.IsFenceComplete(QueueType::Copy, mFences[ticket - 1]));
}

void UploadQueue::WaitOnCpu(Ticket ticket)
{
    u64 fence = 0;
    {
        std::lock_guard lock(mMutex);
        fence = FenceLocked(ticket);
    }
    if (fence != 0)
    {
        mDevice.WaitForFence(QueueType::Copy, fence);
    }
}

void UploadQueue::WaitOnGpu(QueueType queue, Ticket ticket)
{
    AIR_CHECK(queue != QueueType::Copy, "UploadQueue::WaitOnGpu: the copy queue runs its batches in order anyway");
    std::lock_guard lock(mMutex);
    const u64 fence = FenceLocked(ticket);
    if (fence != 0)
    {
        mDevice.QueueWait(queue, QueueType::Copy, fence);
    }
}

UploadQueue::Stats UploadQueue::GetStats() const
{
    std::lock_guard lock(mMutex);
    return mStats;
}

} // namespace air::renderer
