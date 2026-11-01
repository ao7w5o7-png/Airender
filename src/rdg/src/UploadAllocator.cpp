#include "air/rdg/UploadAllocator.h"

#include "air/core/Assert.h"

#include <algorithm>
#include <format>
#include <utility>

namespace air::rdg
{

UploadAllocator::UploadAllocator(rhi::Device& device, Config config) : mDevice(device), mConfig(config)
{
    AIR_CHECK(mConfig.pageBytes >= 64 * 1024, "UploadAllocator: pages of {} bytes are too small", mConfig.pageBytes);
    AIR_CHECK(mConfig.location != rhi::MemoryLocation::DeviceLocal && mConfig.location != rhi::MemoryLocation::Readback,
              "UploadAllocator: {} memory is not written by the CPU", rhi::ToString(mConfig.location));
}

UploadAllocator::~UploadAllocator()
{
    for (const Page& page : mPages)
    {
        mDevice.DestroyBuffer(page.buffer); // deferred until the GPU is done with it
    }
}

u32 UploadAllocator::TakePage(u64 bytes)
{
    const bool dedicated = bytes > mConfig.pageBytes;
    if (!dedicated)
    {
        for (u32 i = 0; i < mPages.size(); ++i)
        {
            Page& page = mPages[i];
            if (!page.inUse && !page.dedicated && mDevice.IsFenceComplete(rhi::QueueType::Graphics, page.fence))
            {
                page.inUse = true;
                mFramePages.push_back(i);
                return i;
            }
        }
    }
    Page page;
    page.size = dedicated ? AlignUp(bytes, 64 * 1024) : mConfig.pageBytes;
    page.dedicated = dedicated;
    page.buffer =
        mDevice.CreateBuffer({.size = page.size,
                              .usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::CopySrc,
                              .memory = mConfig.location,
                              .debugName = std::format("Upload{}[{}]", dedicated ? ".Dedicated" : "", mPages.size())});
    AIR_CHECK(page.buffer.IsValid(), "UploadAllocator: out of memory creating a page of {} bytes", page.size);
    page.cpu = static_cast<u8*>(mDevice.GetMappedPointer(page.buffer));
    AIR_CHECK(page.cpu, "UploadAllocator: {} memory is not mapped", rhi::ToString(mConfig.location));
    page.srv = mDevice.GetBindlessSRV(page.buffer);
    page.inUse = true;
    mPages.push_back(page);
    mFramePages.push_back(u32(mPages.size() - 1));
    ++mCounters.pagesCreated;
    return u32(mPages.size() - 1);
}

UploadAllocator::Allocation UploadAllocator::Allocate(u64 size, u64 alignment)
{
    AIR_CHECK(IsPow2(alignment), "UploadAllocator: alignment {} is not a power of two", alignment);
    size = std::max<u64>(size, 1);
    std::lock_guard lock(mMutex);
    u32 index = ~0u;
    u64 offset = 0;
    if (mCurrent != ~0u && AlignUp(mOffset, alignment) + size <= mPages[mCurrent].size)
    {
        index = mCurrent;
        offset = AlignUp(mOffset, alignment);
        mOffset = offset + size;
    }
    else
    {
        index = TakePage(size);
        if (!mPages[index].dedicated)
        {
            mCurrent = index; // what is left of the old page is not worth searching
            mOffset = size;
        }
    }
    ++mCounters.allocations;
    mCounters.bytes += size;
    const Page& page = mPages[index];
    return {page.buffer, offset, size, page.cpu + offset, page.srv};
}

void UploadAllocator::EndFrame(u64 graphicsFence)
{
    std::lock_guard lock(mMutex);
    mCounters.pagesUsed = u32(mFramePages.size());
    for (const u32 index : mFramePages)
    {
        Page& page = mPages[index];
        page.fence = graphicsFence;
        page.lastUsed = mFrame;
        page.inUse = false;
    }
    mFramePages.clear();
    mCurrent = ~0u;
    mOffset = 0;
    // Dedicated pages go after their frame, pages unused for long after retireAfterFrames.
    const auto retire = [&](const Page& page)
    {
        const bool release = page.dedicated || mFrame - page.lastUsed >= mConfig.retireAfterFrames;
        if (release)
        {
            mDevice.DestroyBuffer(page.buffer); // deferred until the GPU is done with it
            ++mCounters.pagesReleased;
        }
        return release;
    };
    mPages.erase(std::remove_if(mPages.begin(), mPages.end(), retire), mPages.end());
    ++mFrame;

    mCounters.pages = u32(mPages.size());
    mCounters.pageBytes = 0;
    for (const Page& page : mPages)
    {
        mCounters.pageBytes += page.size;
    }
    mStats = std::exchange(mCounters, {});
}

UploadAllocator::Stats UploadAllocator::GetStats() const
{
    std::lock_guard lock(mMutex);
    return mStats;
}

} // namespace air::rdg
