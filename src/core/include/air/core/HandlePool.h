#pragma once

// Generational handle pool. Handles are plain values (cheap to copy, safe to log, compare and
// hash) and a stale handle is detected instead of dereferencing freed memory.
// All member functions are thread-safe. Get() returns a pointer that stays valid until the
// handle is freed; the RHI only frees after the GPU is done (deferred deletion).

#include "air/core/Assert.h"
#include "air/core/Types.h"

#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace air
{

template <typename Tag> struct Handle
{
    u32 index = 0;
    u32 generation = 0; // 0 == invalid

    bool IsValid() const { return generation != 0; }
    explicit operator bool() const { return IsValid(); }
    friend bool operator==(Handle, Handle) = default;
};

template <typename Tag, typename T> class HandlePool
{
public:
    using HandleType = Handle<Tag>;

    template <typename... Args> HandleType Allocate(Args&&... args)
    {
        std::lock_guard lock(mMutex);
        u32 index;
        if (!mFreeList.empty())
        {
            index = mFreeList.back();
            mFreeList.pop_back();
        }
        else
        {
            index = u32(mSlots.size());
            mSlots.emplace_back(); // deque: element addresses stay stable on growth
        }
        Slot& slot = mSlots[index];
        slot.value.emplace(std::forward<Args>(args)...);
        ++mLiveCount;
        return {index, slot.generation};
    }

    void Free(HandleType handle)
    {
        std::lock_guard lock(mMutex);
        Slot* slot = Lookup(handle);
        AIR_CHECK(slot, "freeing invalid or stale handle (index {}, gen {})", handle.index, handle.generation);
        slot->value.reset();
        slot->generation = slot->generation + 1 == 0 ? 1 : slot->generation + 1;
        mFreeList.push_back(handle.index);
        --mLiveCount;
    }

    T* Get(HandleType handle)
    {
        std::lock_guard lock(mMutex);
        Slot* slot = Lookup(handle);
        return slot && slot->value.has_value() ? &slot->value.value() : nullptr;
    }

    // Calls fn(value) for a live element while holding the pool lock, so that ForEach sees the
    // change atomically. Returns false for an invalid or stale handle. fn must not call back into
    // this pool.
    template <typename Fn> bool Modify(HandleType handle, Fn&& fn)
    {
        std::lock_guard lock(mMutex);
        Slot* slot = Lookup(handle);
        if (!slot)
        {
            return false;
        }
        fn(*slot->value);
        return true;
    }

    // Calls fn(handle, value) for every live element while holding the pool lock: fn must not
    // call back into this pool.
    template <typename Fn> void ForEach(Fn&& fn) const
    {
        std::lock_guard lock(mMutex);
        for (u32 i = 0; i < mSlots.size(); ++i)
        {
            const Slot& slot = mSlots[i];
            if (slot.value.has_value())
            {
                fn(HandleType{i, slot.generation}, *slot.value);
            }
        }
    }

    u32 LiveCount() const
    {
        std::lock_guard lock(mMutex);
        return mLiveCount;
    }

private:
    struct Slot
    {
        u32 generation = 1;
        std::optional<T> value;
    };

    Slot* Lookup(HandleType handle)
    {
        if (!handle.IsValid() || handle.index >= mSlots.size())
        {
            return nullptr;
        }
        Slot& slot = mSlots[handle.index];
        return slot.generation == handle.generation && slot.value ? &slot : nullptr;
    }

    mutable std::mutex mMutex;
    std::deque<Slot> mSlots;
    std::vector<u32> mFreeList;
    u32 mLiveCount = 0;
};

} // namespace air
