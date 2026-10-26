// OpenVDB files without the OpenVDB library (renderer/Vdb.h) and the decoders they need
// (core/Compression.h). The zlib streams below were made by zlib itself; the files by a small writer
// that follows the layout Vdb.cpp describes (checked against files from openvdb.org and the Walt
// Disney Animation Studios cloud, which decode to the same sums as an independent reader), in every
// encoding the reader handles.

#include "TestFramework.h"

#include "air/core/Compression.h"
#include "air/renderer/Scene.h"
#include "air/renderer/Transfer.h"
#include "air/renderer/Vdb.h"
#include "air/rhi/RHIDevice.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace air;

namespace
{

// The data the zlib streams hold (zlib_vectors.py made them with Python's zlib): a sentence and
// numbers, and letters of skewed frequencies from a linear congruential generator.
std::vector<u8> Sentences(size_t n)
{
    static const char kText[] = "smoke rises from the brazier, ";
    std::vector<u8> data(n);
    for (size_t i = 0; i < n; ++i)
    {
        data[i] = (i / 30) % 4 == 3 ? u8((i * 37 + i / 3) & 0xFF) : u8(kText[i % 30]);
    }
    return data;
}

std::vector<u8> Letters(size_t n)
{
    static const char kLetters[] = "eeeeettttaaoinsh";
    std::vector<u8> data(n);
    u32 x = 1;
    for (size_t i = 0; i < n; ++i)
    {
        x = (x * 1103515245u + 12345u) & 0x7FFFFFFFu;
        data[i] = u8(kLetters[(x >> 16) % 16]);
    }
    return data;
}

// 70 bytes at level 0: stored blocks.
constexpr u8 kStored[] = {
    0x78, 0x01, 0x01, 0x46, 0x00, 0xb9, 0xff, 0x73, 0x6d, 0x6f, 0x6b, 0x65, 0x20, 0x72, 0x69, 0x73, 0x65,
    0x73, 0x20, 0x66, 0x72, 0x6f, 0x6d, 0x20, 0x74, 0x68, 0x65, 0x20, 0x62, 0x72, 0x61, 0x7a, 0x69, 0x65,
    0x72, 0x2c, 0x20, 0x73, 0x6d, 0x6f, 0x6b, 0x65, 0x20, 0x72, 0x69, 0x73, 0x65, 0x73, 0x20, 0x66, 0x72,
    0x6f, 0x6d, 0x20, 0x74, 0x68, 0x65, 0x20, 0x62, 0x72, 0x61, 0x7a, 0x69, 0x65, 0x72, 0x2c, 0x20, 0x73,
    0x6d, 0x6f, 0x6b, 0x65, 0x20, 0x72, 0x69, 0x73, 0x65, 0x9b, 0x21, 0x19, 0xdd,
};
// 45 bytes at level 1: a fixed Huffman block with a match.
constexpr u8 kFixed[] = {
    0x78, 0x01, 0x2b, 0xce, 0xcd, 0xcf, 0x4e, 0x55, 0x28, 0xca, 0x2c, 0x4e, 0x2d, 0x56,
    0x48, 0x2b, 0xca, 0xcf, 0x55, 0x28, 0xc9, 0x48, 0x55, 0x48, 0x2a, 0x4a, 0xac, 0xca,
    0x4c, 0x2d, 0xd2, 0x51, 0x28, 0x46, 0x95, 0x05, 0x00, 0x84, 0x01, 0x10, 0xc2,
};
// 3000 bytes at level 9: fixed Huffman blocks, long matches.
constexpr u8 kFixedLong[] = {
    0x78, 0xda, 0x2b, 0xce, 0xcd, 0xcf, 0x4e, 0x55, 0x28, 0xca, 0x2c, 0x4e, 0x2d, 0x56, 0x48, 0x2b, 0xca, 0xcf, 0x55,
    0x28, 0xc9, 0x48, 0x55, 0x48, 0x2a, 0x4a, 0xac, 0xca, 0x4c, 0x2d, 0xd2, 0x51, 0x28, 0xa6, 0x40, 0x56, 0xc1, 0x35,
    0x6b, 0xc2, 0xd6, 0x5b, 0x0c, 0xaa, 0x5e, 0x05, 0x53, 0x77, 0x3d, 0x60, 0xd5, 0x0a, 0x28, 0x9d, 0x75, 0xe0, 0x29,
    0x97, 0x41, 0x68, 0xd5, 0x82, 0xa3, 0xaf, 0x04, 0x4c, 0xa3, 0x8a, 0x69, 0x66, 0x2f, 0xc4, 0xfc, 0x86, 0xa5, 0xa7,
    0x3e, 0x88, 0x5a, 0x25, 0xb4, 0xae, 0xba, 0xf0, 0x55, 0xca, 0x21, 0xb5, 0x6b, 0xc3, 0xd5, 0x5f, 0x10, 0xf7, 0x14,
    0x8f, 0xfa, 0x77, 0xd4, 0xbf, 0xa3, 0xfe, 0x1d, 0xf5, 0xef, 0xa8, 0x7f, 0x47, 0xfd, 0x3b, 0xea, 0xdf, 0x51, 0xff,
    0x8e, 0xfa, 0x77, 0xd4, 0xbf, 0x43, 0xd0, 0xbf, 0x00, 0x89, 0x50, 0xa2, 0xba,
};
// 400 skewed letters at level 9: a dynamic Huffman block.
constexpr u8 kDynamic[] = {
    0x78, 0xda, 0x3d, 0x50, 0x4b, 0x0a, 0x44, 0x31, 0x08, 0x3b, 0xab, 0x8b, 0x40, 0xdc, 0xe8, 0xc2, 0xdc, 0x9f, 0x49,
    0xfa, 0x60, 0x0a, 0x2d, 0xad, 0xe6, 0x67, 0x75, 0xd8, 0x5d, 0x2c, 0x34, 0xec, 0x86, 0x40, 0x9c, 0x0f, 0x79, 0x55,
    0xa9, 0xbd, 0xd3, 0x83, 0xdc, 0xa2, 0x4e, 0x5b, 0x70, 0xd5, 0xfb, 0x58, 0x45, 0xba, 0x64, 0xe2, 0xd4, 0xa9, 0xa0,
    0xeb, 0x9a, 0xa3, 0x00, 0x2c, 0x21, 0x73, 0xaa, 0x31, 0xb5, 0x6e, 0x01, 0x23, 0x5a, 0x29, 0x35, 0xe0, 0x62, 0xd3,
    0x0f, 0x61, 0x9f, 0x1a, 0xa5, 0xa3, 0xb6, 0xec, 0x0e, 0x8e, 0x88, 0x25, 0x31, 0x21, 0xcd, 0x2c, 0xe6, 0x8e, 0xa7,
    0x8f, 0x8a, 0x48, 0xec, 0xbb, 0xd0, 0x8e, 0x79, 0xe2, 0x09, 0x89, 0x77, 0xb5, 0x0e, 0xef, 0x2c, 0xb6, 0x44, 0xa4,
    0xdb, 0xe2, 0x6a, 0xcf, 0xf6, 0x67, 0xbe, 0xd5, 0xce, 0xf7, 0x9e, 0x8e, 0x60, 0xb0, 0x7b, 0x36, 0x31, 0xbb, 0x03,
    0x7b, 0x93, 0x74, 0x92, 0xae, 0x65, 0x33, 0xa2, 0xb5, 0xe6, 0xb9, 0x7a, 0x40, 0x4e, 0xe8, 0x86, 0xa3, 0x37, 0x98,
    0xf9, 0xdc, 0x73, 0x88, 0xc9, 0x9b, 0x04, 0xca, 0x0f, 0x99, 0xe6, 0x7c, 0x8a, 0x51, 0x7c, 0x1e, 0xb6, 0x7a, 0x52,
    0x75, 0xf4, 0xcd, 0x1f, 0xba, 0xbc, 0x4f, 0xc0, 0xc0, 0x92, 0xc6, 0x6f, 0xff, 0xfc, 0x0f, 0x1c, 0xdb, 0xa7, 0x19,
};

bool Inflates(std::span<const u8> stream, const std::vector<u8>& expected)
{
    std::vector<u8> out(expected.size());
    std::string error;
    if (!InflateZlib(stream, out, &error))
    {
        std::fprintf(stderr, "  %s\n", error.c_str());
        return false;
    }
    return out == expected;
}

// --- Encoders for the tests: valid streams without compressing much ---------------------------------

void PutLe32(std::vector<u8>& out, u32 v)
{
    for (int i = 0; i < 4; ++i)
    {
        out.push_back(u8(v >> (8 * i)));
    }
}

// An LZ4 block of literals only.
std::vector<u8> Lz4Literals(std::span<const u8> data)
{
    std::vector<u8> out;
    const size_t n = data.size();
    out.push_back(u8(std::min<size_t>(n, 15) << 4));
    if (n >= 15)
    {
        size_t rest = n - 15;
        while (rest >= 255)
        {
            out.push_back(255);
            rest -= 255;
        }
        out.push_back(u8(rest));
    }
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

// A Blosc 1 frame of LZ4 blocks of `blockSize` bytes (the last one shorter), byte shuffled, split
// as c-blosc splits; `stored`: a memcpyed frame, whose shuffle flag must be ignored.
std::vector<u8> BloscFrame(std::span<const u8> data, u32 typeSize, u32 blockSize, bool stored)
{
    const u32 bytes = u32(data.size());
    // c-blosc keeps blocks a multiple of the element size: what is left over makes a last block.
    blockSize = std::min(blockSize, std::max(bytes, 1u));
    if (blockSize > typeSize)
    {
        blockSize = blockSize / typeSize * typeSize;
    }
    std::vector<u8> out = {2, 1, u8(0x01 | (stored ? 0x02 : 0x00) | (1 << 5)), u8(typeSize)};
    PutLe32(out, bytes);
    PutLe32(out, blockSize);
    PutLe32(out, 0); // total, below
    if (stored)
    {
        out.insert(out.end(), data.begin(), data.end());
    }
    else if (bytes > 0)
    {
        const u32 blocks = (bytes + blockSize - 1) / blockSize;
        const size_t starts = out.size();
        out.resize(out.size() + 4 * blocks);
        for (u32 b = 0; b < blocks; ++b)
        {
            const u32 offset = u32(out.size());
            std::memcpy(out.data() + starts + 4 * b, &offset, 4);
            const bool leftover = b == blocks - 1 && bytes % blockSize != 0;
            const u32 size = leftover ? bytes % blockSize : blockSize;
            const std::span<const u8> block = data.subspan(size_t(b) * blockSize, size);
            // Shuffled: byte j of every element together.
            const u32 elements = size / typeSize;
            std::vector<u8> shuffled(block.begin(), block.end());
            for (u32 j = 0; j < typeSize; ++j)
            {
                for (u32 i = 0; i < elements; ++i)
                {
                    shuffled[size_t(j) * elements + i] = block[size_t(i) * typeSize + j];
                }
            }
            const bool split = typeSize <= 16 && size / typeSize >= 128 && !leftover;
            const u32 splits = split ? typeSize : 1;
            for (u32 s = 0; s < splits; ++s)
            {
                const std::span<const u8> part(shuffled.data() + size_t(s) * (size / splits), size / splits);
                // Every other split stored as it is (its size equal to its length).
                const std::vector<u8> lz4 = s % 2 == 1 ? std::vector<u8>(part.begin(), part.end()) : Lz4Literals(part);
                PutLe32(out, u32(lz4.size()));
                out.insert(out.end(), lz4.begin(), lz4.end());
            }
        }
    }
    const u32 total = u32(out.size());
    std::memcpy(out.data() + 12, &total, 4);
    return out;
}

// A zlib stream of stored deflate blocks.
std::vector<u8> ZlibStored(std::span<const u8> data)
{
    std::vector<u8> out = {0x78, 0x01};
    size_t at = 0;
    do
    {
        const size_t n = std::min<size_t>(data.size() - at, 65535);
        const bool last = at + n == data.size();
        out.push_back(last ? 1 : 0);
        out.push_back(u8(n));
        out.push_back(u8(n >> 8));
        out.push_back(u8(~n));
        out.push_back(u8(~n >> 8));
        out.insert(out.end(), data.begin() + std::ptrdiff_t(at), data.begin() + std::ptrdiff_t(at + n));
        at += n;
    } while (at < data.size());
    u32 a = 1;
    u32 b = 0;
    for (u8 byte : data)
    {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }
    const u32 adler = (b << 16) | a;
    for (int i = 3; i >= 0; --i)
    {
        out.push_back(u8(adler >> (8 * i)));
    }
    return out;
}

// --- A small OpenVDB writer -------------------------------------------------------------------------

enum class Packing
{
    Raw,           // no compression flags: every value written
    ActiveRaw,     // active-mask compression only (the Disney cloud)
    ActiveZip,     // zlib, as OpenVDB 2 wrote
    ActiveBlosc,   // Blosc, one block per chunk (OpenVDB 3.3 on)
    ActiveBlosc256 // Blosc in 256-byte blocks (OpenVDB 3.0 to 3.2), with leftover blocks
};

struct Writer
{
    std::vector<u8> bytes;

    void Put(const void* data, size_t n)
    {
        const u8* p = static_cast<const u8*>(data);
        bytes.insert(bytes.end(), p, p + n);
    }
    template <typename T> void Put(T value) { Put(&value, sizeof(T)); }
    void String(std::string_view s)
    {
        Put(u32(s.size()));
        Put(s.data(), s.size());
    }
    void Patch(size_t at, i64 value) { std::memcpy(bytes.data() + at, &value, 8); }
};

// A voxel of the test grid: its value (multiples of 1/16, exact as halves) and whether it is active.
struct Voxel
{
    float value = 0.0f;
    bool active = false;
};

constexpr float kTileValue = 0.75f;     // the active tile over [16, 24) x [0, 8) x [0, 8)
constexpr float kInactiveValue = 0.25f; // the inactive voxels of the partial leaf
constexpr i32 kLeafOrigins[5][3] = {{0, 0, 0}, {0, 8, 0}, {0, 16, 0}, {8, 0, 0}, {8, 8, 0}}; // in buffer order

float Pattern(i32 x, i32 y, i32 z)
{
    return float((3 * x + 5 * y + 7 * z) % 17) / 16.0f;
}

// Leaf voxels: four full leaves over [0, 16) x [0, 16) x [0, 8); the leaf at y 16 has every other
// voxel active.
Voxel LeafVoxel(i32 x, i32 y, i32 z)
{
    if (y >= 16)
    {
        return (x + y + z) % 2 == 0 ? Voxel{Pattern(x, y, z), true} : Voxel{kInactiveValue, false};
    }
    return {Pattern(x, y, z), true};
}

// What the grid should read as: [0, 24) x [0, 24) x [0, 8), x fastest.
std::vector<float> ExpectedDensity()
{
    std::vector<float> d(24 * 24 * 8, 0.0f);
    for (i32 z = 0; z < 8; ++z)
    {
        for (i32 y = 0; y < 24; ++y)
        {
            for (i32 x = 0; x < 24; ++x)
            {
                float v = 0.0f;
                if ((x < 16 && y < 16) || (x < 8 && y >= 16))
                {
                    v = LeafVoxel(x, y, z).value;
                }
                else if (x >= 16 && y < 8)
                {
                    v = kTileValue;
                }
                d[(size_t(z) * 24 + y) * 24 + x] = v;
            }
        }
    }
    return d;
}

struct GridSpec
{
    std::string name = "density";
    std::string valueType = "float";
    std::string gridClass = "fog volume";
    Packing packing = Packing::ActiveBlosc;
    bool half = false;
    bool empty = false; // no tree (only its header, for ReadVdbInfo)
};

class VdbFile
{
public:
    std::vector<u8> Write(const std::vector<GridSpec>& grids)
    {
        mOut.bytes.clear();
        mOut.Put(u64(0x56444220));
        mOut.Put(u32(224));
        mOut.Put(u32(11));
        mOut.Put(u32(0));
        mOut.Put(u8(1)); // grid offsets
        mOut.Put("01234567-89ab-cdef-0123-456789abcdef", 36);
        mOut.Put(u32(1)); // file metadata: one entry
        mOut.String("creator");
        mOut.String("string");
        mOut.String("VdbTests");
        mOut.Put(i32(grids.size()));
        for (const GridSpec& g : grids)
        {
            mGrid = g;
            mOut.String(g.name + "\x1e"
                                 "0");
            mOut.String("Tree_" + g.valueType + "_5_4_3" + (g.half ? "_HalfFloat" : ""));
            mOut.String("");
            const size_t offsets = mOut.bytes.size();
            mOut.Put(i64(0));
            mOut.Put(i64(0));
            mOut.Put(i64(0));
            mOut.Patch(offsets, i64(mOut.bytes.size()));
            WriteGrid(offsets + 8);
            mOut.Patch(offsets + 16, i64(mOut.bytes.size()));
        }
        return mOut.bytes;
    }

private:
    u32 Compression() const
    {
        switch (mGrid.packing)
        {
        case Packing::Raw: return 0;
        case Packing::ActiveRaw: return 2;
        case Packing::ActiveZip: return 3;
        default: return 6;
        }
    }

    void Metadata()
    {
        mOut.Put(u32(4));
        mOut.String("class");
        mOut.String("string");
        mOut.String(mGrid.gridClass);
        mOut.String("file_bbox_min");
        mOut.String("vec3i");
        mOut.Put(u32(12));
        const i32 lo[3] = {0, 0, 0};
        mOut.Put(lo, 12);
        mOut.String("file_bbox_max");
        mOut.String("vec3i");
        mOut.Put(u32(12));
        const i32 hi[3] = {23, 23, 7};
        mOut.Put(hi, 12);
        mOut.String("file_delayed_load"); // an unknown type the reader skips
        mOut.String("__delayedload");
        mOut.Put(u32(3));
        mOut.Put("abc", 3);
    }

    void Transform()
    {
        // UniformScaleTranslateMap: translation, scale and four derived vectors.
        mOut.String("UniformScaleTranslateMap");
        const double translation[3] = {1.0, 2.0, 3.0};
        const double scale[3] = {0.1, 0.1, 0.1};
        const double derived[3] = {10.0, 10.0, 10.0};
        mOut.Put(translation, sizeof(translation));
        mOut.Put(scale, sizeof(scale));
        for (int i = 0; i < 4; ++i)
        {
            mOut.Put(derived, sizeof(derived));
        }
    }

    // A chunk of values as the grid's packing writes it.
    void Chunk(const std::vector<float>& values)
    {
        std::vector<u8> raw;
        if (mGrid.half)
        {
            if (values.empty())
            {
                return; // halves write nothing for no values
            }
            for (float v : values)
            {
                const u16 h = renderer::FloatToHalf(v);
                raw.push_back(u8(h));
                raw.push_back(u8(h >> 8));
            }
        }
        else
        {
            raw.resize(values.size() * 4);
            std::memcpy(raw.data(), values.data(), raw.size());
        }
        const u32 typeSize = 4; // as OpenVDB 3.3 on, for halves too
        switch (mGrid.packing)
        {
        case Packing::Raw:
        case Packing::ActiveRaw: mOut.Put(raw.data(), raw.size()); return;
        case Packing::ActiveZip:
        {
            if (raw.size() < 16)
            {
                mOut.Put(-i64(raw.size())); // not worth compressing: raw, its size negated
                mOut.Put(raw.data(), raw.size());
                return;
            }
            const std::vector<u8> z = ZlibStored(raw);
            mOut.Put(i64(z.size()));
            mOut.Put(z.data(), z.size());
            return;
        }
        default:
        {
            const u32 block = mGrid.packing == Packing::ActiveBlosc256 ? 256u : std::max<u32>(u32(raw.size()), 1u);
            const std::vector<u8> frame = BloscFrame(raw, typeSize, block, raw.size() < 64);
            mOut.Put(i64(frame.size()));
            mOut.Put(frame.data(), frame.size());
            return;
        }
        }
    }

    // writeCompressedValues: `values` of a node, `active` its value mask.
    void Values(const std::vector<float>& values, const std::vector<bool>& active, u8 metadata, float inactive)
    {
        const bool masked = Compression() & 2u;
        mOut.Put(masked ? metadata : u8(6));
        if (masked && metadata == 2)
        {
            mOut.Put(inactive); // full width, also for halves
        }
        std::vector<float> stored;
        for (size_t i = 0; i < values.size(); ++i)
        {
            if (!masked || active[i])
            {
                stored.push_back(values[i]);
            }
        }
        Chunk(stored);
    }

    static void Mask(std::vector<u64>& words, u32 n) { words[n >> 6] |= u64(1) << (n & 63u); }

    void WriteGrid(size_t blockPositionAt)
    {
        mOut.Put(Compression());
        Metadata();
        Transform();
        mOut.Put(i32(1)); // one buffer
        if (mGrid.valueType == "vec3s")
        {
            const float background[3] = {};
            mOut.Put(background, sizeof(background));
        }
        else
        {
            mOut.Put(0.0f); // background
        }
        if (mGrid.empty)
        {
            mOut.Put(u32(0));
            mOut.Put(u32(0));
            mOut.Patch(blockPositionAt, i64(mOut.bytes.size()));
            return;
        }
        mOut.Put(u32(0)); // root tiles
        mOut.Put(u32(1)); // root children
        const i32 origin[3] = {0, 0, 0};
        mOut.Put(origin, sizeof(origin));
        // Node5: its child 0 is the Node4 at the origin; no tiles.
        std::vector<u64> childMask5(512, 0);
        std::vector<u64> valueMask5(512, 0);
        Mask(childMask5, 0);
        mOut.Put(childMask5.data(), 4096);
        mOut.Put(valueMask5.data(), 4096);
        Values(std::vector<float>(32768, 0.0f), std::vector<bool>(32768, false), 0, 0.0f);
        // Node4: five leaves and the tile at child (2, 0, 0).
        std::vector<u64> childMask4(64, 0);
        std::vector<u64> valueMask4(64, 0);
        auto slot = [](i32 x, i32 y, i32 z) { return u32(((x >> 3) << 8) | ((y >> 3) << 4) | (z >> 3)); };
        for (const auto& o : kLeafOrigins)
        {
            Mask(childMask4, slot(o[0], o[1], o[2]));
        }
        Mask(valueMask4, slot(16, 0, 0));
        mOut.Put(childMask4.data(), 512);
        mOut.Put(valueMask4.data(), 512);
        std::vector<float> tiles(4096, 0.0f);
        std::vector<bool> activeTiles(4096, false);
        tiles[slot(16, 0, 0)] = kTileValue;
        activeTiles[slot(16, 0, 0)] = true;
        Values(tiles, activeTiles, 0, 0.0f);
        // The leaves' masks, in the order of their slots (the order of kLeafOrigins).
        std::vector<std::vector<u64>> leafMasks;
        for (const auto& o : kLeafOrigins)
        {
            std::vector<u64> mask(8, 0);
            for (u32 n = 0; n < 512; ++n)
            {
                if (LeafVoxel(o[0] + i32(n >> 6), o[1] + i32((n >> 3) & 7u), o[2] + i32(n & 7u)).active)
                {
                    Mask(mask, n);
                }
            }
            mOut.Put(mask.data(), 64);
            leafMasks.push_back(mask);
        }
        mOut.Patch(blockPositionAt, i64(mOut.bytes.size()));
        // Buffers: each leaf's mask again, then its values.
        for (size_t l = 0; l < std::size(kLeafOrigins); ++l)
        {
            const auto& o = kLeafOrigins[l];
            mOut.Put(leafMasks[l].data(), 64);
            std::vector<float> values(512);
            std::vector<bool> active(512);
            for (u32 n = 0; n < 512; ++n)
            {
                const Voxel v = LeafVoxel(o[0] + i32(n >> 6), o[1] + i32((n >> 3) & 7u), o[2] + i32(n & 7u));
                values[n] = v.value;
                active[n] = v.active;
            }
            const bool partial = o[1] == 16;
            Values(values, active, partial ? 2 : 0, kInactiveValue);
        }
    }

    Writer mOut;
    GridSpec mGrid;
};

std::filesystem::path WriteTemporary(const std::string& name, const std::vector<u8>& bytes)
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / ("air_vdb_test_" + name + ".vdb");
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    return path;
}

} // namespace

TEST(Compression_InflateDecodesZlibStreams)
{
    EXPECT(Inflates(kStored, Sentences(70)));
    EXPECT(Inflates(kFixed, Sentences(45)));
    EXPECT(Inflates(kFixedLong, Sentences(3000)));
    EXPECT(Inflates(kDynamic, Letters(400)));
    // Stored blocks longer than one block, as the test's own encoder writes them.
    const std::vector<u8> large = Letters(150000);
    EXPECT(Inflates(ZlibStored(large), large));
}

TEST(Compression_RejectsCorruptStreams)
{
    std::string error;
    std::vector<u8> out(400);
    // Truncated, a changed byte (the checksum or the codes catch it), the wrong expected size.
    EXPECT(!InflateZlib(std::span(kDynamic, sizeof(kDynamic) - 9), out, &error) && !error.empty());
    std::vector<u8> changed(std::begin(kDynamic), std::end(kDynamic));
    changed[60] ^= 0x10;
    EXPECT(!InflateZlib(changed, out, &error));
    std::vector<u8> small(399);
    EXPECT(!InflateZlib(kDynamic, small, &error));
    const u8 notZlib[] = {0x12, 0x34, 0, 0, 0, 0, 0, 0};
    EXPECT(!InflateZlib(notZlib, out, &error));
    // LZ4: an offset before the start, a match beyond the output.
    const u8 badOffset[] = {0x14, 'a', 0x05, 0x00};
    std::vector<u8> lz(9);
    EXPECT(!DecompressLz4Block(badOffset, lz, &error));
    const u8 tooLong[] = {0x1F, 'a', 0x01, 0x00, 0x40};
    EXPECT(!DecompressLz4Block(tooLong, lz, &error));
    // Blosc: a frame for another size.
    const std::vector<u8> data = Letters(1000);
    const std::vector<u8> frame = BloscFrame(data, 4, 1000, false);
    std::vector<u8> wrong(999);
    EXPECT(!DecompressBlosc(frame, wrong, &error) && !error.empty());
}

TEST(Compression_Lz4AndBloscFrames)
{
    // LZ4: literals, then an overlapping match ("abc" repeated), then literals with a long length.
    const u8 block[] = {0x35, 'a', 'b', 'c', 0x03, 0x00, 0xF0, 0x02, 'x', 'x', 'x', 'x', 'x',
                        'x',  'x', 'x', 'x', 'x',  'x',  'x',  'x',  'x', 'x', 'x', 'x'};
    std::vector<u8> out(12 + 17);
    std::string error;
    EXPECT(DecompressLz4Block(block, out, &error));
    EXPECT(std::string(out.begin(), out.end()) == "abcabcabcabc" + std::string(17, 'x'));
    // Blosc: stored, one block split into byte streams, blocks of 256 bytes with a leftover, for
    // elements of 4 and 2 bytes, and an empty frame.
    for (u32 typeSize : {4u, 2u})
    {
        for (size_t size : {size_t(40), size_t(2048), size_t(1000), size_t(1002)})
        {
            const std::vector<u8> data = Letters(size);
            for (u32 blockSize : {u32(size), 256u})
            {
                for (bool stored : {false, true})
                {
                    std::vector<u8> decoded(size);
                    const bool ok = DecompressBlosc(BloscFrame(data, typeSize, blockSize, stored), decoded, &error);
                    if (!ok || decoded != data)
                    {
                        std::fprintf(stderr, "  blosc %zu bytes, type %u, blocks %u, stored %d: %s\n", size, typeSize,
                                     blockSize, int(stored), ok ? "wrong data" : error.c_str());
                    }
                    EXPECT(ok && decoded == data);
                }
            }
        }
    }
    const u8 empty[16] = {2, 1, 0x21, 4, 0, 0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0};
    std::vector<u8> none;
    EXPECT(DecompressBlosc(empty, none, &error));
}

// Every encoding reads to the same voxels, in the same place.
TEST(Vdb_ReadsEveryEncoding)
{
    const std::vector<float> expected = ExpectedDensity();
    for (Packing packing :
         {Packing::Raw, Packing::ActiveRaw, Packing::ActiveZip, Packing::ActiveBlosc, Packing::ActiveBlosc256})
    {
        for (bool half : {false, true})
        {
            VdbFile file;
            const std::filesystem::path path =
                WriteTemporary("encoding", file.Write({{.packing = packing, .half = half}}));
            std::string error;
            const std::optional<renderer::VolumeGrid> grid = renderer::ReadVdbGrid(path, "density", 256, &error);
            if (!grid)
            {
                std::fprintf(stderr, "  packing %d, half %d: %s\n", int(packing), int(half), error.c_str());
            }
            EXPECT(grid.has_value());
            if (!grid)
            {
                continue;
            }
            EXPECT(grid->sizeX == 24u && grid->sizeY == 24u && grid->sizeZ == 8u);
            // Voxel (0, 0, 0)'s centre is at the translation; the box starts half a voxel before.
            EXPECT(std::abs(grid->voxelSize - 0.1f) < 1e-7f);
            EXPECT(std::abs(grid->origin.x - 0.95f) < 1e-6f && std::abs(grid->origin.y - 1.95f) < 1e-6f &&
                   std::abs(grid->origin.z - 2.95f) < 1e-6f);
            size_t wrong = 0;
            for (size_t i = 0; i < expected.size() && i < grid->density.size(); ++i)
            {
                wrong += grid->density[i] != expected[i] ? 1u : 0u;
            }
            if (wrong != 0)
            {
                std::fprintf(stderr, "  packing %d, half %d: %zu voxels differ\n", int(packing), int(half), wrong);
            }
            EXPECT(wrong == 0);
            std::filesystem::remove(path);
        }
    }
}

// A grid larger than maxResolution is averaged over blocks of voxels while it is read: the same as
// averaging the full grid down.
TEST(Vdb_AveragesLargeGridsDown)
{
    VdbFile file;
    const std::filesystem::path path = WriteTemporary("average", file.Write({{}}));
    std::string error;
    const std::optional<renderer::VolumeGrid> full = renderer::ReadVdbGrid(path, "", 256, &error);
    EXPECT(full.has_value());
    for (u32 limit : {12u, 8u, 5u})
    {
        const std::optional<renderer::VolumeGrid> coarse = renderer::ReadVdbGrid(path, "", limit, &error);
        EXPECT(coarse.has_value());
        if (!full || !coarse)
        {
            continue;
        }
        const u32 factor = (24 + limit - 1) / limit;
        const renderer::VolumeGrid reference = renderer::Downsample(*full, factor);
        EXPECT(coarse->sizeX == reference.sizeX && coarse->sizeY == reference.sizeY &&
               coarse->sizeZ == reference.sizeZ);
        EXPECT(std::max({coarse->sizeX, coarse->sizeY, coarse->sizeZ}) <= limit);
        EXPECT(std::abs(coarse->voxelSize - reference.voxelSize) < 1e-6f);
        double largest = 0.0;
        for (size_t i = 0; i < reference.density.size() && i < coarse->density.size(); ++i)
        {
            largest = std::max(largest, double(std::abs(coarse->density[i] - reference.density[i])));
        }
        EXPECT(largest < 1e-6);
    }
    std::filesystem::remove(path);
}

TEST(Vdb_ListsGridsAndSaysWhyOneCannotBeRead)
{
    VdbFile file;
    const std::filesystem::path path = WriteTemporary(
        "list", file.Write({{.name = "velocity", .valueType = "vec3s", .gridClass = "unknown", .empty = true},
                            {.name = "surface", .gridClass = "level set", .empty = true},
                            {.name = "smoke", .packing = Packing::ActiveZip, .half = true}}));
    std::string error;
    const std::optional<renderer::VdbFileInfo> info = renderer::ReadVdbInfo(path, &error);
    EXPECT(info.has_value());
    if (info)
    {
        EXPECT(info->version == 224u && info->libraryMajor == 11u && info->grids.size() == 3);
        if (info->grids.size() == 3)
        {
            const renderer::VdbGridInfo& velocity = info->grids[0];
            EXPECT(velocity.name == "velocity" && velocity.valueType == "vec3s" && velocity.nodes == "5_4_3");
            EXPECT(!velocity.readable && velocity.why.find("vec3s") != std::string::npos);
            EXPECT(!info->grids[1].readable && info->grids[1].why.find("level set") != std::string::npos);
            const renderer::VdbGridInfo& smoke = info->grids[2];
            EXPECT(smoke.readable && smoke.halfFloat && smoke.gridClass == "fog volume");
            EXPECT(smoke.hasBounds && smoke.boundsMax[0] == 23 && smoke.boundsMax[2] == 7);
            EXPECT(smoke.map == "UniformScaleTranslateMap" && smoke.voxelSize[1] == 0.1 && smoke.translation[2] == 3.0);
        }
    }
    // No grid named "density": the first readable one stands in. Another name must exist.
    EXPECT(renderer::ReadVdbGrid(path, "density", 256, &error).has_value());
    EXPECT(!renderer::ReadVdbGrid(path, "temperature", 256, &error).has_value());
    EXPECT(error.find("no grid named 'temperature'") != std::string::npos && error.find("smoke") != std::string::npos);
    EXPECT(!renderer::ReadVdbGrid(path, "velocity", 256, &error).has_value() &&
           error.find("vec3s") != std::string::npos);
    EXPECT(!renderer::ReadVdbGrid(path, "surface", 256, &error).has_value() &&
           error.find("level set") != std::string::npos);
    // Not a file of this format, and one cut short.
    const std::vector<u8> bytes = file.Write({{}});
    const std::filesystem::path text = WriteTemporary("text", std::vector<u8>(64, u8('x')));
    EXPECT(!renderer::ReadVdbInfo(text, &error).has_value() && error.find("not an OpenVDB file") != std::string::npos);
    const std::filesystem::path cut =
        WriteTemporary("cut", std::vector<u8>(bytes.begin(), bytes.begin() + std::ptrdiff_t(bytes.size() - 100)));
    EXPECT(!renderer::ReadVdbGrid(cut, "density", 256, &error).has_value() && !error.empty());
    EXPECT(!renderer::ReadVdbInfo("does/not/exist.vdb", &error).has_value() &&
           error.find("cannot open") != std::string::npos);
    for (const auto& p : {path, text, cut})
    {
        std::filesystem::remove(p);
    }
}

// A volume made from a file: BuildVolume reads it at its maxResolution and places it.
TEST(Vdb_VolumeFromAFile)
{
    VdbFile file;
    const std::filesystem::path path = WriteTemporary("volume", file.Write({{}}));
    const renderer::VolumeResources volume = renderer::BuildVolume(
        {.enabled = true, .file = path.string(), .position = {0.0f, 1.0f, 0.0f}, .size = 4.8f, .maxResolution = 12});
    EXPECT(volume.error.empty() && volume.data.enabled == 1u);
    EXPECT(volume.data.sizeX == 12u && volume.data.sizeY == 12u && volume.data.sizeZ == 4u);
    EXPECT(std::abs(volume.data.voxelSize - 0.4f) < 1e-6f); // 4.8 m over 12 voxels
    EXPECT(std::abs(volume.data.origin.x + 2.4f) < 1e-5f && std::abs(volume.data.origin.y - 1.0f) < 1e-6f);
    std::filesystem::remove(path);
}

// A scene keeps the voxels it read: edits of the optics or the placement do not read the file again
// (here it is gone by then), edits of where the voxels come from do.
TEST(Vdb_SceneReadsTheFileOnlyWhenItsSourceChanges)
{
    auto device = rhi::CreateDevice({.backend = rhi::Backend::Null});
    {
        VdbFile file;
        const std::filesystem::path path = WriteTemporary("scene", file.Write({{}}));
        renderer::Scene scene(*device, *renderer::MakeBuiltinScene("smoke"));
        renderer::Volume volume = scene.Desc().volumes[0];
        volume.file = path.string();
        scene.SetVolume(0, volume);
        scene.Update();
        EXPECT(scene.Get().volumes.size() == 1 && scene.Get().volumes[0].sizeX == 24u && scene.VolumeError(0).empty());
        std::filesystem::remove(path);
        volume.extinction *= 2.0f;
        volume.position.y += 1.0f;
        scene.SetVolume(0, volume);
        scene.Update();
        EXPECT(scene.Get().volumes.size() == 1 && scene.VolumeError(0).empty());
        EXPECT(scene.Get().volumes[0].extinction == volume.extinction);
        // A second volume from the same file shares its voxels, which are not read again either.
        scene.SetVolumes({volume, volume});
        scene.Update();
        EXPECT(scene.Get().volumes.size() == 2 && scene.VolumeError(1).empty());
        volume.maxResolution = 12; // read again: the file is gone
        scene.SetVolume(1, volume);
        scene.Update();
        EXPECT(scene.Get().volumes.size() == 1 && scene.VolumeError(0).empty() &&
               scene.VolumeError(1).find("cannot open") != std::string::npos);
    }
    device->WaitIdle();
    EXPECT(device->GetMemoryStats().liveBuffers == 0 && device->GetMemoryStats().liveTextures == 0);
}
