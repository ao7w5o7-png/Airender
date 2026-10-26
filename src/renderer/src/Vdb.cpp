// The OpenVDB file format, as the OpenVDB library writes it (io/Archive.cc, io/Compression.h,
// tree/RootNode.h, tree/InternalNode.h, tree/LeafNode.h, math/Maps.h):
//
//   header      u64 magic 0x56444220, u32 format version, u32 x 2 library version, u8 "has grid
//               offsets", 36 characters of UUID, the file's metadata, i32 grid count
//   descriptor  per grid: its unique name ("name" + "\x1e" + suffix), its type ("Tree_float_5_4_3",
//               "_HalfFloat" appended when its values are stored as halves), the name of the grid it
//               is an instance of, and i64 offsets of its data, its buffers and its end
//   grid        u32 compression flags, metadata, transform (a map type name and its numbers),
//               topology, buffers
//   metadata    u32 count; per entry its name, type name, u32 byte count and value
//   topology    i32 buffer count (1); root: background, u32 tile count, u32 child count, tiles
//               (i32 x 3 origin, value, u8 active), children (i32 x 3 origin, then the node);
//               internal node: child mask, value mask, its tiles' values (compressed, below), its
//               children in the masks' order; leaf: value mask
//   buffers     per leaf, in the topology's order: value mask, then its 512 values (compressed)
//
// Masks are 2^(3 log2dim) bits in u64 words; entry n of a node of 2^log2dim per side sits at
// x = n >> 2 log2dim, y = (n >> log2dim) & (dim - 1), z = n & (dim - 1). A node's values are
// written by writeCompressedValues: a byte of metadata saying which inactive values are left out
// (background, -background, one or two others written after it as full values, and a mask choosing
// between two), then with the active-mask flag only the active values, as one chunk: raw, or an i64
// size followed by a zlib stream or a Blosc frame (a size <= 0: -size raw bytes).

#include "air/renderer/Vdb.h"

#include "air/core/Compression.h"
#include "air/renderer/Transfer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <span>
#include <string_view>

namespace air::renderer
{
namespace
{

constexpr u64 kMagic = 0x56444220;
constexpr u32 kOldestVersion = 222; // per-node metadata of compressed values (OpenVDB 2.0)
constexpr u32 kCompressZip = 0x1;
constexpr u32 kCompressActiveMask = 0x2;
constexpr u32 kCompressBlosc = 0x4;

// Which inactive values a node's mask-compressed values leave out (io/Compression.h).
enum : u8
{
    NoMaskOrInactiveValues,   // all inactive values are the background
    NoMaskAndMinusBackground, // all are -background
    NoMaskAndOneInactiveValue,
    MaskAndNoInactiveValues, // a mask chooses between -background and background
    MaskAndOneInactiveValue, // between background and one other value
    MaskAndTwoInactiveValues,
    NoMaskAndAllValues, // all values are written
};

// Little-endian reading with a sticky error: after the first failure every read gives zeros and
// the first message stays.
class Stream
{
public:
    explicit Stream(const std::filesystem::path& path) : mFile(path, std::ios::binary), mPath(path.string())
    {
        if (!mFile)
        {
            Fail("cannot open the file");
        }
    }

    bool Ok() const { return mError.empty(); }
    const std::string& Error() const { return mError; }
    void Fail(const std::string& message)
    {
        if (mError.empty())
        {
            mError = std::format("{}: {}", mPath, message);
        }
    }

    void Read(void* data, size_t bytes)
    {
        if (Ok())
        {
            mFile.read(static_cast<char*>(data), std::streamsize(bytes));
            if (mFile)
            {
                return;
            }
            Fail("unexpected end of file");
        }
        std::memset(data, 0, bytes);
    }
    template <typename T> T Read()
    {
        T value{};
        Read(&value, sizeof(T));
        return value;
    }
    std::string ReadString()
    {
        const u32 length = Read<u32>();
        if (length > (1u << 20))
        {
            Fail("corrupt string");
            return {};
        }
        std::string text(length, '\0');
        Read(text.data(), length);
        return text;
    }
    void Skip(u64 bytes)
    {
        if (Ok() && !mFile.seekg(std::streamoff(bytes), std::ios::cur))
        {
            Fail("unexpected end of file");
        }
    }
    void Seek(i64 position)
    {
        if (Ok() && (position < 0 || !mFile.seekg(std::streamoff(position))))
        {
            Fail("corrupt grid offset");
        }
    }

private:
    std::ifstream mFile;
    std::string mPath;
    std::string mError;
};

struct Descriptor
{
    std::string name; // without its unique suffix
    std::string type; // without "_HalfFloat"
    bool half = false;
    std::string instanceParent;
    i64 gridPosition = 0;
    i64 endPosition = 0;
    std::string valueType; // "float", from "Tree_float_5_4_3"
    std::string nodes;     // "5_4_3"
};

struct Header
{
    u32 version = 0;
    u32 libraryMajor = 0;
    u32 libraryMinor = 0;
    std::vector<Descriptor> grids;
};

struct Metadata
{
    std::string gridClass;
    bool hasMin = false;
    bool hasMax = false;
    i32 boundsMin[3] = {};
    i32 boundsMax[3] = {};
    u64 activeVoxels = 0;
};

void ReadMetadata(Stream& s, Metadata* out)
{
    const u32 count = s.Read<u32>();
    if (count > (1u << 16))
    {
        s.Fail("corrupt metadata");
        return;
    }
    std::vector<u8> value;
    for (u32 i = 0; i < count && s.Ok(); ++i)
    {
        const std::string name = s.ReadString();
        const std::string type = s.ReadString();
        const u32 bytes = s.Read<u32>();
        if (bytes > (1u << 24))
        {
            s.Fail("corrupt metadata");
            return;
        }
        value.resize(bytes);
        s.Read(value.data(), bytes);
        if (!out)
        {
            continue;
        }
        if (name == "class" && type == "string")
        {
            out->gridClass.assign(value.begin(), value.end());
        }
        else if ((name == "file_bbox_min" || name == "file_bbox_max") && type == "vec3i" && bytes == 12)
        {
            const bool min = name == "file_bbox_min";
            std::memcpy(min ? out->boundsMin : out->boundsMax, value.data(), 12);
            (min ? out->hasMin : out->hasMax) = true;
        }
        else if (name == "file_voxel_count" && type == "int64" && bytes == 8)
        {
            std::memcpy(&out->activeVoxels, value.data(), 8);
        }
    }
}

bool ReadHeader(Stream& s, Header& header)
{
    if (s.Read<u64>() != kMagic)
    {
        s.Fail("not an OpenVDB file");
        return false;
    }
    header.version = s.Read<u32>();
    if (s.Ok() && header.version < kOldestVersion)
    {
        s.Fail(std::format("file format version {} predates {} (OpenVDB 2.0), the oldest this reader reads",
                           header.version, kOldestVersion));
        return false;
    }
    header.libraryMajor = s.Read<u32>();
    header.libraryMinor = s.Read<u32>();
    const bool offsets = s.Read<u8>() != 0;
    s.Skip(36); // the file's UUID as text
    ReadMetadata(s, nullptr);
    if (s.Ok() && !offsets)
    {
        s.Fail("written as a stream, without the grid offsets this reader needs");
        return false;
    }
    const i32 count = s.Read<i32>();
    if (count < 0 || count > (1 << 16))
    {
        s.Fail("corrupt grid count");
        return false;
    }
    for (i32 i = 0; i < count && s.Ok(); ++i)
    {
        Descriptor d;
        const std::string unique = s.ReadString();
        d.name = unique.substr(0, unique.find('\x1e'));
        d.type = s.ReadString();
        constexpr std::string_view kHalf = "_HalfFloat";
        if (d.type.ends_with(kHalf))
        {
            d.half = true;
            d.type.resize(d.type.size() - kHalf.size());
        }
        d.instanceParent = s.ReadString();
        d.gridPosition = s.Read<i64>();
        s.Read<i64>(); // where its buffers start
        d.endPosition = s.Read<i64>();
        // "Tree_float_5_4_3": the value type, then the nodes' log2 sizes from the root's children down.
        if (d.type.starts_with("Tree_"))
        {
            const size_t start = 5;
            size_t split = d.type.find('_', start);
            while (split != std::string::npos && split + 1 < d.type.size() &&
                   !(d.type[split + 1] >= '0' && d.type[split + 1] <= '9'))
            {
                split = d.type.find('_', split + 1);
            }
            d.valueType = d.type.substr(start, split == std::string::npos ? std::string::npos : split - start);
            d.nodes = split == std::string::npos ? std::string() : d.type.substr(split + 1);
        }
        header.grids.push_back(std::move(d));
        s.Seek(header.grids.back().endPosition);
    }
    return s.Ok();
}

struct Transform
{
    std::string map;
    double scale[3] = {1.0, 1.0, 1.0};
    double translation[3] = {};
    std::string unsupported; // why this reader cannot place the voxels (empty: it can)
};

void ReadTransform(Stream& s, Transform& t)
{
    t.map = s.ReadString();
    auto vec3 = [&](double* v) { s.Read(v, 3 * sizeof(double)); };
    double derived[3];
    if (t.map == "ScaleMap" || t.map == "UniformScaleMap")
    {
        vec3(t.scale);
        for (int i = 0; i < 4; ++i)
        {
            vec3(derived); // voxel size, 1 / scale, 1 / scale^2, 1 / (2 scale)
        }
    }
    else if (t.map == "ScaleTranslateMap" || t.map == "UniformScaleTranslateMap")
    {
        vec3(t.translation);
        vec3(t.scale);
        for (int i = 0; i < 4; ++i)
        {
            vec3(derived);
        }
    }
    else if (t.map == "TranslationMap")
    {
        vec3(t.translation);
    }
    else if (t.map == "AffineMap" || t.map == "UnitaryMap")
    {
        // A 4 x 4 matrix applied to row vectors: world = (index, 1) * M, the translation in its last row.
        double m[16];
        s.Read(m, sizeof(m));
        t.scale[0] = m[0];
        t.scale[1] = m[5];
        t.scale[2] = m[10];
        t.translation[0] = m[12];
        t.translation[1] = m[13];
        t.translation[2] = m[14];
        const double largest = std::max({std::abs(m[0]), std::abs(m[5]), std::abs(m[10])});
        for (int i : {1, 2, 4, 6, 8, 9})
        {
            if (std::abs(m[i]) > 1e-9 * largest)
            {
                t.unsupported = std::format("its {} rotates or shears the voxels", t.map);
            }
        }
    }
    else
    {
        // Its size is unknown: the grid cannot be read on.
        t.unsupported = std::format("its transform '{}' is not supported", t.map);
        return;
    }
    if (t.unsupported.empty())
    {
        const double s0 = t.scale[0];
        if (!(s0 > 0.0) || std::abs(t.scale[1] - s0) > 1e-5 * s0 || std::abs(t.scale[2] - s0) > 1e-5 * s0)
        {
            t.unsupported = std::format("its voxels are not cubes ({} x {} x {})", t.scale[0], t.scale[1], t.scale[2]);
        }
    }
}

// What a grid's values look like in the file.
struct Format
{
    u32 compression = 0;
    u32 valueSize = 4; // half 2, float 4, double 8: tiles, the background and inactive values
    bool half = false; // "_HalfFloat": the nodes' values converted to halves
    double background = 0.0;
};

std::string WhyUnreadable(const Descriptor& d)
{
    if (d.valueType != "float" && d.valueType != "double" && d.valueType != "half")
    {
        return std::format("it holds {} values, not scalars", d.valueType.empty() ? d.type : d.valueType);
    }
    if (d.nodes != "5_4_3")
    {
        return std::format("its tree has {} nodes, not 5_4_3", d.nodes);
    }
    return {};
}

double ReadScalar(Stream& s, const Format& f)
{
    if (f.valueSize == 2)
    {
        return double(HalfToFloat(s.Read<u16>()));
    }
    return f.valueSize == 8 ? s.Read<double>() : double(s.Read<float>());
}

// One chunk of `out.size()` bytes: raw, or an i64 size and a zlib stream or Blosc frame.
bool ReadChunk(Stream& s, u32 compression, std::span<u8> out, std::vector<u8>& packed)
{
    if ((compression & (kCompressBlosc | kCompressZip)) == 0)
    {
        s.Read(out.data(), out.size());
        return s.Ok();
    }
    const i64 size = s.Read<i64>();
    if (size <= 0)
    {
        if (u64(-size) != out.size())
        {
            s.Fail(std::format("a chunk of {} bytes where {} were expected", -size, out.size()));
            return false;
        }
        s.Read(out.data(), out.size());
        return s.Ok();
    }
    // Compressed data a little larger than raw (incompressible blocks) is possible; much larger is not.
    if (u64(size) > out.size() + out.size() / 8 + 1024)
    {
        s.Fail(std::format("a compressed chunk of {} bytes for {} bytes of values", size, out.size()));
        return false;
    }
    packed.resize(size_t(size));
    s.Read(packed.data(), packed.size());
    std::string error;
    const bool ok =
        (compression & kCompressBlosc) != 0 ? DecompressBlosc(packed, out, &error) : InflateZlib(packed, out, &error);
    if (!ok)
    {
        s.Fail(error);
    }
    return s.Ok();
}

bool Bit(const u64* mask, u32 n)
{
    return ((mask[n >> 6] >> (n & 63u)) & 1u) != 0;
}

// A node's `count` values (readCompressedValues), as floats.
bool ReadValues(Stream& s, const Format& f, const u64* valueMask, u32 count, float* values, std::vector<u8>& raw,
                std::vector<u8>& packed)
{
    const u8 metadata = s.Read<u8>();
    if (metadata > NoMaskAndAllValues)
    {
        s.Fail("corrupt node metadata");
        return false;
    }
    double inactive0 = metadata == NoMaskOrInactiveValues ? f.background : -f.background;
    double inactive1 = f.background;
    if (metadata == NoMaskAndOneInactiveValue || metadata == MaskAndOneInactiveValue ||
        metadata == MaskAndTwoInactiveValues)
    {
        inactive0 = ReadScalar(s, f);
        if (metadata == MaskAndTwoInactiveValues)
        {
            inactive1 = ReadScalar(s, f);
        }
    }
    u64 selection[512] = {}; // chooses inactive1 over inactive0
    if (metadata == MaskAndNoInactiveValues || metadata == MaskAndOneInactiveValue ||
        metadata == MaskAndTwoInactiveValues)
    {
        s.Read(selection, count / 8);
    }
    const bool maskCompressed = (f.compression & kCompressActiveMask) != 0 && metadata != NoMaskAndAllValues;
    u32 stored = count;
    if (maskCompressed)
    {
        stored = 0;
        for (u32 w = 0; w < count / 64; ++w)
        {
            stored += u32(std::popcount(valueMask[w]));
        }
    }
    const u32 size = f.half ? 2u : f.valueSize;
    raw.resize(size_t(stored) * size);
    // Values converted to halves write nothing at all when there are none (not even a chunk size).
    if (!(f.half && stored == 0) && !ReadChunk(s, f.compression, raw, packed))
    {
        return false;
    }
    auto value = [&](u32 i) -> float
    {
        const u8* p = raw.data() + size_t(i) * size;
        if (size == 2)
        {
            u16 h;
            std::memcpy(&h, p, 2);
            return HalfToFloat(h);
        }
        if (size == 8)
        {
            double d;
            std::memcpy(&d, p, 8);
            return float(d);
        }
        float v;
        std::memcpy(&v, p, 4);
        return v;
    };
    if (stored == count)
    {
        for (u32 i = 0; i < count; ++i)
        {
            values[i] = value(i);
        }
        return true;
    }
    for (u32 i = 0, k = 0; i < count; ++i)
    {
        values[i] = Bit(valueMask, i) ? value(k++) : float(Bit(selection, i) ? inactive1 : inactive0);
    }
    return true;
}

// Index bounds, in 64 bits: a tile next to the coordinates' limits must not overflow.
struct Box
{
    i64 lo[3] = {INT64_MAX, INT64_MAX, INT64_MAX};
    i64 hi[3] = {INT64_MIN, INT64_MIN, INT64_MIN};

    bool Empty() const { return lo[0] > hi[0]; }
    void Add(i32 x, i32 y, i32 z, i32 extent)
    {
        const i64 p[3] = {x, y, z};
        for (int k = 0; k < 3; ++k)
        {
            lo[k] = std::min(lo[k], p[k]);
            hi[k] = std::max(hi[k], p[k] + extent - 1);
        }
    }
};

struct Tile
{
    i32 origin[3];
    i32 extent; // voxels per side
    float value;
};

struct Leaf
{
    i32 origin[3];
};

// The tree's topology: its leaves in the order their buffers follow, its active tiles and the
// bounds of everything active.
struct Topology
{
    std::vector<Leaf> leaves;
    std::vector<Tile> tiles;
    Box bounds;
};

// An internal node of 2^log2 children per side, each 2^childLog2 voxels wide; its children are
// internal nodes of log2 4 or leaves.
void ReadInternal(Stream& s, const Format& f, u32 log2, u32 childLog2, const i32 origin[3], Topology& t,
                  std::vector<float>& values, std::vector<u8>& raw, std::vector<u8>& packed)
{
    const u32 count = 1u << (3 * log2);
    std::vector<u64> masks(count / 64 * 2);
    u64* childMask = masks.data();
    u64* valueMask = masks.data() + count / 64;
    s.Read(childMask, count / 8);
    s.Read(valueMask, count / 8);
    values.resize(count);
    if (!ReadValues(s, f, valueMask, count, values.data(), raw, packed))
    {
        return;
    }
    const i32 childExtent = 1 << childLog2;
    auto childOrigin = [&](u32 n, i32* o)
    {
        o[0] = origin[0] + i32(n >> (2 * log2)) * childExtent;
        o[1] = origin[1] + i32((n >> log2) & ((1u << log2) - 1)) * childExtent;
        o[2] = origin[2] + i32(n & ((1u << log2) - 1)) * childExtent;
    };
    for (u32 n = 0; n < count; ++n)
    {
        if (Bit(valueMask, n) && !Bit(childMask, n) && values[n] > 0.0f)
        {
            Tile tile{};
            childOrigin(n, tile.origin);
            tile.extent = childExtent;
            tile.value = values[n];
            t.tiles.push_back(tile);
            t.bounds.Add(tile.origin[0], tile.origin[1], tile.origin[2], childExtent);
        }
    }
    for (u32 n = 0; n < count && s.Ok(); ++n)
    {
        if (!Bit(childMask, n))
        {
            continue;
        }
        i32 o[3];
        childOrigin(n, o);
        if (childLog2 > 3)
        {
            ReadInternal(s, f, 4, 3, o, t, values, raw, packed);
            continue;
        }
        u64 mask[8];
        s.Read(mask, sizeof(mask));
        t.leaves.push_back({{o[0], o[1], o[2]}});
        for (u32 w = 0; w < 8; ++w)
        {
            // Word w holds the voxels with x = w; bit y * 8 + z.
            if (mask[w] == 0)
            {
                continue;
            }
            u64 bits = mask[w];
            while (bits != 0)
            {
                const u32 b = u32(std::countr_zero(bits));
                bits &= bits - 1;
                t.bounds.Add(o[0] + i32(w), o[1] + i32(b >> 3), o[2] + i32(b & 7u), 1);
            }
        }
    }
}

// Sums of voxel values over blocks of factor^3 voxels of the active bounds.
struct Accumulator
{
    i64 lo[3] = {};
    i64 hi[3] = {};
    i64 factor = 1;
    u32 size[3] = {};
    std::vector<float> sums;

    float& At(i64 x, i64 y, i64 z) { return sums[(size_t(z) * size[1] + size_t(y)) * size[0] + size_t(x)]; }

    // `value` times the overlap of the voxels [origin, origin + extent) with every block.
    void AddBox(const i32 origin[3], i32 extent, float value)
    {
        i64 a[3];
        i64 b[3];
        for (int k = 0; k < 3; ++k)
        {
            a[k] = std::max<i64>(origin[k], lo[k]) - lo[k];
            b[k] = std::min<i64>(i64(origin[k]) + extent - 1, hi[k]) - lo[k];
            if (a[k] > b[k])
            {
                return;
            }
        }
        auto overlap = [&](int k, i64 c)
        { return std::min(b[k], c * factor + factor - 1) - std::max(a[k], c * factor) + 1; };
        for (i64 cz = a[2] / factor; cz <= b[2] / factor; ++cz)
        {
            for (i64 cy = a[1] / factor; cy <= b[1] / factor; ++cy)
            {
                for (i64 cx = a[0] / factor; cx <= b[0] / factor; ++cx)
                {
                    At(cx, cy, cz) += value * float(overlap(0, cx)) * float(overlap(1, cy)) * float(overlap(2, cz));
                }
            }
        }
    }
};

std::optional<VolumeGrid> ReadGrid(Stream& s, const Header& header, const Descriptor& d, u32 maxResolution)
{
    if (const std::string why = WhyUnreadable(d); !why.empty())
    {
        s.Fail(std::format("grid '{}' cannot be rendered: {}", d.name, why));
        return std::nullopt;
    }
    s.Seek(d.gridPosition);
    Format f;
    f.compression = s.Read<u32>();
    f.valueSize = d.valueType == "double" ? 8u : (d.valueType == "half" ? 2u : 4u);
    f.half = d.half;
    Metadata metadata;
    ReadMetadata(s, &metadata);
    if (s.Ok() && metadata.gridClass == "level set")
    {
        s.Fail(std::format("grid '{}' is a level set (a surface's signed distance), not a density", d.name));
        return std::nullopt;
    }
    Transform transform;
    ReadTransform(s, transform);
    if (s.Ok() && !transform.unsupported.empty())
    {
        s.Fail(std::format("grid '{}' cannot be placed: {}", d.name, transform.unsupported));
        return std::nullopt;
    }
    if (!d.instanceParent.empty())
    {
        // An instance shares its parent's tree: read that one with this transform.
        const auto parent = std::find_if(header.grids.begin(), header.grids.end(),
                                         [&](const Descriptor& g) { return g.name == d.instanceParent; });
        if (parent == header.grids.end())
        {
            s.Fail(std::format("grid '{}' is an instance of a missing grid '{}'", d.name, d.instanceParent));
            return std::nullopt;
        }
        s.Seek(parent->gridPosition);
        f.compression = s.Read<u32>();
        ReadMetadata(s, nullptr);
        Transform ignored;
        ReadTransform(s, ignored);
    }
    if (s.Read<i32>() != 1 && s.Ok())
    {
        s.Fail("trees with several buffers are not supported");
        return std::nullopt;
    }

    // Topology: the root's tiles and children, down to the leaves' masks.
    f.background = ReadScalar(s, f);
    const u32 tileCount = s.Read<u32>();
    const u32 childCount = s.Read<u32>();
    if (tileCount > (1u << 24) || childCount > (1u << 24))
    {
        s.Fail("corrupt root node");
        return std::nullopt;
    }
    Topology topology;
    for (u32 i = 0; i < tileCount && s.Ok(); ++i)
    {
        Tile tile{};
        s.Read(tile.origin, sizeof(tile.origin));
        tile.value = float(ReadScalar(s, f));
        tile.extent = 1 << 12;
        if (s.Read<u8>() != 0 && tile.value > 0.0f)
        {
            topology.tiles.push_back(tile);
            topology.bounds.Add(tile.origin[0], tile.origin[1], tile.origin[2], tile.extent);
        }
    }
    std::vector<float> values;
    std::vector<u8> raw;
    std::vector<u8> packed;
    for (u32 i = 0; i < childCount && s.Ok(); ++i)
    {
        i32 origin[3];
        s.Read(origin, sizeof(origin));
        ReadInternal(s, f, 5, 7, origin, topology, values, raw, packed);
    }
    if (!s.Ok())
    {
        return std::nullopt;
    }
    if (topology.bounds.Empty())
    {
        s.Fail(std::format("grid '{}' has no active voxels", d.name));
        return std::nullopt;
    }

    // The voxels over the active bounds, averaged down to maxResolution.
    Accumulator a;
    i64 longest = 0;
    for (int k = 0; k < 3; ++k)
    {
        a.lo[k] = topology.bounds.lo[k];
        a.hi[k] = topology.bounds.hi[k];
        longest = std::max(longest, a.hi[k] - a.lo[k] + 1);
    }
    const i64 limit = std::max<i64>(maxResolution, 1);
    a.factor = (longest + limit - 1) / limit;
    for (int k = 0; k < 3; ++k)
    {
        a.size[k] = u32((a.hi[k] - a.lo[k] + a.factor) / a.factor);
    }
    a.sums.assign(size_t(a.size[0]) * a.size[1] * a.size[2], 0.0f);
    for (const Tile& tile : topology.tiles)
    {
        a.AddBox(tile.origin, tile.extent, tile.value);
    }
    values.resize(512);
    for (const Leaf& leaf : topology.leaves)
    {
        u64 mask[8];
        s.Read(mask, sizeof(mask));
        if (!ReadValues(s, f, mask, 512, values.data(), raw, packed))
        {
            return std::nullopt;
        }
        for (u32 n = 0; n < 512; ++n)
        {
            const float v = values[n];
            if (!(v > 0.0f))
            {
                continue;
            }
            const i64 x = i64(leaf.origin[0]) + (n >> 6) - a.lo[0];
            const i64 y = i64(leaf.origin[1]) + ((n >> 3) & 7u) - a.lo[1];
            const i64 z = i64(leaf.origin[2]) + (n & 7u) - a.lo[2];
            if (x < 0 || y < 0 || z < 0 || x > a.hi[0] - a.lo[0] || y > a.hi[1] - a.lo[1] || z > a.hi[2] - a.lo[2])
            {
                continue; // an inactive voxel beyond the active bounds
            }
            a.At(x / a.factor, y / a.factor, z / a.factor) += v;
        }
    }
    if (!s.Ok())
    {
        return std::nullopt;
    }

    VolumeGrid grid;
    grid.sizeX = a.size[0];
    grid.sizeY = a.size[1];
    grid.sizeZ = a.size[2];
    const double scale = transform.scale[0];
    grid.voxelSize = float(scale * a.factor);
    // Voxel i's centre is at i * scale + translation: the box of the first one starts half a voxel
    // before it.
    grid.origin = hlsl::float3(float((double(a.lo[0]) - 0.5) * scale + transform.translation[0]),
                               float((double(a.lo[1]) - 0.5) * scale + transform.translation[1]),
                               float((double(a.lo[2]) - 0.5) * scale + transform.translation[2]));
    const float norm = float(1.0 / (double(a.factor) * double(a.factor) * double(a.factor)));
    grid.density = std::move(a.sums);
    for (float& v : grid.density)
    {
        v *= norm;
    }
    return grid;
}

} // namespace

std::optional<VdbFileInfo> ReadVdbInfo(const std::filesystem::path& path, std::string* error)
{
    Stream s(path);
    Header header;
    if (!ReadHeader(s, header))
    {
        if (error)
        {
            *error = s.Error();
        }
        return std::nullopt;
    }
    VdbFileInfo info;
    info.version = header.version;
    info.libraryMajor = header.libraryMajor;
    info.libraryMinor = header.libraryMinor;
    for (const Descriptor& d : header.grids)
    {
        VdbGridInfo g;
        g.name = d.name;
        g.valueType = d.valueType;
        g.nodes = d.nodes;
        g.halfFloat = d.half;
        s.Seek(d.gridPosition);
        s.Read<u32>(); // compression
        Metadata metadata;
        ReadMetadata(s, &metadata);
        Transform transform;
        ReadTransform(s, transform);
        if (!s.Ok())
        {
            break;
        }
        g.gridClass = metadata.gridClass;
        g.hasBounds = metadata.hasMin && metadata.hasMax;
        std::copy(metadata.boundsMin, metadata.boundsMin + 3, g.boundsMin);
        std::copy(metadata.boundsMax, metadata.boundsMax + 3, g.boundsMax);
        g.activeVoxels = metadata.activeVoxels;
        g.map = transform.map;
        std::copy(transform.scale, transform.scale + 3, g.voxelSize);
        std::copy(transform.translation, transform.translation + 3, g.translation);
        g.why = WhyUnreadable(d);
        if (g.why.empty() && metadata.gridClass == "level set")
        {
            g.why = "it is a level set (a surface's signed distance), not a density";
        }
        if (g.why.empty())
        {
            g.why = transform.unsupported;
        }
        g.readable = g.why.empty();
        info.grids.push_back(std::move(g));
    }
    if (!s.Ok())
    {
        if (error)
        {
            *error = s.Error();
        }
        return std::nullopt;
    }
    return info;
}

std::optional<VolumeGrid> ReadVdbGrid(const std::filesystem::path& path, const std::string& name, u32 maxResolution,
                                      std::string* error)
{
    Stream s(path);
    Header header;
    std::optional<VolumeGrid> grid;
    if (ReadHeader(s, header))
    {
        const Descriptor* chosen = nullptr;
        for (const Descriptor& d : header.grids)
        {
            if (d.name == name)
            {
                chosen = &d;
                break;
            }
        }
        if (!chosen && (name.empty() || name == "density"))
        {
            // The first grid of densities: scalars, and not a level set.
            for (const Descriptor& d : header.grids)
            {
                if (!WhyUnreadable(d).empty())
                {
                    continue;
                }
                s.Seek(d.gridPosition);
                s.Read<u32>(); // compression
                Metadata metadata;
                ReadMetadata(s, &metadata);
                if (s.Ok() && metadata.gridClass != "level set")
                {
                    chosen = &d;
                    break;
                }
            }
        }
        if (chosen)
        {
            grid = ReadGrid(s, header, *chosen, maxResolution);
        }
        else
        {
            std::string grids;
            for (const Descriptor& d : header.grids)
            {
                grids += std::format("{}{} ({})", grids.empty() ? "" : ", ", d.name, d.valueType);
            }
            s.Fail(std::format("no grid named '{}' (the file has {})", name, grids.empty() ? "none" : grids));
        }
    }
    if (!s.Ok())
    {
        if (error)
        {
            *error = s.Error();
        }
        return std::nullopt;
    }
    return grid;
}

} // namespace air::renderer
