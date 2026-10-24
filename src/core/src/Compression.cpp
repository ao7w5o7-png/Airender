#include "air/core/Compression.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <vector>

namespace air
{
namespace
{

bool Fail(std::string* error, std::string message)
{
    if (error)
    {
        *error = std::move(message);
    }
    return false;
}

u32 LoadLe32(const u8* p)
{
    return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24);
}

// --- Deflate (RFC 1951) ------------------------------------------------------------------------

// Bits in the order deflate packs them: least significant first. Reading past the end yields
// zeros and is reported by Overrun().
class BitReader
{
public:
    explicit BitReader(std::span<const u8> in) : mIn(in) {}

    u32 Peek(u32 count)
    {
        while (mCount < count)
        {
            mBuffer |= u64(mPos < mIn.size() ? mIn[mPos] : 0) << mCount;
            ++mPos;
            mCount += 8;
        }
        return u32(mBuffer & ((u64(1) << count) - 1));
    }
    void Drop(u32 count)
    {
        mBuffer >>= count;
        mCount -= count;
    }
    u32 Bits(u32 count)
    {
        if (count == 0)
        {
            return 0;
        }
        const u32 value = Peek(count);
        Drop(count);
        return value;
    }
    // Skips to the next byte boundary.
    void Align() { Drop(mCount % 8); }
    // Bytes consumed so far (after Align).
    size_t Position() const { return mPos - mCount / 8; }
    void SetPosition(size_t position)
    {
        mPos = position;
        mBuffer = 0;
        mCount = 0;
    }
    bool Overrun() const { return Position() > mIn.size(); }
    std::span<const u8> Input() const { return mIn; }

private:
    std::span<const u8> mIn;
    size_t mPos = 0;
    u64 mBuffer = 0;
    u32 mCount = 0;
};

constexpr u32 kFastBits = 10;

// A canonical Huffman code: codes of up to 15 bits, with a table that decodes those of up to
// kFastBits bits in one lookup (entry: symbol | length << 9, 0 for longer codes).
struct Huffman
{
    u16 counts[16] = {};
    u16 symbols[288] = {};
    u16 fast[1u << kFastBits] = {};
};

// False for an over-subscribed set of lengths. Incomplete codes are accepted (a distance code
// may have a single symbol); decoding a missing code fails.
bool BuildHuffman(Huffman& h, const u8* lengths, u32 count)
{
    std::fill(std::begin(h.counts), std::end(h.counts), u16(0));
    std::fill(std::begin(h.fast), std::end(h.fast), u16(0));
    for (u32 s = 0; s < count; ++s)
    {
        ++h.counts[lengths[s]];
    }
    h.counts[0] = 0;
    int left = 1;
    for (u32 length = 1; length < 16; ++length)
    {
        left = left * 2 - h.counts[length];
        if (left < 0)
        {
            return false;
        }
    }
    u16 offsets[16] = {};
    u32 next[16] = {};
    u32 code = 0;
    for (u32 length = 1; length < 16; ++length)
    {
        offsets[length] = u16(length == 1 ? 0 : offsets[length - 1] + h.counts[length - 1]);
        code = (code + (length == 1 ? 0 : h.counts[length - 1])) << 1;
        next[length] = code;
    }
    for (u32 s = 0; s < count; ++s)
    {
        const u32 length = lengths[s];
        if (length == 0)
        {
            continue;
        }
        h.symbols[offsets[length]++] = u16(s);
        const u32 c = next[length]++;
        if (length <= kFastBits)
        {
            // The stream holds the code's bits most significant first: index the table reversed.
            u32 reversed = 0;
            for (u32 b = 0; b < length; ++b)
            {
                reversed |= ((c >> b) & 1u) << (length - 1 - b);
            }
            for (u32 fill = reversed; fill < (1u << kFastBits); fill += 1u << length)
            {
                h.fast[fill] = u16(s | (length << 9));
            }
        }
    }
    return true;
}

int DecodeSymbol(BitReader& bits, const Huffman& h)
{
    const u16 entry = h.fast[bits.Peek(kFastBits)];
    if (entry != 0)
    {
        bits.Drop(entry >> 9);
        return entry & 511;
    }
    // A longer code: canonical decoding bit by bit.
    int code = 0;
    int first = 0;
    int index = 0;
    for (u32 length = 1; length < 16; ++length)
    {
        code |= int(bits.Bits(1));
        const int count = h.counts[length];
        if (code - count < first)
        {
            return h.symbols[index + (code - first)];
        }
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    return -1;
}

constexpr u16 kLengthBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr u8 kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr u16 kDistanceBase[30] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                                   33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                                   1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr u8 kDistanceExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                   6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

bool InflateBlock(BitReader& bits, const Huffman& lengths, const Huffman& distances, std::span<u8> out, size_t& written,
                  std::string* error)
{
    for (;;)
    {
        const int symbol = DecodeSymbol(bits, lengths);
        if (symbol < 0 || bits.Overrun())
        {
            return Fail(error, "deflate: invalid or truncated literal/length code");
        }
        if (symbol < 256)
        {
            if (written == out.size())
            {
                return Fail(error, std::format("deflate: data beyond the expected {} bytes", out.size()));
            }
            out[written++] = u8(symbol);
            continue;
        }
        if (symbol == 256)
        {
            return true;
        }
        if (symbol > 285)
        {
            return Fail(error, "deflate: invalid length symbol");
        }
        const u32 length = kLengthBase[symbol - 257] + bits.Bits(kLengthExtra[symbol - 257]);
        const int d = DecodeSymbol(bits, distances);
        if (d < 0 || d > 29)
        {
            return Fail(error, "deflate: invalid distance code");
        }
        const u32 distance = kDistanceBase[d] + bits.Bits(kDistanceExtra[d]);
        if (bits.Overrun())
        {
            return Fail(error, "deflate: truncated input");
        }
        if (distance > written)
        {
            return Fail(error, "deflate: distance before the start of the data");
        }
        if (length > out.size() - written)
        {
            return Fail(error, std::format("deflate: data beyond the expected {} bytes", out.size()));
        }
        // Byte by byte: the copy may overlap what it writes.
        for (u32 i = 0; i < length; ++i, ++written)
        {
            out[written] = out[written - distance];
        }
    }
}

// --- Blosc ----------------------------------------------------------------------------------------

constexpr u8 kBloscShuffle = 0x1;
constexpr u8 kBloscMemcpyed = 0x2;
constexpr u8 kBloscBitShuffle = 0x4;
constexpr u8 kBloscNoSplit = 0x10;
constexpr u32 kBloscOverhead = 16;
constexpr u32 kBloscLz4 = 1;

} // namespace

bool Inflate(std::span<const u8> in, std::span<u8> out, std::string* error)
{
    BitReader bits(in);
    size_t written = 0;
    // Allocated once: the tables are 4 KiB each.
    std::vector<Huffman> tables(3);
    Huffman& lengths = tables[0];
    Huffman& distances = tables[1];
    Huffman& codeLengths = tables[2];
    bool last = false;
    while (!last)
    {
        last = bits.Bits(1) != 0;
        const u32 type = bits.Bits(2);
        if (type == 0)
        {
            // Stored: LEN and its complement, then the bytes as they are.
            bits.Align();
            const u32 length = bits.Bits(16);
            const u32 complement = bits.Bits(16);
            if (bits.Overrun() || (length ^ 0xFFFFu) != complement)
            {
                return Fail(error, "deflate: corrupt stored block");
            }
            const size_t position = bits.Position();
            if (length > in.size() - std::min(position, in.size()))
            {
                return Fail(error, "deflate: truncated stored block");
            }
            if (length > out.size() - written)
            {
                return Fail(error, std::format("deflate: data beyond the expected {} bytes", out.size()));
            }
            std::memcpy(out.data() + written, in.data() + position, length);
            written += length;
            bits.SetPosition(position + length);
        }
        else if (type == 1)
        {
            u8 l[288];
            std::fill(l, l + 144, u8(8));
            std::fill(l + 144, l + 256, u8(9));
            std::fill(l + 256, l + 280, u8(7));
            std::fill(l + 280, l + 288, u8(8));
            u8 d[30];
            std::fill(d, d + 30, u8(5));
            BuildHuffman(lengths, l, 288);
            BuildHuffman(distances, d, 30);
            if (!InflateBlock(bits, lengths, distances, out, written, error))
            {
                return false;
            }
        }
        else if (type == 2)
        {
            const u32 literalCount = bits.Bits(5) + 257;
            const u32 distanceCount = bits.Bits(5) + 1;
            const u32 codeLengthCount = bits.Bits(4) + 4;
            if (literalCount > 286 || distanceCount > 30)
            {
                return Fail(error, "deflate: too many codes");
            }
            constexpr u8 kOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            u8 c[19] = {};
            for (u32 i = 0; i < codeLengthCount; ++i)
            {
                c[kOrder[i]] = u8(bits.Bits(3));
            }
            if (!BuildHuffman(codeLengths, c, 19))
            {
                return Fail(error, "deflate: invalid code length code");
            }
            u8 l[320] = {};
            u32 n = 0;
            while (n < literalCount + distanceCount)
            {
                const int symbol = DecodeSymbol(bits, codeLengths);
                if (symbol < 0 || bits.Overrun())
                {
                    return Fail(error, "deflate: invalid code lengths");
                }
                if (symbol < 16)
                {
                    l[n++] = u8(symbol);
                    continue;
                }
                u8 value = 0;
                u32 repeat = 0;
                if (symbol == 16)
                {
                    if (n == 0)
                    {
                        return Fail(error, "deflate: repeat with no previous length");
                    }
                    value = l[n - 1];
                    repeat = 3 + bits.Bits(2);
                }
                else if (symbol == 17)
                {
                    repeat = 3 + bits.Bits(3);
                }
                else
                {
                    repeat = 11 + bits.Bits(7);
                }
                if (n + repeat > literalCount + distanceCount)
                {
                    return Fail(error, "deflate: too many code lengths");
                }
                std::fill(l + n, l + n + repeat, value);
                n += repeat;
            }
            if (l[256] == 0)
            {
                return Fail(error, "deflate: no end-of-block code");
            }
            if (!BuildHuffman(lengths, l, literalCount) || !BuildHuffman(distances, l + literalCount, distanceCount))
            {
                return Fail(error, "deflate: over-subscribed code");
            }
            if (!InflateBlock(bits, lengths, distances, out, written, error))
            {
                return false;
            }
        }
        else
        {
            return Fail(error, "deflate: invalid block type");
        }
        if (bits.Overrun())
        {
            return Fail(error, "deflate: truncated input");
        }
    }
    if (written != out.size())
    {
        return Fail(error, std::format("deflate: {} bytes, expected {}", written, out.size()));
    }
    return true;
}

bool InflateZlib(std::span<const u8> in, std::span<u8> out, std::string* error)
{
    if (in.size() < 6)
    {
        return Fail(error, "zlib: truncated stream");
    }
    const u32 cmf = in[0];
    const u32 flags = in[1];
    if ((cmf & 15u) != 8 || (cmf >> 4) > 7 || (cmf * 256 + flags) % 31 != 0 || (flags & 0x20u) != 0)
    {
        return Fail(error, "zlib: invalid header");
    }
    if (!Inflate(in.subspan(2, in.size() - 6), out, error))
    {
        return false;
    }
    u32 a = 1;
    u32 b = 0;
    for (size_t i = 0; i < out.size();)
    {
        // 5552 bytes keep the sums within 32 bits before the modulo.
        const size_t end = std::min(out.size(), i + 5552);
        for (; i < end; ++i)
        {
            a += out[i];
            b += a;
        }
        a %= 65521u;
        b %= 65521u;
    }
    const u8* tail = in.data() + in.size() - 4;
    const u32 stored = (u32(tail[0]) << 24) | (u32(tail[1]) << 16) | (u32(tail[2]) << 8) | u32(tail[3]);
    if (((b << 16) | a) != stored)
    {
        return Fail(error, "zlib: checksum mismatch");
    }
    return true;
}

bool DecompressLz4Block(std::span<const u8> in, std::span<u8> out, std::string* error)
{
    size_t ip = 0;
    size_t op = 0;
    auto extend = [&](size_t& length)
    {
        u8 byte = 255;
        while (byte == 255)
        {
            if (ip >= in.size())
            {
                return false;
            }
            byte = in[ip++];
            length += byte;
        }
        return true;
    };
    while (ip < in.size())
    {
        const u8 token = in[ip++];
        size_t literals = token >> 4;
        if (literals == 15 && !extend(literals))
        {
            return Fail(error, "lz4: truncated literal length");
        }
        if (literals > in.size() - ip || literals > out.size() - op)
        {
            return Fail(error, "lz4: literals beyond the input or the output");
        }
        std::memcpy(out.data() + op, in.data() + ip, literals);
        ip += literals;
        op += literals;
        if (ip == in.size())
        {
            break; // the last sequence holds literals only
        }
        if (in.size() - ip < 2)
        {
            return Fail(error, "lz4: truncated offset");
        }
        const size_t offset = size_t(in[ip]) | (size_t(in[ip + 1]) << 8);
        ip += 2;
        size_t match = token & 15u;
        if (match == 15 && !extend(match))
        {
            return Fail(error, "lz4: truncated match length");
        }
        match += 4;
        if (offset == 0 || offset > op)
        {
            return Fail(error, "lz4: offset before the start of the data");
        }
        if (match > out.size() - op)
        {
            return Fail(error, "lz4: match beyond the output");
        }
        for (size_t i = 0; i < match; ++i, ++op)
        {
            out[op] = out[op - offset];
        }
    }
    if (op != out.size())
    {
        return Fail(error, std::format("lz4: {} bytes, expected {}", op, out.size()));
    }
    return true;
}

bool DecompressBlosc(std::span<const u8> in, std::span<u8> out, std::string* error)
{
    if (in.size() < kBloscOverhead)
    {
        return Fail(error, "blosc: truncated header");
    }
    const u8 flags = in[2];
    const u32 typeSize = std::max<u32>(in[3], 1u);
    const u32 bytes = LoadLe32(in.data() + 4);
    const u32 blockSize = LoadLe32(in.data() + 8);
    const u32 compressedBytes = LoadLe32(in.data() + 12);
    if (bytes != out.size())
    {
        return Fail(error, std::format("blosc: frame of {} bytes, expected {}", bytes, out.size()));
    }
    if (compressedBytes > in.size())
    {
        return Fail(error, "blosc: truncated frame");
    }
    if (bytes == 0)
    {
        return true;
    }
    if ((flags & kBloscMemcpyed) != 0)
    {
        if (in.size() - kBloscOverhead < bytes)
        {
            return Fail(error, "blosc: truncated stored frame");
        }
        std::memcpy(out.data(), in.data() + kBloscOverhead, bytes);
        return true;
    }
    const u32 codec = flags >> 5;
    if (codec != kBloscLz4)
    {
        return Fail(error, std::format("blosc: codec {} is not supported (LZ4 only)", codec));
    }
    if ((flags & kBloscBitShuffle) != 0 && (flags & kBloscShuffle) == 0)
    {
        return Fail(error, "blosc: bit shuffling is not supported");
    }
    if (blockSize == 0)
    {
        return Fail(error, "blosc: zero block size");
    }
    const bool shuffled = (flags & kBloscShuffle) != 0 && typeSize > 1;
    const u32 blocks = bytes / blockSize + (bytes % blockSize != 0 ? 1u : 0u);
    if (kBloscOverhead + size_t(blocks) * 4 > in.size())
    {
        return Fail(error, "blosc: truncated block offsets");
    }
    std::vector<u8> temporary(shuffled ? blockSize : 0);
    for (u32 block = 0; block < blocks; ++block)
    {
        const bool leftover = block == blocks - 1 && bytes % blockSize != 0;
        const u32 size = leftover ? bytes % blockSize : blockSize;
        // Split into one stream per byte of the type, as c-blosc 1.x decides (compatible with
        // frames written before the no-split flag).
        const bool split = (flags & kBloscNoSplit) == 0 && typeSize <= 16 && size / typeSize >= 128 && !leftover;
        const u32 splits = split ? typeSize : 1u;
        const u32 splitSize = size / splits;
        u8* destination = shuffled ? temporary.data() : out.data() + size_t(block) * blockSize;
        size_t position = LoadLe32(in.data() + kBloscOverhead + size_t(block) * 4);
        for (u32 s = 0; s < splits; ++s)
        {
            if (position + 4 > in.size())
            {
                return Fail(error, "blosc: truncated block");
            }
            const u32 streamBytes = LoadLe32(in.data() + position);
            position += 4;
            if (streamBytes > in.size() - position)
            {
                return Fail(error, "blosc: truncated block");
            }
            const std::span<u8> target(destination + size_t(s) * splitSize, splitSize);
            if (streamBytes == splitSize)
            {
                std::memcpy(target.data(), in.data() + position, splitSize);
            }
            else if (!DecompressLz4Block(in.subspan(position, streamBytes), target, error))
            {
                return false;
            }
            position += streamBytes;
        }
        if (shuffled)
        {
            // Byte j of element i was stored at j * elements + i; a tail shorter than one element
            // stays where it is.
            u8* target = out.data() + size_t(block) * blockSize;
            const u32 elements = size / typeSize;
            for (u32 j = 0; j < typeSize; ++j)
            {
                for (u32 i = 0; i < elements; ++i)
                {
                    target[size_t(i) * typeSize + j] = temporary[size_t(j) * elements + i];
                }
            }
            const u32 tail = elements * typeSize;
            std::memcpy(target + tail, temporary.data() + tail, size - tail);
        }
    }
    return true;
}

} // namespace air
