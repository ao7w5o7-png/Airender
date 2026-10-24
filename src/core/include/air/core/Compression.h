#pragma once

// Decoders for the compressed blocks of file formats the renderer reads (OpenVDB, renderer/Vdb.h):
// zlib streams (RFC 1950 around RFC 1951 deflate), LZ4 blocks and Blosc 1 frames holding LZ4 with
// byte shuffling. Each fills `out` exactly, or fails with a message: corrupt or truncated input,
// or more or less data than `out` holds.

#include "air/core/Types.h"

#include <span>
#include <string>

namespace air
{

// A zlib stream: its header, deflate blocks and Adler-32 checksum (checked).
bool InflateZlib(std::span<const u8> in, std::span<u8> out, std::string* error = nullptr);
// Raw deflate blocks.
bool Inflate(std::span<const u8> in, std::span<u8> out, std::string* error = nullptr);
// One LZ4 block (the block format, without a frame).
bool DecompressLz4Block(std::span<const u8> in, std::span<u8> out, std::string* error = nullptr);
// One Blosc 1 frame (c-blosc 1.x): stored, or LZ4 blocks, byte shuffled or not.
bool DecompressBlosc(std::span<const u8> in, std::span<u8> out, std::string* error = nullptr);

} // namespace air
