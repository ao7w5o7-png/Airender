#pragma once

// Backend-neutral RHI vocabulary: enums, handles and resource descriptions.
// Mirrors the split in UE's RHI: the same descriptions feed D3D12, Vulkan and the Null backend.

#include "air/core/HandlePool.h"
#include "air/core/Types.h"

#include <span>
#include <string>
#include <vector>

namespace air::rhi {

enum class Backend : u8 { Null, D3D12, Vulkan };

enum class QueueType : u8 { Graphics, Compute, Copy, Count };

enum class Format : u16 {
    Unknown,
    R8_UNORM,
    RG8_UNORM,
    RGBA8_UNORM,
    RGBA8_SRGB,
    BGRA8_UNORM,
    R16_FLOAT,
    RG16_FLOAT,
    RGBA16_FLOAT,
    R32_FLOAT,
    RG32_FLOAT,
    RGB32_FLOAT,
    RGBA32_FLOAT,
    R32_UINT,
    RG32_UINT,
    RGBA32_UINT,
    R11G11B10_FLOAT,
    D32_FLOAT,
    D24_UNORM_S8_UINT,
};

u32 FormatBytesPerPixel(Format format);
bool IsDepthFormat(Format format);
const char* ToString(Format format);

// Where a resource lives. Backends map these onto D3D12 heap types / Vulkan memory types.
enum class MemoryLocation : u8 {
    DeviceLocal,     // VRAM, GPU only (D3D12 DEFAULT, VK DEVICE_LOCAL)
    Upload,          // CPU write -> GPU read, write-combined (D3D12 UPLOAD, VK HOST_VISIBLE|COHERENT)
    Readback,        // GPU write -> CPU read, cached (D3D12 READBACK, VK HOST_VISIBLE|CACHED)
    DeviceLocalHost, // VRAM visible to the CPU (ReBAR / D3D12 GPU_UPLOAD); falls back to Upload
    Count
};
const char* ToString(MemoryLocation location);

enum class BufferUsage : u32 {
    None = 0,
    Vertex = 1 << 0,
    Index = 1 << 1,
    Constant = 1 << 2,
    ShaderResource = 1 << 3,
    UnorderedAccess = 1 << 4,
    Indirect = 1 << 5,
    CopySrc = 1 << 6,
    CopyDst = 1 << 7,
    AccelStructStorage = 1 << 8,    // backing memory of a BLAS/TLAS
    AccelStructBuildInput = 1 << 9, // vertex/index/instance data read by a build
    ShaderTable = 1 << 10,
};
AIR_ENUM_FLAGS(BufferUsage)

enum class TextureUsage : u32 {
    None = 0,
    ShaderResource = 1 << 0,
    UnorderedAccess = 1 << 1,
    RenderTarget = 1 << 2,
    DepthStencil = 1 << 3,
    CopySrc = 1 << 4,
    CopyDst = 1 << 5,
    Present = 1 << 6,
};
AIR_ENUM_FLAGS(TextureUsage)

// Resource access state, in the spirit of UE's ERHIAccess. Barriers are expressed as
// (before access -> after access); backends translate to D3D12 enhanced barriers / VK sync2.
enum class Access : u32 {
    None = 0, // undefined contents (fresh / aliased transient resource)
    CopySrc = 1 << 0,
    CopyDst = 1 << 1,
    VertexOrIndex = 1 << 2,
    Constant = 1 << 3,
    SRVGraphics = 1 << 4,
    SRVCompute = 1 << 5,
    UAVGraphics = 1 << 6,
    UAVCompute = 1 << 7,
    RenderTarget = 1 << 8,
    DepthRead = 1 << 9,
    DepthWrite = 1 << 10,
    IndirectArgs = 1 << 11,
    AccelStructRead = 1 << 12,
    AccelStructWrite = 1 << 13,
    Present = 1 << 14,
    HostRead = 1 << 15,

    ReadOnlyMask = CopySrc | VertexOrIndex | Constant | SRVGraphics | SRVCompute | DepthRead | IndirectArgs |
                   AccelStructRead | Present | HostRead,
    WritableMask = CopyDst | UAVGraphics | UAVCompute | RenderTarget | DepthWrite | AccelStructWrite,
    UAVMask = UAVGraphics | UAVCompute,
};
AIR_ENUM_FLAGS(Access)
std::string ToString(Access access);

inline bool IsWriteAccess(Access access) {
    return Any(access & Access::WritableMask);
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------
struct BufferTag;
struct TextureTag;
struct PipelineTag;
struct AccelStructTag;
struct HeapTag;
using BufferHandle = Handle<BufferTag>;
using TextureHandle = Handle<TextureTag>;
using PipelineHandle = Handle<PipelineTag>;
using AccelStructHandle = Handle<AccelStructTag>;
using HeapHandle = Handle<HeapTag>; // placed-resource heap (RDG transient memory)

// Index into the global bindless descriptor heap (SM 6.6 ResourceDescriptorHeap / VK descriptor indexing).
using BindlessIndex = u32;
constexpr BindlessIndex kInvalidBindless = ~0u;

// ---------------------------------------------------------------------------
// Descriptions
// ---------------------------------------------------------------------------
struct BufferDesc {
    u64 size = 0;
    u32 stride = 0; // > 0: structured buffer element size
    BufferUsage usage = BufferUsage::None;
    MemoryLocation memory = MemoryLocation::DeviceLocal;
    std::string debugName;
};

enum class TextureDimension : u8 { Tex2D, Tex3D, TexCube };

struct TextureDesc {
    TextureDimension dimension = TextureDimension::Tex2D;
    Format format = Format::Unknown;
    u32 width = 1;
    u32 height = 1;
    u32 depthOrArraySize = 1;
    u32 mipLevels = 1;
    u32 sampleCount = 1;
    TextureUsage usage = TextureUsage::None;
    float clearColor[4] = {0, 0, 0, 0};
    std::string debugName;
};

// Size/alignment a resource needs when placed in a heap.
struct AllocationInfo {
    u64 size = 0;
    u64 alignment = 0;
};

// Acceleration structures (hardware ray tracing).
struct RayTracingGeometryTriangles {
    BufferHandle vertexBuffer;
    u64 vertexOffset = 0;
    u32 vertexCount = 0;
    u32 vertexStride = 12;
    Format vertexFormat = Format::RGB32_FLOAT;
    BufferHandle indexBuffer; // 32-bit indices
    u64 indexOffset = 0;
    u32 indexCount = 0;
    bool opaque = true;
};

struct BlasDesc {
    std::vector<RayTracingGeometryTriangles> geometries;
    bool allowUpdate = false;
    bool preferFastTrace = true; // offline path tracing wants fast trace over fast build
    std::string debugName;
};

struct TlasInstance {
    float transform[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}; // row-major 3x4
    u32 instanceId = 0;                                                 // InstanceID() in shaders, 24 bits
    u32 mask = 0xFF;
    u32 hitGroupOffset = 0;
    AccelStructHandle blas;
};

struct TlasDesc {
    u32 maxInstances = 0;
    bool allowUpdate = true;
    std::string debugName;
};

// Shaders are compiled offline by DXC to DXIL (D3D12) and SPIR-V (Vulkan) from the same HLSL.
struct ShaderBlob {
    std::vector<u8> bytecode;
    std::string entryPoint;
};

struct ComputePipelineDesc {
    ShaderBlob shader;
    u32 pushConstantBytes = 0; // root constants / push constants, max 128
    std::string debugName;
};

struct RayTracingHitGroup {
    std::string name;
    std::string closestHit;
    std::string anyHit;
};

struct RayTracingPipelineDesc {
    ShaderBlob library; // one DXIL/SPIR-V library holding all entry points
    std::string rayGen;
    std::vector<std::string> missShaders;
    std::vector<RayTracingHitGroup> hitGroups;
    u32 maxPayloadBytes = 32;
    u32 maxAttributeBytes = 8;
    u32 maxRecursionDepth = 1; // path tracer loops in raygen; keep recursion at 1
    u32 pushConstantBytes = 0;
    std::string debugName;
};

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------
struct Barrier {
    enum class Type : u8 { Buffer, Texture, AccelStruct, Aliasing };
    Type type = Type::Texture;
    BufferHandle buffer;
    TextureHandle texture;
    AccelStructHandle accelStruct;
    Access before = Access::None;
    Access after = Access::None;
    bool discard = false; // contents may be discarded (first use of a transient resource)

    static Barrier Tex(TextureHandle t, Access before, Access after, bool discard = false) {
        Barrier b;
        b.type = Type::Texture;
        b.texture = t;
        b.before = before;
        b.after = after;
        b.discard = discard;
        return b;
    }
    static Barrier Buf(BufferHandle buf, Access before, Access after) {
        Barrier b;
        b.type = Type::Buffer;
        b.buffer = buf;
        b.before = before;
        b.after = after;
        return b;
    }
};

struct DeviceCapabilities {
    bool rayTracing = false; // DXR 1.1 / VK_KHR_ray_tracing_pipeline + ray_query
    bool meshShaders = false;
    bool bindless = false; // SM 6.6 dynamic resources / descriptor indexing
    bool rebar = false;    // DeviceLocalHost memory available
    u64 dedicatedVideoMemory = 0;
    std::string adapterName;
};

} // namespace air::rhi
