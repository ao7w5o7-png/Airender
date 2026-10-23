// Direct3D 12 backend (work in progress).
//
// Plan (see docs/architecture.md, "RHI"):
//   * DXGI factory -> pick the high-performance adapter -> ID3D12Device5+ (DXR 1.1 required for the path tracer)
//   * Debug layer + GPU-based validation when DeviceDesc::enableValidation
//   * One ID3D12CommandQueue per QueueType, one ID3D12Fence per queue
//   * GpuMemoryAllocator with a D3D12 IGpuHeapProvider (ID3D12Heap per block, placed resources)
//     memory type key = heap type (DEFAULT/UPLOAD/READBACK/GPU_UPLOAD) x tier-1 category
//   * Shader-visible CBV/SRV/UAV heap of 1M descriptors for bindless (SM 6.6 ResourceDescriptorHeap)
//   * Enhanced barriers (ID3D12GraphicsCommandList7::Barrier) when available, legacy fallback otherwise

#include "../Backends.h"

#include "air/core/Log.h"

namespace air::rhi {

std::unique_ptr<Device> CreateD3D12Device(const DeviceDesc& desc) {
    AIR_LOG(Error, "D3D12", "D3D12 backend is not implemented yet");
    return nullptr;
}

} // namespace air::rhi
