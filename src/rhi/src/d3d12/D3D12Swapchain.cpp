// DXGI flip-model swapchain. Back buffers are wrapped as ordinary TextureHandles so the render
// graph and command lists treat them like any other texture.

#include "D3D12Backend.h"

#include "air/core/Assert.h"
#include "air/core/Log.h"

#include <format>

namespace air::rhi::d3d12
{
namespace
{

class D3D12Swapchain final : public Swapchain
{
public:
    D3D12Swapchain(D3D12Device& device, const SwapchainDesc& desc) : mDevice(device), mDesc(desc) {}

    ~D3D12Swapchain() override
    {
        if (mSwapchain)
        {
            ReleaseBuffers();
            mDevice.WaitIdle(); // the back buffers must be released before the swapchain
        }
    }

    bool Initialize()
    {
        AIR_CHECK(mDesc.window, "SwapchainDesc::window (HWND) is required on D3D12");
        AIR_CHECK(mDesc.format == Format::BGRA8_UNORM || mDesc.format == Format::RGBA8_UNORM ||
                      mDesc.format == Format::RGBA16_FLOAT,
                  "swapchain format {} is not supported by the flip model", ToString(mDesc.format));
        DXGI_SWAP_CHAIN_DESC1 desc = {};
        desc.Width = mDesc.width;
        desc.Height = mDesc.height;
        desc.Format = ToDxgi(mDesc.format);
        desc.SampleDesc.Count = 1;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = mDesc.bufferCount;
        desc.Scaling = DXGI_SCALING_STRETCH;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        desc.Flags = mDevice.AllowTearing() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
        mFlags = desc.Flags;

        auto* hwnd = static_cast<HWND>(mDesc.window);
        ComPtr<IDXGISwapChain1> swapchain;
        const HRESULT hr = mDevice.Factory()->CreateSwapChainForHwnd(mDevice.GetQueue(QueueType::Graphics).queue.Get(),
                                                                     hwnd, &desc, nullptr, nullptr, &swapchain);
        if (FAILED(hr))
        {
            AIR_LOG(Error, "D3D12", "CreateSwapChainForHwnd failed: {}", HResultToString(hr));
            return false;
        }
        mDevice.Factory()->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        AIR_D3D12_CHECK(swapchain.As(&mSwapchain));
        AcquireBuffers();
        return true;
    }

    TextureHandle GetCurrentTexture() override { return mBuffers[mSwapchain->GetCurrentBackBufferIndex()]; }

    bool Present() override
    {
        const UINT interval = mDesc.vsync ? 1 : 0;
        const UINT flags =
            !mDesc.vsync && (mFlags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) ? DXGI_PRESENT_ALLOW_TEARING : 0;
        const HRESULT hr = mSwapchain->Present(interval, flags);
        if (FAILED(hr))
        {
            mDevice.LogDeviceRemoved(hr, "Present");
            return false;
        }
        return true;
    }

    void Resize(u32 width, u32 height) override
    {
        if (width == 0 || height == 0 || (width == mDesc.width && height == mDesc.height))
        {
            return;
        }
        ReleaseBuffers();
        mDevice.WaitIdle(); // flushes the deferred releases of the old back buffers
        AIR_D3D12_CHECK(mSwapchain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, mFlags));
        mDesc.width = width;
        mDesc.height = height;
        AcquireBuffers();
    }

    void SetVsync(bool vsync) override { mDesc.vsync = vsync; }
    const SwapchainDesc& GetDesc() const override { return mDesc; }

private:
    void AcquireBuffers()
    {
        for (u32 i = 0; i < mDesc.bufferCount; ++i)
        {
            ComPtr<ID3D12Resource> buffer;
            AIR_D3D12_CHECK(mSwapchain->GetBuffer(i, IID_PPV_ARGS(&buffer)));
            TextureDesc desc;
            desc.format = mDesc.format;
            desc.width = mDesc.width;
            desc.height = mDesc.height;
            desc.usage = TextureUsage::RenderTarget | TextureUsage::CopyDst | TextureUsage::Present;
            desc.debugName = std::format("Swapchain[{}]", i);
            mBuffers.push_back(mDevice.RegisterExternalTexture(std::move(buffer), desc));
        }
    }

    void ReleaseBuffers()
    {
        for (TextureHandle buffer : mBuffers)
        {
            mDevice.DestroyTexture(buffer);
        }
        mBuffers.clear();
    }

    D3D12Device& mDevice;
    SwapchainDesc mDesc;
    ComPtr<IDXGISwapChain3> mSwapchain;
    UINT mFlags = 0;
    std::vector<TextureHandle> mBuffers;
};

} // namespace

std::unique_ptr<Swapchain> CreateD3D12Swapchain(D3D12Device& device, const SwapchainDesc& desc)
{
    auto swapchain = std::make_unique<D3D12Swapchain>(device, desc);
    if (!swapchain->Initialize())
    {
        return nullptr;
    }
    return swapchain;
}

} // namespace air::rhi::d3d12
