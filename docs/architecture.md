# Airender 架构设计

> 目标：一个能让 AI 直接写代码、调用、调试的渲染器；同时支持实时渲染和影视级离线渲染，
> 最终能用 AI + 渲染器产出影视化视频，也作为图形算法试验场，快速验证新效果。

本文描述第一版骨架的架构。设计上大量参考 Unreal Engine（RHI / RDG / Tasks），但做了简化，
优先保证 **线程安全、可调试、对 AI 友好**。

---

## 1. 分层总览

```
 apps/aircli            命令行驱动：render / features / compare，JSON 报告
 tools/air.py           开发 harness：check / test / run / golden-update / new-feature / layers
      │
 features/<name>        渲染特性插件：pathtracer（GPU 路径追踪）、example（新算法模板）……
      │                 互不依赖，CMake 自动发现，按名字注册
 renderer               扩展点：RenderFeature 接口、FeatureRegistry、ParameterSet、FrameRunner、GpuScene
      │
 rdg (RenderGraph)      声明式 pass：自动裁剪、自动 barrier、瞬态资源显存复用、并行录制
      │
 rhi                    设备抽象：D3D12 / Vulkan / Null 三个后端，bindless，显存分配器
      │
 core                   任务系统（Task / Pipe / ParallelFor）、句柄池、日志、断言
```

依赖严格单向：上层只依赖下层，`core` 不依赖任何图形 API，特性之间互不依赖。
`python3 tools/air.py layers` 检查每一个 `#include "air/..."` 是否越层，CI 会跑。

| 模块 | 目录 | 对应 UE 概念 |
|---|---|---|
| 任务系统 | `src/core` | `UE::Tasks`（`Launch`、prerequisites、`FPipe`、`ParallelFor`） |
| RHI | `src/rhi` | `FDynamicRHI` / `FRHICommandList` / `ERHIAccess` |
| 显存管理 | `src/rhi/GpuMemoryAllocator` | D3D12 的 `FD3D12PoolAllocator`、Vulkan 的 `FVulkanResourceHeap` |
| RenderGraph | `src/rdg` | `FRDGBuilder`、transient allocator |
| 渲染特性框架 | `src/renderer` | UE 的 `FSceneViewExtension` / 渲染模块插件 |
| 路径追踪 | `src/features/pathtracer` | UE Path Tracer（`PathTracing.cpp`） |

---

## 2. 任务系统（多线程并发）

文件：`src/core/include/air/core/TaskSystem.h`

### 2.1 模型

- **Task**：`TaskSystem::Launch(name, fn, prerequisites, priority)` 返回 `TaskHandle`。
  所有前置任务完成后才会被调度。任务之间形成 DAG。
- **Wait 会帮忙干活**：`TaskHandle::Wait()` 在等待期间会执行队列里的其他任务（UE 的 "retraction"
  思路的简化版）。因此在 worker 线程里等待子任务不会把线程池卡死
  （见测试 `TaskSystem_NestedWaitOnSingleWorkerDoesNotDeadlock`）。
- **Pipe**：同一个 `Pipe` 上的任务按提交顺序**串行**执行，但可以落在任意线程上。
  这是保证线程安全的首选手段：把一份非线程安全的状态"交给"一个 Pipe，所有访问都作为 Pipe 任务提交，
  就不需要锁（UE 用 `FPipe` 替代 named thread 也是这个原因）。
- **ParallelFor**：调用线程本身也参与，动态分块。
- **优先级**：High / Normal / Background 三档队列。

### 2.2 线程安全约定（全工程通用）

| 对象 | 规则 |
|---|---|
| `TaskSystem` / `TaskHandle` / `Pipe` | 所有函数任意线程可调 |
| `rhi::Device` | 所有函数线程安全；资源可在任意线程创建；`Destroy*` 延迟到 GPU 用完后才真正释放 |
| `rhi::CommandList` | **非**线程安全：一个列表同一时间只由一个线程录制；不同列表可并行录制 |
| `rhi::Device::Submit` | 线程安全，同一队列按调用顺序串行化 |
| `rdg::RenderGraph` 构建 | 单线程（构造它的线程），有运行时检查 |
| RDG pass 的 execute lambda | 可能在 worker 上与其他 pass 并发执行：只能用 `PassContext` 和按值捕获/只读的数据 |
| `GpuMemoryAllocator` | 线程安全，每个内存池一把锁 |
| `FrameRunner` / `RenderFeature` / `ParameterSet` | 在驱动帧循环的线程上调用；参数只在帧之间修改 |
| `HandlePool` | 线程安全；句柄带 generation，用到已释放的句柄会被检测出来而不是野指针 |

### 2.3 后续计划

- 把全局队列换成每线程 work-stealing 双端队列（Chase-Lev），减少锁竞争。
- 引入 Render 线程 / RHI 线程两条 Pipe：游戏逻辑 → 渲染命令构建 → RHI 提交 三级流水，
  对应 UE 的 GameThread / RenderThread / RHIThread。

---

## 3. RHI（DX12 + Vulkan）

文件：`src/rhi/include/air/rhi/`

### 3.1 设计要点

1. **句柄而不是裸指针**：`BufferHandle`、`TextureHandle` 等是 `{index, generation}` 值类型，
   可以打印、比较、跨线程传递，用到失效句柄会报错。对 AI 调试很友好。
2. **Bindless 优先**：所有着色器资源通过全局描述符堆的索引访问
   （D3D12 SM 6.6 `ResourceDescriptorHeap`，Vulkan descriptor indexing）。
   pass 只需要把索引塞进 push constants，不需要 root signature / descriptor set 布局。
   这样写新 pass 几乎没有样板代码，也是 AI 最不容易写错的绑定模型。
3. **Access 状态 + barrier**：资源状态用 `rhi::Access` 位标志表达（类似 UE 的 `ERHIAccess`），
   barrier 写成 `before -> after`。后端翻译为 D3D12 Enhanced Barriers / Vulkan synchronization2。
   上层（RDG）自动生成 barrier，pass 代码不手写 barrier。
4. **一份 HLSL，两种字节码**：DXC 编译为 DXIL（D3D12）和 SPIR-V（Vulkan）。
5. **命令捕获**：`Device::SetCommandCapture(true)` 后，每条提交的命令都变成一行可读文本，例如
   `[Graphics] Barrier [tex 'SceneColor' UAVCompute -> SRVCompute]`。
   Null 后端还会校验 barrier 的 `before` 是否等于资源的真实状态，不一致时输出 `VALIDATION:` 行。
   AI 不看屏幕也能知道一帧做了什么、错在哪。

### 3.2 后端

| 后端 | 状态 | 说明 |
|---|---|---|
| Null | ✅ 可用 | 无 GPU；真实的显存记账、命令捕获、barrier 校验。用于单元测试、CI、无显卡环境调试 |
| D3D12 | 🚧 桩 | 计划：DXGI 选卡 → `ID3D12Device5+`，DXR 1.1，Enhanced Barriers，GPU-based validation |
| Vulkan | 🚧 桩 | 计划：Vulkan 1.3（dynamic rendering、sync2、timeline semaphore），KHR ray tracing pipeline + ray query |

### 3.3 队列

`Graphics` / `Compute` / `Copy` 三种队列。第一版 RDG 全部提交到 Graphics；
异步计算（光追 BVH 构建、降噪与光栅并行）留作后续，RDG 编译期已有足够信息来拆分。

---

## 4. 显存管理（多种显存类型）

文件：`src/rhi/include/air/rhi/GpuMemoryAllocator.h`

### 4.1 显存类型

`MemoryLocation` 描述资源在哪里：

| MemoryLocation | D3D12 | Vulkan | 用途 |
|---|---|---|---|
| `DeviceLocal` | `HEAP_TYPE_DEFAULT` | `DEVICE_LOCAL` | 纹理、RT、BVH、绝大多数 buffer |
| `Upload` | `HEAP_TYPE_UPLOAD` | `HOST_VISIBLE \| HOST_COHERENT` | CPU 每帧写的常量、上传暂存 |
| `Readback` | `HEAP_TYPE_READBACK` | `HOST_VISIBLE \| HOST_CACHED` | 截图、离线渲染结果回读、GPU 调试数据 |
| `DeviceLocalHost` | `HEAP_TYPE_GPU_UPLOAD`（ReBAR） | `DEVICE_LOCAL \| HOST_VISIBLE` | CPU 直接写显存，省一次拷贝；不支持时退化到 Upload |

### 4.2 三层结构

```
GpuMemoryAllocator           线程安全；按 memoryTypeKey 分池，每池一把锁
  └─ Pool (memoryTypeKey)    D3D12: 堆类型 × tier-1 类别（buffer / RT-DS 纹理 / 其他纹理）
       │                     Vulkan: memoryTypeIndex
       ├─ Block (256 MiB 堆)  由后端的 IGpuHeapProvider 创建 ID3D12Heap / VkDeviceMemory
       │    └─ TlsfAllocator  O(1) 分配/释放，自动合并相邻空闲块，支持任意 2 的幂对齐
       └─ Dedicated           ≥ 64 MiB 的资源单独一个堆（大 RT、超大 BVH）
```

- **TLSF**（Two-Level Segregated Fit）：分配和释放都是常数时间，碎片低，是游戏引擎 GPU 分配器的常见选择。
  它只管偏移量，不碰 GPU 对象，所以能单独做随机压力测试（`Tlsf_RandomStress`）。
- 空堆策略：每个池最多保留一个空堆，避免反复创建/销毁。
- Upload / Readback 用更小的块（64 MiB），因为它们占用的是系统内存/BAR 空间。
- 统计：`Device::GetMemoryStats()` 按显存类型给出 堆数 / 保留字节 / 已用字节 / 分配数 / dedicated 数。

### 4.3 RDG 瞬态显存（aliasing）

RenderGraph 内部创建的临时资源不走上面的池，而是：

1. 编译期算出每个资源的生命周期 `[首次使用 pass, 最后使用 pass]`；
2. 按大小降序做 first-fit 打包：生命周期不重叠的资源可以占用同一段显存；
3. 整帧只申请一个 placed heap，在里面 `CreatePlacedTexture/Buffer`。

`RenderGraph::GetStats()` 会给出 `transientHeapSize`（复用后）和 `transientBytesUnaliased`（不复用时）。
资源首次使用时 barrier 带 `discard` 标记，后端据此发 aliasing barrier / 丢弃旧内容。

后续：跨帧缓存 transient heap（UE 的 `FRHITransientResourceAllocator`），避免每帧建堆。

### 4.4 延迟释放

`Destroy*` 只是登记；`EndFrame()` 时释放 `framesInFlight` 帧之前登记的对象，保证 GPU 不会访问已释放的内存。

---

## 5. RenderGraph（方便写 pass）

文件：`src/rdg/include/air/rdg/RenderGraph.h`

### 5.1 写一个 pass

```cpp
rdg::RenderGraph graph(device, &tasks);
auto color = graph.CreateTexture({.format = Format::RGBA16_FLOAT, .width = w, .height = h,
                                  .usage = TextureUsage::UnorderedAccess | TextureUsage::ShaderResource,
                                  .debugName = "SceneColor"});
auto out = graph.ImportTexture(backbuffer, Access::None, Access::Present);

graph.AddPass("Shade", PassFlags::Compute,
    [&](PassBuilder& b) { b.Write(color, Access::UAVCompute); },          // 声明：写 color
    [=](PassContext& ctx) {                                               // 执行：只录命令
        ShadeConstants c{.output = ctx.UAV(color)};
        ctx.cmd.SetPipeline(shadePso);
        ctx.cmd.PushConstants(&c, sizeof(c));
        ctx.cmd.Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    });
graph.AddPass("Present", PassFlags::Compute,
    [&](PassBuilder& b) { b.Read(color, Access::SRVCompute); b.Write(out, Access::UAVCompute); },
    [=](PassContext& ctx) { ... });
graph.Execute();
```

pass 作者只做两件事：**声明用到哪些资源、怎么用**；**录制绘制/计算命令**。其余全部自动：

| 自动完成 | 说明 |
|---|---|
| 裁剪 | 反向遍历：输出没人用的 pass 被裁掉（`NeverCull` 可强制保留） |
| barrier | 按声明顺序推导状态变化；连续 UAV 写自动插 UAV barrier；导入资源结束时转到指定 final 状态 |
| 瞬态资源 | 首次使用时创建、显存复用、执行完延迟释放 |
| 并行录制 | 存活 pass 按 `passesPerCommandList` 分块，每块一个 command list，在任务系统上并行录制，按声明顺序提交 |
| 跨帧资源 | `ImportTexture`（外部资源）/ `ExtractTexture`（把本帧临时资源留到下一帧，例如 TAA history、路径追踪累积） |

### 5.2 调试与错误提示（面向 AI）

- 声明期就校验：读声明用了可写状态、写声明用了只读状态、资源缺少对应 usage 标志
  （例如写 UAV 但创建时没有 `UnorderedAccess`）、同一 pass 对同一资源既读又写不同状态 —— 报错信息都带 pass 名和资源名。
- 执行期校验：pass 里用了没在 setup 中声明的资源会直接报错。
- `DumpText()`：每个 pass 的状态、barrier、读写；每个资源的生命周期、堆偏移。
- `DumpGraphviz()`：输出 dot 图（被裁剪的 pass 灰色，导入资源蓝色，提取资源绿色）。

### 5.3 与 UE RDG 的差异

UE 用 shader parameter struct 让 RDG 反射出资源依赖；这里用显式 `PassBuilder` 声明（Frostbite 风格），
不需要宏和反射，代码更短，AI 更容易写对。后续可在 HLSL/C++ 共享头之上加代码生成，把 push constant 结构和声明自动对齐。

---

## 6. 扩展性：渲染特性（RenderFeature）

文件：`src/renderer/include/air/renderer/`

作为图形算法试验场，最常见的改动是"加一个新效果"。这件事被设计成**不碰核心代码**：

```cpp
class MyFeature final : public renderer::RenderFeature {
public:
    MyFeature() { mParameters.Add("strength", &mStrength, "Effect strength", 0.0f, 10.0f); }
    bool Initialize(const renderer::FeatureInitContext& ctx) override;  // 建管线、持久资源
    void AddPasses(renderer::FrameContext& ctx) override;               // 每帧往 RenderGraph 加 pass
    json::Value GetStatus() const override;                             // 可选：报告里的状态
};
namespace air::features {
void Register_my_feature(renderer::FeatureRegistry& r) { r.Register({"my_feature", "...", [] { ... }}); }
}
```

| 组件 | 作用 |
|---|---|
| `RenderFeature` | 一个自包含的渲染技术：自己的 C++、着色器、测试都在 `src/features/<name>/` |
| `FeatureRegistry` + `air_add_feature()` | CMake 扫描 `src/features/*`，生成 `RegisterBuiltinFeatures()`；新目录即新特性 |
| `ParameterSet` | 有类型、有范围、有说明的参数；CLI `--set feature.param=value`，报告里自动列出 |
| `FrameRunner` | 每帧：建 RenderGraph → 导入输出纹理 → 依次调用特性 → 执行 → 收集统计/命令流/校验错误。app、测试、harness 共用这一条路径 |
| `GpuScene` | 特性读取的场景数据（TLAS、实例、材质）；特性不拥有场景资源 |

特性按添加顺序执行并写同一张输出纹理，后面的特性可以做前面结果的后处理
（例如 `--features pathtracer,example`）。

新建特性：`python3 tools/air.py new-feature <name>` 从 `src/features/example` 生成完整骨架（含测试），立即可编译可运行。

---

## 7. AI 迭代 harness

目标：AI（或人）改一行代码后，**一条命令**就能知道"对不对、哪里不对"，不需要显卡、不需要看屏幕。

| 工具 | 命令 | 输出 |
|---|---|---|
| 门禁 | `python3 tools/air.py check [--full]` | 每步 `ok/FAIL`，失败时附最后几十行输出；`--json` 给机器可读汇总；退出码 |
| 单测 | `python3 tools/air.py test [filter]` | core / rhi / rdg / renderer / 各特性的测试，全部跑在 Null 后端 |
| 渲染 | `python3 tools/air.py run -- render ... --report r.json` | JSON 报告：每帧 pass/barrier/显存复用统计、特性参数与状态、校验错误、关闭后的显存泄漏 |
| 命令流 | `aircli render --dump-commands` | 一帧的每条 GPU 命令（barrier、dispatch、事件），带资源名 |
| 快照 | `tests/golden/*.txt` + `air.py golden-update` | 参考场景的 RenderGraph 和命令流；任何行为变化都是可审阅的文本 diff |
| 图像对比 | `aircli compare ref.pfm test.pfm --diff d.ppm` | rmse / psnr / 最大误差 / 差异像素比例（JSON），差异热力图 |
| 脚手架 | `air.py new-feature <name>` | 新特性骨架 |
| 规范 | `air.py format` / `tidy` / `layers` | 格式、静态检查（命名 + bug 类）、模块分层 |

设计原则：

1. **一切可文本化**：句柄可打印、命令可捕获、RenderGraph 可 dump、显存可统计、报告是 JSON。
2. **Null 后端 + 校验**：barrier 状态错误、未声明的资源访问、usage 缺失、参数越界都会带着 pass/资源/参数名报错。
3. **确定性**：并行录制与串行录制产生完全相同的命令流；golden 快照不含时间、线程等不稳定信息。
4. **退出码有意义**：0 通过，1 检查失败（校验错误、泄漏、图像差异超阈值），2 用法或环境错误。
5. **CI 跑同一套命令**：`.github/workflows/ci.yml` 调用的就是 `tools/air.py`，本地通过即 CI 通过。

图像 golden 测试：`Image` 支持 PFM（线性 HDR，精确）和 PPM（8-bit 预览）。真机后端加上回读后，
`render --screenshot` 输出 PFM，与 `tests/golden/images/` 比对；比对工具和指标已经就位。

---

## 8. GPU 路径追踪（离线也要快）

文件：`src/features/pathtracer/`（含着色器 `shaders/` 与测试 `tests/`）。
作为特性注册为 `pathtracer`，参数：`spp`、`maxBounces`、`targetSamples`、`exposure`。

### 8.1 当前实现

- 硬件光追：DXR 1.1 / `VK_KHR_ray_tracing_pipeline`，一个 raygen + miss + closest-hit。
- **弹射循环写在 raygen 里**，`maxRecursionDepth = 1`：栈小、调度友好，是 UE Path Tracer 和大多数 GPU 离线渲染器的做法。
- 渐进累积：持久的 `RGBA32F` 累积 buffer（rgb = radiance 和，a = 样本数），每帧追加 `samplesPerPixelPerFrame` 个样本；
  `Resolve` pass 做平均 + 曝光 + ACES tonemap。
- 相机或场景变化 → `ResetAccumulation()`；`IsConverged()` 达到 `targetSamples` 即可输出一帧。
- 材质：目前只有 Lambert + 自发光，天空为简单渐变；Russian roulette 从第 3 次弹射开始。
- 着色器与 C++ 共享 `PathTracerShared.h`，push constant / 结构体布局一处定义，`static_assert` 检查大小。

### 8.2 "离线也要快" 的路线图

| 优先级 | 项目 | 收益 |
|---|---|---|
| P0 | 下一事件估计（NEE）+ MIS，面光源/环境光重要性采样 | 同样 spp 下噪声大幅下降 |
| P0 | GGX 金属/粗糙度材质、纹理（bindless） | 能出像样的画面 |
| P0 | 离线模式：每帧高 spp、多次 DispatchRays 合批、不等待 vsync，跑满 GPU | 吞吐 |
| P1 | AI 降噪（OIDN / OptiX，albedo+normal 辅助 buffer） | 低 spp 出成片 |
| P1 | 蓝噪声 / Sobol 低差异序列替代 PCG | 收敛更快 |
| P1 | 自适应采样：按像素方差停止 | 把样本花在噪声大的地方 |
| P2 | Shader Execution Reordering（SER，RTX 40+）/ 材质排序 | 发散路径的 SIMD 效率 |
| P2 | 分块渲染（tile）避免 TDR，支持超大分辨率 | 稳定性 |
| P2 | 运动模糊、景深、体积 | 影视化效果 |

### 8.3 输出影视化视频

离线渲染一帧 = 循环跑帧直到收敛 → `Readback` 回读 → 写 EXR/PNG 序列 → ffmpeg 合成。
相机路径/动画由上层脚本（后续可由 AI 生成）驱动。

---

## 9. 构建

```bash
python3 tools/air.py check                     # 配置 + 构建 + 全部检查
python3 tools/air.py run -- render --frames 2 --dump-commands
# 或直接用 CMake：
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build && ctest --test-dir build
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `AIR_RHI_D3D12` | Windows 上 ON | D3D12 后端 |
| `AIR_RHI_VULKAN` | OFF | Vulkan 后端（需要 Vulkan SDK） |
| `AIR_BUILD_TESTS` | ON | 单元测试 |
| `AIR_ENABLE_ASAN` / `AIR_ENABLE_TSAN` | OFF | 内存 / 线程错误检测 |

着色器：安装 DXC 后可编译各特性的 `shaders/`（`-T lib_6_6` 出 DXIL，加 `-spirv` 出 SPIR-V），构建集成会在 D3D12 后端落地时加入。

---

## 10. 下一步

1. D3D12 后端：设备、队列/fence、堆提供者、bindless 描述符堆、Enhanced Barriers、DXR 管线与 shader table。
2. 着色器编译集成（DXC，CMake 自动编译 + 运行时热重载）。
3. 场景系统：glTF 加载 → 顶点/索引 buffer → BLAS/TLAS → 实例/材质 buffer。
4. 路径追踪 P0 项（NEE+MIS、GGX、离线模式）。
5. 图片回读 + `render --screenshot`（PFM/EXR）+ 图像 golden 测试 + 序列帧渲染。
6. Vulkan 后端。
