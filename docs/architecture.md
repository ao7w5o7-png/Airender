# Airender Architecture

> Goal: a renderer that AI can write code for, drive and debug directly. It supports both
> real-time rendering and film-quality offline rendering, should eventually produce cinematic
> video from AI + renderer, and doubles as a testbed for quickly validating graphics algorithms.

This document describes the architecture of the first skeleton. The design follows established
RHI / render graph / task system patterns, simplified to prioritise **thread safety,
debuggability and AI friendliness**.

---

## 1. Layers

```
 apps/aircli            command line driver: render / features / compare, JSON reports
 tools/air.py           developer harness: check / test / run / golden-update / new-feature / layers
      │
 features/<name>        rendering feature plugins: pathtracer (GPU path tracing), example (template) ...
      │                 independent of each other, discovered by CMake, registered by name
 renderer               extension point: RenderFeature, FeatureRegistry, ParameterSet, FrameRunner, GpuScene
      │
 rdg (RenderGraph)      declarative passes: culling, automatic barriers, transient memory aliasing, parallel recording
      │
 rhi                    device abstraction: D3D12 / Vulkan / Null backends, bindless, GPU memory allocator
      │
 core                   task system (Task / Pipe / ParallelFor), handle pool, logging, asserts, JSON, images
```

Dependencies point strictly downwards: a layer only depends on the layers below it, `core` knows
no graphics API, and features never depend on each other. `python3 tools/air.py layers` checks
every `#include "air/..."` against this rule and runs in CI.

| Module | Directory | Contents |
|---|---|---|
| Task system | `src/core` | `Launch`, prerequisites, `Pipe`, `ParallelFor` |
| RHI | `src/rhi` | `Device` / `CommandList` / `Access` states, backends |
| GPU memory | `src/rhi/GpuMemoryAllocator` | TLSF pools per memory type, dedicated heaps |
| Render graph | `src/rdg` | `RenderGraph`, barriers, transient allocator |
| Feature framework | `src/renderer` | `RenderFeature` plugins, `FrameRunner` |
| Path tracer | `src/features/pathtracer` | Progressive hardware ray traced path tracer |

---

## 2. Task System (Multithreading)

File: `src/core/include/air/core/TaskSystem.h`

### 2.1 Model

- **Task**: `TaskSystem::Launch(name, fn, prerequisites, priority)` returns a `TaskHandle`. A task
  is scheduled once all its prerequisites have completed; tasks form a DAG.
- **Waiting helps**: while waiting, `TaskHandle::Wait()` executes other queued tasks (task
  retraction). Waiting for a subtask on a worker thread therefore cannot stall
  the pool (see `TaskSystem_NestedWaitOnSingleWorkerDoesNotDeadlock`).
- **Pipe**: tasks launched through the same `Pipe` run **one at a time** in launch order, on any
  thread. This is the preferred way to get thread safety: hand non-thread-safe state to a pipe and
  submit every access as a pipe task, and no lock is needed.
- **ParallelFor**: the calling thread participates; work is split dynamically.
- **Priorities**: High / Normal / Background queues.

### 2.2 Thread-safety contract (whole codebase)

| Object | Rule |
|---|---|
| `TaskSystem` / `TaskHandle` / `Pipe` | Every function may be called from any thread |
| `rhi::Device` | Every function is thread-safe; resources may be created on any thread; `Destroy*` is deferred until the GPU is done |
| `rhi::CommandList` | **Not** thread-safe: one thread records a list at a time; different lists may be recorded in parallel |
| `rhi::Device::Submit` | Thread-safe; submissions to one queue are serialised in call order |
| Building an `rdg::RenderGraph` | Single threaded (the constructing thread), checked at runtime |
| RDG pass execute lambdas | May run concurrently with other passes on workers: only use `PassContext` and by-value / immutable captures |
| `GpuMemoryAllocator` | Thread-safe, one mutex per pool |
| `FrameRunner` / `RenderFeature` / `ParameterSet` | Called on the thread that drives the frame loop; parameters change only between frames |
| `HandlePool` | Thread-safe; handles carry a generation, so a stale handle is detected instead of dereferencing freed memory |

### 2.3 Next steps

- Replace the global queue with per-thread work-stealing deques (Chase-Lev) to reduce contention.
- Add render-thread and RHI-thread pipes for a three-stage pipeline: game logic → render command
  building → RHI submission.

---

## 3. RHI (D3D12 + Vulkan)

Files: `src/rhi/include/air/rhi/`

### 3.1 Design points

1. **Handles, not raw pointers**: `BufferHandle`, `TextureHandle`, ... are `{index, generation}`
   value types. They can be printed, compared and passed across threads, and a stale handle is
   reported as an error. This makes AI-driven debugging much easier.
2. **Bindless first**: every shader resource is accessed through an index into the global
   descriptor heap (D3D12 SM 6.6 `ResourceDescriptorHeap`, Vulkan descriptor indexing). A pass only
   puts indices into push constants; there are no root signature or descriptor set layouts to
   manage. New passes need almost no boilerplate, and it is the binding model that AI gets wrong
   least often.
3. **Access states + barriers**: resource state is the `rhi::Access` bit set and a barrier is
   `before -> after`. The D3D12 backend maps them to resource states (`ResourceBarrier`); Enhanced
   Barriers and Vulkan synchronization2 fit the same model. The render graph generates barriers;
   pass code never writes them by hand.
4. **One HLSL source, two bytecodes**: DXC compiles to DXIL (D3D12) and SPIR-V (Vulkan).
5. **Command capture**: after `Device::SetCommandCapture(true)` every submitted command becomes a
   readable line such as `[Graphics] Barrier [tex 'SceneColor' UAVCompute -> SRVCompute]`. The Null
   backend also validates that each barrier's `before` matches the resource's real state and emits
   a `VALIDATION:` line when it does not. AI can see what a frame did, and what went wrong,
   without looking at a screen.

### 3.2 Backends

| Backend | Status | Notes |
|---|---|---|
| Null | ✅ working | No GPU; real memory bookkeeping, command capture and barrier validation. Used by unit tests, CI and GPU-less debugging |
| D3D12 | ✅ working | Agility SDK 1.721 preview (1.619 retail without `AIR_D3D12_PREVIEW`), SM 6.6 bindless, DXR 1.1 pipelines (1.2 shader execution reordering when available), Shader Model 6.10 linear algebra (§3.4), graphics + async compute queues, per-queue GPU timestamps, debug layer and DRED with `enableValidation` |
| Vulkan | 🚧 stub | Plan: Vulkan 1.3 (dynamic rendering, sync2, timeline semaphores), KHR ray tracing pipeline + ray query |

### 3.3 Queues

There are `Graphics`, `Compute` and `Copy` queues. The render graph submits to Graphics, and to
Compute for passes flagged `PassFlags::AsyncCompute` (§5.4).

- `Device::Submit(queue, lists)` returns the fence value the submission signals on its queue.
  `Device::QueueWait(waiter, signaler, value)` makes one queue wait on the GPU until another
  queue's fence reaches `value`; the CPU does not block.
- `IsAccessLegalOnQueue(access, queue)`: the compute queue cannot use, or transition from,
  graphics-only states (render targets, depth, vertex input, pixel-shader reads `SRVGraphics`,
  present); the copy queue only copies. On D3D12 `SRVCompute` maps to
  `NON_PIXEL_SHADER_RESOURCE`, so compute reads are legal on both queues.
- The Null device validates cross-queue use: touching a resource last used on another queue
  without waiting for that use, or using a state the queue does not support, is a `VALIDATION:`
  error. Captured commands carry the queue (`[Compute] Dispatch ...`, `[Graphics] Wait Compute fence 3`).
- GPU timestamps are per queue: each queue has its own query heap and readback buffer (queues
  running at the same time must not write one resource). `GetClockCalibration` maps every
  queue's ticks onto the CPU clock, so `GpuFrameTimings` puts all queues on one timeline and each
  scope names its `queue`. `frameMs` spans the first to the last timestamp; `busyMs` is the time
  some queue was inside a scope (`ComputeBusyMs`). They differ by waits: a queue waiting for
  another, or, at vsync, a submission that writes a swapchain buffer waiting on the GPU for it
  (the editor's last batch, once async compute splits the frame). The editor shows busy time.

### 3.4 Linear algebra (Shader Model 6.10)

Small neural networks (§8.11) multiply on the GPU's matrix units through Shader Model 6.10's
thread-scope matrices (`dx/linalg.h`, the successor of cooperative vectors): each thread multiplies
its own vector by a matrix in memory, and the hardware gathers a wave's.

- **Toolchain** (`AIR_D3D12_PREVIEW`, on by default): the preview Agility SDK (its runtime goes to
  `.\D3D12\`, and the executable exports `D3D12_PREVIEW_SDK_VERSION`) and the preview DXC, which
  compiles only the shaders marked `// air-shader-target: cs_6_10` (with 16-bit types, DXIL only;
  the manifest records each shader's compiler for hot reload and `air.py shaders`). Without it
  those shaders are not built and their features fall back to float32 kernels.
- **Device**: `DeviceDesc::experimentalShaderModels` enables D3D12's experimental shader models
  before the device exists (Windows Developer Mode only; a device that fails with them is created
  again without). `DeviceCaps::linearAlgebra` reports linear algebra tier 1 with what the networks
  use: float16 matrix x vector products, transposed too, float16 outer products accumulated into
  float32 matrices in a buffer. The Null device pretends, so the passes run headless.
- **Layouts**: the optimal layouts (`MatrixLayout::MulOptimal`, `OuterProductOptimal`) are the
  device's own; `Device::GetMatrixSize` gives their sizes and `CommandList::ConvertMatrices` copies
  matrices between layouts and types (source in `SRVCompute`, destination in `UAVCompute`, offsets
  128-byte aligned).
- **What this driver does** (RTX 5070, driver 617.42, Agility SDK 1.721.3-preview), measured with
  standalone probes and `aircli mlp-check`:
  - a ray tracing pipeline whose ray generation shader multiplies crashed when created; the same code
    in a compute shader runs (the cache's trace kernels are compute shaders with inline ray queries);
  - multiplications made only by the threads that took a branch after a loop with breaks returned
    NaN for about 1% of threads, although simple divergence (a branch, an early return, loops of
    different lengths) was exact: the kernels call them with every thread of the wave;
  - float16 x float16 products are summed in float16 blocks (about 7e-4 relative error per output,
    against 1.4e-4 for a single rounding);
  - the debug layer does not validate Shader Model 6.10 shaders (it says so once per pipeline).

### 3.5 Shader language: HLSL, not Slang

Every shader is HLSL compiled by DXC. Slang (a shading language with automatic differentiation,
modules and generics, the language of NVIDIA's neural shading samples) was assessed for the neural
work (§8.11, §8.16, §8.17) and is not needed:

- **What neural rendering needs from the language**: matrix x vector products on the matrix units
  (DXC's `dx/linalg.h`, §3.4), and gradients for training. The networks are small and of one shape
  (`MlpShared.h`), so their backward pass is written by hand once (`MlpGradients.hlsli`) and checked
  against the CPU twin (`aircli mlp-check`); automatic differentiation would save that one file.
- **Third-party code**: LibNTC's shader headers compile with DXC (`Inference.hlsli`; its
  cooperative-vector variant includes `dx/linalg.h` unless compiled as Slang); its only Slang shader
  is compiled into `libntc.dll`. NRD and DLSS ship HLSL or binaries.
- **What Slang would cost**: a second compiler beside DXC (which the preview linear algebra needs
  anyway) in the build, hot reload and `air.py shaders`; and the CPU twins compile the integrator's
  `.hlsli` as C++ (`HlslCompat.h`), which Slang's generics and interfaces would not.
- **When to revisit**: networks of many shapes whose gradients are a burden to write and check by
  hand, or a backend DXC does not reach (Metal). Slang reads HLSL, so `.slang` files could be added
  through a second compiler entry in `AirShaders.cmake` without porting the existing shaders.

---

## 4. GPU Memory Management (Multiple Memory Types)

File: `src/rhi/include/air/rhi/GpuMemoryAllocator.h`

### 4.1 Memory types

`MemoryLocation` says where a resource lives:

| MemoryLocation | D3D12 | Vulkan | Used for |
|---|---|---|---|
| `DeviceLocal` | `HEAP_TYPE_DEFAULT` | `DEVICE_LOCAL` | Textures, render targets, BVHs, most buffers |
| `Upload` | `HEAP_TYPE_UPLOAD` | `HOST_VISIBLE \| HOST_COHERENT` | Per-frame CPU constants, upload staging |
| `Readback` | `HEAP_TYPE_READBACK` | `HOST_VISIBLE \| HOST_CACHED` | Screenshots, offline render readback, GPU debug data |
| `DeviceLocalHost` | `HEAP_TYPE_GPU_UPLOAD` (ReBAR) | `DEVICE_LOCAL \| HOST_VISIBLE` | CPU writes straight into VRAM, saving a copy; falls back to Upload when unsupported |

### 4.2 Structure

```
GpuMemoryAllocator           thread-safe; one pool per memoryTypeKey, one mutex per pool
  └─ Pool (memoryTypeKey)    D3D12: heap type × tier-1 category (buffers / RT-DS textures / other textures)
       │                     Vulkan: memoryTypeIndex
       ├─ Block (256 MiB)     ID3D12Heap / VkDeviceMemory created by the backend's IGpuHeapProvider
       │    └─ TlsfAllocator  O(1) allocate/free, coalesces neighbours, any power-of-two alignment
       └─ Dedicated           resources ≥ 64 MiB get their own heap (large render targets, huge BVHs)
```

- **TLSF** (Two-Level Segregated Fit): constant-time allocation and free with low fragmentation,
  a common choice for GPU allocators in game engines. It only manages offsets and never touches
  GPU objects, so it can be stress tested on its own (`Tlsf_RandomStress`).
- Empty heaps: each pool keeps at most one empty heap to avoid create/destroy thrashing.
- Upload / Readback use smaller blocks (64 MiB) because they consume system memory / BAR space.
- Statistics: `Device::GetMemoryStats()` reports heaps / reserved bytes / used bytes / allocation
  count / dedicated count per memory type.

### 4.3 Transient memory in the render graph (aliasing and reuse)

Temporary resources created inside the render graph do not use the pools above. Memory is reused
twice: within a frame, by resources whose lifetimes do not overlap, and across frames, by the
transient pool.

**Within a frame** (`AllocateTransientMemory`, after the barriers so the cross-queue waits are
known):

1. each resource records its first and last pass on every queue it is used on;
2. `Precedes(a, b)` holds when every use of `a` happens before every use of `b` on the GPU: on one
   queue by pass order, across queues when `b`'s first pass starts after a wait that covers `a`'s
   last batch (`mReach`: per batch, the latest batch of each queue it is ordered after, through
   chains of waits too);
3. resources are packed first-fit in descending size order; two may share memory when one
   precedes the other. A transient of an async compute pass therefore shares memory with
   graphics transients its waits order it against (e.g. one consumed before the pass's input was
   produced), and with nothing that may run beside it.

The first barrier of a resource carries a `discard` flag: the backend issues an aliasing barrier
and transitions from the barrier's `before` state.

**Across frames** (`rdg::TransientResourcePool`, owned by FrameRunner; `poolTransients = false`
or `aircli render --no-transient-pool` turns it off):

- a frame's transients go into a pooled heap whose last frame the GPU has finished (its graphics
  fence passed); the least recently used fitting heap is taken, so with N frames in flight N heaps
  alternate, each laid out like the frame before it. Heap sizes are rounded up to 16 MiB.
- a transient described exactly like one placed at the same offset of that heap before reuses the
  placed resource and its bindless descriptors. The pool remembers the state the frame left it in
  and the graph patches the next first barrier's `before` to it. A transient first used on the
  compute queue that a frame would leave in a graphics-only state ends that frame in common
  (a final barrier), or the compute queue could never take it back.
- heaps and resources unused for 120 frames are released; FrameRunner calls `Trim()` sooner when
  the process uses over 95% of its VRAM budget (checked every 120 frames).

In a steady frame nothing is created: `RenderGraph::Stats` reports `transientHeapsCreated`,
`transientResourcesCreated` and `transientResourcesReused` next to `transientHeapSize` (with
aliasing) and `transientBytesUnaliased`; FrameResult (and its JSON) adds the pool's totals
(`transientPool`). Measured on door at 1080p with DLSS (7 transients, D3D12 debug build): 4.7 ms
per frame on the CPU instead of 5.7 ms, two 48 MiB heaps instead of a 41 MiB heap per frame
(three alive at a time while the GPU catches up), and no creation after the second frame.

### 4.4 Deferred release

`Destroy*` only records the request; `EndFrame()` releases objects recorded `framesInFlight`
frames ago, so the GPU never touches freed memory.

---

## 5. Render Graph (Easy Pass Authoring)

File: `src/rdg/include/air/rdg/RenderGraph.h`

### 5.1 Writing a pass

```cpp
rdg::RenderGraph graph(device, &tasks);
auto color = graph.CreateTexture({.format = Format::RGBA16_FLOAT, .width = w, .height = h,
                                  .usage = TextureUsage::UnorderedAccess | TextureUsage::ShaderResource,
                                  .debugName = "SceneColor"});
auto out = graph.ImportTexture(backbuffer, Access::None, Access::Present);

graph.AddPass("Shade", PassFlags::Compute,
    [&](PassBuilder& b) { b.Write(color, Access::UAVCompute); },          // declare: writes color
    [=](PassContext& ctx) {                                               // execute: record commands only
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

A pass author does two things: **declare which resources are used and how**, and **record draw
/ dispatch commands**. Everything else is automatic:

| Automatic | Details |
|---|---|
| Culling | Reverse walk: passes whose outputs nobody consumes are removed (`NeverCull` keeps them) |
| Barriers | State changes derived in declaration order; consecutive UAV writes get UAV barriers; imported resources end in their requested final state |
| Transient resources | Created on first use, memory aliased, released (deferred) after execution |
| Parallel recording | Live passes are split into chunks of `passesPerCommandList`, each chunk is recorded into its own command list on the task system, and lists are submitted in declaration order |
| Async compute | Passes flagged `AsyncCompute` run on the compute queue; cross-queue waits and hand-over barriers are inserted (§5.4) |
| Cross-frame resources | `ImportTexture` (external resources) / `ExtractTexture` (keep a transient resource for the next frame, e.g. TAA history, path tracing accumulation) |

### 5.2 Debugging and error messages (for AI)

- Checked at declaration time: a read declared with a writable state, a write declared with a
  read-only state, a missing usage flag (e.g. a UAV write on a texture created without
  `UnorderedAccess`), one pass using one resource in two different states. Every message names
  the pass and the resource.
- Checked at execution time: using a resource that was not declared in setup is an error.
- `DumpText()`: per pass state, barriers, reads and writes; per resource lifetime and heap offset;
  with async compute, the submissions: queue, passes, waits and hand-over barriers, e.g.
  `(3) Graphics: [3], waits for (2), final barriers`.
- `DumpGraphviz()`: a dot graph (culled passes grey, imported resources blue, extracted resources green).

### 5.3 Explicit dependency declaration

Some render graphs reflect resource dependencies from shader parameter structs. Here passes
declare them explicitly through `PassBuilder`: no macros or reflection, shorter
code, and easier for AI to get right. Code generation on top of the shared HLSL/C++ headers can
later keep push constant structs and declarations in sync automatically.

### 5.4 Async compute

Flag a compute pass `PassFlags::Compute | PassFlags::AsyncCompute` and it runs on the compute
queue, overlapping the graphics passes around it. Good candidates depend on little and are
needed late: the example feature's pattern today; BVH refits, light structure builds and
denoiser pre-passes later. The pass is written as usual, except that it may only use states
the compute queue supports: `AddPass` rejects `AsyncCompute | Raster`, and `Read`/`Write` reject
graphics-only accesses on an async pass.

The graph schedules it in `Compile()`:

1. **Batches**: live passes are split, in declaration order, into runs on one queue; each batch
   is one submission (its command lists still record in parallel). Batch 0 is a graphics
   prologue without passes.
2. **Fork**: the first compute batch waits at least for the prologue, which is then submitted
   even when empty. Without it, a compute batch with no graphics input starts as soon as it is
   submitted, while the graphics queue (frames in flight behind the CPU) may still be on the
   previous frame and using an imported resource the batch writes (a LUT, a reservoir buffer).
   With it, compute work overlaps this frame's graphics passes, and a frame's GPU timings
   (first to last timestamp) cover only that frame.
3. **Waits**: a use of a resource waits for its last write or change of state on another queue
   (read after write); a write or a change of state also waits for every read on the other queues
   since then (write after read). Reads that leave the state as it is run side by side on both
   queues (the volume probes read the shadow maps on the compute queue while the trace reads them
   on the graphics queue, §8.13), and so does the first use of an imported resource in the state
   its import declared. Waiting for a later batch of a queue covers its earlier ones, so a batch
   keeps one wait per queue.
4. **Hand-over**: a transition the compute queue cannot make (e.g. `RenderTarget -> SRVCompute`)
   is recorded at the end of the graphics batch that used the resource last (the prologue for an
   imported resource), and the compute batch waits for that batch.
5. **Join**: the graph's final barriers are recorded on the graphics queue after it waits for the
   last compute batch (an epilogue batch is added when needed), so the frame's graphics fence and
   imported resources' final states include the compute work. The next frame's graphics passes
   therefore come after this frame's compute work, and its compute passes after the fork.
6. **Memory**: a transient touched by an async pass shares memory only with transients the waits
   order it against (§4.3).

`DumpText()` prints the schedule under `Submissions:`, one line per submission, e.g.
`(0) Graphics: fork`, `(1) Graphics: [0] [1]`, `(2) Compute: [2], waits for (0)`,
`(3) Graphics: [3], waits for (2), final barriers`.

`RenderGraph::Config::enableAsyncCompute = false` runs async passes on the graphics queue in
declaration order, for A/B timing: `aircli render --no-async-compute`, or the editor's Debug >
Frame > Async compute. `Stats` has `asyncComputePassCount`, `submissionCount` and
`queueWaitCount` (frame JSON: `asyncComputePasses`, `submissions`, `queueWaits`). The editor's
GPU profiler draws one lane per queue; `aircli render --profile --report r.json` writes every
GPU scope with its `queue`, `beginMs` and `ms`, so overlap can be checked from the report.

---

## 6. Extensibility: Render Features

Files: `src/renderer/include/air/renderer/`

In a graphics testbed the most common change is "add a new effect". That is designed to happen
**without touching core code**:

```cpp
class MyFeature final : public renderer::RenderFeature {
public:
    MyFeature() { mParameters.Add("strength", &mStrength, "Effect strength", 0.0f, 10.0f); }
    bool Initialize(const renderer::FeatureInitContext& ctx) override;  // pipelines, persistent resources
    void AddPasses(renderer::FrameContext& ctx) override;               // add this frame's passes
    json::Value GetStatus() const override;                             // optional: state for reports
};
namespace air::features {
void Register_my_feature(renderer::FeatureRegistry& r) { r.Register({"my_feature", "...", [] { ... }}); }
}
```

| Component | Role |
|---|---|
| `RenderFeature` | One self-contained rendering technique: its C++, shaders and tests live in `src/features/<name>/` |
| `FeatureRegistry` + `air_add_feature()` | CMake scans `src/features/*` and generates `RegisterBuiltinFeatures()`; a new directory is a new feature |
| `ParameterSet` | Typed, range-checked, documented parameters; `--set feature.param=value` on the CLI, listed in reports |
| `FrameRunner` | Per frame: create the render graph → import the output texture → run features in order → execute → collect stats / commands / validation errors. The app, tests and harness all use this one path |
| `GpuScene` | Scene data features read (TLAS, instances, materials); features never own scene resources |

Features run in the order they were added and write the same output texture, so a later feature
can post-process an earlier one (e.g. `--features pathtracer,example`).

New feature: `python3 tools/air.py new-feature <name>` generates a complete skeleton (including
tests) from `src/features/example`; it builds and runs immediately.

---

## 7. AI Iteration Harness

Goal: after changing a line of code, an AI (or a human) learns **with one command** whether it is
right and, if not, what is wrong, without a GPU and without looking at a screen.

| Tool | Command | Output |
|---|---|---|
| Gate | `python3 tools/air.py check [--full]` | `ok/FAIL` per step, the tail of the output on failure, `--json` summary, exit code |
| Unit tests | `python3 tools/air.py test [filter]` | Tests for core / rhi / rdg / renderer / every feature, all on the Null backend |
| Render | `python3 tools/air.py run -- render ... --report r.json` | JSON report: per-frame pass / barrier / aliasing stats, feature parameters and status, validation errors, memory leaks after shutdown |
| Command stream | `aircli render --dump-commands` | Every GPU command of a frame (barriers, dispatches, events) with resource names |
| Snapshots | `tests/golden/*.txt` + `air.py golden-update` | Render graph and command stream of reference scenarios; any behaviour change is a reviewable text diff |
| Image comparison | `aircli compare ref.pfm test.pfm --diff d.ppm` | rmse / psnr / max error / differing pixel fraction (JSON) and a diff heatmap |
| Neural network kernels | `aircli mlp-check --backend d3d12` | The GPU's float32 and linear algebra network kernels against the CPU's: outputs, input and weight gradients (JSON, exit 1 when they disagree; §8.11) |
| Scaffolding | `air.py new-feature <name>` | New feature skeleton |
| Conventions | `air.py format` / `tidy` / `layers` | Formatting, static analysis (naming + bug patterns), module layering |

Principles:

1. **Everything is text**: handles print, commands are captured, the render graph dumps, memory
   is accounted for, reports are JSON.
2. **Null backend + validation**: wrong barrier states, undeclared resource use, missing usage
   flags and out-of-range parameters all fail with the pass / resource / parameter name.
3. **Determinism**: parallel and serial recording produce identical command streams; golden
   snapshots contain no timing or thread information.
4. **Meaningful exit codes**: 0 passed, 1 a check failed (validation error, leak, image
   difference over threshold), 2 usage or environment error.
5. **CI runs the same commands**: `.github/workflows/ci.yml` calls `tools/air.py`, so passing
   locally means passing in CI.

Image tests: `Image` reads and writes PFM (linear HDR, exact) and PPM (8-bit preview).
`render --output` reads the final frame back; `aircli compare` measures rmse / psnr against a
reference (see §8.3). For what happens over time in real time (temporal filters, denoisers):
`--output-frames N` also writes the N - 1 frames before the last (`image.<frame>.pfm`), and
`--camera-motion RIGHT,UP,FORWARD[,YAW[,PITCH]]` moves the camera every frame (metres along its own
axes, degrees). Flicker is the change between a static camera's frames; trails are the difference
between a moving camera's frame and a static camera's at the same place (§8.13).

---

## 8. GPU Path Tracer (Physically Based, Energy Conserving)

Files: `src/features/pathtracer/` (shaders in `shaders/`, tests in `tests/`). Registered as the
`pathtracer` feature with parameters `spp`, `maxBounces`, `targetSamples`, `accumulate`, `guides`,
`nee`, `lightTreeBounces`, `cloudEvents` (§8.10), `volumeEvents` (§8.12), `volumeMarch` and the other
`volume*` (§8.13), `restir*` (§8.7), `nrc*` (§8.11), `furnace`, `debugView`, `ser` and `debugScale`.

### 8.1 Units and scene

- Metres for distance, radiance in nits (cd/m²). Emissive materials store a color normalized to
  luminance 1 times an intensity in nits; the environment is a constant radiance.
- Camera exposure follows photographic EV100: `exposure = 1 / (1.2 * 2^EV100)`. The Cornell box
  (3000 nit light) uses EV100 8.
- `renderer::Scene` uploads a `SceneDesc` (meshes, instances, materials, camera, environment):
  device-local vertex/index buffers, one BLAS per mesh and a TLAS. Materials, the light list and
  the per-instance light offsets live in per-frame upload rings so they can be edited live;
  `GpuScene::version` changes whenever a frame's data must be re-accumulated.
- The light list holds the triangles of emissive instances only (all of an instance's triangles,
  in order), so it costs nothing for the non-emissive bulk of a large scene. `instanceLights[i]`
  is instance i's first entry or `kInvalidIndex`; the integrator looks up a hit emitter's entry
  (for its MIS light pdf) as `instanceLights[instance] + primitive`. Making a material emissive in
  the editor rebuilds both in the next ring slot.
- Built-in scenes (`MakeBuiltinScene`, `--scene NAME`, the editor's scene list), all procedural and
  deterministic (fixed-seed PCG, no assets):

  | Scene | Content | What it tests |
  |---|---|---|
  | `cornell_box` | the 2 m box, 3000 nit panel, tall box, gold sphere | default; GPU vs CPU reference |
  | `furnace` | spheres of roughness 0..1 in a uniform environment | energy conservation (`furnace=1`) |
  | `veach_mis` | 4 glossy plates, 4 spherical lights of equal power, r = 3 cm..90 cm | MIS: NEE and BSDF sampling each fail somewhere |
  | `material_grid` | white / red / gold spheres, roughness 0..1, one area light | BSDF appearance |
  | `sphere_field` | ~2400 instances of 2 meshes, 32 materials, physical sky and sun | TLAS scale, dynamic range, light imbalance |
  | `many_lights` | hall with 66 panels + 300 LEDs, radiance over 2 orders of magnitude | light selection (baseline for many-light sampling) |
  | `door` | light reaches the camera's room only through a door left ajar | indirect-dominated transport |
  | `terrain` | 2.1M triangle height field, lake, physical sky and sun | dense geometry, BLAS build and memory |
  | `sunset` | a plain, two spheres and five pillars under a sun 4 degrees high | the physical sky (§8.9), long shadows |
  | `cumulus` | a 4 km meadow, a farmstead and trees under fair-weather cumulus | volumetric clouds (§8.10): their light, shadows, cost |
  | `smoke` | smoke rising from a brazier in a walled yard, the low sun behind it | volumes (§8.12): scattering, NEE in a medium, its shadow |
  | `braziers` | three overlapping plumes over braziers at dusk, a lantern among them | several volumes and light groups near them (§8.13) |
  | `textured` | a courtyard corner of tiles, bricks and riveted metal plates in the afternoon sun | material textures and normal maps (§8.15), neural texture compression (§8.16) |
  | `coated` | the courtyard with clear coats: red car paint, blue lacquer, amber-lacquered tiles, coated plates | clear coats and their network (§8.17) |
  | `car_paint` | candy red, gold pearl, midnight blue, silver and solid black paints on a stretched body and five spheres | metallic flakes under clear coats and the network's flake inputs (§8.17) |
  | `subsurface` | marble, jade, skin and wax balls and a thin jade slab against a low sun behind them | subsurface scattering: light carried beneath surfaces, through thin parts and from lit backs (§8.20) |

  Daylight scenes use the physical sky (§8.9), whose sun has the real angular radius and
  luminance, so they use daylight exposure (EV100 14..15; 11.5 for the low sun of `sunset`).
- File scenes (§8.18; `MakeScene` and `SceneNames` cover both kinds), made from model files that
  `python3 tools/air.py fetch-scenes` downloads into `.deps/scenes`:

  | Scene | Content | What it tests |
  |---|---|---|
  | `picapica` | SEED's PICA PICA assets: a toy factory of plastic machines, pallets, books and seven robots around the mini diorama, on a table in the afternoon sun (886k triangles, 592 instances) | real assets, many small parts and materials, decals (cutouts), emissive LEDs, a clear coat |
  | `sponza` | the Sponza atrium under the sky through its open roof (262k triangles, 25 texture sets) | textures and normal maps at scale, cutout foliage, indirect light in arcades |
  | `helmets` | the flight, damaged and sci-fi helmets of the glTF sample assets on plinths (134k triangles) | physically based texture sets (metal, roughness, normal maps) |

### 8.2 Integrator

The integrator is written once in `shaders/Integrator.hlsli` and `shaders/Bsdf.hlsli` and compiled
both as HLSL and as C++ through `air/renderer/shader/HlslCompat.h` (see coding standards §8). The
GPU path tracer and the CPU reference renderer therefore run the same code; only the hooks
(`TraceClosest`, `TraceVisible`, `LoadInstance`, `LoadMaterial`, `LoadLight`, `BsdfLutLoad`, ...)
differ.

- **BSDF**: GGX with height-correlated Smith G2, sampled with visible normals (spherical caps).
  Kulla-Conty multiple scattering restores the energy lost by single-scatter microfacets; the
  dielectric is a GGX coat (F0 = 0.04) over a Lambert base that receives exactly the energy the coat
  did not reflect. A white material reflects 100% at every roughness and angle (white furnace).
  The albedo tables are built on the CPU at startup (`GetBsdfLut`) and uploaded once.
- **Shading normals** (`EvaluateSurfaceBsdf`, `SampleSurfaceBsdf`): where the shading normal (the
  interpolated one, or a normal map's) leans away from the triangle's, part of the diffuse lobe points
  below the surface, where no light comes from. That part is mirrored in the surface's plane back
  above it (the reflection off a facet of Schuessler et al.'s microfacet-based normal mapping, 2017,
  with the surface as the facet): a direction above the surface also gets the diffuse lobe's value and
  pdf at its mirror image, and a diffuse sample below the surface is mirrored, so sampling, evaluation
  and MIS agree. White furnace, CPU reference at 128 bounces: `textured` -1.76% -> -0.58%, `terrain`
  -0.19% -> -0.03%, `sphere_field` -0.19% -> -0.15% (mostly metal). Mirroring the specular lobes too
  kept all of it (-0.04%, -0.001%, -0.05%) but takes a second evaluation of the whole BSDF, which cost
  every kernel 5-7% of its frame in registers (17% with coats), also where nothing is mirrored; the
  diffuse lobe's mirror costs nothing measurable (`coated` +4%). Coated materials are not mirrored.
- **Light transport**: next event estimation over emissive triangles chosen by power, combined
  with BSDF sampling by the power heuristic (MIS). Emitters are one sided (front = `cross(e1, e2)`).
  The light list is also a Walker/Vose alias table, so choosing a light is O(1) (one or two loads)
  whatever the light count; the alias decision uses its own random number, so the selection
  probability is exactly `probability`, as MIS assumes. A light tree can choose instead (§8.6).
  Russian roulette from the third bounce ends paths without bias; `maxBounces` (32) is a cap only.
  At 8 (the earlier default) `door`, lit through a door ajar, lacked 13% of its indirect light
  (`cornell_box` 1%, `sponza` and `picapica` under 0.05%); 32 costs 1-7% of a real-time frame.
- **Hardware ray tracing**: a DXR ray generation shader runs the whole path (bounce loop in raygen);
  primary, bounce and shadow rays are inline `RayQuery`s. `TraceRay` into a closest-hit shader (20
  byte payload) measured 19-29% slower on every test scene at 1080p (cornell_box 2.00 -> 1.53 ms,
  many_lights 4.00 -> 2.93 ms): the call saves the megakernel's live state around each ray. The miss
  and closest-hit shaders remain for the SER variant. Push constants only carry bindless indices;
  per-frame data comes from rings in the GPU upload heap (`DeviceLocalHost`) where available.
- **Kernel variants** of `PathTracerKernel.hlsli`: `PathTracer.hlsl` (SM 6.6), `PathTracerSer.hlsl`
  (SM 6.9, DXR 1.2 shader execution reordering: bounce rays trace into hit objects, threads regroup
  by hit instance, the hit is read from the object without a closest-hit call; `pathtracer.ser=1` on
  GPUs that reorder, `DeviceCaps::shaderExecutionReordering`) and `PathTracerGi.hlsl`
  (`AIR_PT_RESTIR_GI`: records ReSTIR GI's candidates, §8.8; compiled apart because that state
  halves the plain kernel's speed). `PathTracerNrc.hlsl` and `PathTracerNrcPortable.hlsl` (§8.11)
  end paths in the neural radiance cache; they are compute shaders (`AIR_PT_COMPUTE`) that run the
  same `TracePixel` with inline ray queries. `PathTracerVolume.hlsl` (`AIR_PT_VOLUME`) tracks the
  scene's volume (§8.12) and runs instead of the plain kernel while there is one;
  `PathTracerVolumeShadows.hlsl` (`AIR_PT_VOLUME_SHADOWS`) lights surfaces through the real-time
  volumes' moment shadow maps instead, when they are marched (§8.13).
- **Guides** (albedos, normal/roughness, depth, motion) are written only when an active feature
  lists them in `RenderFeature::ConsumedSlots()` (dlss does): they cost about a tenth of the trace.

#### Performance notes (RTX 5070, 1280x720, 1 spp, `aircli render --profile --no-validation`)

| Scene | Before | Alias table + guides on demand | Change |
|---|---|---|---|
| cornell_box | 0.914 ms | 0.807 ms | -12% |
| veach_mis | 0.694 ms | 0.516 ms | -26% |
| material_grid | 0.940 ms | 0.793 ms | -16% |
| sphere_field | 0.773 ms | 0.627 ms | -19% |
| many_lights | 2.389 ms | 1.700 ms | -29% |
| door | 1.694 ms | 1.555 ms | -8% |
| terrain | 0.671 ms | 0.572 ms | -15% |

- SER measured slower everywhere (+4% to +48%, still +3% to +46% when primary rays skip the
  reorder): shading here is cheap and uniform, and each reorder moves the whole path state of the
  megakernel. That is why it is off by default; revisit it when materials get expensive (textures,
  layered BSDFs) or with a wavefront split that keeps the reordered state small.
- `door` and `many_lights` are dominated by long paths in closed, bright rooms (halving
  `maxBounces` halves the time), not by per-bounce cost.
- Measure on an idle machine. With other processes holding most of the VRAM (two graphics
  debugger sessions held 8 GB here), WDDM pages continuously (the System process shows on a copy
  engine in `\GPU Engine(*)\Utilization Percentage`) and every pass slows down 4x to 180x:
  the same Cornell frame measured 3.17 ms of trace and 1.85 ms of resolve, against 0.91 ms and
  0.010 ms once that memory was freed. Device-local heaps get a high residency priority, which
  helps a little, not enough.
- **Accumulation**: a persistent `RGBA32F` buffer averages samples until `targetSamples`, then
  tracing stops. Camera, scene version or image-affecting setting changes reset it. With
  `accumulate=0` the camera is jittered (Halton) every frame for a temporal denoiser.

### 8.3 Validation

- `tests/BsdfTests.cpp`: white furnace for metal and dielectric, energy bound, reciprocity, sampled
  weight equals evaluation, sampling matches uniform integration; on a surface whose shading normal
  leans 35 degrees away the diffuse lobe keeps its albedo and nothing is gained, and its sampling
  matches uniform integration around the surface (`Bsdf_Surface*`).
- `tests/IntegratorTests.cpp`: an emissive white room collects exactly `B + 1` times the emission
  (B = bounces), NEE and BSDF-only estimators agree, furnace spheres average 0.18.
- `aircli reference` renders the same scene with the CPU reference renderer (BVH + the shared
  integrator, multithreaded). Compare a GPU frame with it:
  ```bash
  aircli reference --scene cornell_box --spp 1024 --output ref.pfm
  aircli render --backend d3d12 --scene cornell_box --frames 300 \
                --output gpu.pfm
  aircli compare ref.pfm gpu.pfm --diff diff.ppm
  ```

### 8.4 Debug views

`pathtracer.debugView` selects what `Resolve` writes: 0 lit, 1 normals, 2 base color,
3 roughness, 4 metallic, 5 first-hit distance, 6 intersection count heatmap (Turbo colormap),
7 path length heatmap, 8 direct light only, 9 indirect light only (the global illumination, before
denoising; `nrd.view=1` shows it denoised, §8.14), 10 what the neural radiance cache predicts the
first surface reflects (§8.11), for the real-time volumes (§8.13) 11 their march's steps, 12 its
adaptive sampling rate, 13 the volumes per ray, and 14 the neural coat (§8.17): blue where the
clear coats' network shades the first surface, green / red where it brightens / darkens the
analytic approximation towards the mirror direction (full at 2x / 0.5x), purple for a coat another
model evaluates, grey elsewhere. `debugScale` scales the distance and heatmap views. `furnace=1` turns every
material white and every emitter off with a uniform environment, so any visible structure is an
energy error.

### 8.5 Feature chain and the blackboard

Features run in three stages (`RenderFeature::GetStage()`), and within a stage in the order they
were added:

| Stage | Resolution | Writes | Features |
|---|---|---|---|
| `Scene` | render (`FrameContext::renderWidth/Height`) | `FrameContext::sceneColor` | pathtracer, nrd |
| `Upscale` | render -> output | `FrameContext::output`, sets `outputWritten` | dlss, taa |
| `Post` | output (`FrameContext::width/height`) | `FrameContext::output` in place | tonemap, example |

The render resolution is the output resolution times `FrameRunnerDesc::renderScale` (`aircli
render --render-scale S`). At scale 1 `sceneColor` *is* the output texture and nothing extra runs.
Below 1 `sceneColor` is a transient texture (`FrameRunner.SceneColor`), and if no Upscale feature
wrote the output (dlss missing, disabled, failed, or a debug view) FrameRunner adds
`FrameRunner.Upscale` (bilinear, `src/renderer/shaders/Upscale.hlsl`) before the Post stage, so
the output is always defined.

Features communicate only through `renderer::Blackboard` (`FrameContext::blackboard`), never by
including each other. FrameRunner asks every feature for the slots it consumes each frame (a feature
a parameter switched off consumes none), and producers skip what nobody consumes. Slots live in
`renderer::slots`:

| Slot | Producer | Contents |
|---|---|---|
| `kDiffuseAlbedo`, `kSpecularAlbedo` | pathtracer (`guides=1`) | First-hit directional albedos of the BSDF's diffuse and other lobes (`GuideAlbedos`) |
| `kNormalRoughness` | pathtracer | World normal + roughness |
| `kViewDepth` | pathtracer | Distance along the view axis in metres |
| `kMotionVectors` | pathtracer | Previous minus current position, in pixels |
| `kColorBeforeTransparency` | pathtracer (real-time volumes, §8.13) | The frame resolved without the volumes over it; consuming it marks a denoiser of the whole frame |
| `kDiffuseSignal`, `kSpecularSignal` | pathtracer (real time, lit view) | The first surface's reflected indirect light per lobe, a = its reflected ray's length (§8.14) |
| `kSurfaceTransmittance` | pathtracer (real-time volumes) | What the volumes leave of the surfaces' light (absent: 1) |
| `kRealTime` | pathtracer | The frame holds its own jittered samples only (not accumulated) |
| `kJitterX/Y`, `kHistoryReset` | pathtracer | Sub-pixel jitter; history is invalid this frame |
| `kOutputIsDisplay` | pathtracer | Output is a display-referred debug view: do not tonemap or denoise |

- **dlss** (Windows, D3D12, `AIR_WITH_DLSS`): NVIDIA DLSS Ray Reconstruction (NGX DLSS-D), the
  Upscale stage: denoises `sceneColor` with the guides above and writes the output. The NGX mode
  follows the render scale: 1 = DLAA, 0.667 = Quality, 0.58 = Balanced, 0.5 = Performance,
  0.333 = Ultra Performance. Use Quality or lower for path tracing: at 2790x1125 Quality traces
  44% of the pixels and RR itself is far cheaper than DLAA, with no visible loss on the Cornell
  box (PSNR against a 2048 spp reference: DLAA 43.9 dB, Quality 46.4 dB). Run the path tracer
  with `accumulate=0, spp=1` in front of it; its jitter sequence has 8 x (output / render)^2
  phases (at least 32). The jitter is passed negated (NGX takes the image shift, the opposite of
  the sample offset); a static camera must give a steady image (`dlss.flipJitter=1` shows the
  wrong sign for comparison). The NGX feature is created in `Initialize()`; if NGX rejects the
  size it passes the frame through. It refuses to run on other backends. When a producer publishes
  `kColorBeforeTransparency` it passes it as RR's color-before-transparency guide, from which RR
  tells transparent layers from the surfaces behind them (§8.13).
- GPU timings are only meaningful when the GPU has VRAM to spare: with other processes (graphics
  debuggers, IDEs) holding most of it, WDDM pages resources to system memory and every pass slows
  down by an order of magnitude. Check `vramUsageBytes` / `vramBudgetBytes` in the report and the
  per-process usage before trusting a profile.
- **tonemap**: exposure is applied by the path tracer; tonemap maps HDR to display with `clamp`,
  `aces` or `neutral` (Khronos PBR Neutral), highlights NaN/Inf (magenta) and negative values
  (cyan), and optionally encodes sRGB.

### 8.6 Many lights: the light tree

Choosing the light by power ignores where the shading point is: in `many_lights` most samples go
to panels across the hall that barely light the point. The light tree (Conty Estevez and Kulla,
"Importance Sampling of Many Lights with Adaptive Tree Splitting", 2018, in the form of pbrt-v4)
chooses by power, distance and orientation as seen from the point instead.

- **Build** (`renderer::BuildLightTree`, `src/renderer/src/LightTree.cpp`): a binary tree over the
  light list, split by the surface area orientation heuristic (12 buckets per axis; median splits
  past depth 40 bound the depth). A node (`gpu::LightTreeNode`, 64 bytes) stores its box, its power
  and an orientation cone: the emitters' normals lie within `cosThetaO` of `axis`, and each emits
  within `cosThetaE` (90 degrees for these one-sided triangles) beyond that. Every light stores the
  bits of its path from the root (`treeTrailLow/High`, `treeDepth`); lights without power are not
  in the tree. Built with the light list and uploaded with it in the same ring slot, so editing
  emission rebuilds both.
- **Sampling** (`SampleLightTree` in `Integrator.hlsli`): from the root, compute the importance of
  both children (power x a bound of the emitter cosine / squared distance, times a bound of the
  receiver cosine) and descend into one with probability proportional to it; the product of the
  choices is the light's probability. Two exact tests give zero importance, so samples are not
  spent where nothing can arrive: the whole box is below the receiver's tangent plane (the box's
  support function), or the node's emitters all face one way (`cosThetaO` about 1, such as a group
  of coplanar panels) and the whole box faces away from the point. When both children are zero no
  light is chosen: a valid sample of zero.
- **MIS**: a BSDF-sampled ray that hits an emitter needs the probability the tree would have given
  that light at the previous vertex; `LightTreePmf` follows the light's trail from the root.
- **Where**: `pathtracer.lightTreeBounces=N` uses the tree at the first N path vertices and the
  alias table beyond. `aircli reference --light-tree-bounces N` does the same on the CPU.

Measured (RTX 5070, idle VRAM). Noise is relMSE at 16 spp, 640x360, against an 8192 spp
reference; time is the `PathTracer.Trace` pass at 1920x1080, 4 spp:

| Scene | Power | Tree at vertex 1 | Tree everywhere |
|---|---|---|---|
| `many_lights` noise | 2.27 | 2.01 | 1.16 |
| `many_lights` trace | 13.4 ms | 27.9 ms | 42.2 ms |
| `sphere_field` noise | 0.082 | 0.077 | 0.111 |
| `sphere_field` trace | 5.1 ms | 11.3 ms | 13.1 ms |

The tree halves the noise per sample among many lights, but in `many_lights` (36k emissive
triangles) each choice walks 16 to 19 levels of dependent loads in divergent warps, so in
time-to-quality it loses to the O(1) alias table, and in
`sphere_field`, where the sun dominates every point, it adds noise. That is why
`lightTreeBounces` defaults to 0. Use it for offline references of many-light scenes, where samples
are expensive anyway. The images are unbiased either way: means agree within 0.2% at 16 spp and
0.01% at 8192 spp. `tests/LightTreeTests.cpp` checks that every light has a leaf, that the
sampled frequencies match `LightTreePmf` and sum to at most 1, and that nearby lights win and lights
behind the horizon get nothing.

### 8.7 ReSTIR DI

ReSTIR DI estimates the direct light at each pixel's first surface with spatiotemporal reservoir
resampling (Bitterli et al. 2020) instead of one NEE sample per path, in the generalized RIS
formulation (Lin et al. 2022; the 2023 SIGGRAPH course "A Gentle Introduction to ReSTIR").
`pathtracer.restirDi` (and `restirGi`, §8.8): 0 off, 1 (default) only when frames do not accumulate
(`accumulate=0`, the real-time path under a denoiser), 2 always (to check it is unbiased by
accumulating). Files: `shaders/ReSTIRDI.hlsli` (the DI math), `shaders/ReSTIRCommon.hlsli` (shared
with GI: receivers, neighbours, push constants), `ReSTIRDIInitial/Temporal/Spatial.hlsl`,
`shaders/ReSTIRShared.h`, `src/ReSTIR.h/.cpp` (buffers and passes of both).

- **Split with the path tracer**: with `kFlagReSTIR`, `PathTracer.Trace` leaves out the direct light
  from emissive triangles at the first surface (no NEE there, and no emission found by its BSDF
  sample), and writes the first sample's hit (`PrimaryHit`: instance, primitive, barycentrics,
  distance, the shading normal and the material after its textures in float16). The passes rebuild
  the surface's geometry from it with the integrator's own `LoadSurface` and take its normal and
  material as stored, so they shade exactly what the path tracer did without reading textures again
  (a neural texture set decoded on sample, §8.16, would cost a decode per pass and neighbour), and add
  the result (times the samples per pixel) to the accumulation before `PathTracer.Resolve`. The
  environment is still reached only by BSDF sampling, and deeper vertices keep NEE.
- **Samples**: a point on an emissive triangle (light index, barycentrics), the same point in every
  pixel's domain, so shifts are the identity with Jacobian 1. The target function p̂ is the luminance
  of the unshadowed contribution f · Le · cos / d². A reservoir (16 bytes) holds the sample, its
  unbiased contribution weight W and its confidence M.
- **Passes** (one thread per pixel, 8x8 groups):

  | Pass | Inputs | What it does | Shadow rays |
  |---|---|---|---|
  | `ReSTIRDI.Initial` | primary hits | RIS over `restirLights` points on lights chosen by power and `restirBsdf` BSDF rays that hit an emitter, balance heuristic between the two strategies | 1 BSDF + 1 |
  | `ReSTIRDI.Temporal` | + last frame's hits and final reservoirs | reprojects the point with the previous camera; if the surface there is similar, generalized balance heuristic between the two domains, last frame's confidence capped at `restirHistory` | 1 + 1 |
  | `ReSTIRDI.Spatial` | temporal reservoirs | `restirSpatial` neighbours in a disc of `restirRadius` pixels (`restirDisocclusion` where the pixel has no history), defensive pairwise MIS (course eq. 7.8) with this pixel's visibility in the target; shades and stores the result as next frame's history | k + k |

- **Visibility reuse, without bias**: each pass traces the shadow ray of the sample it chose when the
  sample is new to the pixel, and zeroes W when it is occluded, so a reservoir only ever holds light
  its own surface sees. That domain restriction must appear in the MIS weights: another pixel's (or
  frame's) reservoir may get weight for a sample only if that surface sees it. With
  `restirUnbiased=1` (default) the weights trace exactly that visibility, one ray per reused
  reservoir. `restirUnbiased=0` assumes the other surface sees the sample: the classic biased
  visibility reuse, fewer rays, penumbrae and contact shadows darker (-2% on `many_lights`, -2.8% on
  `cornell_box` direct light).
- **Shading with every candidate**: the spatial pass traces this pixel's visibility for each reused
  sample, so its target is p̂ times visibility and the chosen sample needs no shading ray. The
  direct light it adds is the sum of all candidates' contributions weighted by their resampling
  weights (the expectation of f(Y) W over the choice, as in ReSTIR PT Enhanced, Lin et al. 2026)
  instead of the chosen sample's alone: with the scalar target choosing by luminance, one light per
  pixel otherwise shows as colour noise. Per-channel relMSE on `many_lights` drops 4x (5.4 -> 1.3),
  for about 0.15 ms.
- **Neighbour choice**: from geometry only (normal within 60 degrees, distance to the eye within
  10%), never from the samples, so heuristics change the noise but not the expected value. Pixels
  without a hit keep empty reservoirs.
- **History**: primary hits and final reservoirs are double-buffered by frame parity (`ReSTIR` owns
  them; they exist only once ReSTIR has been used). A scene or setting change, or a frame without
  ReSTIR, starts without history; a camera move does not (the reprojection handles it).
- **Limits**: lights must not move between frames (samples are reused as light-list points); light
  indices have 24 bits. Reuse is per-pixel only: large disocclusions recover over a few
  frames (the disocclusion boost helps); stochastic pairwise MIS over screen-space cells (Hedstrom et
  al. 2026) is the known fix.

Validation (`aircli render --set pathtracer.debugView=8` isolates the direct light):

- Unbiased: accumulating ReSTIR frames (`accumulate=1`) converges to the NEE reference. Mean
  difference after 8192 frames: `many_lights` -0.013%, full image -0.010%; `cornell_box` -0.012%
  and `veach_mis` -0.07% after 4096 (NEE alone: -0.010% and +0.09%). With `spp=4` the scaling holds.
- Noise of one frame (`accumulate=0, spp=1`, after 30 frames of history), relMSE of the direct
  light at 1920x1080: `many_lights` NEE 21.8, ReSTIR 0.20 (k=3: 0.18, 16 light candidates: 0.13,
  no BSDF candidate: 1.95, which the glossy floor needs). At 640x360: `veach_mis` 1.88 -> 0.0098,
  `cornell_box` 0.0089 -> 0.0019.
- Cost at 1920x1080 (RTX 5070, defaults, `accumulate=0, spp=1`):

  | Scene | Trace | Initial | Temporal | Spatial |
  |---|---|---|---|---|
  | `cornell_box` | 1.45 ms | 0.32 ms | 0.35 ms | 0.73 ms |
  | `door` | 2.73 ms | 0.46 ms | 0.39 ms | 0.76 ms |
  | `many_lights` | 2.74 ms | 1.10 ms | 0.65 ms | 1.62 ms |
  | `sphere_field` | 1.12 ms | 0.60 ms | 0.61 ms | 1.22 ms |

  The unbiased MIS rays cost about 0.1 ms; each neighbour about 0.4 ms on `many_lights`, each light
  candidate about 0.1 ms. Rebuilding a surface from its `PrimaryHit` costs about 0.1 ms per use
  (six per pixel), cheaper in total than storing a full-precision G-buffer twice. Storing the material
  (40 bytes a hit, from 24) also made the receivers of normal-mapped surfaces agree with the trace:
  rebuilt without the camera ray, a bent normal's side was judged differently, and ReSTIR GI on
  `textured` was 0.31% dark against NEE, now 0.10%.
- When frames accumulate, ReSTIR's correlated samples converge about as fast per unit of time as
  plain paths (§8.8 has the measurement), and with several samples per pixel per frame only the first
  one's light is resampled; hence the default of running it in real time only.
- Tests: `PathTracer_ReSTIRResamplesAcrossFrames` and `PathTracer_ReSTIROnlyRunsForLitViews`
  (Null backend: passes, buffer ping-pong, modes, which views run it).

### 8.8 ReSTIR GI

`pathtracer.restirGi` (same modes as `restirDi`) resamples the indirect light at each pixel's first
surface, all paths of two or more bounces, across pixels and frames (Ouyang et al. 2021), in the
path-space form of ReSTIR PT (Lin et al. 2022) with the reconnection shift always at the second
vertex x2. Files: `shaders/ReSTIRGI.hlsli`, `ReSTIRGITemporal/Spatial.hlsl`, the `GiPath` part of
`TracePath` (Integrator.hlsli), `PathTracerGi.hlsl`.

- **Initial samples from the path tracer**: with `kFlagReSTIRGI` the trace kernel (its GI variant)
  leaves every path of two or more bounces out of the radiance. Its first sample records them at x2
  instead, as two candidate paths: NEE at x2 (the light point), and x2's BSDF continuation, whose
  radiance arriving at x2 splits into the emission at x3 (weighted against NEE at x2 by MIS) and the
  rest, gathered with throughputs from x3 on. RIS picks one by the luminance of its contribution and
  writes it as the pixel's initial reservoir (`GiReservoir`, 64 bytes): x2's hit, the outgoing
  direction at x2, the two radiances, the light pdf the MIS needs, W, M and which of the two it is.
  The two candidates' sum is exactly what the path tracer would have added.
- **The shift, without a Lambertian assumption**: a sample moves to another pixel by connecting
  that pixel's x1 to the same x2 and keeping the rest: in area measure at x2 times solid angle at x2
  the shift is the identity. Its value there is evaluated in full: x1's BSDF towards x2, the
  geometry term, x2's BSDF between the new incoming and the kept outgoing direction (x2's shading
  frame rebuilt as `TracePath` builds it), and the power heuristic between NEE and BSDF sampling at
  x2, whose BSDF pdf depends on the incoming direction. Only the radiance arriving at x2 is reused as
  it is, and that does not depend on x1, so the reuse stays unbiased on glossy x2 too.
- **Where reconnection is allowed**: through a near-mirror vertex a moved path's value swings
  between nothing and a spike, which reuse turns into fireflies that persist. The shift exists only
  when x1 on both sides and x2 have a roughness of at least 0.25 (`kGiMinRoughness`; ReSTIR PT
  Enhanced, Lin et al. 2026, keeps such a threshold before the reconnection too): smoother pixels
  keep their own sample, neighbours are chosen among rough pixels, and a sample with a smooth x2
  stays where it was found. The criteria depend on surfaces only, so the reuse stays unbiased.
- **Passes**, sharing DI's receivers, neighbour choice, confidence cap and visibility handling (a
  reservoir only holds samples whose x2 its surface sees; MIS weights trace the other surfaces'
  visibility with `restirUnbiased`):

  | Pass | Inputs | What it does | Shadow rays |
  |---|---|---|---|
  | `ReSTIRGI.Temporal` | initial reservoirs, last frame's hits and history | generalized balance heuristic with the reprojected pixel | 1 + 1 |
  | `ReSTIRGI.Spatial` | temporal reservoirs | defensive pairwise MIS with neighbours; shades every candidate by its resampling weight; history | k + k |

Validation (`--set pathtracer.debugView=9` isolates the indirect light):

- Unbiased: accumulated ReSTIR GI (`restirGi=2`, 4096 frames, 640x360) matches 8192 spp NEE
  references within 0.06% on `cornell_box`, `many_lights`, `door` and `material_grid` (NEE alone:
  within 0.03%); DI and GI together on the full image within 0.04%. In the white furnace the
  accumulated mean is the same 0.1799 as without it.
- Noise of one frame (`accumulate=0, spp=1`, after 30 frames, 640x360), relMSE of the indirect
  light: `cornell_box` 0.284 -> 0.0089 (temporal reuse alone 0.0138, spatial alone 0.0606), `door`
  52.9 -> 0.68, `many_lights` 78.9 -> 5.0, `material_grid` 0.044 -> 0.026. Of the whole image with
  DI and GI: `door` 50.7 -> 0.57, `cornell_box` 0.223 -> 0.0083, `many_lights` 40.4 -> 2.94.
- The white furnace is the exception: white, energy-conserving BSDFs under a uniform sky make plain
  BSDF sampling almost noiseless, so any reuse adds noise there (std of one frame 0.016 -> 0.021).
  It stays a test of the mean only.
- Cost at 1920x1080 (RTX 5070, `accumulate=0, spp=1`, frame time NEE -> DI -> GI -> both):
  `cornell_box` 1.44 -> 2.89 -> 5.52 -> 6.87 ms, `door` 2.75 -> 4.37 -> 8.47 -> 9.98 ms,
  `many_lights` 2.88 -> 6.17 -> 8.88 -> 11.65 ms. Of GI's share, the trace kernel's recording costs
  1.1-1.3 ms (register pressure), the temporal pass 1.0-1.5 ms and the spatial pass 2.0-3.3 ms
  (per neighbour two surfaces rebuilt, three BSDF pairs, two shadow rays). Next steps there: ReSTIR
  PT Enhanced's paired spatial reuse (one shift serves both pixels) and a cheaper target function.
- Accumulating instead (`accumulate=1`): at equal time, 4096 frames of DI+GI against 16384 frames of
  NEE reach about the same error (`many_lights` 0.0067 vs 0.0066, `door` 0.0077 vs 0.0093,
  `cornell_box` rmse 0.0027 vs 0.0019), which is why both default to real time only.
- Tests: `PathTracer_ReSTIRGIResamplesTheIndirectLight` (Null backend: passes, history ping-pong,
  with and without DI, the furnace).


### 8.9 Physical sky

`SceneDesc::sky` (`renderer/Sky.h`) replaces the uniform `environment` with an Earth-like
atmosphere lit by the sun: sun elevation, azimuth and angular radius, the scene's altitude, the
aerosol (haze) optical depth at 550 nm (0.005 pristine, 0.05 the default clear day, 0.3 and up
hazy) and the albedo of the planet seen below the horizon. `sphere_field`, `terrain` and `sunset`
use it; `aircli render|reference --sun ELEVATION,AZIMUTH [--aerosols DEPTH]` gives it to any scene,
and the editor's Scene panel edits it.

- **Model** (`src/renderer/src/Sky.cpp`): the medium and data of Bruneton's 2017 reference
  implementation (Rayleigh scattering 1.24062e-6 / lambda^4 per metre over an 8 km scale height;
  aerosols over 1.2 km with single-scattering albedo 0.9 and a Cornette-Shanks phase, g = 0.8; 300
  Dobson units of ozone peaking at 25 km; the measured solar spectrum outside the atmosphere), and
  the light transport of Hillaire 2020: a transmittance table (256 x 64, Bruneton's
  parameterisation, 500 steps each), a multiple-scattering table (32 x 32; 64 directions, all
  orders as the geometric series of the second) and a 96-step ray march per sky direction with the
  planet's sunlit ground below the horizon.
- **Spectral, then RGB**: every table is computed for 16 bins of 25 nm (380-780 nm) and the sky
  radiance converted to linear sRGB in nits through the CIE 1931 colour matching functions (the
  analytic fit of Wyman, Sloan and Shirley 2013) and 683 lm/W. Colours come from spectra: the
  reddening of a low sun, the blue zenith, the twilight sky.
- **One table, two renderers**: the result is a 256 x 256 table of sky radiance over all
  directions as seen from the scene (`u = sqrt(azimuth from the sun / pi)`, elevation through a
  signed square root: texels gather near the sun's azimuth and the horizon) plus the sun's
  radiance after the atmosphere (`gpu::SkyData`, `shader/SkyShared.h`). The CPU builds it when the
  sky changes (0.75 s in debug builds, on all cores) and `Scene` uploads it into its ring; the GPU
  kernels and the CPU reference read it through the same `SkyLookup.hlsli`, so their skies are
  identical.
- **The sun is a disc light**: every path vertex samples a direction uniformly within the disc
  (next event estimation, one shadow ray), weighted by the power heuristic against BSDF samples
  that reach the disc, exactly like the emissive triangles. The sky itself is only reached by BSDF
  sampling. ReSTIR DI keeps resampling the triangles only; in ReSTIR GI the sun's NEE at x2 is one
  more NEE candidate (streamed with the triangle's), and a continuation that escapes into the disc
  keeps the sun as its MIS-weighted emission. The furnace replaces the sky with a uniform white one
  and switches the sun off.
- **Aerial perspective**: the same ray march, stopped at 32 distances (quadratically spaced up to
  100 km, the first at 104 m), fills two 32 x 64 x 32 RGBA16F tables over the sky table's
  directions: the light the air scatters towards the viewer over that distance and its
  transmittance (spectral, turned into RGB as the factor by which it changes sunlight's colour).
  What a camera ray sees on its surface becomes `inscatter + transmittance * radiance`
  (`AddAerialPerspective`; ReSTIR's light at that surface gets the same transmittance), and so does
  everything a ray sees on the clouds (§8.10); the sky table includes the air already. Segments
  between surfaces of the scene are taken as clear. The GPU samples the tables with hardware
  filtering (linear clamp), the CPU reference emulates it (half-float texels, 8-bit weights).

Validation:

- `tests/SkyTests.cpp`: the zenith optical depth at 550 nm (0.19: Rayleigh 0.11, ozone 0.03,
  aerosols 0.05) and its transmittance; the solar spectrum's illuminance outside the atmosphere
  (about 128 klux); with the sun 60 degrees high a sun of 1.3-1.75e9 nits, 75-100 klux direct and
  8-30 klux of sky light on a horizontal surface, a blue zenith of 1.5-12 kcd/m^2; a red sun and
  warm horizon at 2 degrees; no sun and a dim sky in civil twilight; the lookup returns the table and
  is mirror-symmetric; sun samples stay in the disc.
- `Integrator_SunNeeAndBsdfSamplingAgree`: with a 12-degree sun, NEE + MIS and BSDF sampling alone
  agree (0.5% apart at 256 spp).
- GPU against the CPU reference on `sunset` (96x54): the means differ by 0.06%, by 0.13% with
  ReSTIR DI and GI accumulated.
- `Sky_AerialPerspectiveMatchesTheSkyAndTheAir`: no air at the viewer; looking up through the whole
  atmosphere the air's light is the sky's within 1% (2% near the sun and 3% at the zenith, where the
  table's texels are wide); 10 km along the ground let through exp(-optical depth) in green (within
  0.03), red more than blue; the air's light grows with distance.
- GPU against the CPU reference on `terrain` (160x90): the means differ by 0.023%, by 0.024% with
  ReSTIR DI and GI accumulated.
- Not modelled yet: the air's light in cloud shadows (crepuscular rays: the tables assume clear
  sunlight); a viewer far above the scene's altitude (the tables are computed for that altitude);
  the sun's limb darkening.

### 8.10 Volumetric clouds

`SceneDesc::clouds` (`renderer/Clouds.h`) adds a layer of cumulus above the scene: the heights of
its base and tops, coverage, extinction at density 1, the droplets' effective diameter and
single-scattering albedo, a shape scale, a seed, the wind (speed and heading) and the high cirrus
above (coverage, altitude, optical depth). `cumulus` uses it;
`aircli render|reference --clouds COVERAGE[,BOTTOM,TOP[,WIND[,HEADING]]]` adds it to any scene (0
removes it; it switches on the physical sky, whose sun lights the clouds; the wind in m/s, heading in
degrees from +x towards +z; `--cirrus COVERAGE[,ALTITUDE[,DEPTH]]` sets the cirrus over them), and
the editor's Scene panel edits it.

- **Density** (`src/renderer/src/Clouds.cpp`), procedural, baked on the CPU into two 8-bit 3D
  textures when the clouds change (2.1 s for `cumulus` in debug builds, all cores):
  - the base, 512 x 512 texels over a 25.6 km tile that repeats horizontally (too far away to show
    from the ground) and 25 m layers in height. A 2D weather field is thresholded at its
    1 - coverage quantile, so the requested fraction of the tile lies under cloud outlines; how far
    it exceeds the threshold sets a cloud's height and density. Gradient noise from 12.8 to 1.6 km
    gathers the clouds into fields with clear gaps between; cellular noise at 3.2, 1.6 and 0.8 km,
    over coordinates warped by about 700 m, makes clouds of many sizes with uneven outlines. Bases are
    flat at the condensation level; above, a cloud narrows into a dome, billowy cellular noise in
    cells of 640 to 160 m carves its flanks and top (barely its base) and liquid water grows with
    height above the base. As in Horizon's Nubis (Schneider 2015, 2017) a second field, of cloud
    types (gradient noise of 12.8 and 3.2 km), varies over the sky: stratocumulus (tops at
    0.06 + 0.18 strength of the layer, hardly narrowing: flat-topped sheets, the weather raised there
    so they spread), cumulus (0.12 + 0.88 strength^1.6: mostly shallow, the strongest tall; tall ones
    everywhere made a wall of clouds along the horizon) and congestus (0.3 + 0.7 strength^0.7, in
    fewer, larger cells, their tops spreading into anvils). The billows' and the weather's
    coordinates are swirled by curl noise (up to about 200 m, more higher up) and leaned by the
    wind's shear (500 m over the layer's height downwind, 150 m across).
  - the detail, 128^3 texels of cellular noise repeating every 1.024 km (25 to the tile), which
    erodes the base's fringe a little at render time (strength 0.4); the eroded density is then
    sharpened (times 5, clamped to 1): crisp outlines, even interiors. The earlier generator (a
    12.8 km tile, one size of cell, the detail at 0.75 and times 10) made a sky of same-sized
    cauliflowers.
  - majorants, one per 8 x 4 x 8 base texels plus the texels around them that trilinear filtering
    reaches. They bound the density everywhere because the erosion only lowers it and the
    sharpening is monotonic.
- **Cirrus** (`CirrusOpticalDepth` in `CloudLookup.hlsli`): high clouds of ice crystals, a sheet at
  9 km by default (above the layer's tops), too thin to march through: a ray crosses it once and
  scatters there with probability 1 - exp(-tau / mu) (its vertical optical depth over the cosine at
  which the ray crosses it, at least 0.05), with the clouds' phase function and albedo (ice crystals
  scatter forward like droplets; their halos are not modelled); the transmittance through it is
  exp(-tau / mu), exact. Its optical depth (up to 1 by default) is procedural, so nothing is baked
  and the CPU reference evaluates the same code. Cirrus is wisps of ice drawn out by the wind's shear:
  filaments a few hundred metres across and kilometres long that curl into hooks and taper, in patches
  with clear sky between and a faint veil among them. So: ridges of gradient noise (`1 - |n|`, thin
  lines; plain noise made evenly spaced bands, a sky combed into stripes) stretched 6:1 along the
  wind, three octaves each turned a little against the others, over coordinates warped by swirls of
  18 and 6 km, tapering along their length, under a soft mask of patches of 6 to 15 km over the
  fraction of the sky asked for (0.3).
- **The planet's curvature**: the layer's heights are altitudes above a sphere of 6371 km under the
  scene's origin, `a(p) = p.y + (p.x^2 + p.z^2) / 2R` (exact to a metre within the 100 km a ray goes):
  100 km away the layer is 785 m lower, so it sinks towards the horizon instead of a flat sheet whose
  edge-on view fills it. In (x, altitude, z) the grid stays flat and a ray is a parabola: the DDA
  crosses its altitude boundaries at the parabola's roots (`CloudCrossing`), its x and z boundaries
  as before. The shadow map (§8.19) starts at the base's altitude and its lookup follows the sun's
  ray to it.
- **One lookup, two renderers**: `shader/CloudLookup.hlsli` (the density, a 3D DDA over the
  majorant cells that skips empty ones, delta and ratio tracking, the phase function) is shared
  like the integrator. The GPU samples the textures with hardware trilinear filtering (the root
  signature's linear-wrap static sampler); the CPU reference emulates that filter including its
  8-bit weights (`renderer::SampleTexture3D`). Hardware filtering halved the cost of a cloud frame
  compared with 8 buffer loads per lookup.
- **Phase function**: Mie scattering by water droplets as fitted by Jendersie and d'Eon 2023:
  Henyey-Greenstein for the diffraction peak (g = 0.995 at 20 micrometres) blended with Draine's
  function for the rest (g = 0.59, alpha = 27), all four parameters from the droplet diameter
  (5-50 micrometres); both lobes are sampled exactly (Draine's CDF inverts in closed form).
- **Delta-M scaling** by default (`Clouds::exactPhase` false, Wiscombe 1977): the diffraction peak,
  about half of all scattering within a fraction of a degree, counts as unscattered light. The
  extinction becomes sigma (1 - f albedo) and the albedo albedo (1 - f) / (1 - f albedo) for the
  peak's weight f, and Draine's function alone is the phase function. Transport away from the
  forward direction is unchanged (the means of the two agree to 0.1%), but the rare paths that line
  up with the sun inside the peak, each carrying a large share of its 1.6e9 nits, are gone: with the
  exact phase function sunlit clouds sparkle with fireflies for thousands of samples. It only
  changes the image within a degree or so of the sun (the sun seen through a thin cloud keeps its
  sharp disc instead of a glowing peak around it).
- **Transport** (`EscapeRadiance` in `Integrator.hlsli`): the layer lies above all geometry
  (`Validate` checks it), so a path meets it only on its way to the sky. Every ray that leaves the
  geometry, and a camera ray from above the cloud base before it reaches its surface, walks through
  the clouds by delta tracking. At each scattering event the sun is sampled by next event
  estimation, its transmittance by ratio tracking (Russian roulette below 0.1), and weighted by the
  power heuristic against phase sampling, which continues the walk. A walk that leaves the layer
  gets the sky table's radiance; below the horizon that is the planet's sunlit ground: light that
  the scene's own geometry reflects back up into the clouds is not traced. The sun's NEE at a
  surface is dimmed by the clouds' transmittance, which casts their shadows. `pathtracer.cloudEvents`
  (256) caps the scattering events of a walk; light beyond it is lost (raising it to 4096 changes
  `cumulus` by less than its noise).
- **ReSTIR**: DI is unchanged (it resamples triangle lights only). In GI the sun's NEE at x2
  carries the clouds' transmittance into its candidate, and a continuation through the clouds
  splits into the sun seen straight through (the MIS-weighted emission) and the rest.

Validation:

- `tests/CloudTests.cpp`: the phase fit at 20 micrometres; the phase function integrates to 1 for
  5, 20 and 50 micrometres with a mean cosine of 0.75-0.92; sampled cosines follow its CDF at 8 cuts
  (400k samples, within 0.003); the volume covers about the requested sky and, along 2000 random
  rays, the density never exceeds the majorant of the cell the DDA is in; the density repeats with
  the tile; the wind's offsets carry the density and the majorant cells with it (the detail drifting
  on over the base stays bounded); ratio and delta tracking both match a 0.5 m ray march of
  exp(-optical depth) for rays of optical depth 0.1-3 (times the cirrus' transmittance); the cirrus
  covers the fraction of the sky asked for (0.15, 0.3, 0.6, within 20%; the noise is close to normal and its quantile sets the threshold) and a ray through a gap of
  the layer collides with it at 1 - exp(-depth); disabled clouds are transparent.
- `Integrator_CloudsVanishInAWhiteFurnace`: with albedo 1 in a uniform environment every path
  through the clouds returns the environment exactly, so every pixel is 0.18.
- `Integrator_CloudSunNeeAndPhaseSamplingAgree`: with a 12-degree sun, NEE + MIS and phase sampling
  alone agree within 0.5% at 512 spp, with and without delta-M scaling.
- `Scene_CloudsStayAboveTheGeometry`, `Scene_CloudResourcesFollowEdits`.
- GPU against the CPU reference on `cumulus` (160x90, `--sun 38,-20`, 8192 against 2048 spp): the
  means differ by 0.014%; sky, horizon and ground separately by at most 0.2%.
  With the 25.6 km generator and the curved layer (240x135, 1024 frames against 64 spp on the CPU):
  0.00%, 15 x 15 blocks within -1.2% .. +0.9% (the CPU's noise).
- Cost (RTX 5070, 960x540, 4 spp per frame, front-lit `cumulus`): 78 ms per frame. Multiple
  scattering in thick media is the most expensive transport here, and noisy: a cloud pixel's
  samples have a relative standard deviation of about 0.8, so clouds need hundreds of samples (or
  a denoiser).
  In real time the clouds are ray marched into maps instead (§8.19).
- The air between a ray's origin and its first collision with the clouds dims and adds to their
  light (§8.9's aerial perspective), so distant clouds fade into the haze.
- Not modelled: the scene's own geometry lighting the clouds, the clouds' shadows in the air between
  them and the viewer; a ray ends 100 km out. The wind (§8.19) only moves them in real time.

### 8.11 Neural radiance cache

`pathtracer.nrc` (0 off, 1 when not accumulating, 2 always; off by default) ends paths in a neural
radiance cache (Mueller et al. 2021, "Real-time Neural Radiance Caching for Path Tracing"): a small
network, trained every frame on a few longer paths, predicts the light a path vertex reflects. A
path ends at the first vertex where its footprint has spread past `nrcSpread` (0.01) times the
primary vertex's (as if the camera emitted uniformly; usually the first diffuse bounce, while
mirror-like chains go on), or at its last vertex, and adds the cache's prediction there instead of
its continuation: far less noise from long paths and fewer rays, for some bias. While the cache
runs it replaces ReSTIR GI (§8.8); ReSTIR DI keeps the first surface's direct light.

- **Network** (`renderer/shader/MlpShared.h`, `renderer/NeuralNetwork.h`): 64 inputs, two hidden
  layers of 64 ReLU neurons, 3 outputs (16 rows, the rest zero), no biases (an input held at 1).
  The inputs are a multiresolution hash grid of the position (Mueller et al. 2022: 16 levels of 2
  features from 16 to 512 cells across the scene's box, 2^`nrcTableLog2` (16) entries per level),
  spherical harmonics up to degree 3 of the direction the light leaves in, the shading normal, the
  roughness and both albedos. The outputs are reflected radiance times the camera's exposure (O(1)
  values, safe in float16; when the exposure changes the output layer is rescaled) divided by the
  albedo sum (the network learns the light, not the materials).
- **Training paths** (`shaders/NrcShared.h`): one pixel per tile of pixels (`nrcTrainingPaths`,
  32768 per frame; tiles of 2 x 2 at 480 x 270, 8 x 8 at 1080p) traces a training path, at a pixel
  that moves every frame. Its own render query is deferred to `NRC.Resolve`, so that every thread of
  the trace makes exactly one cache lookup; the path goes on for a suffix that ends in the cache in
  turn (self-training: the cache learns infinite bounces from its own predictions), except 1 in
  `nrcUnbiased` (16) paths, which go on until Russian roulette or their last vertex. Every vertex of
  a training path (at most 6; not the first with ReSTIR DI, whose direct light is not in the path)
  becomes a record: its inputs and the light the rest of the path found it reflect per unit of
  throughput up to it, the tail's prediction included.
- **Optimizer**: `nrcSteps` (4) steps per frame, each on a scattered quarter of the training paths
  (a multiplicative permutation of the path indices); the relative L2 loss of the luminance
  (prediction error over the prediction's squared luminance plus 0.01, which trains dark and bright
  regions alike); Adam at `nrcLearningRate` (0.01, beta 0.9 / 0.99), L2 regularization 1e-6 of the
  MLP's weights, sparse updates of the hash grid (entries no sample touched keep their moments).
  Loss gradients are clamped only far out (65536 per output): a cache learns the mean of its noisy
  targets, which their rare bright samples carry, and clamping them at 64 made `door` 28% dark.
- **Kernels**: on the matrix units (`nrcLinearAlgebra`, §3.4) the weights are float16 MulOptimal
  matrices: forward passes multiply, backpropagation multiplies by the same matrices loaded as
  MulOptimalTranspose, and every weight gradient is an outer product accumulated atomically into
  float32 OuterProductOptimal matrices, converted to row major for the optimizer, which writes new
  float16 weights that are converted back. Backpropagated gradients are scaled by 1/64 for float16
  range (the optimizer divides it out). Without linear algebra the float32 kernels run
  `renderer/shader/Mlp.hlsli`, the code the CPU twin (`renderer/Mlp.h`) runs, summing weight
  gradients over each wave before 64-bit fixed point atomics. Hash grid gradients are 64-bit fixed
  point atomics in both (exact and order independent). The float32 master parameters, Adam's
  moments and the gradients live in `renderer::NeuralNetwork`, whose passes any feature can use.
- **Passes**: `PathTracer.Trace` (the cache's kernel variant, a compute shader: every thread
  predicts at the end of its path, those without a query on a dummy one), `NRC.Resolve` (the
  training paths' render queries into the accumulation), then per step `NRC.Train[i]`,
  `NRC.Mlp.ConvertGradients`, `NRC.Mlp.Optimize`, `NRC.Mlp.ConvertWeights`. A new scene starts a
  new network. The cache's pipelines are created the first time it runs
  (`renderer::LazyComputePipeline`): the driver takes 0.7 s to compile them.
- `debugView=10` shows what the cache predicts the first surface reflects.

Validation:

- `tests/MlpTests.cpp`: backpropagation agrees with finite differences for 400 weights, all 64
  inputs and the hash grid's entries at all 16 levels; the spherical harmonics are orthonormal
  (4e-5); Adam fits a function of position and direction from 0.87 to 0.033 mean relative error
  in 400 steps; the GPU network's passes on the Null device. `PathTracer_RadianceCacheTrainsEveryFrame`
  checks the cache's passes, the float32 kernels and the views.
- `aircli mlp-check --backend d3d12` runs the GPU's network kernels on 1024 random inputs and loss
  gradients against the CPU: the float32 kernels agree to 2e-7 (relative RMS); the linear algebra
  ones to 7e-4 for the outputs, 1.4% for the weight gradients and 2.7% for the input gradients
  (float16 sums, and the ReLUs float16 rounding switches near zero); one transposed load made
  untransposed shows as 143%.
- White furnace (`furnace=1`): the `furnace` scene's spheres vanish (accumulated, -0.01%). Inside
  the Cornell box, where albedo 1 keeps nearly all light and a self-trained cache feeds on its own
  predictions, it reads +1.4% (1 in 1 paths unbiased) to +2.6% (1 in 16) over the true 0.18, which
  8 bounces of plain path tracing (then the default) miss by 14%.
- Against a same-jitter reference (2048 spp), 480 x 270, 4 spp per frame after 62 frames, relative
  MSE (bias of the mean):

  | Scene | Plain paths | ReSTIR DI + GI | ReSTIR DI + cache |
  |---|---|---|---|
  | cornell_box | 0.052 | 0.0092 | 0.0127 (-1.4%) |
  | door | 11.4 | 0.84 | 0.98 (+2.7%) |
  | terrain | 0.0076 | 0.0082 | 0.0075 (-0.02%) |
  | sunset | 0.135 | 0.190 | 0.057 (-0.16%) |
  | sphere_field | 0.231 | 0.432 | 0.062 (+0.10%) |

  The cache's mean moves by about 0.5% from frame to frame as it trains. In closed rooms whose
  light arrives through few paths (the box's ceiling, the door) the first bounce's sampling
  dominates the noise, and ReSTIR GI's reuse across pixels wins there.
- Cost (RTX 5070, cornell_box, 960 x 540, 1 spp, ReSTIR DI off): trace 0.99 ms (0.49 ms without the
  cache), `NRC.Resolve` 0.04 ms, 4 steps of training 4 x 0.28 ms and optimizer 4 x 0.05 ms: about
  2.4 ms. The float32 kernels take 15 ms (trace 4.1 ms, training 9.7 ms): the matrix units are 4.2
  times faster at inference and 8.4 at training.
- Not done yet: ReSTIR GI's candidates ending in the cache (the reuse of the first bounce together
  with the cache's tail); averaging the weights over steps for inference (steadier predictions);
  scenes that move (the network restarts with the scene).

### 8.12 Volumes: smoke and OpenVDB grids

`SceneDesc::volumes` (`renderer/Volume.h`, up to 16) put boxes of voxels among the surfaces that
absorb and scatter light: grids of OpenVDB files, or procedural plumes of smoke. They may overlap
(their densities add up). `smoke` uses one plume; `aircli render|reference --volume
FILE.vdb|plume|none[,EXTINCTION[,ALBEDO[,ANISOTROPY]]] [--volume-grid NAME] [--volume-at
X,Y,Z[,SIZE]]` sets a scene's volume, once per volume (the n-th `--volume` sets or adds the n-th,
`--volume-grid` and `--volume-at` apply to the last one given, `none` removes them all), `aircli vdb
FILE [--read GRID]` lists a file's grids (and reads one as a volume would), and the editor's Scene
panel lists, adds, edits and removes them. In real time (not accumulating) they are ray marched over
the path traced surfaces by default instead of tracked (§8.13).

- **OpenVDB without the library** (`renderer/Vdb.h`; the format is described at the top of
  `src/renderer/src/Vdb.cpp`): files of format version 222 (OpenVDB 2.0) on, float, double and half
  grids (and floats saved as halves) of 5-4-3 trees, their values raw or compressed with zlib or
  Blosc (LZ4, byte shuffled), with or without active-mask compression, and transforms that scale
  uniformly and translate. The decoders are `core/Compression.h` (inflate, LZ4 blocks, Blosc 1
  frames). Level sets, vector grids, rotated or anisotropic transforms and frustum maps are refused
  with the reason. A grid is read over the bounds of its active voxels (leaves and active tiles) and
  averaged while it is read over blocks of factor^3 voxels so that its longest side fits
  `maxResolution` (256, at most 2048): the full-resolution Disney cloud (1.5e9 active voxels) never
  becomes a 6 GB dense grid. Voxel i's centre is at i x scale + translation (OpenVDB's convention).
- **Placement**: `size` > 0 scales the box to that many metres along its longest side, its bottom
  centred on `position`; 0 keeps the file's own size and place, moved by `position`. Grids are read
  as stored: a file from a Z-up tool lies on its side. `renderer::Scene` builds each volume on its
  own: one that did not change keeps its resources, and it keeps the voxels it read (up to 512 MiB in
  all, shared by volumes of the same source), so edits of the optics or the placement do not read the
  file again; another file, grid, resolution or seed does.
- **The plume** (`MakeSmokePlume`): 64 x 128 x 64 voxels of 2.5 cm, a swaying column of value noise
  that widens with height and fades out in its top 30%.
- **On the GPU** (`renderer::BuildVolume`, `renderer::Scene`): per volume, the densities as an
  R16_FLOAT 3D texture with a zero voxel around them (hardware trilinear filtering through the static
  linear-clamp sampler reads zero outside), and majorants for cells of 8^3 texels: the largest
  texel trilinear filtering reaches from inside a cell, as rounded to half floats, times the
  extinction. `GpuScene::volumeTable` holds a `gpu::VolumeData` per volume that loaded, with the
  bindless indices of its resources. `shader/VolumeLookup.hlsli`, shared with the CPU reference (which emulates the filter,
  `renderer::SampleTexture3DClampedR16`), marches a 3D DDA over the cells that skips empty ones:
  delta tracking finds collisions, ratio tracking (Russian roulette below 0.1) transmittance; the
  Henyey-Greenstein phase function is sampled exactly.
- **Transport** (`Integrator.hlsli`): every ray, camera or bounce, is delta tracked through the
  volumes up to its surface, each volume on its own: the nearest collision wins (the first collision
  in a sum of media is distributed as the nearest of each medium's first collision, and the medium
  it is in is the one that scattered), and each volume is tracked only up to the nearest collision
  found so far. A collision is a path vertex: the throughput takes that volume's albedo, NEE samples
  a triangle light (the light tree without a receiver cosine, or by power) and the sun, their
  transmittance through the volumes (the product of each one's ratio tracking estimate; and the
  clouds, for the sun), weighed by the power heuristic against phase sampling, which continues the
  path. NEE from surfaces is dimmed by the volumes' transmittance too, which casts their shadows. Scattering events are not
  bounces: a path takes up to `pathtracer.volumeEvents` (256) of them on top of its `maxBounces`
  surfaces, and Russian roulette ends a walk only once its throughput has fallen below 0.25 (as the
  clouds' walks do), so walks through a bright medium are not cut short by chance. ReSTIR DI and GI
  and the radiance cache stay off while the scene has a volume: they do not model media.
- At a collision on the first segment the guides and debug views show the medium: its albedo, a
  normal facing the camera, the collision's distance.

Validation:

- `tests/VolumeTests.cpp`: placement and padding; along 3000 random rays the density never exceeds
  the majorant of the DDA's segment (its ends included); ratio and delta tracking match a 1 mm ray
  march of exp(-optical depth) for rays of optical depth 0.1-3, also up to a surface halfway;
  averaging down keeps the mass; the phase function integrates to 1 and its samples follow its CDF
  for g from -0.5 to 0.9; a disabled volume is transparent, a missing file gives none and says why.
- `tests/VdbTests.cpp`: zlib streams made by zlib (stored, fixed and dynamic Huffman blocks)
  inflate; corrupt ones fail with a message; LZ4 blocks and Blosc frames (stored, split into byte
  streams, 256-byte blocks with leftovers, 2- and 4-byte elements); a small writer's files in every
  encoding (raw, active mask, zlib, Blosc in one or 256-byte blocks, floats or halves, tiles and
  inactive values) read to the same voxels in the same place; reading at a lower resolution equals
  averaging the full grid down; listing grids, and why one cannot be read.
- Real files (not in the repository): openvdb.org's `smoke`, `smoke2`, `fire`, `explosion` and
  `bunny_cloud` and the Walt Disney Animation Studios cloud (eighth and sixteenth resolution) decode
  to the sums of densities (leaves and tiles) an independent reader finds, to 1e-11.
- `Integrator_VolumeVanishesInAWhiteFurnace`: with albedo 1 in a uniform environment every pixel is
  0.18 within 0.1%; `Integrator_OverlappingVolumesVanishInAWhiteFurnace` the same with two crossing
  plumes of different densities and anisotropies.
- `Integrator_TwoHalvesOfAVolumeLookLikeTheWhole`: two copies of the plume at half its extinction,
  lit from the side by the sun, give the image of the whole within 2% (512 spp).
- `Scene_VolumeResourcesFollowEdits`: a volume that does not change keeps its resources when another
  is added or fails to load; the table holds the volumes that loaded.
- `Integrator_VolumeNeeAndPhaseSamplingAgree`: three views, the plume's shadow on the wall and the
  plume lit from the side in a 12-degree sun, and the plume with its shadow on the ground under a
  3 m lamp: NEE with MIS and sampling alone agree within 4% at 512-1024 spp (at 16 times that, to
  0.15%, 0.26% and 0.04%); taking the transmittance out of any of the four NEE estimates fails it
  (the side-lit plume doubles). Sampling alone carries the noise: under a 1.2 m lamp it was 2.5% at
  1024 spp, too close to the tolerance.
- GPU against the CPU reference: `smoke` (320 x 180, 16384 against 1024 spp) -0.03%, relative MSE
  3e-4; the Disney cloud (eighth resolution, 700 m wide, extinction 0.05 /m, g 0.85) above
  `cumulus` +0.06%.
- Cost (RTX 5070, 1920 x 1080, 1 spp): `smoke` traces in 7.2 ms (1.6 ms without its plume, 7.7 ms
  with openvdb.org's smoke in its place), the Disney cloud over `cumulus` in 2.3 ms (0.75 ms
  without). A dense, bright medium is noisy: the cloud (albedo 1, g 0.85) still sparkles at 256 spp.
- The volume has a kernel of its own (`PathTracerVolume.hlsl`, `AIR_PT_VOLUME`; its pipeline is
  created the first time a scene has a volume): compiled into the plain kernel, the tracking code
  cost every other scene 1-6% (cornell_box +4.9%, sphere_field +5.8%, door +1.3%, sunset +5.4%,
  three alternating rounds each), with no volume to track.
- Not modelled: emission (a fire's temperature grid), extinction varying with wavelength, a sparse
  layout for the path tracer (its grids are dense: 2 bytes a voxel), Z-up files.

### 8.13 Real-time volumes: ray marched, lit through moment shadow maps

Tracking the volumes along every path (§8.12) is exact but costs a frame what the volumes' many
scattering events cost. In real time (`pathtracer.volumeMarch=1`, the default, when not accumulating;
`2` also when accumulating, to measure it) the paths ignore the volumes and every pixel ray marches
them over its path traced surface instead. The design follows Mueller's description of a production
volume renderer (SIGGRAPH 2026): per-pixel marching of sparse 4-bit bricks; a moment shadow map
rendered once per light (b0 in R16, four moments in RGBA16) and looked up everywhere, on async
compute; segments of up to four overlapping volumes per ray; light from the volumes' surroundings
(here probes) scaled by an occlusion grid; blue-noise jitter that accumulation or a temporal filter
resolves; adaptive sampling groups that share lighting over 2 x 2 and 4 x 4 pixels; mip levels of the
volumes; lights cast volume shadows by default, those that matter most to the volumes. Files:
`src/features/pathtracer/src/VolumeMarcher.*`, `VolumeShadowLayout.*`, `shaders/VolumeMarch*`,
`VolumeShadow*`, `VolumeProbes*`; `src/renderer/` `VolumeBricks.cpp`, `BlueNoise.cpp`,
`shader/VolumeBricks*`, `shader/MomentShadow*`.

**Data** (`renderer::Scene`, `GpuScene::volumeMarch`, `volumeLevels`, `lightGroups`):

- *Bricks* (`renderer::BuildVolumeBricks`, `shader/VolumeBricks.hlsli`): each volume at three levels of
  detail (2^l voxels averaged), cut into bricks of 4^3 voxels stored as 5^3 4-bit codes (the apron
  voxel makes trilinear filtering local to a brick) scaled between the brick's half-float minimum and
  maximum, an indirection table (empty bricks hold none) and per brick position its Chebyshev
  distance to the nearest allocated brick (8 bits, counted up to 15), so that a march leaves all the
  empty bricks closer than that at once. The plume keeps 2177 of 9537 bricks; its densities read back
  within 0.0003 on average of the path tracer's texture.
- *Occlusion grid*: per 8^3 voxels the volume's own mean transmittance over 32 directions, marched at
  the coarsest level with diffusion's effective extinction, sigma_t sqrt(3 (1 - albedo)(1 - g
  albedo)): light that scatters on its way in reaches deeper than Beer's law says.
- *Light groups* (`renderer::BuildLightGroups`): one per emissive instance, its centre, area, summed
  normal, mean radiance, power and radius; the light triangles carry their group.

**Passes** each frame, around `PathTracer.Trace`:

| Pass | Queue | What it does |
|---|---|---|
| `VolumeMarch.Shadows` | compute, async | the moment shadow maps, again only when the scene's version or the shadow settings change |
| `PathTracer.Trace` | graphics | `PathTracerVolumeShadows.hlsl` (`AIR_PT_VOLUME_SHADOWS`): paths that light surfaces through the maps; writes the frame's samples and their first hit's distance (`PathTracer.FrameSamples`) |
| `VolumeMarch.Probes` | compute, async, beside the trace | 512 probes per volume, 16 paths each |
| `VolumeMarch.March` | graphics | every pixel's ray through the volumes in front of its hit, over its samples, into the accumulation; under a denoiser it also rewrites the guides where the volumes hide the surfaces |
| `PathTracer.Resolve` | graphics | under a denoiser, also the frame without the volumes (`PathTracer.ColorBeforeVolumes`) |

- **Moment shadow maps** (`MomentShadow.hlsli`, `VolumeShadowLayout.cpp`): up to
  `pathtracer.volumeLights` (8) light groups, ranked by power over (distance to the volumes' box^2 +
  radius^2), and the sun get maps: orthographic over the box for the sun, octahedral around a point
  of the group, its depth through a square root (rays from a point cross the volumes in a short
  stretch of distances). A texel's ray is marched (`VolumeShadowMarch.hlsli`, `volumeShadowLevel` 1,
  `volumeShadowStep` 1 voxel) and its optical depth's power moments stored as half floats; a lookup
  reconstructs the transmittance in front of a depth with moment-based order-independent
  transparency's algorithm (Muenstermann et al. 2018; overestimation 0.25, bias 1e-3: rounded to
  half floats, moments need a stronger pull than 32-bit ones, `MomentTests.cpp`). A third slice holds
  the depth of the texel ray's first opaque hit, ray traced (past the light's own instance; for the
  sun also towards it), filtered as percentage-closer for points inside the volumes; surfaces trace
  their own opaque shadows and read the moments only.
- **Large emitters** (`volumeLightOrigins`, 4): a point sees a column of smoke over a bed of embers
  through its dense foot, which hides the embers' centre from the smoke above while their edges light
  it past the foot. An open emitter whose radius is at least a quarter of its distance to the volumes
  plus that radius gets a map from the centre of its area in each quadrant of its plane (around its
  centre, along the principal axes of the area). A surface reads the one nearest the point its path
  sampled on the emitter; the march reads one per pixel and frame (`MarchLightSample`: the pixel's
  quarter of the blue noise ranks, moved on every frame), so every pixel reads each in turn and
  accumulation or TAA averages them, at the cost of one.
- **Probes** (`VolumeProbes.hlsl`): 8^3 per volume at the cell centres of a lattice over its box
  trace the path tracer's paths (`AIR_PT_VOLUME` and `AIR_PT_PROBE`: they track the volumes), 16 a
  frame along a Fibonacci lattice turned at random per probe and frame. With `kFlagProbe` a path
  leaves out the light that reaches the probe without scattering from the sun and from the march's
  light groups, which the march adds itself; the rest (the sky and the surfaces through the volumes,
  the light the volumes scatter, other emitters) is projected onto L1 spherical harmonics per colour
  channel. The march scatters it once more, so its single scattering of the probes' light gives all
  orders beyond the first. The probe field (an RGBA32F 3D texture, `VolumeMarchShared.h`) keeps per
  probe the three channels and the occlusion grid's value there, filtered trilinearly within a volume.
  It is a running mean of the frames while the scene stands still and accumulates, 5% a frame when it
  does not.
- **The march** (`VolumeMarch.hlsl`, shared with the CPU in `VolumeMarch.hlsli`): a thread group
  covers 8 x 8 pixels, its lanes in Morton order; it gathers the volumes its rays cross in front of
  their surfaces and marches the first four. A ray steps from the first box it enters to its surface,
  `volumeStep` (1) voxels of `volumeLevel` (0) of the smallest voxel among its volumes, from a jitter
  in [0, 1) (its blue-noise rank moved on by the golden ratio every frame), and jumps over what the
  bricks' distances say is empty. The thread group reads its volumes and their level into groupshared
  memory once (`LoadRayVolume`, `LoadRayLevel`), and a step walks a volume's tables once
  (`BrickSample`: the indirection, then the brick or, where it is empty, the distance); a ray chooses
  each light group's map once (`MarchChooseMaps`), and volumes of one anisotropy share their phase
  function's evaluation. Each step: the volumes' extinction and scattering, the sun's light
  through its map, each light group's (exactly per triangle for up to 8 triangles, by their solid
  angle, Van Oosterom and Strackee, and the phase function at their centre; else a sphere or a disc of
  its area) through its maps, and the probes' light (the harmonics convolved with Henyey-Greenstein,
  which scales band l by g^l) times the ratio of the occlusion grid at the step to the probes' (at
  most 4): the grid is finer than the probes. A step's light is integrated as if constant over it
  (energy conserving); the ray ends where the transmittance falls below 1e-3. Over the surface: the
  volumes' light plus the surface's samples times their transmittance, the air (aerial perspective)
  fused at the depth the volumes' light comes from (its transmittance-weighted mean).
- **Adaptive sampling groups** (`volumeAdaptive`, `MarchRayShared`): lighting is most of a step's cost
  (70-85% here, 80-85% in Mueller's measurements), so a wave lights its steps together where
  neighbouring pixels would light them alike. It marches one grid of steps (from its rays'
  nearest entry, at their smallest step, so that a block's steps lie at the same depths on all its
  rays) and decides per block of steps. Once every ray it still marches has less than
  `volumeShareBelow2` (0.3) of the light from behind left (Mueller's criterion), or where a voxel at
  the block's depth covers `volumeShareFootprint2` (2) pixels or more (volumes close to the camera,
  whose pixels are many and lie within a voxel of each other), each 2 x 2 block of pixels lights 4
  steps, one per pixel at its own ray's depth, and every pixel's step takes the light of the pixel
  that lit it; below `volumeShareBelow4` (0.1) or from `volumeShareFootprint4` (4) pixels per voxel,
  4 x 4 blocks and 16 steps. A pixel lights per unit of scattering (`MarchLightAt`, valid in the empty
  parts of a box too), and only once a step of the block scatters, so empty space costs no light.
  Which pixel lights which step turns with a hash of the block, its first step and the frame: with a
  fixed order the front of the smoke, which shows most, took its light from one side, and wherever the
  light changes across the image it was biased (-5% over the embers at night). The tiles move by an R2
  offset every frame, so that blocks do not stay in place (else the converged light stepped every 4
  pixels). Densities, depths and transmittances stay every pixel's own; the wave skips emptiness where
  all its rays are empty, and all its lanes stay in the loop.
- **Under a denoiser** (DLSS Ray Reconstruction, §8.5): it reprojects its history with the guides,
  which describe the surfaces. Smoke over the sky moved with the sky, and smoke over a far wall with
  the wall, so moving the camera left trails of smoke behind and the smoke's light flickered as the
  history failed to match. As RR's integration guide asks for transparent layers, the march rewrites
  the guides where the volumes hide most of the surface (`DescribeVolumesInGuides`): from an opacity
  of 0.5, depth, motion and normal together describe the volumes (at the depth their light comes from,
  facing the camera), and the albedos (towards the volume's albedo, no specular) and roughness (towards
  1) blend by their opacity. The resolve also writes the frame without the volumes (the trace's samples
  resolved alike, equal to the frame where no volume is in front), `slots::kColorBeforeTransparency`,
  which RR takes as its color-before-transparency guide. Strafing past the `braziers` plumes up close
  at 0.015 m a frame (1920 x 1080, Quality), the last frame against a static camera's at its place
  (display luminance after x / (1 + x)): mean difference 0.0145 -> 0.0010, 99th percentile 0.158 ->
  0.0065 (the guides alone 0.0069, the frame without the volumes alone 0.0085); moving forwards at
  0.02 m a frame 0.152 -> 0.017. Over the last four frames the difference's own change from frame to
  frame (flicker) fell from 0.0126 to 0.0045 (99th percentile). Turning in place had no trails (no
  parallax), and a static camera's image is unchanged (within 0.0005).
- **Over frames** (`volumeTemporal`, `VolumeTemporal.hlsl`): what the march lights changes from frame
  to frame (its jitter, the maps it reads in turn, the blocks that share light). Accumulation and a
  denoiser of the whole frame average that; under a denoiser of the surfaces alone (NRD, §8.14) or
  none, `VolumeMarch.Temporal` does: the march writes its layer (the light in front, the transmittance,
  the depth that light comes from) instead of laying it over the surfaces, and the pass follows each
  pixel's layer point back to the previous frame along that depth, clamps the history found there to
  the 3 x 3 pixels around and blends in `volumeTemporalBlend` (0.1) of this frame. Under NRD at the
  `braziers` default view, the frame-to-frame deviation's 99th percentile went from 0.0079 to 0.0047
  (display luminance), and the grain over the embers is gone. 0.1 ms at 1080p.
- **Debug views** 11 (march steps per pixel, `debugScale` steps to the top of the Turbo colormap),
  12 (sampling rate: blue every pixel, green 2 x 2, red 4 x 4) and 13 (volumes per ray). Direct-only
  (8) and indirect-only (9) leave the probes' and the lights' light out of the march respectively.
- **Render graph**: the probes read the maps on the compute queue while the trace reads them on the
  graphics queue (reads in the same state no longer wait for each other, §5.4).

Parameters: `volumeMarch`, `volumeStep`, `volumeLevel`, `volumeShadowSize` (256), `volumeShadowLevel`,
`volumeShadowStep`, `volumeLights`, `volumeLightOrigins`, `volumeAdaptive`, `volumeShareBelow2`,
`volumeShareBelow4`, `volumeShareFootprint2`, `volumeShareFootprint4`, `volumeAir`,
`volumeShadowsEveryFrame` (render the maps every frame, to time them), `volumeTemporal`,
`volumeTemporalBlend`. Status: `volumeMarch` (whether it runs), `volumeMarchLights` (light groups with maps).

**CPU twin** (`pt::RenderVolumeMarchReference`, `VolumeMarchReference.cpp`; `aircli reference
--volume-march [--frames N] [--probe-frames N] [--volume-step S] [--volume-shadow-size N]
[--volume-lights N]`): the same shader code on the CPU, the maps marched and ray traced against the
reference's BVH and stored as half floats, the probes traced by the reference's kernel
(`AIR_PT_PROBE`), the surfaces by one with `AIR_PT_VOLUME_SHADOWS`, the march at full rate. It
accumulates `--frames` frames of one camera ray per pixel at the frame's sub-pixel offset, `--spp`
paths along it (16 by default). So the method can be measured against the path tracer that tracks
the volumes without a GPU, and the GPU against the method. `pt::VolumeMarchProbe` exposes the maps
and the light groups' light to tests.

Validation (`src/features/pathtracer/tests/VolumeMarchTests.cpp`, `tests/MomentTests.cpp`,
`tests/VolumeBricksTests.cpp`):

- An octant-sized triangle lights the march with exactly an eighth of its radiance (isotropic); a
  sphere and a square of 32 triangles are within 1% of the integral over their triangles, within 8%
  scattering forwards (g 0.5; the phase function at the centre, +5.6% at worst). A single triangle
  spanning an octant is 12% dark at g 0.5.
- The sun's map against the transmittance marched at a quarter voxel through the finest bricks, over
  1246 points the plume shades: 0.049 on average, 0.26 at worst. It is slightly too dark where the
  light enters (-0.05 to -0.08 at transmittances 0.7-1) and too bright deep inside (+0.08 at 0.1-0.4),
  the reconstruction's own error.
- The embers' four maps against the transmittance towards their four points: 0.027 on average; above
  1.6 m they let through 0.39 of the light, the embers' centre alone 0.025.
- Under adaptive sampling, the light a pixel lights for its neighbours times the scattering of their
  step is that step's own light where it is in smoke: within 3e-7 over 370 points of the `braziers`
  (three plumes of one anisotropy, the sun and four light groups).
- White furnace: the plume vanishes, 0.18000 at every pixel.
- Against tracked volumes on the CPU (24 x 24, the plume close up): in the sun from its side -2.9%, at
  night over its embers +0.9% (above the embers). With the probes on a lattice of 4^3 the first was
  -11%; with the embers' map from their centre alone the second was -44%, and -99% for the single
  scattering above the plume's foot.
- `PathTracer_RealTimeVolumesAreMarched` (Null backend): the passes and pipelines, the maps rendered
  once, the probes on the compute queue beside the trace, the status, the debug views, the maps
  rendered again after a change, tracking while accumulating and the march with `volumeMarch=2`.
  `PathTracer_RealTimeVolumesGuideDenoisers`: with a feature that reads the guides, the march writes
  them and the resolve writes the frame without the volumes, not when accumulating.
- GPU (`volumeMarch=2`, accumulating 512 frames of 4 spp, 160 x 90) against the tracked CPU reference
  (512 spp): `smoke` -0.01% (the plume +0.55%, relative MSE 0.006), `braziers` +1.1% (the plumes
  +1.4%); the GPU and the CPU twin agree within 0.3% in the plume.
- Adaptive sampling against the full rate (`volumeAdaptive=0`; GPU, 1920 x 1080, 256 frames
  accumulated, where a voxel covers about 5 pixels): the image means within 0.04% in the sun and at
  night; per 8 x 8 pixels at most 2.1%, 99% of them within 0.6-1.5%, where hard shadow edges on the
  smoke (a wall's) soften by up to 3 pixels. From 24 m away, where nothing shares by footprint, within
  0.25% everywhere.

Cost (RTX 5070, 1920 x 1080, 1 spp, `accumulate=0`):

| Scene | Tracked | Trace | March | Probes (async) | Busy |
|---|---|---|---|---|---|
| `smoke` | 5.3 ms | 1.9 ms | 0.48 ms | 0.44 ms | 2.4 ms |
| `braziers` | 10.1 ms | 2.4 ms | 1.19 ms | 0.92 ms | 3.6 ms |

Without adaptive sampling the march takes 0.86 ms (`smoke`) and 3.4 ms (`braziers`), 0.83 and 2.7 ms
sharing by transmittance alone: in these close-ups (a voxel covers about 5 pixels) the footprint does
most of it. Without any lighting it takes 0.30 and 0.61 ms. The brick distances, the volumes in
groupshared memory and one walk of the tables per step took a tenth off. Four maps per ember bed
cost the `braziers` march 0.03 ms over one map. Mueller reports 0.63-1.82 ms for the main view of
three production scenes at 1080p on an RTX 5060 Ti (the shadows on async compute aside).

Not modelled: the light the volumes scatter onto surfaces (a wall next to the glowing plume at night
is 8-11% darker than tracked), volumes seen in reflections and refractions (BSDF-sampled rays from
surfaces do not meet them), more than four volumes per 8 x 8 pixels, the march through the cloud
layer (§8.10, which the trace kernels still render), detail finer than the probes beyond what the
occlusion grid restores, emission.

### 8.14 Real-time denoising: NRD and TAA

In real time (`pathtracer.accumulate=0`, 1 spp) the path tracer's frame is noise. Two chains turn it
into an image:

| Chain | Features | What it denoises |
|---|---|---|
| default (editor) | `pathtracer,nrd,taa,tonemap` | NRD the surfaces' reflected indirect light, TAA the edges and the direct light; the volumes filter themselves (§8.13) |
| DLSS | `pathtracer,dlss,tonemap` | Ray Reconstruction the whole frame (§8.5), also anti-aliasing and upscaling |

A feature that consumes `kColorBeforeTransparency` denoises the whole frame: nrd and taa then stand
aside, and the path tracer describes the volumes in the guides (§8.13).

**Signals** (`slots::kDiffuseSignal`, `kSpecularSignal`, `Integrator.hlsli`): NRD denoises diffuse
and specular indirect light separately, divided by their albedos, with the length of the ray that
brought it. `TracePath` (with `kFlagSignals`) splits the light its first vertex reflects beyond its
next event estimation by the diffuse part of the BSDF sample it continued with
(`BsdfSample::diffuseWeight`), a medium's scattering as diffuse; its own emission, its direct light
and a miss's sky are in neither. ReSTIR GI splits its shading alike and adds it to the signals.
The direct light (next event estimation, ReSTIR DI) passes NRD: filtered by it, the shadows spread
into the lit surfaces around them (edge-stopping by normals and depth does not see a shadow's edge),
and the frame lost 2% (`textured`) to 10% (`many_lights`) of its light; without it the frame is also
closer to the reference (`textured`: one frame's RMSE 0.019 -> 0.0065), its residual noise left to
TAA. The albedos (`GuideAlbedos`, `Bsdf.hlsli`) are the lobes' directional albedos as the BSDF weighs
them (the diffuse lobe's `1 - Es(wo)`, a coat's inner part). The signals are exposed like sceneColor
and dimmed by the air like it, but not by the volumes in front: those pass `kSurfaceTransmittance`.

**nrd** (`src/features/nrd/`, D3D12, `AIR_WITH_NRD`): NVIDIA's NRD 4.18 built from source
(`cmake/AirThirdParty.cmake`: NRD, ShaderMake and MathLib downloaded into `.deps`, its shaders compiled
to DXIL with our dxc; normal encoding R10G10B10A2, linear roughness). Three passes after the path
tracer, at the render resolution:

| Pass | What it does |
|---|---|
| `NRD.Prepare` | each lobe's light divided by its albedo (at least 0.01), sanitized and packed with NRD.hlsli (RELAX: with the hit distance; REBLUR: YCoCg and the normalized hit distance), the normal and roughness packed |
| `NRD.Denoise` | NRD's own dispatches, recorded with D3D12 into the RHI's command list: a root signature and pipeline per NRD pipeline (root CBV, one descriptor table, static samplers), its textures (committed, states tracked across dispatches), a shader-visible descriptor heap and an upload buffer for its constants, both rings over the frames in flight. The render graph's textures end in the states it declared |
| `NRD.Composite` | `color += transmittance * (diffuseAlbedo * (denoised - noisy) + specularAlbedo * (...))`: exactly the signals NRD saw are replaced; emission, sky, volumes and anything the signals do not carry stay as they were |

The editor renders below the viewport under NRD (Render panel, "Render scale", 0.67 by default) and
TAA upsamples: at 1920 x 1080 a frame of `sponza` took 22.8 ms at full size, 10.3 ms at 0.67 and
5.8 ms at 0.5, with the image at 0.67 hard to tell apart. Primary rays are about 1 ms of such a frame
(`maxBounces=0` without NEE or ReSTIR), so a rasterized first hit would save at most that.

Parameters: `enabled`, `denoiser` (0 RELAX, the default; 1 REBLUR), `historyFrames` (30),
`antiFirefly` (off: it clamps bright samples, 4-5% of `sphere_field`'s light), `view` (1: the
denoised signals alone, the global illumination, in place of the frame; the editor's "GI only"). Camera
matrices are the left-handed view with the path tracer's view depth, the motion vectors the path
tracer's (pixels, previous - current) scaled to UV, the jitter its own.

**taa** (`src/features/taa/`, any backend): the Upscale stage behind nrd. Each output pixel gathers
the 3 x 3 render samples around it (at pixel + 0.5 + jitter, a Gaussian of the distance), reprojects
its last result with the motion vectors (Catmull-Rom, kept within the 2 x 2 texels around it), clips
it to a box in YCoCg and blends in 0.1 of the frame; so it also upsamples a frame rendered below the
output size. It stands aside for accumulated frames (`kRealTime` 0) and debug views. The box decides
whether the mean survives (`Taa.hlsli`): around the frame's own noisy samples (mean +- 1.25
deviations) it is drawn from skewed noise, mostly dark with a few bright samples, so its centre lies
below the true mean in most frames and its width is small, and the history (which holds the true
mean) was clipped down: TAA alone took 5% of `sphere_field`'s light. So `TAA.Fast` first keeps a fast
history (0.25 of each frame, clipped only far out: 3 deviations widened by the pixel's temporal
deviation, whose second moment its alpha keeps), and `TAA` clips a still pixel's history to the fast
history's 3 x 3 box (centre unskewed by the frames it averages; 1.6 x `clip` deviations widened by
what the frames' temporal variance leaves in it). A moving pixel (half a pixel of motion, or any
camera motion: smoke over the sky has the sky's motion vectors) uses the frame's own box as before,
the fast history too, so that nothing trails; it gives up some of the mean while it moves. Both
histories round to half precision at random (`StochasticHalf`): converting on store truncates on this
GPU, and an average that rounds every update down settles half a step times the frames it averages
below its inputs.
Weighing samples by 1 / (1 + luminance) against fireflies (Karis) took 5% of the Cornell box's ceiling
light, whose few pixels hold half the image's energy: the input is denoised, so it is not done.

Measured (RTX 5070, D3D12; display luminance after x / (1 + x)):

- `cornell_box` 960 x 540: one frame's RMSE against 4096 accumulated spp 0.027 noisy, 0.013 NRD,
  0.0067 NRD + TAA.
- `braziers` 960 x 540: RMSE 0.128 noisy, 0.016 NRD, 0.014 NRD + TAA; mean +0.01%.
- `braziers` 1920 x 1080 static: the frame-to-frame deviation's 99.9th percentile 0.073 with NRD
  alone (jittered edges crawl), 0.0084 with TAA (DLSS RR 0.0064). Strafing past the plumes up close
  at 0.015 m a frame, the last frame against a static camera's at its place: 99th percentile 0.013,
  the error's flicker 0.0035 (DLSS RR 0.012 and 0.0045).
- Cost at 1920 x 1080: `NRD.Denoise` 1.7 ms (`cornell_box`) to 2.0 ms (`braziers`), `NRD.Prepare`
  and `NRD.Composite` 0.1 ms each, TAA 0.1 ms.

#### Energy

A denoiser of noise that is skewed (rare bright samples) loses light wherever it clips or clamps, and a
filter that spreads shadows loses it where they meet the light. The mean of 32 real-time frames (after
96, 480 x 270) against 2048 accumulated ones, and one frame's RMSE in display luminance x / (1 + x):

| Scene | Before (energy / RMSE) | Now | NRD alone | TAA alone | Frames alone |
|---|---|---|---|---|---|
| `sphere_field` | -9.4% / 0.046 | -2.2% / 0.029 | -1.5% | -0.9% | -0.1% |
| `many_lights` | -12.9% / 0.067 | -2.7% / 0.053 | -1.7% | -1.7% | -0.1% |
| `textured` | -3.0% / 0.019 | -1.2% / 0.0065 | -1.3% | -0.3% | -0.2% |
| `coated` | -3.5% / 0.023 | -1.7% / 0.0088 | -1.7% | -0.8% | -0.6% |
| `cornell_box` | -1.5% / 0.0051 | -0.5% / 0.0083 | -0.5% | -0.1% | 0.0% |
| `material_grid` | -1.7% | -0.5% / 0.0036 | -0.5% | -0.2% | -0.1% |
| `braziers` | | +1.0% / 0.011 | +1.0% | +0.9% | +1.0% (the march's own, §8.13) |

- What went: NRD's anti-firefly (4-5% of `sphere_field`), NRD on the direct light (shadows spread:
  2-10%), TAA's clipping to the frame's skewed neighbourhood (5% of `sphere_field` alone), half
  precision truncation in TAA's history (0.1-0.2%), guides that did not follow the lobes.
- What is left: NRD on the indirect light, 0.5-1.7%, and none of RELAX's settings (history clamping,
  antilag, luminance edge stopping, pre-pass blur, history length, 32-bit textures) moves it; about
  two thirds of it at `sphere_field`'s edges, where the jitter switches a pixel between surfaces. TAA
  0.1-1.7%. In a white furnace (all light indirect, ReSTIR GI's noise heaviest) the chain is still
  3-4% below accumulation on `textured` and `coated`.
- Cornell box: the RMSE rose (0.0051 -> 0.0083) where the light's edge meets the dark ceiling. TAA's
  Gaussian reconstruction (`blend` 1: the frame's own mean) puts 0.38 nits outside the edge where the
  reference's box filter has 0.05; the old clipping cut that halo, and some of the light with it.
- Trails (the last frame of a strafing camera, 64 frames, against a static camera's at its place,
  99th percentile of the error): `braziers` 0.0153 (before 0.0153), `cornell_box` 0.0110 (0.0106),
  `sphere_field` 0.096 (0.085).

Validation: `Nrd_RefusesOnNullDevice`, `Taa_FiltersRealTimeFramesOnly`, `Taa_KnowsWhenTheCameraMoves`,
`PathTracer_SignalsForDenoisersOfSurfaces` (the trace and ReSTIR write the signals, the march leaves
the guides alone and filters its layer), `Editor_DenoiserChoice`; on the GPU the D3D12 debug layer
reports no errors for `cornell_box` and `braziers` with `pathtracer,nrd,taa`.

### 8.15 Material textures

A material may take a texture set (`SceneDesc::textureSets`, `Material::textures`): two RGBA8
textures of one size that repeat, `color` (the base colour, sRGB) and `surface` (the tangent-space
normal's x and y mapped to [0, 1] with z following, roughness, metallic). They multiply the
material's base colour, roughness and metallic (set those to 1 to take the textures as they are) at
the mesh's texture coordinates times `Material::uvScale`.

- **Shading** (`LoadSurface`, `Integrator.hlsli`): both textures bilinear at level 0 (a path has no
  footprint to choose a level by; the real-time denoisers and TAA filter what it aliases). The normal
  map bends the shading normal in the triangle's tangent frame along increasing u and v (from its
  positions and texture coordinates), orthogonalized against the interpolated normal; a bent normal
  below the geometric surface is ignored. What its lobes send below the surface is mirrored back above
  it (§8.2, "Shading normals"), the specular lobes' is lost. Hooks: `LoadTextureSet` (a bindless
  `StructuredBuffer<TextureSetData>`, `IntegratorParams::textureSets` from `FrameData::textureSets`)
  and `MaterialTextureSample` (bindless `Texture2D`, the static linear wrap sampler); on the CPU,
  `SampleTexture2DWrapRGBA8` emulates the hardware: 8 fractional bits, sRGB decoded before filtering.
- **Cutouts** (`Material::alphaCutoff`, `Cutout.hlsli`): where the colour texture's alpha is below
  the cutoff the surface is a hole (leaves, chains, decals). The instances of cutout materials are
  non-opaque in the TLAS (`TlasInstance::opaque`, D3D12 `FORCE_NON_OPAQUE`), every other instance is
  forced opaque, so the ray queries (`TraceClosest`, `TraceVisible`, ReSTIR's, the probes'; the SER
  variant makes its hit object from such a query) see candidates only on cutouts and commit those that
  `AlphaCutsOut` keeps. The CPU reference's BVH runs the same function on a cutout's triangles. Scenes
  without cutouts measured the same trace time as before. The cutoff is fixed once the scene is
  uploaded (`SetMaterial` keeps it); neural texture compression leaves the cutouts' sets alone (the
  codec drops the colour's alpha); the volumes' moment shadow maps take every triangle as opaque.
- **Procedural sets** (`MaterialTextures.cpp`): a height field per material whose slopes give the
  normal map, with colour, roughness and metallic from the same pattern, from integer hashes and
  value noise that repeat with the texture. `MakeBrickTextures` (red bricks in mortar),
  `MakeTileTextures` (glazed tiles in grout), `MakeMetalPlateTextures` (brushed metal, seams, rivets).
  The `textured` scene uses all three at 512 x 512.
- **Validation**: `MaterialTextures_*` (sampling, the sets, Validate), `Integrator_TexturesScaleTheMaterial`
  (a flat grey texture renders as that grey), `Integrator_NormalMapsKeepTheLightInAWhiteFurnace`:
  `textured` in a white furnace (16 bounces) is within 1.2% of the environment (3.2% dark before the
  mirror of §8.2) and gains nothing. GPU against the CPU reference on `textured`: +0.05%.
  `Integrator_CutoutsLetLightThrough` (a half-transparent panel before an emitting wall).

### 8.16 Neural texture compression

A scene's texture sets may go through neural texture compression (Vaidyanathan et al., *Random-Access
Neural Compression of Material Textures*, SIGGRAPH 2023) with NVIDIA's LibNTC (RTXNTC 0.10 beta):
a set's seven channels (colour, the normal's x and y, roughness, metallic) become a few grids of
4-bit latents and a small MLP (48 -> 64 -> 48 -> 32 -> 16, int8 weights) that decodes a texel from
them and its position. `cmake/AirThirdParty.cmake` downloads the SDK's prebuilt `libntc.dll` (Windows
x64) and the library's headers and shaders at the commit that release pins (`AIR_WITH_NTC`, never
committed); `renderer/NeuralTextures.h` loads the DLL at run time from next to the executable. CUDA
compresses and decodes through the driver (no toolkit); without the DLL or an NVIDIA GPU the codec
says why and scenes keep their textures.

- **Modes** (`NeuralTextureOptions`, `aircli --neural-textures off|load|sample[,BPP[,STEPS]]
  [--ntc-cache DIR]`, the editor's Scene panel and `aireditor --neural-textures`):
  - **Decode on load**: compressed, then decoded once on CUDA into ordinary textures: the memory on
    disk / in transit shrinks, the renderer is unchanged.
  - **Decode on sample**: the GPU keeps only the latents (an RGBA8 2D array: LibNTC's BGRA4 texels
    widened, which filter the same since x / 15 is exact in 8 bits) and the MLP's constants and
    weights (`TextureSetData::latents`, `neural`), and the path tracer decodes a texel at every
    camera-visible hit with LibNTC's own shader code (`libntc/shaders/Inference.hlsli`, int8 dot
    products). The filter is stochastic (Pharr et al., *Filtering After Shading With Stochastic
    Texture Filtering*, 2024): the hit picks one texel with a hash of the hit and the frame, the
    bilinear weights as its probabilities, so one decode replaces four. Deeper vertices, whose
    footprints are wide, read the set's coarse copy (a quarter of its resolution, box filtered in
    linear light, `kNeuralTextureCoarseFactor`; what ray cones would choose at most of them), and
    ReSTIR reads the first surface's material from `PrimaryHit` (§8.7), so a pixel decodes once a
    frame. The decoder lives in the kernel variants with the neural parts (`PathTracerNeural.hlsl`,
    `PathTracerGiNeural.hlsl`, `AIR_PT_NEURAL`; also the clear coats' models, §8.17), used while the
    scene has such sets: compiled into the other kernels its registers cost them 7-10%. The volume and
    radiance cache kernels read the coarse copy everywhere.
  - Either way the scene's texels become the decoded ones (with the int8 weights, as the shader
    decodes), so the CPU reference renders what the GPU does.
- **Compression** (`NeuralTextureCodec`): `PickLatentShape` for the rate (4 bits per texel asks for
  3.75), 20000 training steps (about 4 s a set on an RTX 5070 whatever its size; LibNTC's default
  100000 adds a few dB), the colour learnt in sRGB. Results are cached by content and settings
  (`<temp>/airender-ntc`). Every channel must belong to a texture of the set's metadata
  (`AddTexture`, `SetChannels`), else LibNTC learns NaN.
- **Measured** (`textured`'s three 512 x 512 sets): 2 MiB of RGBA8 each to 113-140 KiB (14-18x), PSNR
  54-56 dB at 20000 steps (58-63 at 100000; photographed materials reach about 30 dB at 4 bpp). Images
  accumulated over 2048 frames: decode on load +0.04% against the source textures; decode on sample (the
  stochastic filter at the first surface, the coarse copy beyond) -0.12% against decode on load, as
  much as decoding four texels bilinearly at every vertex (the shader's int8 decode against CUDA's);
  the stochastic filter at every vertex gave -0.28%, shading single texels where bilinear shades their
  average (normal maps and roughness are not linear in the image). Cost at 1920 x 1080 in real time:
  `PathTracer.Trace` 3.9 ms with ordinary textures, 7.9 ms decoding on sample (the whole frame 8.4 ->
  12.4 ms); decoding at every vertex and in every ReSTIR pass took 81 ms. LibNTC's cooperative-vector
  path (FP8 weights on the matrix units) needs its weights converted on the device by LibNTC with our
  D3D12 device, not done yet.
- **Validation**: `aircli ntc-check` decodes every texel of `textured`'s sets with the shader
  (`NtcCheck.hlsl`, the code the path tracer runs) and compares with CUDA's decode: under 1 / 255 RMS
  per channel, means within 3e-4. `NeuralTextures_*` (the coarse copy, a scene's latents and its
  kernels on the Null device, and with LibNTC a codec round trip and the cache), and
  `Editor_NeuralTexturesReloadTheScene`; on the GPU the debug layer reports no errors.

### 8.17 Neural shading: clear coats

A material may wear a clear coat (`Material::coat`, `coatRoughness`, `coatColor`): a rough
dielectric interface (index 1.5, GGX) over a layer that absorbs (`coatColor` is what it lets through
at normal incidence, one way) over the material itself. The light the interface reflects has a
closed form; the light that enters, bounces between the base and the underside of the interface and
leaves again (the inner part) has none. Layered materials like this are where neural materials pay
off (Zeltner et al., *Real-Time Neural Appearance Models*, 2024): a small network stands in for an
evaluation that is otherwise a stochastic simulation. `pathtracer.coatModel` chooses how the inner
part is evaluated (`Bsdf.hlsli`):

| Model | What it is |
|---|---|
| 0 neural (default) | the approximation times `exp(z)`, z from an MLP trained for the scene's coated materials: noise free |
| 1 random walk | the reference: one walk between the interface and the base per evaluation, with next event estimation to wi from both ends weighed by the power heuristic (position-free layered BSDF, Guo et al. 2018, as pbrt-v4's `LayeredBxDF` evaluates it); unbiased and noisy |
| 2 approximate | one bounce off the base through a smooth interface, `(1 - F(wo)) (1 - F(wi)) f_base(wo', wi') T(wo') T(wi') / eta^2`, times `1 / (1 - albedo * 0.596)` for the diffuse light the interface sends back down |

- **The interface**: pbrt-v4's rough dielectric (reflection and refraction, either side, radiance and
  importance transport, visible normal sampling restricted to a lobe), the Fresnel term exact.
- **Sampling** serves all three models: the interface's reflection with probability
  `F(wo) / (F(wo) + (1 - F(wo)) * luminance(base * coat^2))` (clamped to [0.1, 0.9]), else the base's
  lobes; the inner part evaluated at what was drawn is weighed by that mixture's pdf, so every model
  is unbiased with respect to its own BSDF and MIS works as for any material. The walk evaluates from
  a seed hashed from the hit, the frame and wi: the same direction gets the same estimate wherever a
  frame evaluates it.
- **The network** (`NeuralCoat.hlsli`, shared by HLSL and C++): 32 inputs (the directions outside and
  refracted into the layer, their half vectors, azimuth, how far the base's lobe peaks there, sines
  and cosines of those angles, the material), two hidden layers of 16 ReLUs, three outputs, 852
  parameters. It corrects the approximation multiplicatively (`|z| <= 6`), so the analytic part keeps
  the peaks and the colour and the network learns a smooth factor.
- **Training** (`NeuralCoat.cpp`, on the CPU): `CollectCoatMaterials` samples the scene's coated
  materials, a textured one by its texels; every step draws 4096 samples (a material, wo from grazing
  to normal incidence, wi half from the sampling above and half uniform), each with the mean of four
  walks as its target, and takes an Adam step on the relative L2 error of the inner part (unbiased in
  the targets' noise: the gradient is linear in them). 1500 steps take about 2.5 s on 23 threads,
  deterministic (chunks summed in order); results are cached by the materials and settings
  (`<temp>/airender-neural-coat`; the version in its key goes up when the coat's code changes). The
  path tracer trains when the coated materials change: at once without a network, else once the scene
  has been unchanged for 30 frames (an editor slider keeps the last network meanwhile); the CPU
  reference trains (or finds) the same network. Its weights reach the shaders as a constant buffer
  (`Device::GetBindlessCBV`): every thread reads the same weights.
- **Kernels**: coats are compiled only into the kernels with the neural parts (`AIR_PT_NEURAL`:
  `PathTracerNeural.hlsl`, `PathTracerGiNeural.hlsl` and `ReSTIR*Neural.hlsl`, used while the scene
  has coated materials or neural textures); elsewhere a coated material is its base. Compiled into
  every kernel, even unused, the coat cost `cornell_box` 8.0 -> 24.3 ms a frame (the walk's loop inlined
  into ReSTIR's resampling loops: 47x in `ReSTIRDI.Initial`), its approximation alone 8%. The volume and
  radiance cache kernels render coated materials without their coats.
- **Measured** (`coated`, RTX 5070): the network's inner part against 256 walks per direction, over
  the scene's material samples: 2.2% relative RMS error (the approximation 12%; much of the 2.2% is the
  reference's own noise); directional albedos within about 1% of the walk's. Images accumulated over
  2048 frames at 320 x 180 against the walk: network +0.01% (blocks of 8 x 8 pixels: median +0.01%,
  5th-95th percentile -0.2% to +0.5%), approximation -0.71%; the CPU reference with the same network
  agrees with the GPU's (block median +0.02%). White furnace (the bases white, the layers clear): the
  walk 0.940 (a rough interface at grazing angles loses some light to its single scattering) to 1.008
  (`Coat_WhiteFurnaceRandomWalk`); with the scene's own shapes (64 bounces) the image means are walk
  -0.13%, network -1.5% (white bases are outside what it trained on), approximation -4.1%. Cost at
  1920 x 1080 in real time (frame / `Trace`): the approximation 11.4 / 5.0 ms, the network 16.7 / 6.3
  ms, the walk 22.6 / 6.1 ms (and noisy).
  A walk limited to 10 events (pbrt's default) lost 8% of a white coated base: light the interface
  traps by total internal reflection keeps much of its weight for several round trips, so the limit is
  64 and Russian roulette ends walks early.
- **Car paint** (`Material::flakes`, `flakeRoughness`, `flakeColor`; scene `car_paint`): under the
  coat, a share of what the light meets is metallic flakes, a metal of `flakeColor` whose flakes'
  normals spread like a GGX surface of `flakeRoughness`; the rest is the material's own base (its
  pigment). The walk's base is that mixture (`CoatBaseContext`: both lobes evaluated, either sampled
  by the share), the approximation takes it too, and from the outside the base's samples draw the
  flakes' GGX lobe about the mirror direction in that share (`BsdfContext::flakeProbability`), so
  sampling, evaluation and MIS agree. A coloured layer over them (`coatColor`) tints the flakes'
  light as a candy paint's does. The network's inputs grew from 32 to 40: the flakes' share,
  roughness and colour, and how far their lobe peaks at the refracted half vector; without flakes
  they are 0, and the first layer's other weights are drawn as before, so a coat over a plain base
  trains exactly the network it did (980 parameters). On `car_paint` (accumulated, 640 x 360, 256
  frames): the network against the walk -0.02% (16 x 16 blocks within -0.4% .. +0.2%), the
  approximation -0.31% (5th percentile -6.1%). Not modelled: the sparkle of single flakes (glints).
- **Validation**: `aircli coat-check [--scene NAME]` (the network and the approximation against the
  walk: errors, albedos), `Coat_*` (white furnace, absorption, reciprocity of the walk, sampling against
  uniform integration for the walk and the approximation, with flakes too, kernels on the Null device),
  `Coat_FlakesConserveEnergy` (white flakes in a white pigment reflect all the light; gold flakes in a
  blue pigment reflect gold towards the mirror direction and blue elsewhere),
  `NeuralCoat_LearnsTheRandomWalk`.

### 8.18 Scenes from model files

`renderer/SceneFiles.h` reads model files into a `SceneDesc` and composes the file scenes of §8.1 from
them. Their assets are never committed: `python3 tools/air.py fetch-scenes [NAMES]` downloads them
from pinned commits into `.deps/scenes/<scene>/` (`AIR_SCENE_ASSETS` elsewhere; the renderer looks
there too), each with its license:

| Scene | Source | License |
|---|---|---|
| `picapica` | SEED (Electronic Arts), `SEED-EA/pica-pica-assets` at b433c2e (an archive, sha256 checked) | CC BY-NC 4.0 |
| `sponza` | Crytek; `KhronosGroup/glTF-Sample-Assets` at 8f49239, `Models/Sponza` | the license that repository names for it |
| `helmets` | the same repository: `FlightHelmet` (CC0), `DamagedHelmet` (CC BY 4.0, ctxwing), `SciFiHelmet` (CC0) | as listed |

A file scene without its assets fails to load with the command that fetches them. `aircli scenes`
lists the file scenes too (loaded when their assets are there), `aircli render|reference --model
FILE[,X,Y,Z[,HEIGHT[,YAW]]]` adds any model file to any scene, and the editor's scene list holds them.

- **Libraries** (`cmake/AirThirdParty.cmake`, `AIR_WITH_MODEL_FILES`, downloaded into `.deps` like the
  SDKs): ufbx 0.23.1 (FBX, OBJ), cgltf 1.15 (glTF), stb_image (PNG, JPEG, TGA), compiled into one
  static library without our warnings. Without them `ModelFilesAvailable()` is false and imports
  report it.
- **Import** (`ImportModel`): a mesh per mesh and material part (glTF primitive), an instance per node
  with its world transform, in metres and y up (ufbx converts FBX's axes and unit; `metresPerUnit`
  overrides a file whose unit is wrong). FBX and OBJ count v upwards: v becomes 1 - v. Normals the file
  lacks come from the faces. `materialFor` replaces a material by its name, `skipMaterial` and
  `skipNode` leave geometry out.
- **Materials**: glTF's metallic-roughness factors and textures (base colour sRGB with its alpha;
  roughness from green, metallic from blue), normal map, alpha mode (`MASK` and `BLEND` become cutouts
  at their cutoff, 0.5 for blend), a constant emission (`emissiveFactor`, strength; emissive textures
  are left out), a clear coat (`KHR_materials_clearcoat` above 0.5). `PackMaterialTextures` resamples a
  material's textures to one size (the largest, at most `maxTextureSize`, 1024 by default, 2048 for
  `helmets`): box filtered down, bilinear up; decoded in parallel, each file once.
- **Normal maps**: glTF's +y points up the image, towards decreasing v, while the renderer bends along
  increasing v (§8.15), so green flips (`normalGreenUp`); the same holds for FBX and OBJ after their v
  flip. Checked with the glTF sample `NormalTangentMirrorTest` (`--model`): the normal-mapped spheres,
  with mirrored texture coordinates too, are lit like the real geometry beside them.
- **picapica**: the PICA PICA files hold geometry and material names only (the demo's materials are
  not in the release), and the names say what they were: `PicaMaterial` makes coloured plastics in
  matte, satin and glossy finishes, metals, rubber and dark glass of them, a clear coat for
  `Coated_Black_Glossy`, emissive LEDs and lights for the robots. The mug, the books, the crayons'
  labels and the decals take the textures in the release's folders (the decals as cutouts); the
  machines' `Plastic_Decals` have none and are left out, as are the blue display stands each model was
  shown on (the diorama's is its floor). Scales differ by file and the mug's unit is wrong, so they are
  given per file. One robot is imported and six more instance its meshes.
- **sponza**: every material of the file scales its texture by 0.588 (a constant diffuse factor the
  original export carried), which leaves its stone reflecting 10-20% where limestone reflects 40-60%;
  as most of the arcades' light arrives by bouncing, that darkened them by the factor per bounce. The
  scene keeps the textures alone (stone 25-36%), at EV100 11 for the shade that fills the view.
- **Measured** (RTX 5070): loading takes 3.9 s (`picapica`), 1.8 s (`sponza`), 1.9 s (`helmets`), most
  of it decoding textures; a real-time frame at 1920 x 1080 with `pathtracer,nrd,taa,tonemap`: `sponza`
  21 ms (`Trace` 8.2 ms), `picapica` 18 ms (5.3 ms). The D3D12 debug layer reports no errors for the
  three in real time. GPU accumulation against the CPU reference: `sponza` -0.08%, `picapica` -0.04%.
- **Limits**: emissive textures, transmission and texture transforms are not read; textures are
  sampled at level 0 (§8.15).
- **Validation**: `SceneFiles_*` (texture packing and the normal's flip, a glTF and an OBJ written by the
  test: transforms, units, v, materials replaced and skipped; the names and the missing-assets
  message), `Integrator_CutoutsLetLightThrough`.

### 8.19 Real-time clouds

In real time (`pathtracer.cloudMarch`, 0 off, 1 when not accumulating (the default), 2 always) the
clouds of §8.10 are not walked through by every path: a `cumulus` frame at 1920 x 1080 under
`pathtracer,nrd,taa,tonemap` took 93 ms (`Trace` 89 ms), and takes 8.0 ms with them marched (the
clouds' passes 0.76 ms). `src/CloudMarcher.h` follows Schneider's Nubis (the clouds of Horizon Zero
Dawn and Forbidden West, SIGGRAPH 2015-2023): the density is ray marched rather than tracked, empty
space is skipped whole (the majorant cells of `CloudLookup.hlsli`, which bound the density the way
Nubis' signed distance field bounds its voxels), a frame marches a sixteenth of the render's pixels
and reprojects the rest. Its passes run before `PathTracer.Trace`, into three maps the path tracer
reads with `kFlagCloudMaps` (`CloudMaps.hlsli`):

| Pass | Map | Read by |
|---|---|---|
| `Clouds.Shadow` | R32F, 512 x 512 over one tile of the clouds (they repeat): the optical depth along the sun from each point of their base up through the layer (steps of at most 25 m), indexed where the drifting base lies; rendered when the scene changes and whenever the wind has moved the detail 16 m over the base | every sun NEE below the clouds (`CloudSunTransmittance`): the transmittance of the point where the sun's ray crosses their base, instead of ratio tracking |
| `Clouds.Sky` | RGBA32F, 128 x 128 over the upper hemisphere (the octahedron's upper half): what the clouds add to and leave of the sky towards each direction, seen from the camera (48 steps, fixed offsets) | rays that leave the scene after a bounce (`EscapeRadiance`): the clouds are kilometres away, so one viewpoint serves the scene |
| `Clouds.March` | one pixel of every 4 x 4 block of the layer, a different one each frame in a Bayer order (96 steps; the ray, its steps and its samples towards the sun jittered) | `Clouds.Reconstruct` |
| `Clouds.Reconstruct` | RGBA32F + R32F at the render's size: the marched pixels blended into their history (0.5), the others reprojected along the depth of the clouds' light (and back along the wind's step), read through a Catmull-Rom filter and clamped to the march's 3 x 3 neighbourhood as far as parallax moves it (from a quarter of a pixel, fully from one) | the camera's rays (`AddCloudLayer`): the clouds' light added and what they let through scaling the sample and the denoiser's signals, where the ray found nothing before their front |

- **Light at a step**: the sun's single scattering with the clouds' phase function (§8.10, delta-M
  scaled) and an optical depth towards the sun from 8 samples whose steps grow by 1.6 from 20 m (to
  1.4 km), each at a place jittered along its step every frame: at fixed places the far, long steps
  stamped the outlines of the clouds they reached into the light as bands; the
  light that scattered more than once, isotropic, falling off as the two-stream solution of a
  conservative slab, `4.4 sqrt(density) / (1 + 0.3 tau)` times the sun's irradiance over 4 pi (octaves of
  single scattering, Wrenninge et al. 2013, came out 50% dark inside cumulus, whose optical depths
  towards the sun are tens); the sky's light from 8 directions above and below, `0.4 (up (0.3 + 0.7 h)
  + down (1 - 0.7 h))` at a height h in the layer. Each step integrates its light with its own
  extinction (Hillaire 2016). The air in front dims the clouds' light and adds its own where they hide
  the sky, as a path's walk sees it.
- **Matched** to the path traced clouds (`cloudMarch=0`, accumulated) on `cumulus` at 640 x 360, the
  march accumulated with `cloudMarch=2`: the image's mean -2.4% (8 x 8 blocks: 5th percentile -9.3%,
  95th +20.5%; the march's sunlit sides a little dark, their shaded sides a little bright), -0.1%
  with the sun at 25 degrees, azimuth 160 (-6.3%, +24.6%); cirrus included. (The diffused light scaled with the
  density itself made the thinner stratocumulus 8% dark: with its square root.)
- The layer was half the render's size at first (a quarter of its pixels; under the editor's render
  scale of 0.67 a ninth of the output's): the clouds were soft and their outlines blocky against the
  sky. At the render's size with a 4 x 4 pattern the march lights as many pixels a frame as before,
  and a still view is sharp after 16 frames, its outlines antialiased by the jittered rays.
- **Cirrus**: the march crosses the sheet behind the layer once (`MarchCirrus`): the sun's single
  scattering through a thin slab in closed form, `p (1 / muV) exp(-tau / muV) (1 - exp(-tau (1 / muS -
  1 / muV))) / (1 / muS - 1 / muV)` for the cosines muV and muS of the view and the sun, and the sky's
  light from above scattered as much as the sheet dims, through the air in front of it. It dims the sun
  of the layer's steps (taken where the ray enters the layer) and of the shadow map's lookups (exactly:
  the map does not hold it, its wind is not the layer's). The sun's term grows by `1 + 0.6 tau / muV`
  for the light the sheet scatters on forwards more than once. Matched to the path traced sheet
  (`cumulus` without its layer, `--clouds 0.00001`): +0.4% (the upper sky 0.0%), +0.1% with the sun
  at 25 degrees, azimuth 160.
- **Fine structure**: clamping the history to the sparse march (texels 4 pixels apart) erased
  whatever was finer, the cirrus' fibres most (they came out smeared and 18% dark in places). The
  clamp only guards against parallax, clouds at other depths coming into view: where the cloud's
  depth moves the reprojection less than a quarter of a pixel from that of a point at infinity the
  history is kept as it is (a camera that turns, or stands still), and fully clamped from a pixel.
  A camera flying 30 m a frame while turning shows no trails.
- **Flow** (`Clouds::flow`, 250 m, `flowPeriod`, 16 s; `CloudDetailFlow`): while the bodies drift, their
  fringe churns in place. The detail's coordinates move along a swirl (the curl of two products of
  sines, cells of 300 to 600 m, rising at 0.2 to 0.6 of it) by up to `flow` metres; two phases half a
  period apart are cross-faded, each restarting where the other has all the weight (a flow map: motion
  without the stretching of advecting the texture for ever). The detail only erodes, so the majorants
  still bound the density and the paths' walks see the same clouds. The shadow map is rendered again
  every 32nd of the period. Against the march accumulated at the same clock the real-time layer's
  error is 0.00090 at 1/60 s a frame (0.00085 without the flow). The editor drives the clouds' clock
  with its frames' wall time (`cloudWindStep` each frame) times a time-lapse factor. The clouds'
  passes at 1920 x 1080: 1.17 ms (sky map 0.42, march 0.52, reconstruction 0.23).
- **Wind** (`CloudData::windX` / `Z`, from `Clouds::windSpeed` and `windHeading`, 12 m/s by
  default): real-time frames advance the clouds' clock, `pathtracer.cloudTime` (seconds), by
  `pathtracer.cloudWindStep` (1/60 s; 0 stills them); accumulation holds it, so the paths' walks and
  an accumulated march see the clouds where it stopped (the CPU reference at time 0). The base density drifts with the wind; the detail half again as fast and rising at
  a quarter of its speed, so the fringes billow and change while the bodies hold their shape (Nubis
  animates its detail the same way). The offsets wrap to their periods (the tile holds 25 detail
  periods; the cirrus, twice as fast aloft, every 1000 km), the DDA's cells move with the base, the
  shadow map is indexed where the base lies, and
  `Clouds.Reconstruct` follows its history back along the wind's step as well as the camera's motion.
  The shear and the curl (§8.10) are baked, so the clouds lean downwind. Against the march accumulated
  at the same clock (`cumulus`, 960 x 540, 160 real-time frames, the sky's rows): the mean absolute
  error of the tone-mapped layer is 0.00085 at 12 m/s, as still (0.00091), and 0.00118 at six times
  that (without the cirrus); its gradients match the reference's. Blending fresh pixels at 0.25 instead of 0.5 lost to it
  in all three (0.00097, 0.00100, 0.00152), and the history read bilinearly lost a little sharpness
  in a strong wind.
- A yawing camera (0.5 degrees a frame) shows no trails. What a moving camera brings in at the edges
  has no history to follow back: `Clouds.Reconstruct` marches those pixels at once (upsampled from
  the sparse march they showed as streaks along the edge for 16 frames); yawing a degree a frame it
  takes 0.31 ms instead of 0.10.
- Not done: the clouds in the real-time volumes' shadow maps (with `volumeMarch` the paths walk
  through the clouds); the sky map's parallax for points far from the camera; the furnace walks
  through the clouds (`furnace=1` checks their energy, §8.10).
- **Validation**: `PathTracer_RealTimeCloudsMarchIntoMaps` (the passes, the shadow map only when the
  scene changes, the layer's history, none while accumulating).

### 8.20 Subsurface scattering

A dielectric material may scatter the light it does not reflect beneath its surface
(`Material::subsurfaceRadius`, per channel, metres; `subsurfaceAnisotropy`): skin, marble, jade, wax,
milk. The material's dielectric BSDF already splits its light that way (§8.3): of the light the
specular lobes do not reflect, `1 - Es(wo)` enters, and its diffuse lobe returns it as if where it
entered, `(1 - Es(wi)) / (pi (1 - EsAvg))` on the way out. Subsurface scattering keeps both factors
and moves the second: the light that enters at x_i is carried beneath the surface to where it leaves,
x_o, and leaves through the exit lobe there (`BsdfMaterial::subsurfaceExit`, white, integrating to 1).
The base colour stays what the material reflects in all; with the radius going to 0 the material is
the plain dielectric again, exactly. (`Subsurface.hlsli`, `SubsurfaceProfile.hlsli`.)

- **The medium**: homogeneous beneath the surface, its single-scattering albedo chosen so that a
  half-space of it reflects the base colour in all: a table of that reflectance by Monte Carlo
  (400000 walks per entry, single-scattering albedos `1 - (1 - i/31)^3`, g = 0, 0.3, 0.6, 0.9),
  inverted into the BSDF LUT as `sqrt(1 - albedo)` per colour (`kLutSubsurface`; textured colours
  look it up per shading point). Its extinction is `1 / (radius s(A) (1 - g))`, Chiang et al. 2016's
  scale `s = 1.9 - A + 3.5 (A - 0.8)^2` with the anisotropy's similarity scaling: the light spreads
  about `radius` whatever g (without it g = 0.9 spread ten times as far, past any profile's reach).
- **Three models** (`pathtracer.subsurfaceModel`, `aircli reference --subsurface-model`):

  | Model | What finds x_o |
  |---|---|
  | 1 random walk (default) | the medium inside the mesh itself: free flights against rays of the scene, Henyey-Greenstein scattering, out where a ray reaches a surface first. Unbiased for any shape: light through thin parts, from a lit back, around corners |
  | 2 Burley | a profile R(r) of the distance from x_i, Burley's normalized diffusion (Christensen and Burley 2015), the exit found by a probe ray through a disc around x_i (on the plane of the normal half of the time, of either tangent a quarter; one of up to 4 hits on the entering mesh, every axis and channel in its pdf) |
  | 0 neural profile | the same probe, R(r) Burley's times `exp(z(log(r/d), A, g))` from a 3-16-16-1 MLP fit to the half-space walk |

- **Colour channels** walk together: one channel, at random, draws every distance of a walk, and each
  channel's light is weighed by the balance heuristic over the three channels' pdfs of the whole
  walk (one-sample MIS, as pbrt-v4 weighs its wavelengths): the weight stays below 3 times the
  product of the channel's albedos. Weighing each segment on its own bounded the segments but not
  their product: in the white furnace 1.1% of the light went missing into the rare enormous weights
  of long walks, even at 4096 frames.
- **In the path tracer** (`TracePath`): at a subsurface vertex the path takes the specular lobes
  (the vertex's BSDF, the base colour black) or goes beneath, by the lobes' energy towards wo (`(1 -
  Es(wo)) A` against `Es(wo)`, clamped to [0.1, 0.9]); beneath, the vertex becomes x_o with the exit
  lobe, its normals outwards. ReSTIR DI and GI see the specular lobes at the first surface; a path
  that went beneath there gathers its own light (NEE at x_o included) and records no GI candidate, and
  the GI candidate's pdf includes the lobe's probability. The denoisers' guides take the base colour
  for the diffuse albedo, and the light from beneath is the diffuse signal. Compiled into the kernels
  with the neural parts (`AIR_PT_SUBSURFACE` with `AIR_PT_NEURAL`, chosen while the scene has
  subsurface materials); elsewhere (volumes, the radiance cache) the material is the plain
  dielectric. Meshes must be closed with their triangles winding outwards (the probe's exits must
  face the way x_i does: counting the back of a thin slab as a half-space created 50% more light there
  in the white furnace).
- **The neural profile** (`SubsurfaceFit.cpp`, `aircli subsurface-fit [--write FILE]`): the half-space
  walk at 22 colours (0.05 - 0.99) and 7 anisotropies (0 - 0.9), 200000 walks each (16 s on 24
  threads), histogrammed in 50 bins of log(r / d) from -6 to 4; the MLP (353 parameters, compiled into
  the shaders as constants, `SubsurfaceProfileNetwork.h`) fit to `log(walk / Burley)` per bin, weighed
  by the bin's share of the light, by Adam (6000 steps, 2.4 s). Its profile follows the walk's with a
  relative RMS of 0.056 against Burley's 0.196 (the light-weighted error per colour and g, averaged),
  the light in all within 1.4% (Burley 1.0%); worst near colour 0.99, whose tails pass the last bin.
  Evaluated with its loops unrolled (immediate weights, activations in registers): looped, the
  network made the trace 2.6 times slower.
- **The probe's budget**: up to 12 rays along the chord, of which only hits on the entering mesh count
  towards its 4: a budget of all hits lost the light where the chord ran through the neighbouring balls
  first, a dark cross on each.

Validation (`tests/SubsurfaceTests.cpp`, scene `subsurface`):

- The half-space walk reflects the colour it was given (6 colours and anisotropies, 40000 walks:
  within 1% plus the walk's noise); the shaders' network equals the CPU's; the compiled network follows
  the walk at least twice as closely as Burley with its energy within 3%.
- White furnace (the furnace scene's spheres, radius 5 / 2 / 1 cm): walk 0.1797, Burley 0.1793, network
  0.1766 (a white furnace's colour 1 lies past the fit's 0.99); on the GPU over `subsurface`, the
  objects -0.2% with the walk (the plain dielectric's own -0.05%), the profiles -2.5% (the balls'
  curvature).
- A thick slab under a uniform sky reflects as the plain dielectric of its colour: all three models
  within 0.3%.
- GPU against the CPU reference (`subsurface`, the walk, 320 x 180): the means agree to 0.0003%, 20 x 20
  blocks within the CPU's noise.
- `subsurface` lit from behind (accumulated, 320 x 180): against the walk, the profiles' median 8 x 8
  block -2.3%, the balls 15-44% dark (the walk carries the light of their lit backs through them; jade
  spreads 8 cm in a ball 44 cm across), the slab 70% dark (the sun through it). Neural and Burley differ
  little there: the profile's shape matters less than what a profile cannot do.
- Cost at 1920 x 1080 in real time (`PathTracer.Trace`): walk 6.4 ms, Burley 5.8, network 6.25; the noise
  of a frame on the balls alike for the three (relative deviation 1.0-1.5). The walk, exact and no
  dearer here, is the default.
- Not modelled: refraction at the boundary (the exit lobe is the dielectric's diffuse one), single
  scattering's glints, a medium that varies inside (textured colours set the medium at x_i).

---

## 9. Editor

`apps/aireditor/` is split in two:

- `air_editor_ui` (all platforms): `EditorApp` plus the diagnostics panels, built on ImGui core.
  `tests/EditorTests.cpp` drives it headless on the Null backend.
- `aireditor` (Windows, D3D12): Win32 window, swap chain and the ImGui DX12 backend. ImGui's
  descriptors come from the bindless heap; the UI is recorded by an RDG pass `Editor.UI` added
  through `FrameRunner::RenderFrame`'s `extraPasses` hook, so it gets automatic barriers like any
  other pass.

Panels: Viewport (fly camera: hold right mouse to look, WASD/QE to move, wheel for speed, Shift
for fast; F12 saves a screenshot), Scene (scene picker, camera,
EV100, materials), Render (VSync, the denoiser: accumulation, NRD with TAA (the default where NRD
runs) or DLSS Ray Reconstruction and its mode; shader hot reload, feature parameters),
Debug (debug views, furnace, NEE), Profiler (GPU pass timeline with one lane per queue, per-thread CPU scopes,
frame/GPU/CPU/samples-per-second history; pause with P or Pause and step frame by frame; Export
HTML writes the profile report of §9.2), Profiler (HTML) (that report, live, inside the editor),
Memory (VRAM budget, process memory, allocator pools and heaps, a filterable resource table) and Log.
Command line (default features `pathtracer,nrd,taa,tonemap`, real time where nrd runs):
`aireditor [--scene NAME] [--features pathtracer,example,tonemap] [--no-vsync] [--dlss]
[--dlss-mode quality|...] [--focus WINDOW]`; `--features` replaces the feature chain, e.g. to try a
new feature or an async compute pass in the editor; `--focus "Profiler (HTML)"` selects that
window's dock tab at startup (handy for screenshots, as nothing else selects a tab without a click).

### 9.1 Shader hot reload

Saving an `.hlsl`/`.hlsli` (or a header it includes) in a running editor recompiles the shaders
that use it and shows the result within about 100 ms:

- CMake writes `<build>/shaders/manifest.json` at configure time: the dxc path and, per output,
  the exact command line of the build and the files it depends on (`cmake/AirShaders.cmake`).
- `renderer::ShaderHotReload` checks those files' times every 250 ms, waits until they have been
  quiet for 150 ms (editors save in steps), and compiles the changed shaders on a worker thread
  into temporary files. Outputs are replaced only if every one compiled; otherwise the previous
  shaders keep running.
- The editor then recreates its FrameRunner (parameters are kept), so every feature reloads its
  pipelines. Compiler errors go to the Log one line each, and a red banner stays on the viewport
  until a compile succeeds. Ctrl+R or Render > Recompile all forces a full recompile.
- `*Shared.h` headers are compiled into C++ too: changing a struct layout still needs a rebuild,
  which the log points out.
- New shader files need a CMake reconfigure (the manifest lists the files that existed then).

`python3 tools/air.py shaders [filter]` compiles every shader from the same manifest into
`<build>/shader-check/`, a quick HLSL check that needs no C++ build and leaves a running editor's
shaders alone. (It does not write `<build>/shaders`: after editing shaders outside the editor, build
the feature's `air_feature_<name>_shaders` target.)

Shaders carry no debug information unless `-DAIR_SHADER_DEBUG_INFO=ON` (source in GPU debuggers):
with it the driver's DXIL verifier prints, at every pipeline creation, the scopes DXC misplaces for
loops in inlined functions, such as the ray queries' candidate loops of alpha cutouts.

### 9.2 Profile report (HTML)

`aircli render ... --html out/profile.html` (the last frame; implies `--profile`) or the editor's
Profiler > Export HTML (also File > Export frame profile; written to
`out/profiles/<scene>-frame<N>.html` under the working directory, then opened in the browser)
writes one self-contained page about one frame, with no network access needed:

- **Overview**: GPU busy time and span, async overlap, CPU recording, passes, submissions,
  barriers, transient heap, VRAM.
- **GPU timeline**: one lane per queue; colour is the kind of work (ray tracing, compute, raster,
  copy), brackets are the submissions, arrows the cross-queue waits (the fork, the join), shaded
  bands where queues overlap. Wheel zooms, drag pans, clicking a pass selects it on every view.
- **GPU pipeline**: every captured command mapped to the stages and units it occupies
  (`renderer::CommandStages`): the command processor (front end, queue waits, barriers), copy,
  the raster stages, compute shader, ray generation / traversal (ray tracing units) / hit and
  miss shaders, AS build. A pass x stage matrix; a selected pass lists its commands and barriers.
- **Submissions & barriers**: every submission with its waits; every barrier with the stages it
  waits for and the stages it unblocks (`renderer::AccessStages`), narrowed to the passes that use
  the resource before and after it.
- **Memory**: the transient heap over the frame (offset against pass: aliased resources share
  rows in disjoint columns); every allocator heap with its resources at their offsets
  (`ResourceInfo::heapId` / `heapOffset`, `GpuMemoryAllocator::HeapUsage::id`), heaps released but
  still waiting for the GPU, memory outside the allocator.
- **Graph resources**. Every chart has a table view; Download JSON saves the data.

How it is built: `FrameRunner::RequestCapture()` makes the next `RenderFrame()` fill
`FrameResult::capture`: `rdg::RenderGraph::DumpJson()`, the command stream, the resource list and
pool statistics after submission, and the device frame index. That frame's GPU timings arrive a
few frames later (`GpuFrameTimings::frameIndex == capture->deviceFrame`; aircli waits for the GPU).
`renderer::MakeProfileReport()` joins them, splitting the command stream per pass
(`SplitCommandsByPass()`: a queue's waits and barriers between two passes count for the next), and
`WriteProfileHtml()` puts the JSON into `src/renderer/report/ProfileReport.html`, which
`cmake/AirEmbed.cmake` compiles into the binary. The page's address keeps the view
(`#pass=3&theme=light`). Tools read the data from the page's `<script id="report-data">` element.
To change the page, edit the HTML and rebuild; the data contract is `ProfileReport.h`.

Inside the editor, the **Profiler (HTML)** panel shows the same page live. `EditorOptions::
createHtmlView` supplies an `HtmlView` (`apps/aireditor/HtmlView.h`); on Windows that is
`WebViewHost.cpp`, a WebView2 control (the runtime is part of Windows 11; the SDK is downloaded,
see `cmake/AirThirdParty.cmake`) in a child window kept over the panel's content rectangle and
hidden while the tab is not shown or an ImGui popup is open (it would cover it). The page loads
once without data; then, while the panel is visible, Live is on and the profiler is not paused, the
editor captures a frame every `htmlProfilerInterval` seconds (1 s) and posts its report with
`PostWebMessageAsJson`; the page's `window.chrome.webview` message listener redraws everything,
keeping the selected pass and a zoomed timeline. Capture takes one now, Open in browser writes
the file of the export. Without a web view (other platforms, tests) the panel offers the browser.

---

## 10. Building

```bash
python3 tools/air.py check                     # configure + build + every check
python3 tools/air.py run -- render --frames 2 --dump-commands
# or plain CMake:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build && ctest --test-dir build
```

| Option | Default | Description |
|---|---|---|
| `AIR_RHI_D3D12` | ON on Windows | D3D12 backend (Agility SDK and DXC are downloaded) |
| `AIR_RHI_VULKAN` | OFF | Vulkan backend (requires the Vulkan SDK) |
| `AIR_BUILD_EDITOR` | ON | Editor UI library and tests everywhere, `aireditor.exe` with D3D12 |
| `AIR_WITH_DLSS` | ON with D3D12 | `dlss` feature; the SDK is downloaded or taken from `AIR_DLSS_SDK_DIR` |
| `AIR_D3D12_PREVIEW` | ON with D3D12 | Preview Agility SDK and DXC for Shader Model 6.10 linear algebra (§3.4); off: the retail SDK and float32 network kernels |
| `AIR_BUILD_TESTS` | ON | Unit tests |
| `AIR_ENABLE_ASAN` / `AIR_ENABLE_TSAN` | OFF | Address / thread sanitizers |

Shaders are compiled by CMake with DXC when a GPU backend is enabled and copied next to the
executables.

---

## 11. Next Steps

1. Vulkan backend (ray tracing pipeline, bindless descriptor buffers).
2. Textures and glTF scene loading; environment map importance sampling.
3. Low-discrepancy sampling (Sobol / blue noise) and adaptive sampling.
4. Offline mode: batched dispatches, tiling for huge resolutions, image sequence output.
5. Shader hot reload in the editor.
