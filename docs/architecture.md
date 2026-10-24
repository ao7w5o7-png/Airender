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
3. **Waits**: any use of a resource last used on another queue waits for the batch of that use.
   This covers read after write, write after read and state changes; two reads in a row also
   wait (conservative). Waiting for a later batch of a queue covers its earlier ones, so a batch
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
reference (see §8.3).

---

## 8. GPU Path Tracer (Physically Based, Energy Conserving)

Files: `src/features/pathtracer/` (shaders in `shaders/`, tests in `tests/`). Registered as the
`pathtracer` feature with parameters `spp`, `maxBounces`, `targetSamples`, `accumulate`, `guides`,
`nee`, `lightTreeBounces`, `cloudEvents` (§8.10), `volumeEvents` (§8.12), `restir*` (§8.7), `nrc*`
(§8.11), `furnace`, `debugView`, `ser` and `debugScale`.

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

  Daylight scenes use the physical sky (§8.9), whose sun has the real angular radius and
  luminance, so they use daylight exposure (EV100 14..15; 11.5 for the low sun of `sunset`).

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
- **Light transport**: next event estimation over emissive triangles chosen by power, combined
  with BSDF sampling by the power heuristic (MIS). Emitters are one sided (front = `cross(e1, e2)`).
  The light list is also a Walker/Vose alias table, so choosing a light is O(1) (one or two loads)
  whatever the light count; the alias decision uses its own random number, so the selection
  probability is exactly `probability`, as MIS assumes. A light tree can choose instead (§8.6).
  Russian roulette from the third bounce.
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
  scene's volume (§8.12) and runs instead of the plain kernel while there is one.
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
  weight equals evaluation, sampling matches uniform integration.
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
7 path length heatmap, 8 direct light only, 9 indirect light only, 10 what the neural radiance
cache predicts the first surface reflects (§8.11). `debugScale` scales the distance and heatmap views. `furnace=1` turns every
material white and every emitter off with a uniform environment, so any visible structure is an
energy error.

### 8.5 Feature chain and the blackboard

Features run in three stages (`RenderFeature::GetStage()`), and within a stage in the order they
were added:

| Stage | Resolution | Writes | Features |
|---|---|---|---|
| `Scene` | render (`FrameContext::renderWidth/Height`) | `FrameContext::sceneColor` | pathtracer |
| `Upscale` | render -> output | `FrameContext::output`, sets `outputWritten` | dlss |
| `Post` | output (`FrameContext::width/height`) | `FrameContext::output` in place | tonemap, example |

The render resolution is the output resolution times `FrameRunnerDesc::renderScale` (`aircli
render --render-scale S`). At scale 1 `sceneColor` *is* the output texture and nothing extra runs.
Below 1 `sceneColor` is a transient texture (`FrameRunner.SceneColor`), and if no Upscale feature
wrote the output (dlss missing, disabled, failed, or a debug view) FrameRunner adds
`FrameRunner.Upscale` (bilinear, `src/renderer/shaders/Upscale.hlsl`) before the Post stage, so
the output is always defined.

Features communicate only through `renderer::Blackboard` (`FrameContext::blackboard`), never by
including each other. Slots live in `renderer::slots`:

| Slot | Producer | Contents |
|---|---|---|
| `kDiffuseAlbedo`, `kSpecularAlbedo` | pathtracer (`guides=1`) | First-hit albedos |
| `kNormalRoughness` | pathtracer | World normal + roughness |
| `kViewDepth` | pathtracer | Distance along the view axis in metres |
| `kMotionVectors` | pathtracer | Previous minus current position, in pixels |
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
  size it passes the frame through. It refuses to run on other backends.
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
  sample), and writes the first sample's hit (`PrimaryHit`: instance, primitive, barycentrics, plus a
  normal, roughness and distance for the choice of neighbours). The passes rebuild the surface from it with the
  integrator's own `LoadSurface`, so they shade exactly what the path tracer would have, and add
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
  (six per pixel), cheaper in total than storing a full-precision G-buffer twice.
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
single-scattering albedo, a shape scale and a seed. `cumulus` uses it; `aircli render|reference
--clouds COVERAGE[,BOTTOM,TOP]` adds it to any scene (0 removes it; it switches on the physical sky,
whose sun lights the clouds), and the editor's Scene panel edits it.

- **Density** (`src/renderer/src/Clouds.cpp`), procedural, baked on the CPU into two 8-bit 3D
  textures when the clouds change (0.2 s in debug builds, all cores):
  - the base, 256 x 256 texels over a 12.8 km tile that repeats horizontally and 25 m layers in
    height. A 2D weather field (gradient noise for regional variation, cellular noise that gathers
    clouds into cells) is thresholded at its 1 - coverage quantile, so the requested fraction of the
    tile lies under cloud outlines; how far it exceeds the threshold sets a cloud's height and
    density. Bases are flat at the condensation level, tops are domes, billowy cellular noise
    carves the outline and liquid water grows with height above the base.
  - the detail, 64^3 texels of cellular noise repeating every 400 m, which erodes the base's fringe
    at render time; the eroded density is then sharpened (times 10, clamped to 1): crisp outlines,
    even interiors.
  - majorants, one per 8 x 4 x 8 base texels plus the texels around them that trilinear filtering
    reaches. They bound the density everywhere because the erosion only lowers it and the
    sharpening is monotonic.
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
  the tile; ratio and delta tracking both match a 0.5 m ray march of exp(-optical depth) for rays of
  optical depth 0.1-3; disabled clouds are transparent.
- `Integrator_CloudsVanishInAWhiteFurnace`: with albedo 1 in a uniform environment every path
  through the clouds returns the environment exactly, so every pixel is 0.18.
- `Integrator_CloudSunNeeAndPhaseSamplingAgree`: with a 12-degree sun, NEE + MIS and phase sampling
  alone agree within 0.5% at 512 spp, with and without delta-M scaling.
- `Scene_CloudsStayAboveTheGeometry`, `Scene_CloudResourcesFollowEdits`.
- GPU against the CPU reference on `cumulus` (160x90, `--sun 38,-20`, 8192 against 2048 spp): the
  means differ by 0.014%; sky, horizon and ground separately by at most 0.2%.
- Cost (RTX 5070, 960x540, 4 spp per frame, front-lit `cumulus`): 78 ms per frame. Multiple
  scattering in thick media is the most expensive transport here, and noisy: a cloud pixel's
  samples have a relative standard deviation of about 0.8, so clouds need hundreds of samples (or
  a denoiser).
- The air between a ray's origin and its first collision with the clouds dims and adds to their
  light (§8.9's aerial perspective), so distant clouds fade into the haze.
- Not modelled: the planet's curvature (the layer is flat and ends 100 km along a ray), wind, the
  scene's own geometry lighting the clouds, the clouds' shadows in the air between them and the
  viewer.

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
  the box's 8 bounces of plain path tracing miss by 14%.
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
panel lists, adds, edits and removes them.

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
EV100, materials), Render (VSync, DLSS toggle and mode, shader hot reload, feature parameters),
Debug (debug views, furnace, NEE), Profiler (GPU pass timeline with one lane per queue, per-thread CPU scopes,
frame/GPU/CPU/samples-per-second history; pause with P or Pause and step frame by frame; Export
HTML writes the profile report of §9.2), Profiler (HTML) (that report, live, inside the editor),
Memory (VRAM budget, process memory, allocator pools and heaps, a filterable resource table) and Log.
Command line: `aireditor [--scene NAME] [--features pathtracer,example,tonemap] [--no-vsync] [--dlss]
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
shaders alone.

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
