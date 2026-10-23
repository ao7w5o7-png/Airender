# Airender Architecture

> Goal: a renderer that AI can write code for, drive and debug directly. It supports both
> real-time rendering and film-quality offline rendering, should eventually produce cinematic
> video from AI + renderer, and doubles as a testbed for quickly validating graphics algorithms.

This document describes the architecture of the first skeleton. The design borrows heavily from
Unreal Engine (RHI / RDG / Tasks) but is simplified, prioritising **thread safety, debuggability
and AI friendliness**.

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

| Module | Directory | Unreal Engine counterpart |
|---|---|---|
| Task system | `src/core` | `UE::Tasks` (`Launch`, prerequisites, `FPipe`, `ParallelFor`) |
| RHI | `src/rhi` | `FDynamicRHI` / `FRHICommandList` / `ERHIAccess` |
| GPU memory | `src/rhi/GpuMemoryAllocator` | `FD3D12PoolAllocator` (D3D12), `FVulkanResourceHeap` (Vulkan) |
| Render graph | `src/rdg` | `FRDGBuilder`, transient allocator |
| Feature framework | `src/renderer` | `FSceneViewExtension` / renderer module plugins |
| Path tracer | `src/features/pathtracer` | UE Path Tracer (`PathTracing.cpp`) |

---

## 2. Task System (Multithreading)

File: `src/core/include/air/core/TaskSystem.h`

### 2.1 Model

- **Task**: `TaskSystem::Launch(name, fn, prerequisites, priority)` returns a `TaskHandle`. A task
  is scheduled once all its prerequisites have completed; tasks form a DAG.
- **Waiting helps**: while waiting, `TaskHandle::Wait()` executes other queued tasks (a simplified
  form of UE's task retraction). Waiting for a subtask on a worker thread therefore cannot stall
  the pool (see `TaskSystem_NestedWaitOnSingleWorkerDoesNotDeadlock`).
- **Pipe**: tasks launched through the same `Pipe` run **one at a time** in launch order, on any
  thread. This is the preferred way to get thread safety: hand non-thread-safe state to a pipe and
  submit every access as a pipe task, and no lock is needed (the same reason UE replaced named
  threads with `FPipe`).
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
  building → RHI submission (UE's GameThread / RenderThread / RHIThread).

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
3. **Access states + barriers**: resource state is the `rhi::Access` bit set (similar to UE's
   `ERHIAccess`) and a barrier is `before -> after`. Backends translate them to D3D12 Enhanced
   Barriers / Vulkan synchronization2. The render graph generates barriers; pass code never
   writes them by hand.
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
| D3D12 | 🚧 stub | Plan: DXGI adapter selection → `ID3D12Device5+`, DXR 1.1, Enhanced Barriers, GPU-based validation |
| Vulkan | 🚧 stub | Plan: Vulkan 1.3 (dynamic rendering, sync2, timeline semaphores), KHR ray tracing pipeline + ray query |

### 3.3 Queues

There are `Graphics`, `Compute` and `Copy` queues. The first render graph version submits
everything to Graphics. Async compute (BVH builds, denoising in parallel with raster) comes later;
the render graph already has the information it needs to split work across queues.

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

### 4.3 Transient memory in the render graph (aliasing)

Temporary resources created inside the render graph do not use the pools above. Instead:

1. at compile time each resource gets a lifetime `[first pass, last pass]`;
2. resources are packed first-fit in descending size order: resources whose lifetimes do not
   overlap may share the same memory;
3. the frame allocates a single placed heap and calls `CreatePlacedTexture/Buffer` inside it.

`RenderGraph::GetStats()` reports `transientHeapSize` (with aliasing) and
`transientBytesUnaliased` (without). The first barrier of a resource carries a `discard` flag so
the backend can issue an aliasing barrier / drop old contents.

Next: cache transient heaps across frames (UE's `FRHITransientResourceAllocator`) instead of
creating one per frame.

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
| Cross-frame resources | `ImportTexture` (external resources) / `ExtractTexture` (keep a transient resource for the next frame, e.g. TAA history, path tracing accumulation) |

### 5.2 Debugging and error messages (for AI)

- Checked at declaration time: a read declared with a writable state, a write declared with a
  read-only state, a missing usage flag (e.g. a UAV write on a texture created without
  `UnorderedAccess`), one pass using one resource in two different states. Every message names
  the pass and the resource.
- Checked at execution time: using a resource that was not declared in setup is an error.
- `DumpText()`: per pass state, barriers, reads and writes; per resource lifetime and heap offset.
- `DumpGraphviz()`: a dot graph (culled passes grey, imported resources blue, extracted resources green).

### 5.3 Differences from UE's RDG

UE uses shader parameter structs so that RDG can reflect resource dependencies. Here passes
declare them explicitly through `PassBuilder` (Frostbite style): no macros or reflection, shorter
code, and easier for AI to get right. Code generation on top of the shared HLSL/C++ headers can
later keep push constant structs and declarations in sync automatically.

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

Image golden tests: `Image` reads and writes PFM (linear HDR, exact) and PPM (8-bit preview).
Once a real backend supports readback, `render --screenshot` will write PFM to compare against
`tests/golden/images/`; the comparison tool and metrics are already in place.

---

## 8. GPU Path Tracer (Fast Offline Rendering)

Files: `src/features/pathtracer/` (shaders in `shaders/`, tests in `tests/`). Registered as the
`pathtracer` feature with parameters `spp`, `maxBounces`, `targetSamples` and `exposure`.

### 8.1 Current implementation

- Hardware ray tracing: DXR 1.1 / `VK_KHR_ray_tracing_pipeline`, one raygen + miss + closest-hit.
- **The bounce loop lives in raygen** with `maxRecursionDepth = 1`: a small stack and scheduler
  friendly, as in UE's path tracer and most GPU offline renderers.
- Progressive accumulation: a persistent `RGBA32F` accumulation buffer (rgb = radiance sum,
  a = sample count) gains `samplesPerPixelPerFrame` samples per frame; the `Resolve` pass averages,
  applies exposure and an ACES tonemap.
- Camera or scene changes call `ResetAccumulation()`; once `IsConverged()` reaches
  `targetSamples` the frame is final.
- Materials: Lambert + emission only for now, with a gradient sky; Russian roulette from the
  third bounce.
- Shaders and C++ share `PathTracerShared.h`, so push constant / struct layouts are defined once
  and their sizes are checked with `static_assert`.

### 8.2 Roadmap for fast offline rendering

| Priority | Item | Benefit |
|---|---|---|
| P0 | Next event estimation (NEE) + MIS, area light / environment importance sampling | Much less noise at the same spp |
| P0 | GGX metal/roughness materials, bindless textures | Presentable images |
| P0 | Offline mode: high spp per frame, batched DispatchRays, no vsync, saturate the GPU | Throughput |
| P1 | AI denoising (OIDN / OptiX with albedo + normal guides) | Final frames at low spp |
| P1 | Blue noise / Sobol low-discrepancy sequences instead of PCG | Faster convergence |
| P1 | Adaptive sampling: stop per pixel by variance | Spend samples where the noise is |
| P2 | Shader Execution Reordering (SER, RTX 40+) / material sorting | SIMD efficiency for divergent paths |
| P2 | Tiled rendering to avoid TDRs and support huge resolutions | Stability |
| P2 | Motion blur, depth of field, volumes | Cinematic effects |

### 8.3 Producing cinematic video

One offline frame = run frames until converged → read back → write an EXR/PNG sequence →
assemble with ffmpeg. Camera paths and animation come from a higher-level script (which AI can
generate later).

---

## 9. Building

```bash
python3 tools/air.py check                     # configure + build + every check
python3 tools/air.py run -- render --frames 2 --dump-commands
# or plain CMake:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build build && ctest --test-dir build
```

| Option | Default | Description |
|---|---|---|
| `AIR_RHI_D3D12` | ON on Windows | D3D12 backend |
| `AIR_RHI_VULKAN` | OFF | Vulkan backend (requires the Vulkan SDK) |
| `AIR_BUILD_TESTS` | ON | Unit tests |
| `AIR_ENABLE_ASAN` / `AIR_ENABLE_TSAN` | OFF | Address / thread sanitizers |

Shaders: with DXC installed, each feature's `shaders/` can be compiled (`-T lib_6_6` for DXIL,
add `-spirv` for SPIR-V). Build integration lands together with the D3D12 backend.

---

## 10. Next Steps

1. D3D12 backend: device, queues/fences, heap provider, bindless descriptor heap, Enhanced
   Barriers, DXR pipelines and shader tables.
2. Shader compilation integration (DXC, compiled by CMake + runtime hot reload).
3. Scene system: glTF loading → vertex/index buffers → BLAS/TLAS → instance/material buffers.
4. Path tracer P0 items (NEE + MIS, GGX, offline mode).
5. Readback + `render --screenshot` (PFM/EXR) + image golden tests + image sequence rendering.
6. Vulkan backend.
