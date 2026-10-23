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
3. **Access states + barriers**: resource state is the `rhi::Access` bit set and a barrier is `before -> after`. Backends translate them to D3D12 Enhanced
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

Next: cache transient heaps across frames instead of
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

### 5.3 Explicit dependency declaration

Some render graphs reflect resource dependencies from shader parameter structs. Here passes
declare them explicitly through `PassBuilder`: no macros or reflection, shorter
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

Image tests: `Image` reads and writes PFM (linear HDR, exact) and PPM (8-bit preview).
`render --output` reads the final frame back; `aircli compare` measures rmse / psnr against a
reference (see §8.3).

---

## 8. GPU Path Tracer (Physically Based, Energy Conserving)

Files: `src/features/pathtracer/` (shaders in `shaders/`, tests in `tests/`). Registered as the
`pathtracer` feature with parameters `spp`, `maxBounces`, `targetSamples`, `accumulate`, `guides`,
`nee`, `furnace`, `debugView` and `debugScale`.

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
  | `sphere_field` | ~2400 instances of 2 meshes, 32 materials, sun (1.6e9 nits) + sky | TLAS scale, dynamic range, light imbalance |
  | `many_lights` | hall with 66 panels + 300 LEDs, radiance over 2 orders of magnitude | light selection (baseline for many-light sampling) |
  | `door` | light reaches the camera's room only through a door left ajar | indirect-dominated transport |
  | `terrain` | 2.1M triangle height field, lake, sun + sky | dense geometry, BLAS build and memory |

  The sun is a disc 10 km away with the real angular radius (0.2665 degrees) and luminance, so
  daylight scenes use daylight exposure (EV100 14..15).

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
  probability is exactly `probability`, as MIS assumes. Russian roulette from the third bounce.
- **Hardware ray tracing**: DXR raygen + closest hit for primary and bounce rays (20 byte payload
  with access qualifiers, bounce loop in raygen, recursion depth 1); inline `RayQuery` for shadow
  rays and the intersection heatmap. Push constants only carry bindless indices; per-frame data comes
  from rings in the GPU upload heap (`DeviceLocalHost`) where available.
- **Kernel variants**: `PathTracer.hlsl` (SM 6.6) and `PathTracerSer.hlsl` (SM 6.9, DXR 1.2 shader
  execution reordering: bounce rays trace into hit objects, threads regroup by hit instance, the
  hit is read from the object without a closest-hit call) share `PathTracerKernel.hlsli`. SER is
  chosen with `pathtracer.ser=1` on GPUs that reorder (`DeviceCaps::shaderExecutionReordering`).
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
7 path length heatmap, 8 direct light only, 9 indirect light only. `debugScale` scales the distance and heatmap views. `furnace=1` turns every
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
Debug (debug views, furnace, NEE), Profiler (GPU pass timeline, per-thread CPU scopes,
frame/GPU/CPU/samples-per-second history; pause with P or Pause and step frame by frame), Memory
(VRAM budget, process memory, allocator pools and heaps, a filterable resource table) and Log.
Command line: `aireditor [--scene NAME] [--no-vsync] [--dlss] [--dlss-mode quality|...]`.

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
