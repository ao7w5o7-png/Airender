# Airender: conventions for AI agents

Talk to the user in Chinese. Everything written into the repository (code comments, docs, commit
messages, PR descriptions) is in English. Full standards: docs/coding-standards.md; architecture:
docs/architecture.md. Do not name or reference other game engines anywhere in the repository.

## Iteration loop (run this first)
- Gate before pushing: `python3 tools/air.py check` (format + layering + build + all tests). Add
  `--full` (clang-tidy + TSAN) after touching concurrent code.
- Run a subset of tests: `python3 tools/air.py test <name substring>`.
- After editing HLSL: `python3 tools/air.py shaders` compiles every shader with the build's exact
  dxc command lines (no C++ build needed). A running editor hot-reloads saved shaders by itself
  (docs/architecture.md §9.1).
- Run the renderer: `python3 tools/air.py run -- render --features pathtracer,example --frames 2 --report out/report.json`
  - The report is JSON: per-frame stats, feature parameters and status, `errors`, `passed`. Exit
    code 0 = passed, 1 = validation errors or leaks, 2 = usage or environment error.
  - `--dump-commands` prints the full command stream; `--dump-graph x.dot` writes the render
    graph; `aircli features` lists every feature and its parameters.
- Profile one frame as an HTML page: `aircli render --backend d3d12 ... --html out/profile.html`
  (GPU timeline per queue with async overlap and waits, the pipeline stages of every command,
  barriers, transient aliasing and heap offsets; docs/architecture.md §9.2). The data is the JSON
  in its `<script id="report-data">` element. The editor shows the same page live in its
  Profiler (HTML) panel; `aireditor --focus "Profiler (HTML)"` opens with that tab selected.
- Image comparison: `aircli compare ref.pfm test.pfm --diff diff.ppm` prints rmse / psnr /
  differing pixels as JSON.
- A failing golden snapshot (`tests/golden/`) means the command stream or render graph changed:
  once the change is confirmed as intended, run `python3 tools/air.py golden-update` and say so in
  the PR.
- Rendering correctness: `aircli reference --scene cornell_box --output ref.pfm` renders on the CPU
  with the same `.hlsli` integrator; compare a GPU `render --output gpu.pfm` against it. BSDF and
  integrator changes must keep `BsdfTests` / `IntegratorTests` (white furnace, energy) green.
- Sample reuse (ReSTIR DI / GI, on by default when `accumulate=0`; docs/architecture.md §8.7, §8.8)
  must stay unbiased: force it on while accumulating (`--set pathtracer.restirDi=2 --set
  pathtracer.restirGi=2`, thousands of frames) and compare the mean against an NEE reference of the
  same scene; `--set pathtracer.debugView=8` / `9` isolates the direct / indirect light.
- Neural radiance cache (`--set pathtracer.nrc=1`, real time; `2` also when accumulating;
  docs/architecture.md §8.11): biased by design, so measure its bias and noise against a reference
  rendered with the same jitter (`accumulate=0`, same `--frames`, plain paths at high `spp`), and
  look at the cache itself with `--set pathtracer.debugView=10`. After changing the network code
  run `aircli mlp-check --backend d3d12 --shaders build/shaders`: the GPU's float32 and matrix-unit
  (Shader Model 6.10 linear algebra, docs/architecture.md §3.4) kernels against the CPU's.
  `--set pathtracer.nrcLinearAlgebra=0` runs the float32 kernels for an A/B comparison.
- Test scenes (`aircli scenes` lists them with sizes; docs/architecture.md §8.1): pick the one that
  stresses the change, e.g. `veach_mis` for MIS / light sampling, `many_lights` for light
  selection (light tree: `--set pathtracer.lightTreeBounces=N`, docs/architecture.md §8.6), `door`
  for indirect transport, `sphere_field` / `terrain` for instancing, dense geometry and memory,
  `sunset` for the physical sky (`--sun ELEVATION,AZIMUTH` gives any scene that sky, §8.9),
  `cumulus` for volumetric clouds (`--clouds COVERAGE[,BOTTOM,TOP]` adds them to any scene, §8.10),
  `smoke` for volumes (`--volume FILE.vdb|plume[,EXTINCTION,ALBEDO,G] --volume-at X,Y,Z,SIZE` puts
  an OpenVDB grid in any scene; `aircli vdb FILE` lists a file's grids; §8.12).
  `--camera EX,EY,EZ,TX,TY,TZ[,FOV]` and `--ev EV100` move a scene's camera.
  Validate GPU against `aircli reference` on the same scene.
- Render below the output size with `--render-scale S` (dlss upscales; without it FrameRunner
  upscales bilinearly). Stages: Scene (render size, writes `sceneColor`) -> Upscale -> Post
  (output size); see docs/architecture.md §8.5.
- Before trusting GPU timings, check the report's `vramBudgetBytes`: other processes holding VRAM
  (graphics debuggers, IDEs) make WDDM page resources to system memory and inflate every pass.
- Debug a frame with `--set pathtracer.debugView=N` (normals, albedo, depth, heatmaps, direct /
  indirect) or `--set pathtracer.furnace=1`; see docs/architecture.md §8.4.
- All of the above works without a GPU (Null backend), except GPU renders.

## New algorithm = new feature
- `python3 tools/air.py new-feature <lower_snake_name>` creates a skeleton from
  `src/features/example`; CMake discovers it automatically.
- Features depend only on core / rhi / rdg / renderer and never include another feature
  (`air.py layers` enforces this).
- Expose tunables with `mParameters.Add(...)`; set them at runtime with `--set <feature>.<param>=<value>`.

## Writing passes
- Use `rdg::RenderGraph::AddPass`; declare every resource read/write and its `Access` in setup.
  Never write barriers by hand.
- Shader resources are always bindless: get indices with `ctx.SRV()/ctx.UAV()` in execute and
  pass them through push constants (≤ 128 bytes).
- Structs shared between C++ and HLSL live in the feature's `shaders/XxxShared.h` with
  `static_assert` size checks.
- Execute lambdas may run concurrently on worker threads: capture by value or read only immutable data.
- Every new pass gets at least one Null backend test (`Headless` in `tests/TestHelpers.h`)
  asserting there are no validation errors.
- A compute pass that depends on little and is needed late can run on the compute queue: add
  `PassFlags::AsyncCompute`. The graph inserts the cross-queue waits and hand-over barriers; the
  pass may only use compute-queue states (no `SRVGraphics`, render targets, depth). Check the
  schedule in `DumpText()` (`Submissions:`), and compare GPU timings with `--no-async-compute`
  (docs/architecture.md §5.4).

## Features talk through the blackboard
- Publish and consume per-frame textures and values via `ctx.blackboard` with the keys in
  `renderer::slots` (guides, jitter, history reset); add new slots there rather than coupling features.
- The editor (`apps/aireditor`) is a UI over `FrameRunner`: its logic lives in `air_editor_ui`
  and is tested headless in `tests/EditorTests.cpp`.

## Thread safety
Device is thread-safe; CommandList is not; render graph construction is single threaded; prefer
serialising shared mutable state through a `Pipe` over adding locks.

## Style
- C++20, as defined by `.clang-format` / `.clang-tidy`: types and functions `PascalCase`, private
  members `mName`, constants `kName`.
- Braces of multi-line blocks on their own line (Allman), in C++ and HLSL alike; one-line inline
  functions and lambdas stay on one line. Run `python3 tools/air.py format` after editing: it runs
  clang-format 18 on C++ and fixes HLSL brace placement (clang-format cannot parse HLSL).
- Invariants use `AIR_CHECK` (always on, messages name the pass/resource); debug-only checks use `AIR_DCHECK`.
