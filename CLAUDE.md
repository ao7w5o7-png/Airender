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
  (docs/architecture.md §9.1). It only checks them (into `build/shader-check`): `aircli` reads
  `build/shaders`, which `cmake --build build --config Release --target air_feature_<name>_shaders`
  updates (a restored file keeps its old time: touch it, or the build keeps the newer binary).
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
- GPU captures (docs/architecture.md §9, WinPixEventRuntime's pix3.h): `aireditor --pix` lets PIX
  attach to the running editor, and F11 writes the next frame's capture to `out/captures/*.wpix`; open
  it in PIX, or read it with `pixtool open-capture X.wpix save-event-list events.csv`. Shader sources
  for PIX's shader debugger are in `build/shaders/pdb/*.pdb` (add it to PIX's PDB search paths;
  `-DAIR_SHADER_OPTIMIZE=OFF` keeps every variable).
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
  `cumulus` for volumetric clouds (`--clouds COVERAGE[,BOTTOM,TOP[,WIND[,HEADING]]]` adds them to
  any scene, `--cirrus COVERAGE[,ALTITUDE[,DEPTH]]` the high cirrus over them, §8.10; in real time
  they are ray marched into maps, `--set pathtracer.cloudMarch=0|1|2`, and drift with the wind,
  `pathtracer.cloudTime` / `cloudWindStep`, §8.19),
  `smoke` for volumes (`--volume FILE.vdb|plume[,EXTINCTION,ALBEDO,G] --volume-at X,Y,Z,SIZE` puts
  an OpenVDB grid in any scene, repeated for several; `aircli vdb FILE` lists a file's grids; §8.12),
  `braziers` for several volumes and lights close to them (§8.13), `textured` for material textures
  and normal maps (§8.15), `coated` for clear coats and their network (§8.17), `subsurface` for
  subsurface scattering (§8.20). File scenes made from
  real assets (§8.18): `picapica`, `sponza` (textures at scale, cutout foliage), `helmets`; fetch them
  first with `python3 tools/air.py fetch-scenes` (into `.deps/scenes`, never committed).
  `--model FILE[,X,Y,Z[,HEIGHT[,YAW]]]` adds any glTF, FBX or OBJ to any scene.
  `--camera EX,EY,EZ,TX,TY,TZ[,FOV]` and `--ev EV100` move a scene's camera.
  Validate GPU against `aircli reference` on the same scene.
- Real-time volumes (`pathtracer.volumeMarch`, on when `accumulate=0`; docs/architecture.md §8.13):
  measure the method against tracked volumes with its CPU twin, `aircli reference --volume-march`
  (the same shader code; `--frames` frames of `--spp` paths), and the GPU against both with
  `--set pathtracer.accumulate=1 --set pathtracer.volumeMarch=2`. Debug views 11 / 12 / 13 show the
  march's steps, adaptive sampling rate and volumes per ray. Adaptive sampling (lighting shared over
  2 x 2 / 4 x 4 pixels) has no CPU twin: compare the GPU's accumulated image at the full output size
  (it shares by the pixels a voxel covers) with `--set pathtracer.volumeAdaptive=0`, per 8 x 8 block.
  Time with `--set pathtracer.spp=1` and enough frames that the GPU's clocks have risen.
- Render below the output size with `--render-scale S` (dlss upscales; without it FrameRunner
  upscales bilinearly). Stages: Scene (render size, writes `sceneColor`) -> Upscale -> Post
  (output size); see docs/architecture.md §8.5.
- Neural texture compression (LibNTC, docs/architecture.md §8.16): `--neural-textures load|sample[,BPP]`
  on `render` / `reference` (the `textured` scene); decoding on sample runs in the path tracer's
  `AIR_PT_NEURAL` kernels. After touching the decode path run `aircli ntc-check --shaders build/shaders`
  (the shader's decode against LibNTC's CUDA decode, every texel), and compare an accumulated
  `sample` image with a `load` one (they agree within about 0.1%).
- Subsurface scattering (docs/architecture.md §8.20, scene `subsurface`): `--set
  pathtracer.subsurfaceModel=1|0|2` (random walk inside the mesh, the default and reference; neural
  profile; Burley's profile). Check energy with `--set pathtracer.furnace=1` (white, must stay 0.18)
  and the GPU against `aircli reference --subsurface-model N`. The neural profile's network is fit by
  `aircli subsurface-fit --write src/features/pathtracer/shaders/SubsurfaceProfileNetwork.h` (it prints
  its error against the half-space walk); refit after changing the medium or the walk.
- Neural shading, clear coats (docs/architecture.md §8.17, scene `coated`): `--set pathtracer.coatModel=0|1|2`
  (network, random walk reference, approximation). After changing the coat or its network run
  `aircli coat-check` and compare an accumulated `coatModel=0` image with a `coatModel=1` one; raise the
  version in `TrainNeuralCoat`'s cache key so stale networks are not reused.
- Probe GI (docs/architecture.md §8.21): indirect light without ray tracing (voxel cascades, screen
  probes, a world-space radiance cache): `--features pathtracer,probe_gi,taa,tonemap --set
  pathtracer.accumulate=0` (nrd before taa denoises the direct light). Compare `--set probe_gi.view=1`
  (its GI alone) with the path tracer's reference of it, `--set pathtracer.debugView=15` accumulated,
  over a few hundred frames (the voxels' bounces converge over frames), and by lobe `view=2` / `3`
  (diffuse irradiance / pi, specular light) with `pathtracer.debugView=16` / `17`; `view=4..7` show the voxels
  (albedo, light, steps, cascades), `multiBounce=0` against `maxBounces=2` checks one bounce, and the
  report's `voxelization` status shows each cascade's voxels and what overflowed. Move the camera
  (`--camera-motion`) to check the cascades' scrolling. Compare the mean of the last 100 of 400 frames
  (`--output-frames 100`), not one frame: the voxels' light drifts slowly (many_lights about 1%).
  `probe_gi.reflections=0` / `rayFilterRadius=0` / `filterRadius=1` switch the traced reflections and the
  probes' per-direction filter for comparison; the specular light of smooth pixels alone: mask with
  `pathtracer.debugView=3` (roughness).
- Real-time denoising (docs/architecture.md §8.14): two alternatives, never combined:
  `pathtracer,dlss,tonemap` (DLSS Ray Reconstruction denoises, anti-aliases and upscales; the editor's
  default, `--no-dlss` or a GPU without it falls back) and `pathtracer,nrd,taa,tonemap` (NRD on the
  surfaces' indirect light split by lobe, TAA for edges and the direct light; nrd needs D3D12). Indirect light added at the
  first surface must also add its lobe split to `kDiffuseSignal` / `kSpecularSignal` (as ReSTIR GI
  does), or NRD leaves it noisy. Direct light: only the diffuse part of the emissive triangles' goes to
  them (`pathtracer.denoiseDirect`, on); the sun's and specular direct light stay out (NRD washed out
  sharp shadows and highlights). The editor's ReSTIR DI / GI checkboxes switch them for comparison. Energy is part of correctness: check the mean of 32 real-time frames against an
  accumulated reference, in real lighting and in the white furnace (`--set pathtracer.furnace=1`;
  docs/architecture.md §8.14 "Energy"); a filter that clips or clamps noisy samples loses light.
- Real time is judged over frames, and a static camera hides trails: `--camera-motion
  RIGHT,UP,FORWARD[,YAW[,PITCH[,FRAMES]]]` moves the camera every frame (only the first FRAMES, then
  holds it: what a move leaves behind) and `--output-frames N` writes the last N frames. Compare a moving
  camera's last frame with a static camera's at the same place (the report's `finalCamera` is that place
  in `--camera` syntax; a static camera elsewhere is no baseline, even a few centimetres away) (trails,
  e.g. a denoiser reprojecting with the wrong motion), and a static camera's consecutive frames
  (flicker). Anything laid over the surfaces must keep the denoiser's guides consistent with it
  (docs/architecture.md §8.13, "Under a denoiser").
- Small GPU allocations (docs/architecture.md §4.2): with the driver's tight alignment a buffer takes
  its own size, else small buffers are packed into shared ones; `--no-tight-alignment` forces that
  fallback. A buffer written once by `renderer::UploadQueue` and then only read should carry
  `BufferUsage::Immutable` so the fallback can pack it.
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
- CPU data of one frame goes through the frame's upload memory (docs/architecture.md §4.6):
  `graph.AllocateUpload()` for data shaders read in place (`ByteAddressBuffer` at the allocation's
  offset), `graph.UploadBuffer` / `UploadTexture` / `UploadToBuffer` for a Copy pass the graph orders.
  Do not keep host-visible rings of your own. Loading and rebuilding stream through
  `renderer::UploadQueue` (copy queue; `MakeAvailable` hands the resources to a queue).
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
