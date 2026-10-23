# Airender

A graphics renderer that AI can write code for, drive and debug directly. It targets both
real-time rendering and film-quality offline rendering through GPU path tracing, and doubles as a
testbed for graphics algorithms. C++20, with an RHI over D3D12 and Vulkan.

- Architecture: [docs/architecture.md](docs/architecture.md)
- Coding standards: [docs/coding-standards.md](docs/coding-standards.md)
- Conventions for AI agents: [CLAUDE.md](CLAUDE.md)

## Modules

| Directory | Contents |
|---|---|
| `src/core` | Task system (task DAG / Pipe / ParallelFor), handle pool, logging, JSON, image IO and comparison |
| `src/rhi` | RHI interface, GPU memory allocator (TLSF + pools + dedicated heaps), Null and D3D12 (DXR) backends, Vulkan backend (in progress) |
| `src/rdg` | Render graph: culling, automatic barriers, transient memory aliasing, parallel recording |
| `src/renderer` | Extension point: RenderFeature interface, feature registry, parameters, FrameRunner |
| `src/renderer` (scene) | Scene description, GPU scene upload (BLAS/TLAS, materials, emissive triangles), built-in scenes |
| `src/features/*` | Rendering feature plugins, each with its own shaders and tests: `pathtracer` (DXR, energy-conserving GGX, NEE + MIS), `dlss` (Ray Reconstruction), `tonemap`, `example` (template) |
| `apps/aireditor` | ImGui editor: viewport, scene and render settings, debug views, GPU/CPU profiler, memory panels |
| `apps/aircli` | Headless command line: `render` / `features` / `reference` / `compare`, JSON reports |
| `tools/air.py` | Developer harness: check / test / run / golden-update / new-feature |
| `tests/` | Unit tests and golden snapshots (no external dependencies) |

## Quick start

```bash
python3 tools/air.py check                         # format + layering + build + all tests
python3 tools/air.py run -- render --features pathtracer,example --frames 2 --report out/report.json
python3 tools/air.py new-feature my_effect         # skeleton for a new algorithm
```

On Windows with a DXR GPU, run the editor (`build/apps/aireditor/aireditor.exe`) or render from
the command line and validate against the CPU reference:

```bash
aircli render --backend d3d12 --scene cornell_box --frames 300 --output gpu.pfm
aircli reference --scene cornell_box --spp 512 --output ref.pfm
aircli compare ref.pfm gpu.pfm --diff diff.ppm
```

Requires CMake ≥ 3.24 and a C++20 compiler (GCC 13 / Clang 18 / MSVC 2022); Ninja and
clang-format/clang-tidy 18 are optional. On Windows the D3D12 backend is built by default; enable
the Vulkan backend with `-DAIR_RHI_VULKAN=ON`.
