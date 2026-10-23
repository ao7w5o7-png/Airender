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
| `src/rhi` | RHI interface, GPU memory allocator (TLSF + pools + dedicated heaps), Null backend, D3D12/Vulkan backends (in progress) |
| `src/rdg` | Render graph: culling, automatic barriers, transient memory aliasing, parallel recording |
| `src/renderer` | Extension point: RenderFeature interface, feature registry, parameters, FrameRunner |
| `src/features/*` | Rendering feature plugins, each with its own shaders and tests: `pathtracer`, `example` (template) |
| `apps/aircli` | Headless command line: `render` / `features` / `compare`, JSON reports |
| `tools/air.py` | Developer harness: check / test / run / golden-update / new-feature |
| `tests/` | Unit tests and golden snapshots (no external dependencies) |

## Quick start

```bash
python3 tools/air.py check                         # format + layering + build + all tests
python3 tools/air.py run -- render --features pathtracer,example --frames 2 --report out/report.json
python3 tools/air.py new-feature my_effect         # skeleton for a new algorithm
```

Requires CMake ≥ 3.24 and a C++20 compiler (GCC 13 / Clang 18 / MSVC 2022); Ninja and
clang-format/clang-tidy 18 are optional. On Windows the D3D12 backend is built by default; enable
the Vulkan backend with `-DAIR_RHI_VULKAN=ON`.
