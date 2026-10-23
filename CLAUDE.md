# Airender: conventions for AI agents

Talk to the user in Chinese. Everything written into the repository (code comments, docs, commit
messages, PR descriptions) is in English. Full standards: docs/coding-standards.md; architecture:
docs/architecture.md. Do not mention Unreal Engine in README.md.

## Iteration loop (run this first)
- Gate before pushing: `python3 tools/air.py check` (format + layering + build + all tests). Add
  `--full` (clang-tidy + TSAN) after touching concurrent code.
- Run a subset of tests: `python3 tools/air.py test <name substring>`.
- Run the renderer: `python3 tools/air.py run -- render --features pathtracer,example --frames 2 --report out/report.json`
  - The report is JSON: per-frame stats, feature parameters and status, `errors`, `passed`. Exit
    code 0 = passed, 1 = validation errors or leaks, 2 = usage or environment error.
  - `--dump-commands` prints the full command stream; `--dump-graph x.dot` writes the render
    graph; `aircli features` lists every feature and its parameters.
- Image comparison: `aircli compare ref.pfm test.pfm --diff diff.ppm` prints rmse / psnr /
  differing pixels as JSON.
- A failing golden snapshot (`tests/golden/`) means the command stream or render graph changed:
  once the change is confirmed as intended, run `python3 tools/air.py golden-update` and say so in
  the PR.
- All of the above works without a GPU (Null backend).

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

## Thread safety
Device is thread-safe; CommandList is not; render graph construction is single threaded; prefer
serialising shared mutable state through a `Pipe` over adding locks.

## Style
- C++20, as defined by `.clang-format` / `.clang-tidy`: types and functions `PascalCase`, private
  members `mName`, constants `kName`.
- Invariants use `AIR_CHECK` (always on, messages name the pass/resource); debug-only checks use `AIR_DCHECK`.
