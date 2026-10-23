# Airender Coding Standards

The standards are enforced by tools rather than memory: code conforms when
`python3 tools/air.py check` passes.

| Area | Tool | Configuration |
|---|---|---|
| Formatting | clang-format 18 | `.clang-format` |
| Naming, common bugs, concurrency, performance | clang-tidy 18 | `.clang-tidy` (bug patterns and naming are errors) |
| Module layering | `tools/air.py layers` | `LAYERS` table in the script |
| Editors | EditorConfig | `.editorconfig` |
| Behaviour | Unit tests + golden snapshots | `tests/`, `src/features/*/tests/`, `tests/golden/` |

### Layout

- The opening brace of every multi-line block goes on its own line, aligned with the statement it
  belongs to (Allman): functions, classes/structs/enums, namespaces, `if`/`else`/`for`/`while`/
  `switch`/`case` blocks and multi-line lambda bodies. `} else {` is three lines.
- One-line inline functions and one-line lambdas stay on one line (`bool IsValid() const { return
  mIndex != 0; }`); braced initializers are not blocks (`Desc{.width = 8}` stays as is).
- C++ is laid out by clang-format 18 (`BreakBeforeBraces: Allman`). clang-format cannot parse HLSL,
  so `.hlsl`/`.hlsli` brace placement is checked and fixed by `tools/air.py format` itself, which
  touches nothing else in shader files.
- Never hand-format around the tools: run `python3 tools/air.py format` and commit the result.

## 1. Modules and dependencies (low coupling)

```
core  ←  rhi  ←  rdg  ←  renderer  ←  features/<name>      apps / tests may depend on anything
```

- A module may only include modules to its left (lower layers); `tools/air.py layers` checks every
  `#include "air/..."`.
- **Features never depend on each other**: `features/a` must not include `features/b`. Move shared
  code down into `renderer`.
- One CMake library per module, linking only its direct dependencies. Feature libraries are created
  by `air_add_feature()` and link only `air_renderer`.
- Public headers go in `include/air/<module>/`, implementation in `src/`; implementation details
  (e.g. backend classes) stay in `.cpp` files inside an anonymous namespace.
- Headers include only what their interface needs; everything else is included in the `.cpp`.

## 2. Extension points

| Goal | How | Core code changes? |
|---|---|---|
| New rendering algorithm / effect | `python3 tools/air.py new-feature <name>`, implement `RenderFeature` | No |
| New tunable parameter | `mParameters.Add(...)` in the feature constructor; `--set <feature>.<param>=v` on the CLI | No |
| New RHI backend | Implement `rhi::Device` in `src/rhi/src/<backend>/`, add a case to `CreateDevice()` | Factory function only |
| New resource format / state | Add the enum value in `RHITypes.h` and update its `ToString` | Yes (RHI vocabulary) |

## 3. Naming

| Kind | Rule | Example |
|---|---|---|
| Types (class/struct/enum) | `PascalCase` | `RenderGraph`, `TextureDesc` |
| Functions, methods | `PascalCase` | `AddPass()`, `GetStats()` |
| Locals, parameters | `camelCase` | `passIndex`, `heapSize` |
| Private/protected members | `m` + `PascalCase` | `mDevice`, `mPasses` |
| Public struct fields | `camelCase` | `desc.debugName` |
| `constexpr` constants | `k` + `PascalCase` | `kMinBlockSize` |
| Enum values | `PascalCase`; texture formats keep DXGI spelling | `Access::UAVCompute`, `Format::RGBA16_FLOAT` |
| Namespaces | lower case | `air::rdg` |
| Feature names | `lower_snake_case` (also the directory and CLI name) | `pathtracer`, `soft_shadows` |
| Files | Named after the main type | `RenderGraph.h / RenderGraph.cpp` |

Debug names (`debugName`, pass names) use the `Feature.Thing` form, e.g.
`PathTracer.Accumulation`, so they are easy to find in command streams and in PIX / RenderDoc.

## 4. Ownership and lifetime

- GPU resources are handles (`TextureHandle`, ...); whoever creates one destroys it. `Destroy*` is
  deferred and may be called while the GPU still uses the resource.
- CPU objects with a single owner use `std::unique_ptr`; no raw `new/delete` (low-level
  allocators excepted).
- Transient resources belong to the render graph; never keep a `TextureRef` outside a pass. Use
  `ImportTexture` / `ExtractTexture` for data that crosses frames.
- Non-copyable classes explicitly `= delete` their copy constructor and assignment.

## 5. Thread safety

See `docs/architecture.md` §2.2. Additional rules:

- Prefer serialising shared mutable state through a `Pipe`. When a lock is unavoidable, keep the
  mutex next to the data it protects and comment it with "guards ...".
- Pass execute lambdas capture by value only; never capture `this` to read members that change.
- After changing concurrent code in core / rhi / rdg, run `python3 tools/air.py test --preset tsan`.

## 6. Error handling

- Invariants whose violation would corrupt GPU state use `AIR_CHECK(cond, "fmt", args...)`, which
  is always enabled; messages name the pass / resource involved.
- Debug-only checks use `AIR_DCHECK`.
- Recoverable failures (missing files, invalid parameters, out of memory) return `bool` /
  `std::optional` / an invalid handle and explain why with `AIR_LOG(Error, ...)`.
- No exceptions on the rendering path; the task system catches and logs exceptions thrown by tasks.

## 7. Testing

- Every new feature has at least one Null backend test asserting that `validationErrors` is empty
  (use `Headless` from `tests/TestHelpers.h`).
- Changing pass order, barriers, culling or aliasing makes the golden snapshots fail: once the
  change is confirmed as intended, run `python3 tools/air.py golden-update` and review the
  `tests/golden/` diff in the PR.
- Tests must not depend on timing, thread scheduling, or randomness other than fixed seeds.

## 8. Shaders

- HLSL, SM 6.6, bindless (`ResourceDescriptorHeap[index]`), parameters through push constants
  (≤ 128 bytes).
- Structs shared with C++ go in `src/features/<name>/shaders/<Name>Shared.h`, compile on both
  sides, and have their sizes locked with `static_assert`.
- One entry point per `.hlsl` file (Vulkan allows one push constant block per entry point);
  compiled outputs are named `<Stem>.cs.dxil|spv` and `<Stem>.lib.dxil|spv`.
- Code that must match on the GPU and the CPU (BSDFs, the integrator) goes in `.hlsli` files that
  compile as both HLSL and C++ through `air/renderer/shader/HlslCompat.h`: use only the types and
  intrinsics it provides, `AIR_INOUT` / `AIR_OUT` for out parameters and `AIR_CONST` for constants.
  Resource access goes through hook functions that the including file defines (HLSL: descriptor
  loads; C++: a test kernel or the reference renderer), and CPU tests cover the shared code.

## 9. Language

- Code comments, documentation, commit messages and pull request descriptions are written in English.

## 10. Commits

- Run `python3 tools/air.py check` before pushing (CI runs the same checks, plus TSAN and the
  Windows build).
- One pull request does one thing; the first line of a commit message is an imperative English
  summary of the change.
