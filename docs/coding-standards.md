# Airender 代码规范

规范由工具强制执行，而不是靠人记：`python3 tools/air.py check` 通过才算符合规范。

| 方面 | 工具 | 配置 |
|---|---|---|
| 格式 | clang-format 18 | `.clang-format` |
| 命名、常见 bug、并发、性能 | clang-tidy 18 | `.clang-tidy`（bug 类与命名规则是 error） |
| 模块分层 | `tools/air.py layers` | 脚本里的 `LAYERS` 表 |
| 编辑器 | EditorConfig | `.editorconfig` |
| 行为 | 单元测试 + golden 快照 | `tests/`、`src/features/*/tests/`、`tests/golden/` |

## 1. 模块与依赖（低耦合）

```
core  ←  rhi  ←  rdg  ←  renderer  ←  features/<name>      apps / tests 可以依赖任何模块
```

- 只能 include 左边（下层）的模块，`tools/air.py layers` 会检查每一个 `#include "air/..."`。
- **特性之间互不依赖**：`features/a` 不能 include `features/b`。需要共享的东西下沉到 `renderer`。
- 每个模块一个 CMake 库，只 `target_link_libraries` 它直接依赖的库。特性库由 `air_add_feature()` 创建，只链接 `air_renderer`。
- 公共头文件放在 `include/air/<module>/`，实现放在 `src/`；实现细节（例如后端类）只在 `.cpp` 里，用匿名命名空间。
- 头文件里只放接口需要的 include，其余放 `.cpp`。

## 2. 扩展点

| 想做的事 | 做法 | 需要改核心代码吗 |
|---|---|---|
| 新渲染算法 / 效果 | `python3 tools/air.py new-feature <name>`，实现 `RenderFeature` | 不需要 |
| 新的可调参数 | 在特性构造函数里 `mParameters.Add(...)`，CLI 用 `--set <feature>.<param>=v` | 不需要 |
| 新 RHI 后端 | 在 `src/rhi/src/<backend>/` 实现 `rhi::Device`，在 `CreateDevice()` 里加一个分支 | 只改工厂函数 |
| 新资源格式 / 状态 | `RHITypes.h` 加枚举值，更新对应 `ToString` | 是（RHI 词汇表） |

## 3. 命名

| 种类 | 规则 | 例子 |
|---|---|---|
| 类型（class/struct/enum） | `PascalCase` | `RenderGraph`, `TextureDesc` |
| 函数、方法 | `PascalCase` | `AddPass()`, `GetStats()` |
| 局部变量、参数 | `camelCase` | `passIndex`, `heapSize` |
| 私有/受保护成员 | `m` + `PascalCase` | `mDevice`, `mPasses` |
| 公有结构体字段 | `camelCase` | `desc.debugName` |
| `constexpr` 常量 | `k` + `PascalCase` | `kMinBlockSize` |
| 枚举值 | `PascalCase`；纹理格式沿用 DXGI 写法 | `Access::UAVCompute`, `Format::RGBA16_FLOAT` |
| 命名空间 | 小写 | `air::rdg` |
| 特性名 | `lower_snake_case`（也是目录名、CLI 名） | `pathtracer`, `soft_shadows` |
| 文件 | 与主类型同名 | `RenderGraph.h / RenderGraph.cpp` |

调试名（`debugName`、pass 名）统一用 `Feature.Thing` 形式，例如 `PathTracer.Accumulation`，便于在命令流和 PIX/RenderDoc 里定位。

## 4. 所有权与生命周期

- GPU 资源用句柄（`TextureHandle` 等），谁创建谁销毁；`Destroy*` 是延迟的，可以在 GPU 仍在使用时调用。
- CPU 对象用 `std::unique_ptr` 表达唯一所有权；不要用裸 `new/delete`（底层分配器除外）。
- 渲染图里的临时资源由渲染图管理，不要在 pass 外保存 `TextureRef`。跨帧数据用 `ImportTexture` / `ExtractTexture`。
- 禁止拷贝的类显式 `= delete` 拷贝构造和赋值。

## 5. 线程安全

见 `docs/architecture.md` §2.2。补充规则：

- 共享可变状态优先交给 `Pipe` 串行化；必须加锁时，锁和它保护的数据放在一起并注释 "guards ..."。
- pass 的 execute lambda 只按值捕获；不要捕获 `this` 去读会变化的成员。
- 改动 core / rhi / rdg 的并发代码后必须跑 `python3 tools/air.py test --preset tsan`。

## 6. 错误处理

- 不变量（违反会导致 GPU 状态错乱）用 `AIR_CHECK(cond, "fmt", args...)`，始终开启；消息里带上 pass 名 / 资源名。
- 仅调试期检查用 `AIR_DCHECK`。
- 可恢复的失败（文件缺失、参数非法、显存不足）返回 `bool` / `std::optional` / 空句柄，并用 `AIR_LOG(Error, ...)` 说明原因。
- 不在渲染路径上抛异常；任务系统会捕获任务里的异常并记录日志。

## 7. 测试

- 每个新特性至少一个 Null 后端测试，断言 `validationErrors` 为空（用 `tests/TestHelpers.h` 里的 `Headless`）。
- 改变 pass 顺序、barrier、裁剪或复用行为时，golden 快照会失败：确认是预期变化后 `python3 tools/air.py golden-update`，并在 PR 里审阅 `tests/golden/` 的 diff。
- 测试不能依赖时间、线程调度或随机种子之外的随机性。

## 8. 着色器

- HLSL，SM 6.6，bindless（`ResourceDescriptorHeap[index]`），参数走 push constants（≤128 字节）。
- 与 C++ 共享的结构体放 `src/features/<name>/shaders/<Name>Shared.h`，两边都能编译，并用 `static_assert` 锁定大小。
- 一个 `.hlsl` 文件一个入口（Vulkan 每个入口只能有一个 push constant 块）；编译产物命名 `<Stem>.cs.dxil|spv`、`<Stem>.lib.dxil|spv`。

## 9. 提交

- 推送前跑 `python3 tools/air.py check`（CI 跑的是同一套检查，外加 TSAN 和 Windows 构建）。
- 一个 PR 做一件事；提交信息第一行用英文祈使句概括改动。
