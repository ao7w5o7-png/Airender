# Airender — AI 协作约定

用户用中文交流。代码注释和标识符用英文。规范全文：docs/coding-standards.md；架构：docs/architecture.md。

## 迭代循环（先跑这个）
- 推送前的门禁：`python3 tools/air.py check`（格式 + 分层 + 构建 + 全部测试）。改了并发代码加 `--full`（clang-tidy + TSAN）。
- 只跑部分测试：`python3 tools/air.py test <名称子串>`。
- 跑渲染器：`python3 tools/air.py run -- render --features pathtracer,example --frames 2 --report out/report.json`
  - 报告是 JSON：每帧统计、特性参数与状态、`errors`、`passed`。退出码 0 = 通过，1 = 有校验错误/泄漏，2 = 参数或环境错误。
  - `--dump-commands` 打印完整命令流；`--dump-graph x.dot` 输出 RenderGraph；`aircli features` 列出所有特性和参数。
- 图像对比：`aircli compare ref.pfm test.pfm --diff diff.ppm` 输出 rmse / psnr / 差异像素（JSON）。
- golden 快照（`tests/golden/`）失败说明命令流或 RenderGraph 变了：确认是预期变化后跑 `python3 tools/air.py golden-update`，并在 PR 里说明。
- 没有显卡也能完成以上全部（Null 后端）。

## 加新算法 = 加一个特性
- `python3 tools/air.py new-feature <lower_snake_name>` 从 `src/features/example` 生成骨架，CMake 自动发现。
- 特性只能依赖 core / rhi / rdg / renderer，不能 include 其他特性（`air.py layers` 会拦）。
- 可调参数用 `mParameters.Add(...)` 暴露，运行时 `--set <feature>.<param>=<value>`。

## 写 pass
- 用 `rdg::RenderGraph::AddPass`，在 setup 里声明每个资源的读写和 `Access`，不要手写 barrier。
- 着色器资源一律 bindless：execute 里用 `ctx.SRV()/ctx.UAV()` 取索引，放进 push constants（≤128 字节）。
- C++ 与 HLSL 共享的结构体放在特性的 `shaders/XxxShared.h`，加 `static_assert` 检查大小。
- execute lambda 可能在 worker 线程并发执行：只按值捕获，或只读共享数据。
- 每个新 pass 至少一个 Null 后端测试（`tests/TestHelpers.h` 的 `Headless`），断言没有校验错误。

## 线程安全
Device 线程安全；CommandList 不是；RenderGraph 构建单线程；共享可变状态优先交给 `Pipe` 串行化，而不是加锁。

## 风格
- C++20，以 `.clang-format` / `.clang-tidy` 为准：类型和函数 `PascalCase`，私有成员 `mName`，常量 `kName`。
- 不变量用 `AIR_CHECK`（始终开启，错误信息带上 pass/资源名），调试期检查用 `AIR_DCHECK`。
