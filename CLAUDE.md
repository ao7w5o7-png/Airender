# Airender — AI 协作约定

用户用中文交流。代码注释和标识符用英文。

## 构建与验证
- 构建：`cmake -S . -B build -G Ninja && cmake --build build`
- 测试：`./build/tests/air_tests [名称子串]`。改动 core / rhi / rdg 后必须跑全部测试。
- 线程相关改动额外跑 TSAN：`cmake -S . -B build-tsan -G Ninja -DAIR_ENABLE_TSAN=ON`（用 GCC）。
- 无显卡调试：`aircli --dump-commands` 打印一帧的完整命令流；`--dump-graph x.dot` 输出 RenderGraph。

## 写新 pass
- 用 `rdg::RenderGraph::AddPass`，在 setup 里声明每个资源的读写和 `Access`，不要手写 barrier。
- 着色器资源一律 bindless：在 execute 里用 `ctx.SRV()/ctx.UAV()` 取索引，放进 push constants（≤128 字节）。
- C++ 与 HLSL 共享的结构体放在 `shaders/**/XxxShared.h`，并加 `static_assert` 检查大小。
- execute lambda 可能在 worker 线程并发执行：只按值捕获，或只读共享数据。
- 每个新 pass 至少有一个 Null 后端测试，断言命令流里没有 `VALIDATION` 行。

## 线程安全规则
见 docs/architecture.md §2.2。要点：Device 线程安全；CommandList 不是；RenderGraph 构建单线程；
共享可变状态优先交给 `Pipe` 串行化，而不是加锁。

## 风格
- C++20，`.clang-format` 为准；类型 `PascalCase`，成员 `mName`，函数 `PascalCase`。
- 不变量用 `AIR_CHECK`（始终开启，错误信息带上 pass/资源名），调试期检查用 `AIR_DCHECK`。
