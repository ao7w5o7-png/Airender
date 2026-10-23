# Airender

一个让 AI 能直接写代码、调用和调试的图形渲染器：支持实时渲染与 GPU 路径追踪的影视级离线渲染，
同时作为图形算法试验场。C++20，RHI 抽象 D3D12 + Vulkan，参考 Unreal Engine 的 RHI / RDG / Tasks 设计。

- 架构设计：[docs/architecture.md](docs/architecture.md)
- AI 协作约定：[CLAUDE.md](CLAUDE.md)

## 模块

| 目录 | 内容 |
|---|---|
| `src/core` | 任务系统（Task DAG / Pipe / ParallelFor）、句柄池、日志、JSON、图像读写与对比 |
| `src/rhi` | RHI 接口、显存分配器（TLSF + 分池 + dedicated）、Null 后端、D3D12/Vulkan 后端（开发中） |
| `src/rdg` | RenderGraph：自动裁剪、自动 barrier、瞬态显存复用、并行录制 |
| `src/renderer` | 扩展点：RenderFeature 接口、特性注册表、参数系统、FrameRunner |
| `src/features/*` | 渲染特性插件（各自带着色器和测试）：`pathtracer`、`example`（新算法模板） |
| `apps/aircli` | 无头命令行：`render` / `features` / `compare`，JSON 报告 |
| `tools/air.py` | 开发 harness：check / test / run / golden-update / new-feature |
| `tests/` | 单元测试与 golden 快照（无外部依赖） |

## 快速开始

```bash
python3 tools/air.py check                         # 格式 + 分层 + 构建 + 全部测试
python3 tools/air.py run -- render --features pathtracer,example --frames 2 --report out/report.json
python3 tools/air.py new-feature my_effect         # 新算法骨架
```

需要 CMake ≥ 3.24、C++20 编译器（GCC 13 / Clang 18 / MSVC 2022），可选 Ninja、clang-format/clang-tidy 18。
Windows 上默认同时编译 D3D12 后端；Vulkan 后端用 `-DAIR_RHI_VULKAN=ON` 打开。
规范见 [docs/coding-standards.md](docs/coding-standards.md)。
