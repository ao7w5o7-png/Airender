# Airender

一个让 AI 能直接写代码、调用和调试的图形渲染器：支持实时渲染与 GPU 路径追踪的影视级离线渲染，
同时作为图形算法试验场。C++20，RHI 抽象 D3D12 + Vulkan，参考 Unreal Engine 的 RHI / RDG / Tasks 设计。

- 架构设计：[docs/architecture.md](docs/architecture.md)
- AI 协作约定：[CLAUDE.md](CLAUDE.md)

## 模块

| 目录 | 内容 |
|---|---|
| `src/core` | 任务系统（Task DAG / Pipe / ParallelFor）、句柄池、日志 |
| `src/rhi` | RHI 接口、显存分配器（TLSF + 分池 + dedicated）、Null 后端、D3D12/Vulkan 后端（开发中） |
| `src/rdg` | RenderGraph：自动裁剪、自动 barrier、瞬态显存复用、并行录制 |
| `src/pathtracer` | 硬件光追渐进路径追踪 |
| `shaders/` | HLSL（DXC → DXIL / SPIR-V） |
| `apps/aircli` | 无头命令行驱动，打印 RenderGraph / 命令流 / 显存统计 |
| `tests/` | 单元测试（无外部依赖） |

## 构建

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/apps/aircli/aircli --frames 2 --dump-commands
```

Windows 上默认同时编译 D3D12 后端；Vulkan 后端用 `-DAIR_RHI_VULKAN=ON` 打开。
