# Colab：五策略完整原生 ADK pipeline

完整 notebook 在 [maxkernel_v6e.ipynb](maxkernel_v6e.ipynb)。它替代旧 notebook 的简化实验入口，并整合了此前分散的安装、启动、查看结果 cells。

在 Colab 选择 TPU runtime，上传 notebook，按顺序执行。已有 PallasKernels 和 Python 环境可以复用；先停止旧实验。Cell 3 会从 `yezi333/accelerator-agents` 的 `codex/adk-openai-colab` 分支快进更新代码，不再固定旧 commit。如果本地分支发生分叉或修改冲突，会停止更新并显示 Git 错误。

## 配置

| 参数 | notebook 初始值 | 含义 |
| --- | --- | --- |
| `STRATEGIES` | 下面列出的 5 条 | 每条策略产生一条原生 Parallel Search 路线 |
| `FULL_ITERATIONS` | `5` | 每个 worker 的原生 pipeline 最大优化轮数 |
| `MAX_CONCURRENCY` | `2` | 同时运行的 worker 上限，其余排队 |
| `FULL_RUN_NAME` | `gqa-decode-native-five-01` | 每次运行使用新名称，避免覆盖已有结果 |

五条初始策略来自原生策略池，可在 Cell 1 编辑：

1. `Loop tiling`
2. `Overlap data movement and compute`
3. `Cache reused data in local memory instead of reloading from main memory`
4. `Loop reordering and restructuring`
5. `Eliminate redundant computation`

5 条路线分别从同一份 Pallas baseline/初始候选开始。最多共 25 个优化轮次，编译检查、修复、调参等单轮内部操作按原生上限执行，并不计入这 25 轮。两条路线可以同时进行模型规划；TPU 执行请求仍由原 TPU 服务排队处理。

模型和代理地址保留 notebook 中原有配置。API key 从环境变量、Colab Secrets 中的 `OPENAI_API_KEY` 或隐藏输入读取，不写入 notebook 源码。代码也支持 `MODEL_NAME`；同时设置时 `MAXKERNEL_MODEL` 优先。

## Cell 对应关系

| Colab 标题编号 | 内容 |
| --- | --- |
| 1 | 五条策略、轮数、并发、模型和结果目录 |
| 2 | 检查配置、显示策略 |
| 3 | 更新 fork、上传或复用 PallasKernels、安装完整原生依赖 |
| 3.5 | 模型 API 连通性测试，会发送一次真实请求 |
| 4 | 启动五策略完整搜索；中断时先请求清理服务并保存结果 |
| 5 | 查看各策略、各轮编译/正确性/调参/Profile，以及全局最佳 |
| 6 | 下载完整结果包 |

不要在 notebook 主进程初始化 JAX/TPU；协调进程使用 CPU，TPU 服务派发的子进程使用 TPU。依赖安装和 CPU-only 导入检查均在独立 Python 3.11 环境进行。

## 实际入口与保留的流程

```text
experiments.full_adk.run
  ├─ prepare_input：原 gqa_decode + computation/get_inputs 接口
  ├─ 原 cpu_server / tpu_server / eval_server
  └─ experiments.full_adk.search.run_search
       └─ SimpleParallelSearchOrchestrator（5 条策略，并发 2）
            └─ ADKSessionWorker（每条路线独立目录与会话）
                 └─ create_root_agent / AutoAgentClient
                      └─ AutonomousPipelineAgent（每条最多 5 轮）
```

`ColabSessionWorker` 只补充复制 `inputs/` 中的依赖源码和说明；执行、状态处理与结果选择沿用原 worker 和搜索图。原生顺序仍为：准备 baseline → 生成/验证测试 → 每轮规划、实现、编译与修复、正确性测试、autotune、profiling/分析 → 历史最佳选择。编译或正确性失败的轮次按原逻辑进入下一轮；初始测试生成/验证耗尽重试仍失败时，该 worker 失败。

当前输入适配针对 PallasKernels 的单设备 `gqa_decode`，保留原 YAML 工作负载和容差，优化现有 Pallas 实现。原 agent 生成测试，比较对象是原 Pallas baseline。`inputs/jax_ops/gqa_decode.py` 是补充源码，本入口没有替换成固定 JAX oracle。

原 prompts、API 查询、MCP 文件工具、autotune 和 XPlane 分析工具均保留；可选 Vertex RAG 未配置。原 `get_hlo_dump` 仍为占位实现。旧简化实验的 `TPU_JOBS`、`MODEL_CALLS`、`WALL_SECONDS`、`STRUCTURE_ROUNDS` 和 3%/MAD 晋级规则不参与此流程。启动器未传 `--strategies` 时仍支持原来的单 agent 模式。

## 结果

所有文件位于 `full_adk_results/<FULL_RUN_NAME>/`：

- `search_config.json`：本次策略、轮数和并发。
- `search_graph.json`：原生搜索图，运行中随节点更新持久化。
- `nodes/node_*_attempt_1/session.json`：每条路线的完整事件和 state/history，以及同目录的 kernel、plan、profile 文件。
- `search_summary.json`：路线状态、各轮结果和全局最佳；中断时尽量保存已完成结果。
- `optimized_kernel.py`、`base_kernel_plan.md`：有有效候选时导出的全局最佳。
- `services/*.log`、`input_config.json`、`input_provenance.json`：服务日志、输入配置和来源。

没有有效候选时，summary 的 `best_node_id` 和 `optimized_kernel_path` 为 `null`；根目录初始 `optimized_kernel.py` 仍可能存在，不能将它当作优化成功。`completed` 表示搜索任务执行结束，候选是否有效应查看最佳节点和各路线 evaluation。

原生 speedup 使用原测试/调参计时规则，每轮 profiling 不代表最终选择经过独立 device-only 确认。本入口不提供阶段级断点续跑，每次使用新的实验目录。强制终止进程可能来不及写 summary，此时结果 cell 会尝试展示已保存的搜索图。
