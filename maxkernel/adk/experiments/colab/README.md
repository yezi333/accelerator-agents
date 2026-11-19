# Colab：有限预算的 PallasKernels 优化实验

本入口保留 ADK 作为 agent runtime，用 OpenAI Responses API 完成规划、代码修改与失败修复。它是现有 Auto Agent 的**受限实验入口**，不是论文 Parallel/Beam Search 的复现，也不是直接启动原有全部子 agent。

## 实施方案与边界

- 输入：现有 `gqa_decode` 或 `gqa_bwd` 模块、仓库 YAML、纯 JAX reference。
- 复用：ADK `LlmAgent` / `Runner`、MaxKernel TPU 硬件表、PallasKernels registry、reference、XProf 设备时间提取器。
- 模型只返回单个 kernel 模块的精确文本替换；没有 shell 或文件工具。调度器写候选副本，原仓库保持不变。
- 测试由固定 Python evaluator 驱动，使用两个确定性随机用例及一个非平凡 logits 用例；decode 额外改变有效 KV 长度。模型不能选择 shape、dtype、容限或 reference。
- 这些有限用例不构成正确性证明。最终接入算子库前，还须在目标 TPU 上运行该库完整的相关正确性测试。
- 模型生成的 Python 仍在本机进程中执行。本工具没有 OS 安全沙箱，不应用于执行不可信第三方代码。

两个例子：

| Kernel | 默认 shape | 本轮目标 |
| --- | --- | --- |
| `gqa_decode` | B8/Hq32/Hkv8/KV1024/D128，BF16 | 小规模 `block_b` / `block_kv` 搜索，再研究具体结构瓶颈 |
| `gqa_bwd` | B2/Hq32/Hkv8/S1024/D128，BF16，split | 固定 O/LSE 输入，完整 dQ+dK+dV；研究 causal 无效 tile 跳过 |

Decode 使用连续 KV cache，不包含分页或 KV cache 更新。`block_kv` 不会自动缩小完整 KV BlockSpec。Backward 的 O/LSE 由固定 forward reference 在计时前生成；不会把 forward 算入 backward 时间。首版不切换 fused backward。

## 默认流程与预算

```text
冻结源码/config → 原始实现验证与 profile
→ 原始实现最多 4 个调参配置
→ 最多 2 轮 ADK 规划 → 修改 → 编译/测试 → 失败修复
→ 最佳筛选候选与原始实现的设备时间确认 → 报告
```

参数搜索采用固定的小集合，顺序执行；不展开 Cartesian product。结构改写沿用已经选中的 tiling，本版不在每轮结构修改后再展开一次参数搜索。

| 预算 | 默认 |
| --- | --- |
| 同时使用 TPU 的 evaluator | 1 |
| TPU evaluation jobs | 12，包含 baseline、调参、失败尝试和最终确认 |
| 每个 job | 最多 300 秒 |
| API 调用 | 最多 10 次，不自动重试 HTTP 请求 |
| 每个 API 响应 | 最多 8000 输出 token（包括 reasoning）；用量保存在 usage.json |
| 结构优化 | 最多 2 轮，每轮最多 2 次修复 |
| profiling jobs | 最多 2，每个分别采集 baseline 与 candidate，因此最多 4 次 trace |
| 总时间 | 首次 run 起 3600 秒，含模型等待、编译、测试、断线时间 |

**job 数不是物理编译次数**：一个 job 验证并编译 baseline 与 candidate；内部也可能有多个 custom call。重复编译尽量使用持久编译缓存。任务在提交前扣预算，失败也计数；超时会杀掉整个 evaluator 进程组。最终确认预留一个 job，但 API/总时间预算耗尽仍可能使确认无法完成。

恢复不会重置预算。失联期间也计入总时间；断电后遗留的进程仍应先停止。主机锁和目录锁防止本工具在同一主机并发运行多个实验；外部 notebook/server 的 TPU 占用仍需先释放。

## Colab 环境

PallasKernels 当前要求 Python 3.11 和 JAX 0.8.1。请使用独立 Python 3.11 环境，不直接升级 notebook 已加载的 JAX。以下假设两个仓库已经在 Colab 可读目录中，且 accelerator-agents 包含本分支改动：

```bash
# 在 Colab shell 中执行，路径按实际 checkout 调整。
python -m pip install uv
uv python install 3.11
uv venv --python 3.11 /content/maxkernel-env
uv pip install --python /content/maxkernel-env/bin/python \
  -r /content/accelerator-agents/maxkernel/adk/dependency/openai_requirements.txt
uv pip install --python /content/maxkernel-env/bin/python \
  -e '/content/PallasKernels[tests,tpu]'
```

**TPU 由 evaluator 子进程独占**。启动前不要在 notebook 主进程调用 `jax.devices()` 或运行 TPU 运算；如果主进程已经初始化 TPU，先重启 runtime。不能同时启动原有 MaxKernel TPU server。所有 TPU 工作都由本入口顺序派发，避免多个进程争用 libtpu。

模型 key 放在环境变量中，不写入 task、源码或 notebook 输出。Notebook 可以使用 Colab Secrets：

```python
import os
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["MAXKERNEL_MODEL"] = "替换为账户可用且支持 Responses API 的模型 ID"
# 可选，必须是所选模型支持的 reasoning effort：
# os.environ['MAXKERNEL_REASONING_EFFORT'] = 'medium'
```

不是 ChatGPT/Codex 登录 token；本入口使用 OpenAI Platform API key，按 API 用量计费。不需要 Gemini key、Vertex RAG 或 Codex CLI。

## 使用

所有命令在 `accelerator-agents/maxkernel/adk` 下运行。Notebook 用 `%cd` 切换目录后，以 `!` 调用对应命令。

```bash
cd /content/accelerator-agents/maxkernel/adk

# prepare 仅复制和冻结任务描述，不调用模型、不初始化 JAX/TPU。
/content/maxkernel-env/bin/python -m experiments.colab.run prepare \
  --repo /content/PallasKernels --kernel gqa_decode \
  --run-dir /content/experiments/decode

/content/maxkernel-env/bin/python -m experiments.colab.run run \
  --run-dir /content/experiments/decode

# 第二个例子，等第一个进程完全退出后再运行。
/content/maxkernel-env/bin/python -m experiments.colab.run prepare \
  --repo /content/PallasKernels --kernel gqa_bwd \
  --run-dir /content/experiments/backward
/content/maxkernel-env/bin/python -m experiments.colab.run run \
  --run-dir /content/experiments/backward
```

`prepare` 可指定 `--wall-seconds`、`--tpu-jobs`、`--model-calls`，也可用 `--seq-len` 做独立扩展实验。延长序列可能超过 VMEM 或时间预算，不自动提高资源上限。

先做不用 API 的控制组：prepare 一个新目录，再执行 `run --mode autotune-only`。这种模式仍然使用真实 TPU 和 profile。它不与 agent 实验自动构成等预算比较；报告中必须分别列出实际消耗。

恢复同一实验：

```bash
/content/maxkernel-env/bin/python -m experiments.colab.run run \
  --run-dir /content/experiments/decode --resume
```

`autotune-only` 恢复时也要带同一 `--mode`。恢复在阶段/任务边界进行；中断的结构轮次不会自动重放，已扣预算不退回。原仓库、baseline、task/config 或硬件环境变化时，创建新实验并重新测 baseline。

运行目录及源码在 Colab runtime 消失后不会自动保留。可把目录放到持久盘，或在任务间备份。持久缓存可能很大；缓存不是必须归档的证据。

## 结果解释

- `task.json`：冻结的工作负载、源码指纹和预算。
- `state.json` / `budget.json`：阶段、已花预算、全部候选及失败反馈。
- `model_calls/`：实际 prompt、回复、返回的 token 用量。失败请求的 token 用量可能不可用，调用次数仍计入。
- `jobs/`：固定输入请求、日志、正确性结果、baseline/candidate HLO，以及选定阶段的 XPlane/设备时间。
- `report.json`：原始→仅调参→结构优化的筛选收益、设备确认和实际预算消耗。
- `best.py` + `best_config.json`：必须同时使用。默认保留原始版本，仅在候选通过测试且设备 Program 时间改善超过 3% 和 3 倍 MAD 噪声时发布候选。

筛选采用预热后交替运行 baseline/candidate 的同步 **host latency**，只是低成本代理指标，可能漏掉只有设备时间改善的候选。最终使用 XProf **整个 public registry 入口的设备 Program 时间**确认；同时保留各 custom-call 的时间用于诊断。不能把 dQ/dKV custom-call 时间简单相加冒充 Program 时间，也不能将 host speedup 写成 device speedup。

本版给模型提供设备类型、软件版本、已有硬件表、profile 的 Program/custom-call 汇总、历史验证和计时结果。完整 trace 与 HLO 保存在磁盘；没有实现完整 DMA/SyncWait/寄存器压力自动诊断。设备采集不完整时不声称设备加速。

`best.py` 是替换同名 kernel 模块的候选，不是脱离 PallasKernels 的独立库。实验通过临时模块加载接入原 registry；没有修改原仓库。进入正式库前还需相关 TPU 回归测试。

## 验证改造本身（无需 API key / TPU）

```bash
python -m pip install pytest pytest-asyncio
python -m pytest tests/openai_colab -q
```

测试覆盖真实 OpenAI SDK 的 HTTP 序列化、真实 ADK 工具往返、会话 JSON 恢复、预算不重置、超时杀子进程、设备计时口径与发布回退。真实模型可用性、Colab backend 编译和性能收益必须另行实测。

本次本地验证：上述 21 项测试通过；PallasKernels 中选取的 6 项 CPU reference 测试通过；实际仓库的 `prepare` 成功。未调用付费 API，未完成 TPU 编译、正确性或性能验证。原有 `test_pipeline_agent_metrics.py` 因本地缺少 TensorFlow 无法收集，因此不视为原 pipeline 全量回归通过。

## 通用 Auto Agent 的 OpenAI 模式

本分支也为原有 `auto_agent` / `auto_search` 增加模型适配。安装原始 agent 依赖后额外安装 `dependency/openai_requirements.txt`，启动前设置：

```bash
export MAXKERNEL_LLM_PROVIDER=openai
export MAXKERNEL_MODEL='账户可用的模型 ID'
# OPENAI_API_KEY 从秘密存储注入；可选 OPENAI_BASE_URL 必须支持 Responses API。
# 可选 MAXKERNEL_REASONING_EFFORT / MAXKERNEL_API_TIMEOUT / MAXKERNEL_MAX_OUTPUT_TOKENS
# 可选 MAXKERNEL_COMPACTION_TOKENS，默认 32000，需为系统 prompt 和输出留空间。
unset RAG_CORPUS
```

通用 Auto Agent 没有本实验入口的全局硬预算；不要用它代替上述 Colab 命令。Gemini 仍是默认 provider。当前适配支持文本和 function tools、buffered response；不支持 Gemini 原生搜索、多模态、ADK `output_schema`。不支持的输入会明确报错，不会静默丢弃。`hitl_agent` 的独立 Gemini 配置不在此次改造范围。

官方接口依据：[Function calling](https://developers.openai.com/api/docs/guides/function-calling)、[Reasoning state](https://developers.openai.com/api/docs/guides/reasoning)。
