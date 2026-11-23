# 用 OpenAI 兼容 API 运行原 Claude Code pipeline

这个入口提供原 Claude Code 工具宿主的 Python 实现，通过 OpenAI **Responses API** 驱动原 MaxKernel pipeline。它直接渲染原始 orchestrator、worker 和全部 22 个 subagent prompts，沿用原工具、知识库、编译/正确性检查、autotune 和 XProf profiling。无需 Claude Code 登录或 Codex CLI；它不是 ADK search，也没有额外的五条并行策略。

## 快速开始

Colab TPU v6e：打开 [maxkernel_codex_v6e.ipynb](maxkernel_codex_v6e.ipynb)，选择 TPU runtime，按顺序执行。默认使用独立的 matmul + bias + ReLU 示例，并沿用此前实验的代理、模型和 6 分钟超时。换成自己的 kernel 时，推荐提供原格式的 job.json，明确输入、reference、容差和轮数。

Linux TPU 服务器：在仓库根目录创建独立的 Python 3.12 环境，安装**原 Claude Code pipeline 的依赖**及新增宿主依赖。不要用 ADK 实验的 JAX 版本覆盖这套环境。

```bash
uv venv --python 3.12 /tmp/maxkernel-codex-venv
uv pip install --python /tmp/maxkernel-codex-venv/bin/python \
  -r maxkernel/claude-code/requirements.txt \
  -r maxkernel/claude-code/openai_runtime/requirements.txt
uv pip check --python /tmp/maxkernel-codex-venv/bin/python
cd maxkernel/claude-code
```

原 requirements 同时包含工具的检索依赖，包括 google-adk；本入口的调度本身不使用 ADK。服务器需要 `rg`（原 requirements 已包含 ripgrep）、Bash，以及本地 TPU 或按 [TPU setup](../docs/tpu-setup.md) 配置的远程 TPU。远程模式仍使用原工具约定的远端环境；`--python` 指定的是宿主工具环境。

配置 API（密钥通过交互读取，不写入文件）：

```bash
export OPENAI_BASE_URL=https://agent.infiscale-tech.com/v1
export MAXKERNEL_MODEL=gpt-5.6-sol
export MAXKERNEL_REASONING_EFFORT=high
export MAXKERNEL_MAX_OUTPUT_TOKENS=8000
export MAXKERNEL_API_TIMEOUT=360
export MAXKERNEL_API_MAX_RETRIES=2
read -r -s -p 'OPENAI_API_KEY: ' OPENAI_API_KEY  # Bash
export OPENAI_API_KEY
```

以下命令均在 `maxkernel/claude-code/` 执行。先做不调用模型和 TPU 的预检查：

```bash
/tmp/maxkernel-codex-venv/bin/python -m openai_runtime \
  --job openai_runtime/examples/matmul_bias_relu.job.json \
  --run-dir workspace/codex-matmul-01 \
  --dry-run
```

随后开始真实实验：

```bash
/tmp/maxkernel-codex-venv/bin/python -m openai_runtime \
  --run-dir workspace/codex-matmul-01 --resume
```

也可以首次直接运行，去掉 `--dry-run` 即可。单个自包含源码用 `--source /absolute/path/kernel.py` 代替 `--job`。新实验必须使用新的 `--run-dir`，且位于该 checkout 的 `workspace/` 下。job 的相对路径按 job 所在目录解释；默认示例为 BF16 512×512 matmul，5 轮，atol/rtol 均为 0.01。

## 运行过程与代码位置

```text
python -m openai_runtime
  __main__.py：参数、原 job 校验、运行目录、恢复与结果检查
  prompts.py：用原 install.py 渲染完整 prompts / guides / general_rules
  runtime.py：Responses 对话、独立子 agent 上下文、工具回传与上下文压缩
  tools.py：Agent / Bash / Read / Write / Edit / Glob / Grep / AskUserQuestion
    原 skill/SKILL.md（orchestrator）
      原 agents/maxkernel-worker.md（每次负责一轮）
        原阶段 specialists → 原 tools/*.py → 原 server/tpu_server.py
```

每轮仍按原 worker 的条件执行输入处理与 reference 验证、plan → implement → compile/fix → test → autotune → profile → 更新 state。源码转换、oracle、extraction 等阶段是否执行由原输入类型与 job 决定。原编译重试和 worker 重派规则保留；原默认优化轮数为 5，job 的 `loop.max_iterations` 可覆盖。子 agent 同步运行，各自拥有独立上下文，父 agent 收到子 agent 的最终报告。

原 prompts 和知识文件没有被改写成简化版。宿主前缀只说明工具接口、路径、同步执行和暂停/恢复规则；每次启动保存渲染结果与源文件 SHA256。上下文达到默认 32000 input tokens 且超过五个工具轮次后，使用同一模型总结较早的上下文，保留初始任务和最近五个完整轮次；每个 agent 最多 500 次模型响应，这不是 kernel 优化轮数。

## API 配置与超时

| 配置 | 行为 |
| --- | --- |
| `MAXKERNEL_MODEL` | 必填；也支持 `MODEL_NAME`，`--model` 优先 |
| `OPENAI_API_KEY` | 实际调用必填；dry-run 不需要 |
| `OPENAI_BASE_URL` | 默认 `https://api.openai.com/v1`；notebook 显式设为此前代理 |
| `MAXKERNEL_API_TIMEOUT` | 默认 360 秒，传给 OpenAI SDK |
| `MAXKERNEL_API_MAX_RETRIES` | 默认 2，即 SDK 最多三次请求尝试 |
| `MAXKERNEL_REASONING_EFFORT` | 默认 `high`；空字符串时不发送 reasoning 参数 |
| `MAXKERNEL_MAX_OUTPUT_TOKENS` | 默认 8000 |
| `MAXKERNEL_COMPACTION_TOKENS` | 默认 32000；上下文压缩阈值 |

代理必须支持 `/responses`、function calling、reasoning 及加密 reasoning 的回传。仅支持 `/chat/completions` 的服务不能直接使用。配置与 ADK 实验相同不代表代理的所有接口都已验证。

360 秒是 SDK/HTTP 的连接、读写等超时设置，不是整个 agent 或整轮优化的总时限；一次调用重试两次时可能接近 18 分钟，另加退避等开销。日志会记录每次宿主模型调用的耗时和错误。Bash 工具另有独立超时（默认一小时，模型可为命令指定毫秒值）。

## 结果、暂停与恢复

主要结果在 `workspace/<run>/`：

- `state.json`、`iter<n>/`、`optimized.py`：原 pipeline 的状态、各轮产物与最终胜出 kernel。
- `openai_host/status.json`、`report.md`：宿主状态及模型报告。宿主额外核对原 history 中正确且最快的候选、最终代码和 plan/autotune/profile 产物；这不替代 TPU 实测。
- `openai_host/events.jsonl`、`sessions/`、`tools/`：模型耗时/用量、对话与工具输出。
- `openai_host/prompts/<id>/`：每次启动的渲染快照及配置。

`completed` 表示原优化轮数已执行完；还需看 `has_valid_candidate`，它可能为 false。宿主退出码：0 完成，1 失败或未完成，2 等待人工输入，130 用户中断。

原流程需要人批准生成的 baseline、抽取的源码或补充信息时，整个运行暂停，写入 `openai_host/pending_question.json`。阅读问题和引用的产物后，将自己的决定写入文本文件，再执行：

```bash
/tmp/maxkernel-codex-venv/bin/python -m openai_runtime \
  --run-dir workspace/codex-matmul-01 --resume \
  --reply-file /absolute/path/my-reply.txt
```

回复可以批准、拒绝或要求修改；宿主不会自动设置 approved。恢复使用原请求和磁盘 state，开启新的模型上下文，已完成轮次由原 orchestrator 读取。失败的当前阶段可能重做；这不是对任意工具调用的精确断点恢复。中断或超时后也可用 `--resume`，同时继续设置所需 API 环境变量和 `--python`/`--with-guard` 等启动选项。

`--with-guard` 会在每个工具调用前执行原可选 workspace guard 并尊重其拒绝结果。文件工具限制了写入运行目录；**Bash 执行模型生成的本机命令，宿主及原 guard 都不是操作系统沙箱**。仅在用于实验的可信环境运行。API key 不传给 Bash 子进程，也会从宿主日志中脱敏；会话和产物仍可能含源码与数据。

原 TPU server 可在宿主结束后继续运行。不要与 ADK 实验同时占用 TPU；原 emergency cleanup 可能取消服务器中的全部任务，因此建议独占实验 runtime。需要手动清理时先按原 TPU 文档查看队列，再按具体 job ID 取消。

## 已知边界与验证

当前上游 checkout 的 prompts 引用了缺失的 `evaluation/compare_kernel.py`、`evaluation/templates/reference_model.py.template` 等文件。需要这些文件的 external evaluation 或 specification job 会在入口预检查时明确失败，不会绕过相应阶段。自包含 JAX 示例走原内部 harness，不依赖它们。原 `torch_xla` 的 TPU 兼容性限制也保留，见 [原 requirements 注释](../requirements.txt)。

本次离线测试使用真实 OpenAI SDK 的 mock HTTP transport，覆盖多级 agent 分发、reasoning 回传、人工暂停、上下文压缩、工具超时、原 guard 和产物检查；没有据此宣称模型能稳定完成全部阶段或获得加速。真实代理接口、TPU 编译/精度/性能及整套 Colab 依赖安装仍需实际实验验证。

在已安装宿主依赖、pytest、pytest-asyncio 和 httpx 的环境中执行：

```bash
python -m pytest openai_runtime/tests -q
```
