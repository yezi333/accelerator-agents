# 用 Codex CLI 执行原 MaxKernel pipeline

这个入口让 **Codex CLI 本身**负责模型调用、原生子 agent、文件操作、Shell 执行和上下文管理。原 orchestrator、worker、全部 22 个 subagent prompts、guides、知识库及 Python/TPU 工具继续使用。优化阶段不缩减：输入处理与 reference 验证 → plan → implement → compile/fix → correctness/timing → autotune → profile → 更新 state。

`codex_cli` 的 Python 代码只准备配置、启动 `codex exec`、记录日志、处理人工暂停及检查产物；不会调用 OpenAI SDK，也不会运行 `openai_runtime` 的模型/工具循环。它复用其中的纯 Python prompt 渲染辅助函数、job 校验与产物检查。

## Colab 与服务器入口

Colab：打开 [maxkernel_codex_cli_v6e.ipynb](maxkernel_codex_cli_v6e.ipynb)，选择 TPU runtime，顺序执行。默认是原 BF16 512×512 matmul + bias + ReLU 示例，原流程 5 轮；没有 ADK 的多策略 parallel search。源码示例与前一个宿主共用，便于比较。

服务器：需要 Node.js >= 16、npm、Python 3.12、Bash、原 pipeline 的完整依赖，以及本地 TPU 或 [远程 TPU 配置](../docs/tpu-setup.md)。固定安装已验证的 CLI 版本：

```bash
npm install --prefix /tmp/maxkernel-codex-cli @openai/codex@0.171.0-alpha.13
uv venv --python 3.12 /tmp/maxkernel-codex-venv
uv pip install --python /tmp/maxkernel-codex-venv/bin/python \
  -r maxkernel/claude-code/requirements.txt
uv pip check --python /tmp/maxkernel-codex-venv/bin/python
cd maxkernel/claude-code
```

当前固定版本为 `0.171.0-alpha.13`，来自实际验证的 CLI；升级版本前需重跑原生角色与工具集成测试。启动器会检查版本及全部角色是否注册。这里不需要安装 `openai_runtime/requirements.txt`。

配置沿用此前代理与模型。以下密钥输入命令用于 Bash：

```bash
export OPENAI_BASE_URL=https://agent.infiscale-tech.com/v1
export MAXKERNEL_MODEL=gpt-5.6-sol
export MAXKERNEL_REASONING_EFFORT=high
export MAXKERNEL_API_TIMEOUT=360
export MAXKERNEL_API_MAX_RETRIES=2
read -r -s -p 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
```

预检查不调用模型或 TPU，但会用真实 CLI 验证角色注册：

```bash
/tmp/maxkernel-codex-venv/bin/python -m codex_cli \
  --codex /tmp/maxkernel-codex-cli/node_modules/.bin/codex \
  --job openai_runtime/examples/matmul_bias_relu.job.json \
  --run-dir workspace/codex-cli-matmul-01 --dry-run
```

开始执行：

```bash
/tmp/maxkernel-codex-venv/bin/python -m codex_cli \
  --codex /tmp/maxkernel-codex-cli/node_modules/.bin/codex \
  --run-dir workspace/codex-cli-matmul-01 --resume
```

首次直接执行可以去掉 `--dry-run`。自己的 kernel 推荐使用原格式 `--job /path/to/job.json`，也支持 `--source /path/to/self_contained.py`。job 中的相对路径相对于 job 所在目录。新任务用新 `--run-dir`，必须在当前 checkout 的 `workspace/` 下。

## 角色与工具如何适配

- `prepare.py` 使用原 `install.py` 的渲染函数，原 prompt 正文在路径替换后完整保留；它为角色增加 Codex 工具适配前缀，并生成原生 agent TOML。
- 原 Claude `Agent(subagent_type=X, prompt=P)` 映射为 Codex 的 `collaboration.spawn_agent(agent_type=X, fork_turns="none", ...)`，父 agent 等待结果。每轮 worker 和每次 specialist 分发使用新上下文。
- 原 `Bash/Read/Write/Edit/Glob/Grep` 映射为 CLI 原生 Shell、文件补丁和 `rg`。独立的原 `tools/*.py` 真正执行编译、数值检查、测速、autotune 和 trace 分析。
- 每轮仍按原条件和重试预算执行；`job.loop.max_iterations` 控制优化轮数。配置允许最多 4 个同时活动的子 agent，流程要求同一时刻只推进一个阶段。这是调度容量，不是 4 条搜索策略。
- `request_input.py` 将原人工确认点转成磁盘 pending question；启动器发现后中断 CLI，等待用户回复。

原阶段流程仍由模型遵循 prompts 执行，并非 Python 固定状态机。角色分工与原 `tools:` 的要求被保留为指令；Codex 的工具权限机制与 Claude 不完全等价，不能把这种迁移当作权限隔离的逐项复刻。

## 配置、权限与超时

每个实验使用 `codex_host/home/` 作为独立 `CODEX_HOME`，配置不会写进你的日常 Codex 目录。生成的配置禁用 apps/plugins/memories/web search，固定同一 provider，避免额外模型覆盖；系统或仓库的其他适用规则仍可能生效。

| 环境变量 / 参数 | 默认与含义 |
| --- | --- |
| `MAXKERNEL_MODEL` | `gpt-5.6-sol`；也读取 `MODEL_NAME`；`--model` 优先 |
| `OPENAI_BASE_URL` | `https://agent.infiscale-tech.com/v1`，必须支持 Codex 使用的流式 Responses 与工具协议 |
| `OPENAI_API_KEY` | 仅实际执行时必填；不写入生成的配置，不传给原生 Shell 子进程 |
| `MAXKERNEL_REASONING_EFFORT` | `high` |
| `MAXKERNEL_API_TIMEOUT` | 360 秒，转换为 `stream_idle_timeout_ms=360000` |
| `MAXKERNEL_API_MAX_RETRIES` | 2，分别配置 provider 的 HTTP 请求重试和流中断重试 |
| `--sandbox` | 默认 `workspace-write`，允许原工具的网络访问 |
| `--python` | 原工具使用的解释器，默认当前 Python |
| `--tpu-version` | 新任务默认 `TPU v6e`；可指定其他硬件 |

**360 秒是流式连接无新事件时的空闲超时，不是整次请求或整轮优化的总时限。** HTTP/流重试及 CLI 的错误恢复还可能延长时间，不能按 Python SDK 的三次请求直接计算 CLI 总时长。`MAXKERNEL_MAX_OUTPUT_TOKENS=8000` 是前一个 SDK 宿主的配置；本入口不把它当作有效 CLI 参数，输出预算和上下文压缩由该 CLI/模型管理。

默认通过 `workspace-write` 沙箱执行，CLI 无交互运行设置 `approval_policy=never`，命令权限不足时返回失败；这不批准 baseline/extraction 等业务确认点。Linux 容器/Colab 若不支持 CLI 所需的沙箱，应明确报错，不会自动降级。仅在你选择的独占、可丢弃环境中，可显式传 `--sandbox danger-full-access`；该选项放开 CLI 沙箱，工具可操作整台环境。

原 Claude 的可选 workspace guard hook **未安装到 Codex**。其路径/源码读取约束不能直接由 Claude 工具名匹配到 Codex 的 Shell/补丁工具。当前依靠原 prompt 规则与 CLI sandbox；如实验要求 guard 的强制拦截等价，应先补齐独立 hook 适配，不能据此宣称等价。API key 脱敏不意味着对源码/数据脱敏，原生会话及日志仍需按实验数据保存。

TPU server 保持原生命周期，CLI 结束不代表远程/本地服务器已停止。原 emergency cleanup 可能取消服务器的全部任务，因此在独占 TPU runtime 上比较，避免与 ADK 或另一个 kernel 实验同时运行。

## 暂停、恢复与结果

运行目录下保留原 `state.json`、`iter<n>/`、`optimized.py`。新增内容在 `codex_host/`：

- `status.json`：`prepared / running / needs_input / completed / incomplete / failed / interrupted`。
- `pending_question.json`：待人工确认的问题和相关产物路径。
- `home/agents/*.toml`：Codex 原生角色配置；同目录 `.md` 是渲染后的原始 prompt。
- `home/skills/maxkernel/`、`orchestrator.md`：完整主流程与 guides。
- `invocations/<id>/`：每次执行的配置/角色快照、`events.jsonl`、`stderr.log`、`report.md`、`session.json`。
- `manifest.json`：模型、CLI 版本、超时与原 prompts 的 SHA256。

遇到人工确认时，读取问题及引用文件，将自己的决定写到文本文件，然后：

```bash
/tmp/maxkernel-codex-venv/bin/python -m codex_cli \
  --codex /tmp/maxkernel-codex-cli/node_modules/.bin/codex \
  --run-dir workspace/codex-cli-matmul-01 --resume \
  --reply-file /absolute/path/my-reply.txt
```

拒绝或要求修改同样有效，入口不会自行设置 `approved`。`--resume` 创建新的 CLI 根会话，根据保存的原请求、用户回复与磁盘状态继续，避免复用已经被中断的子 agent 树；不是从任意工具调用精确断点恢复。原生会话 ID 另外保留用于排查。重启时继续提供 API 环境、解释器与所需 sandbox 设置。

退出码：0 表示原轮数完成，1 表示失败或未完成，2 等待人工输入，130 用户中断。完成后仍需检查 `has_valid_candidate` 和实测数据；无正确候选也可能完成全部轮数。产物检查不会把模型的口头“成功”当成 TPU 测试通过。

## 验证与边界

原生离线集成测试使用真实 Codex CLI 与本机模拟 SSE API，验证两层角色分发、新上下文、6 次连续 worker 调度、真实 Shell 写文件、API key 不进入 Shell 环境、人工暂停和 API 失败后状态保留。不会调用付费模型或 TPU。

从 `maxkernel/claude-code/` 执行：

```bash
python -m pytest codex_cli/tests -q
MAXKERNEL_TEST_CODEX=/path/to/codex python -m pytest codex_cli/tests -q
```

测试环境需要 pytest、Python >= 3.11；第二条需允许 localhost 监听和执行固定版本 CLI。本次尚未实测代理接口兼容性、完整 Colab 依赖安装或 TPU 优化效果。当前上游缺失 external evaluation/specification 所需的部分 `evaluation/` 文件；相应 job 会预检查失败，自包含 JAX 示例不受该路径影响。

官方接口参考：[自定义 agents](https://developers.openai.com/codex/multi-agent)、[配置](https://developers.openai.com/codex/config-reference)、[非交互运行](https://developers.openai.com/codex/noninteractive)。
