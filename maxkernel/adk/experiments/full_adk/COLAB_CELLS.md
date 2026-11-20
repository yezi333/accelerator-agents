# 在现有 Colab runtime 中运行完整原生 ADK pipeline

这份文档提供追加/替换 cells，不需要上传新 notebook，也不需要重启 runtime。继续使用已验证的 `gpt-5.6-sol`、代理 URL 和对应 key。不要在 notebook 主进程初始化 JAX/TPU。

保留原 Cell 1/2 的模型配置和已成功安装 PallasKernels 的 Cell 3。停止旧实验后先拉取分支，再执行下面 A、C、D；API 连通性测试可保留，原 Cell 4/5 和修改 `STRUCTURE_ROUNDS` 的 cell 不再执行。原 Cell 6 可以继续下载，因为这里仍设置 `RUN_DIR`。

## 运行范围

- 直接调用原 `create_root_agent()` / `AutoAgentClient`，原 `AutonomousPipelineAgent`、子 agent、prompts、测试生成、autotune 搜索、profile 分析、最优候选选择代码不变。
- 原始流程：准备 baseline → 生成/验证测试 → 每轮规划、实现、编译与修复、正确性测试、autotune、profiling/分析 → 选择结果。编译或测试失败的轮次仍按原逻辑提前返回规划。
- 本输入适配针对当前 `gqa_decode`：保留整个原 Pallas 实现，补 `computation` 入口和来自原 registry 的 `get_inputs`，提供已有源码作为初始候选。不是从头生成另一个 kernel。该完整流程由原 agent 生成测试，正确性/速度比较对象是原 Pallas baseline；不会偷偷沿用简化实验的固定 JAX oracle。
- 三个服务使用原 `cpu_server`、`tpu_server`、`eval_server`；只有 TPU 服务派发的子进程使用 TPU。启动器只管理输入、日志、端口和生命周期。
- `FULL_ITERATIONS` 通过原 factory 的 `max_iterations` 生效，默认 5。旧 `TPU_JOBS`、`MODEL_CALLS`、`WALL_SECONDS`、`STRUCTURE_ROUNDS` 不适用。原生流程没有简化版本的全局硬预算，实际调用和调参次数由原流程决定。
- `MAXKERNEL_API_TIMEOUT`、`MAXKERNEL_MAX_OUTPUT_TOKENS`、`MAXKERNEL_REASONING_EFFORT` 现在走通用模型适配器的配置，不再被简化版 `ask()` 覆盖。
- Vertex RAG 是原流程的可选服务，本配置不启用它，未替换为另一套检索工具。原 API 查询、MCP 文件工具、XPlane 查询和分析工具仍保留。原 `get_hlo_dump` 是占位实现，本启动器不会把它描述为已完整实现。
- 原生测试和 autotune 有 host 计时；每轮 profiling 并不意味着原版的最终选择自动变成 device-only。保留原始逻辑，不沿用简化版的 3%/MAD 晋级规则。
- 使用新 `FULL_RUN_NAME`。原生 pipeline 不是阶段级断点恢复入口；本启动器不把旧实验 `state.json` 伪装成可恢复会话。

## 更新 cell：在已完成的 Cell 3 后新增并执行一次

此 cell 只同步 accelerator-agents，不重新安装 PallasKernels，不重启 runtime。
使用快进合并保留本地文件；如出现冲突会停止，不会自动清除本地修改。
原 Cell 3 可能固定旧 commit，完成更新后不要再执行它。

```python
#@title 3.1 更新完整 ADK 启动器
import subprocess
from pathlib import Path

REPO = Path("/content/maxkernel-colab/accelerator-agents")
subprocess.run([
    "git", "-C", str(REPO), "fetch",
    "https://github.com/yezi333/accelerator-agents.git",
    "refs/heads/codex/adk-openai-colab",
], check=True)
subprocess.run(["git", "-C", str(REPO), "merge", "--ff-only", "FETCH_HEAD"], check=True)
subprocess.run(["git", "-C", str(REPO), "log", "-1", "--oneline"], check=True)
assert (REPO / "maxkernel/adk/experiments/full_adk/run.py").is_file()
```

拉取代码后仍需要 Cell A 补齐原版 agent 的依赖，并把旧 Cell 4 换为 Cell C。
只更新代码而保留旧入口，仍然执行 `experiments.colab.run` 简化实验。
原 Cell B（粘贴启动脚本）已经不需要；Cell C 直接运行仓库中的模块。

## Cell A：补充配置和依赖

原 Cell 3 的 `run_live()` 必须已经定义。此 cell 不调用模型，也不运行 TPU 测试。

```python
#@title A. 完整原生 ADK：配置与补装依赖（原 Cell 3 成功后执行）
FULL_ITERATIONS = 5 #@param {type:"integer"}
FULL_RUN_NAME = "gqa-decode-native-01" #@param {type:"string"}

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/maxkernel-colab")
ADK = ROOT / "accelerator-agents/maxkernel/adk"
PALLAS_REPO = ROOT / "PallasKernels"
ENV_DIR = ROOT / "env"
PYTHON = ENV_DIR / "bin/python"
FULL_WORKDIR = ROOT / "full_adk_results"
RUN_DIR = FULL_WORKDIR / FULL_RUN_NAME

if FULL_ITERATIONS < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", FULL_RUN_NAME):
    raise ValueError("检查轮数和实验名称。")
if RUN_DIR.exists():
    raise RuntimeError("该原生实验目录已存在，请使用新的 FULL_RUN_NAME。")
if not PYTHON.is_file() or not (PALLAS_REPO / "pyproject.toml").is_file():
    raise RuntimeError("请先运行原 Cell 3，完成 Python 环境和 PallasKernels 安装。")
if not all(os.environ.get(k) for k in ("OPENAI_API_KEY", "MAXKERNEL_MODEL", "OPENAI_BASE_URL")):
    raise RuntimeError("先运行 Cell 1/2 设置已经验证成功的模型、URL 和 key。")

# 在子进程中设置 provider；Notebook 主进程不 import ADK/JAX。
os.environ["MAXKERNEL_LLM_PROVIDER"] = "openai"
os.environ["MAXKERNEL_ADK_DIR"] = str(ADK)
os.environ["WORKDIR"] = str(FULL_WORKDIR)
os.environ["TPU_VERSION"] = "TPU v6e"
os.environ.pop("RAG_CORPUS", None)  # 原流程的可选 Vertex RAG 未配置。

# 用原 agent_requirements，补上当前 JAX 场景需要的工具依赖。
constraints = ROOT / "full_adk_constraints.txt"
constraints.write_text(
    "jax==0.8.1\njaxlib==0.8.1\ntensorflow==2.20.0\nxprof==2.22.1\n"
    "google-adk==1.39.1\nopenai==2.54.0\n"
)
install_env = os.environ.copy()
install_env.pop("OPENAI_API_KEY", None)
install_env.pop("GITHUB_TOKEN", None)
run_live([
    sys.executable, "-m", "uv", "pip", "install", "--python", str(PYTHON),
    "-c", str(constraints),
    "-r", str(ADK / "dependency/agent_requirements.txt"),
    "-r", str(ADK / "dependency/openai_requirements.txt"),
    "tensorflow==2.20.0", "xprof==2.22.1", "matplotlib", "pandas",
    "docstring-parser", "tabulate",
], env=install_env)

# 原 MCP 文件工具需要 npx；没有时只装到独立环境。
full_path = str(ENV_DIR / "bin") + os.pathsep + os.environ["PATH"]
if not shutil.which("npx", path=full_path):
    run_live([
        sys.executable, "-m", "uv", "pip", "install", "--python", str(PYTHON),
        "nodejs-wheel>=20,<25",
    ], env=install_env)

full_env = os.environ.copy()
full_env["PATH"] = full_path
full_env["PYTHONPATH"] = str(ADK) + os.pathsep + str(PALLAS_REPO)
full_env["PYTHONUNBUFFERED"] = "1"
full_env["JAX_PLATFORMS"] = "cpu"  # 仅 agent/协调进程用 CPU。
full_env.pop("JAX_PLATFORM_NAME", None)
full_env.pop("GITHUB_TOKEN", None)
FULL_WORKDIR.mkdir(parents=True, exist_ok=True)

run_live([sys.executable, "-m", "uv", "pip", "check", "--python", str(PYTHON)], env=install_env)
# 在 ADK 启动前预下载原版 MCP 包，避免首次下载超过握手时限。
run_live([
    "npx", "-y", "--package=@modelcontextprotocol/server-filesystem@0.5.1",
    "node", "-e", "console.log('MCP filesystem dependency ready')",
], env=full_env)
# CPU-only 导入检查：提前发现原生 profiling 依赖缺失。
run_live([
    str(PYTHON), "-c",
    "from auto_agent.agent import create_root_agent; "
    "from xprof.convert import raw_to_tool_data; "
    "print('Original ADK and XProf imports ready')",
], env=full_env)
print("完整原生流程配置完成。原 experiments.colab 的次数预算和 rounds 配置不会参与此运行。")
```

## Cell C：启动原生 pipeline

```python
#@title C. 启动完整原生 ADK pipeline（替代原 Cell 4）
import os
import signal
import subprocess

# 不使用旧 run_experiment 的 SIGKILL 中断方式，给服务清理留时间。
def run_native(command):
    with subprocess.Popen(
        command, env=full_env, cwd=ADK,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            # 清理该驱动进程组可能残留的 MCP/npx 子进程。
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if code:
            raise RuntimeError(f"完整 ADK 运行失败（{code}）；查看上方日志和 {RUN_DIR}/services/*.log。")

run_native([
    str(PYTHON), "-m", "experiments.full_adk.run",
    "--repo", str(PALLAS_REPO),
    "--session-dir", str(RUN_DIR),
    "--iterations", str(FULL_ITERATIONS),
])
print("原生 pipeline 调用结束；请在下一 cell 检查每轮实际完成的阶段。")
```

## Cell D：查看原生结果

```python
#@title D. 查看完整原生流程结果（替代原 Cell 5）
import json

path = RUN_DIR / "native_summary.json"
if not path.exists():
    print("尚无原生结果，请检查运行错误以及 services/*.log。")
else:
    data = json.loads(path.read_text())
    print("Pipeline 状态：", data["pipeline_status"])
    print("选中 iteration（从 0 开始）：", data["best_iteration"])
    for row in data["iterations"]:
        print(f"\n===== iteration {row['iteration']} =====")
        print("编译：", (row["compilation_status"] or {}).get("success"))
        print("正确性：", (row["test_status"] or {}).get("success"))
        print("原生 harness speedup：", row["speedup"])
        print("Autotuning：", row["autotuning_summary"])
        print("Profiling：", row["profiling_summary"])
    print("\n优化文件：", RUN_DIR / "optimized_kernel.py")
    print("会话和工具调用：session.json / events.jsonl")
    print("注意：原生 speedup 不自动等于 XProf 设备时间加速。")
```

原 Cell 6 可继续使用，将新 `RUN_DIR` 打包下载。没有 `device_gain_confirmed` 字段是正常的：它属于之前的简化实验，不属于原生结果格式。


## 本地验证范围

- 25 项相关测试通过，包括原生 pipeline 两轮阶段顺序、上一轮 profiling 结果进入下一轮规划。
- 原 factory 的各子 agent 可实例化；实际 PallasKernels 输入适配和输入形状在 CPU 上验证通过。
- 原 MCP 文件工具完成初始化和真实文件读取；安装 cell 预下载 MCP 依赖，避免冷启动下载超过握手时限。
- 原服务的 eval → CPU 执行请求和退出清理验证通过。macOS 缺少 XProf 原生扩展，此项服务测试对未使用的 profile 导入作了测试替身处理，不代表已验证 profiling 执行。
- 尚未在真实 Colab TPU 上执行完整的模型工具调用、autotune 与 XProf 闭环。API 文本连通性通过也不能替代这项端到端验证。
