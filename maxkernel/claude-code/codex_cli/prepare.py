"""Render original role instructions into Codex's native agent configuration."""

import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from urllib.parse import urlsplit

from openai_runtime.prompts import ROOT, body, load_installer

CLI_VERSION = "0.171.0-alpha.13"


def quote(value):
  # JSON strings are also valid TOML basic strings, including escaped newlines.
  return json.dumps(str(value), ensure_ascii=False)


def adaptation(run_dir, python, home, role):
  gate = ROOT / "codex_cli/request_input.py"
  gate_command = shlex.join([str(python), str(gate), "--run-dir", str(run_dir)])
  return f"""# Native Codex adapter — registered role: {role}
Execute the complete original MaxKernel workflow below. Preserve every applicable
phase, validation gate, retry budget, autotuning sweep and profile analysis.
The user explicitly authorizes the original orchestrator -> worker -> specialist
agent delegation. Use only these registered roles for this pipeline.

The original document uses Claude Code tool names. Translate their meaning:
- Agent(subagent_type=X, prompt=P) means native collaboration.spawn_agent with
  agent_type=X, fork_turns="none", a unique task_name, and message=P. Explicitly
  select that registered role; never substitute a generic worker or explorer.
  Use collaboration.wait_agent/list_agents and its final report to wait for it.
  Only one phase is active at a time; never start the next before it completes.
  The orchestrator dispatches maxkernel-worker, which dispatches specialists.
  Specialists do not delegate. Worker coordination overrides general_rules #7's
  prohibition on delegation, which applies to the specialists.
  Spawn a fresh worker for each iteration and a fresh specialist per dispatch,
  with fork_turns="none", just as in the original pipeline. A completed child
  frees an active slot. Do not flatten roles into the orchestrator conversation.
- Bash means the native shell execution tool, including functions.exec wrapping
  tools.exec_command when exposed. Read/Grep/Glob use shell reads and rg;
  Write/Edit use native apply_patch or exact file writes. Follow actual exposed
  schemas; do not try to call unavailable Claude tools. Follow running commands
  through their session IDs until completion; do not treat yield as success.
- Read registered original role Markdown at {home}/agents/<role>.md when needed.
  Legacy {ROOT}/subagents references mean that same rendered directory.

Workload files and knowledge: {ROOT}. Run directory: {run_dir} (already exists).
Use {python} for original tools. Commands start at {ROOT}; use absolute paths.
Do not install dependencies or change source prompts/tools during a run. Keep
artifacts inside the run directory, except original TPU config/server caches.
The job.loop.max_iterations / state.max_iterations is authoritative; prose 5 is
only the original default. Verify artifacts on disk before claiming completion.
All numerical results must come from original tools/TPU, never model estimates.

Human gates apply even in non-interactive exec. If ANY role needs user input or
approval (including synthesis/extraction), call the following Python helper via
shell, passing the actual question and artifact paths, then end the role's turn:
{gate_command} --question "<question>" --artifact "<path>"
The launcher stops the CLI when pending_question.json appears. Do not continue,
self-approve, launch another phase, or count this as an optimization failure.
AskUserQuestion in the original text means this helper; it is not a native exec
tool. On resume, only the explicit user reply in the task authorizes a decision.
Original approval flags must remain unset unless that reply actually approves.

--- Full original role instructions (after path rendering) ---
"""


def prepare(run_dir, python, *, sandbox, model=None):
  run_dir, python = Path(run_dir).resolve(), Path(python).absolute()
  host = run_dir / "codex_host"
  home = host / "home"
  home.mkdir(parents=True, exist_ok=True)
  installer = load_installer()
  variables = installer.build_vars(home, python.parent.parent)
  variables["VENV_PYTHON"] = str(python)
  manifest = {}
  agents = []
  sources = [
    (p, Path("agents") / p.name)
    for p in sorted((ROOT / "agents").glob("maxkernel-*.md"))
  ]
  sources += [
    (p, Path("skills/maxkernel") / p.relative_to(ROOT / "skill"))
    for p in sorted((ROOT / "skill").rglob("*.md"))
  ]
  for source, relative in sources:
    original = source.read_text()
    rendered = installer.render(original, variables, source)
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    manifest[str(source.relative_to(ROOT))] = hashlib.sha256(
      original.encode()
    ).hexdigest()
    if relative.parts[0] == "agents":
      name = source.stem
      description = re.search(r"^description: (.+)$", original, re.M)[1]
      rules = installer.render(
        (ROOT / "skill/general_rules.md").read_text(), variables, source
      )
      instructions = (
        adaptation(run_dir, python, home, name)
        + rules
        + "\n\n"
        + body(rendered)
      )
      target.with_suffix(".toml").write_text(
        f"name = {quote(name)}\ndescription = {quote(description)}\ndeveloper_instructions = {quote(instructions)}\n"
      )
      agents.append(name)
  root_instructions = adaptation(run_dir, python, home, "maxkernel") + body(
    (home / "skills/maxkernel/SKILL.md").read_text()
  )
  (host / "orchestrator.md").write_text(root_instructions)

  model = (
    model
    or os.environ.get("MAXKERNEL_MODEL")
    or os.environ.get("MODEL_NAME", "gpt-5.6-sol")
  )
  base_url = os.environ.get(
    "OPENAI_BASE_URL", "https://agent.infiscale-tech.com/v1"
  ).rstrip("/")
  if base_url.endswith("/responses"):
    base_url = base_url[: -len("/responses")]
  url = urlsplit(base_url)
  if (
    url.scheme not in {"http", "https"}
    or not url.hostname
    or url.username
    or url.password
    or url.query
    or url.fragment
    or url.path.endswith("/chat/completions")
  ):
    raise ValueError(
      "OPENAI_BASE_URL must be a Responses base URL without credentials/query"
    )
  idle_ms = int(float(os.environ.get("MAXKERNEL_API_TIMEOUT", "360")) * 1000)
  retries = int(os.environ.get("MAXKERNEL_API_MAX_RETRIES", "2"))
  if idle_ms <= 0 or retries < 0:
    raise ValueError(
      "API timeout must be positive; retries must be nonnegative"
    )
  effort = os.environ.get("MAXKERNEL_REASONING_EFFORT", "high")
  config = f"""# Generated per run; no credentials are written here.
model = {quote(model)}
model_provider = "maxkernel"
model_reasoning_effort = {quote(effort)}
approval_policy = "never"
sandbox_mode = {quote(sandbox)}
developer_instructions = {quote(root_instructions)}
web_search = "disabled"

[model_providers.maxkernel]
name = "MaxKernel OpenAI-compatible API"
base_url = {quote(base_url)}
env_key = "OPENAI_API_KEY"
wire_api = "responses"
request_max_retries = {retries}
stream_max_retries = {retries}
stream_idle_timeout_ms = {idle_ms}
supports_websockets = false

[agents]
enabled = true
max_concurrent_threads_per_session = 4

[sandbox_workspace_write]
network_access = true

[shell_environment_policy]
exclude = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

[features]
apps = false
plugins = false
memories = false
shell_snapshot = false

[projects.{quote(ROOT)}]
trust_level = "trusted"
"""
  (home / "config.toml").write_text(config)
  metadata = {
    "cli_version": CLI_VERSION,
    "model": model,
    "base_url": base_url,
    "stream_idle_timeout_ms": idle_ms,
    "request_max_retries": retries,
    "stream_max_retries": retries,
    "sandbox": sandbox,
    "python": str(python),
    "agents": agents,
    "prompt_sha256": manifest,
  }
  (host / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
  return home, metadata
