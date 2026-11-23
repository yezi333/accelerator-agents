"""Render the original prompts without installing into a user's Claude config."""

import importlib.util
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASIC_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep", "Bash"}


def load_installer():
  spec = importlib.util.spec_from_file_location(
    "maxkernel_installer", ROOT / "install.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def body(text):
  return text.split("---", 2)[2].lstrip() if text.startswith("---\n") else text


class Prompts:
  def __init__(self, run_dir, python):
    self.run_dir = Path(run_dir).resolve()
    self.directory = (
      self.run_dir / "openai_host" / "prompts" / uuid.uuid4().hex[:12]
    )
    self.python = Path(python).absolute()
    installer = load_installer()
    variables = installer.build_vars(self.directory, self.python.parent.parent)
    variables["VENV_PYTHON"] = str(self.python)
    self.agents = {}
    self.allowed = {}
    sources = [
      (p, Path("agents") / p.name)
      for p in sorted((ROOT / "agents").glob("maxkernel-*.md"))
    ]
    sources += [
      (p, Path("skills/maxkernel") / p.relative_to(ROOT / "skill"))
      for p in sorted((ROOT / "skill").rglob("*.md"))
    ]
    sources += [
      (ROOT / "hooks/workspace-guard.py", Path("hooks/workspace-guard.py"))
    ]
    self.manifest = {}
    import hashlib

    for source, relative in sources:
      original = source.read_text()
      rendered = installer.render(original, variables, source)
      destination = self.directory / relative
      destination.parent.mkdir(parents=True, exist_ok=True)
      destination.write_text(rendered)
      self.manifest[str(source.relative_to(ROOT))] = hashlib.sha256(
        original.encode()
      ).hexdigest()
      if relative.parts[0] == "agents":
        self.agents[source.stem] = body(rendered)
        match = re.search(r"^tools: (.+)$", original.split("---", 2)[1], re.M)
        self.allowed[source.stem] = (
          set(match[1].split(", ")) if match else BASIC_TOOLS.copy()
        )
    self.agents["maxkernel"] = body(
      (self.directory / "skills/maxkernel/SKILL.md").read_text()
    )
    self.allowed["maxkernel"] = BASIC_TOOLS.copy()
    self.allowed["maxkernel"].add("Agent")
    self.allowed["maxkernel-worker"].add("Agent")
    for name in self.allowed:
      self.allowed[name].add("AskUserQuestion")
    self.rules = (
      self.directory / "skills/maxkernel/general_rules.md"
    ).read_text()

  def instructions(self, name):
    return f"""You are running the original MaxKernel pipeline through an OpenAI Responses host.
Registered role: {name}
The full original role prompt follows. Preserve its phases, gates, retry loops,
workload, tolerances, deterministic tools, autotuning and profiling requirements.
The API model name is selected by the host; `model: inherit` means this model.
Use the provided function tools with the same names as Claude Code's tools.
Agent calls are synchronous and use a fresh isolated conversation. Do not invent
background agents, TaskOutput, or alternate model subprocesses. The orchestrator
may dispatch only maxkernel-worker; the worker dispatches the phase specialists;
specialists cannot delegate. Its explicit coordination duties take precedence over
general_rules #7, which applies to specialists.
All generated artifacts belong to {self.run_dir}. Bash starts in that directory;
commands do not retain working directory or shell variables across calls. Shell
timeouts are milliseconds. Long output is saved to a file; Read it in chunks.
The rendered prompts are registered under {self.directory}/agents. If a legacy
example mentions {ROOT}/subagents, use the registered prompt directory instead.
Use {self.python} for the original Python tools. Do not install packages in a run.
If you need clarification or any original human approval gate, call AskUserQuestion.
The host pauses the whole run and records the question; never approve on silence.
The initial run directory already exists; use it rather than making another run.
The authoritative loop limit is state.max_iterations / the user's job.loop.max_iterations;
literal 5s in the original prose are its default, not an override of a declared limit.
Verify artifacts and state on disk before claiming completion. A prose summary
is not evidence of successful compilation, correctness, autotuning or profiling.

{self.rules if name != "maxkernel" else ""}

--- Original role prompt ---
{self.agents[name]}
"""
