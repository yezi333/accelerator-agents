"""The small Claude-style tool surface used by the original prompts.

Bash runs trusted model-generated commands, as in the original CLI workflow;
this module is not an OS sandbox. The original optional guard can run before tools.
"""

import asyncio
import json
import os
import signal
import uuid
from pathlib import Path


def function(name, description, properties, required=()):
  return {
    "type": "function",
    "name": name,
    "description": description,
    "strict": False,
    "parameters": {
      "type": "object",
      "properties": properties,
      "required": list(required),
      "additionalProperties": False,
    },
  }


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
BOOLEAN = {"type": "boolean"}
SCHEMAS = [
  function(
    "Read",
    "Read a UTF-8 file. offset is a 1-based line number; follow next_offset for large files.",
    {"file_path": STRING, "offset": INTEGER, "limit": INTEGER},
    ["file_path"],
  ),
  function(
    "Write",
    "Write the complete UTF-8 file, creating parent directories.",
    {"file_path": STRING, "content": STRING},
    ["file_path", "content"],
  ),
  function(
    "Edit",
    "Replace exact text; old_string must be unique unless replace_all=true.",
    {
      "file_path": STRING,
      "old_string": STRING,
      "new_string": STRING,
      "replace_all": BOOLEAN,
    },
    ["file_path", "old_string", "new_string"],
  ),
  function(
    "Glob",
    "List files matching a relative glob under an explicit path.",
    {"pattern": STRING, "path": STRING},
    ["pattern", "path"],
  ),
  function(
    "Grep",
    "Search an explicit file or directory with ripgrep; output_mode is content, files_with_matches, or count.",
    {
      "pattern": STRING,
      "path": STRING,
      "glob": STRING,
      "output_mode": STRING,
      "-i": BOOLEAN,
      "-n": BOOLEAN,
      "-A": INTEGER,
      "-B": INTEGER,
      "-C": INTEGER,
      "head_limit": INTEGER,
      "multiline": BOOLEAN,
    },
    ["pattern", "path"],
  ),
  function(
    "Bash",
    "Run a shell command synchronously in the run directory. timeout is milliseconds; no background mode.",
    {
      "command": STRING,
      "description": STRING,
      "timeout": INTEGER,
      "run_in_background": BOOLEAN,
    },
    ["command"],
  ),
  function(
    "Agent",
    "Run a registered MaxKernel subagent with a fresh isolated context; return only its final report.",
    {
      "subagent_type": STRING,
      "description": STRING,
      "prompt": STRING,
      "run_in_background": BOOLEAN,
    },
    ["subagent_type", "prompt"],
  ),
  function(
    "AskUserQuestion",
    "Pause this run for user clarification or an original baseline/extraction approval gate.",
    {
      "questions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "question": STRING,
            "header": STRING,
            "options": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {"label": STRING, "description": STRING},
                "required": ["label"],
              },
            },
            "multiSelect": BOOLEAN,
          },
          "required": ["question"],
        },
      }
    },
    ["questions"],
  ),
]


class NeedsInput(Exception):
  def __init__(self, questions, agent):
    super().__init__("Run paused for user input")
    self.questions, self.agent = questions, agent


class ToolHost:
  def __init__(
    self, run_dir, prompts, *, read_roots=(), guard=False, shell_timeout=3600
  ):
    self.run_dir = Path(run_dir).resolve()
    self.prompts = prompts
    self.read_roots = [
      self.run_dir,
      prompts.directory,
      Path(__file__).resolve().parents[1],
    ]
    self.read_roots += [Path(p).resolve() for p in read_roots]
    self.guard = guard
    self.shell_timeout = shell_timeout
    self.counter = 0
    self.invocation = uuid.uuid4().hex[:12]
    self.logs = self.run_dir / "openai_host" / "tools"
    self.logs.mkdir(parents=True, exist_ok=True)
    self.secrets = [
      os.environ[k] for k in ("OPENAI_API_KEY",) if os.environ.get(k)
    ]

  def redact(self, value):
    text = str(value)
    for secret in self.secrets:
      text = text.replace(secret, "[REDACTED]")
    return text

  def path(self, value, *, write=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
      path = self.run_dir / path
    path = path.resolve()
    roots = [self.run_dir] if write else self.read_roots
    if not any(path == root or path.is_relative_to(root) for root in roots):
      raise ValueError(
        f"Path outside {'run directory' if write else 'declared read roots'}: {path}"
      )
    return path

  def child_env(self):
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env["PATH"] = (
      str(self.prompts.python.parent) + os.pathsep + env.get("PATH", "")
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env

  async def process(self, argv, *, timeout, stdin=None):
    self.counter += 1
    log = self.logs / f"{self.invocation}_{self.counter:05d}.log"
    # Temporary raw output is rewritten with credential redaction before reading.
    with log.open("w") as output:
      process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=self.run_dir,
        env=self.child_env(),
        start_new_session=True,
        stdin=asyncio.subprocess.PIPE
        if stdin is not None
        else asyncio.subprocess.DEVNULL,
        stdout=output,
        stderr=asyncio.subprocess.STDOUT,
      )
      try:
        if stdin is not None:
          process.stdin.write(stdin.encode())
          await process.stdin.drain()
          process.stdin.close()
        await asyncio.wait_for(process.wait(), timeout)
        timed_out = False
      except TimeoutError:
        timed_out = True
      finally:
        if process.returncode is None:
          try:
            os.killpg(process.pid, signal.SIGTERM)
            await asyncio.wait_for(process.wait(), 5)
          except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
          except ProcessLookupError:
            await process.wait()
    text = self.redact(log.read_text(errors="replace"))
    log.write_text(text)
    return {
      "exit_code": process.returncode,
      "timed_out": timed_out,
      "output": text[:24000],
      "truncated": len(text) > 24000,
      "output_file": str(log),
    }

  async def check_guard(self, name, args):
    if self.guard:
      result = await self.process(
        [
          str(self.prompts.python),
          str(self.prompts.directory / "hooks/workspace-guard.py"),
        ],
        timeout=30,
        stdin=json.dumps(
          {"tool_name": name, "tool_input": args, "cwd": str(self.run_dir)}
        ),
      )
      if result["exit_code"] != 0 or result["timed_out"]:
        raise ValueError(
          "Original workspace guard rejected tool: " + result["output"]
        )
      if result["output"].strip():
        decision = json.loads(result["output"]).get("hookSpecificOutput", {})
        if decision.get("permissionDecision") in {"deny", "ask"}:
          raise ValueError(
            "Original workspace guard rejected tool: "
            + decision.get("permissionDecisionReason", "Permission denied")
          )

  async def execute(self, name, args):
    if name == "Read":
      path = self.path(args["file_path"])
      lines = path.read_text().splitlines(keepends=True)
      start = int(args.get("offset", 1)) - 1
      limit = int(args.get("limit", 200))
      if start < 0 or limit < 1:
        raise ValueError("offset and limit must be positive")
      chosen = lines[start : start + min(limit, 2000)]
      text = "".join(
        f"{i + start + 1}: {line}" for i, line in enumerate(chosen)
      )
      if len(text) > 48000:
        # Do not silently skip the remainder of a long line.
        return {
          "error": "Selected lines exceed 48000 characters; use a smaller limit or Bash to inspect the file."
        }
      return {
        "content": self.redact(text),
        "total_lines": len(lines),
        "next_offset": start + len(chosen) + 1
        if start + len(chosen) < len(lines)
        else None,
      }
    if name in {"Write", "Edit"}:
      path = self.path(args["file_path"], write=True)
      if name == "Write":
        content = args["content"]
      else:
        content = path.read_text()
        old = args["old_string"]
        count = content.count(old)
        if (
          not old
          or count == 0
          or (count > 1 and not args.get("replace_all", False))
        ):
          raise ValueError(
            "old_string must exist and be unique unless replace_all=true"
          )
        content = content.replace(
          old, args["new_string"], -1 if args.get("replace_all") else 1
        )
      path.parent.mkdir(parents=True, exist_ok=True)
      temporary = path.with_name(path.name + ".openai-tmp")
      temporary.write_text(content)
      temporary.replace(path)
      return {"written": str(path)}
    if name == "Glob":
      path = self.path(args["path"])
      if (
        Path(args["pattern"]).is_absolute()
        or ".." in Path(args["pattern"]).parts
      ):
        raise ValueError("Use a relative glob without parent traversal")
      matches = [
        str(self.path(p))
        for p in sorted(path.glob(args["pattern"]))
        if p.is_file()
      ]
      return {"files": matches[:500], "truncated": len(matches) > 500}
    if name == "Grep":
      path = self.path(args["path"])
      command = ["rg", "--color=never"]
      mode = args.get("output_mode", "files_with_matches")
      command += {
        "content": ["-n"],
        "files_with_matches": ["-l"],
        "count": ["-c"],
      }[mode]
      for flag in ("-i", "multiline"):
        if args.get(flag):
          command.append("-U" if flag == "multiline" else flag)
      for flag in ("-A", "-B", "-C"):
        if flag in args:
          command += [flag, str(int(args[flag]))]
      if args.get("glob"):
        command += ["--glob", args["glob"]]
      command += ["--", args["pattern"], str(path)]
      result = await self.process(command, timeout=30)
      if args.get("head_limit", 0) > 0:
        result["output"] = "\n".join(
          result["output"].splitlines()[: args["head_limit"]]
        )
      return result
    if name == "Bash":
      if args.get("run_in_background"):
        raise ValueError(
          "Background Bash is unsupported; use synchronous commands"
        )
      timeout = float(args.get("timeout", self.shell_timeout * 1000)) / 1000
      if timeout <= 0:
        raise ValueError("timeout must be positive")
      return await self.process(
        ["/bin/bash", "-c", args["command"]], timeout=timeout
      )
    raise ValueError(f"Unknown tool: {name}")
