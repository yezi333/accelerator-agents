"""Run the original MaxKernel pipeline through native codex exec."""

import argparse
import fcntl
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

# Reuse only the existing pure-Python checks; no ResponsesHost or SDK is started.
from openai_runtime.__main__ import inspect_result, validate_job
from openai_runtime.runtime import save_json

from .prepare import CLI_VERSION, ROOT, prepare


def arguments(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--job", type=Path)
  source.add_argument("--source", type=Path)
  parser.add_argument("--run-dir", required=True, type=Path)
  parser.add_argument("--python", type=Path, default=Path(sys.executable))
  parser.add_argument(
    "--codex", default="codex", help=f"Codex CLI {CLI_VERSION} executable"
  )
  parser.add_argument(
    "--model", help="MAXKERNEL_MODEL / MODEL_NAME, default gpt-5.6-sol"
  )
  parser.add_argument(
    "--sandbox",
    choices=["workspace-write", "danger-full-access"],
    default="workspace-write",
  )
  parser.add_argument("--tpu-version", default="TPU v6e")
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Render roles and verify CLI tool registration; no API/TPU calls",
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Start a fresh native root context using the saved task and on-disk state",
  )
  parser.add_argument(
    "--reply-file",
    type=Path,
    help="Explicit user response to pending_question.json",
  )
  return parser.parse_args(argv)


def stop(process):
  if process.poll() is not None:
    return
  try:
    os.killpg(process.pid, signal.SIGINT)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=20)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def redact(text):
  key = os.environ.get("OPENAI_API_KEY")
  return text.replace(key, "[REDACTED]") if key else text


def run_cli(executable, home, host, task):
  invocation = uuid.uuid4().hex[:12]
  log_dir = host / "invocations" / invocation
  log_dir.mkdir(parents=True)
  shutil.copyfile(home / "config.toml", log_dir / "config.toml")
  shutil.copyfile(host / "manifest.json", log_dir / "manifest.json")
  for name in ("agents", "skills"):
    shutil.copytree(home / name, log_dir / name)
  env = os.environ.copy()
  env["CODEX_HOME"] = str(home)
  env["PATH"] = (
    str(Path(json.loads((host / "manifest.json").read_text())["python"]).parent)
    + os.pathsep
    + env.get("PATH", "")
  )
  report = log_dir / "report.md"
  cmd = [
    executable,
    "--no-daemon",
    "--strict-config",
    "exec",
    "--json",
    "--output-last-message",
    str(report),
    "-",
  ]
  save_json(log_dir / "launch.json", {"command": cmd, "cwd": str(ROOT)})
  save_json(
    host / "status.json", {"status": "running", "invocation": invocation}
  )
  process = subprocess.Popen(
    cmd,
    cwd=ROOT,
    env=env,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True,
  )
  messages = queue.Queue()

  def drain(stream, name):
    try:
      for line in stream:
        messages.put((name, line))
    finally:
      stream.close()
      messages.put((name, None))

  threads = [
    threading.Thread(target=drain, args=(stream, name), daemon=True)
    for stream, name in [(process.stdout, "events"), (process.stderr, "stderr")]
  ]
  for thread in threads:
    thread.start()
  interrupted, closed = False, set()
  try:
    try:
      process.stdin.write(task)
      process.stdin.close()
    except BrokenPipeError:
      pass
    with (
      (log_dir / "events.jsonl").open("w") as events,
      (log_dir / "stderr.log").open("w") as errors,
    ):
      while len(closed) < 2 or process.poll() is None:
        if (host / "pending_question.json").exists() and process.poll() is None:
          stop(process)
        try:
          name, line = messages.get(timeout=0.1)
        except queue.Empty:
          continue
        if line is None:
          closed.add(name)
          continue
        line = redact(line)
        output = events if name == "events" else errors
        output.write(line)
        output.flush()
        if name == "stderr":
          print(line, end="", file=sys.stderr, flush=True)
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError:
          print(line, end="", flush=True)
          continue
        if event.get("type") == "thread.started":
          save_json(
            log_dir / "session.json", {"thread_id": event.get("thread_id")}
          )
        item = event.get("item", {})
        print(
          f"[codex] {event.get('type', 'event')} {item.get('type', '')}",
          flush=True,
        )
        if item.get("type") == "agent_message" and item.get("text"):
          print(item["text"], flush=True)
    process.wait()
  except KeyboardInterrupt:
    interrupted = True
  finally:
    stop(process)
    for thread in threads:
      thread.join(timeout=2)
    if report.exists():
      report.write_text(redact(report.read_text()))
  if (host / "pending_question.json").exists():
    result, code = (
      {
        "status": "needs_input",
        "question_file": str(host / "pending_question.json"),
      },
      2,
    )
  elif interrupted:
    result, code = {"status": "interrupted"}, 130
  elif process.returncode:
    result, code = {"status": "failed", "cli_exit_code": process.returncode}, 1
  else:
    try:
      result = inspect_result(host.parent)
      code = 0 if result["status"] == "completed" else 1
    except (ValueError, KeyError, TypeError) as exc:
      result, code = {"status": "failed", "error": str(exc)}, 1
  result.update(
    invocation=invocation,
    report=str(report),
    events=str(log_dir / "events.jsonl"),
  )
  save_json(host / "status.json", result)
  print(json.dumps(result, ensure_ascii=False, indent=2))
  return code


def check_cli(executable, home, host, roles):
  version = subprocess.check_output(
    [executable, "--version"], text=True, stderr=subprocess.STDOUT
  )
  if f"codex-cli {CLI_VERSION}" not in version.splitlines():
    raise ValueError(
      f"Install @openai/codex@{CLI_VERSION}; found: {version.strip()}"
    )
  env = {**os.environ, "CODEX_HOME": str(home)}
  result = subprocess.run(
    [executable, "debug", "prompt-input", "MaxKernel registration check only"],
    cwd=ROOT,
    env=env,
    capture_output=True,
    text=True,
    check=True,
  )
  data = json.loads(result.stdout)
  serialized = json.dumps(data)
  if (
    not all(role in serialized for role in roles)
    or '"agent_type"' not in serialized
    or '"spawn_agent"' not in serialized
  ):
    raise ValueError(
      "CLI did not register all MaxKernel roles / native delegation tools"
    )
  save_json(
    host / "cli_check.json",
    {
      "cli_version": CLI_VERSION,
      "registered_roles": roles,
      "native_spawn_agent": True,
    },
  )


def main(argv=None):
  args = arguments(argv)
  try:
    run = args.run_dir.expanduser().resolve()
    if run == ROOT / "workspace" or not run.is_relative_to(ROOT / "workspace"):
      raise ValueError("--run-dir must be inside " + str(ROOT / "workspace"))
    python = args.python.expanduser().absolute()
    if not python.is_file():
      raise ValueError("Pipeline interpreter not found")
    executable = shutil.which(args.codex)
    if not executable:
      raise ValueError(
        f"Install @openai/codex@{CLI_VERSION}, or pass --codex /path/to/codex"
      )
    host = run / "codex_host"
    if args.reply_file and not args.resume:
      raise ValueError("--reply-file requires --resume")
    if args.resume:
      if args.job or args.source:
        raise ValueError(
          "Resume uses the saved request; do not replace the input"
        )
      if not (host / "request.json").is_file():
        raise ValueError("No saved CLI request to resume")
    else:
      source = args.job or args.source
      if not source:
        raise ValueError("Provide --job or --source")
      source = source.expanduser().resolve()
      if not source.is_file():
        raise ValueError("Input is not a file: " + str(source))
      if args.job:
        validate_job(source)
      run.mkdir(parents=True, exist_ok=False)
      host.mkdir()
    with (host / "run.lock").open("w") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      pending = host / "pending_question.json"
      reply = None
      if args.resume:
        request = json.loads((host / "request.json").read_text())
        if pending.exists() and not args.reply_file:
          raise ValueError(
            "Read pending_question.json and resume with --reply-file"
          )
        if args.reply_file:
          if not pending.exists():
            raise ValueError("There is no pending question")
          reply = args.reply_file.expanduser().read_text().strip()
          if not reply:
            raise ValueError("Reply must not be empty")
          if args.dry_run:
            raise ValueError(
              "Do not consume an approval reply during --dry-run"
            )
      else:
        request = {
          "task": f"Run the full original MaxKernel pipeline in {run}.\n{'Job' if args.job else 'Source'}: {source}\nResolve relative job paths against {source.parent}; store absolute paths in state.json.\nTarget: {args.tpu_version}. Preserve workload, tolerance, all phase gates and retry budgets.\n",
          "source": str(source),
          "kind": "job" if args.job else "source",
        }
        save_json(host / "request.json", request)
        if args.job:
          _, validation = validate_job(source)
          save_json(host / "job_validation.json", validation)
      home, metadata = prepare(
        run, python, sandbox=args.sandbox, model=args.model
      )
      check_cli(executable, home, host, metadata["agents"])
      if args.dry_run:
        print(
          f"Prepared {len(metadata['agents'])} original roles plus orchestrator; native CLI registration verified. No API/TPU requests."
        )
        if not (host / "status.json").exists():
          save_json(host / "status.json", {"status": "prepared"})
        return 0
      if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY before launching")
      if reply is not None:
        request["task"] += (
          "\nPrevious question:\n"
          + pending.read_text()
          + "\nUser's explicit reply:\n"
          + reply
          + "\n"
        )
        save_json(host / "request.json", request)
        pending.rename(host / f"answered_question_{uuid.uuid4().hex[:12]}.json")
      task = request["task"]
      if args.resume:
        task += "\nRESUME: Read state.json and existing artifacts. Never reinitialize completed work. Begin only the remaining work; a partially finished phase may need recovery.\n"
      return run_cli(executable, home, host, task)
  except KeyboardInterrupt:
    return 130
  except Exception as exc:
    print(redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
