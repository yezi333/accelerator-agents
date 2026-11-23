"""Pause a native CLI run at an original human decision gate."""

import argparse
import json
from pathlib import Path


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", required=True, type=Path)
  parser.add_argument("--question", required=True)
  parser.add_argument("--artifact", action="append", default=[])
  args = parser.parse_args(argv)
  host = args.run_dir.resolve() / "codex_host"
  if not (host / "request.json").is_file():
    parser.error("Not a prepared Codex CLI run")
  pending = host / "pending_question.json"
  if pending.exists():
    print("Already paused; do not overwrite the pending question.")
    return 0
  temporary = host / "pending_question.tmp"
  temporary.write_text(
    json.dumps(
      {"question": args.question, "artifacts": args.artifact},
      ensure_ascii=False,
      indent=2,
    )
    + "\n"
  )
  temporary.replace(pending)
  print(
    "NEEDS_INPUT: stop all work; wait for the user's explicit reply.",
    flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
