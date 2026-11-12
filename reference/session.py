"""Run any command as a dashboard session: a run row, a log and a live PID.

  reference/session.py --task check-fast -- make check-fast
  reference/session.py --task mamba3 --label "Mamba-3 gates" -- make mamba3

The command's output goes to the terminal and to
`results/<task>/runs/<stamp>-<pid>.log`; the run is `running` while the command
runs and `done` or `failed` (with the exit code) after it. The dashboard
samples the process tree, so a Python test that spawns Bend still shows the
Bend child's CPU, memory and GPU use.

A training run that already writes its own steps should use `make trace`
instead, not this wrapper: two rows would describe one run.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--label")
    ap.add_argument("--kind", default="session")
    ap.add_argument("--log", help="default results/<task>/runs/<stamp>-<pid>.log")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        ap.error("give a command after --")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_path = args.log or os.path.join(
        ROOT, "results", args.task, "runs", f"{stamp}-{os.getpid()}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        run_id = runs.start(
            args.task, "", None, args.label or f"{stamp} · {' '.join(command)}",
            pid=os.getpid(), command=" ".join(command), kind=args.kind,
            config={"cwd": os.getcwd(), "argv": command})
        print(f"session: run {run_id} · {os.path.relpath(log_path, ROOT)}")
    except Exception as exc:
        print(f"session: run store unavailable: {exc}")
        run_id = None

    start = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=os.getcwd())
    if run_id is not None:
        try:
            runs.set_pid(run_id, proc.pid)
        except Exception:
            pass
    with open(log_path, "w") as log:
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
        finally:
            code = proc.wait()
    seconds = time.time() - start
    if run_id is not None:
        try:
            runs.finish(run_id, "done" if code == 0 else "failed",
                        exit_code=code,
                        error=None if code == 0 else f"exit {code}")
        except Exception as exc:
            print(f"session: run store unavailable: {exc}")
    print(f"session: exit {code} · {seconds:.1f} s · {os.path.relpath(log_path, ROOT)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
