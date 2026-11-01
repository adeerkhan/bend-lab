"""The frozen references: the numbers a gate compares against (B5a).

  reference/references.py --list        # which tasks have a frozen file
  reference/references.py --check       # exit 1 when a task has none

A frozen file is `results/reference/<task>.json`: the reference column of the
task's latest run, per seed, with the command that produced it and the time it
was frozen. The gate reads it instead of running NumPy, so a machine without
NumPy still checks the tasks; `--refresh-references` on
`reference/experiment.py` recomputes the numbers with the live reference
functions and rewrites the file.

A task without a frozen file is a gate error, never a skip: a skipped check
reads exactly like a passing one.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "results", "reference")
SCHEMA = "mlbend.reference/1"


def path(task):
    return os.path.join(DIR, task + ".json")


def read(task):
    """The frozen file of one task, or None when it has none."""
    try:
        with open(path(task)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def values(frozen):
    """`{seed: {metric: value}}` from a frozen file, seeds as integers."""
    return {int(seed): dict(row) for seed, row in (frozen or {}).get("seeds", {}).items()}


def from_record(task, record, command):
    """The frozen payload for one task, read off a run record's reference column.

    A reference may depend on what the program printed (the fold assignment, a
    parameter count), so the numbers are taken from the run that computed them,
    not recomputed here: `source` names that record and its command.
    """
    seeds = {}
    for run in record.get("seed_runs") or []:
        # a seed stays in the file even when its reference columns are empty: the
        # file must say which seeds the gate may ask for, not guess
        seeds[str(run.get("seed"))] = {
            name: cell.get("reference") for name, cell in (run.get("metrics") or {}).items()
            if cell.get("reference") is not None}
    return {
        "schema": SCHEMA,
        "task": task,
        "created_at": record.get("created_at"),
        "frozen_at": None,
        "command": command,
        "source": {"record": os.path.relpath(path_of_record(task), ROOT),
                   "created_at": record.get("created_at")},
        "bend": (record.get("bend") or {}).get("version"),
        "git": (record.get("git") or {}).get("commit"),
        "machine": (record.get("machine") or {}).get("platform"),
        "seeds": seeds,
    }


def path_of_record(task, numpy=False):
    return os.path.join(ROOT, "results", "latest-numpy" if numpy else "latest", task + ".json")


def write(task, payload):
    os.makedirs(DIR, exist_ok=True)
    with open(path(task), "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path(task)


def freeze(tasks, refresh=False):
    """Write the frozen file of each task from its latest record.

    The numbers are the reference column of a run that already computed them,
    so this costs no build and no NumPy; `--refresh-references` on
    `reference/experiment.py` is what recomputes them with the live functions.
    """
    from datetime import datetime, timezone

    written, kept, absent = [], [], []
    for name in sorted(tasks):
        target = path(name)
        if os.path.isfile(target) and not refresh:
            kept.append(name)
            continue
        try:
            with open(path_of_record(name)) as f:
                record = json.load(f)
        except (OSError, ValueError):
            absent.append(name)
            continue
        payload = from_record(name, record, f"reference/references.py --freeze --task {name}")
        payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
        write(name, payload)
        written.append(name)
    return written, kept, absent


def missing(tasks):
    """The tasks that declare no frozen file."""
    return [name for name in sorted(tasks) if read(name) is None]


def main():
    if not any(flag in sys.argv for flag in ("--check", "--list", "--freeze")):
        raise SystemExit("usage: references.py --list | --check | --freeze [--all|--task NAME]"
                         " [--refresh-references]")
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    import experiment  # the task list; importing it must not need NumPy

    if "--freeze" in sys.argv:
        names = sorted(experiment.TASKS)
        if "--task" in sys.argv:
            names = [sys.argv[sys.argv.index("--task") + 1]]
        written, kept, absent = freeze(names, "--refresh-references" in sys.argv)
        for name in written:
            print(f"froze {name:<20} -> {os.path.relpath(path(name), ROOT)}")
        for name in kept:
            print(f"kept  {name:<20} (--refresh-references rewrites it)")
        for name in absent:
            print(f"no record for {name}: run make experiment TASK={name} first")
        return 1 if absent else 0

    absent = missing(experiment.TASKS)
    if "--list" in sys.argv:
        for name in sorted(experiment.TASKS):
            frozen = read(name)
            if frozen is None:
                print(f"{name:<20} no frozen reference (make references)")
            else:
                seeds = sorted(values(frozen))
                print(f"{name:<20} seeds {seeds} · frozen at {frozen.get('frozen_at')}"
                      f" · from {frozen.get('source', {}).get('record', '?')}")
    if "--check" in sys.argv:
        if absent:
            print("no frozen reference for " + ", ".join(absent)
                  + "\nthe gate compares against results/reference/<task>.json: run make references")
            return 1
        print(f"every task has a frozen reference ({len(experiment.TASKS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
