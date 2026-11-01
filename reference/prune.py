"""Retention: what a prune may delete, and what it may never (B9).

  reference/prune.py --dry-run        # say what would go
  reference/prune.py                  # delete it

Rules:

- `results/latest/` and `results/latest-numpy/` are never pruned: `RESULTS.md`
  is rendered from them.
- A record any markdown file names (`results/raw/...json`, quoted or linked) is
  never pruned. A results table that points at evidence outlives the prune.
- Otherwise each task keeps its newest `--keep N` stamped records (5 by
  default). The stamps sort, so the newest is a name comparison, not a guess.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = re.compile(r"^(?P<task>.+)-(?P<stamp>\d{8}T\d{6}Z)\.json$")
CITE = re.compile(r"results/raw/([A-Za-z0-9._-]+\.json)")


def cited(root):
    """Every record file a markdown document in the tree names."""
    found = set()
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in
                   (".git", ".venv", ".venv-lucidrains", ".bend", ".local", ".scratch", "node_modules")]
        for name in files:
            if not name.endswith(".md"):
                continue
            try:
                with open(os.path.join(base, name)) as f:
                    found.update(CITE.findall(f.read()))
            except OSError:
                continue
    return found


def plan(root, keep):
    raw = os.path.join(root, "results", "raw")
    if not os.path.isdir(raw):
        return [], [], []
    named = cited(root)
    by_task = {}
    for name in sorted(os.listdir(raw)):
        match = STAMP.match(name)
        if match:
            by_task.setdefault(match.group("task"), []).append((match.group("stamp"), name))
        elif name.endswith(".json") and name not in named:
            by_task.setdefault(name, []).append(("", name))
    kept, gone = [], []
    for task, rows in sorted(by_task.items()):
        rows.sort()
        for index, (_, name) in enumerate(reversed(rows)):
            if index < keep or name in named:
                kept.append(name)
            else:
                gone.append(name)
    return kept, gone, sorted(named)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=int, default=5, help="stamped records per task (default 5)")
    parser.add_argument("--root", default=ROOT, help="the tree to prune (default: this repository)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, delete nothing")
    args = parser.parse_args()
    kept, gone, named = plan(args.root, args.keep)
    for name in sorted(kept):
        print(f"keep    results/raw/{name}" + ("   (cited)" if name in named else ""))
    for name in sorted(gone):
        print(f"remove  results/raw/{name}")
    print(f"\n{len(kept)} kept ({len([n for n in kept if n in named])} cited),"
          f" {len(gone)} to remove, {args.keep} per task")
    if args.dry_run:
        print("dry run: nothing deleted")
        return 0
    for name in gone:
        os.remove(os.path.join(args.root, "results", "raw", name))
    print(f"removed {len(gone)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
