"""The stage split's two guards, as a check (ISSUE-039).

  reference/.venv/bin/python reference/test_stages.py

The capture builds a task's stage program, runs it three times and keeps each
key's median. Two things must hold or the split is a lie rather than a
measurement:

  the cache   the program is the task's own and reads the task's dataset, so
              `stages.ENV` must carry the same `MLBEND_DATA` the runner gives
              the task; without it the program fetches from the hub or fails
  the refusal a program that could not read its dataset still exits 0 and
              prints every `name_ms:` line, so a run that timed an empty batch
              would be recorded as a step. The capture must refuse it

The first two checks are the values themselves; the third builds a four-line
program that prints the marker and asserts `run_program` raises instead of
returning a split. Nothing here writes.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stages

CACHE = os.path.join(stages.ROOT, "reference/.cache/datasets")
PROGRAM = """import Base

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("step_in_ms: 0")
    IO.print("the dataset failed")
"""


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    ok = True

    ok &= check("the stage env carries the task's cache",
                stages.ENV.get("MLBEND_DATA") == CACHE,
                f"MLBEND_DATA={stages.ENV.get('MLBEND_DATA')}")
    # Whether the cache is warm is an environment fact, not a code fact: a
    # fresh checkout fetches on the first experiment. Say which it is, and do
    # not fail the gate on it.
    print(f"note the dataset cache is {'warm' if os.path.isdir(CACHE) else 'cold'} "
          f"({CACHE})")

    ok &= check("a run that timed an empty dataset is refused",
                stages.refuses("step_in_ms: 0\nthe dataset failed\n"),
                "the marker is caught")
    ok &= check("a normal run is not refused",
                not stages.refuses("step_in_ms: 41\nforward_ms: 37\n"),
                "no marker, no refusal")

    path = os.path.join(tempfile.mkdtemp(), "empty_dataset.bend")
    with open(path, "w") as f:
        f.write(PROGRAM)
    refused, why = False, ""
    try:
        stages.run_program(path, repeats=1)
        why = "run_program returned a split"
    except SystemExit as e:
        refused = "could not read the dataset" in str(e)
        why = str(e).strip().splitlines()[0] if str(e).strip() else "SystemExit"
    ok &= check("run_program refuses it end to end", refused, why)

    print("\nPASS: the stage split's guards" if ok else "\nFAIL: the stage split's guards")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
