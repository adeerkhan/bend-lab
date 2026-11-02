"""The artifact writers: a record is rewritten only when what it measures moves.

  reference/.venv/bin/python reference/test_artifacts.py

ISSUE-033: `make check` rewrote five committed records with a new `created_at`
and a new wall time on every run, so the gate left the tree dirty while every
number stayed identical. `reference/artifacts.py` compares the new content
against the committed file before writing; this checks that comparison in a
temporary directory:

  a record, the same       no write
  a record, one number     write
  a record, only the clock no write
  a table, the same text   no write
  a table, a moved value   write
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The records the gate rewrites: each is a run's numbers, plus its clock.
RECORDS = ["results/cifar-10/run.json", "results/cifar-train/run.json",
           "results/transformer/run.json", "results/diffusion/run.json",
           "results/diffusion-digits/run.json"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    fail = 0
    with tempfile.TemporaryDirectory() as tmp:
        # a record: identical content is not written, a moved number is,
        # a moved clock alone is not
        for name in RECORDS:
            source = os.path.join(ROOT, name)
            path = os.path.join(tmp, os.path.basename(os.path.dirname(name)) + ".json")
            shutil.copy(source, path)
            original = read(path)
            obj = json.loads(original)

            if artifacts.save_json(path, obj) or read(path) != original:
                print(f"ERROR {name}: an identical record was rewritten")
                fail += 1
            else:
                print(f"ok   {name}: an identical record is not written")

            moved = json.loads(json.dumps(obj))
            moved["bend"]["first_loss"] = moved["bend"]["first_loss"] + 1e-3
            if artifacts.save_json(path, moved) and read(path) != original:
                print(f"ok   {name}: a moved number is written")
            else:
                print(f"ERROR {name}: a moved number was not written")
                fail += 1

            with open(path, "w", encoding="utf-8") as f:
                f.write(original)
            clocked = json.loads(original)
            clocked["created_at"] = "1999-01-01T00:00:00+00:00"
            clocked["seconds"] = 999.0
            if artifacts.save_json(path, clocked) or read(path) != original:
                print(f"ERROR {name}: a clock rewrite was a move")
                fail += 1
            else:
                print(f"ok   {name}: a clock rewrite is not a move")

        # a table: exact text, never normalised, so the committed bytes win
        table = os.path.join(tmp, "loss.tsv")
        with open(table, "w", encoding="utf-8") as f:
            f.write("step\tbend\tnumpy\n1\t0.5\t0.5\n")
        if artifacts.save_text(table, read(table)) or read(table) != "step\tbend\tnumpy\n1\t0.5\t0.5\n":
            print("ERROR loss.tsv: an identical table was rewritten")
            fail += 1
        else:
            print("ok   loss.tsv: an identical table is not written")
        if artifacts.save_text(table, "step\tbend\tnumpy\n1\t0.6\t0.5\n") \
                and read(table) == "step\tbend\tnumpy\n1\t0.6\t0.5\n":
            print("ok   loss.tsv: a moved value is written")
        else:
            print("ERROR loss.tsv: a moved value was not written")
            fail += 1

    print(f"{fail} checks differ" if fail else "the artifact guard holds")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
