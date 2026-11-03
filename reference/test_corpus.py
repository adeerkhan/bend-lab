"""The QwenBend corpus: 200 model-written Bend programs against this Bend.

  reference/.venv/bin/python reference/test_corpus.py [--fast]

`reference/corpus/qwenbend/` holds 200 small programs written by Qwen3.5-0.8B,
each one checked against **real Bend2** by its author, with the verdict, the
value and the first diagnostic line that Bend2 gave it. We did not write a single
one of them, which is the point: it is an outside conformance corpus for the
compiler this repository pins, and for its error messages.

Four checks, each a count rather than a claim:

  verdict      accept exactly the programs real Bend2 accepted, reject the rest
  value        every accepted program runs to the number Bend2 produced
  diagnostic   a rejected program gets the same first rule line
  can-fail     the comparator above, run over deliberately corrupted
               expectations, must report a failure for each of the three: a gate
               that cannot fail is not a gate

`--fast` skips the value runs (the compiler is exercised, the evaluator is not),
which is what an edit loop wants. Exit code 0 = every check passed.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

CORPUS = os.path.join(ROOT, "reference", "corpus", "qwenbend", "programs.jsonl")


def rows():
    with open(CORPUS) as f:
        return [json.loads(line) for line in f if line.strip()]


def first_rule(text):
    """The first rule a Bend error names: `- expected : ...` or `- message : ...`."""
    for line in text.splitlines():
        if line.startswith("- expected") or line.startswith("- message"):
            return line.split(":", 1)[1].strip()
    return ""


def observe(work, programs):
    """What this Bend does with each program: (accepts, value, first rule)."""
    seen = {}
    for row in programs:
        path = os.path.join(work, f"{row['i']:03d}.bend")
        with open(path, "w") as f:
            f.write(row["program"])
        check = subprocess.run([BEND, path, "--check-only"], capture_output=True,
                               text=True, env=ENV, cwd=ROOT, timeout=300)
        text = check.stdout + check.stderr
        accepts = "ALL PROOFS CHECK" in text
        value, rule = None, first_rule(text)
        if accepts:
            run = subprocess.run([BEND, path], capture_output=True, text=True,
                                 env=ENV, cwd=ROOT, timeout=600)
            out = [line for line in run.stdout.splitlines() if line.strip()]
            value = out[-1].strip() if out else None
        seen[row["i"]] = {"accepts": accepts, "value": value, "rule": rule}
    return seen


def compare(programs, seen, fast=False):
    """Every disagreement with the corpus, as {kind: [row index, ...]}."""
    wrong = {"verdict": [], "value": [], "diagnostic": []}
    for row in programs:
        got = seen[row["i"]]
        if got["accepts"] != bool(row["valid"]):
            wrong["verdict"].append(row["i"])
            continue
        if row["valid"]:
            if not fast and got["value"] != str(row["output"]):
                wrong["value"].append(row["i"])
        elif got["rule"] != first_rule(row["error"] or ""):
            wrong["diagnostic"].append(row["i"])
    return {kind: found for kind, found in wrong.items() if found}


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


def can_fail(programs, seen, fast):
    """The comparator must report each kind when the expectation is corrupted."""
    ok = True
    bad_verdict = [dict(row) for row in programs]
    first_valid = next(row for row in bad_verdict if row["valid"])
    first_valid["valid"] = False
    found = compare(bad_verdict, seen, fast)
    ok &= check("can-fail verdict", "verdict" in found,
                f"a valid program marked invalid is caught: {found.get('verdict')}")

    if not fast:
        bad_value = [dict(row) for row in programs]
        row = next(row for row in bad_value if row["valid"])
        row["output"] = str(int(row["output"]) + 1) if str(row["output"]).lstrip("-").isdigit() else "x"
        found = compare(bad_value, seen, fast)
        ok &= check("can-fail value", "value" in found,
                    f"an output moved by one is caught: {found.get('value')}")

    invalid = [row for row in programs if not row["valid"] and first_rule(row["error"] or "")]
    if invalid:
        bad_rule = [dict(row) for row in programs]
        row = next(row for row in bad_rule if not row["valid"] and first_rule(row["error"] or ""))
        row["error"] = "- expected : a rule this compiler never prints"
        found = compare(bad_rule, seen, fast)
        ok &= check("can-fail diagnostic", "diagnostic" in found,
                    f"a different first rule is caught: {found.get('diagnostic')}")
    return ok


def main():
    fast = "--fast" in sys.argv
    programs = rows()
    work = tempfile.mkdtemp(prefix="mlbend-corpus-")
    seen = observe(work, programs)
    wrong = compare(programs, seen, fast)

    valid = sum(1 for row in programs if row["valid"])
    invalid = len(programs) - valid
    ok = True
    ok &= check("verdict",
                not wrong.get("verdict"),
                f"{valid} accepted, {invalid} rejected, {len(wrong.get('verdict') or [])} wrong"
                + (f" (first {wrong['verdict'][:5]})" if wrong.get("verdict") else ""))
    if not fast:
        ok &= check("value", not wrong.get("value"),
                    f"{valid} values against Bend2's"
                    + (f", {len(wrong['value'])} wrong: {wrong['value'][:5]}" if wrong.get("value") else ""))
    ok &= check("diagnostic", not wrong.get("diagnostic"),
                f"{invalid} first rule lines"
                + (f", {len(wrong['diagnostic'])} wrong: {wrong['diagnostic'][:5]}"
                   if wrong.get("diagnostic") else ""))
    ok &= can_fail(programs, seen, fast)

    print(f"\n{'PASS' if ok else 'FAIL'}: the QwenBend corpus")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
