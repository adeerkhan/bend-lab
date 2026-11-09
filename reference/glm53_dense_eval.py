"""Greedy dev/confirm evaluation of the dense Stage A baseline.

  reference/.venv/bin/python reference/glm53_dense_eval.py \
      --weights results/glm53-dense/final_weights.bin \
      --out results/glm53-dense

The agent is `examples/language/glm53_flash/gen_a.bend` at 192 positions.
The prompts are the vendored reference's frozen tasks (8 per family), the
tokenizer is its byte tokenizer, and the score is its `evaluate_source`.  The
Bend program emits exactly 48 tokens; this driver applies the reference's
own stop rule (an eos, or a completion that parses and ends in a newline) so
the recorded completion is what `scripts/evaluate.py` would record.

The result is `results/glm53-dense/eval-<split>.json` plus a compact frozen
copy in `reference/glm53_dense_eval_golden.json` for `make check`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VENDOR = os.path.join(ROOT, "reference/glm53_flash")
sys.path.insert(0, VENDOR)

import glm53_dense_data as D  # noqa: E402
from glm53_flash import ByteTokenizer  # noqa: E402
from glm53_flash.evaluator import completion_is_parseable, evaluate_source  # noqa: E402
from glm53_flash.tasks import frozen_tasks  # noqa: E402

ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary(source):
    root_tag = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
    out = f"/tmp/glm53-gen-{root_tag}"
    deps = [os.path.join(ROOT, "examples/language/glm53_flash", name)
            for name in ("gen_a.bend", "model_eval.bend", "model_a.bend", "weights_a.bend")]
    if os.path.exists(out) and all(os.path.getmtime(out) >= os.path.getmtime(s) for s in deps):
        return out
    build = subprocess.run([os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend"),
                            "-o", out, os.path.join(ROOT, source)],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"{source} did not build:\n{build.stdout}{build.stderr}")
    return out


def generate(model_dir, data_dir, split, exe):
    out = subprocess.run([exe, "--split", split, "--weights", model_dir, "--data", data_dir],
                         capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit(f"gen_a {split} did not run:\n{out.stdout}{out.stderr}")
    rows = {}
    for line in out.stdout.splitlines():
        m = re.match(r"^episode: (\d+) (.*)$", line)
        if not m:
            continue
        tokens = [int(v) for v in m.group(2).split()]
        rows[int(m.group(1))] = tokens
    if sorted(rows) != list(range(len(rows))):
        raise SystemExit(f"{split}: episodes are not 0..{len(rows)-1}")
    return [rows[i] for i in range(len(rows))]


def stopped(task, tokenizer, tokens):
    kept = []
    for token in tokens:
        kept.append(token)
        text = tokenizer.decode(kept)
        if token == tokenizer.eos_id or completion_is_parseable(task, text):
            break
    return kept


def evaluate_split(args, tokenizer, tasks, rows, completions, exe, weights_sha):
    started = time.perf_counter()
    episodes = []
    for index, task in enumerate(tasks):
        if rows[index].tolist() != tokenizer.encode(task.prompt, bos=True) + [0] * (D.GEN_POSITIONS - len(tokenizer.encode(task.prompt, bos=True))):
            raise SystemExit(f"{task.task_id}: the prompt row is not the tokenizer's prompt")
        had = completions[index]
        if len(had) != args.max_new_tokens:
            raise SystemExit(f"{task.task_id}: {len(had)} tokens, not {args.max_new_tokens}")
        kept = stopped(task, tokenizer, had)
        text = tokenizer.decode(kept)
        evaluation = evaluate_source(task, task.prompt + text).to_dict()
        episodes.append({
            "task_id": task.task_id,
            "family": task.family,
            "prompt": task.prompt,
            "reference_completion": task.reference_completion,
            "generated_completion": text,
            "generated_tokens": len(kept),
            "model_tokens": had,
            "hit_token_cap": len(kept) == args.max_new_tokens and (not kept or kept[-1] != tokenizer.eos_id),
            "evaluation": evaluation,
        })
    solved = sum(row["evaluation"]["passed"] for row in episodes)
    tests_passed = sum(row["evaluation"]["tests_passed"] for row in episodes)
    tests_total = sum(row["evaluation"]["tests_total"] for row in episodes)
    summary = {
        "tasks_solved": solved,
        "tasks_total": len(episodes),
        "pass_at_1": solved / len(episodes),
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_accuracy": tests_passed / tests_total,
        "valid_programs": sum(row["evaluation"]["status"] != "invalid" for row in episodes),
        "token_cap_hits": sum(row["hit_token_cap"] for row in episodes),
    }
    by_family = {}
    for family in sorted({row["family"] for row in episodes}):
        family_rows = [row for row in episodes if row["family"] == family]
        by_family[family] = {
            "solved": sum(row["evaluation"]["passed"] for row in family_rows),
            "tasks": len(family_rows),
            "tests_passed": sum(row["evaluation"]["tests_passed"] for row in family_rows),
            "tests_total": sum(row["evaluation"]["tests_total"] for row in family_rows),
        }
    return {
        "schema_version": "1.0",
        "status": "complete",
        "split": args.split,
        "model": "dense Stage A, 74,144 parameters, 192 positions",
        "checkpoint": "results/glm53-dense/final_weights.bin",
        "checkpoint_sha256": weights_sha,
        "inference": {
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "per_family": args.per_family,
            "seed": 2026,
            "families": "all",
            "prompts": "reference/glm53_flash frozen_tasks",
        },
        "summary": summary,
        "by_family": by_family,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "episodes": episodes,
    }


def compact(payload):
    return {
        "split": payload["split"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "summary": payload["summary"],
        "by_family": payload["by_family"],
        "episodes": [
            {"task_id": row["task_id"], "generated_tokens": row["generated_tokens"],
             "model_tokens": row["model_tokens"], "evaluation": row["evaluation"]}
            for row in payload["episodes"]
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("results/glm53-dense/final_weights.bin"))
    ap.add_argument("--data", type=Path, default=Path(D.DATA))
    ap.add_argument("--out", type=Path, default=Path("results/glm53-dense"))
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--splits", default="dev,confirm")
    ap.add_argument("--binary", default="")
    args = ap.parse_args()
    if args.max_new_tokens != D.GEN_POSITIONS and args.max_new_tokens > 48:
        raise SystemExit("the generator emits 48 tokens")
    D.prepare(str(args.data), args.per_family)
    flat = Path(args.weights)
    if not flat.exists():
        raise SystemExit(f"{flat} does not exist; run train_a --steps 400 first")
    weights_sha = sha256_file(flat)
    model_dir = args.out / "model-400"
    D.write_flat_weights(__import__("numpy").fromfile(flat, dtype="<f4"), model_dir)
    exe = args.binary or binary("examples/language/glm53_flash/gen_a.bend")
    tokenizer = ByteTokenizer()
    args.out.mkdir(parents=True, exist_ok=True)
    golden = {"schema_version": "1.0", "splits": {}}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        tasks, rows, _ = D.eval_prompts(split, args.per_family)
        started = time.perf_counter()
        completions = generate(str(model_dir), str(args.data), split, exe)
        if len(completions) != len(tasks):
            raise SystemExit(f"{split}: {len(completions)} episodes, not {len(tasks)}")
        args.split = split
        payload = evaluate_split(args, tokenizer, tasks, rows, completions, exe, weights_sha)
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        path = args.out / f"eval-{split}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        golden["splits"][split] = compact(payload)
        s = payload["summary"]
        print(f"{split}: solved {s['tasks_solved']}/{s['tasks_total']}, "
              f"tests {s['tests_passed']}/{s['tests_total']}, "
              f"valid {s['valid_programs']}, caps {s['token_cap_hits']}, "
              f"{payload['elapsed_seconds']} s")
        for family, row in payload["by_family"].items():
            print(f"  {family}: {row['solved']}/{row['tasks']} solved, {row['tests_passed']}/{row['tests_total']} tests")
    golden_path = Path("reference/glm53_dense_eval_golden.json")
    golden_path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}/eval-*.json and {golden_path}")


if __name__ == "__main__":
    main()
