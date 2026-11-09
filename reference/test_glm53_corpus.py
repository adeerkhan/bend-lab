"""The corpus-trained dense baseline at 128 positions: data, training and eval.

  reference/.venv/bin/python reference/test_glm53_corpus.py

No PyTorch is used.  The test regenerates the 409,600-token corpus windows,
checks the next-byte alignment and hashes, builds `train_dense128.bend`, runs
the 400 steps again and compares the loss curve with `results/glm53-corpus128/`
at 1e-5 relative.  It then re-checks the dev/confirm manifests against the
frozen evaluator golden, the reference prompts and the reference evaluator.
A corrupted input byte must diverge at step 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VENDOR = os.path.join(ROOT, "reference/glm53_flash")
sys.path.insert(0, VENDOR)

import glm53_dense_data as D  # noqa: E402
from glm53_flash import ByteTokenizer  # noqa: E402
from glm53_flash.evaluator import evaluate_source  # noqa: E402
from glm53_flash.tasks import frozen_tasks  # noqa: E402

BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV.get("LD_LIBRARY_PATH", "").split(":"):
    ENV["LD_LIBRARY_PATH"] = ":".join(filter(None, [LLVM_LIB, ENV.get("LD_LIBRARY_PATH")]))
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-corpus-train-{TAG}"
DATA = D.DATA
RESULTS = os.path.join(ROOT, "results/glm53-corpus128")
GOLDEN = os.path.join(ROOT, "reference/glm53_corpus_eval_golden.json")
B32_DATA = os.path.join(DATA, "corpus128-b32")
B32_RESULTS = os.path.join(ROOT, "results/glm53-corpus128-b32")
B32_GOLDEN = os.path.join(ROOT, "reference/glm53_corpus_b32_eval_golden.json")
EX = os.path.join(ROOT, "examples/language/glm53_flash")
STEPS = 400
PER_FAMILY = 8
NEW = 48


def binary():
    deps = [os.path.join(EX, name) for name in
            ("train_dense128.bend", "model_128.bend", "model_a.bend", "weights_a.bend", "loss_masked.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(EX, "train_dense128.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"train_dense128.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run_training(data, out, steps=STEPS, lr_ppm=3000):
    os.makedirs(out, exist_ok=True)
    proc = subprocess.run([binary(), "--steps", str(steps), "--lr-ppm", str(lr_ppm),
                           "--data", data, "--out", out],
                          capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"train_dense128.bend did not run:\n{proc.stdout}{proc.stderr}")
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", proc.stdout, re.M)]
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", proc.stdout, re.M)
    val = re.search(r"^val_loss: ([0-9.eE+-]+)$", proc.stdout, re.M)
    params = re.search(r"^params: (\d+)$", proc.stdout, re.M)
    if final is None or val is None or params is None:
        raise SystemExit(f"the run did not print final_loss/val_loss/params:\n{proc.stdout}")
    return losses, float(final.group(1)), float(val.group(1)), int(params.group(1))


def relative(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return float("inf")
    return float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))


def gate_losses(got, want, limit=1e-5):
    ok = len(got) == len(want)
    rels = []
    for i, (a, b) in enumerate(zip(got, want), 1):
        a, b = float(a), float(b)
        rel = abs(a - b) / max(abs(b), 1e-30)
        rels.append(rel)
        if rel > limit:
            print(f"FAIL step {i}: {a:.9g} vs {b:.9g}, relative {rel:.3e}")
            ok = False
            break
    if ok:
        print(f"ok   loss: {len(got)} steps, max relative {max(rels):.3e} "
              f"(step 1 {got[0]:.9g} vs {want[0]:.9g})")
    return ok


def validate_data(directory):
    meta = json.load(open(os.path.join(directory, "corpus128.json")))
    x = np.fromfile(os.path.join(directory, "corpus128-train_batch.bin"), dtype=np.uint8)
    y = np.fromfile(os.path.join(directory, "corpus128-train_labels.bin"), dtype=np.uint8)
    vx = np.fromfile(os.path.join(directory, "corpus128-val_batch.bin"), dtype=np.uint8)
    vy = np.fromfile(os.path.join(directory, "corpus128-val_labels.bin"), dtype=np.uint8)
    ok = x.size == meta["tokens"] and vx.size == meta["batch"] * meta["positions"]
    print(f"{'ok  ' if ok else 'FAIL'} data: {directory}: {x.size} train tokens, {vx.size} val tokens")
    align = bool(np.array_equal(x[1:], y[:-1]))
    print(f"{'ok  ' if align else 'FAIL'} data: labels are the next byte")
    hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in
              (("train_batch.bin", x), ("train_labels.bin", y),
               ("val_batch.bin", vx), ("val_labels.bin", vy))}
    same = hashes == meta["sha256"]
    print(f"{'ok  ' if same else 'FAIL'} data: {directory}: hashes match corpus128.json")
    return ok and align and same


def copy_data(source):
    target = tempfile.mkdtemp(prefix="glm53-corpus-corrupt-")
    for name in os.listdir(source):
        src = os.path.join(source, name)
        dst = os.path.join(target, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return target


def validate_manifest(payload, split, golden, weights_path):
    problems = []
    tokenizer = ByteTokenizer()
    tasks = frozen_tasks(split, per_family=PER_FAMILY)
    episodes = payload.get("episodes", [])
    if len(episodes) != len(tasks):
        problems.append(f"{len(episodes)} episodes, not {len(tasks)}")
    if payload.get("split") != split:
        problems.append(f"split is {payload.get('split')!r}")
    if payload.get("inference", {}).get("max_new_tokens") != NEW:
        problems.append("max_new_tokens is not 48")
    if payload.get("checkpoint_sha256") != golden.get("checkpoint_sha256"):
        problems.append("checkpoint hash differs from the frozen manifest")
    if os.path.exists(weights_path) and payload.get("checkpoint_sha256") != hashlib.sha256(
            open(weights_path, "rb").read()).hexdigest():
        problems.append("checkpoint hash does not match results/glm53-corpus128/final_weights.bin")
    by_family = {}
    for task, row in zip(tasks, episodes):
        if row.get("task_id") != task.task_id:
            problems.append(f"task id {row.get('task_id')!r} != {task.task_id!r}")
            continue
        if row.get("prompt") != task.prompt:
            problems.append(f"{task.task_id}: prompt differs from the reference")
        if row.get("family") != task.family:
            problems.append(f"{task.task_id}: family differs")
        completion = row.get("generated_completion", "")
        scored = evaluate_source(task, task.prompt + completion).to_dict()
        if row.get("evaluation") != scored:
            problems.append(f"{task.task_id}: evaluation is not the reference evaluator's")
        if row.get("generated_tokens", 999) > NEW:
            problems.append(f"{task.task_id}: more than 48 tokens")
        fam = by_family.setdefault(task.family, {"solved": 0, "tasks": 0, "tests_passed": 0, "tests_total": 0})
        fam["tasks"] += 1
        fam["solved"] += int(row.get("evaluation", {}).get("passed", False))
        fam["tests_passed"] += int(row.get("evaluation", {}).get("tests_passed", 0))
        fam["tests_total"] += int(row.get("evaluation", {}).get("tests_total", 0))
    summary = {
        "tasks_solved": sum(int(row.get("evaluation", {}).get("passed", False)) for row in episodes),
        "tasks_total": len(episodes),
        "tests_passed": sum(int(row.get("evaluation", {}).get("tests_passed", 0)) for row in episodes),
        "tests_total": sum(int(row.get("evaluation", {}).get("tests_total", 0)) for row in episodes),
        "valid_programs": sum(row.get("evaluation", {}).get("status") != "invalid" for row in episodes),
        "token_cap_hits": sum(bool(row.get("hit_token_cap")) for row in episodes),
    }
    summary["pass_at_1"] = summary["tasks_solved"] / max(1, summary["tasks_total"])
    summary["test_accuracy"] = summary["tests_passed"] / max(1, summary["tests_total"])
    if payload.get("summary") != summary:
        problems.append("summary is not recomputed from the episodes")
    if payload.get("by_family") != by_family:
        problems.append("by_family is not recomputed from the episodes")
    if payload.get("summary") != golden.get("summary"):
        problems.append("summary differs from the frozen manifest")
    if payload.get("by_family") != golden.get("by_family"):
        problems.append("by_family differs from the frozen manifest")
    for i, row in enumerate(episodes):
        if i >= len(golden.get("episodes", [])):
            break
        g = golden["episodes"][i]
        if row.get("model_tokens") != g.get("model_tokens"):
            problems.append(f"{row.get('task_id')}: token sequence differs from the frozen manifest")
        if row.get("evaluation") != g.get("evaluation"):
            problems.append(f"{row.get('task_id')}: evaluation differs from the frozen manifest")
    return problems


def main():
    D.write_windows(DATA)
    D.write_windows(B32_DATA, steps=400, batch=32, positions=128, examples=16000)
    ok = [validate_data(DATA), validate_data(B32_DATA)]
    record = json.load(open(os.path.join(RESULTS, "run.json")))
    frozen_losses = [float(line.split("\t")[1]) for line in
                     open(os.path.join(RESULTS, "loss.tsv")).read().splitlines()[1:]]
    out = tempfile.mkdtemp(prefix="glm53-corpus-400-")
    losses, final, val, params = run_training(DATA, out)
    ok.append(params == 74144)
    print(f"params: {params}")
    ok.append(gate_losses(losses, frozen_losses))
    val_ok = abs(val - float(record["loss"]["val"])) <= 1e-5 * max(abs(float(record["loss"]["val"])), 1e-30)
    print(f"{'ok  ' if val_ok else 'FAIL'} val loss: {val:.9g} vs {record['loss']['val']:.9g}")
    ok.append(val_ok)

    for label, results_dir, golden_path in (
            ("b8", RESULTS, GOLDEN), ("b32", B32_RESULTS, B32_GOLDEN)):
        frozen = json.load(open(golden_path))
        weights_path = os.path.join(results_dir, "final_weights.bin")
        for split in ("dev", "confirm"):
            payload = json.load(open(os.path.join(results_dir, f"eval-{split}.json")))
            problems = validate_manifest(payload, split, frozen["splits"][split], weights_path)
            print(f"{'ok  ' if not problems else 'FAIL'} eval {label}/{split}: "
                  f"{len(payload.get('episodes', []))} episodes, {len(problems)} problems")
            for problem in problems:
                print(f"     {problem}")
            ok.append(not problems)
            bad = json.loads(json.dumps(payload))
            if bad.get("episodes"):
                bad["episodes"][0]["evaluation"]["passed"] = not bad["episodes"][0]["evaluation"]["passed"]
                bad["episodes"][0]["model_tokens"] = -1
            rejected = bool(validate_manifest(bad, split, frozen["splits"][split], weights_path))
            print(f"{'ok  ' if rejected else 'FAIL'} corrupt {label}/{split} manifest: rejected")
            ok.append(rejected)

    bad_dir = copy_data(DATA)
    path = os.path.join(bad_dir, "corpus128-train_batch.bin")
    x = bytearray(open(path, "rb").read())
    x[:16] = bytes([42] * 16)
    open(path, "wb").write(bytes(x))
    bad_out = tempfile.mkdtemp(prefix="glm53-corpus-bad-")
    bad, _, _, _ = run_training(bad_dir, bad_out, steps=1)
    rel = abs(bad[0] - frozen_losses[0]) / max(abs(frozen_losses[0]), 1e-30)
    rejected = rel > 1e-5
    print(f"{'ok  ' if rejected else 'FAIL'} corrupt batch: step-1 relative {rel:.3e} "
          f"({'rejected' if rejected else 'not rejected'})")
    ok.append(rejected)
    shutil.rmtree(bad_dir, ignore_errors=True)

    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
