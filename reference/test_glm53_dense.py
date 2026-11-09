"""The dense Stage A gate: 20 AdamW steps against the PyTorch twin, and the
frozen dev/confirm evaluation manifest.

  reference/.venv/bin/python reference/test_glm53_dense.py

No PyTorch is imported here.  The gate builds `train_a.bend`, runs the 20
steps on the NumPy batch, and compares every step's loss (1e-4 relative) and
the final weights and gradients (1e-3 relative) with
`reference/glm53_dense_golden.npz`, which `glm53_dense_twin.py` wrote from the
same files.  Two can-fail runs (one weight changed, one input byte changed)
must be rejected by the same comparison.  The evaluation manifest is checked
for prompt identity, the reference evaluator's verdicts, the four summary
counts and the frozen token sequence; a corrupted copy must be rejected.
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

DATA = D.DATA
RESULTS = os.path.join(ROOT, "results/glm53-dense")
GOLDEN = os.path.join(ROOT, "reference/glm53_dense_golden.npz")
EVAL_GOLDEN = os.path.join(ROOT, "reference/glm53_dense_eval_golden.json")
TAG = hashlib.sha1(ROOT.encode()).hexdigest()[:8]
BIN = f"/tmp/glm53-dense-train-{TAG}"
STEPS = 20
PER_FAMILY = 8
NEW = 48


def binary():
    deps = [os.path.join(ROOT, "examples/language/glm53_flash", name) for name in
            ("train_a.bend", "model_a.bend", "weights_a.bend", "loss_masked.bend")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) >= os.path.getmtime(s) for s in deps):
        return BIN
    build = subprocess.run([BEND, "-o", BIN, os.path.join(ROOT, "examples/language/glm53_flash/train_a.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"train_a.bend did not build:\n{build.stdout}{build.stderr}")
    return BIN


def run_train(data, out, steps):
    os.makedirs(out, exist_ok=True)
    proc = subprocess.run([binary(), "--steps", str(steps), "--data", data, "--out", out],
                          capture_output=True, text=True, env=ENV, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"train_a.bend did not run:\n{proc.stdout}{proc.stderr}")
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", proc.stdout, re.M)]
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", proc.stdout, re.M)
    params = re.search(r"^params: (\d+)$", proc.stdout, re.M)
    if final is None or params is None:
        raise SystemExit(f"the run did not print final_loss/params:\n{proc.stdout}")
    losses.append(float(final.group(1)))
    return losses, int(params.group(1)), proc.stdout


def relative(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return float("inf")
    scale = max(float(np.abs(want).max()), 1e-30)
    return float(np.abs(got - want).max() / scale)


def gate_losses(got, want, limit=1e-4):
    ok = True
    rels = []
    for i, (a, b) in enumerate(zip(got, want), 1):
        a, b = float(a), float(b)
        rel = abs(a - b) / max(abs(b), 1e-30)
        rels.append(rel)
        if rel > limit:
            print(f"FAIL step {i}: {a:.9g} vs {b:.9g}, relative {rel:.3e}")
            ok = False
            break
    if len(got) != len(want):
        print(f"FAIL loss count: {len(got)} vs {len(want)}")
        ok = False
    if ok:
        print(f"ok   loss: {len(got)} steps, max relative {max(rels):.3e} "
              f"(step 1 {got[0]:.9g} vs {want[0]:.9g})")
    return ok


def gate_tensor(name, got, want, limit=1e-3):
    rel = relative(got, want)
    ok = got.shape == want.shape and rel <= limit
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got.size} values, "
          f"max |diff| {np.abs(got - want).max():.3e}, whole-tensor relative {rel:.3e}")
    return ok


def copy_data(source):
    target = tempfile.mkdtemp(prefix="glm53-corrupt-")
    for name in os.listdir(source):
        src = os.path.join(source, name)
        dst = os.path.join(target, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return target


def corrupt_weight(source):
    target = copy_data(source)
    path = os.path.join(target, "emb.bin")
    x = np.fromfile(path, dtype="<f4")
    x[0] += 2.0
    x.astype("<f4").tofile(path)
    return target


def corrupt_batch(source):
    target = copy_data(source)
    path = os.path.join(target, "train_batch.bin")
    x = bytearray(open(path, "rb").read())
    x[0] = 42
    open(path, "wb").write(bytes(x))
    return target


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(payload, split, golden):
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
    if not payload.get("inference", {}).get("greedy"):
        problems.append("not greedy")
    if payload.get("checkpoint_sha256") != golden.get("checkpoint_sha256"):
        problems.append("checkpoint hash differs from the frozen manifest")
    weight_path = os.path.join(RESULTS, "final_weights.bin")
    if os.path.exists(weight_path) and payload.get("checkpoint_sha256") != sha256_file(weight_path):
        problems.append("checkpoint hash does not match results/glm53-dense/final_weights.bin")
    by_family = {}
    for task, row in zip(tasks, episodes):
        if row.get("task_id") != task.task_id:
            problems.append(f"task id {row.get('task_id')!r} != {task.task_id!r}")
            continue
        if row.get("prompt") != task.prompt:
            problems.append(f"{task.task_id}: prompt differs from the reference")
        if row.get("family") != task.family:
            problems.append(f"{task.task_id}: family differs")
        expected_prompt = tokenizer.encode(task.prompt, bos=True)
        if len(expected_prompt) + NEW > D.GEN_POSITIONS:
            problems.append(f"{task.task_id}: prompt exceeds the context")
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
    if sorted(by_family) != sorted({task.family for task in tasks}):
        problems.append("family set differs")
    for i, task in enumerate(tasks):
        if i >= len(episodes):
            break
        g = golden.get("episodes", [])
        if i >= len(g) or episodes[i].get("model_tokens") != g[i].get("model_tokens"):
            problems.append(f"{task.task_id}: generated token sequence differs from the frozen manifest")
    if payload.get("summary") != golden.get("summary"):
        problems.append("summary differs from the frozen manifest")
    if payload.get("by_family") != golden.get("by_family"):
        problems.append("by_family differs from the frozen manifest")
    for i, row in enumerate(episodes):
        if i >= len(golden.get("episodes", [])):
            break
        g = golden["episodes"][i]
        if row.get("task_id") != g.get("task_id") or row.get("evaluation") != g.get("evaluation"):
            problems.append(f"{row.get('task_id')}: evaluation differs from the frozen manifest")
    return problems


def main():
    D.prepare(DATA, PER_FAMILY)
    golden = np.load(GOLDEN)
    out = tempfile.mkdtemp(prefix="glm53-dense-20-")
    losses, params, text = run_train(DATA, out, STEPS)
    ok = []
    print(f"params: {params}")
    ok.append(params == 74144)
    print(f"steps: {len(losses)}")
    ok.append(gate_losses(losses, golden["losses"]))
    weights = np.fromfile(os.path.join(out, "final_weights.bin"), dtype="<f4")
    grads = np.fromfile(os.path.join(out, "final_grads.bin"), dtype="<f4")
    ok.append(gate_tensor("final weights", weights, golden["weights"]))
    ok.append(gate_tensor("final gradients", grads, golden["grads"]))

    # can-fail: one weight changed, then one input byte changed; both must
    # diverge at step 1 under the same 1e-4 loss gate.
    for name, corrupt in (("weight", corrupt_weight), ("batch", corrupt_batch)):
        bad_dir = corrupt(DATA)
        bad_out = tempfile.mkdtemp(prefix=f"glm53-dense-{name}-")
        bad_loss, _, _ = run_train(bad_dir, bad_out, 1)
        rel = abs(bad_loss[0] - float(golden["losses"][0])) / max(abs(float(golden["losses"][0])), 1e-30)
        rejected = rel > 1e-4
        print(f"{'ok  ' if rejected else 'FAIL'} corrupt {name}: step-1 relative {rel:.3e} "
              f"({'rejected' if rejected else 'not rejected'})")
        ok.append(rejected)
        shutil.rmtree(bad_dir, ignore_errors=True)

    # the evaluation manifest: identity, the reference evaluator, the counts
    # and the frozen tokens; then a corrupted copy must be rejected.
    frozen = json.load(open(EVAL_GOLDEN))
    for split in ("dev", "confirm"):
        path = os.path.join(RESULTS, f"eval-{split}.json")
        if not os.path.exists(path):
            print(f"FAIL {split}: {path} is missing")
            ok.append(False)
            continue
        payload = json.load(open(path))
        problems = validate_manifest(payload, split, frozen["splits"][split])
        print(f"{'ok  ' if not problems else 'FAIL'} eval {split}: "
              f"{len(payload.get('episodes', []))} episodes, {len(problems)} problems")
        for problem in problems:
            print(f"     {problem}")
        ok.append(not problems)
        bad = json.loads(json.dumps(payload))
        if bad.get("episodes"):
            bad["episodes"][0]["generated_tokens"] = -1
            bad["episodes"][0]["evaluation"]["passed"] = not bad["episodes"][0]["evaluation"]["passed"]
        bad_problems = validate_manifest(bad, split, frozen["splits"][split])
        rejected = bool(bad_problems)
        print(f"{'ok  ' if rejected else 'FAIL'} corrupt {split} manifest: "
              f"{len(bad_problems)} problems ({'rejected' if rejected else 'not rejected'})")
        ok.append(rejected)

    print(f"{sum(ok)}/{len(ok)} gates passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
