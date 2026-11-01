"""The experiment runner: one Bend program, one reference, one report card.

  reference/.venv/bin/python reference/experiment.py --list
  reference/.venv/bin/python reference/experiment.py --task iris-crossval
  reference/.venv/bin/python reference/experiment.py --all --wandb

This is the layer MLBend exists for. A **task** is a dataset, a program and
the numbers that program must produce; an **adapter** runs the program and
parses its output; a **reference** computes the same numbers with NumPy or
scikit-learn; and a **manifest** records everything a second person needs to
re-run it: the git commit, the Bend version, the package versions and hub
hashes, the machine, the seed and the tolerances.

The report is written to `reports/` as JSON. With `--wandb` the same manifest
is logged to Weights & Biases (project `mlbend`), offline when there is no
API key, so a run never blocks on the network.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import hashlib
import io
from datetime import datetime, timezone

try:
    import numpy as np
except ImportError:  # the frozen-reference gate runs on a machine without NumPy
    np = None

import bounds as floors
import references as frozen_refs
import report_run
import stages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

SCHEMA = "mlbend.experiment/3"


# --------------------------------------------------------------------------
# the tasks
# --------------------------------------------------------------------------

def parse_iris_crossval(out):
    folds = [float(m) for m in re.findall(r"^fold \d+: ([0-9.eE+-]+)$", out, re.M)]
    mean = float(re.search(r"mean accuracy: ([0-9.eE+-]+)", out).group(1))
    seed = int(re.search(r"^seed: (\d+)$", out, re.M).group(1))
    assignment = [int(v) for v in re.search(r"^folds: (.*)$", out, re.M).group(1).split()]
    return {"mean_accuracy": mean, "fold_min": min(folds), "fold_max": max(folds),
            "folds": assignment, "seed": seed}


def reference_iris_crossval(context=None):
    """The same folds, and the same 60 SGD steps, in NumPy.

    The Bend program prints the fold of every row, so this reference uses
    that partition rather than making its own. The model is the same one:
    standardise on the training rows, then 60 gradient steps at 0.5.
    """
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_crossval import load

    x, y = load()
    assignment = context.get("folds") if isinstance(context, dict) else None
    if assignment is None:
        from test_crossval import sklearn_folds
        folds = sklearn_folds(x, y)
        return {"mean_accuracy": float(np.mean(folds)), "fold_min": min(folds),
                "fold_max": max(folds)}
    scores = []
    for f in range(5):
        test = np.array([i for i, a in enumerate(assignment) if a == f])
        train = np.array([i for i, a in enumerate(assignment) if a != f])
        mean, sd = x[train].mean(axis=0), x[train].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        xt = (x[train] - mean) / sd
        xs = (x[test] - mean) / sd
        w = np.zeros((x.shape[1], 3))
        b = np.zeros(3)
        onehot = np.eye(3)[y[train]]
        for _ in range(60):
            logits = xt @ w + b
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p = p / p.sum(axis=1, keepdims=True)
            grad = (p - onehot) / len(train)
            w = w - 0.5 * (xt.T @ grad)
            b = b - 0.5 * grad.sum(axis=0)
        pred = np.argmax(xs @ w + b, axis=1)
        scores.append(float((pred == y[test]).mean()))
    return {"mean_accuracy": float(np.mean(scores)), "fold_min": min(scores),
            "fold_max": max(scores)}


def parse_iris_mlp(out):
    return {
        "loss_before": float(re.search(r"loss before: ([0-9.eE+-]+)", out).group(1)),
        "loss_after": float(re.search(r"loss after:  ([0-9.eE+-]+)", out).group(1)),
        "accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1)),
    }


def reference_iris_mlp(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_mlp import sklearn_result

    # the reference trains, so its loss is not comparable; only the score is
    return {"accuracy": sklearn_result()}


def parse_wine_kmeans(out):
    inertia = float(re.search(r"inertia: ([0-9.eE+-]+)", out).group(1))
    sizes = [int(n) for n in re.search(r"cluster sizes: ([0-9 ]+)", out).group(1).split()]
    return {"inertia": inertia, "cluster_sizes": sizes}


def reference_wine_kmeans(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from kmeans_wine import kmeans, load, standardise

    inertia, sizes = kmeans(standardise(load()), 3, 10)
    return {"inertia": inertia, "cluster_sizes": sizes}


def parse_cancer_knn(out):
    return {"accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1))}


def parse_attention(out):
    return {"checksum": float(re.search(r"checksum: ([0-9.eE+-]+)", out).group(1))}


def parse_transformer(out):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M) if float(v) > 0.0]
    accuracy = float(re.search(r"^accuracy: ([0-9.eE+-]+)$", out, re.M).group(1))
    return {"first_loss": losses[0], "final_loss": losses[-1], "accuracy": accuracy}


def parse_cancer_nb(out):
    return {"accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1))}


def parse_heart_nb(out):
    return {"accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1))}


def parse_wine_pca(out):
    return {"explained_variance_ratio": float(re.search(r"explained variance ratio: ([0-9.eE+-]+)", out).group(1))}


def reference_cancer_knn(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_knn import sklearn_knn

    return {"accuracy": sklearn_knn()}


def reference_attention(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_attention import expected

    return {"checksum": float(expected().sum(dtype="float32"))}


def parse_diffusion(out):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M)]
    sm = [float(v) for v in re.findall(r"^sample_mean: ([0-9.eE+-]+) ([0-9.eE+-]+)$", out, re.M)[0]]
    sd = [float(v) for v in re.findall(r"^sample_std: ([0-9.eE+-]+) ([0-9.eE+-]+)$", out, re.M)[0]]
    return {"first_loss": losses[0], "final_loss": losses[-1],
            "sample_mean": sm[0], "sample_std": sd[0]}


def reference_diffusion(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_diffusion import numpy_answer

    return numpy_answer()


def parse_diffusion_digits(out):
    # the prints: "loss:" at i=1,99,199,... carries curve[i-1], and the base
    # case's "final_loss:" carries the last computed loss, which is numpy's curve[-1]
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M)]
    vals = {k: float(re.search(rf"^{k}: ([0-9.eE+-]+)$", out, re.M).group(1))
            for k in ("data_mean", "sample_mean", "sample_std")}
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", out, re.M)
    return {"first_loss": losses[0],
            "final_loss": float(final.group(1)) if final else losses[-1],
            **vals}


def parse_vit_step(out):
    """The benchmark task's one number: the loss after three steps (reported, not gated)."""
    match = re.search(r"^loss_after_3: ([0-9.eE+-]+)$", out, re.M)
    return {"loss_after_3": float(match.group(1)) if match else None}


def reference_vit_step(context=None):
    return {"loss_after_3": None}


def parse_diffusion_vit(out):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M)]
    vals = {k: float(re.search(rf"^{k}: ([0-9.eE+-]+)$", out, re.M).group(1))
            for k in ("data_mean", "sample_mean", "sample_std")}
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", out, re.M)
    return {"first_loss": losses[0],
            "final_loss": float(final.group(1)) if final else losses[-1],
            **vals}


def reference_diffusion_vit(context=None):
    """The twin's numbers for the ViT: the step-0 loss and the data's mean.

    The twin does not train (it is the gate, not a second run), so the
    loss curve and the cloud are reported, not compared.
    """
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    import twin_vit as twin
    from test_diffusion_digits import load

    first = twin.first_loss()[0]
    return {"first_loss": float(first), "data_mean": float(load().mean()),
            "final_loss": None, "sample_mean": None, "sample_std": None}


def reference_cifar(context=None):
    """The twin's numbers for cifar-10: the first loss and the data's mean.

    The twin trains the same 40 steps and samples the same 8 images, so the
    gated numbers are the deterministic ones; the rest is report mode.
    """
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_cifar import numpy_answer

    return numpy_answer()


def reference_diffusion_digits(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_diffusion_digits import numpy_answer

    return numpy_answer()


def reference_transformer(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_transformer import numpy_curve

    curve, final, accuracy = numpy_curve()
    return {"first_loss": float(curve[0]), "final_loss": final, "accuracy": accuracy}


def reference_cancer_nb(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_naive_bayes import sklearn_result

    return {"accuracy": sklearn_result()}


def reference_heart_nb(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from sklearn.naive_bayes import GaussianNB
    from test_naive_bayes_heart import load

    x, y = load()
    split = len(x) * 4 // 5
    return {"accuracy": float(GaussianNB().fit(x[:split], y[:split]).score(x[split:], y[split:]))}


def reference_wine_pca(context=None):
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_pca import sklearn_ratio

    return {"explained_variance_ratio": sklearn_ratio()}


TASKS = {
    "iris-crossval": {
        "description": "logistic regression over iris, five stratified folds, scaler fitted on the training rows only",
        "program": "examples/classification/crossval/main.bend",
        "dataset": {"name": "iris", "rows": 150, "features": 4,
                    "source": "https://huggingface.co/datasets/mstz/iris",
                    "files": ["https://huggingface.co/datasets/mstz/iris/resolve/e99306d5df05d2ea66a188ea936037f6588a20e9/iris.data"], "expected_sha256": "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"},
        "seed": 42,
        "seeds": [42, 43, 44, 45, 46],
        "parse": parse_iris_crossval,
        "reference": reference_iris_crossval,
        "metrics": {
            "mean_accuracy": {"tolerance": 0.000001, "mode": "absolute"},
            "fold_min": {"tolerance": 0.000001, "mode": "absolute"},
            "fold_max": {"tolerance": 0.000001, "mode": "absolute"},
        },
    },
    "iris-mlp": {
        "description": "a two-layer rectified network over iris, 400 SGD steps",
        "program": "examples/classification/mlp/main.bend",
        "dataset": {"name": "iris", "rows": 150, "features": 4,
                    "source": "https://huggingface.co/datasets/mstz/iris",
                    "files": ["https://huggingface.co/datasets/mstz/iris/resolve/e99306d5df05d2ea66a188ea936037f6588a20e9/iris.data"], "expected_sha256": "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0"},
        "seed": 0,
        "parse": parse_iris_mlp,
        "reference": reference_iris_mlp,
        "metrics": {
            "accuracy": {"tolerance": 0.06, "mode": "absolute"},
        },
    },
    "wine-pca": {
        "description": "first principal component of wine, by power iteration",
        "program": "examples/unsupervised/pca/main.bend",
        "dataset": {"name": "wine", "rows": 6497, "features": 11,
                    "source": "https://huggingface.co/datasets/mstz/wine",
                    "files": ["https://huggingface.co/datasets/mstz/wine/resolve/2042c73ed074afafc54264ba6ef0ad0450fa1e04/wine/train.csv"], "expected_sha256": "7dcde7c70defa60f2e1a89c1cac712e0f90d940b1f3eaaf68a530b3c3b5dd95c"},
        "seed": 0,
        "parse": parse_wine_pca,
        "reference": reference_wine_pca,
        "metrics": {
            "explained_variance_ratio": {"tolerance": 0.001, "mode": "absolute"},
        },
    },
    "wine-kmeans": {
        "description": "k-means over wine, three clusters for ten rounds",
        "program": "examples/unsupervised/kmeans/main.bend",
        "dataset": {"name": "wine", "rows": 6497, "features": 11,
                    "source": "https://huggingface.co/datasets/mstz/wine",
                    "files": ["https://huggingface.co/datasets/mstz/wine/resolve/2042c73ed074afafc54264ba6ef0ad0450fa1e04/wine/train.csv"], "expected_sha256": "7dcde7c70defa60f2e1a89c1cac712e0f90d940b1f3eaaf68a530b3c3b5dd95c"},
        "seed": 0,
        "parse": parse_wine_kmeans,
        "reference": reference_wine_kmeans,
        "metrics": {
            "inertia": {"tolerance": 0.001, "mode": "relative"},
            "cluster_sizes": {"tolerance": 0, "mode": "exact"},
        },
    },
    "diffusion-2d": {
        "description": "a DDPM denoiser trained and sampled on two moons, BendTorch against NumPy",
        "program": "examples/diffusion/two_moons/main.bend",
        "dataset": {"name": "two-moons", "rows": 128, "features": 2,
                    "source": "generated: two interleaved half-circles on a fixed grid"},
        "seed": 0,
        "parse": parse_diffusion,
        "reference": reference_diffusion,
        "metrics": {
            "first_loss": {"tolerance": 0.001, "mode": "absolute"},
            "final_loss": {"tolerance": 0, "mode": "report"},
            "sample_mean": {"tolerance": 0, "mode": "report"},
            "sample_std": {"tolerance": 0, "mode": "report"},
        },
    },
    "diffusion-digits": {
        "description": "a DDPM denoiser over the first 1024 optdigits digits, BendTorch against NumPy",
        "program": "examples/diffusion/digits/main.bend",
        "dataset": {"name": "optdigits", "rows": 1024, "features": 64,
                    "source": "https://huggingface.co/datasets/mstz/optdigits",
                    "files": ["https://huggingface.co/datasets/mstz/optdigits/resolve/4f20cc827b763fb69e2a80e18e892e4e11498aec/optdigits.data"], "expected_sha256": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd"},
        "seed": 0,
        "parse": parse_diffusion_digits,
        "reference": reference_diffusion_digits,
        "metrics": {
            "first_loss": {"tolerance": 0.0001, "mode": "relative"},
            "data_mean": {"tolerance": 0.00001, "mode": "absolute"},
            "final_loss": {"tolerance": 0, "mode": "report"},
            "sample_mean": {"tolerance": 0, "mode": "report"},
            "sample_std": {"tolerance": 0, "mode": "report"},
        },
    },
    "cifar-10": {
        "description": "a DDPM denoiser over the first 256 CIFAR-10 images, BendTorch against NumPy",
        "program": "examples/cifar/main.bend",
        "dataset": {"name": "cifar10", "rows": 256, "features": 3072,
                    "source": "https://huggingface.co/datasets/arwics/cifar10",
                    "files": ["https://huggingface.co/datasets/arwics/cifar10/resolve/f44619abdb57391606d732724f9fc40a684471bf/train.csv"],
                    "expected_sha256": "80f3ca81e56946a0eabc65f68b69558b4fdc741e1aafb0d9f552fcee768b3161"},
        "seed": 0,
        "parse": parse_diffusion_digits,
        "reference": reference_cifar,
        "metrics": {
            "first_loss": {"tolerance": 0.0001, "mode": "relative"},
            "data_mean": {"tolerance": 0.00001, "mode": "absolute"},
            "final_loss": {"tolerance": 0, "mode": "report"},
            "sample_mean": {"tolerance": 0, "mode": "report"},
            "sample_std": {"tolerance": 0, "mode": "report"},
        },
    },
    "diffusion-vit": {
        "description": "a ViT denoiser (2x2 patches, causal attention) over the first 1024 optdigits digits",
        "program": "examples/diffusion/vit/main.bend",
        "dataset": {"name": "optdigits", "rows": 1024, "features": 64,
                    "source": "https://huggingface.co/datasets/mstz/optdigits",
                    "files": ["https://huggingface.co/datasets/mstz/optdigits/resolve/4f20cc827b763fb69e2a80e18e892e4e11498aec/optdigits.data"], "expected_sha256": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd"},
        "seed": 0,
        "stages": {"program": "bench/plumbing/main.bend"},
        "bounds": floors.vit_step,
        "parse": parse_diffusion_vit,
        "reference": reference_diffusion_vit,
        "metrics": {
            "first_loss": {"tolerance": 0.0001, "mode": "relative"},
            "data_mean": {"tolerance": 0.00001, "mode": "absolute"},
            "final_loss": {"tolerance": 0, "mode": "report"},
            "sample_mean": {"tolerance": 0, "mode": "report"},
            "sample_std": {"tolerance": 0, "mode": "report"},
        },
    },
    "vit-step": {
        "description": "one ViT training step at the plumbing benchmark's sizes, timed by stage",
        "program": "bench/plumbing/main.bend",
        "dataset": {"name": "synthetic-vit-step", "rows": 1024, "features": 64,
                    "source": "built in bench/plumbing/main.bend"},
        "seed": 0,
        "stages": {"program": "bench/plumbing/main.bend"},
        "bounds": floors.vit_step,
        "parse": parse_vit_step,
        "reference": reference_vit_step,
        "metrics": {
            "loss_after_3": {"tolerance": 0, "mode": "report"},
        },
    },
    "transformer-periodic": {
        "description": "a one-block causal transformer trained on a periodic sequence, BendTorch against NumPy",
        "program": "examples/transformers/periodic/main.bend",
        "dataset": {"name": "periodic-0-7", "rows": 32, "features": 8,
                    "source": "generated: two sequences of sixteen positions, vocab 8"},
        "seed": 0,
        "parse": parse_transformer,
        "reference": reference_transformer,
        "metrics": {
            "first_loss": {"tolerance": 0.25, "mode": "absolute"},
            "final_loss": {"tolerance": 0, "mode": "report"},
            "accuracy": {"tolerance": 0.1, "mode": "absolute"},
        },
    },
    "attention-parity": {
        "description": "single-head self-attention forward, four tokens of eight features",
        "program": "examples/transformers/attention/main.bend",
        "dataset": {"name": "synthetic-attention", "rows": 4, "features": 8,
                    "source": "generated by the formula in the program"},
        "seed": 0,
        "parse": parse_attention,
        "reference": reference_attention,
        "metrics": {
            "checksum": {"tolerance": 0.0001, "mode": "absolute"},
        },
    },
    "cancer-naive-bayes": {
        "description": "Gaussian naive Bayes over breast cancer, four fifths train",
        "program": "examples/classification/naive_bayes/main.bend",
        "dataset": {"name": "breast_cancer", "rows": 683, "features": 9,
                    "source": "https://huggingface.co/datasets/mstz/breast",
                    "files": ["https://huggingface.co/datasets/mstz/breast/resolve/3cc7c6eef1160389f64f8760d71cba2b789603ec/cancer/train.csv"], "expected_sha256": "4c35090cb5bf3e3ec8c528a4a837a33bacefc0b06c3b62fbb1331baf7ca865e8"},
        "seed": 0,
        "parse": parse_cancer_nb,
        "reference": reference_cancer_nb,
        "metrics": {
            "accuracy": {"tolerance": 0.000001, "mode": "absolute"},
        },
    },
    "heart-naive-bayes": {
        "description": "Gaussian naive Bayes over heart failure, four fifths train",
        "program": "examples/classification/naive_bayes_heart/main.bend",
        "dataset": {"name": "heart_failure", "rows": 299, "features": 12,
                    "source": "https://huggingface.co/datasets/mstz/heart_failure",
                    "files": ["https://huggingface.co/datasets/mstz/heart_failure/resolve/57c2e1a4b3bb2f28e766c6d3b10bc0ced972c857/death/train.csv"], "expected_sha256": "e0771c8c824b89f5b5fc6a4e4af56a3da940512bb93f255a190673868d28c08c"},
        "seed": 0,
        "parse": parse_heart_nb,
        "reference": reference_heart_nb,
        "metrics": {
            "accuracy": {"tolerance": 0.000001, "mode": "absolute"},
        },
    },
    "cancer-knn": {
        "description": "three-nearest-neighbours over breast cancer, four fifths train",
        "program": "examples/classification/knn/main.bend",
        "dataset": {"name": "breast_cancer", "rows": 683, "features": 9,
                    "source": "https://huggingface.co/datasets/mstz/breast",
                    "files": ["https://huggingface.co/datasets/mstz/breast/resolve/3cc7c6eef1160389f64f8760d71cba2b789603ec/cancer/train.csv"], "expected_sha256": "4c35090cb5bf3e3ec8c528a4a837a33bacefc0b06c3b62fbb1331baf7ca865e8"},
        "seed": 0,
        "parse": parse_cancer_knn,
        "reference": reference_cancer_knn,
        "metrics": {
            "accuracy": {"tolerance": 0.000001, "mode": "absolute"},
        },
    },
}


# --------------------------------------------------------------------------
# running a task
# --------------------------------------------------------------------------

def binary_for(task):
    return os.path.join(tempfile.gettempdir(), "mlbend-exp-" + task)


def build_bend(task, cfg):
    """Build the task's program once; every seed runs the same binary."""
    binary = binary_for(task)
    start = time.perf_counter()
    build = subprocess.run([BEND, "-o", binary, cfg["program"]],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    seconds = time.perf_counter() - start
    if build.returncode != 0:
        raise SystemExit(f"{task}: the program did not build:\n{build.stdout}{build.stderr}")
    return binary, seconds


def run_binary(task, binary, seed=0, repeat=1):
    """Run a built program; the seed is its first argument."""
    times, output = [], None
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        out = subprocess.run([binary, str(seed)], capture_output=True, text=True, env=ENV, cwd=ROOT)
        times.append(time.perf_counter() - start)
        if out.returncode != 0:
            raise SystemExit(f"{task}: the program did not run:\n{out.stdout}{out.stderr}")
        output = out.stdout
    return output, times


def run_bend(task, cfg, repeat=1, seed=0):
    """Build and run once; the convenience the tests use."""
    binary, build_seconds = build_bend(task, cfg)
    output, times = run_binary(task, binary, seed, repeat)
    return output, build_seconds, times, min(times)


def compare(cfg, bend_values, reference_values):
    """The metric table: the Bend number, the reference number, and the verdict."""
    table = {}
    for name, rule in cfg["metrics"].items():
        bend = bend_values.get(name)
        want = reference_values.get(name)
        if bend is None:
            table[name] = {"bend": None, "reference": want, "pass": False,
                           "note": "the program did not print this number"}
            continue
        if want is None:
            table[name] = {"bend": bend, "reference": None, "pass": True,
                           "note": "no reference for this number"}
            continue
        if rule["mode"] == "report":
            table[name] = {"bend": bend, "reference": want, "pass": True,
                           "note": "different folds; reported, not compared"}
            continue
        if rule["mode"] == "exact":
            delta = 0 if bend == want else None
            ok = bend == want
        else:
            delta = bend - want
            limit = rule["tolerance"] * abs(want) if rule["mode"] == "relative" else rule["tolerance"]
            ok = abs(delta) <= limit
        table[name] = {"bend": bend, "reference": want, "delta": delta,
                       "tolerance": rule["tolerance"], "mode": rule["mode"], "pass": ok}
    return table


def cache_file(url):
    """The cache path of a HuggingFace file URL, laid out as <repo>/<path>."""
    m = re.search(r"/datasets/(.+?)/resolve/[^/]+/(.+)$", url)
    if not m:
        return None
    return os.path.join(ROOT, "reference/.cache/datasets", m.group(1), m.group(2))


def dataset_probe(cfg):
    """Hash the pinned dataset bytes, fill the cache, and verify the pin."""
    dataset = cfg["dataset"]
    files = dataset.get("files", [])
    expected = dataset.get("expected_sha256")
    cache_root = os.path.join(ROOT, "reference/.cache/datasets")
    ENV["MLBEND_DATA"] = cache_root
    start = time.perf_counter()
    digest = hashlib.sha256()
    total = 0
    if files:
        for url in files:
            path = cache_file(url)
            data = open(path, "rb").read() if path and os.path.isfile(path) else None
            if data is None or (expected and hashlib.sha256(data).hexdigest() != expected):
                data = urllib.request.urlopen(url, timeout=120).read()
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(data)
            digest.update(data)
            total += len(data)
    else:
        digest.update((dataset.get("name", "") + "\n" + dataset.get("source", "")).encode())
    seconds = time.perf_counter() - start
    actual = digest.hexdigest()
    if expected and actual != expected:
        raise SystemExit(f"{dataset['name']}: the pinned bytes are {expected}, got {actual}")
    return {
        "files": files,
        "bytes": total,
        "sha256": actual,
        "expected_sha256": expected,
        "fetch_seconds": round(seconds, 3),
        "cache": cache_root,
    }


def git_state():
    def run(*args):
        r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
        return r.stdout.strip()

    return {"commit": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def bend_version():
    try:
        return subprocess.run([BEND, "version"], capture_output=True, text=True,
                              env=ENV, timeout=30).stdout.split()[-1]
    except Exception:
        return None


def packages():
    """The published packages in bend.lock: name, version and hub hash."""
    path = os.path.join(ROOT, "bend.lock")
    if not os.path.isfile(path):
        return []
    out, block = [], {}
    for line in open(path):
        line = line.strip()
        if line == "[[package]]":
            if block.get("name"):
                out.append(block)
            block = {}
        elif "=" in line and not line.startswith("["):
            key, value = [s.strip() for s in line.split("=", 1)]
            if key in ("name", "version", "hash"):
                block[key] = value.strip('"')
    if block.get("name"):
        out.append(block)
    return [p for p in out if p["name"].startswith(("mlbend-", "bend-ml-", "bendtorch-"))]


def machine():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__ if np is not None else "absent",
    }


def frozen_error(task, seed):
    """The gate's error when a task or a seed has no frozen reference."""
    return (f"no frozen reference for {task} seed {seed}: "
            f"{os.path.relpath(frozen_refs.path(task), ROOT)} is missing it.\n"
            "a task without a frozen reference is a gate error, not a skip: "
            "run make references (or make references-refresh)")


def evaluate(task, cfg, seed, repeat, trainer, binary, build_seconds,
             reference_mode="live", frozen=None):
    """One seed: run the program (or the adapter), and compare its numbers."""
    ENV["MLBEND_SEED"] = str(seed)
    if trainer == "numpy" and reference_mode == "frozen":
        raise SystemExit("--trainer numpy is the reference, so it cannot read a frozen one")
    if trainer == "numpy":
        times, parsed = [], None
    else:
        out, times = run_binary(task, binary, seed, repeat)
        parsed = cfg["parse"](out)
    if reference_mode == "live" and np is None:
        raise SystemExit("live references need NumPy, which is not installed here: "
                         "use --references frozen (see make references)")
    if reference_mode == "frozen":
        table = (frozen or {}).get(seed)
        if table is None:
            raise SystemExit(frozen_error(task, seed))
        reference_values, reference_seconds = dict(table), 0.0
    else:
        start = time.perf_counter()
        reference_values = cfg["reference"](parsed)
        reference_seconds = time.perf_counter() - start
    if trainer == "numpy":
        table = compare(cfg, reference_values, reference_values)
    else:
        table = compare(cfg, parsed, reference_values)
    return {"seed": seed, "table": table, "run_seconds": min(times) if times else 0.0,
            "times": times, "build_seconds": build_seconds,
            "reference_seconds": reference_seconds}


def mean_sd(values):
    """The mean and the sample deviation, or the one value for a non-number.

    Plain arithmetic, not NumPy: the frozen-reference gate aggregates the
    metric table on a machine where NumPy is not installed.
    """
    if values and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in values):
        mean = sum(values) / len(values)
        sd = None
        if len(values) > 1:
            sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        return float(mean), (float(sd) if sd is not None else None)
    return (values[0] if values else None), None


def aggregate_tables(tables):
    """One table from one table per seed: mean ± sd, pass only if every seed passed."""
    out = {}
    for name in tables[0]:
        rows = [t[name] for t in tables]
        bend, bend_sd = mean_sd([r["bend"] for r in rows])
        reference, reference_sd = mean_sd([r["reference"] for r in rows])
        deltas = [r.get("delta") for r in rows]
        if all(isinstance(d, (int, float)) and not isinstance(d, bool) for d in deltas):
            delta, delta_sd = mean_sd(deltas)
        else:
            delta, delta_sd = None, None
        row = dict(rows[0])
        row.update({"bend": bend, "bend_sd": bend_sd,
                    "reference": reference, "reference_sd": reference_sd,
                    "delta": delta, "delta_sd": delta_sd,
                    "n_seeds": len(rows), "pass": all(r["pass"] for r in rows)})
        out[name] = row
    return out


def peaks_for(machine_info):
    """The machine's measured peaks, if a probe wrote them (B3)."""
    key = re.sub(r"[^A-Za-z0-9._-]+", "-", machine_info.get("platform", "machine")).strip("-")
    for name in (f"peaks-{key}.json", "peaks.json"):
        path = os.path.join(ROOT, "results", name)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (OSError, ValueError):
                return {}
    return {}


def manifest_for(task, cfg, evals, repeat, probe, trainer="bend", lane="cpu", threads=None,
                 stages_list=None, stage_counts=None, stage_notes=None, stage_source=None,
                 reference_mode="live"):
    run_seconds = min(e["run_seconds"] for e in evals) if evals else 0.0
    build_seconds = evals[0]["build_seconds"] if evals else 0.0
    reference_seconds = max(e["reference_seconds"] for e in evals) if evals else 0.0
    table = aggregate_tables([e["table"] for e in evals])
    seeds = [e["seed"] for e in evals]
    notes = list(stage_notes or [])
    if notes:
        notes.append("the stage split is one step, not the whole run")
    if reference_mode == "frozen":
        notes.append("references: frozen from "
                     + os.path.relpath(frozen_refs.path(task), ROOT)
                     + "; the reference functions did not run")
    bounds_value = cfg.get("bounds")
    if callable(bounds_value):
        bounds_value = bounds_value()
    bounds_value = bounds_value or {}
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {"name": task, "description": cfg["description"], "program": cfg["program"],
                 "dataset": dict(cfg["dataset"], **probe), "seed": seeds[0] if seeds else None},
        "seeds": seeds,
        "trainer": trainer,
        "variant": {"trainer": trainer, "lane": lane, "threads": threads,
                    "bend": bend_version()},
        "git": git_state(),
        "bend": {"version": bend_version()},
        "packages": packages(),
        "machine": machine(),
        "run": {
            "build_seconds": round(build_seconds, 3),
            "seconds_min": round(run_seconds, 3),
            "seconds_all": [round(t, 3) for e in evals for t in e["times"]],
            "repeat": repeat,
            "fetch_seconds": probe["fetch_seconds"],
            "compute_seconds": round(max(0.0, run_seconds - probe["fetch_seconds"]), 3),
            "reference_seconds": round(reference_seconds, 3),
            "speed_ratio": round(reference_seconds / run_seconds, 3) if run_seconds else None,
        },
        "stages": stages_list or [],
        "stages_source": stage_source,
        "counts": stage_counts or {},
        "peaks": peaks_for(machine()),
        "bounds": bounds_value,
        "notes": notes,
        "seed_runs": [{"seed": e["seed"], "metrics": e["table"],
                       "run_seconds": round(e["run_seconds"], 3),
                       "reference_seconds": round(e["reference_seconds"], 3)}
                      for e in evals],
        "metrics": table,
        "passed": all(row["pass"] for row in table.values()),
    }
    return manifest


# --------------------------------------------------------------------------
# the report card and Weights & Biases
# --------------------------------------------------------------------------

def show_metric(value, sd, digits=6):
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.{digits}g}"
        return text if not sd else f"{text} ± {sd:.2g}"
    return str(value)


def report_text(manifest):
    """The report card is the page: one renderer, two callers (B2)."""
    return report_run.page(manifest)


def log_wandb(manifest, project, name, offline):
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is not installed: reference/.venv/bin/python -m pip install wandb")

    mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
    run = wandb.init(project=project, name=name, config=manifest, mode=mode,
                     tags=[manifest["task"]["name"]], reinit=True)
    values = {}
    for metric, row in manifest["metrics"].items():
        if isinstance(row["bend"], (int, float)):
            values[metric] = row["bend"]
        if isinstance(row["reference"], (int, float)):
            values["reference/" + metric] = row["reference"]
    for metric, row in manifest["metrics"].items():
        if isinstance(row.get("delta"), (int, float)):
            values["delta/" + metric] = row["delta"]
    run.log(values)
    run.summary["passed"] = manifest["passed"]
    run.summary["git_commit"] = manifest["git"]["commit"]
    run.summary["bend_version"] = manifest["bend"]["version"]
    run.finish()
    return mode


def write_report(manifest, out_dir):
    """Write the timestamped report and the task's latest report."""
    os.makedirs(out_dir, exist_ok=True)
    latest_name = "latest" if manifest.get("trainer", "bend") == "bend" else "latest-numpy"
    latest_dir = os.path.join(os.path.dirname(out_dir), latest_name)
    os.makedirs(latest_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"{manifest['task']['name']}-{stamp}.json")
    for target in (path, os.path.join(latest_dir, manifest["task"]["name"] + ".json")):
        with open(target, "w") as f:
            json.dump(manifest, f, indent=2)
    return path


def summary(root_dir):
    """Write RESULTS.md from the latest report of every task."""
    latest_dir = os.path.join(root_dir, "latest")
    if not os.path.isdir(latest_dir):
        raise SystemExit(f"no reports in {latest_dir}: run --all first")
    rows = [json.load(open(os.path.join(latest_dir, f)))
            for f in sorted(os.listdir(latest_dir)) if f.endswith(".json")]
    lines = [
        "# Results",
        "",
        "The latest report card of every task, written by `reference/experiment.py`.",
        "`fetch` is the time to fill and hash the local cache; the Bend program reads",
        "the same bytes through `MLBEND_DATA`, so on a warm cache it is 0 and `compute`",
        "is the run itself. `Python` is the reference computation. The build time is",
        "Bend's compiler and is reported separately.",
        "",
        f"Machine: {rows[0]['machine']['platform']} · {rows[0]['machine']['cpu_count']} CPUs ·"
        f" Bend {rows[0]['bend']['version']}" if rows else "",
        "",
        "| task | dataset | hash | metric | seeds | Bend | reference | delta | fetch ms | Bend ms | compute ms | Python ms | ok |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for m in rows:
        for name, row in m["metrics"].items():
            b = row["bend"]
            r = row["reference"]
            delta = row.get("delta")
            lines.append(
                f"| `{m['task']['name']}` | {m['task']['dataset']['name']} "
                f"({m['task']['dataset']['rows']}x{m['task']['dataset']['features']}) "
                f"| `{m['task']['dataset']['sha256'][:8]}` "
                f"| {name} | {len(m.get('seeds') or [1])} "
                f"| {show_metric(b, row.get('bend_sd'))} | {show_metric(r, row.get('reference_sd'))} "
                f"| {show_metric(delta, row.get('delta_sd'), 4)} "
                f"| {m['run']['fetch_seconds'] * 1000:.0f} "
                f"| {m['run']['seconds_min'] * 1000:.0f} "
                f"| {m['run']['compute_seconds'] * 1000:.0f} "
                f"| {m['run']['reference_seconds'] * 1000:.0f} "
                f"| {'ok' if row['pass'] else 'FAIL'} |")
    lines += ["", "Bend and reference are mean ± sample deviation over the seeds when a task",
              "has more than one; a single-seed task has no deviation. The dataset, seeds,",
              "hub hashes and the git state of every run are in",
              "`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.", ""]
    path = os.path.join(root_dir, "RESULTS.md")
    open(path, "w").write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), help="the task to run")
    parser.add_argument("--all", action="store_true", help="every task")
    parser.add_argument("--list", action="store_true", help="say what the tasks are")
    parser.add_argument("--wandb", action="store_true", help="log the manifest to Weights & Biases")
    parser.add_argument("--offline", action="store_true", help="force W&B offline mode")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "mlbend"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results/raw"), help="where the reports go")
    parser.add_argument("--repeat", type=int, default=1, help="run each program this many times and keep the minimum")
    parser.add_argument("--trainer", choices=["bend", "numpy"], default="bend",
                        help="which trainer runs the task: the Bend program or the NumPy reference")
    parser.add_argument("--seeds", help="comma-separated seeds, overriding the task's list")
    parser.add_argument("--lane", default="cpu", help="the lane the program runs on (cpu, gpu); recorded in the variant")
    parser.add_argument("--threads", type=int, default=None, help="the runtime's thread count, when known")
    parser.add_argument("--summary", action="store_true", help="write results/RESULTS.md from the latest reports")
    parser.add_argument("--references", choices=["live", "frozen"], default="live",
                        help="the reference numbers: the live functions (needs NumPy), or "
                             "results/reference/<task>.json, which needs none")
    parser.add_argument("--freeze-references", action="store_true",
                        help="write results/reference/<task>.json from this run's reference column")
    parser.add_argument("--refresh-references", action="store_true",
                        help="with --freeze-references: recompute live and overwrite")
    args = parser.parse_args()
    if args.refresh_references:
        args.freeze_references, args.references = True, "live"

    if args.summary:
        print(summary(os.path.join(ROOT, "results")))
        return 0

    if args.list:
        for name, cfg in TASKS.items():
            print(f"{name:<18} {cfg['program']:<32} {cfg['description']}")
        return 0

    names = sorted(TASKS) if args.all else ([args.task] if args.task else [])
    if not names:
        parser.error("give --task, --all or --list")

    failed = False
    for name in names:
        cfg = TASKS[name]
        probe = dataset_probe(cfg)
        seeds = cfg.get("seeds") or [cfg.get("seed", 0)]
        if args.seeds:
            seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
        if args.trainer == "numpy":
            seeds = seeds[:1]
            print(f"{name}: the NumPy trainer (no Bend build)")
            binary, build_seconds = None, 0.0
        else:
            binary, build_seconds = build_bend(name, cfg)
        stages_list, stage_counts, stage_notes, stage_source = [], {}, [], None
        if cfg.get("stages") and args.trainer == "bend":
            stages_list, stage_counts, stage_notes, stage_source = stages.capture(cfg)
        frozen = None
        if args.references == "frozen":
            frozen = frozen_refs.values(frozen_refs.read(name))
            if not frozen:
                raise SystemExit(frozen_error(name, seeds[0]))
        evals = [evaluate(name, cfg, seed, args.repeat, args.trainer, binary, build_seconds,
                          reference_mode=args.references, frozen=frozen)
                 for seed in seeds]
        manifest = manifest_for(name, cfg, evals, args.repeat, probe, args.trainer,
                                lane=args.lane, threads=args.threads,
                                stages_list=stages_list, stage_counts=stage_counts,
                                stage_notes=stage_notes, stage_source=stage_source,
                                reference_mode=args.references)
        path = write_report(manifest, args.out)
        if args.freeze_references:
            target = frozen_refs.path(name)
            if os.path.isfile(target) and not args.refresh_references:
                print(f"reference: {os.path.relpath(target, ROOT)} kept"
                      " (--refresh-references rewrites it)")
            else:
                payload = frozen_refs.from_record(name, manifest,
                                                  " ".join([os.path.relpath(PY, ROOT)] + sys.argv))
                payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
                print(f"reference: {os.path.relpath(frozen_refs.write(name, payload), ROOT)}")
        print(report_text(manifest))
        print(f"report: {path}")
        if args.wandb:
            mode = log_wandb(manifest, args.project, f"{name}-{manifest['git']['commit']}", args.offline)
            print(f"wandb: {mode}")
        print()
        failed = failed or not manifest["passed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
