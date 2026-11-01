"""What a checkpoint is, and why a resume may refuse it (B9).

A checkpoint is two files that travel together:

  <name>.bin   the weights, back to back, in the order `tensors` records;
  <name>.json  the identity: what model wrote them, with which config, by
               which package versions and which git commit.

A resume reads the identity first. `verify(saved, current)` names every field
that differs — the model's structure, its parameter count, a config value, a
package version, the weights' own hash — so a resume refuses with a sentence
and not with a tensor of the wrong shape.

Loading and saving can be filtered by name: a partial save writes only the
tensors asked for, a partial load reads only those it is given, and the identity
still records every tensor the file holds. That is what a fine-tune over one
block needs.

Who writes a checkpoint: the training program. The flow and diffusion examples
print their numbers and keep no weights yet, so this module is the identity and
the loader they will use; `reference/checkpoints.py --identify` is how a
person reads one.

  reference/checkpoints.py --identify results/flow/flow.bin
  reference/checkpoints.py --verify results/flow/flow.json --against other.json
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = "mlbend.checkpoint/1"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def structure(model):
    """The model's identity: its layout, in one digest.

    `model` is what the program declares — for the flow example,
    `[3, 128, 128, 2]` with its activation — so two models that differ in a
    width, a depth or an activation have different structures and a resume
    between them is refused.
    """
    return sha256_bytes(json.dumps(model, sort_keys=True).encode())


def package_map(packages):
    """A checkpoint's package identity as `{name: [version and hash, ...]}`.

    `reference/experiment.py` gives a list (several versions of one package can
    be resolved at once), a lock gives a mapping; both are compared by name, so
    a resumed run that resolved a different version is refused by name.
    """
    if isinstance(packages, dict):
        return {str(k): [str(v)] for k, v in packages.items()}
    out = {}
    for entry in packages or []:
        if isinstance(entry, dict):
            name = str(entry.get("name", "?"))
            out.setdefault(name, []).append(f"{entry.get('version')} {entry.get('hash')}")
        else:
            out.setdefault(str(entry), []).append("")
    return {name: sorted(values) for name, values in out.items()}


def identity(weights_path, model, config, packages, bend=None, git=None):
    """The sidecar of one weights file."""
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": {"file": os.path.basename(weights_path),
                    "sha256": sha256_file(weights_path),
                    "bytes": os.path.getsize(weights_path)},
        "model": {"structure": structure(model), "declared": model},
        "config": {"sha256": sha256_bytes(json.dumps(config, sort_keys=True).encode()),
                   "values": config},
        "packages": package_map(packages),
        "bend": bend,
        "git": git,
    }


def identity_only(model, config, packages, bend=None, git=None):
    """The identity of a run whose weights are not saved yet.

    The flow and diffusion examples print their numbers and keep no weights
    (ISSUE-034), so their identity can still be written and verified: a resume
    between two configs is refused on the field that differs, and the moment the
    example saves weights the same file grows a `weights` block.
    """
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": None,
        "model": {"structure": structure(model), "declared": model},
        "config": {"sha256": sha256_bytes(json.dumps(config, sort_keys=True).encode()),
                   "values": config},
        "packages": package_map(packages),
        "bend": bend,
        "git": git,
        "tensors": [],
    }


def save(directory, name, tensors, model, config, packages, bend=None, git=None, names=None):
    """Write the weights and their identity; `names` saves only those tensors."""
    os.makedirs(directory, exist_ok=True)
    keep = [n for n in tensors if names is None or n in names]
    weights_path = os.path.join(directory, name + ".bin")
    table, offset = [], 0
    with open(weights_path, "wb") as f:
        for tensor in keep:
            data = tensors[tensor]
            table.append({"name": tensor, "offset": offset, "bytes": len(data),
                          "sha256": sha256_bytes(data)})
            f.write(data)
            offset += len(data)
    sidecar = identity(weights_path, model, config, packages, bend=bend, git=git)
    sidecar["tensors"] = table
    sidecar["filtered"] = names is not None
    path = os.path.join(directory, name + ".json")
    with open(path, "w") as f:
        json.dump(sidecar, f, indent=2)
        f.write("\n")
    return weights_path, path


def load(directory, name, wanted=None):
    """The named tensors of a checkpoint; every tensor when `wanted` is None."""
    with open(os.path.join(directory, name + ".json")) as f:
        sidecar = json.load(f)
    out = {}
    with open(os.path.join(directory, name + ".bin"), "rb") as f:
        for tensor in sidecar.get("tensors") or []:
            if wanted is not None and tensor["name"] not in wanted:
                continue
            f.seek(tensor["offset"])
            data = f.read(tensor["bytes"])
            if sha256_bytes(data) != tensor["sha256"]:
                raise SystemExit(f"{name}: {tensor['name']} does not hash to its identity")
            out[tensor["name"]] = data
    return out, sidecar


def verify(saved, current):
    """Every field in which `current` is not the model `saved` was written by."""
    out = []
    if saved.get("schema") != current.get("schema"):
        out.append(f"schema: {saved.get('schema')} -> {current.get('schema')}")
    for field in ("structure", "declared"):
        was = (saved.get("model") or {}).get(field)
        now = (current.get("model") or {}).get(field)
        if was != now:
            out.append(f"model.{field}: {was} -> {now}  (the model differs)")
    was_cfg = (saved.get("config") or {}).get("values") or {}
    now_cfg = (current.get("config") or {}).get("values") or {}
    for key in sorted(set(was_cfg) | set(now_cfg)):
        if was_cfg.get(key) != now_cfg.get(key):
            out.append(f"config.{key}: {was_cfg.get(key)} -> {now_cfg.get(key)}")
    was_pkg = saved.get("packages") or {}
    now_pkg = current.get("packages") or {}
    for key in sorted(set(was_pkg) | set(now_pkg)):
        if was_pkg.get(key) != now_pkg.get(key):
            out.append(f"packages.{key}: {was_pkg.get(key)} -> {now_pkg.get(key)}")
    if saved.get("bend") != current.get("bend"):
        out.append(f"bend: {saved.get('bend')} -> {current.get('bend')}")
    if saved.get("weights") and current.get("weights") \
            and saved["weights"].get("sha256") != current["weights"].get("sha256"):
        out.append(f"weights.sha256: {(saved.get('weights') or {}).get('sha256')}"
                   f" -> {(current.get('weights') or {}).get('sha256')}")
    return out


def refuse(saved, current, name="the checkpoint"):
    """The sentence a resume prints, or None when it may proceed."""
    bad = verify(saved, current)
    if not bad:
        return None
    return (f"refusing to resume {name}: {len(bad)} field(s) differ\n  "
            + "\n  ".join(bad))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="a .json identity file")
    parser.add_argument("--against", help="another identity: print what differs")
    parser.add_argument("--filter", help="comma-separated tensor names to read")
    args = parser.parse_args()
    with open(args.checkpoint) as f:
        saved = json.load(f)
    if args.against:
        with open(args.against) as f:
            other = json.load(f)
        sentence = refuse(saved, other, args.checkpoint)
        print(sentence if sentence else "the same model, config, packages and weights")
        return 0 if sentence is None else 1
    tensors = saved.get("tensors") or []
    weights = saved.get("weights")
    print(f"{saved.get('schema')} · {len(tensors)} tensors · "
          + (f"{weights['bytes']} bytes" if weights else "no weights saved (identity only)"))
    print(f"model      {saved['model']['declared']}  ({saved['model']['structure'][:12]})")
    print(f"config     {saved['config']['values']}")
    print(f"packages   {saved['packages']}")
    print(f"bend       {saved.get('bend')} · git {saved.get('git')}")
    if args.filter:
        wanted = [n.strip() for n in args.filter.split(",") if n.strip()]
        print("tensors    " + ", ".join(wanted))
    else:
        print("tensors    " + ", ".join(f"{t['name']} {t['bytes']}B" for t in tensors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
