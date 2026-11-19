"""The UNet record writer resolves its task and run from the environment.

  reference/.venv/bin/python reference/test_unet_record.py

Commit c31dd1d advertised UNET_RUN and UNET_TASK but left NAME and DEFAULT_RUN
undefined, so the writer died before it could register a --four run. This gate
renders the default and a named run and checks what each one writes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
SCRIPT = os.path.join(ROOT, "reference/unet_record.py")
DEFAULT = os.path.join(ROOT, "results", "diffusion-unet", "trained", "run.json")


def render(task=None, run=None):
    env = dict(os.environ)
    if task:
        env["UNET_TASK"] = task
    else:
        env.pop("UNET_TASK", None)
    if run:
        env["UNET_RUN"] = run
    else:
        env.pop("UNET_RUN", None)
    p = subprocess.run([PY, SCRIPT], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


def main():
    # The module-level names exist and default to the original values.
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import unet_record as m; "
        "print(m.TASK); print(m.DEFAULT_RUN); print(m.OUT)" % (
            os.path.join(ROOT, "reference"),)
    )
    env = dict(os.environ)
    env.pop("UNET_TASK", None)
    env.pop("UNET_RUN", None)
    p = subprocess.run([PY, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    task, default, out = p.stdout.strip().splitlines()
    assert task == "unet-flowers", task
    assert default == DEFAULT, default
    assert out == os.path.join(ROOT, "results", "latest", "unet-flowers.json"), out

    # A named run writes a named record and keeps the original one untouched.
    name = "unet-record-test-" + str(os.getpid())
    target = os.path.join(ROOT, "results", "latest", name + ".json")
    original = os.path.join(ROOT, "results", "latest", "unet-flowers.json")
    before = open(original, "rb").read() if os.path.exists(original) else None
    try:
        stdout = render(task=name)
        assert os.path.exists(target), stdout
        rec = json.load(open(target))
        assert rec["task"]["name"] == name, rec["task"]["name"]
        assert rec["run"]["kind"] == "migration", rec["run"]["kind"]
        assert rec["passed"] is True, rec["passed"]
        assert "0.5" in stdout
        if before is not None:
            assert open(original, "rb").read() == before, "the default record moved"
    finally:
        if os.path.exists(target):
            os.remove(target)

    # A non-default RUN is registered as four-pair, with its own source path.
    name = "unet-record-four-test-" + str(os.getpid())
    target = os.path.join(ROOT, "results", "latest", name + ".json")
    other = os.path.join(ROOT, "results", "diffusion-unet", "trained", "run.json")
    try:
        render(task=name, run=other)
        rec = json.load(open(target))
        assert rec["run"]["kind"] == ("four-pair" if other != DEFAULT else "migration")
        assert rec["run"]["source"] == os.path.relpath(other, ROOT)
    finally:
        if os.path.exists(target):
            os.remove(target)

    print("UNET RECORD PASS")


if __name__ == "__main__":
    main()
