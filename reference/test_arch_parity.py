"""The matched GLM arm keeps the Mamba optimizer and schedule settings.

  reference/.venv/bin/python reference/test_arch_parity.py

Source-level gate for the settings a matched run depends on. It cannot prove
the numbers are a good idea; it proves the two arms still agree on them.
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLM = os.path.join(ROOT, "examples", "language", "glm53_flash", "train_h128.bend")
MAMBA = os.path.join(ROOT, "examples", "language", "mamba3",
                     "train_mamba3_tiny.bend")
RUNNER = os.path.join(ROOT, "reference", "arch_compare.py")


def body(path):
    return open(path).read()


def main():
    glm, mamba, runner = body(GLM), body(MAMBA), body(RUNNER)

    # The Mamba arm's constants are the source of truth.
    def constant(src, name):
        m = re.search(rf"def {name}\(\) -> (?:F32|Nat):\s*\n\s*([0-9.]+)n?", src)
        assert m, f"no {name} in {src}"
        return m.group(1)

    assert "0.9" in constant(mamba, "beta1")
    assert "0.999" in constant(mamba, "beta2")
    assert "0.01" in constant(mamba, "wd")
    assert "0.00000001" in constant(mamba, "eps")
    assert "5" in constant(mamba, "warmup_n")

    # The GLM arm's matched config uses the same values, with c1/c2 = 1-b.
    m = re.search(r"def cfg_matched\(\+lr: F32\) -> T\.OptCfg:\s*\n\s*"
                  r"T\.CAdamW\{lr, ([^}]+)\}", glm)
    assert m, "no cfg_matched in the GLM trainer"
    fields = [f.strip() for f in m.group(1).split(",")]
    assert fields == ["0.9", "0.999", "0.00000001", "0.01", "0.1", "0.001"], fields

    # The schedule is applied through BendTorch, with no state reset.
    assert "T.Train.with_cfg" in glm
    assert "m_lr(i, steps, peak)" in glm
    assert re.search(r"def m_floor\(\+peak: F32\) -> F32:\s*\n\s*\(peak / 10\.0", glm)
    assert '--matched' in glm

    # The runner passes the flag and keeps the unmatched banner set apart.
    assert '"--matched"' in runner
    assert "MATCHED_MD" in runner and "MATCHED_JSON" in runner
    assert "gradient clipping" in runner

    print("ARCH PARITY PASS")


if __name__ == "__main__":
    main()
