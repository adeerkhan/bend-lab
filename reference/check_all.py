"""Runs every MLBend verification. Exit code 0 = everything is fine.

  reference/.venv/bin/python reference/check_all.py
"""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
BEND_PIN = os.environ.get("BEND_PIN") or "2.0.38"
ENV = dict(os.environ, BEND_NO_TELEMETRY="1",
           PATH=os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"])
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    # the environment the packages are checked in: the same one the examples run in
    ENV["BEND_LIB"] = ENV_DIR

# This machine's clang wrapper loads libLLVM from the local tree, and Bend
# builds with it. `--verdict` must run without the path (Lean uses its own
# clang), so the fallback goes to the build checks only.
ENV_BUILD = dict(ENV)
LLVM_LIB = os.path.expanduser("~/.local/lib/llvm22")
if os.path.isdir(LLVM_LIB) and LLVM_LIB not in ENV_BUILD.get("LD_LIBRARY_PATH", "").split(":"):
    ENV_BUILD["LD_LIBRARY_PATH"] = ":".join(
        filter(None, [LLVM_LIB, ENV_BUILD.get("LD_LIBRARY_PATH")]))
results = []
FAST = "--fast" in sys.argv   # the cheap gate: the proofs and the frozen references only


def tidy(text):
    """The output of a program, as the tests compare it: no trailing spaces."""
    return "\n".join(l.rstrip() for l in text.strip().split("\n"))


def golden(path):
    """The expected output of a test file, in its #| lines (None when it has none)."""
    lines = [l[2:].strip() for l in open(path).read().split("\n") if l.startswith("#|")]
    return tidy("\n".join(lines)) if lines else None


def verdict_ok(out, code=None):
    """Whether a file checks.

    A package that does IO cannot have a clean verdict: the socket
    bindings of the http package are foreign code, and the verdict
    refuses a book that holds any. So the complaint must be that, and
    nothing else: a type error has "expected" and "observed" in it.
    """
    if "ALL PROOFS CHECK" in out:
        return True
    return ("rely on unsafe or foreign code" in out
            and "Error:" in out
            and "expected" not in out
            and "observed" not in out)


def native_binary(path):
    """Where the built form of a program goes: the interpreter's stack is small."""
    return os.path.join("/tmp", "mlbend-test-" + path.replace("/", "-").replace(".bend", ""))


def run(name, cmd, ok_if=None, timeout=1800, env=None):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env or ENV_BUILD, timeout=timeout)
    out = r.stdout + r.stderr
    ok = (r.returncode == 0) if ok_if is None else ok_if(out, r.returncode)
    results.append(ok)
    print(f"{'ok  ' if ok else 'ERROR'} {name} ({time.time() - t0:.0f}s)", flush=True)
    if not ok:
        print("     " + out.strip().replace("\n", "\n     ")[-600:], flush=True)
        if FAST:
            print(f"\n{sum(results)}/{len(results)} checks passed before the failure", flush=True)
            sys.exit(1)
    return out


def lean_here():
    """Whether the pinned Lean is on this machine: without it the kernel check is skipped."""
    try:
        lv = subprocess.run(["lean", "--version"], capture_output=True, text=True, env=ENV,
                            timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return "version 4.34.0" in lv.stdout


def check_toolchain():
    """Fail early, with a clear message, when the pinned toolchain is not the one in use."""
    bv = subprocess.run(BEND.split() + ["version"], capture_output=True, text=True, env=ENV).stdout.split()
    if not bv or bv[-1] != BEND_PIN:
        sys.exit(f"ERROR: bend {BEND_PIN} is required (found {' '.join(bv) or 'none'}); run scripts/setup.sh")
    if not lean_here():
        print("note: no Lean 4.34.0 here: the kernel check is skipped (CI runs it)")


def main():
    check_toolchain()
    packages = sorted(d for d in os.listdir(ROOT)
                      if os.path.isdir(os.path.join(ROOT, d)) and os.path.exists(os.path.join(ROOT, d, "main.bend")))

    # 0. the frozen references cover every task, so this gate runs without NumPy
    run("results/reference/*.json: every task has a frozen reference",
        [PY, "reference/references.py", "--check"])

    # 1. the proofs: the checker, then the audited kernel
    for pkg in packages:
        f = f"{pkg}/main.bend"
        run(f"{f}: ALL PROOFS CHECK", BEND.split() + [f], verdict_ok)
        if lean_here():
            run(f"{f} --verdict (the kernel, as Lean)", BEND.split() + [f, "--verdict"],
                verdict_ok, env=ENV)
        else:
            print(f"skip {f} --verdict (no Lean 4.34.0 on this machine; CI runs it)", flush=True)

    # 2. the packages with laws carry no @unsafe and no ?TODO
    for pkg in packages:
        f = os.path.join(ROOT, pkg, "main.bend")
        src = open(f).read()
        clean = not re.search(r"@unsafe|^\s*\?\w+|sorry", re.sub(r"#.*", "", src), re.M)
        results.append(clean)
        print(f"{'ok  ' if clean else 'ERROR'} {pkg}/main.bend: no @unsafe or ?TODO", flush=True)

    if FAST:
        ok = sum(results)
        print(f"\n{ok}/{len(results)} checks passed (--fast: the proofs and the frozen references)", flush=True)
        sys.exit(0 if ok == len(results) else 1)

    # 3. the programs that must be refused
    for pkg in packages:
        tdir = os.path.join(ROOT, pkg, "tests")
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            if not name.startswith("bad_") or not name.endswith(".bend"):
                continue
            f = f"{pkg}/tests/{name}"
            run(f"{f}: refused", BEND.split() + [f, "--check-only"],
                lambda o, c: "SOME PROOFS FAIL" in o and "Error:" in o)
            marked = "MUST FAIL" in open(os.path.join(ROOT, f)).read()
            results.append(marked)
            print(f"{'ok  ' if marked else 'ERROR'} {f}: says it must fail")

    # 4. the programs that must run, against the output pinned in the file
    for pkg in packages:
        tdir = os.path.join(ROOT, pkg, "tests")
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            if name.startswith("bad_") or not name.endswith(".bend"):
                continue
            f = f"{pkg}/tests/{name}"
            want = golden(os.path.join(ROOT, f))
            out = run(f"{f}: builds", BEND.split() + ["-o", native_binary(f), f],
                      lambda o, c: "SOME PROOFS FAIL" not in o and "Error:" not in o)
            if "SOME PROOFS FAIL" not in out and "Error:" not in out:
                out = run(f"{f}: runs", [native_binary(f)],
                          lambda o, c: "SOME PROOFS FAIL" not in o and "Error:" not in o)
            if want is not None:
                got = tidy(out)
                results.append(got == want)
                print(f"{'ok  ' if got == want else 'ERROR'} {f}: the output is the pinned one")
                if got != want:
                    print("     want:\n     " + want.replace("\n", "\n     "))
                    print("     got:\n     " + got.replace("\n", "\n     "))

    # 4b. every bench program type-checks. `bench/` is not run by the gate, and
    # three of its programs had rotted until someone compiled them (ISSUE-045).
    for dirpath, _, files in os.walk(os.path.join(ROOT, "bench")):
        for name in sorted(files):
            if name != "main.bend":
                continue
            f = os.path.relpath(os.path.join(dirpath, name), ROOT)
            run(f"{f}: type-checks", BEND.split() + [f, "--check-only"], verdict_ok)

    # 5. the numbers, against NumPy and scikit-learn
    for name in sorted(os.listdir(os.path.join(ROOT, "reference"))):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        run(f"reference/{name}: the numbers match", [PY, os.path.join("reference", name)])

    ok = sum(results)
    print(f"\n{ok}/{len(results)} checks passed", flush=True)
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
