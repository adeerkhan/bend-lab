"""Write a test's artifact only when what it records moved.

`make check` runs the tests that write `results/<task>/`; a test that rewrites
its record on every run leaves the tree dirty with a new `created_at` and a new
wall time while every measured number stays identical (ISSUE-033). The two
writers here compare the new content against the committed file first:

  save_text(path, text)   exact text: the loss tables and the SVG figures
  save_json(path, obj)    the object, after the fields that are not results
                          are dropped from the comparison

Both return True when the file was written and False when the committed file
already says the same thing. The caller prints what happened.

A field is dropped from the JSON comparison when it is a clock or the machine:
`created_at`, `git`, `machine`, `process`, `seconds` and any `*_seconds`,
`speed_ratio`, `repeat`. Those differ between two runs that measured the same
numbers, so comparing them would write every time. The field stays in the file;
only the comparison ignores it, and a moved number still writes the whole
record, clock included.
"""
import json
import os

# Not results: a timestamp, the checkout, the machine, the wall clock.
VOLATILE = {"created_at", "git", "machine", "process", "speed_ratio", "repeat"}


def _volatile(key):
    return key in VOLATILE or key == "seconds" or key.endswith("_seconds")


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if not _volatile(k)}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _canonical(obj):
    # Text, not objects: a NaN field compares equal to itself this way, and
    # the float repr is the one the file holds.
    return json.dumps(_strip(obj), indent=2)


def save_text(path, text):
    """Write `text` to `path` unless the file already holds it; return whether it wrote."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                if f.read() == text:
                    return False
        except OSError:
            pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def save_json(path, obj):
    """Write `obj` as JSON unless the committed file records the same numbers."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                if _canonical(json.load(f)) == _canonical(obj):
                    return False
        except (OSError, ValueError):
            pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2) + "\n")
    return True
