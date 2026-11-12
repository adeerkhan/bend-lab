"""The run store: every trace and record in one SQLite database.

  reference/runs.py --list
  reference/runs.py --sync TASK [--trace PATH] [--record PATH]

`schema` is one row per run and one row per step. `trace.py` starts a run and
appends each step; the dashboard reads the same database, so a run stays
clickable after the next run overwrites `results/<task>/trace.jsonl`.
Legacy single-file traces are imported on first use.

The database is `results/runs.sqlite3`, ignored by git. A missing row prints
unavailable; it is never filled with zero.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "results", "runs.sqlite3")
KNOWN = ("step", "tokens", "loss", "ms")


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS runs(
      id INTEGER PRIMARY KEY,
      task TEXT NOT NULL,
      label TEXT NOT NULL,
      trace TEXT NOT NULL,
      record TEXT,
      created REAL NOT NULL,
      updated REAL NOT NULL,
      status TEXT NOT NULL DEFAULT 'running',
      steps INTEGER NOT NULL DEFAULT 0,
      tokens INTEGER NOT NULL DEFAULT 0,
      loop_s REAL NOT NULL DEFAULT 0,
      first_loss REAL,
      last_loss REAL,
      best_loss REAL,
      extras TEXT NOT NULL DEFAULT '[]',
      record_json TEXT
    );
    CREATE TABLE IF NOT EXISTS steps(
      run_id INTEGER NOT NULL,
      step INTEGER NOT NULL,
      tokens INTEGER,
      loss REAL,
      ms REAL,
      extra TEXT NOT NULL DEFAULT '{}',
      PRIMARY KEY(run_id, step)
    );
    CREATE INDEX IF NOT EXISTS steps_run ON steps(run_id, step);
    """)
    con.commit()
    return con


def _extras(record):
    return sorted(k for k, v in record.items()
                  if k not in KNOWN and isinstance(v, (int, float))
                  and not isinstance(v, bool))


def start(task, trace, record=None, label=None):
    """Create a run and return its id. `trace` is the run's own file."""
    con = connect()
    now = time.time()
    label = label or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    cur = con.execute(
        "INSERT INTO runs(task,label,trace,record,created,updated) VALUES(?,?,?,?,?,?)",
        (task, label, os.path.relpath(trace, ROOT), record, now, now))
    run_id = cur.lastrowid
    con.commit()
    con.close()
    return run_id


def add_step(run_id, record):
    con = connect()
    step = int(record["step"])
    extra = {k: v for k, v in record.items()
             if k not in KNOWN and isinstance(v, (int, float))
             and not isinstance(v, bool)}
    con.execute(
        "INSERT OR REPLACE INTO steps(run_id,step,tokens,loss,ms,extra) "
        "VALUES(?,?,?,?,?,?)",
        (run_id, step, record.get("tokens"), record.get("loss"),
         record.get("ms"), json.dumps(extra, sort_keys=True)))
    row = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(tokens),0), COALESCE(SUM(ms),0), "
        "MIN(loss), MAX(loss) FROM steps WHERE run_id=?", (run_id,)).fetchone()
    count, tokens, ms, min_loss, max_loss = row
    first = con.execute("SELECT loss FROM steps WHERE run_id=? ORDER BY step LIMIT 1",
                        (run_id,)).fetchone()
    last = con.execute("SELECT loss FROM steps WHERE run_id=? ORDER BY step DESC LIMIT 1",
                       (run_id,)).fetchone()
    extras = sorted({k for r in con.execute(
        "SELECT extra FROM steps WHERE run_id=?", (run_id,)) for k in json.loads(r[0])})
    con.execute(
        "UPDATE runs SET updated=?, steps=?, tokens=?, loop_s=?, first_loss=?, "
        "last_loss=?, best_loss=?, extras=? WHERE id=?",
        (time.time(), count, tokens, ms / 1000.0,
         first[0] if first else None, last[0] if last else None,
         min_loss, json.dumps(extras), run_id))
    con.commit()
    con.close()


def record(task, record_path, label=None, status="done"):
    """Store a finished run that has a record but no per-step trace."""
    rec = None
    try:
        rec = json.load(open(record_path))
    except (OSError, ValueError):
        pass
    now = time.time()
    label = label or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    con = connect()
    cur = con.execute(
        "INSERT INTO runs(task,label,trace,record,created,updated,status,"
        "record_json) VALUES(?,?,?,?,?,?,?,?)",
        (task, label, "", os.path.relpath(record_path, ROOT), now, now, status,
         json.dumps(rec) if rec is not None else None))
    run_id = cur.lastrowid
    con.commit()
    con.close()
    return run_id


def finish(run_id, status="done", record_path=None):
    con = connect()
    rec = None
    if record_path and os.path.exists(record_path):
        try:
            rec = json.load(open(record_path))
        except (OSError, ValueError):
            rec = None
    con.execute("UPDATE runs SET status=?, updated=?, record_json=? WHERE id=?",
                (status, time.time(), json.dumps(rec) if rec else None, run_id))
    con.commit()
    con.close()


def import_file(task, trace, record=None, label=None, replace=True):
    """Import a single-file trace as one run. Returns the id, or None."""
    records = []
    try:
        for line in open(trace):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if all(k in r for k in KNOWN):
                records.append(r)
    except OSError:
        return None
    if not records:
        return None
    rel = os.path.relpath(trace, ROOT)
    con = connect()
    old = con.execute("SELECT id FROM runs WHERE trace=? AND task=? ORDER BY id DESC LIMIT 1",
                      (rel, task)).fetchone()
    con.close()
    if old and replace:
        run_id = old[0]
        con = connect()
        con.execute("DELETE FROM steps WHERE run_id=?", (run_id,))
        con.execute("UPDATE runs SET status='imported', updated=?, record=?, label=? WHERE id=?",
                    (time.time(), record, label or rel, run_id))
        con.commit()
        con.close()
    else:
        run_id = start(task, trace, record, label or rel)
        con = connect()
        con.execute("UPDATE runs SET status='imported' WHERE id=?", (run_id,))
        con.commit()
        con.close()
    for r in records:
        add_step(run_id, r)
    finish(run_id, "imported", record)
    return run_id


def list_runs(task=None):
    con = connect()
    q = ("SELECT id,task,label,trace,record,created,updated,status,steps,tokens,"
         "loop_s,first_loss,last_loss,best_loss,extras FROM runs")
    args = ()
    if task:
        q += " WHERE task=?"
        args = (task,)
    q += " ORDER BY id DESC LIMIT 200"
    rows = con.execute(q, args).fetchall()
    con.close()
    keys = ("id", "task", "label", "trace", "record", "created", "updated",
            "status", "steps", "tokens", "loop_s", "first_loss", "last_loss",
            "best_loss", "extras")
    return [dict(zip(keys, row)) for row in rows]


def get_run(run_id):
    con = connect()
    row = con.execute(
        "SELECT id,task,label,trace,record,created,updated,status,steps,tokens,"
        "loop_s,first_loss,last_loss,best_loss,extras,record_json "
        "FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        con.close()
        return None
    keys = ("id", "task", "label", "trace", "record", "created", "updated",
            "status", "steps", "tokens", "loop_s", "first_loss", "last_loss",
            "best_loss", "extras")
    run = dict(zip(keys, row[:-1]))
    run["record"] = json.loads(row[-1]) if row[-1] else None
    run["extras"] = json.loads(run["extras"])
    steps = []
    for step, tokens, loss, ms, extra in con.execute(
            "SELECT step,tokens,loss,ms,extra FROM steps WHERE run_id=? ORDER BY step",
            (run_id,)):
        d = {"step": step, "tokens": tokens, "loss": loss, "ms": ms}
        d.update(json.loads(extra))
        steps.append(d)
    con.close()
    run["records"] = steps
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sync", metavar="TASK")
    ap.add_argument("--trace")
    ap.add_argument("--record")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.sync:
        trace = args.trace or os.path.join(ROOT, "results", args.sync, "trace.jsonl")
        record = args.record or os.path.join(ROOT, "results", "latest", args.sync + ".json")
        rid = import_file(args.sync, trace, record)
        print(f"imported run {rid}" if rid else "no records")
        return 0 if rid else 1
    runs = list_runs()
    if args.json:
        print(json.dumps(runs, indent=1))
    else:
        for r in runs:
            print(f"{r['id']:4d} {r['task']:22s} {r['status']:9s} "
                  f"{r['steps']:5d} steps  last {r['last_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
