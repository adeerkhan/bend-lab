"""The run store: every trace, record, process and sample in one SQLite database.

  reference/runs.py --list
  reference/runs.py --import-latest
  reference/runs.py --sync TASK [--trace PATH] [--record PATH]

One row per run, one per step and one per system sample. `trace.py` starts a
run and appends each step; `experiment.py` registers a run before the work and
finishes it after; `session.py` wraps any command. The dashboard reads the same
database, so a run stays clickable after the next one overwrites
`results/<task>/trace.jsonl`, and several runs can be live at once.

The database is `results/runs.sqlite3` (or `MLBEND_RUNS_DB`), ignored by git. A
missing measurement is `NULL` and prints `unavailable`; it is never filled with
zero.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.environ.get("MLBEND_RUNS_DB") or os.path.join(ROOT, "results", "runs.sqlite3")
KNOWN = ("step", "tokens", "loss", "ms")
RESERVED = set(KNOWN) | {"train_loss", "eval_loss", "val_loss", "validation_loss"}
META = (
    ("pid", "INTEGER"), ("command", "TEXT"), ("config", "TEXT"),
    ("params", "REAL"), ("kind", "TEXT DEFAULT 'run'"), ("host", "TEXT"),
    ("started", "REAL"), ("finished", "REAL"), ("exit_code", "INTEGER"),
    ("error", "TEXT"), ("git_commit", "TEXT"), ("versions", "TEXT"),
    ("eval_loss", "REAL"),
)


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
    CREATE TABLE IF NOT EXISTS samples(
      run_id INTEGER NOT NULL,
      t REAL NOT NULL,
      cpu REAL,
      rss_kb INTEGER,
      hwm_kb INTEGER,
      gpu_util REAL,
      gpu_mem_kb INTEGER,
      gpu_proc_kb INTEGER,
      PRIMARY KEY(run_id, t)
    );
    CREATE INDEX IF NOT EXISTS samples_run ON samples(run_id, t);
    """)
    cols = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
    for name, decl in META:
        if name not in cols:
            con.execute(f"ALTER TABLE runs ADD COLUMN {name} {decl}")
    con.commit()
    return con


def _rel(path):
    if not path:
        return ""
    return path if not os.path.isabs(path) else os.path.relpath(path, ROOT)


def _record_info(rec):
    """The model's parameter count and the configuration a record carries."""
    if not isinstance(rec, dict):
        return None, {}
    params = None
    metrics = rec.get("metrics") or {}
    value = metrics.get("params")
    if isinstance(value, dict):
        params = value.get("bend")
    elif isinstance(value, (int, float)):
        params = value
    for key in ("params", "parameters", "n_parameters"):
        if params is None:
            value = (rec.get("counts") or {}).get(key)
            if isinstance(value, (int, float)):
                params = value
    config = {}
    for key in ("task", "variant", "trainer", "seeds", "git", "bend",
                "packages", "stages_source", "machine"):
        if key in rec:
            config[key] = rec[key]
    return params, config


def _eval_of(rec):
    if not isinstance(rec, dict):
        return None
    for key in ("eval_loss", "val_loss", "validation_loss"):
        value = rec.get(key)
        metrics = rec.get("metrics") or {}
        if value is None:
            value = metrics.get(key)
        if isinstance(value, dict):
            value = value.get("bend")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def start(task, trace, record=None, label=None, pid=None, command=None,
          config=None, params=None, kind="run", status="running",
          git_commit=None, versions=None):
    """Create a run and return its id. `trace` is the run's own file."""
    con = connect()
    now = time.time()
    label = label or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    cur = con.execute(
        "INSERT INTO runs(task,label,trace,record,created,updated,status,"
        "pid,command,config,params,kind,host,started,git_commit,versions) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (task, label, _rel(trace), _rel(record), now, now, status,
         pid, command, json.dumps(config) if config else None, params, kind,
         socket.gethostname(), now, git_commit,
         json.dumps(versions) if versions else None))
    run_id = cur.lastrowid
    con.commit()
    con.close()
    return run_id


def set_pid(run_id, pid):
    con = connect()
    con.execute("UPDATE runs SET pid=?, updated=? WHERE id=?", (pid, time.time(), run_id))
    con.commit()
    con.close()


def add_step(run_id, record):
    con = connect()
    step = int(record["step"])
    extra = {k: v for k, v in record.items()
             if k not in RESERVED and isinstance(v, (int, float))
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


def record(task, record_path, label=None, status="done", command=None):
    """Store a finished run that has a record but no per-step trace."""
    rec = None
    try:
        rec = json.load(open(record_path))
    except (OSError, ValueError):
        pass
    now = time.time()
    label = label or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    rel = _rel(record_path)
    con = connect()
    old = con.execute("SELECT id FROM runs WHERE task=? AND record=? ORDER BY id DESC LIMIT 1",
                      (task, rel)).fetchone()
    if old:
        con.close()
        return old[0]
    params, config = _record_info(rec)
    cur = con.execute(
        "INSERT INTO runs(task,label,trace,record,created,updated,status,kind,"
        "pid,command,config,params,host,started,finished,record_json,eval_loss) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (task, label, "", rel, now, now, status, "experiment", None, command,
         json.dumps(config) if config else None, params, socket.gethostname(),
         now, now, json.dumps(rec) if rec is not None else None, _eval_of(rec)))
    run_id = cur.lastrowid
    con.commit()
    con.close()
    return run_id


def finish(run_id, status="done", record_path=None, exit_code=None, error=None,
           eval_loss=None):
    con = connect()
    now = time.time()
    row = con.execute("SELECT config, record FROM runs WHERE id=?", (run_id,)).fetchone()
    config = {}
    if row and row[0]:
        try:
            config = json.loads(row[0]) or {}
        except ValueError:
            config = {}
    rec = None
    if record_path and os.path.exists(record_path):
        try:
            rec = json.load(open(record_path))
        except (OSError, ValueError):
            rec = None
    params, found = _record_info(rec)
    config.update(found)
    ev = eval_loss if eval_loss is not None else _eval_of(rec)
    con.execute(
        "UPDATE runs SET status=?, updated=?, finished=?, record=COALESCE(?, record), "
        "record_json=COALESCE(?, record_json), config=?, params=COALESCE(?, params), "
        "eval_loss=COALESCE(?, eval_loss), exit_code=?, error=? WHERE id=?",
        (status, now, now, _rel(record_path) or None,
         json.dumps(rec) if rec is not None else None,
         json.dumps(config) if config else None, params, ev, exit_code, error, run_id))
    con.commit()
    con.close()


def sample(run_id, cpu=None, rss_kb=None, hwm_kb=None, gpu_util=None,
           gpu_mem_kb=None, gpu_proc_kb=None, t=None):
    con = connect()
    con.execute(
        "INSERT OR REPLACE INTO samples(run_id,t,cpu,rss_kb,hwm_kb,gpu_util,"
        "gpu_mem_kb,gpu_proc_kb) VALUES(?,?,?,?,?,?,?,?)",
        (run_id, t or time.time(), cpu, rss_kb, hwm_kb, gpu_util, gpu_mem_kb,
         gpu_proc_kb))
    con.commit()
    con.close()


def peak_sample(run_id):
    """The largest CPU, RSS, HWM and GPU memory one run reached."""
    con = connect()
    row = con.execute(
        "SELECT MAX(cpu), MAX(rss_kb), MAX(hwm_kb), MAX(gpu_mem_kb) "
        "FROM samples WHERE run_id=?", (run_id,)).fetchone()
    con.close()
    if not row or all(value is None for value in row):
        return None
    keys = ("cpu", "rss_kb", "hwm_kb", "gpu_mem_kb")
    return dict(zip(keys, row))


def samples(run_id, limit=7200):
    con = connect()
    rows = con.execute(
        "SELECT t,cpu,rss_kb,hwm_kb,gpu_util,gpu_mem_kb,gpu_proc_kb FROM samples "
        "WHERE run_id=? ORDER BY t DESC LIMIT ?", (run_id, limit)).fetchall()
    con.close()
    keys = ("t", "cpu", "rss_kb", "hwm_kb", "gpu_util", "gpu_mem_kb", "gpu_proc_kb")
    return [dict(zip(keys, row)) for row in reversed(rows)]


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
    rel = _rel(trace)
    con = connect()
    old = con.execute("SELECT id FROM runs WHERE trace=? AND task=? ORDER BY id DESC LIMIT 1",
                      (rel, task)).fetchone()
    con.close()
    if old and replace:
        run_id = old[0]
        con = connect()
        con.execute("DELETE FROM steps WHERE run_id=?", (run_id,))
        con.execute("UPDATE runs SET status='imported', updated=?, record=?, label=?, "
                    "finished=? WHERE id=?",
                    (time.time(), _rel(record), label or rel, time.time(), run_id))
        con.commit()
        con.close()
    else:
        run_id = start(task, trace, record, label or rel, kind="trace",
                       status="imported")
    for r in records:
        add_step(run_id, r)
    finish(run_id, "imported", record)
    return run_id


def list_runs(task=None, status=None, limit=200):
    con = connect()
    q = ("SELECT id,task,label,trace,record,created,updated,status,steps,tokens,"
         "loop_s,first_loss,last_loss,best_loss,extras,pid,command,config,params,"
         "kind,host,started,finished,exit_code,error,git_commit,versions,eval_loss FROM runs")
    where, args = [], []
    if task:
        where.append("task=?")
        args.append(task)
    if status:
        where.append("status=?")
        args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = con.execute(q, args).fetchall()
    con.close()
    keys = ("id", "task", "label", "trace", "record", "created", "updated",
            "status", "steps", "tokens", "loop_s", "first_loss", "last_loss",
            "best_loss", "extras", "pid", "command", "config", "params", "kind",
            "host", "started", "finished", "exit_code", "error", "git_commit",
            "versions", "eval_loss")
    out = []
    for row in rows:
        run = dict(zip(keys, row))
        for key in ("extras", "config", "versions"):
            try:
                run[key] = json.loads(run[key]) if run[key] else ([] if key == "extras" else {})
            except ValueError:
                run[key] = [] if key == "extras" else {}
        out.append(run)
    return out


def get_run(run_id):
    con = connect()
    row = con.execute(
        "SELECT id,task,label,trace,record,created,updated,status,steps,tokens,"
        "loop_s,first_loss,last_loss,best_loss,extras,record_json,pid,command,"
        "config,params,kind,host,started,finished,exit_code,error,git_commit,"
        "versions,eval_loss FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        con.close()
        return None
    keys = ("id", "task", "label", "trace", "record", "created", "updated",
            "status", "steps", "tokens", "loop_s", "first_loss", "last_loss",
            "best_loss", "extras", "record_x", "pid", "command", "config",
            "params", "kind", "host", "started", "finished", "exit_code",
            "error", "git_commit", "versions", "eval_loss")
    run = dict(zip(keys, row))
    run["record_path"] = run.pop("record")
    run["record"] = json.loads(run.pop("record_x")) if run.get("record_x") else None
    for key in ("extras", "config", "versions"):
        try:
            run[key] = json.loads(run[key]) if run[key] else ([] if key == "extras" else {})
        except ValueError:
            run[key] = [] if key == "extras" else {}
    steps = []
    for step, tokens, loss, ms, extra in con.execute(
            "SELECT step,tokens,loss,ms,extra FROM steps WHERE run_id=? ORDER BY step",
            (run_id,)):
        d = {"step": step, "tokens": tokens, "loss": loss, "ms": ms}
        d.update(json.loads(extra))
        steps.append(d)
    run["records"] = steps
    run["samples"] = samples(run_id)
    con.close()
    return run


def reconcile():
    """Mark runs whose process is gone as lost, so the page has no ghosts."""
    con = connect()
    now = time.time()
    changed = []
    for run_id, pid in con.execute("SELECT id,pid FROM runs WHERE status='running'"):
        alive = False
        if pid:
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                alive = False
        if not alive:
            con.execute("UPDATE runs SET status='lost', updated=?, finished=?, "
                        "error=? WHERE id=?",
                        (now, now, "process gone", run_id))
            changed.append(run_id)
    con.commit()
    con.close()
    return changed


def backfill():
    """Fill params and config on rows recorded before those columns existed."""
    con = connect()
    changed = 0
    for run_id, raw, config_raw, params in con.execute(
            "SELECT id,record_json,config,params FROM runs WHERE record_json IS NOT NULL"):
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        found_params, found = _record_info(rec)
        config = {}
        try:
            config = json.loads(config_raw) if config_raw else {}
        except ValueError:
            config = {}
        config.update(found)
        if found or (found_params is not None and params is None):
            con.execute("UPDATE runs SET params=COALESCE(?,params), config=? WHERE id=?",
                        (found_params, json.dumps(config) if config else None, run_id))
            changed += 1
    con.commit()
    con.close()
    return changed


def import_latest():
    """Store every `results/latest/<task>.json` as a record-only run."""
    latest = os.path.join(ROOT, "results", "latest")
    ids = []
    for name in sorted(os.listdir(latest)):
        if not name.endswith(".json") or name.endswith("-numpy.json"):
            continue
        path = os.path.join(latest, name)
        try:
            rec = json.load(open(path))
        except (OSError, ValueError):
            continue
        task = (rec.get("task") or {}).get("name") or name[:-5]
        ids.append(record(task, path, f"{task} · imported record"))
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--import-latest", action="store_true",
                    help="store every results/latest/<task>.json as a run")
    ap.add_argument("--backfill", action="store_true",
                    help="fill params and config on older record-only rows")
    ap.add_argument("--sync", metavar="TASK")
    ap.add_argument("--trace")
    ap.add_argument("--record")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.import_latest:
        ids = import_latest()
        print(f"imported {len(ids)} record-only runs")
        return 0
    if args.backfill:
        print(f"updated {backfill()} runs")
        return 0
    if args.sync:
        trace = args.trace or os.path.join(ROOT, "results", args.sync, "trace.jsonl")
        record = args.record or os.path.join(ROOT, "results", "latest", args.sync + ".json")
        rid = import_file(args.sync, trace, record)
        print(f"imported run {rid}" if rid else "no records")
        return 0 if rid else 1
    rows = list_runs()
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{r['id']:4d} {r['task']:22s} {r['kind']:10s} {r['status']:9s} "
                  f"{r['steps']:5d} steps  last {r['last_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
