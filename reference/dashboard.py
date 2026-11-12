"""The in-house training dashboard: every run, live, on a local port.

  reference/dashboard.py --task glm53-hybrid128 --port 8787
  reference/dashboard.py --task glm53-hybrid128 --port 8787 --open

One page for every run in `results/runs.sqlite3` (`reference/runs.py`), not one
task at a time. The overview lists running and finished runs with their CPU (host share and
cores), resident memory (PSS over the process tree), GPU memory and parameter
count; selecting a run shows its loss
and step time, its extra signals, its system samples, its steps, and the record
its program wrote. Several sessions can be live at once because the sampler
walks each run's process tree, so a Python test that spawns Bend is sampled as
one run.

Standard library only. The page polls `/api/overview` and `/api/state` once a
second; there is no WebSocket, no CDN and no build step. `nvidia-smi` supplies
the GPU numbers, and a machine without it prints `unavailable`, never zero.
`--max-seconds` is the budget the page shows (default one hour); the server does
not kill a run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import runs  # noqa: E402

KEYS = ("step", "tokens", "loss", "ms")
KNOWN = set(KEYS) | {"time", "timestamp", "wall_ms", "elapsed",
                     "train_loss", "eval_loss", "val_loss", "validation_loss"}


# ---------------------------------------------------------------------------
# the machine: CPU, memory, GPU, and a process tree


def cpu_times():
    try:
        parts = open("/proc/stat").readline().split()[1:]
        values = [int(v) for v in parts]
        total = sum(values[:8])
        idle = values[3] + values[4]
        return total, idle
    except (OSError, ValueError, IndexError):
        return None


def memory():
    try:
        info = {}
        for line in open("/proc/meminfo"):
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])
        total = info.get("MemTotal")
        available = info.get("MemAvailable", info.get("MemFree"))
        if total is None or available is None:
            return None
        return {"total_kb": total, "used_kb": total - available,
                "available_kb": available}
    except (OSError, ValueError):
        return None


def gpu_stats():
    """(overall, {pid: used_mb}) from nvidia-smi, or (None, {}) without it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if out.returncode != 0 or not out.stdout.strip():
            return None, {}
        parts = [p.strip() for p in out.stdout.splitlines()[0].split(",")]
        overall = {"util": min(100.0, max(0.0, float(parts[0]))),
                   "used_mb": float(parts[1]), "total_mb": float(parts[2])}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, {}
    procs = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                procs[int(parts[0])] = float(parts[1])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return overall, procs


def _stat(pid):
    """ppid, cpu ticks, rss pages from /proc/<pid>/stat."""
    try:
        data = open(f"/proc/{pid}/stat").read()
        end = data.rfind(")")
        fields = data[end + 2:].split()
        return int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _status(pid):
    values = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                values[key] = int(value.split()[0])
    except (OSError, ValueError):
        pass
    # PSS divides a shared page among its users, so a tree sum is not inflated
    # by the same library mapped in every compiler or worker process.
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            if line.startswith("Pss:"):
                values["Pss"] = int(line.split()[1])
                break
    except (OSError, ValueError):
        pass
    return values


def process_tree(pid):
    """The pid and all its descendants, from the /proc stat table."""
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        stat = _stat(entry)
        if stat is None:
            continue
        children.setdefault(stat[0], []).append(int(entry))
    out, stack = [], [int(pid)]
    while stack:
        current = stack.pop()
        if current in out:
            continue
        out.append(current)
        stack.extend(children.get(current, []))
    return out


# ---------------------------------------------------------------------------
# the sampler: one row per running run per second

class Sampler(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        try:
            self.ncpu = len(os.sched_getaffinity(0))
        except AttributeError:
            self.ncpu = os.cpu_count() or 1
        self.live = {}
        self.system = {}
        self._prev_cpu = None
        self._prev_proc = {}
        self._gpu = (None, {})
        self._gpu_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _overall(self):
        now = time.time()
        total, idle = cpu_times() or (None, None)
        cpu = None
        if total is not None and self._prev_cpu is not None:
            dt = total - self._prev_cpu[0]
            di = idle - self._prev_cpu[1]
            if dt > 0:
                cpu = 100.0 * (1.0 - di / dt)
        if total is not None:
            self._prev_cpu = (total, idle)
        mem = memory()
        if now - self._gpu_at > 2.0:
            self._gpu = gpu_stats()
            self._gpu_at = now
        gpu, procs = self._gpu
        return {"cpu": cpu, "memory": mem, "gpu": gpu, "gpu_proc": procs}

    def _tree_stats(self, pid, now):
        pids = process_tree(pid)
        if not pids:
            return None
        ticks = 0
        rss = 0
        hwm = 0
        cpu = 0.0
        for one in pids:
            stat = _stat(one)
            if stat is None:
                continue
            ticks += stat[1]
            status = _status(one)
            rss += status.get("Pss", status.get(
                "VmRSS", stat[2] * (os.sysconf("SC_PAGE_SIZE") // 1024)))
            hwm = max(hwm, status.get("VmHWM", 0))
            prev = self._prev_proc.get(one)
            if prev is not None:
                dt = now - prev[1]
                if dt > 0:
                    hz = os.sysconf("SC_CLK_TCK")
                    cpu += 100.0 * (stat[1] - prev[0]) / hz / dt
            self._prev_proc[one] = (stat[1], now)
        if not ticks and not rss:
            return None
        # `cpu` is the sum over the tree in percent of one core; the host
        # share is that divided by the cores this process may run on.
        return {"pid": int(pid), "pids": len(pids),
                "cpu": min(100.0, cpu / self.ncpu),
                "cores": min(float(self.ncpu), cpu / 100.0),
                "rss_kb": rss, "hwm_kb": hwm or None}

    def tick(self):
        now = time.time()
        system = self._overall()
        gpu, gpu_procs = system["gpu"], system["gpu_proc"]
        live = {}
        try:
            running = runs.list_runs(status="running", limit=200)
        except Exception:
            running = []
        for run in running:
            pid = run.get("pid")
            if not pid:
                continue
            try:
                os.kill(int(pid), 0)
            except OSError:
                try:
                    runs.finish(run["id"], "lost", error="process gone")
                except Exception:
                    pass
                continue
            stats = self._tree_stats(pid, now)
            if stats is None:
                continue
            gpu_proc = sum(gpu_procs.get(p, 0.0) for p in process_tree(pid))
            item = dict(stats)
            item.update({
                "gpu_util": gpu["util"] if gpu else None,
                "gpu_used_mb": gpu["used_mb"] if gpu else None,
                "gpu_proc_mb": gpu_proc or None,
                "at": now,
            })
            live[run["id"]] = item
            try:
                runs.sample(
                    run["id"], cpu=item["cpu"], rss_kb=item["rss_kb"],
                    hwm_kb=item["hwm_kb"],
                    gpu_util=item["gpu_util"],
                    gpu_mem_kb=(item["gpu_proc_mb"] * 1024
                                if item["gpu_proc_mb"] else None),
                    t=now)
            except Exception:
                pass
        with self._lock:
            self.live = live
            self.system = {
                "cpu": system["cpu"],
                "ncpu": self.ncpu,
                "memory": system["memory"],
                "gpu": system["gpu"],
                "at": now,
            }

    def run(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                print(f"dashboard: sampler: {exc}")
            self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# state


def summary_of(records):
    summary = {"steps": len(records), "tokens": 0, "loop_s": 0.0,
               "median_ms": None, "tokens_s": None}
    if records:
        losses = [float(r["loss"]) for r in records if r.get("loss") is not None]
        ms = [float(r["ms"]) for r in records if r.get("ms") is not None]
        summary["tokens"] = int(sum(int(r.get("tokens") or 0) for r in records))
        summary["loop_s"] = sum(ms) / 1000.0
        summary["median_ms"] = statistics.median(ms) if ms else None
        summary["tokens_s"] = (summary["tokens"] / summary["loop_s"]
                               if summary["loop_s"] else None)
        if losses:
            summary["first"] = losses[0]
            summary["last"] = losses[-1]
            summary["best"] = min(losses)
            summary["worst"] = max(losses)
    return summary


def downsample(rows, limit=600):
    """Keep a sample series readable: an even stride plus the last point."""
    if len(rows) <= limit:
        return rows
    step = max(1, len(rows) // limit)
    out = rows[::step]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def extras_of(records):
    keys = []
    for record in records:
        for key, value in record.items():
            if key not in KNOWN and isinstance(value, (int, float)) \
                    and not isinstance(value, bool) and key not in keys:
                keys.append(key)
    return sorted(keys)


def read_trace(path, limit=20000):
    records = []
    if not path:
        return records
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if all(k in record for k in KEYS):
                    records.append(record)
    except OSError:
        return []
    return records[-limit:]


def read_json(path):
    if not path:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def state(run_id, fallback):
    """The selected run's record, steps, samples and live process stats."""
    run = runs.get_run(run_id) if run_id else None
    if run is None and run_id is None:
        latest = runs.list_runs(limit=1)
        run = runs.get_run(latest[0]["id"]) if latest else None
    if run is None:
        records = read_trace(fallback["trace"])
        record = read_json(fallback["record"])
        path = fallback["record"]
        return {
            "run": None, "records": records, "record": record,
            "extras": extras_of(records), "summary": summary_of(records),
            "samples": [], "proc": None, "budget": fallback["budget"],
            "record_path": (os.path.relpath(path, ROOT) if os.path.isabs(path)
                            else path) if path else "",
        }
    records = run["records"]
    all_samples = run.get("samples") or []
    samples = downsample(all_samples)
    proc = fallback["sampler"].live.get(run["id"])
    if proc is None and all_samples:
        cpu = max((x["cpu"] or 0) for x in all_samples)
        rss = max((x["rss_kb"] or 0) for x in all_samples)
        hwm = max((x["hwm_kb"] or 0) for x in all_samples)
        hwm = max(hwm, rss)
        gpu = max((x["gpu_mem_kb"] or 0) for x in all_samples)
        proc = {"cpu": cpu, "rss_kb": rss, "hwm_kb": hwm or rss,
                "cores": cpu * fallback["sampler"].ncpu / 100.0,
                "gpu_proc_mb": gpu / 1024 if gpu else None,
                "peak": True}
    return {
        "run": {k: run[k] for k in (
            "id", "task", "label", "status", "kind", "params", "pid", "host",
            "command", "created", "updated", "started", "finished", "steps",
            "tokens", "loop_s", "first_loss", "last_loss", "best_loss",
            "exit_code", "error", "config", "versions", "git_commit",
            "eval_loss", "record_path")},
        "records": records,
        "record": run.get("record"),
        "extras": extras_of(records),
        "summary": summary_of(records),
        "samples": samples,
        "proc": proc,
        "budget": fallback["budget"],
        "record_path": run.get("record_path") or "",
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLBend · runs</title>
<style>
:root{--bg:#f0eee6;--panel:#faf8f2;--line:#dad5c8;--fg:#2e2d2a;--muted:#6f6a60;
--accent:#7d70b3;--accent2:#4f8f86;--warn:#a9772c;--bad:#b3564f;--pink:#a879a8;
--ok:#2f7f5f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;padding:14px 20px;
border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:3}
h1{font-size:18px;margin:0;letter-spacing:.02em}.mark{display:inline-block;
width:10px;height:18px;background:var(--accent);margin-left:4px;vertical-align:-3px}
h2{font-size:12px;color:var(--muted);
margin:0 0 8px;text-transform:uppercase;letter-spacing:.08em}
main{max-width:1680px;margin:0 auto;padding:14px 20px 40px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:12px 14px;min-width:0;box-shadow:0 1px 0 rgba(45,42,35,.035)}.s2{grid-column:span 2}.s3{grid-column:span 3}
.s4{grid-column:span 4}.s6{grid-column:span 6}.s8{grid-column:span 8}
.s12{grid-column:span 12}
.big{font-size:22px;font-variant-numeric:tabular-nums}.muted{color:var(--muted)}
.row{display:flex;gap:18px;flex-wrap:wrap}.row>div{min-width:92px}
canvas{width:100%;height:140px;display:block}.tall canvas{height:180px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:5px 7px;border-bottom:1px solid var(--line);
vertical-align:middle;white-space:nowrap}
th{color:var(--muted);font-weight:500}td.num,th.num{text-align:right;
font-variant-numeric:tabular-nums}
#runstable{max-height:350px;overflow:auto;border:1px solid var(--line);
border-radius:8px}
#runstable table{min-width:1120px}
#runstable th{position:sticky;top:0;background:var(--panel);z-index:1}
#config{max-height:360px;overflow:auto}
#config td{white-space:normal;word-break:break-word}
#config td:first-child{color:var(--muted);width:84px}
.chk{float:right;text-transform:none;letter-spacing:0;font-weight:400;
color:var(--muted);font-size:11px}
.filter{float:right;background:var(--bg);border:1px solid var(--line);
color:var(--fg);border-radius:6px;padding:2px 8px;font:11px ui-monospace;
text-transform:none;letter-spacing:0;width:190px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;
background:var(--muted);margin-right:6px;vertical-align:middle}
.dot.live{background:var(--ok);box-shadow:0 0 0 3px rgba(47,127,95,.14)}
.st-running{color:var(--ok);font-weight:600}
.st-done{color:var(--muted)}.st-imported{color:var(--accent2)}
.st-failed{color:var(--bad)}.st-lost{color:var(--warn)}
tr.run{cursor:pointer}tr.run:hover td{background:#ece7da}
tr.run.running td{background:#eef4ee}
tr.run.sel td{background:#e6dfcf}
tr.run.sel td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
border:1px solid var(--line);color:var(--muted)}
.hint{color:var(--muted);font-size:12px;margin:6px 0 0}
a{color:var(--accent2)}
@media(max-width:900px){.s2,.s3,.s4,.s6,.s8{grid-column:span 12}
 main{padding:10px 10px 30px}}
</style></head><body>
<header><h1>MLBend<span class="mark"></span></h1>
<span class="muted">runs &amp; live sessions</span>
<span id="status" class="muted"><i class="dot"></i>connecting…</span>
<span id="budget" class="muted"></span>
<span class="muted">auto-refresh 1 s</span></header>
<main class="grid">
<div class="card s2"><h2>running</h2><div id="nrun" class="big">—</div><div id="nrunsub" class="muted"></div></div>
<div class="card s2"><h2>runs</h2><div id="ntotal" class="big">—</div><div class="muted">in the store</div></div>
<div class="card s2"><h2>cpu</h2><div id="cpu" class="big">—</div><div id="cpusub" class="muted">host, all cores</div></div>
<div class="card s2"><h2>ram</h2><div id="ram" class="big">—</div><div id="ramsub" class="muted"></div></div>
<div class="card s2"><h2>gpu</h2><div id="gpu" class="big">—</div><div id="gpusub" class="muted"></div></div>
<div class="card s2"><h2>gpu mem</h2><div id="gpumem" class="big">—</div><div id="gpumemsub" class="muted"></div></div>

<div class="card s12"><h2>sessions <input id="filter" class="filter" placeholder="filter task / status"></h2><div id="runstable" class="muted">loading…</div>
<div class="hint">click a row for its charts, configuration, steps and record</div></div>

<div class="card s12" id="detail" style="display:none"><h2 id="dtitle">run</h2>
<div id="dmeta" class="row"></div></div>

<div class="card s3" id="cstep" style="display:none"><h2>step</h2><div id="step" class="big">—</div><div id="steps" class="muted"></div></div>
<div class="card s3" id="closs" style="display:none"><h2>loss</h2><div id="loss" class="big">—</div><div id="lossrange" class="muted"></div></div>
<div class="card s3" id="crate" style="display:none"><h2>tokens / s</h2><div id="rate" class="big">—</div><div id="median" class="muted"></div></div>
<div class="card s3" id="cmem" style="display:none"><h2>memory</h2><div id="rss" class="big">—</div><div id="peak" class="muted"></div></div>

<div class="card s8" id="closschart" style="display:none"><h2>loss
<label class="chk"><input type="checkbox" id="logloss"> log</label>
<label class="chk"><input type="checkbox" id="showeval"> eval</label>
<label class="chk"><input type="checkbox" id="showtrain" checked> train</label>
</h2><canvas id="losschart"></canvas></div>
<div class="card s4" id="cmschart" style="display:none"><h2>step time</h2><canvas id="mschart"></canvas></div>
<div class="card s3" id="ccpu" style="display:none"><h2>cpu</h2><canvas id="cpuchart"></canvas></div>
<div class="card s3" id="cram" style="display:none"><h2>ram (pss)</h2><canvas id="ramchart"></canvas></div>
<div class="card s3" id="cgpu" style="display:none"><h2>gpu</h2><canvas id="gpuutilchart"></canvas></div>
<div class="card s3" id="cgpuproc" style="display:none"><h2>gpu memory</h2><canvas id="gpuprocchart"></canvas></div>
<div class="card s8" id="cextra" style="display:none"><h2>extra signals</h2><div id="extras" style="display:grid;grid-template-columns:1fr 1fr;gap:8px"></div><div id="extrahint" class="hint"></div></div>
<div class="card s4" id="cconfig" style="display:none"><h2>configuration</h2><div id="config" class="muted"></div></div>
<div class="card s6" id="ctable" style="display:none"><h2>recent steps</h2><div id="table"></div></div>
<div class="card s6" id="crecord" style="display:none"><h2>record</h2><div id="record" class="muted"></div></div>
<div class="card s12" id="craw" style="display:none"><h2>raw trace</h2><pre id="raw" style="max-height:160px;overflow:auto;color:var(--muted)"></pre></div>
</main>
<script>
let overview=null, selected=null, detail=null, runsList=[], timer=null,
 logLoss=false, showTrain=true, showEval=false, lossRun=null, filter="";
const $=id=>document.getElementById(id);
const qs=new URLSearchParams(location.search);
const wanted=qs.get("run");
if(wanted)selected=Number(wanted);
if(qs.get("log")==="1"){logLoss=true;$("logloss").checked=true}
function esc(x){return String(x??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function fmt(x,n=4){if(x===null||x===undefined||Number.isNaN(x))return"—";
 if(typeof x!=="number")return String(x); if(Math.abs(x)>=1e6||(Math.abs(x)>0&&Math.abs(x)<1e-3))
 return x.toExponential(n); return x.toFixed(n).replace(/0+$/,"").replace(/\.$/,"");}
function intfmt(x){return x===null||x===undefined?"—":Math.round(x).toLocaleString()}
function evalOf(r){return r.eval_loss??r.val_loss??r.validation_loss??null}
function statusCell(s){const cls={running:"st-running",done:"st-done",imported:"st-imported",
 failed:"st-failed",lost:"st-lost"}[s]||""; return `<span class="${cls}">${s==="running"?"● ":""}${esc(s)}</span>`}
function show(ids,on){for(const id of ids)$(id).style.display=on?"":"none"}
function chart(canvas, series, opts={}){
 const dpr=window.devicePixelRatio||1, r=canvas.getBoundingClientRect();
 if(!r.width||!r.height)return;
 canvas.width=r.width*dpr; canvas.height=r.height*dpr;
 const ctx=canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
 const w=r.width,h=r.height,left=46,right=8,top=18,bottom=22;
 ctx.clearRect(0,0,w,h);
 const pts=series.flatMap(s=>(s.points||[]).filter(p=>Number.isFinite(p[1])));
 if(!pts.length){ctx.fillStyle="#6f6a60";ctx.font="11px ui-monospace";
  ctx.fillText("unavailable",left+4,h/2);return}
 const ymin=Math.min(...pts.map(p=>p[1])), ymax=Math.max(...pts.map(p=>p[1]));
 const useLog=opts.log&&ymin>0;
 const V=useLog?(v=>Math.log10(v)):(v=>v);
 let lo=V(ymin), hi=V(ymax);
 if(hi-lo<1e-12){const d=Math.abs(hi)*0.03||1;lo-=d;hi+=d}
 const padY=(hi-lo)*0.08; lo-=padY; hi+=padY;
 if(opts.zero&&!useLog)lo=Math.max(0,lo);
 if(opts.min!=null)lo=Math.max(opts.min,lo);
 if(opts.max!=null)hi=Math.min(opts.max,hi);
 if(hi<=lo)hi=lo+1;
 const xs=pts.map(p=>p[0]), x0=Math.min(...xs), x1=Math.max(...xs);
 const xlo=x0===x1?x0-1:x0, xhi=x0===x1?x1+1:x1;
 const X=v=>left+(v-xlo)/(xhi-xlo)*(w-left-right);
 const Y=v=>top+(hi-V(v))/(hi-lo)*(h-top-bottom);
 ctx.font="10px ui-monospace";
 for(let k=0;k<3;k++){
  const val=hi-(hi-lo)*k/2, y=top+(h-top-bottom)*k/2;
  ctx.strokeStyle="#e4dfd2";ctx.lineWidth=1;ctx.beginPath();
  ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();
  ctx.fillStyle="#6f6a60";ctx.textAlign="right";
  const raw=useLog?Math.pow(10,val):val;
  const label=useLog?fmt(raw,3):(opts.int?(hi-lo<10?fmt(raw,2):intfmt(raw)):fmt(raw,3));
  ctx.fillText(label,left-4,y+3);
 }
 const xlab=opts.xfmt||(v=>fmt(v,0));
 ctx.fillStyle="#6f6a60";ctx.textAlign="left";ctx.fillText(xlab(x0),left,h-5);
 ctx.textAlign="right";ctx.fillText(xlab(x1),w-right,h-5);
 for(const s of series){
  const sp=(s.points||[]).filter(p=>Number.isFinite(p[1]));
  if(!sp.length)continue;
  const color=s.color||"#7d70b3";
  ctx.strokeStyle=color;ctx.lineWidth=s.width||1.6;ctx.beginPath();
  sp.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  if(sp.length>1)ctx.stroke();
  if(sp.length===1){const x=X(sp[0][0]),y=Y(sp[0][1]);
   if(series.length===1){ctx.setLineDash([2,4]);ctx.strokeStyle=color;ctx.globalAlpha=.45;
    ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();
    ctx.globalAlpha=1;ctx.setLineDash([]);}
   ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,3.5,0,6.284);ctx.fill();
   ctx.textAlign="left";ctx.fillText(fmt(sp[0][1],4),Math.min(x+7,w-right-40),y-6);
  } else if(sp.length<=40){
   ctx.fillStyle=color;
   for(const p of sp){ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),2,0,6.284);ctx.fill()}
  }
 }
 ctx.textAlign="right";let lx=w-right;
 for(const s of series){if(!s.points||!s.points.length)continue;
  const color=s.color||"#7d70b3";
  ctx.fillStyle="#6f6a60";ctx.fillText(s.name||"",lx,10);
  const tw=ctx.measureText(s.name||"").width;
  ctx.fillStyle=color;ctx.beginPath();ctx.arc(lx-tw-9,6,3,0,6.284);ctx.fill();
  lx-=tw+26;
 }
}
function renderOverview(){
 if(!overview)return; const sys=overview.system||{}, runs=overview.runs||[];
 const running=runs.filter(r=>r.status==="running").length;
 $("nrun").textContent=running; $("nrun").className=running?"big ok":"big muted";
 $("nrunsub").textContent=running?`${running} live · ${runs.length} total`:"no live session";
 $("ntotal").textContent=runs.length;
 $("cpu").textContent=sys.cpu==null?"—":sys.cpu.toFixed(0)+"%";
 $("cpusub").textContent=sys.ncpu?`of ${sys.ncpu} cores`:"host, all cores";
 const mem=sys.memory;
 $("ram").textContent=mem?(mem.used_kb/1048576).toFixed(1)+" GB":"—";
 $("ramsub").textContent=mem?`of ${(mem.total_kb/1048576).toFixed(1)} GB`:"unavailable";
 const gpu=sys.gpu;
 $("gpu").textContent=gpu?gpu.util.toFixed(0)+"%":"—";
 $("gpusub").textContent=gpu?"nvidia-smi":"unavailable";
 $("gpumem").textContent=gpu?(gpu.used_mb/1024).toFixed(1)+" GB":"—";
 $("gpumemsub").textContent=gpu?`of ${(gpu.total_mb/1024).toFixed(1)} GB`:"unavailable";
 $("budget").innerHTML=`· budget ${overview.budget} s`;
 const shown=runs.filter(r=>!filter||(r.task+" "+r.kind+" "+r.status).toLowerCase().includes(filter));
 let rows='<table><tr><th class="num">run</th><th>task</th><th>kind</th><th>status</th><th class="num">pid</th><th class="num">steps</th><th class="num">loss</th><th class="num">params</th><th class="num">cpu</th><th class="num" title="PSS over the process tree">ram</th><th class="num">gpu</th><th class="num">loop s</th></tr>';
 for(const r of shown){
  const p=r.proc||{};
  rows+=`<tr class="run ${r.status} ${r.id===selected?"sel":""}" data-id="${r.id}"><td class="num">#${r.id}</td><td>${esc(r.task)}</td><td>${esc(r.kind||"run")}</td><td>${statusCell(r.status)}</td><td class="num">${r.pid??"—"}</td><td class="num">${r.steps}</td><td class="num">${fmt(r.last_loss)}</td><td class="num">${r.params==null?"—":intfmt(r.params)}</td><td class="num">${p.cpu==null?"—":p.cpu.toFixed(0)+"%"+(p.cores!=null?` · ${p.cores.toFixed(1)}c`:"")}</td><td class="num">${p.rss_kb?((p.rss_kb/1024).toFixed(0)+" MB"):"—"}</td><td class="num">${p.gpu_proc_mb?(p.gpu_proc_mb.toFixed(0)+" MB"):"—"}</td><td class="num">${Number(r.loop_s||0).toFixed(1)}</td></tr>`;
 }
 $("runstable").innerHTML=rows+"</table>";
 const box=$("runstable"), selRow=box.querySelector("tr.sel");
 if(selRow){
  const rb=box.getBoundingClientRect(), sr=selRow.getBoundingClientRect();
  const top=box.scrollTop+(sr.top-rb.top);
  if(top<box.scrollTop||top+sr.height>box.scrollTop+box.clientHeight)
   box.scrollTop=Math.max(0,top-box.clientHeight+sr.height+8);
 }
}
function lossData(){
 const recs=(detail&&detail.records)||[];
 const train=recs.map(x=>[x.step,x.train_loss??x.loss])
  .filter(p=>p[1]!=null&&Number.isFinite(p[1]));
 const ev=recs.map(x=>[x.step,evalOf(x)])
  .filter(p=>p[1]!=null&&Number.isFinite(p[1]));
 const last=detail&&detail.run?detail.run.eval_loss:null;
 if(ev.length===0&&last!=null&&recs.length)
  ev.push([recs[recs.length-1].step,last]);
 return {train,ev};
}
function renderLoss(){
 if(!detail||!detail.run)return;
 const {train,ev}=lossData(), series=[];
 if(showTrain&&train.length)series.push({name:"train",color:"#7d70b3",points:train});
 if(showEval&&ev.length)series.push({name:"eval",color:"#4f8f86",points:ev});
 chart($("losschart"),series,{xfmt:v=>"step "+v,log:logLoss,zero:true});
 $("showeval").disabled=!ev.length;
 $("showeval").title=!ev.length?"no eval loss in this run"
  :(ev.length===1?"one final held-out eval point":"eval loss");
 const lastEval=ev.length?ev[ev.length-1][1]:null;
 $("lossrange").textContent=detail.summary.last==null?"":
  `train ${fmt(detail.summary.last)}`+(lastEval==null?"":` · eval ${fmt(lastEval)}`);
}
function renderDetail(){
 if(!detail||!detail.run){show(["detail"],false);return}
 const r=detail.run, s=detail.summary, recs=detail.records||[], samples=detail.samples||[],
  p=detail.proc||{}, extras=detail.extras||[];
 const hasSteps=recs.length>0, hasSamples=samples.length>0;
 show(["detail","cconfig"],true);
 show(["cstep","closs","crate","closschart","cmschart","ctable","craw"],hasSteps);
 show(["cmem"],hasSteps||hasSamples);
 show(["ccpu","cram","cgpu","cgpuproc"],hasSamples);
 show(["cextra"],hasSteps&&extras.length>0);
 show(["crecord"],!!detail.record);
 $("dtitle").innerHTML=`#${r.id} · ${esc(r.task)} · ${esc(r.kind)} ${statusCell(r.status)}`;
 $("dmeta").innerHTML=[["status",r.status],["pid",r.pid??"—"],["host",r.host??"—"],
  ["command",r.command??"—"],["started",r.started?new Date(r.started*1000).toLocaleString():"—"],
  ["finished",r.finished?new Date(r.finished*1000).toLocaleString():"—"],
  ["exit",r.exit_code??"—"],["error",r.error??"—"]].map(([k,v])=>
   `<div><div class="muted">${k}</div><div>${k==="status"?statusCell(v):esc(v)}</div></div>`).join("");
 $("step").textContent=recs.length?recs[recs.length-1].step:"—";
 $("steps").textContent=recs.length?`${s.loop_s.toFixed(1)} s in the loop`:"no steps";
 $("loss").textContent=s.last==null?"—":fmt(s.last);
 $("lossrange").textContent=s.last==null?"":`best ${fmt(s.best)} · worst ${fmt(s.worst)}`;
 $("rate").textContent=s.tokens_s?fmt(s.tokens_s,1):"—";
 const med=s.median_ms;
 $("median").textContent=med?(med>=2000?`${(med/1000).toFixed(1)} s median`:`${intfmt(med)} ms median`):"";
 $("rss").textContent=p.rss_kb?(p.rss_kb/1024).toFixed(0)+" MB":"—";
 $("peak").textContent=p.hwm_kb?`${(p.hwm_kb/1024).toFixed(0)} MB peak`+
  (p.cores!=null?` · ${p.cores.toFixed(1)} cores`:"")+
  (r.status==="running"?" · live":(p.peak?" · peak sample":" · last sample")):"";
 if(lossRun!==r.id){
  const {ev}=lossData();
  lossRun=r.id; showTrain=true; $("showtrain").checked=true;
  showEval=ev.length>0; $("showeval").checked=showEval;
 }
 renderLoss();
 chart($("mschart"),[{name:"ms",color:"#4f8f86",points:recs.map(x=>[x.step,x.ms])}],{xfmt:v=>"step "+v,int:true,zero:true});
 const tfmt=v=>new Date(v*1000).toLocaleTimeString();
 chart($("cpuchart"),[{name:"host %",color:"#b07a2a",points:samples.map(x=>[x.t,x.cpu])}],{xfmt:tfmt,int:true,zero:true,min:0,max:100});
 chart($("ramchart"),[{name:"ram MB",color:"#5a86a8",points:samples.map(x=>[x.t,x.rss_kb==null?null:x.rss_kb/1024])}],{xfmt:tfmt,int:true,zero:true});
 chart($("gpuutilchart"),[{name:"gpu %",color:"#a879a8",points:samples.map(x=>[x.t,x.gpu_util])}],{xfmt:tfmt,int:true,zero:true,min:0,max:100});
 chart($("gpuprocchart"),[{name:"proc MB",color:"#6f9b7c",points:samples.map(x=>[x.t,x.gpu_mem_kb==null?null:x.gpu_mem_kb/1024])}],{xfmt:tfmt,int:true,zero:true});
 const box=$("extras");
 if(extras.length){box.innerHTML=""; extras.slice(0,4).forEach((key,i)=>{
   const wrap=document.createElement("div"), lab=document.createElement("div");
   lab.className="muted"; lab.textContent=key; const c=document.createElement("canvas");
   c.style.height="90px"; wrap.appendChild(lab); wrap.appendChild(c); box.appendChild(wrap);
   chart(c,[{name:key,color:["#fbbf24","#4f8f86","#f472b6","#6f9b7c"][i%4],points:recs.map(x=>[x.step,x[key]])}]);
 }); $("extrahint").textContent="signals: "+extras.join(", ")+" · first "+Math.min(4,extras.length)+" shown";}
 else {box.innerHTML="";$("extrahint").textContent="no extra numeric field in the trace";}
 const cfg=r.config||{};
 let c='<table>';
 const add=(k,v)=>{if(v!==undefined&&v!==null&&v!=="")c+=`<tr><td>${esc(k)}</td><td>${esc(typeof v==="object"?JSON.stringify(v):v)}</td></tr>`};
 add("task",r.task); add("kind",r.kind);
 add("params",r.params==null?null:intfmt(r.params)); add("pid",r.pid); add("host",r.host);
 add("command",r.command); add("git",r.git_commit); add("versions",r.versions);
 for(const k of Object.keys(cfg))add(k,cfg[k]);
 c+="</table>"; $("config").innerHTML=c;
 const hasEval=recs.some(x=>evalOf(x)!=null);
 let rows=`<table><tr><th class="num">step</th><th class="num">train</th>${hasEval?'<th class="num">eval</th>':""}<th class="num">ms</th><th class="num">tokens</th></tr>`;
 for(const x of recs.slice(-10).reverse())rows+=`<tr><td class="num">${x.step}</td><td class="num">${fmt(x.train_loss??x.loss)}</td>${hasEval?`<td class="num">${fmt(evalOf(x))}</td>`:""}<td class="num">${fmt(x.ms,0)}</td><td class="num">${x.tokens??"—"}</td></tr>`;
 $("table").innerHTML=rows+"</table>";
 const rec=detail.record;
 if(rec){const keys=Object.keys(rec).filter(k=>["passed","trainer","variant","stages_source"].includes(k));
  $("record").innerHTML=`<div class="row">${keys.map(k=>`<div><div class="muted">${k}</div><div>${esc(rec[k])}</div></div>`).join("")}</div>`+
   `<div class="muted" style="margin-top:8px">${rec.passed?"PASS":"FAIL/unknown"} · ${esc(detail.record_path)}</div>`;
 } else $("record").textContent="unavailable";
 $("raw").textContent=JSON.stringify(recs.slice(-3),null,1);
}
async function poll(){
 try{
  const r=await fetch("/api/overview"); overview=await r.json(); runsList=overview.runs||[];
  $("status").innerHTML=`<i class="dot live"></i>live · ${new Date().toLocaleTimeString()}`;
  renderOverview();
  if(!selected && runsList.length){
   const live=runsList.find(x=>x.status==="running")||runsList.find(x=>x.steps>0)||runsList[0];
   selected=live.id;
  }
  if(selected){
   const s=await fetch(`/api/state?run=${selected}`); detail=await s.json();
   if(detail.run){selected=detail.run.id; renderDetail();}
  }
 }catch(e){$("status").innerHTML='<i class="dot"></i>server unavailable'}
 timer=setTimeout(poll,1000);
}
$("logloss").addEventListener("change",e=>{logLoss=e.target.checked;renderLoss()});
$("showtrain").addEventListener("change",e=>{showTrain=e.target.checked;renderLoss()});
$("showeval").addEventListener("change",e=>{showEval=e.target.checked;renderLoss()});
$("filter").addEventListener("input",e=>{filter=e.target.value.toLowerCase();renderOverview()});
$("runstable").addEventListener("click",e=>{
 const tr=e.target.closest("tr[data-id]"); if(!tr)return;
 selected=Number(tr.dataset.id); detail=null;
 history.replaceState(null,"",`?run=${selected}`); poll();
});
document.addEventListener("keydown",e=>{if(e.key===" "){e.preventDefault();if(timer)clearTimeout(timer);timer=setTimeout(poll,1000)}});
addEventListener("resize",()=>{if(detail)renderDetail()});
poll();
</script></body></html>"""


# ---------------------------------------------------------------------------
# the server

class Handler(BaseHTTPRequestHandler):
    server_version = "MLBendDashboard/2.0"

    def _send(self, body, content_type="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/api/overview":
            try:
                rows = runs.list_runs(limit=200)
            except Exception:
                rows = []
            live = self.server.sampler.live
            ncpu = self.server.sampler.ncpu
            for row in rows:
                row["proc"] = live.get(row["id"])
                if row["proc"] is None:
                    one = runs.peak_sample(row["id"])
                    if one:
                        row["proc"] = {
                            "cpu": one["cpu"],
                            "cores": (one["cpu"] or 0) * ncpu / 100.0,
                            "rss_kb": one["rss_kb"], "hwm_kb": one["hwm_kb"],
                            "gpu_proc_mb": (one["gpu_mem_kb"] / 1024
                                            if one["gpu_mem_kb"] else None),
                            "peak": True,
                        }
            body = {
                "now": time.time(),
                "system": self.server.sampler.system,
                "runs": rows,
                "budget": self.server.budget,
            }
            self._send(json.dumps(body))
            return
        if parsed.path == "/api/runs":
            self._send(json.dumps(runs.list_runs(limit=200)))
            return
        if parsed.path == "/api/state":
            run_id = None
            if qs.get("run"):
                try:
                    run_id = int(qs["run"][0])
                except ValueError:
                    self.send_error(400, "bad run")
                    return
            fallback = {"trace": self.server.trace, "record": self.server.record,
                        "sampler": self.server.sampler, "budget": self.server.budget}
            self._send(json.dumps(state(run_id, fallback)))
            return
        if parsed.path in ("/", "/index.html"):
            self._send(PAGE, "text/html; charset=utf-8")
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task",
                    help="the fallback task when the store has no run selected")
    ap.add_argument("--trace", help="default results/<task>/trace.jsonl")
    ap.add_argument("--record", help="default results/latest/<task>.json")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-seconds", type=float, default=3600.0,
                    help="the training budget shown on the page")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    trace = args.trace or (os.path.join(ROOT, "results", args.task, "trace.jsonl")
                           if args.task else None)
    record = args.record or (os.path.join(ROOT, "results", "latest", args.task + ".json")
                             if args.task else None)
    try:
        runs.reconcile()
        runs.backfill()
        if not runs.list_runs(limit=1):
            runs.import_latest()
        if args.task and not any(r["steps"] for r in runs.list_runs(args.task)) \
                and trace and os.path.exists(trace):
            runs.import_file(args.task, trace, record)
    except Exception as exc:
        print(f"dashboard: run store unavailable: {exc}")

    sampler = Sampler()
    sampler.start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.task = args.task
    httpd.trace = trace
    httpd.record = record
    httpd.sampler = sampler
    httpd.budget = args.max_seconds
    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard: {url}  runs: {len(runs.list_runs(limit=200))}")
    if trace:
        print(f"trace: {os.path.relpath(trace, ROOT)}  record: {os.path.relpath(record, ROOT)}")
    print(f"budget: {args.max_seconds:.0f} s  gpu: "
          f"{'nvidia-smi' if sampler.system.get('gpu') else 'unavailable'}")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard: stopped")
    finally:
        sampler.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
