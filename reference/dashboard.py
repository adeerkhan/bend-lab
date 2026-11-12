"""A browser dashboard for a training trace, on a local port.

  reference/dashboard.py --task glm53-hybrid128 --port 8787
  reference/dashboard.py --task glm53-hybrid128 --pid 12345 --open

`reference/dash.py` is the terminal page. This is the same trace
(`results/<task>/trace.jsonl`) and the same record, served as HTML: loss, step
time, tokens/s, resident memory, the stage split, and any extra numeric fields
the program writes beside `step`, `tokens`, `loss` and `ms`. Extra fields are
the hook for model internals: write `lr`, `grad_norm`, `state_norm` or
`gate_mean` in a trace line and they appear as small charts without changing
the server.

Standard library only. The page polls `/api/state` once a second; no
WebSocket, no build step, no CDN. `--max-seconds` is the training budget the
page shows (default one hour); the server does not kill a run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("step", "tokens", "loss", "ms")
KNOWN = set(KEYS) | {"time", "timestamp", "wall_ms", "elapsed"}


def read_trace(path, limit=20000):
    records = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if all(k in r for k in KEYS):
                    records.append(r)
    except OSError:
        return []
    return records[-limit:]


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class Sampler:
    def __init__(self, pid):
        self.pid = pid
        self.previous = None
        self.previous_at = None
        self.cpu = None

    def sample(self):
        if not self.pid:
            return None
        try:
            status = open(f"/proc/{self.pid}/status").read()
            stat = open(f"/proc/{self.pid}/stat").read()
        except OSError:
            return None
        vals = {}
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                vals[key.rstrip(":")] = int(value.split()[0])
        fields = stat[stat.rfind(")") + 2:].split()
        ticks = int(fields[11]) + int(fields[12])
        now = time.monotonic()
        if self.previous is not None and self.previous_at is not None:
            hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
            self.cpu = 100.0 * (ticks - self.previous["ticks"]) / hz / (now - self.previous_at)
        self.previous = {"ticks": ticks}
        self.previous_at = now
        vals["cpu"] = self.cpu
        return vals


def state(task, trace, record_path, sampler, budget, run_id=None):
    run = runs.get_run(run_id) if run_id else None
    if run is None:
        latest = runs.list_runs(task)
        run = runs.get_run(latest[0]["id"]) if latest else None
    if run is not None:
        records = run["records"]
        record = run.get("record") or read_json(record_path)
        task = run["task"]
    else:
        records = read_trace(trace)
        record = read_json(record_path)
    summary = {"steps": len(records), "tokens": 0, "loop_s": 0.0,
               "median_ms": None, "tokens_s": None}
    extras = []
    if records:
        losses = [float(r["loss"]) for r in records]
        ms = [float(r["ms"]) for r in records]
        summary["tokens"] = int(sum(int(r.get("tokens") or 0) for r in records))
        summary["loop_s"] = sum(ms) / 1000.0
        summary["median_ms"] = statistics.median(ms)
        summary["tokens_s"] = (summary["tokens"] / summary["loop_s"]
                               if summary["loop_s"] else None)
        summary["first"] = losses[0]
        summary["last"] = losses[-1]
        summary["best"] = min(losses)
        summary["worst"] = max(losses)
        for r in records:
            for key, value in r.items():
                if key not in KNOWN and isinstance(value, (int, float)) \
                        and key not in extras:
                    extras.append(key)
    proc = sampler.sample() if sampler else None
    return {
        "task": task,
        "trace": os.path.relpath(trace, ROOT),
        "record_path": os.path.relpath(record_path, ROOT),
        "record": record,
        "records": records,
        "extras": sorted(extras),
        "summary": summary,
        "proc": proc,
        "budget": budget,
        "now": time.time(),
        "run": None if run is None else {
            k: run[k] for k in ("id", "task", "label", "status", "created",
                                "updated", "steps", "tokens", "loop_s",
                                "first_loss", "last_loss", "best_loss")
        },
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TASK__ · training</title>
<style>
:root{--bg:#0e1116;--panel:#161b24;--line:#263041;--fg:#e6edf7;--muted:#8b98ad;
--accent:#6ee7b7;--accent2:#93c5fd;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;padding:16px 20px;
border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:18px;margin:0;letter-spacing:.02em}h2{font-size:13px;color:var(--muted);
margin:0 0 8px;text-transform:uppercase;letter-spacing:.08em}
main{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;padding:16px 20px 40px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;
min-width:0}.span3{grid-column:span 3}.span4{grid-column:span 4}.span6{grid-column:span 6}
.span8{grid-column:span 8}.span12{grid-column:span 12}
.big{font-size:22px;font-variant-numeric:tabular-nums}.muted{color:var(--muted)}
.row{display:flex;gap:18px;flex-wrap:wrap}.row>div{min-width:96px}
canvas{width:100%;height:150px;display:block}.wide canvas{height:190px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;
padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500}td.num,th.num{text-align:right;
font-variant-numeric:tabular-nums}.ok{color:var(--accent)}.bad{color:var(--bad)}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent)}
.hint{color:var(--muted);font-size:12px;margin:6px 0 0}
@media(max-width:900px){.span3,.span4,.span6,.span8{grid-column:span 12}}
</style></head><body>
<header><h1 id="title">__TASK__</h1>
<select id="runsel" style="background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px"></select>
<span id="status" class="muted">connecting…</span>
<span id="budget" class="muted"></span><span class="muted">auto-refresh 1 s</span></header>
<main>
<div class="card span3"><h2>step</h2><div id="step" class="big">—</div><div id="steps" class="muted"></div></div>
<div class="card span3"><h2>loss</h2><div id="loss" class="big">—</div><div id="lossrange" class="muted"></div></div>
<div class="card span3"><h2>tokens / s</h2><div id="rate" class="big">—</div><div id="median" class="muted"></div></div>
<div class="card span3"><h2>memory</h2><div id="rss" class="big">—</div><div id="peak" class="muted"></div></div>
<div class="card span8"><h2>loss</h2><canvas id="losschart"></canvas></div>
<div class="card span4"><h2>step time</h2><canvas id="mschart"></canvas></div>
<div class="card span8"><h2>extra signals</h2>
<div id="extras" style="display:grid;grid-template-columns:1fr 1fr;gap:8px"></div>
<div id="extrahint" class="hint">Add lr, grad_norm, state_norm or gate_mean to a trace line.</div></div>
<div class="card span4"><h2>stage split</h2><div id="stages" class="muted">unavailable</div></div>
<div class="card span6"><h2>recent steps</h2><div id="table"></div></div>
<div class="card span6"><h2>record</h2><div id="record" class="muted">unavailable</div></div>
<div class="card span12"><h2>raw trace</h2><pre id="raw" style="max-height:180px;overflow:auto;color:var(--muted)"></pre></div>
</main>
<script>
const defaultTask = "__TASK__";
let paused=false, timer=null, state=null, currentRun=null, runCount=0, ticks=0;
const $=id=>document.getElementById(id);
async function loadRuns(){
 try{const r=await fetch("/api/runs"); const list=await r.json();
  const sel=$("runsel");
  if(list.length!==runCount || sel.options.length!==list.length){
   runCount=list.length;
   const old=sel.value;
   sel.innerHTML=list.map(x=>`<option value="${x.id}">#${x.id} · ${x.task} · ${x.steps} steps · ${x.last_loss===null?"—":Number(x.last_loss).toFixed(4)}</option>`).join("");
   if(currentRun)sel.value=String(currentRun);
   else if(list.length){currentRun=list[0].id;sel.value=String(currentRun)}
   else if(old)sel.value=old;
  }
  if(!currentRun && list.length){currentRun=list[0].id;sel.value=String(currentRun)}
 }catch(e){}
}
$("runsel").addEventListener("change",e=>{currentRun=Number(e.target.value);paused=false;tick()});
function fmt(x,n=4){if(x===null||x===undefined||Number.isNaN(x))return"—";
 if(typeof x!=="number")return String(x); if(Math.abs(x)>=1e4||(Math.abs(x)>0&&Math.abs(x)<1e-3))
 return x.toExponential(n); return x.toFixed(n).replace(/0+$/,"").replace(/\.$/,"");}
function line(canvas, series, opts={}){
 const dpr=window.devicePixelRatio||1, r=canvas.getBoundingClientRect();
 canvas.width=r.width*dpr; canvas.height=r.height*dpr;
 const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
 const w=r.width,h=r.height,pad=26;
 ctx.clearRect(0,0,w,h); ctx.strokeStyle="#263041"; ctx.lineWidth=1;
 ctx.beginPath(); ctx.moveTo(pad,6); ctx.lineTo(pad,h-18); ctx.lineTo(w-6,h-18); ctx.stroke();
 if(!series.length)return;
 const ys=series.map(p=>p[1]).filter(v=>Number.isFinite(v));
 if(!ys.length)return; let lo=Math.min(...ys),hi=Math.max(...ys);
 if(hi===lo){hi=lo+1;lo-=1}
 const n=Math.max(1,series.length-1);
 const path=(color,idx)=>{
  ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.beginPath();
  series.forEach((p,i)=>{const x=pad+(p[0]-(opts.x0??series[0][0]))/((opts.x1??series[series.length-1][0])-(opts.x0??series[0][0])||1)*(w-pad-8);
   const y=h-18-(p[1]-lo)/(hi-lo)*(h-28); i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  ctx.stroke();
 };
 path(opts.color||"#6ee7b7");
 ctx.fillStyle="#8b98ad";ctx.font="10px ui-monospace";ctx.fillText(fmt(hi,3),2,12);
 ctx.fillText(fmt(lo,3),2,h-5);
 if(opts.label){ctx.fillStyle="#8b98ad";ctx.fillText(opts.label,w-70,12)}
}
function draw(){
 if(!state)return; const s=state.summary, recs=state.records;
 $("status").textContent = recs.length ? `${s.steps} steps · ${s.tokens.toLocaleString()} tokens`
   : (state.run ? `${state.run.status} · record only · no per-step trace` : "waiting for trace");
 $("step").textContent = recs.length ? recs[recs.length-1].step : "—";
 $("steps").textContent = recs.length ? `${s.loop_s.toFixed(1)} s in the loop` : "";
 $("loss").textContent = recs.length ? fmt(s.last) : "—";
 $("lossrange").textContent = recs.length ? `best ${fmt(s.best)} · worst ${fmt(s.worst)}` : "";
 $("rate").textContent = s.tokens_s ? fmt(s.tokens_s,1) : "—";
 $("median").textContent = s.median_ms ? `${fmt(s.median_ms,0)} ms median` : "";
 const p=state.proc; $("rss").textContent = p&&p.VmRSS ? `${(p.VmRSS/1024).toFixed(0)} MB` : "—";
 $("peak").textContent = p&&p.VmHWM ? `${(p.VmHWM/1024).toFixed(0)} MB peak`+(p.cpu!=null?` · ${p.cpu.toFixed(0)}% cpu`:"") : "";
 const b=state.budget, over=s.loop_s>b;
 $("budget").innerHTML = `loop ${s.loop_s.toFixed(1)} / ${b} s <span class="${over?'bad':'ok'}">${over?'over budget':'inside budget'}</span>`;
 line($("losschart"), recs.map(r=>[r.step,r.loss]), {color:"#6ee7b7"});
 line($("mschart"), recs.map(r=>[r.step,r.ms]), {color:"#93c5fd"});
 const extras=state.extras, box=$("extras");
 if(extras.length){
   box.innerHTML="";
   extras.slice(0,4).forEach((key,i)=>{
     const wrap=document.createElement("div");
     const lab=document.createElement("div"); lab.className="muted"; lab.textContent=key;
     const c=document.createElement("canvas"); c.style.height="90px";
     wrap.appendChild(lab); wrap.appendChild(c); box.appendChild(wrap);
     line(c, recs.map(r=>[r.step,r[key]]), {color:["#fbbf24","#93c5fd","#f472b6","#a7f3d0"][i%4],label:key});
   });
   $("extrahint").textContent = "signals: "+extras.join(", ")+" · first "+Math.min(4,extras.length)+" shown";
 } else {
   box.innerHTML=""; $("extrahint").textContent="Add lr, grad_norm, state_norm or gate_mean to a trace line.";
 }
 const rec=state.record;
 if(rec&&rec.stages&&rec.stages.length){
   const root=rec.stages[0].ms||1;
   let h='<table><tr><th>component</th><th class="num">ms</th><th class="num">share</th></tr>';
   for(const st of rec.stages){h+=`<tr><td>${st.path||st.name||""}</td><td class="num">${st.ms??""}</td><td class="num">${((st.ms/root)*100).toFixed(1)}%</td></tr>`}
   h+="</table>"; $("stages").innerHTML=h;
 } else $("stages").textContent="unavailable: no stages in the record";
 if(rec){let keys=Object.keys(rec).filter(k=>["loss","passed","metric","variant","stages"].includes(k));
  $("record").innerHTML=`<div class="row">${keys.map(k=>`<div><div class="muted">${k}</div><div>${fmt(rec[k],6)}</div></div>`).join("")}</div>`+
   `<div class="muted" style="margin-top:8px">${rec.passed?"PASS":"FAIL/unknown"} · ${state.record_path}</div>`;
 } else $("record").textContent="unavailable";
 let rows='<table><tr><th class="num">step</th><th class="num">loss</th><th class="num">ms</th><th class="num">tokens</th></tr>';
 for(const r of recs.slice(-8).reverse()) rows+=`<tr><td class="num">${r.step}</td><td class="num">${fmt(r.loss)}</td><td class="num">${fmt(r.ms,0)}</td><td class="num">${r.tokens}</td></tr>`;
 $("table").innerHTML=rows+"</table>";
 $("raw").textContent=JSON.stringify(recs.slice(-3),null,1);
}
async function tick(){try{
 ticks++;
 if(!runCount || !state || ticks%10===1) await loadRuns();
 const q=currentRun?`run=${currentRun}`:`task=${encodeURIComponent(defaultTask)}`;
 const r=await fetch(`/api/state?${q}`);
 state=await r.json();
 if(state.run){currentRun=state.run.id;$("runsel").value=String(currentRun);
  $("title").textContent=`${state.run.task} · run ${state.run.id}`}
 draw();
}catch(e){$("status").textContent="server unavailable"}
 if(!paused)setTimeout(tick,1000)}
document.addEventListener("keydown",e=>{if(e.key===" "){paused=!paused;e.preventDefault();if(!paused)tick()}});
addEventListener("resize",draw);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MLBendDashboard/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            task = (qs.get("task") or [self.server.task])[0]
            if task != self.server.task:
                self.send_error(404, "unknown task")
                return
            run_id = None
            if qs.get("run"):
                try:
                    run_id = int(qs["run"][0])
                except ValueError:
                    self.send_error(400, "bad run")
                    return
            body = json.dumps(state(self.server.task, self.server.trace,
                                    self.server.record, self.server.sampler,
                                    self.server.budget, run_id)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/runs":
            try:
                items = runs.list_runs()
            except Exception:
                items = []
            body = json.dumps(items).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path in ("/", "/index.html"):
            body = PAGE.replace("__TASK__", self.server.task).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--trace", help="default results/<task>/trace.jsonl")
    ap.add_argument("--record", help="default results/latest/<task>.json")
    ap.add_argument("--pid", type=int, help="sample this process's RSS and CPU")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-seconds", type=float, default=3600.0,
                    help="the training budget shown on the page")
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    args = ap.parse_args()

    trace = args.trace or os.path.join(ROOT, "results", args.task, "trace.jsonl")
    record = args.record or os.path.join(ROOT, "results", "latest", args.task + ".json")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.task = args.task
    httpd.trace = trace
    httpd.record = record
    httpd.sampler = Sampler(args.pid)
    httpd.budget = args.max_seconds
    try:
        if not runs.list_runs(args.task) and os.path.exists(trace):
            runs.import_file(args.task, trace, record)
    except Exception as exc:
        print(f"dashboard: run store unavailable: {exc}")
    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard: {url}  task {args.task}")
    print(f"trace: {os.path.relpath(trace, ROOT)}  record: {os.path.relpath(record, ROOT)}")
    print(f"budget: {args.max_seconds:.0f} s  pid: {args.pid or '-'}")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard: stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
