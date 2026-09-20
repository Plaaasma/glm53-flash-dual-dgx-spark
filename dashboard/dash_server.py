#!/usr/bin/env python3
"""SPARK//CONSOLE web server: serves the dashboard page on :3000 AND exposes everything it shows as JSON.

  GET /                      the dashboard
  GET /api                   this index
  GET /api/all               everything below in one document (add ?viz=1 for the activation frames,
                             ?window=SECONDS&points=N to size the history; defaults 900 s / 60 points)
  GET /api/live              collector live sample: engine state, model, per-tick rates (row), node stats
  GET /api/derived           the numbers the page computes client-side (per-stream tok/s, prefilling,
                             KV pool pages, est. bandwidth, watched host processes per node)
  GET /api/requests          per-request progress (id, prompt/computed/total tokens, phase, progress %)
  GET /api/nodes             both nodes' agent stats (gpu, memory, cpu, net, disk, watched processes)
  GET /api/viz               activation telemetry (expert routing, attention scan, ribbon, 3-D frames, scheduler)
  GET /api/viz/status        just the telemetry health block
  GET /api/totals            lifetime input/output tokens (integrated from the history DB)
  GET /api/history?from&to&points   history series (same query as the collector)
  GET /api/procs             watched host processes per node (MiB, RSS + GPU memory), caps, optional alert file

Data comes from the collector on :9102; this server proxies it so one address and port serve both humans and
programs. All responses carry Access-Control-Allow-Origin: *.
"""
import json, math, os, sys, time, urllib.request, urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
COLLECTOR = os.environ.get("COLLECTOR", "http://127.0.0.1:9102")
PORT = int(os.environ.get("PORT", "3000"))
PAGE_TOKENS = 7936          # hybrid page size on this stack (attention block == KDA state page)
# Watched host processes (agent.py SPARK_WATCH_PROCS): optional per-node and total caps in MiB (0 = none) and an
# optional alert JSON written by an external watch, surfaced as-is under /api/procs.
PROC_CAP_MIB = int(os.environ.get("SPARK_PROC_CAP_MIB", "0") or 0)
PROC_CAP_TOTAL_MIB = int(os.environ.get("SPARK_PROC_CAP_TOTAL_MIB", "0") or 0)
PROC_ALERT_JSON = os.environ.get("SPARK_PROC_ALERT_JSON", "")
ENDPOINTS = ["/api", "/api/all", "/api/live", "/api/derived", "/api/requests", "/api/nodes", "/api/viz",
             "/api/viz/status", "/api/totals", "/api/history", "/api/procs"]


def fetch(path, timeout=4.0):
    try:
        with urllib.request.urlopen(COLLECTOR + path, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"error": f"collector {path}: {type(e).__name__}: {e}"}


def fmt_k(v):
    return f"{v/1e6:.2f}M" if v >= 1e6 else (f"{round(v/1e3)}K" if v >= 1e3 else str(int(v)))


def derive(live, viz):
    """Mirror of the dashboard's client-side arithmetic."""
    r = (live or {}).get("row") or {}
    nodes = (live or {}).get("nodes") or {}
    s = (viz or {}).get("sched") or {}
    sched_fresh = s and (viz.get("sched_age") is not None) and viz["sched_age"] < 5
    run = r.get("run") or 0
    dec = r.get("dec") if r.get("dec") is not None else run
    gen = r.get("gen")
    out = {
        "generation_tok_s": gen,
        "generation_per_stream_tok_s": (gen / dec) if (gen is not None and dec and dec > 0) else None,
        "prefill_tok_s": r.get("pp"),
        "running": run, "decoding": dec, "prefilling": max(0, (run or 0) - (dec or 0)) if r.get("dec") is not None else None,
        "waiting": r.get("wait"),
        "spec_accept_pct": r.get("accpct"), "accept_length": r.get("tau"), "drafted_tok_s": r.get("draftrate"),
        "kv_usage_pct": r.get("kv"), "kv_total_tokens": live.get("kv_total_tokens"),
        "kv_used_tokens": (r.get("kv") / 100.0 * live["kv_total_tokens"]) if (r.get("kv") is not None and live.get("kv_total_tokens")) else None,
        "prefix_hit_pct": r.get("pfx"), "prompt_cached_pct": r.get("cachedpct"),
        "steps_per_s": r.get("steps"), "tokens_per_step": r.get("stepsz"), "est_tflops": r.get("tflops"),
        "ttft_s": {"p50": r.get("ttft50"), "p99": r.get("ttft99")},
        "itl_s": {"p50": r.get("itl50"), "p99": r.get("itl99")},
        "queue_s": {"p50": r.get("q50"), "p99": r.get("q99")},
        "requests_ok": r.get("req_ok"), "preemptions": r.get("preempt"),
    }
    # est. memory bandwidth per rank, as the reactor gauge draws it
    tok = r.get("stepsz") or 0; steps = r.get("steps") or 0
    distinct = 288 * (1 - math.exp(-tok * 9 / 288)) if tok else 0
    out["est_bandwidth_gb_s_per_rank"] = (distinct * 0.0063 * 42 + 7.4 + 0.067 * 33 * max(1, run or 1) / 8) * steps if steps else 0.0
    if s:
        total = s.get("blocks_total"); evict = s.get("blocks_evictable")
        out["kv_pool"] = {
            "snapshot_age_s": viz.get("sched_age"), "stale": not sched_fresh,
            "pool_tokens": s.get("pool_tokens"), "page_tokens": PAGE_TOKENS, "groups": s.get("groups"),
            "pages_total": total, "pages_in_use": (total - evict) if (total is not None and evict is not None) else None,
            "pages_cached": s.get("blocks_cached"), "pages_free": s.get("blocks_free"),
            "usage_pct": (s.get("usage") or 0) * 100, "live_requests": len(s.get("reqs") or []), "waiting": s.get("waiting"),
        }
    out["nodes"] = {}
    for k, name in (("h", "head"), ("w", "worker")):
        n = nodes.get(k) or {}
        g = n.get("gpu") or {}; m = n.get("mem") or {}
        out["nodes"][name] = {
            "host": n.get("host"), "gpu_util_pct": g.get("util"), "gpu_power_w": g.get("power"), "gpu_temp_c": g.get("temp"),
            "gpu_sm_mhz": g.get("sm_mhz"), "mem_used_gib": m.get("used_gib"), "mem_total_gib": m.get("total_gib"),
            "cpu_pct": n.get("cpu_pct"), "net_mb_s": r.get(f"{k}_net"), "load1": n.get("load1"),
            "watched_procs_mib": n.get("procs") or {},
            "procs_over_cap": bool(PROC_CAP_MIB and sum((n.get("procs") or {}).values()) > PROC_CAP_MIB),
        }
    return out


def requests_view(viz):
    s = (viz or {}).get("sched") or {}
    if not s or (viz.get("sched_age") or 99) >= 5:
        return []
    out = []
    for q in s.get("reqs") or []:
        prompt, computed, total = q.get("prompt") or 0, q.get("computed") or 0, q.get("total") or 0
        prefilling = computed < prompt
        out.append({
            "id": q.get("id"), "phase": "prefill" if prefilling else "generating",
            "prompt_tokens": prompt, "computed_tokens": computed, "total_tokens": total,
            "prefill_pct": round(100 * computed / prompt, 1) if (prefilling and prompt) else 100.0,
            "age_s": q.get("age"), "scheduled_tokens_this_step": q.get("sched"),
            "label": (f"{q.get('id')} · {fmt_k(computed)}/{fmt_k(prompt)} · prefill {round(100*computed/max(prompt,1))}%"
                      if prefilling else f"{q.get('id')} · {fmt_k(total)} ctx · generating"),
        })
    return out


def procs_view():
    live = fetch("/live")
    nodes = (live or {}).get("nodes") or {}
    out = {"per_node": {k: ((n or {}).get("procs") or {}) for k, n in nodes.items()},
           "caps_mib": {"per_node": PROC_CAP_MIB or None, "total": PROC_CAP_TOTAL_MIB or None}, "alert": None}
    if PROC_ALERT_JSON:
        try:
            with open(PROC_ALERT_JSON) as f:
                out["alert"] = json.load(f)
        except Exception:  # noqa: BLE001
            pass
    return out


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path.rstrip("/") or "/"
        if not p.startswith("/api"):
            return super().do_GET()
        q = urllib.parse.parse_qs(u.query)
        now = time.time()
        if p == "/api":
            return self._json({"endpoints": ENDPOINTS, "collector": COLLECTOR, "ts": now, "doc": __doc__.strip()})
        if p in ("/api/live", "/api/viz", "/api/totals"):
            return self._json(fetch(p[4:]))
        if p == "/api/history":
            return self._json(fetch("/history" + ("?" + u.query if u.query else "")))
        if p == "/api/viz/status":
            v = fetch("/viz"); return self._json({"status": v.get("status"), "status_age": v.get("status_age"), "act_age": v.get("act_age"), "act3d_age": v.get("act3d_age"), "sched_age": v.get("sched_age")})
        if p == "/api/nodes":
            return self._json((fetch("/live") or {}).get("nodes") or {})
        if p == "/api/derived":
            return self._json(derive(fetch("/live"), fetch("/viz")))
        if p == "/api/requests":
            return self._json({"requests": requests_view(fetch("/viz")), "ts": now})
        if p == "/api/procs":
            return self._json(procs_view())
        if p == "/api/all":
            window = int(q.get("window", ["900"])[0]); points = int(q.get("points", ["60"])[0])
            with_viz = q.get("viz", ["0"])[0] == "1"
            live, viz, totals = fetch("/live"), fetch("/viz"), fetch("/totals")
            hist = fetch(f"/history?from={now - window}&to={now}&points={points}") if q.get("history", ["1"])[0] == "1" else None
            doc = {
                "ts": now, "engine_up": live.get("engine_up"), "model": live.get("model"), "engine": live.get("engine"), "boot": live.get("boot"),
                "derived": derive(live, viz), "requests": requests_view(viz), "row": live.get("row"), "nodes": live.get("nodes"),
                "kv_scheduler_snapshot": viz.get("sched"), "viz_status": viz.get("status"), "totals": totals, "procs": procs_view(),
                "history": hist, "history_window_s": window,
            }
            if with_viz:
                doc["viz"] = {k: viz.get(k) for k in ("act", "act_age", "act3d", "act3d_age")}
            return self._json(doc)
        return self._json({"error": "unknown endpoint", "endpoints": ENDPOINTS}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
