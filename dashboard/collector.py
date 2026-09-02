#!/usr/bin/env python3
"""Spark cluster history collector.

Polls vLLM /metrics + both node agents every 2.5s, derives per-tick rates
(tok/s, spec acceptance, latency percentiles from histogram deltas, per-node
hardware), and stores everything in SQLite. Serves:

  GET /history?from=<unix>&to=<unix>&points=<n>   bucket-averaged series
  GET /live                                        latest sample + raw node data\n  GET /totals                                      lifetime input/output tokens (restart-aware)

CORS is open — consumed directly by the dashboard page in the browser.
"""
import json, sqlite3, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DB_PATH = "/home/liam/cluster-dashboard/history.db"
VLLM = "http://localhost:8000/metrics"
SGLANG = "http://localhost:8888/metrics"
AGENTS = {"h": "http://localhost:9101/stats", "w": "http://169.254.152.37:9101/stats"}
TICK = 2.5
RETAIN_S = 35 * 86400          # keep 35 days
HIST_WINDOW = 120              # seconds of histogram ring for percentiles

COLS = ["gen","pp","accpct","draftrate","tau","kv","pfx",
        "ttft50","ttft99","itl50","itl99","run","wait",
        "pos0","pos1","pos2","pos3","pos4","pos5","pos6",
        "wait_cap","wait_def","steps","stepsz","q50","q99","cachedpct","tflops",
        "tok_total","tok_in","tok_out","req_ok","preempt",
        "h_gpu","h_temp","h_power","h_mem","h_cpu","h_net",
        "w_gpu","w_temp","w_power","w_mem","w_cpu","w_net"]

# Series whose true per-bucket extremes are also returned (as <col>_mx / <col>_mn),
# so the min/max readouts do not change when the timeframe (and thus bucket width)
# changes. Keep this list small — each entry adds two numbers per point.
MINMAX_COLS = ["gen", "pp", "h_temp", "w_temp"]

# ---------------- storage ----------------
def db():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    return c

def init_db():
    c = db()
    c.execute(f"CREATE TABLE IF NOT EXISTS samples (ts REAL PRIMARY KEY, {', '.join(f'{k} REAL' for k in COLS)})")
    have = {r[1] for r in c.execute("PRAGMA table_info(samples)")}
    for k in COLS:                       # add columns introduced after the DB was created
        if k not in have:
            c.execute(f"ALTER TABLE samples ADD COLUMN {k} REAL")
    c.commit(); c.close()

# ---------------- prometheus parsing ----------------
def fetch(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()

def parse_prom(text):
    out = {}
    for ln in text.split("\n"):
        if not ln or ln[0] == "#":
            continue
        sp = ln.rfind(" ")
        key, sval = ln[:sp], ln[sp+1:]
        try: v = float(sval)
        except ValueError: continue
        br = key.find("{")
        name = key if br < 0 else key[:br]
        labels = {}
        if br >= 0:
            for pair in key[br+1:-1].split('",'):
                if "=" in pair:
                    k, _, val = pair.partition("=")
                    labels[k.strip()] = val.strip('"')
        out.setdefault(name, []).append((labels, v))
    return out

def psum(p, n): return sum(v for _, v in p.get(n, []))
def pget(p, n):
    e = p.get(n, [])
    return e[0][1] if e else None

def buckets(p, n):
    b = {}
    for labels, v in p.get(n + "_bucket", []):
        le = labels.get("le", "")
        f = float("inf") if le == "+Inf" else float(le)
        b[f] = b.get(f, 0) + v
    return b

def pctile(now_b, old_b, q):
    ks = sorted(now_b.keys())
    d = [max(0.0, now_b[k] - (old_b or {}).get(k, 0.0)) for k in ks]
    tot = d[-1] if d else 0.0
    if tot <= 0: return None
    target = q * tot
    for i, k in enumerate(ks):
        if d[i] >= target:
            prev = d[i-1] if i else 0.0
            span = d[i] - prev
            lo = ks[i-1] if i else 0.0
            hi = k if k != float("inf") else (lo * 2 or 1.0)
            return lo + ((target - prev) / span if span > 0 else 0.0) * (hi - lo)
    return ks[-1] if ks else None

# ---------------- collector loop ----------------
# prev_vllm/ring_vllm and prev_sglang/ring_sglang are kept SEPARATE (not shared)
# so that restarting one engine can never be misread as a rate against the
# other engine's last counters -- e.g. vLLM's cumulative gen_tokens_total vs
# SGLang's would produce a nonsense delta if they shared one "prev" slot.
state = {"prev_vllm": None, "ring_vllm": [], "prev_sglang": None, "ring_sglang": [],
         "prev_net": {}, "live": {}, "model": None, "engine": None}

def poll_agent(key):
    try: return json.loads(fetch(AGENTS[key]))
    except Exception: return None

def net_rate(key, d):
    tot = sum(c["rx"] + c["tx"] for i, c in d["net"].items()
              if not i.startswith(("tailscale", "docker", "br-", "veth", "virbr")))
    pv = state["prev_net"].get(key)
    state["prev_net"][key] = (tot, d["ts"])
    if pv and d["ts"] > pv[1]:
        return max(0.0, (tot - pv[0]) / (d["ts"] - pv[1]) / 1e6)
    return None

def tick_vllm(now, row):
    p = parse_prom(fetch(VLLM))
    _fill_vllm(p, now, row, "vllm")

def _fill_vllm(p, now, row, slot):
    # Engine-agnostic vLLM-metrics filler. `slot` picks the prev/ring pair so
    # the :8000 and :8888 endpoints never share counter epochs (see note on
    # `state`). Port 8888 serves vLLM since the EXL3-kit cutover (2026-08-31),
    # so tick_sglang delegates here when it sees vllm:-prefixed metrics.
    e = p.get("vllm:num_requests_running", [])
    if e: state["model"] = e[0][0].get("model_name")
    cur = {
        "gen": psum(p, "vllm:generation_tokens_total"),
        "pp": psum(p, "vllm:prompt_tokens_total"),
        "acc": psum(p, "vllm:spec_decode_num_accepted_tokens_total"),
        "draft": psum(p, "vllm:spec_decode_num_draft_tokens_total"),
        "drafts": psum(p, "vllm:spec_decode_num_drafts_total"),
        "pos": [sum(v for l, v in p.get("vllm:spec_decode_num_accepted_tokens_per_pos_total", [])
                    if l.get("position") == str(i)) for i in range(7)],
        # iteration_tokens histogram updates PER ENGINE STEP -- unlike
        # prompt_tokens_total (end-of-request), it sees prefill chunks live.
        "iter_sum": psum(p, "vllm:iteration_tokens_total_sum"),
        "iter_cnt": psum(p, "vllm:iteration_tokens_total_count"),
        "cached": psum(p, "vllm:prompt_tokens_cached_total"),
        "flops": psum(p, "vllm:estimated_flops_per_gpu_total"),
        "t": now,
    }
    kv = pget(p, "vllm:kv_cache_usage_perc")
    pfh, pfq = psum(p, "vllm:prefix_cache_hits_total"), psum(p, "vllm:prefix_cache_queries_total")
    row["kv"] = kv * 100 if kv is not None else None
    row["pfx"] = 100 * pfh / pfq if pfq > 0 else None
    row["run"] = psum(p, "vllm:num_requests_running")
    row["wait"] = psum(p, "vllm:num_requests_waiting")
    for labels, v in p.get("vllm:num_requests_waiting_by_reason", []):
        r = labels.get("reason", "")
        if r == "capacity": row["wait_cap"] = (row.get("wait_cap") or 0) + v
        elif r == "deferred": row["wait_def"] = (row.get("wait_def") or 0) + v
    row["tok_total"] = cur["gen"] + cur["pp"]
    row["tok_in"] = cur["pp"]
    row["tok_out"] = cur["gen"]
    row["req_ok"] = psum(p, "vllm:request_success_total")
    row["preempt"] = psum(p, "vllm:num_preemptions_total")
    state["ring_" + slot].append((now, buckets(p, "vllm:time_to_first_token_seconds"),
                               buckets(p, "vllm:inter_token_latency_seconds"),
                               buckets(p, "vllm:request_queue_time_seconds")))
    state["ring_" + slot] = [r for r in state["ring_" + slot] if now - r[0] <= HIST_WINDOW]
    old = state["ring_" + slot][0]
    row["ttft50"] = pctile(state["ring_" + slot][-1][1], old[1], .5)
    row["ttft99"] = pctile(state["ring_" + slot][-1][1], old[1], .99)
    row["itl50"] = pctile(state["ring_" + slot][-1][2], old[2], .5)
    row["itl99"] = pctile(state["ring_" + slot][-1][2], old[2], .99)
    if len(state["ring_" + slot][-1]) > 3 and len(old) > 3:
        row["q50"] = pctile(state["ring_" + slot][-1][3], old[3], .5)
        row["q99"] = pctile(state["ring_" + slot][-1][3], old[3], .99)
    pv = state["prev_" + slot]
    if pv:
        dt = now - pv["t"] or 1.0
        row["gen"] = max(0.0, cur["gen"] - pv["gen"]) / dt
        dD, dN = cur["draft"] - pv["draft"], cur["drafts"] - pv["drafts"]
        dA = cur["acc"] - pv["acc"]
        # Live prefill rate. prompt_tokens_total only lands at request END, so
        # in-flight chunked prefill is invisible to it (observed: pp=0 for whole
        # minutes while prefill chunks ground away, 2026-09-02). Per engine step
        # scheduled == generated + rejected_drafts + prefill, exactly, so:
        dIt = max(0.0, cur.get("iter_sum", 0) - pv.get("iter_sum", 0))
        dRej = max(0.0, dD - dA)
        pp_live = (dIt - max(0.0, cur["gen"] - pv["gen"]) - dRej) / dt
        dPrompt = max(0.0, cur["pp"] - pv["pp"])
        row["pp"] = max(pp_live, 0.0) if dIt > 0 else dPrompt / dt
        dC = max(0.0, cur.get("iter_cnt", 0) - pv.get("iter_cnt", 0))
        row["steps"] = dC / dt
        row["stepsz"] = dIt / dC if dC > 0 else None
        dCa = cur.get("cached", 0) - pv.get("cached", 0)
        row["cachedpct"] = 100.0 * dCa / dPrompt if dPrompt > 0 else None
        dF = max(0.0, cur.get("flops", 0) - pv.get("flops", 0))
        row["tflops"] = dF / dt / 1e12 if dF > 0 else None
        row["accpct"] = 100 * dA / dD if dD > 0 else None
        row["draftrate"] = max(0.0, dD) / dt
        row["tau"] = 1 + dA / dN if dN > 0 else None
        for i in range(7):
            prev_pos = pv["pos"][i] if i < len(pv.get("pos", [])) else 0
            row[f"pos{i}"] = (cur["pos"][i] - prev_pos) / dN if dN > 0 else None
    state["prev_" + slot] = cur
    state["engine"] = "vllm" if slot == "vllm" else "vllm@8888"


def tick_sglang(now, row):
    p = parse_prom(fetch(SGLANG))
    if any(k.startswith("vllm:") for k in p):
        # :8888 is serving vLLM (EXL3 kit) -- parse with the vLLM mapping.
        return _fill_vllm(p, now, row, "sglang")
    if not any(k.startswith("sglang:") for k in p):
        raise RuntimeError("no recognizable engine metrics on :8888")
    e = p.get("sglang:num_running_reqs", [])
    if e: state["model"] = e[0][0].get("model_name")
    # generation_tokens_total / prompt_tokens_total are DECEPTIVE for real-time
    # use: verified empirically (polled every 1-3s across an 80s+ single decode)
    # that SGLang only finalizes them once the WHOLE request completes, not
    # incrementally per token or per decode step like vLLM's equivalents. A
    # delta/dt against them shows 0 tok/s for a request's entire duration, then
    # one spike on the tick after it finishes -- which is exactly the "not
    # real-time" symptom. Still used below for the CUMULATIVE tok_in/tok_out
    # display and for /totals' rate-integration, where end-of-request-only
    # updates are fine (the total is still correct, just not live mid-request).
    cur = {
        "gen": psum(p, "sglang:generation_tokens_total"),
        "pp": psum(p, "sglang:prompt_tokens_total"),
        "t": now,
    }
    # gen_throughput IS a genuine live gauge (verified: moves during an active
    # decode, reports exactly 0.0 when idle, no stale-value carryover) -- use it
    # directly for the real-time "gen" tok/s instead of the dead counter above.
    gt = pget(p, "sglang:gen_throughput")
    row["gen"] = gt if gt is not None else None
    # No equivalent live gauge exists for prefill (checked the full metric list).
    # prompt_tokens_total is still delta/dt'd below for "pp" -- imperfect for a
    # single very long prefill, but chunked_prefill_size splits big prompts into
    # multiple scheduler passes that each seem to land a partial update, and most
    # real prompts finish prefill within one collector tick (2.5s) anyway.
    # token_usage is the scheduler's admission-control KV/state pool fraction --
    # closest analogue to vLLM's kv_cache_usage_perc.
    kv = pget(p, "sglang:token_usage")
    row["kv"] = kv * 100 if kv is not None else None
    pfx = pget(p, "sglang:cache_hit_rate")
    row["pfx"] = pfx * 100 if pfx is not None else None
    row["run"] = psum(p, "sglang:num_running_reqs")
    row["wait"] = psum(p, "sglang:num_queue_reqs")
    row["tok_total"] = cur["gen"] + cur["pp"]
    row["tok_in"] = cur["pp"]
    row["tok_out"] = cur["gen"]
    row["req_ok"] = psum(p, "sglang:num_requests_total")
    # SGLang exposes spec_accept_rate/spec_accept_length as live gauges already
    # (not cumulative counters), so unlike vLLM these need no delta -- they ARE
    # the current-window acceptance rate / mean accepted length.
    accr = pget(p, "sglang:spec_accept_rate")
    row["accpct"] = accr * 100 if accr is not None else None
    row["tau"] = pget(p, "sglang:spec_accept_length")
    # draftrate and pos0-4 have no clean SGLang equivalent: DFlash2 is a
    # block-diffusion drafter (no per-position chain-accept counters like
    # vLLM's MTP), and spec_num_draft_tokens is a static config value, not a
    # cumulative counter. Left None rather than faked.
    state["ring_sglang"].append((now, buckets(p, "sglang:time_to_first_token_seconds"),
                                 buckets(p, "sglang:inter_token_latency_seconds")))
    state["ring_sglang"] = [r for r in state["ring_sglang"] if now - r[0] <= HIST_WINDOW]
    old = state["ring_sglang"][0]
    row["ttft50"] = pctile(state["ring_sglang"][-1][1], old[1], .5)
    row["ttft99"] = pctile(state["ring_sglang"][-1][1], old[1], .99)
    row["itl50"] = pctile(state["ring_sglang"][-1][2], old[2], .5)
    row["itl99"] = pctile(state["ring_sglang"][-1][2], old[2], .99)
    pv = state["prev_sglang"]
    if pv:
        dt = now - pv["t"] or 1.0
        # row["gen"] already set from the live gen_throughput gauge above --
        # do NOT overwrite it here with the dead-counter delta.
        row["pp"] = max(0.0, cur["pp"] - pv["pp"]) / dt
    state["prev_sglang"] = cur
    state["engine"] = "sglang"


def tick():
    now = time.time()
    row = {k: None for k in COLS}
    engine_up = False
    # Try vLLM first, then SGLang. Whichever one is actually up wins the tick;
    # the loser's prev-state is reset so a later switch back doesn't compute a
    # bogus rate against a stale counter epoch from a different engine.
    try:
        tick_vllm(now, row)
        engine_up = True
        state["prev_sglang"] = None
    except Exception:
        state["prev_vllm"] = None
        try:
            tick_sglang(now, row)
            engine_up = True
        except Exception:
            state["prev_sglang"] = None
    # ---- agents ----
    nodes = {}
    for key in ("h", "w"):
        d = poll_agent(key)
        nodes[key] = d
        if d:
            row[f"{key}_gpu"] = d["gpu"]["util"]; row[f"{key}_temp"] = d["gpu"]["temp"]
            row[f"{key}_power"] = d["gpu"]["power"]; row[f"{key}_mem"] = d["mem"]["used_gib"]
            row[f"{key}_cpu"] = d["cpu_pct"]; row[f"{key}_net"] = net_rate(key, d)
    state["live"] = {"ts": now, "engine_up": engine_up, "model": state["model"],
                     "engine": state["engine"], "row": row, "nodes": nodes}
    return now, row

def loop():
    conn = db()
    last_prune = 0.0
    while True:
        t0 = time.time()
        try:
            ts, row = tick()
            conn.execute(f"INSERT OR REPLACE INTO samples (ts, {','.join(COLS)}) VALUES ({','.join(['?']*(len(COLS)+1))})",
                         [ts] + [row[k] for k in COLS])
            conn.commit()
            if ts - last_prune > 3600:
                conn.execute("DELETE FROM samples WHERE ts < ?", (ts - RETAIN_S,))
                conn.commit(); last_prune = ts
        except Exception:
            pass
        time.sleep(max(0.2, TICK - (time.time() - t0)))

# ---------------- API ----------------
class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/live":
            self._send(state["live"] or {}); return
        if u.path == "/totals":
            # Lifetime totals. The raw vLLM counters reset on every engine
            # restart, so instead of differencing them we integrate the per-tick
            # rates (tok/s) that were already derived from counter deltas — the
            # collector nulls the rate across a restart, so resets contribute 0.
            # dt is capped so downtime gaps between samples are not counted.
            conn = db()
            rows = conn.execute(
                "SELECT ts, pp, gen FROM samples ORDER BY ts").fetchall()
            conn.close()
            tin = tout = 0.0
            prev_ts = None
            for ts, pp, gen in rows:
                if prev_ts is not None:
                    dt = min(ts - prev_ts, TICK * 4)   # ignore long gaps
                    if dt > 0:
                        if pp:  tin += pp * dt
                        if gen: tout += gen * dt
                prev_ts = ts
            self._send({"input_tokens": int(tin), "output_tokens": int(tout),
                        "samples": len(rows),
                        "since": rows[0][0] if rows else None})
            return
        if u.path == "/history":
            q = parse_qs(u.query)
            try:
                t_from = float(q["from"][0]); t_to = float(q["to"][0])
                points = min(2000, max(10, int(q.get("points", ["520"])[0])))
            except Exception:
                self._send({"error": "bad params"}, 400); return
            w = max(TICK, (t_to - t_from) / points)
            conn = db()
            # Bucket AVG drives the plotted line, but avg-of-avg destroys peaks: a
            # 1-month window buckets ~83 min vs ~19 min at 1 week, so "max" taken over
            # averaged points SHRANK as the timeframe grew. Carry true per-bucket
            # extremes for the series that report min/max so they stay timeframe-stable.
            aggs = ", ".join(f"avg({k})" for k in COLS)
            aggs += ", " + ", ".join(f"max({k}), min({k})" for k in MINMAX_COLS)
            rows = conn.execute(
                f"SELECT CAST((ts-?)/? AS INTEGER) AS b, {aggs} FROM samples "
                f"WHERE ts >= ? AND ts <= ? GROUP BY b ORDER BY b",
                (t_from, w, t_from, t_to)).fetchall()
            conn.close()
            out = {"t": [], **{k: [] for k in COLS}, "bucket_s": w}
            for k in MINMAX_COLS:
                out[k + "_mx"] = []; out[k + "_mn"] = []
            n = len(COLS)
            for r in rows:
                out["t"].append(t_from + (r[0] + 0.5) * w)
                for i, k in enumerate(COLS):
                    v = r[1 + i]
                    out[k].append(round(v, 4) if isinstance(v, float) else v)
                for j, k in enumerate(MINMAX_COLS):
                    vmx = r[1 + n + 2 * j]; vmn = r[1 + n + 2 * j + 1]
                    out[k + "_mx"].append(round(vmx, 4) if isinstance(vmx, float) else vmx)
                    out[k + "_mn"].append(round(vmn, 4) if isinstance(vmn, float) else vmn)
            self._send(out); return
        self.send_response(404); self.end_headers()

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    init_db()
    threading.Thread(target=loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 9102), H).serve_forever()
