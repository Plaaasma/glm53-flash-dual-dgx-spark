#!/usr/bin/env python3
"""Spark cluster history collector.

Polls vLLM /metrics + both node agents every 2.5s, derives per-tick rates
(tok/s, spec acceptance, latency percentiles from histogram deltas, per-node
hardware), and stores everything in SQLite. Serves:

  GET /history?from=<unix>&to=<unix>&points=<n>   bucket-averaged series
  GET /live                                        latest sample + raw node data\n  GET /totals                                      lifetime input/output tokens (restart-aware)

CORS is open — consumed directly by the dashboard page in the browser.
"""
import json, os, sqlite3, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DB_PATH = os.environ.get("SPARK_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db"))
VLLM = "http://localhost:8000/metrics"
# Where the vLLM head runs. Empty SPARK_HEAD_SSH (default) = this node: local docker, metrics on localhost.
# (Every SPARK_HEAD_* variable also accepts its older GLM53_HEAD_* name.)
# Set it to user@<head fabric IP> when the head is the other node: the head container is then reached through
# docker's ssh transport (one multiplexed connection), metrics over VLLM_METRICS_URL, the boot phase file over
# ssh, and the engine's viz UDP must be pointed at this node (MIMO26_VIZ_UDP / GLM53_VIZ_UDP in the head's .env).
_env = lambda k, d="": os.environ.get("SPARK_" + k, os.environ.get("GLM53_" + k, d))
HEAD_SSH = _env("HEAD_SSH")
HEAD_KIT = _env("HEAD_KIT", os.path.expanduser("~/mimo26/kit"))
# Serving kits that can own the head, as "container:kit_dir" pairs (SPARK_HEAD_KITS). The first pair whose
# container is running wins (falls back to the first pair that exists at all); re-checked every 15 s, so the
# boot bar / KV pool / logs follow whichever kit is booting without a collector restart.
_DEFAULT_HEAD_CTN = "mimo26-head"
KITS = [tuple(k.split(":", 1)) for k in _env("HEAD_KITS").split(",") if ":" in k] \
    or [(_DEFAULT_HEAD_CTN, HEAD_KIT)]
DOCKER = ["docker", "-H", f"ssh://{HEAD_SSH}"] if HEAD_SSH else ["docker"]
SGLANG = os.environ.get("VLLM_METRICS_URL", "http://localhost:8888/metrics")
# Node agents (agent.py on :9101): "h" is the node this collector runs on, "w" the other node.
AGENTS = {"h": os.environ.get("SPARK_AGENT_H", "http://localhost:9101/stats"),
          "w": os.environ.get("SPARK_AGENT_W", "")}
NODE_LABELS = os.environ.get("SPARK_NODE_LABELS", "HEAD · API,WORKER").split(",")   # role labels for the page, h then w
PROC_CAP_MIB = int(os.environ.get("SPARK_PROC_CAP_MIB", "0") or 0)                  # per-node cap for watched processes, 0 = none
TICK = 2.5
RETAIN_S = 35 * 86400          # keep 35 days
HIST_WINDOW = 120              # seconds of histogram ring for percentiles

COLS = ["gen","pp","accpct","draftrate","tau","kv","pfx",
        "ttft50","ttft99","itl50","itl99","run","wait","dec",
        "pos0","pos1","pos2","pos3","pos4","pos5","pos6",
        "wait_cap","wait_def","steps","stepsz","q50","q99","cachedpct","tflops",
        "tok_total","tok_in","tok_out","req_ok","preempt",
        "h_gpu","h_temp","h_power","h_mem","h_cpu","h_net",
        "w_gpu","w_temp","w_power","w_mem","w_cpu","w_net"]

# Per-model numbers the page and dash_server need (served model name prefix -> profile), exposed as /live "profile".
#   flops_per_token : ~2 x active params, for the TF/s fallback (vLLM's estimator does not know these archs)
#   page_tokens     : tokens per KV page of the attention group, for the pool map
#   bw              : per-rank memory traffic per engine step for the reactor gauge:
#                     distinct routed experts x expert_gb x moe_layers + fixed_gb (dense weights, drafter, lm_head)
#                     + KV reads (context tokens x kv_bytes_per_tok + per-request swa bytes), or kv_gb_per_run
MODEL_PROFILES = {
    # MiMo-V2.6-Flash (TP=2, NVFP4 KV): active ~14.8B = 48 x attention ~90M + 47 MoE x 8 x 3*4096*2048 + dense layer 0
    # (3*4096*16384) + lm_head 152576*4096. Expert MXFP4 (4.25 bit) 13.4 MB -> 6.68 MB per rank. Fixed per rank per step:
    # attention fp8 qkv + bf16 o_proj 3.0 GB, dense MLP 0.1, routers 0.1, lm_head 0.63, DFlash drafter (bf16 5 layers,
    # 2.9 GB) 1.47 + its lm_head pass 0.63 = ~5.9 GB. KV: 9 global layers x 2 kv heads/rank x 180 B = 3,240 B per
    # context token; 39 sliding-window layers x 128 tokens x 4 heads x 180 B = 3.6 MB per request.
    # With MTP instead of DFlash the drafter's ~2.1 GB/rank/step (5 bf16 layers + its lm_head pass) becomes one MTP
    # layer (~0.3 GB/rank incl. its routed experts at low batch): "fixed_gb_mtp".
    "mimo": {"name": "mimo-v2.6-flash", "flops_per_token": 29.6e9, "page_tokens": 64, "fixed_gb_mtp": 4.1,
             "bw": {"experts": 256, "topk": 8, "moe_layers": 47, "expert_gb": 0.00668, "fixed_gb": 5.9,
                    "kv_bytes_per_tok": 3240, "swa_bytes_per_req": 3.6e6, "kv_gb_per_run": 0.0, "peak_gb_s": 273}},
    # GLM-5.3-Flash EXL3 (retired 2026-09-22): derivation at the tflops fallback below
    "glm": {"name": "glm-5.3-flash", "flops_per_token": 33.4e9, "page_tokens": 7936,
            "bw": {"experts": 288, "topk": 9, "moe_layers": 42, "expert_gb": 0.0063, "fixed_gb": 7.4,
                   "kv_bytes_per_tok": 0, "swa_bytes_per_req": 0, "kv_gb_per_run": 0.067 * 33 / 8, "peak_gb_s": 273}},
}


def model_profile(model=None):
    m = (model if model is not None else state.get("model")) or ""
    for k, p in MODEL_PROFILES.items():
        if m.startswith(k) or m == p["name"]:
            return p
    return MODEL_PROFILES["mimo"]

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


# ---------------- boot progress (EXL3 kit) ----------------
# Stage-based progress: the phases start.sh reports through logs/boot-state.json (teardown, launch, health,
# ready, failed) plus milestones parsed from the head container's own log with real timestamps. Percent is
# weighted by each stage's typical duration (measured on boot 34, 2026-09-17: 159 s end to end) and the ETA is the
# typical time of the stages still ahead, so it reflects where the boot actually is, not wall-clock guessing.
import re as _re
import subprocess as _sp
import json as _json
import datetime as _dt
_kit_cache = {"t": 0.0, "v": KITS[0]}

# The head node's agent (SPARK_AGENT_W, :9101) answers container state and the kit's boot-state file locally, so the
# collector does not poll over ssh: each ssh command is a PAM/logind session on the head, and ~80 a minute kept its
# desktop daemons churning and leaking (bluetoothd re-registered audio profiles per session, polkitd grew ~0.6 GB/day,
# the head ran out of headroom and the memory watchdog killed vLLM on 2026-09-29). ssh stays as the fallback.
_AGENT_HEAD = os.environ.get("SPARK_AGENT_W", "").rsplit("/stats", 1)[0] if HEAD_SSH else ""

def _agent_json(path):
    if not _AGENT_HEAD:
        return None
    try:
        with urllib.request.urlopen(_AGENT_HEAD + path, timeout=3) as r:
            return _json.loads(r.read())
    except Exception:
        return None

def _active_kit():
    """(container, kit_dir) of the kit that currently owns the head."""
    now = time.time()
    if now - _kit_cache["t"] < 15.0:
        return _kit_cache["v"]
    v = _kit_cache["v"]
    try:
        d = _agent_json("/docker?names=" + ",".join(k[0] for k in KITS))
        if d is not None:
            states = {n: x.get("status") for n, x in d.items()}
        else:
            out = _sp.run(DOCKER + ["ps", "-a", "--format", "{{.Names}} {{.State}}"], capture_output=True, text=True, timeout=4)
            states = dict(ln.split(None, 1) for ln in out.stdout.splitlines() if " " in ln)
        running = [k for k in KITS if states.get(k[0]) == "running"]
        present = [k for k in KITS if k[0] in states]
        v = (running or present or [KITS[0]])[0]
    except Exception:
        pass
    _kit_cache.update(t=now, v=v)
    return v

def head_ctn():
    return _active_kit()[0]

def boot_state_path():
    return _active_kit()[1] + "/logs/boot-state.json"
# (key, label, typical seconds, marker regex in the head log that ENDS the stage; None = ended by state file).
# Per serving kit (container name -> stages): the same log markers, different labels and typical durations.
_BOOT_STAGES_GLM = [
    ("teardown", "stopping old containers, shipping files", 20, None),
    ("patch",    "applying kit patches (both containers)",    8, _re.compile(r"launching: vllm serve")),
    ("init",     "engine init + worker join over CX7",       16, _re.compile(r"Initializing a V1 LLM engine")),
    ("weights",  "streaming weights (InstantTensor)",         52, _re.compile(r"Loading weights took")),
    ("reclaim",  "post-load reclaim + draft weights",          4, _re.compile(r"Loading weights took.*\n(?:.*\n)*?.*Loading weights took")),
    ("kv",       "MoE kernels + KV cache allocation",         28, _re.compile(r"GPU KV cache size")),
    ("graphs",   "CUDA graph capture",                        14, _re.compile(r"Graph capturing finished")),
    ("api",      "API server startup",                        10, _re.compile(r"Application startup complete")),
    ("health",   "health check + post-ready reclaim",          7, None),
]
# MiMo kit, measured on boot 35 (2026-09-22, 160 s launch to healthy): container start 9 s after start.sh, patchers
# <1 s, engine init 21 s, CX7 join + 35 s InstantTensor load 54 s, DFlash weights 11 s, compile + KV 30 s, kernel
# warmup + graphs 30 s, API 5 s.
_BOOT_STAGES_MIMO = [
    ("teardown", "stopping old containers, shm cleanup, headroom check",  9, None),
    ("patch",    "applying kit patches (overlay patchers)",                1, _re.compile(r"launching: vllm serve")),
    ("init",     "API server + engine init",                              21, _re.compile(r"Initializing a V1 LLM engine")),
    ("weights",  "worker join over CX7 + weights (InstantTensor)",        54, _re.compile(r"Loading weights took")),
    ("reclaim",  "DFlash drafter weights",                                11, _re.compile(r"Loading weights took.*\n(?:.*\n)*?.*Loading weights took")),
    ("kv",       "torch.compile + KV cache allocation (NVFP4)",           30, _re.compile(r"GPU KV cache size")),
    ("graphs",   "kernel warmup + CUDA graph capture",                    30, _re.compile(r"Graph capturing finished")),
    ("api",      "API server startup",                                     5, _re.compile(r"Application startup complete")),
    ("health",   "health check",                                           9, None),
]
_BOOT_PROFILES = {"glm53-exl3-head": _BOOT_STAGES_GLM, "mimo26-head": _BOOT_STAGES_MIMO}


def _boot_stages():
    return _BOOT_PROFILES.get(head_ctn(), _BOOT_STAGES_MIMO)
_boot_cache = {"t": 0.0, "log": "", "started": None}

_cstate_cache = {"t": 0.0, "v": (None, None, None)}

def _container_state():
    """(status, started, finished) of the head container; cached 2 s (it is asked several times per tick)."""
    now = time.time()
    if now - _cstate_cache["t"] < 2.0:
        return _cstate_cache["v"]
    p = lambda s: _dt.datetime.fromisoformat(s[:26].rstrip("Z") + "+00:00").timestamp() if s and not s.startswith("0001") else None
    v = (None, None, None)
    try:
        d = _agent_json("/docker?names=" + head_ctn())
        if d is not None:
            x = d.get(head_ctn())
            if x:
                v = (x["status"], p(x["started"]), p(x["finished"]))
        else:
            out = _sp.run(DOCKER + ["inspect", "-f", "{{.State.Status}} {{.State.StartedAt}} {{.State.FinishedAt}}",
                           head_ctn()], capture_output=True, text=True, timeout=3)
            if out.returncode == 0:
                st, started, finished = out.stdout.split()
                v = (st, p(started), p(finished))
    except Exception:
        v = (None, None, None)
    _cstate_cache.update(t=now, v=v)
    return v

def _head_log(started):
    """Head container log since it started, with docker's RFC3339 timestamps (cached 2 s)."""
    now = time.time()
    if now - _boot_cache["t"] < 2.0 and _boot_cache["started"] == started:
        return _boot_cache["log"]
    try:
        out = _sp.run(DOCKER + ["logs", "--timestamps", head_ctn()], capture_output=True, text=True, timeout=8)
        log = (out.stdout or "") + (out.stderr or "")
    except Exception:
        log = ""
    _boot_cache.update(t=now, log=log, started=started)
    return log

def _line_ts(line):
    m = _re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z", line)
    if not m:
        return None
    s = m.group(1)
    return _dt.datetime.fromisoformat(s[:26] + "+00:00").timestamp()

_state_cache = {"t": 0.0, "v": None}

def _read_state():
    """start.sh's phase file lives on the head node; read it over the multiplexed ssh connection, cached 2 s."""
    now = time.time()
    if now - _state_cache["t"] < 2.0:
        return _state_cache["v"]
    v = None
    try:
        a = _agent_json("/bootstate?path=" + boot_state_path()) if HEAD_SSH else None
        if a is not None:
            v = a
        elif HEAD_SSH:
            r = _sp.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "-o", "ControlMaster=auto",
                         "-o", "ControlPath=/tmp/glm53-mux-%C", "-o", "ControlPersist=120", HEAD_SSH, "cat", boot_state_path()],
                        capture_output=True, text=True, timeout=5)
            v = _json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
        else:
            with open(boot_state_path()) as f:
                v = _json.load(f)
    except Exception:
        v = None
    _state_cache.update(t=now, v=v)
    return v

def boot_progress():
    """None when serving; else {stage, stage_key, stage_idx, n_stages, pct, eta_s, elapsed_s, stages:[...], failed}."""
    st, started, finished = _container_state()
    state = _read_state() or {}
    phase, ptime = state.get("phase"), float(state.get("t") or 0)
    now = time.time()
    # A start.sh run in progress (teardown/launch phases) may predate the new container.
    shell_boot = phase in ("teardown", "launching", "containers", "healthy") and now - ptime < 1800
    if not shell_boot and (st is None or st != "running"):
        if phase == "failed" and now - ptime < 6 * 3600:
            return {"stage": "boot failed", "stage_key": "failed", "failed": True, "pct": 0.0, "eta_s": None,
                    "elapsed_s": None, "note": state.get("note"), "stages": []}
        return {"stage": "server stopped", "stage_key": "stopped", "pct": 0.0, "eta_s": None, "elapsed_s": None, "stages": []}
    # Stage timeline: t0 = the shell's launch (state file) or the container start
    t0 = ptime if phase in ("teardown", "launching") and (started is None or ptime <= started) else (started or ptime or now)
    if phase in ("teardown", "launching") and started and started < ptime:
        started = None                     # the running container belongs to the previous boot
    ends = {}                              # stage key -> end timestamp
    if started:
        ends["teardown"] = started
        log = _head_log(started)
        lines = log.splitlines()
        first_weights = None
        for ln in lines:
            ts = _line_ts(ln)
            if ts is None:
                continue
            if "launching: vllm serve" in ln and "patch" not in ends:
                ends["patch"] = ts
            elif "Initializing a V1 LLM engine" in ln and "init" not in ends:
                ends["init"] = ts
            elif "Loading weights took" in ln:
                if first_weights is None:
                    first_weights = ts; ends["weights"] = ts
                elif "reclaim" not in ends:
                    ends["reclaim"] = ts
            elif "GPU KV cache size" in ln and "kv" not in ends:
                ends["kv"] = ts
            elif "Graph capturing finished" in ln and "graphs" not in ends:
                ends["graphs"] = ts
            elif "Application startup complete" in ln and "api" not in ends:
                ends["api"] = ts
    if phase in ("healthy", "ready") and ptime >= (started or 0):
        ends["api"] = min(ends.get("api", ptime), ptime)
        ends["health"] = ptime             # "healthy" = API answering; post-ready reclaim is not part of the boot
    # Walk the stages in order; the first one without an end is the current stage.
    stages, done_typ, cur = [], 0.0, None
    prev_end = t0
    boot_stages = _boot_stages()
    for key, label, typ, _rx in boot_stages:
        end = ends.get(key)
        if end is not None and cur is None:
            stages.append({"key": key, "label": label, "typ_s": typ, "state": "done", "took_s": round(max(0.0, end - prev_end))})
            done_typ += typ; prev_end = end
        elif cur is None:
            cur = (key, label, typ, prev_end)
            stages.append({"key": key, "label": label, "typ_s": typ, "state": "current", "elapsed_s": round(max(0.0, now - prev_end))})
        else:
            stages.append({"key": key, "label": label, "typ_s": typ, "state": "pending"})
    if cur is None:                        # every stage ended: ready
        return None
    key, label, typ, cstart = cur
    in_stage = max(0.0, now - cstart)
    pct = (done_typ + min(in_stage, typ)) / float(sum(s[2] for s in boot_stages)) * 100.0
    remaining = sum(s["typ_s"] for s in stages if s["state"] == "pending") + max(0.0, typ - in_stage)
    idx = next(i for i, s in enumerate(stages) if s["state"] == "current")
    late = in_stage > 1.5 * typ
    return {"stage": label, "stage_key": key, "stage_idx": idx + 1, "n_stages": len(stages),
            "pct": round(min(pct, 99.0), 1), "eta_s": round(remaining), "elapsed_s": round(now - t0),
            "stage_elapsed_s": round(in_stage), "stage_typ_s": typ, "late": late, "stages": stages,
            "failed": False}

# ---------------- KV pool capacity (tokens) ----------------
# The engine prints its real capacity once per boot ("GPU KV cache size: N tokens"): the number that already
# accounts for every cache group sharing the block pool. Parsed from the head container log, cached per
# container start; the scheduler snapshot's pool_tokens is attention-page tokens and overstates it ~2.4x.
_kv_total_cache = {"started": None, "tokens": None, "t": 0.0}

def _effective_profile(drafter):
    """Model profile with the bandwidth model's fixed weights matched to the running draft method."""
    prof = model_profile()
    if drafter and drafter.get("method") == "mtp" and prof.get("fixed_gb_mtp"):
        prof = {**prof, "bw": {**prof["bw"], "fixed_gb": prof["fixed_gb_mtp"]}}
    return prof


_drafter_cache = {"started": None, "v": None}

def drafter_info():
    """Speculative-decoding method of the running engine, from its log (cached per container start):
    {"method": "mtp"|"dflash"|..., "arch": draft architecture, "k": draft tokens per step}."""
    st, started, _fin = _container_state()
    c = _drafter_cache
    if c["started"] == started and c["v"] is not None:
        return c["v"]
    v = None
    if st == "running":
        try:
            out = _sp.run(DOCKER + ["logs", head_ctn()], capture_output=True, text=True, timeout=8)
            log = out.stdout + out.stderr
            archs = _re.findall(r"Resolved architecture: (\w+)", log)
            m = _re.search(r"'speculative_config': \{'method': '(\w+)'.*?'num_speculative_tokens': (\d+)", log)
            if m:
                v = {"method": m.group(1), "k": int(m.group(2)), "arch": archs[1] if len(archs) > 1 else None}
        except Exception:
            v = None
    c["started"], c["v"] = started, v
    return v


def kv_total_tokens():
    st, started, _fin = _container_state()
    now = time.time()
    c = _kv_total_cache
    if c["started"] == started and (c["tokens"] is not None or now - c["t"] < 10.0):
        return c["tokens"]
    c["started"], c["t"] = started, now
    tokens = None
    if st == "running":
        try:
            out = _sp.run(DOCKER + ["logs", head_ctn()], capture_output=True, text=True, timeout=8)
            for ln in reversed((out.stdout + out.stderr).splitlines()):
                m = _re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", ln)
                if m:
                    tokens = int(m.group(1).replace(",", "")); break
        except Exception:
            tokens = None
    c["tokens"] = tokens
    return tokens

# ---------------- collector loop ----------------
# prev_vllm/ring_vllm and prev_sglang/ring_sglang are kept SEPARATE (not shared)
# so that restarting one engine can never be misread as a rate against the
# other engine's last counters -- e.g. vLLM's cumulative gen_tokens_total vs
# SGLang's would produce a nonsense delta if they shared one "prev" slot.
state = {"prev_vllm": None, "ring_vllm": [], "prev_sglang": None, "ring_sglang": [],
         "prev_net": {}, "live": {}, "model": None, "engine": None}

def poll_agent(key):
    if not AGENTS.get(key): return None
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
        # per-step context (prefill) tokens from the glm53 logger patch: counted
        # every engine step, including pure-prefill steps that produce no output
        "ctx_live": psum(p, "vllm:glm53_ctx_tokens_total") or state.get("viz_prefill_total", 0.0),
        "t": now,
    }
    kv = pget(p, "vllm:kv_cache_usage_perc")
    pfh, pfq = psum(p, "vllm:prefix_cache_hits_total"), psum(p, "vllm:prefix_cache_queries_total")
    row["kv"] = kv * 100 if kv is not None else None
    row["pfx"] = 100 * pfh / pfq if pfq > 0 else None
    row["run"] = psum(p, "vllm:num_requests_running")
    # Streams actually decoding (past their prompt) per the scheduler snapshot; a request still
    # prefilling contributes 0 generated tokens, so dividing by "running" understated per-stream
    # decode whenever a prefill was in flight (observed 2026-09-08). None = snapshot stale -> UI falls back to run.
    sch = state.get("viz_sched")
    if sch and now - sch.get("ts", 0) < 5 and isinstance(sch.get("reqs"), list):
        row["dec"] = sum(1 for q in sch["reqs"] if (q.get("prompt") or 0) > 0 and (q.get("computed") or 0) >= q["prompt"])
    else:
        row["dec"] = None
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
        dCtx = cur.get("ctx_live", 0) - pv.get("ctx_live", 0)
        if cur.get("ctx_live", 0) > 0:
            # live: every step counted, no output needed. Averaged over ~12 s: at long contexts one 4K prefill chunk
            # takes several seconds, so a single 2.5 s tick alternates between 0 and a spike.
            win = state.setdefault("ctx_win_" + slot, [])
            if win and cur["ctx_live"] < win[-1][1]:
                win.clear()                        # counter reset (engine or collector restart)
            win.append((now, cur["ctx_live"]))
            while len(win) > 2 and now - win[0][0] > 12.0:
                win.pop(0)
            span = now - win[0][0]
            row["pp"] = (win[-1][1] - win[0][1]) / span if span > 0 else max(dCtx, 0.0) / dt
        else:
            row["pp"] = max(pp_live, 0.0) if dIt > 0 else dPrompt / dt
        dC = max(0.0, cur.get("iter_cnt", 0) - pv.get("iter_cnt", 0))
        row["steps"] = dC / dt
        row["stepsz"] = dIt / dC if dC > 0 else None
        dCa = cur.get("cached", 0) - pv.get("cached", 0)
        row["cachedpct"] = 100.0 * dCa / dPrompt if dPrompt > 0 else None
        # vllm:estimated_flops_per_gpu_total exists but never increments (the
        # analytic estimator doesn't understand this kit's Glm5Next hybrid
        # arch), so fall back to deriving it: tokens forwarded per second
        # (dIt = generated + rejected drafts + prefill, exactly) x 2 FLOPs per
        # active weight. Active ~16.7B from config.json geometry: 45 layers x
        # (MLA ~117M + indexer ~17M), 3 dense FFN x 151M, 42 MoE layers x
        # ((8 routed + 1 shared) x 3*4096*2048 + router) = 227.7M, lm_head
        # 634M. Cluster-total FLOPs, not per-GPU.
        dF = max(0.0, cur.get("flops", 0) - pv.get("flops", 0))
        row["tflops"] = dF / dt / 1e12 if dF > 0 else (
            model_profile()["flops_per_token"] * (dIt / dt) / 1e12 if dIt > 0 else None)
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
    state["live"] = {"ts": now, "engine_up": engine_up, "model": state["model"], "kv_total_tokens": kv_total_tokens(),
                     "node_labels": NODE_LABELS, "proc_cap_mib": PROC_CAP_MIB,
                     "engine": state["engine"], "row": row, "nodes": nodes,
                     "profile": _effective_profile(drafter_info() if engine_up else None), "kit": _active_kit()[1],
                     "drafter": drafter_info() if engine_up else None,
                     "boot": None if engine_up else boot_progress()}
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
def _prefill_counter(f):
    """Live prefill tokens for the throughput panel. The MiMo scheduler hook sends `prefill_total` (prompt tokens
    computed, counted every step); without it, derive the same from how far each running request's prompt progress
    moved since its previous snapshot (a request's first snapshot is not counted: it includes prefix-cache hits)."""
    if "prefill_total" in f:
        state["viz_prefill_total"] = float(f["prefill_total"]) + state.get("viz_prefill_offset", 0.0)
        return
    last = state.setdefault("pf_last", {})
    total = state.get("viz_prefill_total", 0.0)
    seen = {}
    for q in f.get("reqs") or []:
        done = min(q.get("computed") or 0, q.get("prompt") or 0)
        prev = last.get(q.get("id"))
        if prev is not None and done > prev:
            total += done - prev
        seen[q.get("id")] = done
    state["pf_last"] = seen
    state["viz_prefill_total"] = total
    state["viz_prefill_offset"] = total          # keeps the counter monotonic if the engine counter takes over


PROXY_ATTR_URL = os.environ.get("SPARK_PROXY_ATTR_URL", "http://127.0.0.1:8890/_proxy/attribution")


def attribution_poller():
    """Who sent each request: the optional API proxy on this node (dashboard/api-proxy.py, :8888 -> head) lists recent
    generation requests with the client's Tailscale user / machine or LAN name (never content). Matched to the
    scheduler's requests in attribute_reqs()."""
    while True:
        try:
            with urllib.request.urlopen(f"{PROXY_ATTR_URL}?since={time.time() - 3600:.0f}", timeout=3) as r:
                d = json.load(r)
            state["attr_records"] = d.get("records") or []
            state["attr_ok"] = time.time()
        except Exception:
            pass
        time.sleep(1.0)


def _who_label(w):
    w = w or {}
    user, machine, src = w.get("user"), w.get("machine"), w.get("source")
    if src == "tailscale":
        first = (user or "?").split("@")[0].split()[0]
        return f"{first}@{machine}" if machine else first
    if src == "lan":
        return f"{machine} ({user})" if machine and user else (machine or user)
    return None


def attribute_reqs(sch):
    """Attach `who` to each scheduler request: the proxy record forwarded closest before the request reached vLLM
    (arrival = frame ts - age; vLLM stamps arrival when its API server receives the request: 0.3 s after the
    proxy forwarded an 839K-token image request, so the window is tight and requests that bypass the proxy cannot
    steal a proxied request's record). One
    HTTP request can become several engine requests (multi-prompt completions), up to the record's `n`.
    Assignments stick per request id; requests that came in through the old socat forwarder stay unattributed."""
    recs = state.get("attr_records") or []
    amap = state.setdefault("attr_map", {})
    used = state.setdefault("attr_used", {})
    now = time.time()
    out = []
    for q in sch.get("reqs") or []:
        key = q.get("id")
        hit = amap.get(key)
        if hit is None and recs:
            arrival = float(sch.get("ts") or now) - float(q.get("age") or 0)
            best, best_d = None, None
            for r in recs:
                t = r.get("t_fwd") or r.get("t_recv")
                if t is None or used.get(r["rid"], 0) >= (r.get("n") or 1):
                    continue
                d = arrival - t
                if d < -1.5 or d > 5.0:
                    continue
                if r.get("t_end") and r["t_end"] < arrival - 1.0:
                    continue
                if best_d is None or abs(d) < abs(best_d):
                    best, best_d = r, d
            if best is not None:
                used[best["rid"]] = used.get(best["rid"], 0) + 1
                hit = amap[key] = {"who": best.get("who") or {}, "ip": best.get("ip"), "dt": round(best_d, 2), "seen": now}
        if hit:
            hit["seen"] = now
            w = hit["who"]
            q = dict(q, who={"label": _who_label(w) or hit.get("ip"), "user": w.get("user"), "login": w.get("login"),
                             "machine": w.get("machine"), "source": w.get("source") or "ip"})
        try:
            stats = req_stats(key, now)
        except Exception:
            stats = None
        if stats:
            q = dict(q, stats=stats)
        out.append(q)
    if len(amap) > 500:
        for k in [k for k, v in amap.items() if now - v.get("seen", 0) > 900]:
            del amap[k]
        live = {r["rid"] for r in recs}
        for k in [k for k in used if k not in live]:
            del used[k]
    return dict(sch, reqs=out, attribution=bool(state.get("attr_ok") and now - state["attr_ok"] < 10))


REQ_T0 = time.time()   # requests that arrived before this collector started have no exact TTFT / queue time


def _req_stats_update(f):
    """Per-request timeline from the scheduler snapshots (counts only): arrival, first time seen running, first
    generated token, and a short (ts, computed, total) history for live prefill / generation rates."""
    import collections as _c
    ts = float(f.get("ts") or time.time())
    rs = state.setdefault("rstats", {})
    for q in f.get("reqs") or []:
        key = q.get("id")
        if not key:
            continue
        prompt, computed, total = int(q.get("prompt") or 0), int(q.get("computed") or 0), int(q.get("total") or 0)
        st = rs.get(key)
        if st is None:
            arrival = ts - float(q.get("age") or 0)
            fresh = arrival >= REQ_T0
            st = rs[key] = {"arrival": arrival, "first_seen": ts, "fresh": fresh, "prompt": prompt,
                            "cached": None, "first_tok": None, "tok_total0": None, "tok_exact": total <= prompt,   # prefill->decode seen = exact TTFT
                            "tok_upper": fresh and total > prompt,   # prefill done between two frames: first sight bounds TTFT
                            "hist": _c.deque(maxlen=400), "acc": _c.deque(maxlen=64), "acc_steps": set()}
        st["last_seen"] = ts
        st["hist"].append((ts, computed, total))
        if q.get("cached") is not None:          # exact prefix-cache hit from vLLM (kit hook >= 2026-09-24)
            st["cached"] = int(q["cached"])
        if st["first_tok"] is None and total > prompt:
            st["first_tok"], st["tok_total0"] = ts, total
    if len(rs) > 64:
        for k in [k for k, v in rs.items() if ts - v.get("last_seen", 0) > 600]:
            del rs[k]


def _req_acc_update(a3):
    """DFlash tokens emitted per step for each request (1 + accepted drafts), from the activation frames."""
    rs = state.get("rstats") or {}
    step = a3.get("step")
    for key, n in a3.get("acc") or []:
        st = rs.get(key)
        if st is not None and (step is None or step not in st["acc_steps"]):
            st["acc"].append(int(n))
            if step is not None:
                st["acc_steps"].add(step)
                if len(st["acc_steps"]) > 256:
                    st["acc_steps"] = set(sorted(st["acc_steps"])[-128:])


def _rate(hist, idx, window=20.0):
    """tokens/s of hist[*][idx] over the last `window` seconds (None without two points)."""
    if len(hist) < 2:
        return None
    t1, v1 = hist[-1][0], hist[-1][idx]
    t0, v0 = hist[0][0], hist[0][idx]
    for t, *vals in reversed(hist):
        if t1 - t >= window:
            t0, v0 = t, vals[idx - 1]
            break
    return (v1 - v0) / (t1 - t0) if t1 - t0 >= 0.5 else None


def req_stats(key, now):
    st = (state.get("rstats") or {}).get(key)
    if not st:
        return None
    h = list(st["hist"])          # snapshot: the UDP thread keeps appending
    acc = list(st["acc"])
    last = h[-1] if h else (now, 0, 0)
    prompt, computed, total = st["prompt"], last[1], last[2]
    prefilling = computed < prompt
    out = {"phase": "prefill" if prefilling else "decode",
           "prefill_tps": _rate(h, 1) if prefilling else None,
           "gen_tps": None if prefilling else _rate(h, 2),
           "generated": max(0, total - prompt),
           "queue_s": round(st["first_seen"] - st["arrival"], 2) if st["fresh"] else None,
           "cached": st["cached"],
           "ttft_s": round(st["first_tok"] - st["arrival"], 2) if (st["first_tok"] and (st["tok_exact"] or st["tok_upper"])) else None,
           "ttft_upper": bool(st.get("tok_upper")),
           "tau": round(sum(acc[-32:]) / len(acc[-32:]), 2) if acc else None}
    if st["first_tok"] and not prefilling and last[0] - st["first_tok"] >= 1.0:
        out["gen_tps_avg"] = round((total - st["tok_total0"]) / (last[0] - st["first_tok"]), 2)
    if st["fresh"] and st["first_tok"] and st["tok_exact"]:
        pf_t = st["first_tok"] - st["first_seen"]
        out["prefill_tps_avg"] = round((prompt - (st["cached"] or 0)) / pf_t, 1) if pf_t >= 0.5 else None
    for k in ("prefill_tps", "gen_tps"):
        if out[k] is not None:
            out[k] = round(out[k], 2)
    return out


def viz_udp_listener(port=9103):
    """Receive live-activation frames ('act') and scheduler snapshots ('sched')
    sent by the engine hooks (mimo26_viz_runtime / glm53_viz_runtime + the kit's patch_viz_hooks) as UDP JSON."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    s.bind(("0.0.0.0", port))   # the engine hooks send from the head node over the fabric
    while True:
        try:
            data, _ = s.recvfrom(65535)
            f = json.loads(data.decode())
            k = f.get("kind")
            if k == "act": state["viz_act"] = f
            elif k == "sched":
                state["viz_sched"] = f
                _prefill_counter(f)
                _req_stats_update(f)
            elif k == "viz_status": state["viz_status"] = f
            elif k == "act3d":
                state["viz_act3d"] = f
                _req_acc_update(f)
            elif k == "act3h": state["viz_act3h"] = f
        except Exception:
            pass


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
        if u.path == "/viz":
            # Latest activation frame (from the TP0 worker) and scheduler
            # snapshot (from the EngineCore), both arriving over UDP :9103.
            now = time.time()
            act, sch = state.get("viz_act"), state.get("viz_sched")
            vs = state.get("viz_status"); a3 = state.get("viz_act3d"); ah = state.get("viz_act3h")
            self._send({"act": act, "act_age": (now - act["ts"]) if act else None,
                        "act3d": a3, "act3d_age": (now - a3["ts"]) if a3 else None,
                        "act3h": ah, "act3h_age": (now - ah["ts"]) if ah else None,
                        "sched": attribute_reqs(sch) if sch else None, "sched_age": (now - sch["ts"]) if sch else None,
                        "status": vs, "status_age": (now - vs["ts"]) if vs else None})
            return
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
    threading.Thread(target=viz_udp_listener, daemon=True, name="viz-udp").start()
    threading.Thread(target=attribution_poller, daemon=True, name="attribution").start()
    ThreadingHTTPServer(("0.0.0.0", 9102), H).serve_forever()
