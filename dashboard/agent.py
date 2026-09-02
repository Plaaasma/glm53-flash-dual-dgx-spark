#!/usr/bin/env python3
"""Tiny node metrics agent for the Spark cluster dashboard.

Serves GET /stats as JSON on :9101 (CORS *). Sources chosen for GB10 quirks:
nvidia-smi memory fields are N/A on unified memory, so RAM comes from
/proc/meminfo (MemTotal-MemAvailable = what's actually committed).
"""
import json, os, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST_NAME = os.uname().nodename
_prev_cpu = None
_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()

# nvidia-smi goes unresponsive when the GPU is saturated (heavy model loads wedge
# it for minutes at a time). It used to be called inline while holding _lock, so
# one stuck call blocked every request for 5s+ and piled the rest up behind the
# lock -- past the collector's 2s timeout, which showed the whole node as OFFLINE
# exactly when there was most to look at. GPU polling now lives on a background
# thread and requests serve the last good sample, so /stats never blocks on it.
GPU_POLL_S = 2.0
_gpu = {"t": 0.0, "data": {"util": None, "temp": None, "power": None, "sm_mhz": None}}
_gpu_lock = threading.Lock()


def read_cpu_pct():
    global _prev_cpu
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + vals[4]
    total = sum(vals)
    if _prev_cpu is None:
        _prev_cpu = (idle, total)
        return 0.0
    didle, dtotal = idle - _prev_cpu[0], total - _prev_cpu[1]
    _prev_cpu = (idle, total)
    return round(100.0 * (1 - didle / dtotal), 1) if dtotal > 0 else 0.0


def read_mem():
    m = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            k, v = ln.split(":", 1)
            m[k] = int(v.split()[0])  # kB
    total = m["MemTotal"] / 1048576
    avail = m["MemAvailable"] / 1048576
    return {"total_gib": round(total, 1), "used_gib": round(total - avail, 1)}


def read_gpu():
    """Blocking nvidia-smi query. Only ever called from the poller thread."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        f = lambda s: None if "N/A" in s else float(s.strip())
        return {"util": f(out[0]), "temp": f(out[1]), "power": f(out[2]), "sm_mhz": f(out[3])}
    except Exception:
        return None


def gpu_poller():
    """Refresh the GPU sample off the request path, forever.

    A failed/timed-out read keeps the previous sample rather than blanking it --
    the consumer decides staleness from gpu_age_s, so a briefly wedged driver
    reads as 'last known' instead of a node that dropped off the dashboard.
    """
    while True:
        d = read_gpu()
        if d is not None:
            with _gpu_lock:
                _gpu.update(t=time.time(), data=d)
        time.sleep(GPU_POLL_S)


def get_gpu():
    with _gpu_lock:
        return dict(_gpu["data"]), _gpu["t"]


def read_net():
    ifaces = {}
    base = "/sys/class/net"
    for i in os.listdir(base):
        if i == "lo":
            continue
        try:
            with open(f"{base}/{i}/operstate") as f:
                if f.read().strip() != "up":
                    continue
            rx = int(open(f"{base}/{i}/statistics/rx_bytes").read())
            tx = int(open(f"{base}/{i}/statistics/tx_bytes").read())
            ifaces[i] = {"rx": rx, "tx": tx}
        except OSError:
            continue
    return ifaces


def read_disk():
    st = os.statvfs("/")
    total = st.f_blocks * st.f_frsize / 2**30
    free = st.f_bavail * st.f_frsize / 2**30
    return {"total_gib": round(total, 1), "used_gib": round(total - free, 1)}


def collect():
    now = time.time()
    with _lock:
        if _cache["data"] and now - _cache["t"] < 1.0:
            return _cache["data"]
        gpu, gpu_t = get_gpu()
        d = {
            "host": HOST_NAME, "ts": now,
            # Everything below GPU is a plain /proc or statvfs read -- cheap and
            # non-blocking, so /stats stays fast no matter what the driver is doing.
            "gpu": gpu, "mem": read_mem(), "cpu_pct": read_cpu_pct(),
            "net": read_net(), "disk": read_disk(),
            "load1": round(os.getloadavg()[0], 2),
            # Age of the GPU sample. None = never got one since start.
            "gpu_age_s": None if gpu_t == 0.0 else round(now - gpu_t, 1),
        }
        _cache.update(t=now, data=d)
        return d


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/delay"):
            # test helper: hold the response N ms (default 6000) then 204 —
            # lets a screenshot harness postpone the page load event until data arrives
            try: ms = int(self.path.split("ms=")[1])
            except Exception: ms = 6000
            time.sleep(min(ms, 15000) / 1000)
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); return
        if self.path != "/stats":
            self.send_response(404); self.end_headers(); return
        body = json.dumps(collect()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    read_cpu_pct()  # prime the delta
    threading.Thread(target=gpu_poller, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 9101), H).serve_forever()
