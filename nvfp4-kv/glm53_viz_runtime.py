# SPDX-License-Identifier: MIT
"""Live activation telemetry for the Spark dashboard (GLM-5.3-Flash EXL3 stack).

Records three signals per decode step from inside the CUDA graphs, with no
host syncs: async device->pinned copies only (memcpy nodes in the graph).

  routing : per MoE layer, the routed expert ids of every token   [L_moe, T, K]
  attn    : per sparse-MLA layer, logical top-k context positions  [L_mla, T, 2048]
  ribbon  : per decoder layer, mean L2 norm of the residual stream [L_all]

A publisher thread polls a pinned step counter and, when it advances, folds
the buffers into small histograms and sends one UDP datagram (JSON) to the
collector. Everything is gated on env GLM53_VIZ=1 and degrades to no-ops.
Overhead budget: ~60 memcpy nodes + (ribbon) ~3 tiny kernels/layer per step.
"""
import base64, json, os, socket, threading, time
import numpy as np
import torch

_ON = os.environ.get("GLM53_VIZ", "0") == "1"
_RIBBON = os.environ.get("GLM53_VIZ_RIBBON", "1") == "1"
_UDP = os.environ.get("GLM53_VIZ_UDP", "127.0.0.1:9103")
_HZ = float(os.environ.get("GLM53_VIZ_HZ", "12"))
MAX_T = int(os.environ.get("GLM53_VIZ_MAX_T", "256"))   # tokens per step we record (decode+mixed steps)
N_EXPERTS, TOPK, N_MOE, N_MLA, N_ALL, TOPK_ATTN, ATTN_BINS = 288, 9, 43, 13, 46, 2048, 256
MAX_T3, MAX_TA, ATTN3_BINS = 32, 16, 64      # per-token 3D frame: routing tokens, attention tokens, context bins


def enabled() -> bool:
    return _ON


class _State:
    def __init__(self):
        self.ready = False
        self.lock = threading.Lock()
        self.moe_ord: dict = {}     # layer identity -> ordinal (first-call order)
        self.mla_ord: dict = {}
        self.rank = -1
        self.capture_seen = 0.0     # last time a hook ran inside a CUDA graph capture
        self.batch = []             # [(request id tail, first token index, token count)] for the current step

    def init(self, device: torch.device):
        if self.ready:
            return
        with self.lock:
            if self.ready:
                return
            self.dev = device
            # vLLM runs the forward under torch.inference_mode(); tensors created
            # there are "inference tensors" that no other context may update
            # in place (the publisher thread hit exactly that). Allocate as
            # normal tensors: in-graph updates from inference mode stay legal.
            self._im = torch.inference_mode(False); self._im.__enter__()
            # Hooks record into DEVICE buffers only (device->device copies are
            # always graph-capturable; host-pinned copies are rejected by this
            # fork's piecewise capture). The publisher pulls them to the host
            # on its own stream, outside any capture.
            dz = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt, device=device)
            self.d_routing = dz(N_MOE, MAX_T, TOPK)
            self.d_attn = dz(N_MLA, MAX_T, TOPK_ATTN)
            self.d_ribbon = dz(N_ALL, dt=torch.float32)
            self.d_meta = dz(4)                        # [step, T, -, -]
            self.d_h3 = dz(MAX_T, 3, dt=torch.float32) # final hidden state, random-projected to 3D
            self.h_h3 = None                           # pinned twin, created with the others below
            self.proj = None                           # [hidden, 3] fixed random projection (lazy, eager step)
            self.d_one = torch.ones(1, dtype=torch.int32, device=device)
            pin = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt, pin_memory=True)
            self.h_routing = pin(N_MOE, MAX_T, TOPK)
            self.h_attn = pin(N_MLA, MAX_T, TOPK_ATTN)
            self.h_ribbon = pin(N_ALL, dt=torch.float32)
            self.h_meta = pin(4)
            self.h_h3 = pin(MAX_T, 3, dt=torch.float32)
            for nm in ("h_routing", "h_attn", "h_ribbon", "h_meta", "h_h3"):
                b = getattr(self, nm)
                if not b.is_pinned():
                    setattr(self, nm, b.pin_memory())
            self.pub_stream = torch.cuda.Stream(device=device)
            self._im.__exit__(None, None, None); del self._im
            self.ready = True
            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
                self.rank = get_tensor_model_parallel_rank()
            except Exception:
                self.rank = 0
            if self.rank == 0:
                threading.Thread(target=_publisher, args=(self,), daemon=True, name="glm53-viz").start()


S = _State()


def _ordinal(table: dict, key, limit: int) -> int:
    o = table.get(key)
    if o is None:
        o = len(table)
        table[key] = o
    return o if o < limit else -1


# ---------------------------------------------------------------- hooks ----
_DEAD = False


_DEAD_REASON = ""


def _disarm(where: str, exc: BaseException) -> None:
    global _DEAD, _DEAD_REASON
    if not _DEAD:
        _DEAD = True
        _DEAD_REASON = f"{where}: {str(exc)[:160]}"
        try:
            from vllm.logger import init_logger
            init_logger("vllm.glm53_viz").warning("viz telemetry disabled after %s failed: %s", where, exc)
        except Exception:
            pass


def begin_step(num_tokens: int, device: torch.device) -> None:
    """Call once at the top of the model forward (captured into the graph)."""
    if not _ON or _DEAD:
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            S.capture_seen = time.time()            # publisher must stay off the GPU while captures run
            if not S.ready:
                return                              # never allocate inside a capture; arm on the next eager step
        S.init(device)
        S.d_meta[0].add_(S.d_one[0])          # step counter (device-side, graph-safe)
        S.d_meta[1].fill_(min(num_tokens, MAX_T))
    except Exception as e:
        _disarm("begin_step", e)


def record_routing(layer_key, topk_ids: torch.Tensor) -> None:
    """topk_ids: [T, K] int on device. Async copy into the pinned slot."""
    if not _ON or _DEAD or not S.ready:
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            S.capture_seen = time.time()            # drafter captures reach here without begin_step
        o = _ordinal(S.moe_ord, layer_key, N_MOE)
        if o < 0:
            return
        t = min(topk_ids.shape[0], MAX_T)
        S.d_routing[o, :t, : topk_ids.shape[1]].copy_(topk_ids[:t])
    except Exception as e:
        _disarm("record_routing", e)


def record_attn(layer_key, topk_indices: torch.Tensor) -> None:
    """topk_indices: [T, 2048] int32 logical positions (-1 padded)."""
    if not _ON or _DEAD or not S.ready:
        return
    try:
        o = _ordinal(S.mla_ord, layer_key, N_MLA)
        if o < 0:
            return
        t = min(topk_indices.shape[0], MAX_T)
        S.d_attn[o, :t, : topk_indices.shape[1]].copy_(topk_indices[:t])
    except Exception as e:
        _disarm("record_attn", e)


def record_ribbon(layer_idx: int, hidden: torch.Tensor) -> None:
    """Mean residual-stream L2 norm for one layer (3 tiny kernels)."""
    if not _ON or _DEAD or not _RIBBON or not S.ready or layer_idx >= N_ALL:
        return
    try:
        S.d_ribbon[layer_idx].copy_(hidden.detach().float().norm(dim=-1).mean())
    except Exception as e:
        _disarm("record_ribbon", e)


_MAINT_EVERY = float(os.environ.get("GLM53_MEM_MAINT_S", "600"))
_MAINT_EMPTY = os.environ.get("GLM53_MEM_MAINT_EMPTY_CACHE", "1") == "1"
_maint_last = [time.time()]


def _mem_maint() -> None:
    """Every GLM53_MEM_MAINT_S: log process/host/torch memory (both ranks) and, unless a
    capture is in progress, return caching-allocator slack. Added 2026-09-05 after both
    nodes ramped ~0.17 GiB/h over 14 h of serving until the worker's watchdog tripped."""
    now = time.time()
    if now - _maint_last[0] < _MAINT_EVERY:
        return
    _maint_last[0] = now
    try:
        import gc
        from vllm.logger import init_logger
        log = init_logger("vllm.glm53_mem")
        st = {}
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith(("VmRSS", "RssAnon", "RssShmem", "VmSwap")):
                    p = l.split(); st[p[0].rstrip(":")] = int(p[1]) // 1024
        avail = 0
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemAvailable"):
                    avail = int(l.split()[1]) // 1024; break
        a0, r0 = torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20
        released = 0.0
        if _MAINT_EMPTY and not torch.cuda.is_current_stream_capturing():
            gc.collect(); torch.cuda.empty_cache()
            released = r0 - torch.cuda.memory_reserved() / 2**20
        log.info("[glm53-mem] MemAvailable=%d MiB | proc VmRSS=%s RssAnon=%s RssShmem=%s VmSwap=%s MiB | torch allocated=%.0f reserved=%.0f MiB | empty_cache released %.0f MiB",
                 avail, st.get("VmRSS"), st.get("RssAnon"), st.get("RssShmem"), st.get("VmSwap"), a0, r0, released)
    except Exception:
        pass


def set_batch(input_batch) -> None:
    """Per-request token spans of the step (CPU data only; called by the model runner each step)."""
    _mem_maint()
    if not _ON or _DEAD:
        return
    try:
        ids = list(input_batch.req_ids)[: input_batch.num_reqs]
        qsl = input_batch.query_start_loc_np
        S.batch = [(str(r)[-8:], int(qsl[i]), int(qsl[i + 1] - qsl[i])) for i, r in enumerate(ids)]
    except Exception as e:
        _disarm("set_batch", e)


def record_hidden3d(hidden: torch.Tensor) -> None:
    """Final residual stream [T, hidden] -> fixed random 3-D projection (one tiny matmul)."""
    if not _ON or _DEAD or not S.ready:
        return
    try:
        if S.proj is None or S.proj.shape[0] != hidden.shape[-1]:
            if torch.cuda.is_current_stream_capturing():
                return                              # created on an eager step (profile run), never inside a capture
            g = torch.Generator(device="cpu").manual_seed(53)
            p = torch.randn(hidden.shape[-1], 3, generator=g)
            S.proj = (p / p.norm(dim=0, keepdim=True)).to(hidden.device)
        t = min(hidden.shape[0], MAX_T)
        S.d_h3[:t].copy_(hidden[:t].float() @ S.proj)
    except Exception as e:
        _disarm("record_hidden3d", e)


def end_step() -> None:
    """Nothing to enqueue: all recording is device-side; the publisher pulls."""
    return


# ------------------------------------------------------------ publisher ----
def _fold(state: _State, T: int, n_moe: int, n_mla: int) -> dict:
    T = max(1, min(T, MAX_T))
    r = state.h_routing[:n_moe, :T].numpy()                       # [L, T, K]
    hist = np.zeros((N_MOE, N_EXPERTS), dtype=np.uint16)
    for l in range(n_moe):
        ids = r[l].reshape(-1)
        ids = ids[(ids >= 0) & (ids < N_EXPERTS)]
        np.add.at(hist[l], ids, 1)
    a = state.h_attn[:n_mla, :T].numpy()                          # [L, T, 2048]
    ah = np.zeros((N_MLA, ATTN_BINS), dtype=np.uint8)
    for l in range(n_mla):
        idx = a[l]
        valid = idx >= 0
        if not valid.any():
            continue
        ctx = int(idx.max()) + 1
        rel = (idx[valid].astype(np.float32) * (ATTN_BINS / max(ctx, 1))).astype(np.int32)
        h = np.bincount(np.clip(rel, 0, ATTN_BINS - 1), minlength=ATTN_BINS).astype(np.float32)
        ah[l] = np.clip(h * (255.0 / max(h.max(), 1.0)), 0, 255).astype(np.uint8)
    rib = state.h_ribbon.numpy().astype(np.float32)
    return {
        "kind": "act", "ts": time.time(), "T": T,
        "n_moe": n_moe, "n_mla": n_mla, "n_all": N_ALL,
        "routing": base64.b64encode(hist.tobytes()).decode(),
        "attn": base64.b64encode(ah.tobytes()).decode(),
        "ribbon": [round(float(x), 3) for x in rib],
    }


def _fold3d(state: _State, T: int, n_moe: int, n_mla: int) -> dict:
    """Per-token frame for the 3-D views (second datagram; ~50 KB)."""
    T3, TA = max(1, min(T, MAX_T3)), max(1, min(T, MAX_TA))
    r = state.h_routing[:n_moe, :T3].numpy().astype(np.int16)              # [L, T3, K]
    a = state.h_attn[:n_mla, :TA].numpy()                                   # [L, TA, 2048]
    ah = np.zeros((N_MLA, MAX_TA, ATTN3_BINS), dtype=np.uint8)
    for l in range(n_mla):
        for t in range(TA):
            idx = a[l, t]; valid = idx >= 0
            if not valid.any():
                continue
            ctx = int(idx.max()) + 1
            rel = np.clip((idx[valid].astype(np.float32) * (ATTN3_BINS / max(ctx, 1))).astype(np.int32), 0, ATTN3_BINS - 1)
            hcount = np.bincount(rel, minlength=ATTN3_BINS).astype(np.float32)
            ah[l, t] = np.clip(hcount * (255.0 / max(hcount.max(), 1.0)), 0, 255).astype(np.uint8)
    h3 = state.h_h3[:T3].numpy().astype(np.float16)
    return {"kind": "act3d", "ts": time.time(), "T3": T3, "TA": TA, "n_moe": n_moe, "n_mla": n_mla, "bins": ATTN3_BINS,
            "reqs": [list(b) for b in state.batch if b[1] < T3],
            "routing3d": base64.b64encode(r.tobytes()).decode(),
            "attn3d": base64.b64encode(ah[:n_mla, :TA].tobytes()).decode(),
            "h3": base64.b64encode(h3.tobytes()).decode()}


def _publisher(state: _State) -> None:
    host, port = _UDP.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last = -1
    period = 1.0 / _HZ
    hb = 0.0
    errs = 0; last_err = ""; last_meta = None
    while True:
        time.sleep(period)
        now = time.time()
        if now - hb > 2.0:
            hb = now
            try:
                st = {"kind": "viz_status", "ts": now, "ready": state.ready, "dead": _DEAD, "reason": _DEAD_REASON,
                      "rank": state.rank, "n_moe": len(state.moe_ord), "n_mla": len(state.mla_ord), "last_step": last,
                      "errors": errs, "last_error": last_err, "meta": last_meta}
                sock.sendto(json.dumps(st).encode(), (host, int(port)))
            except Exception:
                pass
        # Never touch the GPU until at least one CUDA graph capture has been seen AND
        # 6 s have passed since the last one. vLLM captures only at boot (global
        # capture mode): a sync/copy from this thread during a capture invalidates it,
        # and the warmup forward right before the first capture is eager, so a stamp-
        # only guard has a race window at the start of every capture.
        if state.capture_seen == 0.0 or now - state.capture_seen < 6.0:
            continue
        try:
          with torch.inference_mode():
            with torch.cuda.stream(state.pub_stream):
                state.h_meta.copy_(state.d_meta, non_blocking=True)
            state.pub_stream.synchronize()
            step = int(state.h_meta[0]); T = int(state.h_meta[1])
            last_meta = [step, T]
            if step == last or T <= 0:
                continue
            last = step
            n_moe, n_mla = min(len(state.moe_ord), N_MOE), min(len(state.mla_ord), N_MLA)
            with torch.cuda.stream(state.pub_stream):
                state.h_routing[:n_moe, :T].copy_(state.d_routing[:n_moe, :T], non_blocking=True)
                state.h_attn[:n_mla, :T].copy_(state.d_attn[:n_mla, :T], non_blocking=True)
                state.h_ribbon.copy_(state.d_ribbon, non_blocking=True)
                state.h_h3.copy_(state.d_h3, non_blocking=True)
            state.pub_stream.synchronize()
            frame = _fold(state, T, n_moe, n_mla)
            frame3 = _fold3d(state, T, n_moe, n_mla)
          frame["step"] = step; frame3["step"] = step
          sock.sendto(json.dumps(frame).encode(), (host, int(port)))
          sock.sendto(json.dumps(frame3).encode(), (host, int(port)))
        except Exception as e:
            errs += 1; last_err = f"{type(e).__name__}: {str(e)[:140]}"


# ----------------------------------------------------------- self-test ----
def synth(seconds: float = 30.0, hz: float = 10.0, T: int = 16) -> None:
    """Send synthetic frames (no GPU) so the collector/dashboard can be built."""
    host, port = _UDP.rsplit(":", 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rng = np.random.default_rng(0)
    hot = rng.choice(N_EXPERTS, 12, replace=False)
    t0 = time.time(); step = 0
    while time.time() - t0 < seconds:
        step += 1
        hist = np.zeros((N_MOE, N_EXPERTS), dtype=np.uint16)
        for l in range(N_MOE):
            ids = rng.choice(N_EXPERTS, T * TOPK, p=_p(hot, N_EXPERTS, rng))
            np.add.at(hist[l], ids, 1)
        ah = np.zeros((N_MLA, ATTN_BINS), dtype=np.uint8)
        for l in range(N_MLA):
            c = int(ATTN_BINS * (0.15 + 0.7 * ((step * 0.01 + l * 0.08) % 1.0)))
            x = np.arange(ATTN_BINS); ah[l] = np.clip(255 * np.exp(-((x - c) / 12.0) ** 2) + 40 * np.exp(-((x - ATTN_BINS + 3) / 6.0) ** 2), 0, 255)
        rib = (20 + 8 * np.sin(np.arange(N_ALL) / 5.0 + step * 0.2) + rng.normal(0, 0.5, N_ALL)).tolist()
        frame = {"kind": "act", "ts": time.time(), "step": step, "T": T, "n_moe": N_MOE, "n_mla": N_MLA, "n_all": N_ALL,
                 "routing": base64.b64encode(hist.tobytes()).decode(), "attn": base64.b64encode(ah.tobytes()).decode(),
                 "ribbon": [round(float(v), 3) for v in rib]}
        sock.sendto(json.dumps(frame).encode(), (host, int(port)))
        T3, TA = min(T, MAX_T3), min(T, MAX_TA)
        r3 = np.zeros((N_MOE, T3, TOPK), dtype=np.int16)
        reqs = [["req-aaaa1111", 0, T3 // 2], ["req-bbbb2222", T3 // 2, T3 - T3 // 2]]
        hot2 = [hot, (hot + 97) % N_EXPERTS]
        for t in range(T3):
            base = rng.choice(N_EXPERTS, TOPK, replace=False, p=_p(hot2[0 if t < T3 // 2 else 1], N_EXPERTS, rng))
            for l in range(N_MOE):
                drift = rng.integers(-2, 3, TOPK) if rng.random() < 0.6 else 0
                r3[l, t] = np.clip(base + drift + (l % 3), 0, N_EXPERTS - 1)
        a3 = np.zeros((N_MLA, TA, ATTN3_BINS), dtype=np.uint8); x = np.arange(ATTN3_BINS)
        for l in range(N_MLA):
            for t in range(TA):
                c = ATTN3_BINS * (0.2 + 0.6 * ((step * 0.01 + t * 0.05 + l * 0.07) % 1.0))
                a3[l, t] = np.clip(255 * np.exp(-((x - c) / 4.0) ** 2) + 60 * np.exp(-((x - ATTN3_BINS + 2) / 3.0) ** 2), 0, 255)
        ang = step * 0.05 + np.arange(T3) * 0.4
        h3 = np.stack([np.cos(ang) * (1 + 0.1 * np.arange(T3)), np.sin(ang) * (1 + 0.1 * np.arange(T3)), 0.3 * np.sin(2 * ang)], 1).astype(np.float16)
        f3 = {"kind": "act3d", "ts": time.time(), "step": step, "T3": T3, "TA": TA, "n_moe": N_MOE, "n_mla": N_MLA, "bins": ATTN3_BINS, "reqs": reqs,
              "routing3d": base64.b64encode(r3.tobytes()).decode(), "attn3d": base64.b64encode(a3.tobytes()).decode(), "h3": base64.b64encode(h3.tobytes()).decode()}
        sock.sendto(json.dumps(f3).encode(), (host, int(port)))
        time.sleep(1.0 / hz)


def _p(hot, n, rng):
    p = np.full(n, 1.0); p[hot] = 14.0; return p / p.sum()


if __name__ == "__main__":
    import sys
    synth(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
