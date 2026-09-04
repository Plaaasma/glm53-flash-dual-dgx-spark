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
            self.d_one = torch.ones(1, dtype=torch.int32, device=device)
            pin = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt, pin_memory=True)
            self.h_routing = pin(N_MOE, MAX_T, TOPK)
            self.h_attn = pin(N_MLA, MAX_T, TOPK_ATTN)
            self.h_ribbon = pin(N_ALL, dt=torch.float32)
            self.h_meta = pin(4)
            for nm in ("h_routing", "h_attn", "h_ribbon", "h_meta"):
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
        if now - state.capture_seen < 3.0:
            continue                                # a CUDA graph capture is (or just was) in progress: any
        try:                                        # sync/copy from this thread would invalidate it (global mode)
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
            state.pub_stream.synchronize()
            frame = _fold(state, T, n_moe, n_mla)
          frame["step"] = step
          sock.sendto(json.dumps(frame).encode(), (host, int(port)))
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
        time.sleep(1.0 / hz)


def _p(hot, n, rng):
    p = np.full(n, 1.0); p[hot] = 14.0; return p / p.sum()


if __name__ == "__main__":
    import sys
    synth(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
