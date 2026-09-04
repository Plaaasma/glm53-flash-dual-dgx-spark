#!/usr/bin/env python3
"""Install live activation telemetry hooks (dashboard 'ambient' panels).

Runtime: /opt/glm53/glm53_viz_runtime.py -> vllm/glm53_viz_runtime.py.
Hooks (all no-ops unless env GLM53_VIZ=1; graph-safe async copies only):
  exl3.py apply_exl3_experts      -> record_routing(layer_name, topk_ids)
  flashinfer_mla_sparse_sm120.py  -> record_attn(layer, topk_indices)
  glm5next/nvidia/model.py        -> begin_step / record_ribbon / end_step
  v1/core/sched/scheduler.py      -> rate-limited KV/request snapshot (UDP)
Idempotent; every anchor must match exactly once or the patcher refuses.
"""
import os, shutil, sys
from pathlib import Path
V = Path("/usr/local/lib/python3.12/dist-packages/vllm")
V = Path(os.environ.get("GLM53_VLLM_DIR", str(V)))
RT_SRC = Path(os.environ.get("GLM53_VIZ_RT_SRC", "/opt/glm53/glm53_viz_runtime.py"))
MARK = "# [glm53-viz]"

def patch(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if MARK in text and label in text:
        print(f"{path.name}: {label} already present — skipping"); return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one anchor for {label}, found {n} — refusing")
    path.write_text(text.replace(old, new, 1)); print(f"patched {path.name}: {label}")

def main() -> int:
    if RT_SRC.exists():
        shutil.copyfile(RT_SRC, V / "glm53_viz_runtime.py"); print("installed glm53_viz_runtime.py")
    # 1) MoE routing
    patch(V / "model_executor/layers/quantization/exl3.py",
          "    tokens, hidden = x.shape[-2], x.shape[-1]\n",
          "    tokens, hidden = x.shape[-2], x.shape[-1]\n"
          "    from vllm import glm53_viz_runtime as _viz  " + MARK + " viz-routing\n"
          "    _viz.record_routing(getattr(layer, 'layer_name', id(layer)), topk_ids)\n",
          "viz-routing")
    # 2) sparse-MLA attention scan
    patch(V / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
          "        topk_indices = self.topk_indices_buffer[:num_actual_toks]\n",
          "        topk_indices = self.topk_indices_buffer[:num_actual_toks]\n"
          "        from vllm import glm53_viz_runtime as _viz  " + MARK + " viz-attn\n"
          "        _viz.record_attn(getattr(layer, 'layer_name', id(self)), topk_indices)\n",
          "viz-attn")
    # 3a) step begin in the base model forward
    m = V / "models/glm5next/nvidia/model.py"
    patch(m,
          "        aux_hidden_states: list[torch.Tensor] = []\n        for idx, layer in enumerate(\n",
          "        from vllm import glm53_viz_runtime as _viz  " + MARK + " viz-step\n"
          "        if self.start_layer == 0:\n"
          "            _viz.begin_step(hidden_states.shape[0], hidden_states.device)\n"
          "        aux_hidden_states: list[torch.Tensor] = []\n        for idx, layer in enumerate(\n",
          "viz-step")
    # 3b) ribbon + end_step in the decoder layer
    patch(m,
          "        if self.layer_idx == self.num_hidden_layers - 1:\n"
          "            x = self.hc_post(x, residual, post, comb)\n"
          "            x = hc_contract(x, self.n)\n"
          "            return x, None, None, None\n\n"
          "        return x, residual, post, comb\n",
          "        from vllm import glm53_viz_runtime as _viz  " + MARK + " viz-ribbon\n"
          "        if self.layer_idx == self.num_hidden_layers - 1:\n"
          "            x = self.hc_post(x, residual, post, comb)\n"
          "            x = hc_contract(x, self.n)\n"
          "            _viz.record_ribbon(self.layer_idx, x)\n"
          "            _viz.end_step()\n"
          "            return x, None, None, None\n\n"
          "        if self.layer_idx < self.num_hidden_layers:\n"
          "            _viz.record_ribbon(self.layer_idx, x)\n"
          "        return x, residual, post, comb\n",
          "viz-ribbon")
    # 4) scheduler snapshot
    s = V / "v1/core/sched/scheduler.py"
    helper = '''
def _glm53_viz_sched(sched, scheduler_output):  ''' + MARK + ''' viz-sched
    """Rate-limited KV-pool / request snapshot for the dashboard (UDP JSON)."""
    import os, time, json, socket
    if os.environ.get("GLM53_VIZ", "0") != "1":
        return
    now = time.monotonic()
    if now - getattr(sched, "_g53_viz_ts", 0.0) < 0.25:
        return
    sched._g53_viz_ts = now
    try:
        sock = getattr(sched, "_g53_viz_sock", None)
        if sock is None:
            sock = sched._g53_viz_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host, port = os.environ.get("GLM53_VIZ_UDP", "127.0.0.1:9103").rsplit(":", 1)
        nb = getattr(sched.cache_config, "num_gpu_blocks", 0) or 0
        reqs = []
        sched_tok = scheduler_output.num_scheduled_tokens if scheduler_output is not None else {}
        for r in list(sched.running)[:64]:
            reqs.append({"id": r.request_id[-8:], "computed": int(r.num_computed_tokens),
                         "prompt": int(r.num_prompt_tokens), "total": int(r.num_tokens),
                         "age": round(time.time() - float(r.arrival_time), 1),
                         "sched": int(sched_tok.get(r.request_id, 0))})
        frame = {"kind": "sched", "ts": time.time(), "usage": float(sched.kv_cache_manager.usage),
                 "pool_tokens": int(nb) * int(getattr(sched, "block_size", 0) or 0),
                 "waiting": len(sched.waiting), "reqs": reqs}
        sock.sendto(json.dumps(frame).encode(), (host, int(port)))
    except Exception:
        pass


'''
    text = s.read_text()
    if "viz-sched" not in text:
        needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{s}: helper insert point not unique")
        text = text.replace(needle, helper + needle, 1)
        old = "        return scheduler_output\n"
        if text.count(old) != 1:
            raise SystemExit(f"{s}: expected one 'return scheduler_output'")
        text = text.replace(old, "        _glm53_viz_sched(self, scheduler_output)  " + MARK + "\n" + old, 1)
        s.write_text(text); print("patched scheduler.py: viz-sched")
    else:
        print("scheduler.py: viz-sched already present — skipping")
    return 0

if __name__ == "__main__":
    sys.exit(main())
