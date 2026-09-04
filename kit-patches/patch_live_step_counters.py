#!/usr/bin/env python3
"""Per-step token counters that update even on output-less (pure prefill) steps.

PrometheusStatLogger.record() returns early when `iteration_stats is None`,
which is every step that produced no output tokens -- i.e. all of a long
prefill. `scheduler_stats.iteration_details` (present on every step when
--enable-logging-iteration-details is on) carries num_ctx_tokens /
num_generation_tokens, so count those into two counters:
  vllm:glm53_ctx_tokens_total   context (prefill) tokens scheduled
  vllm:glm53_gen_tokens_total   generation tokens scheduled
The dashboard collector uses the first for a real-time prefill rate.
Idempotent; fails closed on anchor drift.
"""
import os, sys
from pathlib import Path
MARK = "# [glm53-live-step-counters]"
P = Path(os.environ.get("GLM53_LOGGERS_PY", "/usr/local/lib/python3.12/dist-packages/vllm/v1/metrics/loggers.py"))
CREATE_OLD = """        histogram_iteration_tokens = self._histogram_cls(
            name="vllm:iteration_tokens_total",
"""
CREATE_NEW = """        counter_g53_ctx = self._counter_cls(  """ + MARK + """
            name="vllm:glm53_ctx_tokens",
            documentation="Context (prefill) tokens scheduled per engine step, counted every step.",
            labelnames=labelnames,
        )
        self.counter_g53_ctx = create_metric_per_engine(counter_g53_ctx, per_engine_labelvalues)
        counter_g53_gen = self._counter_cls(
            name="vllm:glm53_gen_tokens",
            documentation="Generation tokens scheduled per engine step, counted every step.",
            labelnames=labelnames,
        )
        self.counter_g53_gen = create_metric_per_engine(counter_g53_gen, per_engine_labelvalues)
        histogram_iteration_tokens = self._histogram_cls(
            name="vllm:iteration_tokens_total",
"""
REC_OLD = """            self.gauge_kv_cache_usage[engine_idx].set(scheduler_stats.kv_cache_usage)
"""
REC_NEW = """            self.gauge_kv_cache_usage[engine_idx].set(scheduler_stats.kv_cache_usage)
            _d = scheduler_stats.iteration_details  """ + MARK + """
            if _d is not None and not getattr(_d, "is_dummy", False):
                self.counter_g53_ctx[engine_idx].inc(int(_d.num_ctx_tokens))
                self.counter_g53_gen[engine_idx].inc(int(_d.num_generation_tokens))
"""
def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    for old in (CREATE_OLD, REC_OLD):
        if t.count(old) != 1:
            raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(old)} — refusing to patch")
    t = t.replace(CREATE_OLD, CREATE_NEW, 1).replace(REC_OLD, REC_NEW, 1)
    P.write_text(t); print(f"patched {P.name} (live per-step token counters)"); return 0
if __name__ == "__main__":
    sys.exit(main())
