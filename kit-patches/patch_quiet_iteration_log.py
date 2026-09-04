#!/usr/bin/env python3
"""Silence per-iteration INFO logging while keeping per-step scheduler stats.

`--enable-logging-iteration-details` is the only way this vLLM build
attaches SchedulerStats to steps that produce no output tokens (pure
prefill): `_attach_iteration_details` creates an EngineCoreOutputs for the
step, so kv usage / running / iteration-token metrics update live and the
dashboard can show prefill rate in real time. The same flag makes
LoggingStatLogger._log_iteration_details emit one INFO line per engine
step (5-10/s). This patch turns that method into a no-op; Prometheus
recording is untouched. Idempotent; fails closed on anchor drift.
"""
import os, sys
from pathlib import Path
MARK = "# [glm53-quiet-iteration-log]"
P = Path(os.environ.get("GLM53_LOGGERS_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/metrics/loggers.py"))
OLD = """        details = scheduler_stats.iteration_details
        if details is None:
            return

        encoder_msg = ""
"""
NEW = """        details = scheduler_stats.iteration_details
        if details is None:
            return
        return  """ + MARK + """ per-step INFO line suppressed; stats still recorded

        encoder_msg = ""
"""
def main() -> int:
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if text.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {text.count(OLD)} — refusing to patch")
    P.write_text(text.replace(OLD, NEW, 1)); print(f"patched {P.name} (per-iteration INFO log silenced)"); return 0
if __name__ == "__main__":
    sys.exit(main())
