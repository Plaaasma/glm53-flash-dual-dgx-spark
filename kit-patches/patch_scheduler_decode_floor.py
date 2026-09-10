#!/usr/bin/env python3
"""Keep decode from sharing an engine step with a long sparse-MLA prefill.

Issue #6: max_num_batched_tokens=1024 is the whole engine step. A decode
lane needs ~8 tokens (1 + DFlash2 k=7); the leftover ~1016 go to a peer
FLASHINFER_MLA_SPARSE_SM120 prefill chunk (~1.5 s). Decode still runs, but
at ~5 tok/s instead of ~50.

A 128-token mixed cap is not enough on 80k KV: the indexer has a large
per-step cost, so mixed decode stays ~10 tok/s. Default is therefore to
skip scheduling that prefill this step (it resumes when no peer is
decoding). Solo prefill is unchanged (1024).

GLM53_MIXED_PREFILL_CHUNK:
  skip / -1  — do not mix prefill with decode (default)
  N>0        — cap mixed prefill chunks to N tokens (128 still stalls ~10 tok/s)
  ladder     — cap by number of decoding peers, GLM53_MIXED_PREFILL_LADDER="1:1024,2:512,4:256,*:128"
  0 / off    — disable

Fail closed if the vLLM scheduler anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-decode-floor]"

IMPORT_OLD = "import itertools\nimport time\n"
IMPORT_NEW = "import itertools\nimport os\nimport time\n"

HELPER = '''
def _glm53_mixed_prefill_policy(running, current):
    """Mixed-step prefill policy when a peer in `running` is decoding.

    None = no extra policy. 0 = skip this prefill this step. N>0 = cap.
    """
    raw = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip").strip().lower()
    if raw in ("0", "off", "no"):
        return None
    if raw == "interleave":
        # Periodic admission: while a peer decodes, admit prefill chunks only
        # inside a WINDOW every PERIOD seconds. Between windows: decode-only
        # steps at full speed. During the admitted mixed step decode rides the
        # slow (~1.5 s) sparse-MLA prefill step, so per period decode keeps
        # roughly (PERIOD - 1.5)/PERIOD of solo throughput while prefill makes
        # ~one chunk per period instead of starving outright (issue #6 default
        # `skip` starves prefill for the whole peer decode -- observed as
        # minutes-long TTFT and client retry storms, 2026-09-01).
        # Solo prefill and solo decode are unchanged: no decoding peer -> None.
        import time as _t
        cur_id = getattr(current, "request_id", None)
        decoding = False
        for r in running:
            if r is current or getattr(r, "request_id", None) == cur_id:
                continue
            if r.num_computed_tokens >= r.num_prompt_tokens:
                decoding = True
                break
        if not decoding:
            return None
        period = float(os.environ.get("GLM53_MIXED_INTERLEAVE_PERIOD", "3.5"))
        window = float(os.environ.get("GLM53_MIXED_INTERLEAVE_WINDOW", "0.3"))
        return None if (_t.monotonic() % period) < window else 0
    cur_id = getattr(current, "request_id", None)
    n_decoding = 0
    for r in running:
        if r is current or getattr(r, "request_id", None) == cur_id:
            continue
        if r.num_computed_tokens >= r.num_prompt_tokens:
            n_decoding += 1
    # Context-aware chunk cap (2026-09-10): the sparse-MLA indexer's per-step scratch scales with
    # chunk x context (~1 GB at 2048 x 588K), which is the whole memory margin of the head node.
    # Keep chunk * context <= GLM53_PREFILL_CHUNK_CTX_BUDGET (token^2, default 3e8 ~ 300 MB of fp8
    # logits) once the context is deep; per-token MoE cost is flat above ~256-token chunks so this
    # costs little throughput. Applies with or without decoding peers.
    ctx_cap = None
    try:
        budget = int(float(os.environ.get("GLM53_PREFILL_CHUNK_CTX_BUDGET", "300000000")))
    except ValueError:
        budget = 300000000
    if budget > 0:
        ctx = int(getattr(current, "num_computed_tokens", 0) or 0)
        if ctx > 0:
            c = budget // ctx
            if c < 2048:
                ctx_cap = max(256, (c // 128) * 128)
    if n_decoding == 0:
        return ctx_cap
    if raw == "ladder":
        # Adaptive cap by the number of decoding peers (2026-09-10, max prefill speed): every mixed step
        # reads the whole expert set once whatever the chunk, so bigger chunks are almost free for prefill
        # throughput; the cost is longer inter-token gaps on the decoding streams. GLM53_MIXED_PREFILL_LADDER
        # = "1:1024,2:512,4:256,*:128" -> cap 1024 with <=1 decoding peer, 512 with <=2, 256 with <=4, else 128.
        ladder = os.environ.get("GLM53_MIXED_PREFILL_LADDER", "1:1024,2:512,4:256,*:128")
        default_cap = 128
        for item in ladder.split(","):
            k, _, v = item.strip().partition(":")
            try:
                cap_v = int(v)
            except ValueError:
                continue
            if k.strip() == "*":
                default_cap = cap_v
            else:
                try:
                    if n_decoding <= int(k):
                        return min(cap_v, ctx_cap) if ctx_cap else cap_v
                except ValueError:
                    continue
        return min(default_cap, ctx_cap) if ctx_cap else default_cap
    if raw in ("skip", "-1"):
        return 0
    try:
        cap = int(raw)
    except ValueError:
        return 0
    if cap <= 0:
        return ctx_cap
    return min(cap, ctx_cap) if ctx_cap else cap


'''

RUNNING_OLD = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
"""

RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""

WAITING_OLD = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
"""

WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
                    if mixed_cap is not None and num_computed_tokens < request.num_prompt_tokens:
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""


DBG_CAP_OLD = """            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)
"""

DBG_CAP_NEW = """            _g53_pre = num_new_tokens  # [glm53-decode-floor] dbg
            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)
            _g53_post_cap = num_new_tokens  # [glm53-decode-floor] dbg
            if request.num_computed_tokens < request.num_prompt_tokens:  # [glm53-decode-floor] dbg
                import time as _g53t2
                if _g53t2.monotonic() - getattr(self, "_g53_dbg2_ts", 0.0) > 1.0:
                    self._g53_dbg2_ts = _g53t2.monotonic()
                    logger.info(
                        "[glm53-dbg] prefill pass: req=%s pre=%s cap=%s post=%s computed=%d"
                        " prompt=%d ntws=%d placeholders=%d budget=%s",
                        request.request_id, _g53_pre, mixed_cap, _g53_post_cap,
                        request.num_computed_tokens, request.num_prompt_tokens,
                        request.num_tokens_with_spec, request.num_output_placeholders,
                        token_budget,
                    )
"""

ZERO_OLD = """                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue
"""

ZERO_NEW = """                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                if request.num_computed_tokens < request.num_prompt_tokens:  # [glm53-decode-floor] dbg
                    import time as _g53t
                    if _g53t.monotonic() - getattr(self, "_g53_dbg_ts", 0.0) > 1.0:
                        self._g53_dbg_ts = _g53t.monotonic()
                        logger.info(
                            "[glm53-dbg] prefill got 0 tokens: req=%s pre_min=%s post_cap=%s"
                            " mixed_cap=%s computed=%d prompt=%d token_budget=%s",
                            request.request_id, _g53_pre, _g53_post_cap, mixed_cap,
                            request.num_computed_tokens, request.num_prompt_tokens,
                            token_budget,
                        )
                req_index += 1
                continue
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK not in text:
        if "import os\n" not in text.split("import time\n", 1)[0]:
            text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")
        if "def _glm53_mixed_prefill_policy(" not in text:
            needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
            if text.count(needle) != 1:
                raise SystemExit(f"{P}: helper insert point not unique")
            text = text.replace(needle, HELPER + needle, 1)
        text = replace_once(text, RUNNING_OLD, RUNNING_NEW, "running-prefill")
        text = replace_once(text, WAITING_OLD, WAITING_NEW, "waiting-prefill")
    else:
        print(f"{P.name}: {MARK} present — core seams kept")
    # Upgrade a baked-in older policy helper in place (the image ships v1; "ladder" arrived 2026-09-10).
    if "def _glm53_mixed_prefill_policy(" in text and ("\"ladder\"" not in text or "GLM53_PREFILL_CHUNK_CTX_BUDGET" not in text):
        import re as _re
        start = text.index("def _glm53_mixed_prefill_policy(")
        m = _re.compile(r"\n(?=(def |class |from |import |@))").search(text, start + 1)
        if m is None:
            raise SystemExit(f"{P}: cannot find the end of the old policy helper")
        new_fn = HELPER.strip("\n") + "\n"
        text = text[:start] + new_fn + text[m.start() + 1:]
        print(f"{P.name}: policy helper upgraded in place (ladder mode available)")
    if "prefill pass" not in text:
        if "[glm53-decode-floor] dbg" in text:
            # v2 dbg seams present: upgrade the cap-capture block in place
            v2 = DBG_CAP_NEW.split("            if request.num_computed_tokens < request.num_prompt_tokens:  # [glm53-decode-floor] dbg")[0]
            text = replace_once(text, v2, DBG_CAP_NEW, "dbg-cap-upgrade")
        else:
            text = replace_once(text, DBG_CAP_OLD, DBG_CAP_NEW, "dbg-cap-capture")
            text = replace_once(text, ZERO_OLD, ZERO_NEW, "zero-sched-dbg")
    else:
        print(f"{P.name}: dbg seams current")
    P.write_text(text)
    cap = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip")
    print(f"patched {P.name} (mixed prefill policy={cap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
