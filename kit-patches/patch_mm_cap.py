#!/usr/bin/env python3
"""Keep long multimodal sessions under `--limit-mm-per-prompt` instead of rejecting them.

vLLM validates the number of image/video items per prompt during preprocessing and answers
`400 At most N image(s) may be provided in one prompt` once a long agent session has accumulated
more screenshots than the limit. This patch runs before rendering (OpenAI chat and the Anthropic
/v1/messages shim both pass through `render_chat_request`): when a modality exceeds its limit it
keeps the NEWEST N items and replaces the older ones with one short text placeholder per message,
so the text context is untouched and the request proceeds.

Prefix-cache friendliness (2026-09-08, second version): a naive "keep the newest N" changes the prompt
every time an image is added (the window slides, and a count in the placeholder text changes), which moved the
first difference ~85K tokens back in a 334K-token agent session and forced a full re-prefill on every
read_image call. Now the dropped set only grows in batches of GLM53_MM_CAP_BATCH (default 8) images: with a
limit of 16, a session carries between 9 and 16 images and the prompt prefix is stable for 8 consecutive image
additions; the placeholder text is constant.

Cold guard (v3): the memory cost of a request is dominated by images the server has never processed
(preprocessing, IPC copies, encoder work); images seen before are cheap (processor cache + KV prefix cache).
So the pass keeps every image it has accepted before (a bounded LRU of content hashes) and admits at most
GLM53_MM_COLD_MAX (default 24) never-seen images per request, newest first; the rest get the constant
placeholder and, since they never enter the seen-set, stay dropped on later turns. A session therefore grows
append-only after its first (cold) turn, up to the limit, without ever exceeding the cold budget.

Runtime knobs: GLM53_MM_CAP=0 disables the pass (request then fails the way upstream does);
GLM53_MM_CAP_BATCH=16 batch size for over-limit dropping; GLM53_MM_COLD_MAX=24 never-seen images per request
(0 = no cold guard).
Fail closed if the vLLM seams drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_CHAT_SERVING_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py",
    )
)
MARK = "# [glm53-mm-cap]"
MARK_V2 = "_glm53_part_key"

HELPER = '''
import collections as _collections  # [glm53-mm-cap]
import hashlib as _hashlib

_GLM53_MM_SEEN: "collections.OrderedDict[str, None]" = _collections.OrderedDict()     # accepted image keys (LRU, bounded)
_GLM53_MM_DROPPED: "collections.OrderedDict[str, None]" = _collections.OrderedDict()  # rejected-when-cold keys: stay dropped
_GLM53_MM_SEEN_MAX = 50000


def _glm53_part_key(part) -> str | None:
    """Cheap content key for an image/video part without decoding it: hash of the URL/data-URL string."""
    try:
        if isinstance(part, dict):
            for k in ("image_url", "video_url", "input_image", "image", "video"):
                v = part.get(k)
                if isinstance(v, dict):
                    v = v.get("url") or v.get("data") or v.get("image_url")
                if isinstance(v, str) and v:
                    return _hashlib.blake2b(v.encode("utf-8", "ignore"), digest_size=16).hexdigest()
    except Exception:  # noqa: BLE001
        return None
    return None


def _glm53_lru_add(s, key: str) -> None:
    if key in s:
        s.move_to_end(key)
    else:
        s[key] = None
        while len(s) > _GLM53_MM_SEEN_MAX:
            s.popitem(last=False)


_GLM53_MM_PART_TYPES = {  # [glm53-mm-cap]
    "image": ("image_url", "input_image", "image", "image_embeds", "image_pil"),
    "video": ("video_url", "input_video", "video", "video_embeds"),
}


def _glm53_part_type(part):
    if isinstance(part, dict):
        return part.get("type")
    return getattr(part, "type", None)


def _glm53_mm_drop_count(n_items: int, limit: int, batch: int) -> int:
    """How many of the oldest items to drop: 0 within the limit, else the excess rounded UP to a multiple of
    `batch`, so the dropped set (and thus the prompt prefix) only changes every `batch` additions."""
    if n_items <= limit:
        return 0
    batch = max(1, batch)
    drop = -(-(n_items - limit) // batch) * batch
    return min(drop, n_items - 1)


def _glm53_cap_mm_parts(request, model_config) -> None:
    """Keep the newest `--limit-mm-per-prompt` items per modality (batched, see module doc); older ones
    become a constant text note so the prompt prefix stays cacheable."""
    import os as _os

    if _os.environ.get("GLM53_MM_CAP", "1") != "1":
        return
    try:
        batch = int(_os.environ.get("GLM53_MM_CAP_BATCH", "16"))
    except ValueError:
        batch = 16
    try:
        cold_max = int(_os.environ.get("GLM53_MM_COLD_MAX", "24"))
    except ValueError:
        cold_max = 24
    mm_config = getattr(model_config, "multimodal_config", None)
    messages = getattr(request, "messages", None)
    if mm_config is None or not isinstance(messages, list):
        return
    for modality, types in _GLM53_MM_PART_TYPES.items():
        try:
            limit = int(mm_config.get_limit_per_prompt(modality))
        except Exception:  # noqa: BLE001
            continue
        if limit <= 0:
            continue
        locs: list[tuple[int, int]] = []
        for mi, msg in enumerate(messages):
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if not isinstance(content, list):
                continue
            for pi, part in enumerate(content):
                if _glm53_part_type(part) in types:
                    locs.append((mi, pi))
        n_items = len(locs)
        n_drop = _glm53_mm_drop_count(n_items, limit, batch)
        drop = set(locs[:n_drop])
        n_cold_dropped = 0
        if modality == "image" and cold_max > 0:
            keys = {}
            for loc in locs[n_drop:]:
                mi, pi = loc
                msg = messages[mi]
                content = msg["content"] if isinstance(msg, dict) else msg.content
                keys[loc] = _glm53_part_key(content[pi])
            # once rejected when cold, an image stays rejected (monotone -> the prompt prefix never changes)
            for loc in locs[n_drop:]:
                if keys[loc] is not None and keys[loc] in _GLM53_MM_DROPPED:
                    drop.add(loc)
                    n_cold_dropped += 1
            unseen = [
                loc for loc in locs[n_drop:]
                if loc not in drop and (keys[loc] is None or keys[loc] not in _GLM53_MM_SEEN)
            ]
            if len(unseen) > cold_max:
                for loc in unseen[: len(unseen) - cold_max]:
                    drop.add(loc)
                    n_cold_dropped += 1
                    if keys[loc] is not None:
                        _glm53_lru_add(_GLM53_MM_DROPPED, keys[loc])
            for loc in locs[n_drop:]:
                if loc not in drop and keys[loc] is not None:
                    _glm53_lru_add(_GLM53_MM_SEEN, keys[loc])
        if not drop:
            continue
        by_msg: dict[int, set[int]] = {}
        for mi, pi in drop:
            by_msg.setdefault(mi, set()).add(pi)
        for mi, drop_idx in by_msg.items():
            msg = messages[mi]
            content = msg["content"] if isinstance(msg, dict) else msg.content
            new_content = []
            placed = False
            for pi, part in enumerate(content):
                if pi in drop_idx:
                    if not placed:
                        new_content.append({
                            "type": "text",
                            "text": f"[earlier {modality}(s) omitted by the server: per-prompt {modality} limit]",
                        })
                        placed = True
                    continue
                new_content.append(part)
            if isinstance(msg, dict):
                msg["content"] = new_content
            else:
                msg.content = new_content
        logger.warning(
            "[glm53-mm-cap] %d %s parts (limit %d, batch %d, cold max %d): kept %d, replaced %d with placeholders "
            "(%d over the limit, %d never-seen beyond the cold budget) across %d message(s)",
            n_items, modality, limit, batch, cold_max, n_items - len(drop), len(drop), n_drop, n_cold_dropped, len(by_msg),
        )


'''

CLASS_ANCHOR = "class OpenAIServingChat(GenerateBaseServing):\n"
SEAM_OLD = "        return await self.online_renderer.render_chat(request)\n"
SEAM_NEW = (
    "        _glm53_cap_mm_parts(request, self.model_config)  # [glm53-mm-cap] newest N images, text untouched\n"
    "        return await self.online_renderer.render_chat(request)\n"
)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        if MARK_V2 in text:
            print(f"{P.name}: {MARK} v2 already present — skipping")
            return 0
        raise SystemExit(f"{P}: carries the v1 [glm53-mm-cap] patch; rebuild from a clean image (no in-place upgrade)")
    for label, needle in (("class anchor", CLASS_ANCHOR), ("render seam", SEAM_OLD), ("logger", "logger = init_logger(__name__)\n")):
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: expected exactly one {label}, found {text.count(needle)} (upstream drift)")
    text = text.replace(CLASS_ANCHOR, HELPER + CLASS_ANCHOR, 1)
    text = text.replace(SEAM_OLD, SEAM_NEW, 1)
    P.write_text(text)
    print(f"patched {P.name}: multimodal items over --limit-mm-per-prompt are truncated (newest kept) instead of rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
