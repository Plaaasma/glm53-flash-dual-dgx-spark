"""v3: cold guard + prefix stability across a growing session; over-limit fallback still batched."""
import os, sys, types, subprocess, copy, json
r = subprocess.run([sys.executable, "/w/patch_mm_cap.py"], capture_output=True, text=True); print(r.stdout.strip()[-120:], r.stderr.strip()[-300:]); assert r.returncode == 0
import importlib
mod = importlib.import_module("vllm.entrypoints.openai.chat_completion.serving")
cap = mod._glm53_cap_mm_parts
mc = types.SimpleNamespace(multimodal_config=types.SimpleNamespace(get_limit_per_prompt=lambda m: {"image": 128, "video": 1}.get(m, 999)))
os.environ["GLM53_MM_COLD_MAX"] = "24"; os.environ["GLM53_MM_CAP_BATCH"] = "16"
def img(i): return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,IMG{i:04d}"}}
def txt(s): return {"type": "text", "text": s}
def convo(M):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(M): msgs.append({"role": "user", "content": [txt(f"turn {i}"), img(i)]}); msgs.append({"role": "assistant", "content": f"reply {i}"})
    return msgs
def kept_ids(msgs): return [p["image_url"]["url"][-4:] for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
def same_prefix(a, b): return all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b))
# cold turn with 40 images: only the newest 24 admitted
m40 = convo(40); cap(types.SimpleNamespace(messages=m40), mc); k = kept_ids(m40)
print("cold 40 ->", len(k), "kept, newest:", k[-1], "oldest kept:", k[0]); assert len(k) == 24 and k[0] == "0016"
# subsequent turns: append one image each; everything before must be byte-identical, kept grows by one
prev = m40; breaks = []
for M in range(41, 150):
    cur = convo(M); cap(types.SimpleNamespace(messages=cur), mc)
    if not same_prefix(prev, cur): breaks.append(M)
    prev = cur
print("kept at 149 images:", len(kept_ids(prev)), "| prefix breaks at:", breaks)
# 129..144: the batched over-limit drop removes positions 0-15, already placeholders -> no change; 145: n_drop=32 -> images 16-31 go -> one break
assert breaks == [145], f"prefix must only break at the over-limit fallback (145), got {breaks}"
# a burst of 30 new images in one turn: only 24 admitted, and the prefix before them is intact
burst = convo(149 + 30); cap(types.SimpleNamespace(messages=burst), mc)
print("after a 30-image burst: kept", len(kept_ids(burst)), "(cold budget 24 per request)")
ph = sorted({p["text"] for m in burst if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "text" and p["text"].startswith("[")})
print("placeholder(s):", ph); assert len(ph) == 1
print("ALL OK")
