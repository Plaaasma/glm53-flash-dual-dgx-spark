"""v2 unit test: cap correctness + prompt-prefix stability across a growing agent session."""
import os, sys, types, subprocess, copy, json
r = subprocess.run([sys.executable, "/w/patch_mm_cap.py"], capture_output=True, text=True); print(r.stdout.strip(), r.stderr.strip()[-300:]); assert r.returncode == 0
import importlib
mod = importlib.import_module("vllm.entrypoints.openai.chat_completion.serving")
cap = mod._glm53_cap_mm_parts
mc = types.SimpleNamespace(multimodal_config=types.SimpleNamespace(get_limit_per_prompt=lambda m: {"image": 16, "video": 1}.get(m, 999)))
def img(i): return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,IMG{i}"}}
def txt(s): return {"type": "text", "text": s}
def n_img(msgs): return sum(1 for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url")
def capped(M):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(M):
        msgs.append({"role": "user", "content": [txt(f"turn {i}"), img(i), txt(f"after {i}")]}); msgs.append({"role": "assistant", "content": f"reply {i}"})
    cap(types.SimpleNamespace(messages=msgs), mc); return msgs
def common_prefix_msgs(a, b):
    n = 0
    for x, y in zip(a, b):
        if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True): break
        n += 1
    return n
jumps = []; kept = {}
prev = capped(1)
for M in range(2, 41):
    cur = capped(M); kept[M] = n_img(cur)
    # the conversation grew by 2 messages; everything before them must be byte-identical unless a batch jump happened
    stable = common_prefix_msgs(prev, cur) >= len(prev)
    if not stable: jumps.append(M)
    prev = cur
print("kept images by M:", {m: kept[m] for m in (16, 17, 20, 24, 25, 32, 33, 40)})
print("prefix-changing turns (batch jumps):", jumps)
assert all(9 <= kept[m] <= 16 for m in range(17, 41)), "kept count out of [9,16]"
assert jumps == [17, 25, 33], f"expected jumps only at 17, 25, 33; got {jumps}"
ph = [p["text"] for m in capped(30) if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "text" and p["text"].startswith("[")]
assert len(set(ph)) == 1 and "carries" not in ph[0]; print("placeholder (constant):", ph[0])
os.environ["GLM53_MM_CAP_BATCH"] = "1"; jumps1 = []
prev = capped(16)
for M in range(17, 25):
    cur = capped(M); jumps1.append(M) if common_prefix_msgs(prev, cur) < len(prev) else None; prev = cur
print("with GLM53_MM_CAP_BATCH=1 (plain sliding window) prefix changes at:", jumps1); assert jumps1 == list(range(17, 25))
print("ALL OK")
