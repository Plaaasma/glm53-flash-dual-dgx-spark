"""Unit test for the [glm53-mm-cap] helper after the patcher has been applied in this container."""
import os, sys, types, subprocess, copy
r = subprocess.run([sys.executable, "/w/patch_mm_cap.py"], capture_output=True, text=True); print(r.stdout.strip(), r.stderr.strip()[-400:])
assert r.returncode == 0
import importlib
mod = importlib.import_module("vllm.entrypoints.openai.chat_completion.serving")
cap = mod._glm53_cap_mm_parts
mc = types.SimpleNamespace(multimodal_config=types.SimpleNamespace(get_limit_per_prompt=lambda m: {"image": 32, "video": 1}.get(m, 999)))
def img(i): return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,IMG{i}"}}
def txt(s): return {"type": "text", "text": s}
def count(msgs, t): return sum(1 for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == t)
def texts(msgs): return [p["text"] for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "text" and not p["text"].startswith("[")] + [m["content"] for m in msgs if isinstance(m.get("content"), str)]

# 1) 40 images spread over 12 messages (some with several), plus text everywhere: keep newest 32, text untouched
msgs = [{"role": "system", "content": "sys prompt"}]
k = 0
for j in range(12):
    parts = [txt(f"user text {j}")]
    for _ in range(3 if j % 3 == 0 else 3 if j < 10 else 5):
        parts.append(img(k)); k += 1
    parts.append(txt(f"trailing text {j}"))
    msgs.append({"role": "user", "content": parts}); msgs.append({"role": "assistant", "content": f"reply {j}"})
before = copy.deepcopy(msgs); n_before = count(msgs, "image_url")
req = types.SimpleNamespace(messages=msgs)
cap(req, mc)
kept = [p["image_url"]["url"] for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
print(f"case1: images {n_before} -> {len(kept)}; newest kept: {kept[0][-6:]}..{kept[-1][-6:]}; text parts intact: {texts(before) == texts(msgs)}; placeholders: {sum(1 for m in msgs if isinstance(m.get('content'), list) for p in m['content'] if p.get('type')=='text' and p['text'].startswith('['))}")
assert len(kept) == 32 and kept == [p["image_url"]["url"] for m in before if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"][-32:]
assert texts(before) == texts(msgs)
ph = [p["text"] for m in msgs if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "text" and p["text"].startswith("[")]
print("  placeholder example:", ph[0])
# 2) under the limit: untouched
msgs2 = [{"role": "user", "content": [txt("a"), img(1), img(2)]}]; b2 = copy.deepcopy(msgs2); cap(types.SimpleNamespace(messages=msgs2), mc); assert msgs2 == b2; print("case2: under limit untouched OK")
# 3) plain-string contents only: untouched; 4) videos: 3 -> 1
msgs3 = [{"role": "user", "content": "hello"}]; cap(types.SimpleNamespace(messages=msgs3), mc); assert msgs3 == [{"role": "user", "content": "hello"}]; print("case3: string content OK")
vid = lambda i: {"type": "video_url", "video_url": {"url": f"v{i}"}}
msgs4 = [{"role": "user", "content": [vid(1), txt("t"), vid(2)]}, {"role": "user", "content": [vid(3)]}]; cap(types.SimpleNamespace(messages=msgs4), mc)
assert count(msgs4, "video_url") == 1 and msgs4[1]["content"][0]["type"] == "video_url"; print("case4: videos 3 -> 1 (newest) OK")
# 5) disabled via env
os.environ["GLM53_MM_CAP"] = "0"; msgs5 = copy.deepcopy(before); cap(types.SimpleNamespace(messages=msgs5), mc); assert msgs5 == before; print("case5: GLM53_MM_CAP=0 leaves request untouched OK")
print("ALL OK")
