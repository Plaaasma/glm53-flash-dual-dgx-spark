import os, sys, time, json, socket, torch
os.environ["GLM53_VIZ"] = "1"; os.environ["GLM53_VIZ_UDP"] = "127.0.0.1:9198"; os.environ["GLM53_VIZ_HZ"] = "30"
sys.path.insert(0, "/r"); import glm53_viz_runtime as v
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rx.bind(("127.0.0.1", 9198)); rx.settimeout(0.5)
dev = torch.device("cuda:0"); T = 8
ids = torch.randint(0, 288, (T, 9), dtype=torch.int32, device=dev); tk = torch.randint(0, 3000, (T, 2048), dtype=torch.int32, device=dev); hid = torch.randn(T, 4096, device=dev, dtype=torch.bfloat16)
import numpy as np
class FB: pass
fb = FB(); fb.req_ids = ['chatcmpl-aaaa1111', 'chatcmpl-bbbb2222']; fb.num_reqs = 2; fb.query_start_loc_np = np.array([0, 5, 8])
def hooks():
    v.set_batch(fb); v.begin_step(T, dev)
    for l in range(43): v.record_routing(f"moe{l}", ids)
    for l in range(12): v.record_attn(f"mla{l}", tk)
    for l in range(46): v.record_ribbon(l, hid)
    v.record_hidden3d(hid)
    v.end_step()
with torch.inference_mode():
    hooks(); torch.cuda.synchronize()          # eager: arms buffers + starts publisher thread
    time.sleep(2.5)
    early = 0
    while True:
        try: rx.recv(65535); early += 1
        except socket.timeout: break
    print("frames before any capture (must be status only):", early)
    # 6 back-to-back captures (global error mode, like torch's default) with hooks inside, publisher live
    graphs = []
    for i in range(6):
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            hooks(); torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s, capture_error_mode="global"):
                hooks()
        graphs.append(g); time.sleep(0.2)
    torch.cuda.synchronize(); print("6 captures OK with publisher live")
    for _ in range(40):
        for g in graphs: g.replay()
        time.sleep(0.05)
    torch.cuda.synchronize()
kinds = {}; sizes = {}
t0 = time.time()
while time.time() - t0 < 12:
    try: raw = rx.recv(65535); f = json.loads(raw.decode()); kinds[f["kind"]] = f; sizes[f["kind"]] = max(sizes.get(f["kind"], 0), len(raw))
    except socket.timeout: pass
    with torch.inference_mode():
        for g in graphs: g.replay()
st = kinds.get("viz_status", {}); act = kinds.get("act")
print("status:", {k: st.get(k) for k in ("ready","dead","errors","last_error","last_step")})
print("act frame after captures:", bool(act), act and f"step {act['step']} T {act['T']}")
a3 = kinds.get("act3d"); print("act3d:", bool(a3), a3 and f"T3 {a3['T3']} TA {a3['TA']} n_moe {a3['n_moe']} reqs {a3['reqs']}", "| datagram bytes:", sizes)
assert a3 and a3["reqs"] == [["aaaa1111", 0, 5], ["bbbb2222", 5, 3]], "reqs missing"
assert act and a3 and not st.get("errors") and max(sizes.values()) < 65000, "3d frame problem"
print("PASS")
