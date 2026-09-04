import os, sys, time, json, socket, torch
os.environ["GLM53_VIZ"] = "1"; os.environ["GLM53_VIZ_UDP"] = "127.0.0.1:9198"; os.environ["GLM53_VIZ_HZ"] = "30"
sys.path.insert(0, "/r"); import glm53_viz_runtime as v
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rx.bind(("127.0.0.1", 9198)); rx.settimeout(0.5)
dev = torch.device("cuda:0"); T = 8
ids = torch.randint(0, 288, (T, 9), dtype=torch.int32, device=dev); tk = torch.randint(0, 3000, (T, 2048), dtype=torch.int32, device=dev); hid = torch.randn(T, 4096, device=dev, dtype=torch.bfloat16)
def hooks():
    v.begin_step(T, dev)
    for l in range(43): v.record_routing(f"moe{l}", ids)
    for l in range(12): v.record_attn(f"mla{l}", tk)
    for l in range(46): v.record_ribbon(l, hid)
    v.end_step()
with torch.inference_mode():
    hooks(); torch.cuda.synchronize()          # eager: arms buffers + starts publisher thread
    time.sleep(0.5)                             # publisher now actively polling the GPU
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
kinds = {}
t0 = time.time()
while time.time() - t0 < 5:
    try: f = json.loads(rx.recv(65535).decode()); kinds[f["kind"]] = f
    except socket.timeout: pass
    with torch.inference_mode():
        for g in graphs: g.replay()
st = kinds.get("viz_status", {}); act = kinds.get("act")
print("status:", {k: st.get(k) for k in ("ready","dead","errors","last_error","last_step")})
print("act frame after captures:", bool(act), act and f"step {act['step']} T {act['T']}")
assert act and not st.get("errors"), "publisher not recovering after captures"
print("PASS")
