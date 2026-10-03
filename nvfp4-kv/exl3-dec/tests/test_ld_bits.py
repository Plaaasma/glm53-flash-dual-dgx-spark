"""The 128-bit load path (ld=1) gives the 32-bit path's bits for the same (nt, warps, splits), any depth."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import common as C
import glm53_exl3_dec_rt as rt

dev = torch.device("cuda")
layer = C.load_layer(20, list(range(0, 288, 18))[:16], dev)
gen = torch.Generator().manual_seed(3)
pairs = [(((8, 8, 2, 1, 0), (8, 8, 1, 1, 0)), ((8, 8, 2, 2, 1), (8, 8, 1, 2, 1))),
         (((8, 8, 2, 1, 0), (8, 8, 1, 1, 0)), ((8, 8, 2, 1, 1), (8, 8, 1, 1, 1))),
         (((8, 4, 4, 1, 0), (8, 4, 1, 1, 0)), ((8, 4, 4, 4, 1), (8, 4, 1, 4, 1))),
         (((4, 8, 2, 2, 0), (4, 8, 1, 2, 0)), ((4, 8, 2, 4, 1), (4, 8, 1, 4, 1)))]
ok = True
for T in (1, 4, 16, 64):
    x = C.hidden(T, gen).to(dev); ids = C.routing_random(T, list(range(16)), gen).to(dev); w = C.routing_weights(T, gen).to(dev)
    for (ga, da), (gb, db) in pairs:
        rt.prepare_layer(layer, max_rows=64, slots=8, cfg_gu=ga, cfg_d=da); a = rt.decode_moe(x, ids, w, layer, C.LIMIT)
        rt.prepare_layer(layer, max_rows=64, slots=8, cfg_gu=gb, cfg_d=db); b = rt.decode_moe(x, ids, w, layer, C.LIMIT)
        same = torch.equal(a, b); ok &= same
        print(f"T={T:2d} {ga}/{da} vs {gb}/{db}: bitwise {same}")
print("ALL BITWISE:", ok)
