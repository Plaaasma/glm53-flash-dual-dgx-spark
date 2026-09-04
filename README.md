# GLM-5.3-Flash (Uncensored, EXL3 4bpw) on 2× NVIDIA DGX Spark — 2.6M-token KV pool, NVFP4 KV cache, 16-way serving

A complete, battle-tested recipe for serving **GLM-5.3-Flash** (320B/18B MoE,
vision included) across **two DGX Sparks (GB10, sm_121)** with:

- **2,600,787-token KV pool** (≈ 9.9 concurrent full 262K-context sessions)
- **NVFP4 KV cache** — 288 B/token vs the stock 656 B `fp8_ds_mla` (2.28×
  denser), via a gather-dequant Triton path feeding the stock prebuilt kernel
- **16 concurrency slots**, 900K max context per request
- **67.6 tok/s aggregate** at 12-way concurrency, ~22 tok/s single-stream code
- GSM8K **59/60** under 12-way load, **zero** request errors, greedy outputs
  byte-identical to the fp8-KV baseline
- Host-side guard rails that turn this platform's infamous memory livelocks
  into ordinary process restarts

Builds on [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(their vLLM-fork image + EXL3 kernels). This repo adds the NVFP4 KV pool, an
interleave scheduler policy, the memory guard rails, a monitoring dashboard,
and the exact configuration that survives on real hardware.

Measured speeds: [`bench/llama-benchy.md`](bench/llama-benchy.md).

---

## 0. What you need

| thing | value |
|---|---|
| Hardware | 2× DGX Spark (GB10, 121.63 GiB unified each), joined by the CX7 200GbE QSFP cable |
| Disk | ≥ 200 GB free per node (weights are ~164 GiB, on BOTH nodes) |
| Software | stock DGX OS, Docker with NVIDIA runtime, ssh between nodes as the same user |
| Accounts | a HuggingFace token (weights are public but rate-limits bite) |
| Time | ~2 h: ~1 h weight download, ~20 min image, ~10 min per server boot |

Terminology: the node you run commands on is the **head**; the other is the
**worker**. All commands run on the head unless said otherwise.

## 1. Host preparation — do not skip this

On **each** node:

```bash
./host-setup/setup-node.sh
```

What it does and why (each of these was learned from a real failure):

1. **6 GiB zram swap, priority 100.** These boxes ship with a 16G *file* swap
   that is too slow to absorb allocation spikes. Under unified memory almost
   nothing is reclaimable, so a spike stalls the kernel in reclaim and the
   whole node **livelocks — pingable, ssh dead, power cycle required. The OOM
   killer never fires.** Fast compressed swap gives reclaim a real target.
2. **`vm.swappiness=180`, `vm.watermark_scale_factor=100`.** Early, proactive
   reclaim. Do NOT raise the watermark factor further: at 500 it silently
   walls off ~12 GiB from `MemAvailable` and looks exactly like a memory leak.
3. **A memory watchdog** (`vllm-watchdog.sh`): if `MemAvailable` drops below
   0.75 GiB it SIGTERMs vLLM, waits 3 s, then SIGKILLs. TERM-first matters:
   a raw SIGKILL mid-CUDA leaks ~20 GiB of driver memory that only returns
   after minutes of idle (or a reboot).

**None of this survives a reboot.** Re-run the script after every reboot.

### 1.5 Optional: fast boots (InstantTensor + post-load reclaim)

The stock loader mmaps the safetensors and reads them at ~150–300 MB/s: 259 s
for 164 GB. [InstantTensor](https://pypi.org/project/instanttensor/) streams
them with Direct I/O at 4.7–5.6 GB/s: 35 s. Launch-to-API drops from ~9 min
to ~4.5 min. Two pieces, both nodes:

1. Build the image (source build, no aarch64 wheel; ~2 min) and copy it to
   the worker:

   ```bash
   docker build -f image/Dockerfile.instanttensor \
       --build-arg BASE=glm53-flash-sm121:local-0831 -t glm53-flash-sm121:local-0904-it .
   docker save glm53-flash-sm121:local-0904-it | ssh <worker> docker load
   ```

2. Install the reclaim helper (root-owned) and its sudoers entry:

   ```bash
   sudo install -o root -g root -m 0755 host-setup/glm53-reclaim /usr/local/sbin/glm53-reclaim
   echo "$USER ALL=(root) NOPASSWD: /usr/local/sbin/glm53-reclaim" | sudo tee /etc/sudoers.d/glm53-reclaim
   sudo chmod 0440 /etc/sudoers.d/glm53-reclaim
   ```

Why the helper: on 121 GiB unified memory the 4-minute mmap load *incidentally*
pushes ~3–5 GiB of the serving processes' cold pages to zram through page-cache
pressure, and graph capture relies on that headroom. The 35 s load applies no
pressure, nothing gets swapped, and capture trips the memory watchdog (we lost
six boots to this before measuring swap). The helper writes the container
cgroup's `memory.reclaim`; `start.sh` calls it right after weights load
(`GLM53_POSTLOAD_RECLAIM=4`) and again once the API is up
(`GLM53_POSTREADY_RECLAIM=3`). `env.example` enables all of it; set
`--load-format` back to `auto` and the reclaim knobs to 0 to use the stock loader.

## 2. Get the kit and apply the patches

```bash
git clone <this repo> && cd glm53-spark-recipe
./apply-kit-patches.sh
```

This clones the MiaAI-Lab kit at the tested commit and applies
`kit-patches/exl3-kit.patch`, which adds:

- **NVFP4 KV wiring** — ships `nvfp4-kv/patch_nvfp4_kv.py` +
  `glm53_nvfp4_runtime.py` into both containers at boot (same mount-and-patch
  mechanism the kit itself uses).
- **Mixed-prefill policy `GLM53_MIXED_PREFILL_CHUNK=128`** — the kit's
  default (`skip`) starves every new prompt's prefill while any decode runs
  (minutes of dead TTFT, client retry storms). Our first fix (`interleave`)
  ended starvation but delivered decode in waves. The shipped setting caps
  mixed prefill at 128 tokens/step, which on this stack (flat ~700 tok/s
  prefill at any depth — the kit's old per-step-cost measurement no longer
  applies) bounds every engine step, measured with a streaming client while
  a 35K prefill lands concurrently:

  | policy | median gap | p95 | p99 | max |
  |---|---|---|---|---|
  | interleave (waves) | 179 ms | 3122 ms | 3272 ms | 3312 ms |
  | cap 256 | 550 ms | 721 ms | 1294 ms | 1345 ms |
  | **cap 128 (shipped)** | **426 ms** | **502 ms** | **936 ms** | **1245 ms** |

  A continuous stream at any concurrency, at the cost of prefill running
  ~350 tok/s during overlap (solo prefill unaffected).
- **Extended CUDA-graph capture sizes** so full-batch decode steps at 16
  concurrency stay inside graphs.

Then edit `exl3-kit/.env` (copied from `kit-patches/env.example`):

| key | set to |
|---|---|
| `HF_TOKEN` | your token |
| `HEAD_IP` / `WORKER_IP` | the 169.254.x.x addresses of the CX7 link (`ip -br a`) |
| `HEAD_CX7_IF` etc. | your NIC names — find with `ip -br a` + `ls /sys/class/infiniband`. On our pair BOTH nodes use `enp1s0f1np1` / `rocep1s0f1`; the kit's default wrongly assumes the worker uses f0 |
| `NCCL_IB_GID_INDEX` | the GID index whose entry contains `::ffff:<your fabric IP>` — check `/sys/class/infiniband/<dev>/ports/1/gids/*`. Ours: 3 on both |

Everything else in `env.example` is the tested configuration. The
load-bearing values, so you don't "clean them up":

- `SPEC_METHOD=mtp` — DFlash2 speculative decode is ~5 tok/s faster on code
  but its ~10 GiB footprint makes the 2.6M pool impossible. Flip to `dflash`
  + restart if you want speed over pool (pool must then shrink: lower
  `EXTRA_ARGS` pin to ≤ `--kv-cache-memory 8000000000`).
- `EXTRA_ARGS="--kv-cache-memory 11239802020"` — pins the pool. Without it
  the auto-sizer eats every byte to the watchdog line and boots die.
- `MAX_MODEL_LEN=900000` — do NOT lower it "to save memory": hybrid block-id
  overhead then *doubles* the per-token pool cost (measured 8.1 → 18.5 KB).
- `GPU_MEM_UTIL=0.875`, `CG_ESTIMATE=1`, `GLM53_BOOT_SHAPE_WARMUP=0`,
  `EXL3_MOE_ROW_TILE=1` — each guards a specific boot failure; see
  Troubleshooting.

## 3. Weights

The kit downloads `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` (163.65 GiB)
itself on first `./start.sh`, and rsyncs to the worker. If your internet is
slow, run `./download.sh` first and go do something else.

## 4. Launch

```bash
cd exl3-kit && ./start.sh
```

First boot: image pull (20.9 GB) + ship to worker + ~10 min load. Success
looks like:

```
GPU KV cache size: 2,600,787 tokens
```

and `curl localhost:8888/v1/models` answering with `glm-5.3-flash`
(OpenAI-compatible API, port 8888, vision + tool calling enabled).

Verify the guard rails: `MemAvailable` ≥ 3 GiB on both nodes
(`awk '/MemAvailable/{print $2/1048576}' /proc/meminfo`) and
`/var/tmp/vllm-watchdog.log` shows `armed`.

## 5. Monitoring dashboard (optional, recommended)

`dashboard/` is a zero-dependency live dashboard (vLLM metrics + per-node
GPU/memory/network): run `agent.py` on both nodes (port 9101), `collector.py`
on the head (port 9102), and serve `index.html` (e.g.
`python3 -m http.server 3000`). It understands vLLM's metric quirks on this
stack — e.g. deriving *live* prefill throughput from the per-step iteration
histogram, because `prompt_tokens_total` only updates at request completion.

## 6. What the NVFP4 KV cache actually is

`nvfp4-kv/` stores the sparse-MLA latent as **288 B/token** (256 B packed
e2m1 pairs + 32 B e4m3 per-16 block scales) instead of the stock 656 B
`fp8_ds_mla` layout. The prebuilt FlashInfer sparse-MLA kernel can't read
NVFP4 — so we never ask it to: DSA attention only touches its top-k selected
rows, and a Triton kernel gathers + dequantizes exactly those rows into a
page-shaped fp8 scratch the kernel accepts (4 × fp32 per-128-group scales,
empirically matched to the real cache-write op byte-for-byte).

- decode gather: 0.36 ms/layer; prefill union: ≤ 5.6 ms/layer; write: free
- accuracy: greedy outputs byte-identical to fp8 KV on probe prompts;
  GSM8K parity (59/60 @ c12)
- validated standalone against the real kernel before integration:
  cos 0.995 vs an fp8 pool — pure NVFP4 quantization noise

`nvfp4-kv/PLAN.md` documents the full design, the test methodology, and the
two integration traps (the *executing* backend is
`flashinfer_mla_sparse_sm120.py`, not its identically-shaped sibling; and the
kernel routes decode-vs-paged on page *geometry*, so the scratch must be
`[P, 64, 656]`-shaped).

Rollback: `GLM53_NVFP4_KV=0` in `.env` + restart.

## 7. Troubleshooting — every failure we actually hit

| symptom | cause | fix |
|---|---|---|
| node pings but ssh hangs, needs power cycle | memory livelock (no swap target for reclaim) | you skipped step 1; run `setup-node.sh` |
| `Free memory on device ... is less than desired GPU memory utilization` | UMA: page cache (weights streaming) suppresses the CUDA free-memory query | `sync; echo 3 > /proc/sys/vm/drop_caches` on both nodes, relaunch. If it persists, check `watermark_scale_factor` isn't inflated |
| watchdog trips ~15 s after graph capture | DFlash2 boot shape warmup burns big-batch shapes | `GLM53_BOOT_SHAPE_WARMUP=0` (set in env.example) |
| boot OK but ~20 GiB "missing" afterwards with nothing running | driver leak from SIGKILLed CUDA procs | wait ~10 min quiesced; it returns. Watchdog TERM-first prevents it |
| pool much smaller than expected | `MAX_MODEL_LEN` lowered, or auto-sizer vs pin mismatch | keep 900000 + the `--kv-cache-memory` pin |
| generation stops ~1 s every few s during overlap | interleave admitting a prefill chunk (by design) | tune `GLM53_MIXED_INTERLEAVE_PERIOD`/`_WINDOW` |
| engine wedges minutes at 100 % with big prompts | fat-expert Python fallback (per-expert host syncs) | `EXL3_MOE_ROW_TILE=1` **and** the kit image rebuilt so Python + `exllamav3_ext` match (`BUILD=1 ./start.sh`) — mounting new Python onto an older compiled ext hangs |
| `SM120 sparse MLA ... expects [num_pages,1,page_size,656]` | NVFP4 scratch shape | already fixed in `glm53_nvfp4_runtime.py` (page-shaped scratch) |
| `Decode (num_tokens <= 64) must go through ...decode_dsv3_2` | scratch page geometry routed decode to the paged kernel | same fix as above |

## 8. Benchmarks (llama-benchy, this exact configuration)

Single stream ([`bench/benchy-c1.md`](bench/benchy-c1.md)): prefill is a flat
**~680-705 tok/s** from 2K to 32K prompts at any depth up to 32K; decode is
**20-24 tok/s** (peak 27-29) and does not degrade with depth.

| test | prefill t/s | decode t/s (tg128) |
|---|---|---|
| pp2048 | 676 | 22.7 |
| pp8192 | 681 | 21.6 |
| pp32768 | 706 | 22.8 |
| pp32768 @ depth 32768 | 700 | 23.2 |

Concurrency ([`bench/benchy-concurrency.md`](bench/benchy-concurrency.md)),
with the honest caveat: when N long prompts arrive TOGETHER, the interleave
policy serializes their prefills (total prefill stays ~585 tok/s = near solo
speed) while active decoders run in waves (~50% duty, bursts to 48-55 tok/s
total). Expect large TTFT spread under simultaneous long-context arrivals
(c16: 122 s ± 66 s for 8K prompts). Once the prefill backlog clears, steady
mixed-load decode measured **67.6 tok/s aggregate** (GSM8K, 12-way). This is
the fundamental trade of one TP group sharing every GPU cycle: the
alternatives are prefill starvation (kit default `skip`) or uniformly ~5-10
tok/s decode (full mixing; the sparse-MLA indexer per-step cost dominates).
Tune the wave cadence with `GLM53_MIXED_INTERLEAVE_PERIOD`/`_WINDOW`.

## Credits

- [MiaAI-Lab](https://github.com/MiaAI-Lab) — the EXL3 kit this builds on
- [turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) — EXL3/TR3
- [neko-legends](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3) — the uncensored 4bpw encode
- zai-org — GLM-5.3-Flash
