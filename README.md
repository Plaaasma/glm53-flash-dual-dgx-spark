# GLM-5.3-Flash (Uncensored, EXL3 4bpw) on 2× NVIDIA DGX Spark — 2.05M-token KV pool, NVFP4 KV cache, 16-way serving

A complete, battle-tested recipe for serving **GLM-5.3-Flash** (320B/18B MoE,
vision included) across **two DGX Sparks (GB10, sm_121)** with:

- **2,049,230-token KV pool** at a 9.06 GB per-node pin (2.28× the 900K max
  context; two ~460K-token agent sessions stay prefix-cached side by side)
- **NVFP4 KV cache** — 288 B/token vs the stock 656 B `fp8_ds_mla` (2.28×
  denser), via a gather-dequant Triton path feeding the stock prebuilt kernel
- **16 concurrency slots**, 900K max context per request, MTP speculative
  decode (2.7 accepted tokens per step at 1-2 streams)
- **22.6 tok/s single-stream** averaged over 48 h of production (17 tok/s at
  460K context), 32 tok/s aggregate at 5 streams, 67.6 tok/s at 12-way GSM8K
- **Prefill ~830 tok/s** at short context with the M-tiled EXL3 MoE kernel
  (2× per MoE layer over the stock kernel), 600-760 tok/s out to 450K tokens,
  ~330 tok/s at 700K
- GSM8K **59/60** under 12-way load, **zero** request errors, greedy outputs
  byte-identical to the fp8-KV baseline
- Host-side guard rails that turn this platform's infamous memory livelocks
  into ordinary process restarts, and a live dashboard that draws the model
  itself (tokenizer → 45 layers → sampler) with a JSON API

Builds on [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(their vLLM-fork image + EXL3 kernels). This repo adds the NVFP4 KV pool, the
M-tiled prefill kernel, the mixed-prefill ladder and context-aware chunking,
the multimodal caps, the memory guard rails, a monitoring dashboard, and the
exact configuration that survives on real hardware.

Measured speeds: section 8 (production logs, 2026-09-10/11) and
[`bench/benchy-c1.md`](bench/benchy-c1.md) (llama-benchy, 2026-09-02).

### Configuration at a glance (live on 2026-09-11)

| item | value |
|---|---|
| Image / engine | `glm53-flash-sm121:local-0904-it` (21.8 GB, built 2026-09-04), vLLM fork `v0.1.dev20051+g487ecf187` |
| Host | DGX OS kernel 6.17.0-1026-nvidia, driver 580.159.03, GB10 121.63 GiB unified per node |
| Weights | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` rev `1fac3dbe`, 92 shards, 163.65 GiB, on both nodes |
| KV pool | `--kv-cache-memory 9060000000` per node → 2,049,230 tokens, 296 page IDs of 7936 tokens shared by 5 cache groups |
| Limits | `MAX_MODEL_LEN=900000`, `MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=2048`, `--prefix-match-unit 64` |
| Speculative decode | `SPEC_METHOD=mtp`, `MTP_TOKENS=3`, dynamic `[[1,2,3],[3,16,2]]` (3 drafts at 1-2 streams, 2 at 3-16) |
| Prefill | `GLM53_EXL3_MT=1` variant 8 (auto), ladder `1:1024,2:512,4:256,*:128`, `GLM53_PREFILL_CHUNK_CTX_BUDGET=3e8`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` |
| Prefix cache | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=63488` (KDA checkpoint every 8 pages) |
| Multimodal | 128 images per prompt, `max_image_tokens=1024` (1008 per 1080p), lru processor cache 1 GiB, chunks of 4, 16 cold images per request |
| Memory guard | watchdog at 0.75 GiB, boot guard at 2.5 GiB, post-load/ready reclaim 4/3 GiB, cron reclaim 2 GiB every 20 min + every minute under 2.5 GiB, in-engine maintenance every 30 s |
| Boot | launch → healthy 249-346 s over the last 17 boots (typical ~290 s) |
| Ports | API 8888, dashboard + JSON 3000, collector 9102, node agents 9101, viz UDP 9103 |

---

## 0. What you need

| thing | value |
|---|---|
| Hardware | 2× DGX Spark (GB10, 121.63 GiB unified each), joined by the CX7 200GbE QSFP cable |
| Disk | ≥ 200 GB free per node (weights are ~164 GiB, on BOTH nodes) |
| Software | stock DGX OS (kernel 6.17.0-1026-nvidia, driver 580.159.03 as tested), Docker with NVIDIA runtime, ssh between nodes as the same user |
| Accounts | a HuggingFace token (weights are public but rate-limits bite) |
| Time | ~2 h: ~1 h weight download, ~20 min image, ~5 min per server boot (InstantTensor) |

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
to ~5 min (249-346 s over the last 17 boots, both reclaims included). Two pieces, both nodes:

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

**Boot-time headroom.** `start.sh` now fires the post-load reclaim at the first
`Loading weights took` line and runs both nodes in parallel, and a boot guard
polls MemAvailable on both nodes every 2 s until the API is healthy, reclaiming
2 GiB on any node that drops under `GLM53_BOOT_GUARD_MIB` (2500). Boot 14 lost
the old head-then-worker sequence by ~10 s and the worker's watchdog killed it
mid-capture; boot 15 with the guard saw both nodes dip (worker 2.45 GiB, head
1.56 GiB) and survived.

**Runtime headroom guard.** Alongside the 20-minute reclaim, a one-minute cron on each node reclaims 2 GiB
from the vLLM container whenever MemAvailable drops under 2.5 GiB:
```bash
* * * * * [ $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) -lt 2500 ] && /usr/local/sbin/glm53-reclaim glm53-exl3-head 2 >> /var/tmp/glm53-reclaim.log 2>&1
```
It exists because a 468K-token session prefilling with uncapped 2048-token chunks drained the head from 3.4 GiB
to the watchdog line in under 15 minutes (the sparse-MLA indexer's per-step scratch scales with chunk x context).
The in-engine allocator maintenance now runs every 30 s (`GLM53_MEM_MAINT_S=30`) and logs the torch
allocation peak since the previous tick (`peak=... (+N transient)`), the multimodal processor cache is 1 GiB
and the cold-image budget 16, all to keep the head's structural headroom (~3.5 GiB) usable.

**Orphaned shared memory.** vLLM's processes create `/dev/shm/psm_*` segments and an object-storage buffer;
a watchdog kill never unlinks them and with `--ipc=host` they outlive the container. After a week of trips the
head carried 133 leftovers holding 690 MiB (mostly pushed into swap by the reclaims). `start.sh` now runs
`kit-patches/shm_cleanup.py` on both nodes after tearing the old containers down: a throwaway container of the
kit image (`--pid=host --ipc=host`, root) unlinks only the segments no process maps or holds open.

**Long-running headroom.** The post-boot headroom is not permanent: over ~14 h
of deep-context serving both nodes crept up ~0.17 GiB/h (reclaimed cold pages
fault back in, allocator slack accumulates, other host processes grow) until
the worker's watchdog tripped at MemAvailable 0.75 GiB. Two mitigations, both
optional:

1. Periodic cgroup reclaim from root's crontab on **each** node (2 GiB every
   20 min; a few seconds idle, up to ~3 min under heavy prefill load, no
   serving impact; it only pushes untouched pages to zram). Hourly was not
   enough: under sustained deep-context prefill the worker dipped from ~3 GiB
   to 1.4 GiB between runs and the watchdog tripped once at 0.75 GiB.
   ```bash
   (sudo crontab -l 2>/dev/null; echo "*/20 * * * * /usr/local/sbin/glm53-reclaim glm53-exl3-head 2 >> /var/tmp/glm53-reclaim.log 2>&1") | sudo crontab -   # worker: glm53-exl3-worker
   ```
2. In-engine maintenance (`nvfp4-kv/glm53_viz_runtime.py`, `set_batch` seam,
   on every rank): every `GLM53_MEM_MAINT_S` seconds (default 600; the live kit runs 30) it logs
   `[glm53-mem]` (MemAvailable, process RSS/swap, torch allocated/reserved) and
   returns caching-allocator slack with `torch.cuda.empty_cache()` outside
   graph capture (`GLM53_MEM_MAINT_EMPTY_CACHE=0` to log only). Grep the head
   container log for `glm53-mem` to see the trend before the watchdog does.

### 1.6 Optional: faster prefill (M-tiled EXL3 MoE kernel)

Pure prefill on the stock EXL3 kernels runs at ~30 to 45% of the GPU's compute ceiling because the fused MoE
kernel re-decodes the trellis weights for every 16 rows. `nvfp4-kv/exl3-mt/` carries a 64-row-tile version of
the same kernel (1.8 to 2.5x faster per MoE layer on prefill, decode untouched), the patcher that wires it in,
and its tests. Build the extension with `nvfp4-kv/exl3-mt/build.sh`, point `EXL3MT_SO_HOST` at it (or copy it
to `/home/liam/glm53/nvfp4-vllm/exl3-mt/`), and set `GLM53_EXL3_MT=1`. See `nvfp4-kv/exl3-mt/README.md`.

### 1.7 Long multimodal sessions: keep the prompt prefix stable, bound the memory

The KV prefix cache only helps an agent session if every turn is append-only. Anything that rewrites earlier
messages server-side (dropping an old screenshot, a placeholder whose text changes) moves the first differing
token back to that point and forces a re-prefill of everything after it; in a 334K-token session that was
~250K tokens per `read_image` call at the 128-token mixed rate, about 15 minutes. So the design is:

1. **Do not drop images in normal operation.** `--limit-mm-per-prompt` is 128 images (about 129K tokens at
   1024 tokens each). Above that, `kit-patches/patch_mm_cap.py` keeps the newest ones in batches of
   `GLM53_MM_CAP_BATCH` (16) with a constant placeholder, which is a rare fallback, not the steady state.
2. **Bound the memory instead.** The shipped image processor allows 8000 tokens per image;
   `--mm-processor-kwargs {"max_image_tokens":1024}` caps a 1080p screenshot at 1008 tokens. Cold
   preprocessing costs ~95 MB of host RAM per image (measured with vLLM's own processor) and upstream hands
   all of a request's uncached images to the HF processor in one call, so `kit-patches/patch_mm_chunk.py`
   runs it in chunks of `GLM53_MM_CHUNK` (4) images: 32 cold images peak at 1.2 GB instead of 3.2 GB, with
   byte-identical outputs (`kit-patches/tests/test_mm_chunk.py`).
3. **Cache repeated images.** `--mm-processor-cache-type lru --mm-processor-cache-gb 1` keeps processed images
   in the API server's sender cache and the engine core's mirrored receiver cache (up to 2 GiB on the head when
   full), so a turn that adds one screenshot preprocesses one image and ships the rest as hashes; a warm repeat
   of 32 images costs 0.3 s and no memory. The `shm` cache type does NOT work on a two-node TP setup: the
   worker on the other machine tries to open the head's POSIX shm segment and dies at startup.

4. **Cold guard.** Chunking bounds preprocessing, but every never-seen image still costs ~60 to 100 MB across
   the rest of the pipeline (decoded PIL image, processed tensors in the API server, the IPC copy, the engine
   core's cache, the encoder input on the GPU): a cold 32-screenshot request dipped the head by 2.8 GB. So
   the cap pass admits at most `GLM53_MM_COLD_MAX` (16 today; 24 when measured below) never-seen images per request, newest first, and keeps
   every image it accepted before (a bounded LRU of content hashes; rejected ones stay rejected). A session's
   first turn after a restart is capped, then it grows append-only to the limit with the prefix cache hitting
   (`kit-patches/tests/test_mm_cap3.py`). Measured live: a cold 40-screenshot turn admitted 24 (57 s, head
   dipped 4.8 -> 2.3 GiB); the next turn with one more screenshot kept 25, found 25,856 of 25,914 tokens in
   the prefix cache, and answered in 10 s.

On this box the head node has 3 to 6 GB of headroom next to the serving processes, which is why all four
matter. The earlier 400 (`At most N image(s) may be provided in one prompt`) no longer occurs below the limit,
and `GLM53_MM_CAP=0` restores the upstream rejection above it.

### 1.75 Max prefill speed: the two knobs

- `GLM53_EXL3_MT=1` turns on the M-tiled MoE kernel from 1.6 (1.8 to 2.5x on the MoE share of a prefill step).
- `GLM53_MIXED_PREFILL_CHUNK=ladder` with `GLM53_MIXED_PREFILL_LADDER="1:1024,2:512,4:256,*:128"` sizes the
  mixed-prefill chunk by how many other requests are decoding. A mixed step reads the whole expert set once
  whatever the chunk, so a 1024-token chunk next to one decoding stream prefills ~8x faster than the old fixed
  128 at the cost of longer gaps in that stream's output; with five or more decoders it falls back to 128.
  Note: the serving image ships an older copy of the policy helper baked in; the patcher now upgrades it in place
  at boot. Before that fix, `ladder` fell through the old helper as "unparseable" and mixed prefill ran with NO
  cap (full 2048-token chunks next to decoders): the fastest possible prefill, ~800 tok/s next to one decoder,
  with ~2.5 s gaps in that decoder's output.

### 1.76 Context-aware prefill chunks

The sparse-MLA indexer scores every query of a chunk against the whole context, so its per-step scratch scales
with chunk x context: about 1 GB at 2048 x 588K, which is the head node's entire memory margin (two watchdog
trips on 2026-09-10). `GLM53_PREFILL_CHUNK_CTX_BUDGET` (default 3e8 token^2) caps the chunk so that product
stays bounded: full 2048-token chunks below ~146K tokens of context, 896 at 300K, 384 at 588K. Per-token MoE
cost is flat above ~256-token chunks, so throughput at deep context barely changes. The mixed-prefill ladder is
applied on top. `kit-patches/patch_apc_probe.py` adds a per-group prefix-cache hit log for prompts over ~96K
tokens, to tell a prompt change from an evicted KDA state.

**What the indexer scratch actually is (2026-09-10, fourth trip at 702K).** The per-sub-chunk logits tensor is
fp32 `[query rows x context]`, and vLLM already sub-chunks the query rows so it never exceeds
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default 512 MiB); the top-k runs in place with a 1 MiB workspace and the
K gather uses a persistent workspace. So the torch-side transient is bounded at ~0.5-0.6 GiB per step, which is
exactly what every maintenance tick released at 700K. The rest of the swing the head showed before the trip
(MemAvailable 1.4-5.7 GiB between ticks) was host-side: the worker process's ~1 GiB anonymous working set being
pushed to zram by the one-minute reclaim and faulted straight back. Two settings follow from that:
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` (both TP ranks; smaller sub-chunks, same total work), and the chunk
budget stays at 3e8 (2048 below 146K, 896 at 300K, 512 at 588K, 384 at 700K, 256 at 800K). An 8e7 stopgap
with a 128-token floor was tried for one boot: at 691K context it held every step to 128 tokens and prefill
fell to ~230 tok/s, because the fixed per-step cost (one full expert-set read plus the indexer pass) dominates
below ~384-token chunks. Do not go below 3e8; raise it once the maintenance log's `peak` line shows the real
per-step maximum at deep context.

### 1.8 Why idle sessions went cold: the KV pool is ~300 page IDs shared by four cache groups

The engine reports a 2.05M-token KV cache (2.15M at the earlier 9.5 GB pin), but on this hybrid model that is
one group's view. vLLM sets the attention block to 7936 tokens so its page equals the KDA state page, and the
block pool is a single set of 296 page IDs (310 at 9.5 GB) shared by the MLA group and the three KDA groups. With prefix caching in `align` mode and no
retention interval, every 7936-token page of a conversation leaves one cached MLA page plus one cached KDA
state page per KDA group in the LRU queue: four IDs per page, so the cache holds only ~600K conversation
tokens before the least recently used session is evicted. Observed: a 417K-token session plus a 216K-token
session, and the second went cold after 38 idle minutes although "kv usage" showed 32%.

`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=63488` keeps a KDA checkpoint every 8 pages instead of every page
(the state at each prompt's end is always kept, so follow-up turns still hit in full); cost per conversation
page drops to ~1.4 IDs and the cache holds ~1.7M tokens. The only thing that gets slower is a partial-prefix
hit (a branch, or a cancelled prefill resuming), which falls back to the last checkpoint, up to 63K tokens
back. The dashboard's KV panels now show the pool in page IDs: in use, cached, free.

**Capacity and eviction order (2026-09-11).** A ~460K-token session costs 59 attention pages plus its KDA
snapshots and per-turn leftovers; two of them fill the pool (observed 271 of 296 IDs cached, 0 free). Eviction
is plain LRU by free time, and a hit refreshes every attention page but only the one KDA boundary state, so a
session's KDA checkpoints keep the age of its original prefill. Under a sub-agent fan-out they are the oldest
blocks in the pool and go first, and losing them zeroes the hybrid hit: the scheduler's probe logged
`final=0 longest=461312` (461K of attention cache present, no KDA state) 19 minutes after the session's last
turn, followed by a full re-prefill. The fix in progress refreshes a session's KDA snapshots on every hit and
logs evictions by group; until then, expect a main-agent context to go cold when a fan-out of ~8 sub-agents
runs next to a second large session.

### 1.9 Dashboard data as JSON on the same port

`dashboard/dash_server.py` replaces `python -m http.server` for port 3000: it serves the page and mirrors
everything on it as JSON, proxied from the collector on :9102 with the page's client-side arithmetic applied
server-side. `GET /api` lists the endpoints:

| endpoint | contents |
|---|---|
| `/api/all` | everything below in one document (`?viz=1` adds the activation frames, `?window=900&points=60` sizes the history) |
| `/api/derived` | per-stream tok/s over decoding streams, prefilling count, spec-decode stats, KV pool pages (in use / cached / free), est. bandwidth, per-node GPU/memory/engined |
| `/api/requests` | per-request progress: prompt/computed/total tokens, phase, prefill % |
| `/api/live`, `/api/nodes`, `/api/viz`, `/api/viz/status`, `/api/totals`, `/api/history?from&to&points` | the collector's own endpoints |
| `/api/engined` | the engined memory watch: both nodes, caps, alert state |

The page itself uses `/api` when served from :3000 (`?api=` overrides). Unit file: `ExecStart=/usr/bin/python3
/home/liam/cluster-dashboard/dash_server.py`, `Environment=PORT=3000`.

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
- **Mixed-prefill policy `GLM53_MIXED_PREFILL_CHUNK=ladder`** — the kit's
  default (`skip`) starves every new prompt's prefill while any decode runs
  (minutes of dead TTFT, client retry storms). Our first fix (`interleave`)
  ended starvation but delivered decode in waves; a fixed 128-token cap then
  bounded every engine step (table below, measured 2026-09-02 with a
  streaming client while a 35K prefill landed concurrently). The shipped
  setting today is the ladder from section 1.75 (1024/512/256/128 by the
  number of decoding peers) under the context-aware cap of 1.76: same
  bounded gaps with one or two decoders, 3-8× faster mixed prefill.

  | policy | median gap | p95 | p99 | max |
  |---|---|---|---|---|
  | interleave (waves) | 179 ms | 3122 ms | 3272 ms | 3312 ms |
  | cap 256 | 550 ms | 721 ms | 1294 ms | 1345 ms |
  | **cap 128 (shipped)** | **426 ms** | **502 ms** | **936 ms** | **1245 ms** |

  A continuous stream at any concurrency; with the fixed 128 cap prefill ran
  ~350 tok/s during overlap, with the ladder it runs at the numbers in
  section 8 (solo prefill unaffected either way).
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

- `SPEC_METHOD=mtp` with `MTP_TOKENS=3` and `MTP_DYNAMIC='[[1,2,3],[3,16,2]]'`
  — 2.7 accepted tokens per step at 1-2 streams (58% acceptance), 2.4 at 3-5.
  DFlash2 speculative decode is ~5 tok/s faster on code but its ~10 GiB
  footprint does not fit next to the 2.05M pool. Flip to `dflash` + restart
  if you want speed over pool (pool must then shrink: lower the `EXTRA_ARGS`
  pin to ≤ `--kv-cache-memory 8000000000`).
- `EXTRA_ARGS="--kv-cache-memory 9060000000 ..."` — pins the pool. Without it
  the auto-sizer eats every byte to the watchdog line and boots die. The pin
  came down from 11.24 GB (2.6M tokens) to 9.5 GB and then 9.06 GB (2.05M)
  during 2026-09-10 to buy the head node headroom at 600-700K-token contexts;
  each 0.45 GB is ~100K tokens of pool.
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

First boot: image pull (21.8 GB) + ship to worker + ~5 min load. Success
looks like:

```
GPU KV cache size: 2,049,230 tokens, Maximum concurrency for 900,000 tokens per request: 2.28x
```

and `curl localhost:8888/v1/models` answering with `glm-5.3-flash`
(OpenAI-compatible API, port 8888, vision + tool calling enabled).

Verify the guard rails: `MemAvailable` ≥ 3 GiB on both nodes after the
post-ready reclaim (head settles at 3.6-4.1 GiB, worker at ~5.5 GiB;
`awk '/MemAvailable/{print $2/1048576}' /proc/meminfo`) and
`/var/tmp/vllm-watchdog.log` shows `armed`.

## 5. Monitoring dashboard (optional, recommended)

`dashboard/` is a zero-dependency live dashboard (vLLM metrics + per-node
GPU/memory/network): run `agent.py` on both nodes (port 9101), `collector.py`
on the head (port 9102), and serve `index.html` (e.g.
`python3 -m http.server 3000`).

The 3-D "cortex" panel draws the model itself rather than a cuboid: tokenizer/embedding
at the bottom, the 45 decoder layers as plates (34 KDA linear-attention layers with per-head
cells, 11 DSA sparse-attention layers with 64 context bins and rays from the reading token to
the bins it selects; 3 dense MLP plates, then 42 MoE plates of 24x12 = 288 experts plus the
shared expert), the residual-stream spine on the left, the LM head/sampler on top and the MTP
draft layer beside it. Strands are the step's tokens through the experts they were routed to,
coloured by request. The runtime (`nvfp4-kv/glm53_viz_runtime.py`, hooks in
`kit-patches/patch_viz_hooks.py`) additionally sends per-head KDA output norms (`kda`) and the
number of tokens each request had accepted by the sampler (`acc`). No token ids or text ever
leave the engine: the tokenizer/sampler stages show per-request counts only, so nothing a user
sends or receives can be read off the dashboard. It understands vLLM's metric quirks on this
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
| head watchdog trips during prefill at 600-700K context | indexer logits scratch (≤512 MiB/step by default) + host swap churn on a node with 1-2 GiB headroom | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, chunk budget 3e8 (section 1.76), pin lowered to 9.06 GB |
| prefill ~230 tok/s at deep context | chunk budget too low (128-token chunks; the fixed per-step cost dominates) | never set `GLM53_PREFILL_CHUNK_CTX_BUDGET` below 3e8 |
| a 460K session re-prefills from zero minutes after its last turn | its KDA checkpoints were the oldest blocks and got evicted during a fan-out (section 1.8) | fix in progress; keep concurrent large sessions to two |
| server dies the moment you run a diagnostic inside the container | a `docker exec python3` that imports vLLM/torch costs ~1 GB and the watchdog counts it | read the image from a throwaway `docker run --rm` container with shell tools; never exec Python in the serving container |
| 133 orphaned `/dev/shm/psm_*` segments, 690 MiB | watchdog kills never unlink vLLM's shared memory (`--ipc=host`) | `start.sh` runs `kit-patches/shm_cleanup.py` on both nodes at launch |

## 8. Benchmarks

### Production numbers (scheduler and dashboard logs, 2026-09-10/11)

Prefill of one cold 475K-token prompt on an otherwise idle server (boot 27, M-tiled kernel, chunk budget 3e8;
the chunk cap is what the context-aware policy allowed at that depth):

| context | chunk cap | prefill tok/s |
|---|---|---|
| < 146K | 2048 | ~830 (pure prefill benchmark) |
| 150-250K | 1536-1408 | 725-760 |
| 250-300K | 1152 | 717 |
| 300-350K | 896 | 675 |
| 350-450K | 768-640 | 600-640 |
| 446-573K (boot 26) | 640-512 | 407 |
| 573-695K (boot 26) | 512-384 | 331 |

Decode over 48 h of real agent traffic (dashboard history, 69K samples):

| decoding streams | aggregate tok/s (avg) | per stream | acceptance | accepted tokens/step |
|---|---|---|---|---|
| 1 | 22.6 (17.4 at 460K context) | 22.6 | 58% | 2.7 |
| 2 | 24.3 | 12.1 | 58% | 2.7 |
| 3 | 20.0 | 6.7 | 66% | 2.4 |
| 5 | 32.0 | 6.4 | 69% | 2.4 |

Median inter-token latency while decoding is ~220 ms at the p50 the collector reports; time-to-first-token is
dominated by whether the prefix cache hits (tens of milliseconds) or a cold prefill runs (minutes at 400K+).
MoE prefill kernel: 75.1 → 38.0 ms per MoE layer on the real 288-expert routing (2.0×), 33-39 TFLOPS on the
shared-memory-B variants at ≥128 rows per expert (`nvfp4-kv/exl3-mt/README.md`).

### llama-benchy (2026-09-02, before the M-tiled kernel and the ladder)

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
