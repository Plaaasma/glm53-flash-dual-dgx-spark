# GLM-5.3-Flash (Uncensored, EXL3 4bpw) on 2× NVIDIA DGX Spark

A complete recipe for serving **GLM-5.3-Flash** (320B/18B MoE, vision included) across **two DGX Sparks
(GB10, sm_121)** joined by their CX7 cable, with the memory guard rails this platform needs, a 2.7M-token
KV pool, and a live dashboard. Everything here was measured on the hardware; the numbers are dated.

- **2,713,846-token KV pool** (`--kv-cache-memory 12000000000` per node): three full 900K contexts, or five
  ~460K-token agent sessions, stay prefix-cached side by side
- **NVFP4 KV cache**: 288 B/token vs the stock 656 B `fp8_ds_mla` (2.28× denser) through a gather-dequant
  Triton path that feeds the stock prebuilt kernel; greedy outputs byte-identical to the fp8 baseline
- **16 concurrency slots**, 900K max context, MTP speculative decode (2.75 accepted tokens per step)
- **Decode 22.7 tok/s single stream**, 35 tok/s aggregate at 2 streams, 43 at 3, ~49 at 5 (pure decode,
  from ten days of real traffic); 17 tok/s single stream at 460K context
- **Prefill 743 tok/s averaged over every pure-prefill step** since the M-tiled EXL3 MoE kernel (2× the
  stock kernel per MoE layer): ~830 tok/s at short context, 600-760 out to 450K tokens, ~330 at 700K
- **133-139 s from `start.sh` to a healthy API** (InstantTensor weight streaming, no wasted work in the
  patch and draft-model stages)
- Host guard rails that turn the platform's memory livelocks into ordinary process restarts, and a
  dashboard that draws the model itself (tokenizer → 45 layers → sampler), with everything it shows also
  served as JSON

Builds on [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(their vLLM-fork image + EXL3 kernels). This repo adds the NVFP4 KV pool, the M-tiled prefill kernel, the
mixed-prefill policies, the multimodal caps, the memory work, the dashboard, and the configuration that
survives on real hardware.

### Configuration at a glance (2026-09-20)

| item | value |
|---|---|
| Image / engine | `glm53-flash-sm121:local-0904-it` (21.8 GB), vLLM fork `v0.1.dev20051+g487ecf187`, ExLlamaV3 EXL3 kernels |
| Host | DGX OS, kernel 6.17.0-1026-nvidia, driver 580.159.03, GB10 with 121.6 GiB visible of 128 GB unified memory per node |
| Weights | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` rev `1fac3dbe`, 92 shards, 163.6 GiB, on both nodes |
| KV pool | `--kv-cache-memory 12000000000` per node → 2,713,846 tokens (page IDs of 7936 tokens shared by 5 cache groups) |
| Limits | `MAX_MODEL_LEN=900000`, `MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=2048`, `--prefix-match-unit 64` |
| Speculative decode | `SPEC_METHOD=mtp`, `MTP_TOKENS=3`, dynamic `[[1,2,3],[3,16,2]]` (3 drafts at 1-2 streams, 2 at 3-16) |
| Prefill | `GLM53_EXL3_MT=1` (M-tiled MoE kernel, auto variant), mixed ladder `1:1024,2:512,4:256,*:128`, `GLM53_PREFILL_CHUNK_CTX_BUDGET=3e8`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, `GLM53_INDEXER_PREFILL_MULT=1` |
| Prefix cache | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=63488` (KDA checkpoint every 8 pages) + checkpoint refresh on hit |
| Multimodal | 800 images per prompt, `max_image_tokens=1024` (1008 per 1080p), lru processor cache 0.5 GiB, preprocessing in chunks of 4, 16 cold images per request |
| Memory guard | watchdog at 0.75 GiB; boot guard at 4.5 GiB (reclaims 3 GiB, 6 s throttle); post-load / pre-API / post-ready reclaims 4/3/3 GiB; cron reclaim 2 GiB every 20 min and every minute under 2.5 GiB; in-engine maintenance every 15 s |
| Boot | `start.sh` → healthy API in 133-139 s |
| Ports | API 8888, dashboard + JSON 3000, collector 9102, node agents 9101, viz UDP 9103 |

---

## 0. What you need

| thing | value |
|---|---|
| Hardware | 2× DGX Spark (GB10, 121.6 GiB unified each), joined by the CX7 200GbE QSFP cable |
| Disk | ≥ 200 GB free per node (the weights are ~164 GiB on BOTH nodes) |
| Software | stock DGX OS, Docker with the NVIDIA runtime, ssh between the nodes as the same user, `socat` if you ever move the head (section 5.4) |
| Accounts | a HuggingFace token (the weights are public, but rate limits bite) |
| Time | ~2 h: ~1 h weight download, ~20 min image, ~2.5 min per server boot |

Terminology: the node you run `start.sh` on is the **head** (API server, engine core, TP rank 0); the other
is the **worker**. All commands run on the head unless said otherwise.

## 1. Host preparation — do not skip this

On the node that will run `start.sh` and on the other one:

```bash
./host-setup/setup-node.sh head      # on the head
./host-setup/setup-node.sh worker    # on the worker
```

What it does and why, each learned from a real failure:

1. **zram swap, priority 100.** The boxes ship with a 16G *file* swap that is too slow to absorb allocation
   spikes. Under unified memory almost nothing is reclaimable, so a spike stalls the kernel in reclaim and
   the whole node **livelocks: pingable, ssh dead, power cycle required. The OOM killer never fires.** Fast
   compressed swap gives reclaim a real target. `start.sh` re-creates the device with `zstd` at 10 GiB at
   every boot (`GLM53_ZRAM_ALGO`/`GLM53_ZRAM_GIB`): zstd compresses the serving processes' cold pages
   3.6-4.3:1 against lzo-rle's 2.9:1, and 10 GiB keeps the periodic reclaims from spilling into the file swap.
2. **`vm.swappiness=180`, `vm.watermark_scale_factor=100`.** Early, proactive reclaim. Do not raise the
   watermark factor further: at 500 it silently walls off ~12 GiB from `MemAvailable` and looks exactly like
   a memory leak.
3. **A memory watchdog** (`vllm-watchdog.sh`): if `MemAvailable` drops below 0.75 GiB it SIGTERMs vLLM,
   waits 3 s, then SIGKILLs. TERM-first matters: a raw SIGKILL mid-CUDA leaks ~20 GiB of driver memory that
   only returns after minutes of idle, or a reboot.
4. **Two root helpers** `start.sh` calls through `sudo -n`, installed with a sudoers entry:
   `glm53-reclaim <container> <GiB>` writes the container cgroup's `memory.reclaim` (pushes cold pages to
   zram) and `glm53-zram <algo> <GiB>` re-creates the zram device. Section 5.2 explains when each runs.
5. **Root cron reclaims** for this node's container: 2 GiB every 20 minutes, and every minute while
   `MemAvailable` is under 2.5 GiB.

**Steps 1-3 do not survive a reboot.** Re-run `setup-node.sh` after every reboot.

## 2. Get the kit and apply the patches

```bash
git clone <this repo> && cd glm53-spark-recipe
./apply-kit-patches.sh
```

This clones the MiaAI-Lab kit at the tested commit (`493cb88`), applies `kit-patches/exl3-kit.patch` to
the kit's own overlay files, then lays this repo's files over it:

- `exl3-kit/start.sh` (this repo's launch script: parallel node setup over one multiplexed ssh connection,
  the reclaim sequence, the boot guard, orphaned-shm cleanup, boot phases for the dashboard) and
  `exl3-kit/overlay/patch_*.py` + `shm_cleanup.py` (the patchers; each is idempotent and fails closed if
  the vLLM anchor it edits has drifted).
- `nvfp4-vllm/` next to the kit: the NVFP4 KV runtime and patcher, the viz/maintenance runtime, and the
  M-tiled MoE kernel sources. `start.sh` finds them there (`NVFP4_DIR` overrides). Build the kernel once
  with `nvfp4-vllm/exl3-mt/build.sh` (a `docker run` of the kit image, ~2 min); without the `.so` the
  patcher skips itself and the stock kernel is used.
- `exl3-kit/.env` from `kit-patches/env.example`.

Edit `exl3-kit/.env`:

| key | set to |
|---|---|
| `HF_TOKEN` | your token |
| `HEAD_IP` / `WORKER_IP` | the 169.254.x.x link-local addresses of the CX7 link (`ip -br a` on each node) |
| `HEAD_CX7_IF` etc. | your NIC names: `ip -br a` + `ls /sys/class/infiniband`. Both nodes here use `enp1s0f1np1` / `rocep1s0f1`; the kit's default wrongly assumes the worker uses f0 |
| `NCCL_IB_GID_INDEX` | the GID index whose entry contains `::ffff:<your fabric IP>` (`/sys/class/infiniband/<dev>/ports/1/gids/*`); 3 on both nodes here |

Everything else in `env.example` is the tested configuration. The load-bearing values, so you don't
"clean them up":

- `EXTRA_ARGS="--kv-cache-memory 12000000000 ..."` pins the pool. Without a pin the auto-sizer eats every
  byte to the watchdog line and boots die. Section 5.1 shows what 12 GB leaves free and what to change
  if your nodes run anything else.
- `MAX_MODEL_LEN=900000`: do NOT lower it "to save memory": hybrid block-id overhead then *doubles* the
  per-token pool cost (measured 8.1 → 18.5 KB/token).
- `SPEC_METHOD=mtp` with `MTP_TOKENS=3` and `MTP_DYNAMIC='[[1,2,3],[3,16,2]]'`: 2.75 accepted tokens per
  step at 1-2 streams (58% acceptance), 2.4 at 3-5. DFlash2 is ~5 tok/s faster on code but its ~10 GiB
  footprint does not fit next to this pool.
- `GPU_MEM_UTIL=0.875`, `CG_ESTIMATE=1`, `GLM53_BOOT_SHAPE_WARMUP=0`, `EXL3_MOE_ROW_TILE=1`: each guards a
  specific boot failure (section 10).

## 3. Weights

The kit downloads `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` (163.65 GiB) on the first `./start.sh`
and rsyncs it to the worker. If your internet is slow, run `./download.sh` first.

## 4. Launch

```bash
cd exl3-kit && ./start.sh        # ./start.sh restart to bounce a running pair
```

First boot: image pull (21.8 GB) + ship to the worker + load. Every later boot, measured from the
command to a healthy API:

| +s | phase |
|---|---|
| 13 | old containers gone, orphaned shm cleaned, zram re-created (both nodes in parallel) |
| 24 | kit patches applied in both containers, `vllm serve` launched |
| 89 | 164 GB of weights streamed (35 s, InstantTensor Direct I/O at ~4.7 GB/s) |
| 94 | draft (MTP) model loaded from its 4 shards (3 s) |
| 111 | KV cache allocated: `GPU KV cache size: 2,713,846 tokens, Maximum concurrency for 900,000 tokens per request: 3.02x` |
| 122 | CUDA graphs captured (10 s) |
| 133 | `Application startup complete`, health check passed, post-ready reclaim running |

`curl localhost:8888/v1/models` answers with `glm-5.3-flash` (OpenAI-compatible API, vision and tool
calling on). Verify the guard rails: `MemAvailable` ≥ 3 GiB on both nodes after the post-ready reclaim
(`awk '/MemAvailable/{print $2/1048576}' /proc/meminfo`) and `/var/tmp/vllm-watchdog.log` shows `armed`.

The stock loader mmaps the safetensors at 150-300 MB/s (259 s for 164 GB); the kit image built with
`image/Dockerfile.instanttensor` streams them in 35 s. The fast load applies no page-cache pressure, so
nothing gets swapped and graph capture would trip the watchdog: that is what the post-load reclaim exists
for. `--load-format auto` plus the reclaim knobs at 0 gives the stock behaviour.

## 5. Memory on a Spark

### 5.1 Where the 121.6 GiB goes

The checkpoint is 163.6 GiB; split over TP=2 with the replicated parts (embeddings, lm_head, vision
tower) it is 83.5 GiB per node. "So ~38 GB is free for KV" is the natural guess and it is wrong by
~25 GB. The budget below is the head node at steady state on 2026-09-20 (12 GB pin, no desktop session),
reconciled to `MemTotal - MemAvailable` from /proc/meminfo, per-process RSS, zram, and
`nvidia-smi --query-compute-apps` (the driver's per-process GPU memory, which on unified memory is host
RAM like everything else):

| GB | what |
|---|---|
| 86.6 | model weights and torch workspaces (torch reserved minus the KV pool) |
| 11.4 | KV pool (`--kv-cache-memory 12000000000`) |
| 1.9 | GPU memory outside torch: CUDA context, NCCL/RoCE buffers, CUDA-graph exec objects, compiled kernels |
| 2.6 | driver-owned shared memory: appears at CUDA context creation, identical on both nodes, not releasable from user space |
| 2.2 | unreclaimable kernel slab (driver page tracking for ~100 GB of mappings) |
| 2.1 | vLLM processes' resident host memory (API server, engine core, worker, resource tracker) |
| 1.5 | zram store holding 5.3 GB of the same processes' cold pages (the reclaims' price) |
| 3.6 | other host processes: DGX OS daemons plus anything else you run. A GNOME desktop session with remote desktop is another ~2.6 GB; a co-tenant process costs its RSS **plus** the GPU memory the driver attributes to it |
| 1.9 | that co-tenant's GPU memory (one CUDA-using service here) |
| 0.3 | kernel stacks, page tables, vmalloc, per-cpu |
| 5.5 | file cache the kernel's watermark math does not count as available (`watermark_scale_factor=100`) |
| 4.8 | `MemAvailable` |
| **121.6** | total visible to Linux |

The other 6.4 GiB of the 128 GB is gone before Linux boots and none of it is recoverable: 3.5 GiB of
firmware-reserved regions in the EFI map (GPU firmware and system carveout), 2.0 GiB of kernel page
bookkeeping (33.5M 4 KB pages × 64 B; a 64 KB-page kernel would shrink it 16× but DGX OS ships 4 KB),
0.35 GiB carved out before the memory map, 0.3 GiB of kernel image, initrd and CMA. The crash-kernel
reservation is already off (`crashkernel=1G-:0M`).

So the serving stack costs ~97 GB per node before a single KV token. TP pins the same KV on both nodes,
so the node with less free memory sets the pool: every GB you add to the pin must exist on both. Rules
of thumb at this pin: keep 3 GB free on the tighter node at steady state; each 0.45 GB of pin is ~100K
tokens; a co-tenant that grows by N GB costs N GB of pin.

### 5.2 Guard rails

- **Boot:** `start.sh` reclaims 4 GiB from both containers at the first `Loading weights took` line
  (`GLM53_POSTLOAD_RECLAIM=4`), 3 GiB right after graph capture (`GLM53_PREAPI_RECLAIM=3`, the API
  server's startup is the boot's peak), and 3 GiB once healthy (`GLM53_POSTREADY_RECLAIM=3`); a boot guard
  polls both nodes every 2 s and reclaims 3 GiB on any node under `GLM53_BOOT_GUARD_MIB` (4500).
- **Runtime:** root's crontab on each node reclaims 2 GiB every 20 minutes and every minute while
  `MemAvailable` is under 2.5 GiB (`setup-node.sh` installs both lines; the container name differs per
  node). Over ~14 h of deep-context serving both nodes otherwise creep up ~0.17 GiB/h until the watchdog
  fires.
- **In-engine maintenance** (`nvfp4-vllm/glm53_viz_runtime.py`, every `GLM53_MEM_MAINT_S`=15 s on every
  rank): logs `[glm53-mem]` (MemAvailable, process RSS/swap, torch allocated, the allocation peak since the
  last tick), returns caching-allocator slack with `torch.cuda.empty_cache()` outside graph capture, and
  runs `malloc_trim(0)` (the engine core does the same every 60 s). Grep the head container log for
  `glm53-mem` to see the trend before the watchdog does.
- **Orphaned shared memory:** a watchdog kill never unlinks vLLM's `/dev/shm/psm_*` segments and with
  `--ipc=host` they outlive the container (133 leftovers holding 690 MiB after a week of trips).
  `start.sh` runs `shm_cleanup.py` on both nodes at launch; it unlinks only segments no process maps.
- **Never run Python inside the serving containers.** A `docker exec ... python3` that imports vLLM costs
  ~1 GB and the watchdog counts it; on a node with 1-2 GB of headroom that kills the server. Read files
  from the image with a throwaway `docker run --rm` container and shell tools.

### 5.3 What else was tried

| result | change |
|---|---|
| −4.3 GB | indexer K-gather workspace: upstream sizes it at `max_model_len × 40` entries (DeepSeek-V3.2's constant), 4.75 GB at 900K. With this model's 4:1 key pooling one full request needs 225K entries; `GLM53_INDEXER_PREFILL_MULT=1` (`patch_indexer_buffer.py`) makes it 119 MB and multi-request prefill steps simply chunk more |
| −0.45 GB | zram zstd instead of lzo-rle (section 1) |
| −0.2 GB | CUDA-graph exec objects for batch sizes 3, 24 and 48 (capture sizes `1 2 4 8 16 32 64`; those batches pad to the next size) |
| ~0 | `malloc_trim` in the worker: its 2.5 GB glibc heap after loading 150K tensors is live cold data, not free chunks (4 MiB released; the engine core gave back 219 MiB of swap). Kept because it is free |
| not cuttable | the 2.6 GB driver shared memory (CUDA context), the 2.2 GB slab (driver page tracking), NCCL's buffers (small for two ranks on one link), the vision tower and draft layer, and the KV bytes per token (the MLA latent is already NVFP4; FP4 indexer keys need sm_100) |

### 5.4 Put the head on the node without the desktop

The head (API server, engine core, resource tracker) costs ~2 GB of host RAM that the worker does not.
DGX OS runs a GNOME session, and a remote-desktop session on top of it, worth another ~2.6 GB. If one
node is the one you sit at, make the other node the head: swap `HEAD_IP`/`WORKER_IP` in its `.env`, run
`start.sh` there (it needs an ssh key into the new worker and the section 1 helpers on both nodes), swap
the container names in the two root crontabs, and keep clients on the old address with
`host-setup/glm53-api-forward.service` (socat over the fabric link). The dashboard stays where it was
(section 9). Measured on this pair: each node has ~2 GB more headroom in its new role.

## 6. The KV cache

### 6.1 NVFP4 pool

`nvfp4-vllm/` stores the sparse-MLA latent as **288 B/token** (256 B packed e2m1 pairs + 32 B e4m3
per-16 block scales) instead of the stock 656 B `fp8_ds_mla` layout. The prebuilt FlashInfer sparse-MLA
kernel cannot read NVFP4, so it never sees it: DSA attention only touches its top-k selected rows, and a
Triton kernel gathers + dequantizes exactly those rows into a page-shaped fp8 scratch the kernel accepts
(4 × fp32 per-128-group scales, matched to the real cache-write op byte for byte).

- decode gather 0.36 ms/layer; prefill union ≤ 5.6 ms/layer; write free
- accuracy: greedy outputs byte-identical to fp8 KV on probe prompts; GSM8K parity (59/60 at 12-way);
  cos 0.995 against an fp8 pool standalone, i.e. pure NVFP4 quantization noise
- `nvfp4-kv/PLAN.md` has the design, the test methodology and the two integration traps (the executing
  backend is `flashinfer_mla_sparse_sm120.py`, not its identically-shaped sibling; the kernel routes
  decode-vs-paged on page *geometry*, so the scratch must be `[P, 64, 656]`-shaped)
- rollback: `GLM53_NVFP4_KV=0` + restart

### 6.2 One block pool for five cache groups

The engine's "GPU KV cache size" already accounts for this, but it matters for prefix caching. vLLM sets
the attention block to 7936 tokens so its page equals the KDA state page, and the block pool is one set of
page IDs shared by the MLA group and the three KDA groups. With prefix caching in `align` mode and dense
checkpoints, every 7936-token page of a conversation leaves one cached MLA page plus one cached KDA state
per KDA group: four IDs per page, so the cache held only ~600K conversation tokens before evicting the
least recently used session. `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=63488` keeps a KDA checkpoint every
8 pages instead of every page (the state at each prompt's end is always kept, so follow-up turns still hit
in full); cost per page drops to ~1.4 IDs. The only thing slower is a partial-prefix hit (a branch, or a
cancelled prefill resuming), which replays from the last checkpoint, up to 63K tokens back.

Eviction is plain LRU by free time, and a hit refreshes every attention page but only the one KDA
boundary state it resumes from, so a session's older checkpoints kept the age of its original prefill and
were the first things evicted under a sub-agent fan-out; losing them zeroes the hybrid hit and the whole
context re-prefills. `patch_apc_refresh.py` moves the hit prefix's cached KDA states to the young end of
the queue on every hit and logs evictions by cache group once a minute (`[glm53-apc] evicted cached
blocks by group ...`). `patch_apc_probe.py` logs the per-group hit length for prompts over ~96K tokens,
which tells a client-side prompt edit (hit stops at the edit) from an eviction (attention hit long, KDA
hit zero).

The prefix cache only helps an agent session if every turn is append-only. A client that rewrites an
earlier message (drops or downscales an old image, changes a placeholder) moves the first differing token
back to that point and everything after it re-prefills; at 500K tokens that is minutes per turn. Several
agent frameworks do this once a conversation's image payload passes a byte budget: raise that budget on
the client.

## 7. Prefill

- **M-tiled EXL3 MoE kernel** (`nvfp4-vllm/exl3-mt/`, `GLM53_EXL3_MT=1`). The stock fused MoE kernel
  re-decodes the trellis weights every 16 rows, so pure prefill ran at 30-45% of the compute ceiling. The
  M-tiled variants decode once per block: 75.1 → 38.0 ms per MoE layer on the real 288-expert routing
  (2.0×), 33-39 TFLOPS on the shared-memory-B variants at ≥128 rows per expert; the auto variant picks
  per expert. Decode is untouched. `nvfp4-kv/exl3-mt/README.md` has the kernel notes and benchmarks.
- **Mixed prefill next to decoding streams** (`GLM53_MIXED_PREFILL_CHUNK=ladder`,
  `GLM53_MIXED_PREFILL_LADDER="1:1024,2:512,4:256,*:128"`). The kit's default `skip` starves a new
  prompt while anything decodes (minutes of dead TTFT); a fixed 128-token chunk bounded every step but
  held prefill to ~350 tok/s during overlap. A mixed step reads the whole expert set once whatever the
  chunk, so the ladder sizes the chunk by the number of decoding peers: 1024 next to one stream down to
  128 with five or more, keeping the gaps in those streams' output bounded. Section 11 has the measured
  prefill and decode rates during overlap.
- **Context-aware chunks** (`GLM53_PREFILL_CHUNK_CTX_BUDGET=3e8` token²). The sparse-MLA indexer scores
  every query of a chunk against the whole context; vLLM bounds the fp32 score tensor at
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (128 here, both ranks) by sub-chunking, but the per-step transient
  still grows with chunk × context. The budget caps the chunk: 2048 below 146K context, 896 at 300K, 512
  at 588K, 384 at 700K. Do not go below 3e8: an 8e7 budget held a 691K-token prompt to 128-token steps and
  prefill fell to ~230 tok/s, because the fixed per-step cost (one expert-set read plus the indexer pass)
  dominates below ~384-token chunks.

## 8. Long multimodal sessions

The vision path is the other place a Spark's headroom goes. Four pieces keep it bounded:

1. **`--limit-mm-per-prompt` is 800 images** (up to 819K tokens at 1024 each, the most that fits in the
   900K context), so the cap below is a theoretical fallback, not the steady state. Above the limit
   `patch_mm_cap.py` keeps the newest images in batches of `GLM53_MM_CAP_BATCH` (16) with a constant
   placeholder, so the prompt prefix stays stable across turns.
2. **`--mm-processor-kwargs {"max_image_tokens":1024}`** caps a 1080p screenshot at 1008 tokens (the
   shipped processor allows 8000). Cold preprocessing costs ~95 MB of host RAM per image, and upstream
   hands all of a request's uncached images to the HF processor in one call; `patch_mm_chunk.py` runs it
   in chunks of `GLM53_MM_CHUNK` (4): 32 cold images peak at 1.2 GB instead of 3.2 GB, byte-identical
   outputs (`kit-patches/tests/test_mm_chunk.py`).
3. **`--mm-processor-cache-type lru --mm-processor-cache-gb 0.5`** keeps processed images in the API
   server's cache and the engine core's mirror (~53 images), so a turn that adds one screenshot
   preprocesses one image and ships the rest as hashes. A prompt with more images than the cache holds
   re-preprocesses all of them every turn (~150 ms each). The `shm` cache type does not work across two
   nodes: the worker tries to open the head's POSIX shm segment and dies at startup.
4. **Cold guard.** Every never-seen image still costs ~60-100 MB across the pipeline (decoded image,
   processed tensors, the IPC copy, the encoder input); a cold 32-screenshot request dipped the head by
   2.8 GB. The cap pass admits at most `GLM53_MM_COLD_MAX` (16) never-seen images per request, newest
   first, and remembers what it accepted (a bounded LRU of content hashes), so a session's first turn after
   a restart is capped and then grows append-only with the prefix cache hitting
   (`kit-patches/tests/test_mm_cap3.py`).

`GLM53_MM_CAP=0` restores upstream's `At most N image(s) may be provided in one prompt` rejection.

## 9. Dashboard

`dashboard/` is a zero-dependency live dashboard: `agent.py` on each node (`:9101`, GPU/memory/cpu/net plus
any watched host processes), `collector.py` on the dashboard node (`:9102`, polls vLLM metrics and the
agents every 2.5 s, keeps 35 days of history in SQLite, receives the engine's activation telemetry), and
`dash_server.py` (`:3000`, serves the page and mirrors everything on it as JSON). Run them as systemd units
with `User=` your user; configuration is by environment:

| variable | meaning (default) |
|---|---|
| `GLM53_HEAD_SSH` | `user@<head fabric IP>` when the vLLM head is the other node; the head container is then reached with `docker -H ssh://…` (empty: this node) |
| `GLM53_HEAD_KIT` | the kit directory on the head, for the boot phase file (`~/glm53/exl3-kit`) |
| `VLLM_METRICS_URL` | `http://localhost:8888/metrics` |
| `SPARK_AGENT_W` | the other node's agent, `http://<ip>:9101/stats` (`SPARK_AGENT_H` defaults to localhost) |
| `SPARK_NODE_LABELS` | role labels for the two node cards, this node first (`HEAD · API,WORKER`) |
| `SPARK_WATCH_PROCS` | agent: comma-separated command names to report, RSS + GPU memory (none) |
| `SPARK_PROC_CAP_MIB`, `SPARK_PROC_CAP_TOTAL_MIB` | flag the node card and `/api/procs` when the watched processes pass these (none) |
| `SPARK_DB_PATH`, `PORT`, `COLLECTOR` | history DB path (next to `collector.py`), page port (3000), collector URL |

If the head is the other node, its `.env` needs `GLM53_VIZ_UDP=<dashboard node fabric IP>:9103` so the
engine hooks reach the collector.

What it shows: live throughput (prefill derived from the per-step iteration histogram, because
`prompt_tokens_total` only updates at request completion; per-stream decode over the streams actually
decoding), the KV pool as tokens used of total (from the engine's own "GPU KV cache size" line) and as
page IDs in use / cached / free, per-request prefill progress, spec-decode acceptance, node meters, and a
startup panel that is a real stage bar during boots (each stage boundary comes from the engine log's
timestamps and `start.sh`'s phase file). The 3-D "cortex" panel draws the model itself: tokenizer at the
bottom, the 45 decoder layers as plates (34 KDA linear-attention layers with per-head cells, 11 DSA
sparse-attention layers with 64 context bins and rays from the reading token to the bins it selects,
3 dense MLP plates, then 42 MoE plates of 24×12 experts plus the shared expert), the residual-stream
spine, the LM head/sampler on top and the MTP draft layer beside it; strands are the step's tokens
through the experts they were routed to, coloured by request. No token ids or text ever leave the
engine: the hooks (`kit-patches/patch_viz_hooks.py`, `nvfp4-vllm/glm53_viz_runtime.py`) send routing,
attention bins, residual norms, per-head KDA activity and per-request token counts only.

`GET /api` lists the JSON endpoints: `/api/all`, `/api/live`, `/api/derived` (the page's client-side
arithmetic done server-side), `/api/requests`, `/api/nodes`, `/api/viz`, `/api/viz/status`,
`/api/totals`, `/api/history?from&to&points`, `/api/procs`. All responses carry
`Access-Control-Allow-Origin: *`.

## 10. Troubleshooting — every failure actually hit

| symptom | cause | fix |
|---|---|---|
| node pings but ssh hangs, needs a power cycle | memory livelock (no swap target for reclaim) | you skipped section 1; run `setup-node.sh` |
| `Free memory on device ... is less than desired GPU memory utilization` | UMA: page cache (weights streaming) suppresses the CUDA free-memory query | `sync; echo 3 > /proc/sys/vm/drop_caches` on both nodes, relaunch. If it persists, check `watermark_scale_factor` isn't inflated |
| watchdog trips ~15 s after graph capture | DFlash2 boot shape warmup burns big-batch shapes | `GLM53_BOOT_SHAPE_WARMUP=0` (in env.example) |
| boot OK but ~20 GiB "missing" afterwards with nothing running | driver leak from SIGKILLed CUDA processes | wait ~10 min quiesced; it returns. The watchdog's TERM-first prevents it |
| pool much smaller than expected | `MAX_MODEL_LEN` lowered, or auto-sizer vs pin mismatch | keep 900000 + the `--kv-cache-memory` pin |
| watchdog trips during a boot's API-server startup | that is the boot's memory peak; another process grew on that node | the pre-API reclaim and boot guard (section 5.2) cover ~3 GB of it; beyond that, lower the pin or the other process |
| watchdog trips hours into a long-context session with no big transient | slow headroom erosion (section 5.2) plus something else growing | the cron reclaims; check `nvidia-smi --query-compute-apps` for a co-tenant's GPU memory, which RSS does not show |
| engine wedges for minutes at 100% with big prompts | fat-expert Python fallback (per-expert host syncs) | `EXL3_MOE_ROW_TILE=1` **and** the kit image rebuilt so Python + `exllamav3_ext` match (`BUILD=1 ./start.sh`); mounting new Python onto an older compiled ext hangs |
| `SM120 sparse MLA ... expects [num_pages,1,page_size,656]` | NVFP4 scratch shape | fixed in `glm53_nvfp4_runtime.py` (page-shaped scratch) |
| `Decode (num_tokens <= 64) must go through ...decode_dsv3_2` | scratch page geometry routed decode to the paged kernel | same fix |
| prefill ~230 tok/s at deep context | chunk budget too low (128-token chunks) | never set `GLM53_PREFILL_CHUNK_CTX_BUDGET` below 3e8 |
| a 460K session re-prefills from zero minutes after its last turn | its KDA checkpoints were the oldest blocks and got evicted during a fan-out (section 6.2), or the client edited an earlier message | `patch_apc_refresh.py`; read the `[glm53-dbg] apc hit` line to tell the two apart |
| `400: At most N image(s) may be provided in one prompt` | the multimodal limit | 800 images + `patch_mm_cap.py` (section 8) |
| server dies the moment you run a diagnostic inside the container | `docker exec ... python3` importing vLLM costs ~1 GB | read the image from a throwaway `docker run --rm` container; never exec Python in the serving container |
| 133 orphaned `/dev/shm/psm_*` segments, 690 MiB | watchdog kills never unlink vLLM's shared memory (`--ipc=host`) | `start.sh` runs `shm_cleanup.py` on both nodes at launch |
| head exits 0 with `RuntimeError: cancelled` after `CUDA error: an illegal memory access` in a CUDA-graph replay; kernel log `Xid 31 ... MMU Fault` | a kernel in the decode graph computed an unmapped address (once in 25 h, 1-token step at 170K context; GPU healthy after) | restart; `glm53_nvfp4_runtime.py` now clamps gathered pool rows to the pool's range; save the container log before `start.sh` removes it |
| worker dies at startup with `--mm-processor-cache-type shm` | it cannot open the head's shm segment across nodes | use `lru` |

## 11. Throughput, from the server's own history

All numbers below are the dashboard's 2.5 s samples of real agent traffic between 2026-09-11 (when the
M-tiled kernel went live) and 2026-09-20, grouped by what the engine was doing in that step. They are
averages over thousands of samples on this exact configuration, not a synthetic benchmark.

**Pure decode** (no prefill chunk in the step):

| decoding streams | samples | aggregate tok/s | per stream | draft acceptance | accepted tokens/step |
|---|---|---|---|---|---|
| 1 | 75,312 | 22.7 | 22.7 | 58% | 2.75 |
| 2 | 12,306 | 35.5 | 17.7 | 58% | 2.71 |
| 3 | 3,421 | 43.4 | 14.5 | 70% | 2.41 |
| 4 | 2,321 | 41.2 | 10.3 | 71% | 2.42 |
| 5 | 231 | 48.7 | 9.7 | 67% | 2.33 |
| 6 | 54 | 55.7 | 9.3 | 65% | 2.30 |

Single-stream decode falls with context: 22.7 tok/s at short context, 17.4 at 460K. Median inter-token
latency ~220 ms. Under 12-way GSM8K load: 67.6 tok/s aggregate, 59/60 correct, zero request errors.

**Mixed steps** (a prefill chunk scheduled next to decoding streams under the ladder policy): what a new
prompt gets while others are generating, and what they keep.

| decoding streams | samples | prefill tok/s | decode tok/s, aggregate | per stream |
|---|---|---|---|---|
| 1 | 7,056 | 555 | 7.6 | 7.6 |
| 2 | 3,546 | 561 | 8.9 | 4.4 |
| 3 | 2,441 | 391 | 12.0 | 4.0 |
| 4 | 763 | 355 | 15.3 | 3.8 |
| 5 | 273 | 261 | 19.3 | 3.9 |
| 6 | 46 | 224 | 23.5 | 3.9 |

**Pure prefill** (nothing decoding): 743 tok/s averaged over 9,015 samples at every context depth. By
depth, one cold 475K-token prompt on an idle server (2026-09-11; the chunk cap is what the context-aware
budget allowed):

| context | chunk cap | prefill tok/s |
|---|---|---|
| < 146K | 2048 | ~830 |
| 150-250K | 1536-1408 | 725-760 |
| 250-300K | 1152 | 717 |
| 300-350K | 896 | 675 |
| 350-450K | 768-640 | 600-640 |
| 446-573K | 640-512 | 407 |
| 573-695K | 512-384 | 331 |

Time to first token is whether the prefix cache hits (tens of milliseconds) or a cold prefill runs
(minutes at 400K+). MoE prefill kernel: 75.1 → 38.0 ms per MoE layer on the real 288-expert routing.

## Credits

- [MiaAI-Lab](https://github.com/MiaAI-Lab): the EXL3 kit this builds on
- [turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3): EXL3/TR3
- [neko-legends](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3): the uncensored 4bpw encode
- zai-org: GLM-5.3-Flash
