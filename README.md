# GLM-5.3-Flash (Uncensored, EXL3 4bpw) on 2× NVIDIA DGX Spark

A complete recipe for serving **GLM-5.3-Flash** (320B/18B MoE, vision included) across **two DGX Sparks (GB10,
sm_121)** joined by their CX7 cable. It covers the memory guard rails this platform needs, a 2.7M-token KV pool,
decode and prefill kernels tuned for GB10, crash recovery and a live dashboard. Everything here was measured on
the hardware, and the numbers are dated.

**Decode on real agent traffic** (pure-decode steps since 2026-10-01; section 12):
- **34 tok/s** at one stream, **54** aggregate at 2 streams, **62** at 3. Before the October kernel work: 22.7 / 35.6 / 43.6.

**Cold prefill, idle server:**
- **~1,100 tok/s** up to 78K tokens; **~1,000 tok/s** still at 300K, so a 321K-token prompt is ready in under 5 minutes. Before: 797-823 at 9-33K, 661 at 78K.

**What makes the speed:**
- **8-bit weights for every BF16 matrix** (attention projections, shared experts, dense MLP, LM head):
  - matrices that came from GLM's FP8 release go back to FP8 on their own 128×128 block grid, near exact;
  - native-BF16 matrices go to INT8 with per-(row, 128) scales;
  - Triton kernels that stream them at ~210 GB/s.
- **A grouped EXL3 decode kernel** that reads each distinct routed expert once at ~230 GB/s.
- **A fused EXL3 prefill kernel**, 1.6-2.0× per MoE layer.
- **NCCL over both PCIe functions of the CX7.**
- **Quality:** GSM8K-100 98-99/100 against 98 before; teacher-forced NLL +0.7%.

**Memory and capacity:**
- **2,713,846-token KV pool** (`--kv-cache-memory 12000000000` per node). Three full 900K contexts, or five
  ~460K-token agent sessions, stay prefix-cached side by side.
- **NVFP4 KV cache:** 288 B/token instead of the stock 656 B `fp8_ds_mla`.
- **16 concurrency slots**, 900K max context, MTP speculative decode (2.7-2.9 accepted tokens per step).
- **~140 s from `start.sh` to a healthy API**, with ~4 GB more free memory per node than before.

**Keeping it up:**
- **Host guard rails** turn the platform's memory livelocks into ordinary process restarts.
- **An optional auto-recovery service** restarts the pair only when a rank has really died.
- **A dashboard** draws the model itself (tokenizer → 45 layers → sampler) and serves everything it shows as JSON.

Builds on [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(their vLLM-fork image and EXL3 kernels).

**This repo adds:**
- the NVFP4 KV pool;
- the 8-bit dense weights;
- the three EXL3 MoE kernels. Two are ported from
  [TensorFold](https://github.com/ashhart/TensorFold) and
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark), both Apache-2.0;
- the mixed-prefill policies and the multimodal caps;
- the memory work, crash recovery and dashboard;
- the configuration that survives on real hardware.

### Configuration at a glance (2026-10-03)

| item | value |
|---|---|
| Image / engine | `glm53-flash-sm121:local-0904-it` (21.8 GB), vLLM fork `v0.1.dev20051+g487ecf187`, ExLlamaV3 EXL3 kernels |
| Host | DGX OS, kernel 6.17.0-1026-nvidia, driver 580.159.03, GB10 with 121.6 GiB visible of 128 GB unified memory per node |
| Weights | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` rev `1fac3dbe`, 92 shards, 163.6 GiB, on both nodes; non-expert matrices converted to 8 bits at load (`GLM53_DENSE_W8=auto`: 8.13 → 4.15 GB a rank) |
| KV pool | `--kv-cache-memory 12000000000` per node → 2,713,846 tokens (page IDs of 7936 tokens shared by 5 cache groups) |
| Limits | `MAX_MODEL_LEN=900000`, `MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=2048`, `--prefix-match-unit 64` |
| Speculative decode | `SPEC_METHOD=mtp`, `MTP_TOKENS=3`, dynamic `[[1,2,3],[3,16,2]]` (3 drafts at 1-2 streams, 2 at 3-16), greedy drafts |
| MoE kernels | decode: `GLM53_EXL3_DEC=1` (grouped, ≤ 64 tokens); prefill: `GLM53_EXL3_FAT=1` (fused, > 64 tokens), M-tiled kernel (`GLM53_EXL3_MT=1`, variant 8) as fallback |
| Prefill | mixed ladder `1:1024,2:512,4:256,*:128`, `GLM53_PREFILL_CHUNK_CTX_BUDGET=3e8`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, `GLM53_INDEXER_PREFILL_MULT=1` |
| Interconnect | NCCL over both CX7 PCIe functions (`HEAD_NCCL_HCA` / `WORKER_NCCL_HCA`), 4 channels; per-node GID index (`HEAD_GID` / `WORKER_GID`) |
| Prefix cache | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=63488` (KDA checkpoint every 8 pages) + checkpoint refresh on hit |
| Multimodal | 800 images per prompt, `max_image_tokens=1024` (1008 per 1080p), lru processor cache 0.5 GiB, preprocessing in chunks of 4, 16 cold images per request |
| Memory guard | watchdog at 0.75 GiB; boot guard at 4.5 GiB (reclaims 3 GiB, 6 s throttle); post-load / pre-API reclaims 4 / 3 GiB; cron reclaim 2 GiB every minute under 2.5 GiB; in-engine maintenance every 15 s (GPU drained first) |
| Recovery | `glm53-autorecover.service` on the worker: restarts only a dead or hung pair, max 3 an hour, saves both containers' logs first |
| Boot | `start.sh` → healthy API in ~140 s |
| Ports | API 8888, dashboard + JSON 3000, collector 9102, node agents 9101, viz UDP 9103, optional attribution proxy 8890 |

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
./host-setup/setup-node.sh head                                        # on the head
./host-setup/setup-node.sh worker --autorecover user@<head fabric IP>  # on the worker (--autorecover optional)
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
5. **A root cron reclaim** for this node's container: 2 GiB every minute while `MemAvailable` is under 2.5 GiB.
   An unconditional reclaim every 20 minutes used to run as well. On a node with a desktop session each pass took
   17-40 minutes at 100% CPU, recovered nothing, and slowed that node's GPU (section 5.4), so it was dropped.
6. **Optional auto-recovery** (`--autorecover`, worker only): `glm53-autorecover.service` checks the pair every 15 s.
   It runs `start.sh restart` on the head (over ssh) only when:
   - a rank logged a worker exception or an illegal memory access;
   - a container stopped outside a boot;
   - the engine hung (a rank silent for 5 minutes);
   - or `/health` failed for 5 minutes.

   It never touches a healthy server and allows at most 3 restarts an hour. Before each restart it saves both
   containers' logs under the head's `exl3-kit/logs/crash-<time>/`. A boot stuck for 15 minutes (e.g. `start.sh`
   died in preflight) counts as failed. **Disable it before you stop the server on purpose**
   (`sudo systemctl disable --now glm53-autorecover`), or it will bring the server back.

**Steps 1-3 do not survive a reboot.** Re-run `setup-node.sh` after every reboot.

## 2. Get the kit and apply the patches

```bash
git clone <this repo> && cd glm53-spark-recipe
./apply-kit-patches.sh --build
```

This clones the MiaAI-Lab kit at the tested commit (`493cb88`) and applies `kit-patches/exl3-kit.patch` to the
kit's own overlay files. It then lays this repo's files over it:

- **`exl3-kit/start.sh`:** this repo's launch script. Parallel node setup over one multiplexed ssh connection,
  the reclaim sequence, the boot guard, orphaned-shm cleanup, boot phases for the dashboard.
- **`exl3-kit/overlay/patch_*.py` + `shm_cleanup.py`:** the patchers. Each is idempotent and fails closed if the
  vLLM anchor it edits has drifted.
- **`exl3-kit/scripts/autorecover.sh`:** the recovery loop.
- **`nvfp4-vllm/` next to the kit:**
  - the NVFP4 KV runtime and patcher;
  - the 8-bit dense runtime (`glm53_dense_rt.py`);
  - the viz/maintenance runtime;
  - the sources of the three EXL3 MoE kernels (`exl3-mt/` prefill fallback, `exl3-fat/` prefill, `exl3-dec/`
    decode).

  `--build` compiles the three kernels with the kit image's nvcc (`docker run`, CPU only, ~2 min). `start.sh`
  finds everything there (`NVFP4_DIR` overrides). A kernel without its `.so` skips itself and the stock path runs.
- **`exl3-kit/.env`**, from `kit-patches/env.example`.

Edit `exl3-kit/.env`:

| key | set to |
|---|---|
| `HF_TOKEN` | your token |
| `HEAD_IP` / `WORKER_IP` | the 169.254.x.x link-local addresses of the CX7 link (`ip -br a` on each node) |
| `HEAD_CX7_IF` etc. | your NIC names, from `ip -br a` and `ls /sys/class/infiniband`. Both nodes here use `enp1s0f1np1` / `rocep1s0f1`; the kit's default wrongly assumes the worker uses f0 |
| `HEAD_GID` / `WORKER_GID` | each node's GID index whose entry is `::ffff:<that node's fabric IP>` with type `RoCE v2` (`/sys/class/infiniband/<dev>/ports/1/gids/*`, `gid_attrs/types/*`). They can differ between nodes and **can move when the peer reboots** (seen: 3 → 4); `start.sh`'s preflight then stops in seconds with `GID index N is EMPTY` |
| `HEAD_NCCL_HCA` / `WORKER_NCCL_HCA` | both CX7 functions, e.g. `rocep1s0f1,roceP2p1s0f1` (each function has its own subnet; check `ibv_devinfo` / `ip -br a`); one device if your second function is not up |

Everything else in `env.example` is the tested configuration. These values are load-bearing, so don't "clean them
up":

- **`EXTRA_ARGS="--kv-cache-memory 12000000000 ..."`** pins the pool. Without a pin the auto-sizer eats every
  byte down to the watchdog line and boots die. Section 5.1 shows what 12 GB leaves free; with the 8-bit weights
  there is room for ~0.9M more tokens if your nodes run nothing else.
- **`MAX_MODEL_LEN=900000`:** do NOT lower it "to save memory". Hybrid block-id overhead then *doubles* the
  per-token pool cost (measured 8.1 → 18.5 KB/token).
- **`SPEC_METHOD=mtp` with `MTP_TOKENS=3` and `MTP_DYNAMIC='[[1,2,3],[3,16,2]]'`:** 2.7-2.9 accepted tokens per
  step at 1-2 streams, 2.4-2.5 at 3. Section 7 has what was measured against it.
- **`GLM53_DENSE_W8=auto`, `GLM53_EXL3_DEC=1`, `GLM53_EXL3_FAT=1`:** the decode and prefill speed (sections 7, 8).
- **`GLM53_MEM_MAINT_S=15`:** the maintenance step now drains the GPU before it releases memory (section 5.2).
- **`GPU_MEM_UTIL=0.875`, `CG_ESTIMATE=1`, `GLM53_BOOT_SHAPE_WARMUP=0`, `EXL3_MOE_ROW_TILE=1`:** each guards a
  specific boot failure (section 11).

## 3. Weights

The kit downloads `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` (163.65 GiB) on the first `./start.sh`
and rsyncs it to the worker. If your internet is slow, run `./download.sh` first.

## 4. Launch

```bash
cd exl3-kit && ./start.sh        # ./start.sh restart to bounce a running pair
```

The first boot also pulls the image (21.8 GB) and ships it to the worker. Every later boot, measured from the
command to a healthy API (2026-10-03):

| +s | phase |
|---|---|
| 13 | old containers gone, orphaned shm cleaned, zram re-created (both nodes in parallel) |
| 24 | kit patches applied in both containers, `vllm serve` launched |
| 87 | 164 GB of weights streamed (37 s, InstantTensor Direct I/O at ~4.5 GB/s) |
| 89 | 203 non-expert matrices converted to 8 bits (1 s) and their Triton kernels warmed |
| 94 | draft (MTP) layer loaded (3 s) and its 6 matrices converted |
| 99 | KV cache allocated: `GPU KV cache size: 2,713,846 tokens, Maximum concurrency for 900,000 tokens per request: 3.02x` |
| 114 | CUDA graphs captured (9 s) |
| ~140 | `Application startup complete`, health check passed |

`curl localhost:8888/v1/models` answers with `glm-5.3-flash` (OpenAI-compatible API, vision and tool calling on).
Check that the boot log shows:

```text
[glm53-exl3-dec] grouped decode MoE on: windows <= 64 tokens x top-8, experts 288, ...
[glm53-dense] W8A16 (auto): 134 fp8-block + 69 int8-rowgroup matrices, 8.13 GB -> 4.15 GB this rank (1.1s) ...
[glm53-exl3-fat] prefill experts on glm53_exl3_fat for calls > 64 tokens (scratch for 2048 tokens)
```

Then verify the guard rails:
- `MemAvailable` ≥ 8 GiB on both nodes after boot
  (`awk '/MemAvailable/{print $2/1048576}' /proc/meminfo`);
- `/var/tmp/vllm-watchdog.log` shows `armed`.

The stock loader mmaps the safetensors at 150-300 MB/s (259 s for 164 GB). The kit image built with
`image/Dockerfile.instanttensor` streams them in ~37 s.
- **The catch:** the fast load applies no page-cache pressure, so nothing gets swapped, and graph capture would trip
  the watchdog. That is what the post-load reclaim exists for.
- **Stock behaviour:** `--load-format auto` plus the reclaim knobs at 0.

**If `start.sh` stops in preflight with `GID index N is EMPTY`, fix the GID in `.env`; the boot never started.**
The boot-state file still shows the previous phase, so read the tail of `start.sh`'s own output.

## 5. Memory on a Spark

### 5.1 Where the 121.6 GiB goes

The checkpoint is 163.6 GiB. Split over TP=2, with the replicated parts (embeddings, lm_head, vision tower), it is
83.5 GiB per node with BF16 non-expert matrices. With the 8-bit conversion (section 7) it is 4.0 GB less.

"So ~38 GB is free for KV" is the natural guess, and it is wrong by ~25 GB. The budget below is the head node at
steady state, measured on 2026-09-20 (12 GB pin, no desktop session) with the BF16 weights. It is reconciled to
`MemTotal - MemAvailable` using:
- `/proc/meminfo`;
- per-process RSS;
- zram;
- `nvidia-smi --query-compute-apps`, the driver's per-process GPU memory, which on unified memory is host RAM
  like everything else.

The 8-bit conversion moves 4.0 GB from the first line to the last; the head reads `MemAvailable` 12.7 GB right
after a boot on 2026-10-03.

| GB | what |
|---|---|
| 86.6 | model weights and torch workspaces (torch reserved minus the KV pool); **82.6 with `GLM53_DENSE_W8=auto`** |
| 11.4 | KV pool (`--kv-cache-memory 12000000000`) |
| 1.9 | GPU memory outside torch: CUDA context, NCCL/RoCE buffers, CUDA-graph exec objects, compiled kernels |
| 2.6 | driver-owned shared memory: appears at CUDA context creation, identical on both nodes, not releasable from user space |
| 2.2 | unreclaimable kernel slab (driver page tracking for ~100 GB of mappings) |
| 2.1 | vLLM processes' resident host memory (API server, engine core, worker, resource tracker) |
| 1.5 | zram store holding 5.3 GB of the same processes' cold pages (the reclaims' price) |
| 3.6 | other host processes: DGX OS daemons plus anything else you run. A GNOME desktop session with remote desktop is another ~2.6 GB. A co-tenant process costs its RSS **plus** the GPU memory the driver attributes to it |
| 1.9 | that co-tenant's GPU memory (one CUDA-using service here) |
| 0.3 | kernel stacks, page tables, vmalloc, per-cpu |
| 5.5 | file cache the kernel's watermark math does not count as available (`watermark_scale_factor=100`) |
| 4.8 | `MemAvailable`; **8.8 with `GLM53_DENSE_W8=auto`** |
| **121.6** | total visible to Linux |

The other 6.4 GiB of the 128 GB is gone before Linux boots and none of it is recoverable: 3.5 GiB of
firmware-reserved regions in the EFI map (GPU firmware and system carveout), 2.0 GiB of kernel page
bookkeeping (33.5M 4 KB pages × 64 B; a 64 KB-page kernel would shrink it 16× but DGX OS ships 4 KB),
0.35 GiB carved out before the memory map, 0.3 GiB of kernel image, initrd and CMA. The crash-kernel
reservation is already off (`crashkernel=1G-:0M`).

**What it costs:** the serving stack takes ~93 GB per node (BF16 weights: ~97) before a single KV token.

**Why the tighter node decides:** TP pins the same KV on both nodes, so the node with less free memory sets the
pool. Every GB you add to the pin must exist on both.

**Rules of thumb at this pin:**
- keep 3 GB free on the tighter node at steady state;
- each 0.45 GB of pin is ~100K tokens;
- a co-tenant that grows by N GB costs N GB of pin.

### 5.2 Guard rails

- **Boot:** `start.sh` reclaims memory from both containers at three points:
  - 4 GiB at the first `Loading weights took` line (`GLM53_POSTLOAD_RECLAIM=4`);
  - 3 GiB right after graph capture (`GLM53_PREAPI_RECLAIM=3`), because the API server's startup is the boot's peak;
  - optionally more once healthy (`GLM53_POSTREADY_RECLAIM`, now 0: with the 8-bit weights it is not needed, and on
    a node with a desktop session it ground for minutes).

  A boot guard polls both nodes every 2 s and reclaims 3 GiB on any node under `GLM53_BOOT_GUARD_MIB` (4500).
- **Runtime:** root's crontab on each node reclaims 2 GiB every minute while `MemAvailable` is under 2.5 GiB
  (`setup-node.sh`; the container name differs per node).
- **In-engine maintenance** (`nvfp4-vllm/glm53_viz_runtime.py`, every `GLM53_MEM_MAINT_S`=15 s on every rank):
  - logs `[glm53-mem]`: MemAvailable, process RSS/swap, torch allocated, the allocation peak since the last tick;
  - drains the GPU with `torch.cuda.synchronize()`;
  - returns caching-allocator slack with `torch.cuda.empty_cache()` outside graph capture;
  - runs `malloc_trim(0)` (the engine core does the same every 60 s).

  **The drain is load-bearing.** The containers run `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, under which
  `empty_cache()` unmaps freed pages at once; the unmap is not stream-ordered. Without the drain, the tick could
  unmap memory that kernels of the step in flight were still reading. That gave Xid 31 MMU read faults on both
  ranks, and once a `cuModuleGetFunction ... CUDA_ERROR_NOT_PERMITTED`, each landing exactly on a tick. It
  reproduced in 3 minutes with an agent-loop load and has been gone since the drain.

  Grep the head container log for `glm53-mem` to see the trend before the watchdog does.
- **Orphaned shared memory:** a watchdog kill never unlinks vLLM's `/dev/shm/psm_*` segments, and with
  `--ipc=host` they outlive the container (133 leftovers holding 690 MiB after a week of trips). `start.sh` runs
  `shm_cleanup.py` on both nodes at launch; it unlinks only segments no process maps.
- **Never run Python inside the serving containers.** A `docker exec ... python3` that imports vLLM costs ~1 GB,
  and the watchdog counts it; on a node with 1-2 GB of headroom that kills the server. Read files from the image
  with a throwaway `docker run --rm` container and shell tools.
- **GPU tests next to a live server need their own guard.** On unified memory a `docker run --memory` cap does not
  cover GPU allocations. A benchmark whose fp32 temporaries reached a few GB took a node under the watchdog line.
  - Before starting a test, check `MemAvailable` ≥ footprint + 3.5 GiB.
  - Run one test at a time.
  - Kill the test container if the node falls under 2.5 GiB.

### 5.3 What else was tried

| result | change |
|---|---|
| −4.3 GB | indexer K-gather workspace: upstream sizes it at `max_model_len × 40` entries (DeepSeek-V3.2's constant), 4.75 GB at 900K. With this model's 4:1 key pooling one full request needs 225K entries; `GLM53_INDEXER_PREFILL_MULT=1` (`patch_indexer_buffer.py`) makes it 119 MB and multi-request prefill steps simply chunk more |
| −0.45 GB | zram zstd instead of lzo-rle (section 1) |
| −0.2 GB | CUDA-graph exec objects for batch sizes 3, 24 and 48 (capture sizes `1 2 4 8 16 32 64`; those batches pad to the next size) |
| ~0 | `malloc_trim` in the worker: its 2.5 GB glibc heap after loading 150K tensors is live cold data, not free chunks (4 MiB released; the engine core gave back 219 MiB of swap). Kept because it is free |
| not cuttable | the 2.6 GB driver shared memory (CUDA context), the 2.2 GB slab (driver page tracking), NCCL's buffers (small for two ranks on one link), the vision tower and draft layer, and the KV bytes per token (the MLA latent is already NVFP4; FP4 indexer keys need sm_100) |

### 5.4 Put the head on the node without the desktop, and mind the desktop's GPU time

**The head is the heavier node.** The head (API server, engine core, resource tracker) costs ~2 GB of host RAM that
the worker does not. DGX OS runs a GNOME session, and a remote-desktop session on top of it, worth another ~2.6 GB.

**To move the head:** if one node is the one you sit at, make the other node the head:
1. swap `HEAD_IP` / `WORKER_IP` (and `HEAD_GID` / `WORKER_GID`) in its `.env`;
2. run `start.sh` there. It needs an ssh key into the new worker and the section 1 helpers on both nodes;
3. swap the container names in the two root crontabs;
4. keep clients on the old address with `host-setup/glm53-api-forward.service` (socat over the fabric link).

The dashboard stays where it was (section 10). Measured on this pair: each node has ~2 GB more headroom in its new
role.

**The desktop also costs GPU time.** An active remote-desktop client means the compositor, Xorg and the
remote-desktop daemon's capture and encode take time slices of that node's GPU (the daemon alone showed ~8% SM use
with the client idle). The same kernels then ran 30-44% slower on that rank. TP=2 runs in lockstep, so the whole
pair runs at the slower rank's pace, and the other rank spends the difference waiting in all-reduces. Disconnecting
the remote-desktop client when you are not using it buys that time back. Clocks are not the cause; both nodes sit at 2184-2190 MHz under load.

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

## 7. Decode

**Where the time went.** A torch-profiler trace of one stream (MTP k=3, i.e. 4-row verify windows plus three 1-row
draft steps) showed a 150 ms step on the old configuration:

| per step | before | after |
|---|---|---|
| BF16 non-expert matrices (attention / KDA projections, shared experts, dense MLP, LM head ×4): cuBLAS sm80 WMMA kernels at 130-190 GB/s | 75.8 ms | 27.5 ms (8-bit, Triton) + 6.5 ms (matrices kept BF16) |
| routed experts (`exl3_moe`, 48 CTAs, ~110-160 GB/s) | 55.4 ms | 29.6 ms (grouped kernel, ~200 GB/s) |
| all-reduces, hyper-connections, KDA recurrence, DSA attention, routing, elementwise | ~19 ms | ~15 ms |

"After" is the head rank's trace. Decode single stream went from 22.9 to 31.9 tok/s on an idle server (code,
temperature 1, short context), and from 20.7 to 31-36 tok/s at 78K context. Long context adds only ~2 ms of DSA
attention + indexer per step. The step reads ~11 GB a rank: ~5.8 GB of experts and ~5.3 GB of 8-bit dense
weights. At GB10's ~230 GB/s that is ~48 ms; the rest is all-reduces, small kernels and the slower rank
(section 5.4).

**8-bit dense weights** (`kit-patches/patch_dense_fp8.py`, `nvfp4-kv/glm53_dense_rt.py`, `GLM53_DENSE_W8`).

The EXL3 checkpoint quantizes only the routed experts. Every other matrix stays BF16, ~8.9 GB a rank read on every
step, with the LM head read four times (verify + 3 drafts). After load, each eligible matrix is converted in place,
one of two ways:

- **fp8 (134 matrices).** Matrices with ≤ 256 distinct values per 128×128 block came from GLM's official FP8
  release: shared experts, dense MLP, DSA/MLA q_a / q_b / kv_a / o, indexer wq_b. They go back to FP8 e4m3 on their
  own block grid. Relative error 0.15%, which is the BF16 rounding of the release itself.
- **int8 (69 matrices).** Native-BF16 matrices (KDA q/k/v/o, LM head, MTP eh_proj) get symmetric INT8 with one
  scale per row and 128 columns. Relative error 0.66%; FP8's 3-bit mantissa would cost 2.6% there.

The matmul keeps activations in BF16 and picks a kernel by batch size:

| M (rows) | kernel |
|---|---|
| ≤ 64 (every decode step) | GEMV-style Triton kernel, split-K for narrow matrices, 190-220 GB/s |
| 65-256 | tiled Triton GEMM |
| > 256 (prefill chunks) | dequantize into one shared 103 MB BF16 scratch, then cuBLAS |

- **Kept in BF16:** `kv_b_proj` (absorbed by MLA), the KDA conv weights, the indexer's fused wk, the MoE router and
  the vision tower.
- **Quality:** GSM8K-100 99/100 (BF16 98/100). Teacher-forced NLL on fixed code/prose/chat texts went from 1.6213
  to 1.633 (+0.7%); two identical BF16 runs differ by 0.004, because the server is not bit-deterministic.
- **Zero numeric change:** `GLM53_DENSE_W8=origin` converts only the FP8-origin matrices, at about half the gain.

**Grouped EXL3 decode kernel** (`kit-patches/patch_exl3_decode.py`, `nvfp4-kv/exl3-dec/`, `GLM53_EXL3_DEC`):
- TensorFold v0.6.0's grouped EXL3 GEMV, ported to vLLM's `exl3.py`, with jayleaton's 128-bit load ring.
- It reads each distinct routed expert of the window once, over the whole GPU: 228-237 GB/s over distinct-expert
  bytes, against 140-199 GB/s for `exl3_moe` in isolation. `exl3_moe` is slower again in the server.
- Deterministic, 2.5× closer to an fp64 reference than `exl3_moe`, CUDA-graph safe, +17 MiB a rank.
- Details and tables: `nvfp4-kv/exl3-dec/README.md`.

**Measured and not adopted:**

| tried | result |
|---|---|
| seeded probabilistic MTP drafts + block verification (`MTP_SPEC_EXTRA=',"draft_sample_method":"probabilistic","rejection_sample_method":"block"'`; this vLLM fork already implements TensorFold-style exact sampling) | on 12 fixed prompts at temperature 1, accepted tokens per step code 3.10 vs 3.16, prose 2.34 vs 2.22, reasoning 3.18 vs 3.21 against greedy drafts: no gain, kept greedy |
| MTP depth 4 for one stream (`[[1,1,4],[2,2,3],[3,16,2]]` + capture size 5) | +3-10% tokens per step, +10% step time: code 32.0 → 29.1 tok/s |
| a trimmed draft vocabulary (jayleaton's study) | +0.5-0.8% at one stream in their numbers; not tried |

## 8. Prefill

- **Fused EXL3 prefill kernels** (`kit-patches/patch_exl3_fat.py`, `nvfp4-kv/exl3-fat/`, `GLM53_EXL3_FAT`):
  jayleaton's "fast2" / "fat" grouped expert GEMMs, ported for MoE calls above 64 tokens.
  - **Speed:** 11.7 ms per MoE layer at a 2,048-token chunk, against 21.7 ms for the M-tiled kernel (1.85×); 1.6-2.0×
    from 256 tokens up.
  - **Numerics:** fp32 SwiGLU, a deterministic combine instead of atomics, 2× closer to fp64.
  - **Cost:** +40 MiB a rank net (a 160 MiB scratch replaces the M-tiled kernel's 120 MiB of temps).
  - **In the server:** cold prefill 797 / 823 / 661 → 1,026 / 1,105 / 1,082 tok/s at 9.4K / 32.7K / 77.8K tokens.
  - Details: `nvfp4-kv/exl3-fat/README.md`.
- **M-tiled EXL3 MoE kernel** (`nvfp4-kv/exl3-mt/`, `GLM53_EXL3_MT=1`), now the fallback:
  - The stock fused MoE kernel re-decodes the trellis weights every 16 rows. The M-tiled variants decode once per
    block: 75.1 → 38.0 ms per MoE layer on the real 288-expert routing (2.0×).
  - `nvfp4-kv/exl3-mt/README.md` has the kernel notes.
- **NCCL over both CX7 PCIe functions.**
  - A Spark's CX7 shows its cable as two RoCE devices (`rocep1s0f1`, `roceP2p1s0f1`), each with its own subnet.
    Listing both in `NCCL_IB_HCA`, with `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=4`, splits the prefill-size
    all-reduces (16 MB a layer at 2,048 tokens) across both links.
  - The idea comes from jayleaton's kit, which measured 4 MiB all-gathers 322 → 142 µs and prefill +1.5%.
  - Decode-size all-reduces are latency-bound and did not change.
- **Mixed prefill next to decoding streams** (`GLM53_MIXED_PREFILL_CHUNK=ladder`,
  `GLM53_MIXED_PREFILL_LADDER="1:1024,2:512,4:256,*:128"`). The kit's default `skip` starves a new
  prompt while anything decodes (minutes of dead TTFT); a fixed 128-token chunk bounded every step but
  held prefill to ~350 tok/s during overlap. A mixed step reads the whole expert set once whatever the
  chunk, so the ladder sizes the chunk by the number of decoding peers: 1024 next to one stream down to
  128 with five or more, keeping the gaps in those streams' output bounded. Section 12 has the measured
  prefill and decode rates during overlap.
- **Context-aware chunks** (`GLM53_PREFILL_CHUNK_CTX_BUDGET=3e8` token²). The sparse-MLA indexer scores
  every query of a chunk against the whole context; vLLM bounds the fp32 score tensor at
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (128 here, both ranks) by sub-chunking, but the per-step transient
  still grows with chunk × context. The budget caps the chunk: 2048 below 146K context, 896 at 300K, 512
  at 588K, 384 at 700K. Do not go below 3e8: an 8e7 budget held a 691K-token prompt to 128-token steps and
  prefill fell to ~230 tok/s, because the fixed per-step cost (one expert-set read plus the indexer pass)
  dominates below ~384-token chunks.

**Prefill by context depth** (one cold 321K-token prompt on an idle server, 2026-10-03; the chunk is what the
context-aware budget allows):

| context depth | chunk (context-aware budget) | prefill tok/s |
|---|---|---|
| 0-33K | 2048 | 1,188 |
| 33-66K | 2048 | 1,221 |
| 66-131K | 2048 | 1,192 |
| 131-197K | 2048 → 1408 | 1,150 |
| 197-262K | 1408 → 1024 | 1,070 |
| 262-321K | 1024 → 896 | 1,002 |

The whole 321,152-token prompt took 287 s to its first token (1,118 tok/s). Before the October kernels the
same server measured 797 / 823 / 661 tok/s at 9.4K / 32.7K / 77.8K.

## 9. Long multimodal sessions

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

## 10. Dashboard

`dashboard/` is a zero-dependency live dashboard:
- **`agent.py`** on each node (`:9101`): GPU, memory, CPU and network, plus any watched host processes. It also
  serves the local container state (`/docker?names=`) and the kit's boot-state file (`/bootstate?path=`), so the
  collector never has to ssh to the other node.
- **`collector.py`** on the dashboard node (`:9102`): polls vLLM metrics and the agents every 2.5 s, keeps 35 days
  of history in SQLite and receives the engine's activation telemetry.
- **`dash_server.py`** (`:3000`): serves the page and mirrors everything on it as JSON.

**Don't poll the other node over ssh every tick.** An earlier collector ran `docker -H ssh://` and `ssh cat` against
the head several times per 2.5 s tick, ~80 ssh logins a minute. Each login is a PAM/logind session, and
`polkitd`, `bluetoothd` and `wireplumber` grew by GBs over days until the head's memory watchdog fired during a big
prefill. The agent endpoints replaced that polling; ssh remains only as a fallback.

Run them as systemd units with `User=` your user; configuration is by environment:

| variable | meaning (default) |
|---|---|
| `GLM53_HEAD_SSH` | `user@<head fabric IP>` when the vLLM head is the other node: its container state and boot phase file then come from that node's agent (`SPARK_AGENT_W`), with ssh only as the fallback (empty: this node) |
| `GLM53_HEAD_KIT` | the kit directory on the head, for the boot phase file (`~/glm53/exl3-kit`) |
| `VLLM_METRICS_URL` | `http://localhost:8888/metrics` |
| `SPARK_AGENT_W` | the other node's agent, `http://<ip>:9101/stats` (`SPARK_AGENT_H` defaults to localhost) |
| `SPARK_NODE_LABELS` | role labels for the two node cards, this node first (`HEAD · API,WORKER`) |
| `SPARK_WATCH_PROCS` | agent: comma-separated command names to report, RSS + GPU memory (none) |
| `SPARK_PROC_CAP_MIB`, `SPARK_PROC_CAP_TOTAL_MIB` | flag the node card and `/api/procs` when the watched processes pass these (none) |
| `SPARK_DB_PATH`, `PORT`, `COLLECTOR` | history DB path (next to `collector.py`), page port (3000), collector URL |
| `SPARK_PROXY_ATTR_URL` | the optional attribution proxy's record feed (`http://127.0.0.1:8890/_proxy/attribution`) |

**Who sent each request (optional).**
- **The proxy:** `dashboard/api-proxy.py` + `api-proxy.service` (a template) sit in front of the API on the node
  clients connect to. A NAT redirect sends new connections on :8888 to the proxy, which forwards them unchanged.
- **What it records:** each generation request's client: its Tailscale user and machine (`tailscale whois`), or a
  name from `lan_names.json`. Never content.
- **On the page:** the collector matches those records to the scheduler's requests by arrival time, and the page
  labels each request with its sender.
- **If it stops:** the redirect is removed and clients fall back to the socat forwarder.

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

## 11. Troubleshooting — every failure actually hit

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
| `400: At most N image(s) may be provided in one prompt` | the multimodal limit | 800 images + `patch_mm_cap.py` (section 9) |
| server dies the moment you run a diagnostic inside the container | `docker exec ... python3` importing vLLM costs ~1 GB | read the image from a throwaway `docker run --rm` container; never exec Python in the serving container |
| 133 orphaned `/dev/shm/psm_*` segments, 690 MiB | watchdog kills never unlink vLLM's shared memory (`--ipc=host`) | `start.sh` runs `shm_cleanup.py` on both nodes at launch |
| head exits 0 with `RuntimeError: cancelled` after `CUDA error: an illegal memory access` in a CUDA-graph replay; kernel log `Xid 31 ... MMU Fault` | a kernel in the decode graph computed an unmapped address (once in 25 h, 1-token step at 170K context; GPU healthy after) | restart; `glm53_nvfp4_runtime.py` now clamps gathered pool rows to the pool's range; save the container log before `start.sh` removes it |
| worker dies at startup with `--mm-processor-cache-type shm` | it cannot open the head's shm segment across nodes | use `lru` |
| both ranks die within a second with `illegal memory access`, kernel log `Xid 31 ... MMU Fault ... FAULT_PTE ACCESS_TYPE_VIRT_READ` at a 1 GiB-aligned address; or `cuModuleGetFunction ... CUDA_ERROR_NOT_PERMITTED` | the 15 s in-engine maintenance ran `empty_cache()` (expandable segments unmap at once) while kernels of the step in flight still read the freed pages | fixed: `glm53_viz_runtime.py` drains the GPU (`torch.cuda.synchronize()`) before releasing memory (section 5.2) |
| `start.sh` stops in seconds: `head GID index N is EMPTY` | the RoCE GID table moved (a peer reboot moved the head's v2 entry 3 → 4) | set `HEAD_GID` / `WORKER_GID` to each node's `::ffff:<fabric IP>` RoCE v2 entry |
| decode well below the numbers here, one rank waits in all-reduces | a desktop / remote-desktop session takes GPU time slices on that node | section 5.4; disconnect the remote-desktop client or put the desktop elsewhere |
| head's `MemAvailable` erodes by GBs over days, `polkitd` / `bluetoothd` / `wireplumber` huge | something logs in over ssh many times a minute (an old dashboard collector) | the agent endpoints (section 10); restart those daemons |
| a GPU benchmark on a serving node kills the server | `docker run --memory` does not cap GPU (= host) memory on unified memory | section 5.2: guard GPU tests by `MemAvailable` |
| worker dies with `CUBLAS_STATUS_INTERNAL_ERROR` in a BF16 GEMM (the indexer's `wk_weights_proj`) during a deep cold prefill; no Xid, memory fine | not found. Seen once (2026-10-03, 238K tokens into a 321K-token prompt, on the rank whose node runs a desktop session); the same prompt then completed cleanly, and 5,425 never-used cuBLAS GEMM shapes loaded without error on that node | the auto-recovery service restarts the pair (~130 s) and keeps both containers' logs under `logs/crash-<time>/` |

## 12. Throughput, from the server's own history

The tables below are the dashboard's 2.5 s samples of real agent traffic, grouped by what the engine was doing in
that step:
- **Before:** 2026-09-11 (the M-tiled kernel went live) to 2026-09-20, BF16 dense weights and the stock decode
  kernel.
- **After:** 2026-10-01 to 2026-10-03, the current configuration. Windows with synthetic test load are excluded.

**Pure decode** (no prefill chunk in the step):

| decoding streams | samples before / after | aggregate tok/s before | **after** | per stream after | accepted tokens/step after |
|---|---|---|---|---|---|
| 1 | 75,684 / 747 | 22.7 | **34.1** | 34.1 | 2.71 |
| 2 | 11,628 / 334 | 35.6 | **53.5** | 26.8 | 2.88 |
| 3 | 3,741 / 118 | 43.6 | **61.5** | 20.5 | 2.45 |
| 4 | 2,504 / - | 41.5 | - | - | - |
| 5 | 226 / - | 48.8 | - | - | - |

Under 12-way GSM8K load the old configuration did 67.6 tok/s aggregate. At 4-way GSM8K load the current one does
51.2 tok/s aggregate, against 37.3 on the same benchmark before the kernel work.

**Mixed steps** (a prefill chunk scheduled next to decoding streams under the ladder policy):

| decoding streams | prefill tok/s before / after | decode tok/s (aggregate) before / after |
|---|---|---|
| 1 | 442 / 232 | 8.5 / **25.5** |
| 2 | 509 / 295 | 11.3 / **37.5** |
| 3 | 367 / 233 | 13.8 / **51.4** |

The two mixed columns are not like for like. Since October the traffic is mostly agent loops, where a mixed step
carries a short prompt tail (tens to hundreds of tokens) rather than a long prompt's 1024-token chunk. So the
prefill column is smaller, while the decoding streams keep 3× more speed.

**Pure prefill** (nothing decoding): 764 tok/s averaged over 6,561 samples before, at every depth. Section 8 has the
current depth table from a controlled cold prompt. The October history has only 175 pure-prefill samples (595 tok/s
mean, dominated by deep contexts).

Time to first token depends on whether the prefix cache hits (tens of milliseconds) or a cold prefill runs.

## Credits

- [MiaAI-Lab](https://github.com/MiaAI-Lab): the EXL3 kit this builds on.
- [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0 from 0.6.0, MIT before): the grouped EXL3 decode
  kernel and the fp64 EXL3 reference.
- [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache-2.0): the
  fast2/fat prefill expert kernels, the 128-bit load ring, the two-function NCCL setting, and its roofline and
  decode analyses of this model on this hardware.
- [turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3): EXL3/TR3.
- [neko-legends](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3): the uncensored 4bpw encode.
- zai-org: GLM-5.3-Flash.

Third-party code keeps its licence and notices: `nvfp4-kv/exl3-dec/LICENSES`, `nvfp4-kv/exl3-fat/LICENSES`, and the
headers of the ported files.
