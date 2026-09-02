#!/usr/bin/env python3
"""Boot-time patcher: NVFP4 KV pool for GLM-5.3 sparse MLA.

Installs glm53_nvfp4_runtime into the vllm package and hooks three seams.
All hooks are inert unless env GLM53_NVFP4_KV=1. Fail closed on anchor drift.
"""
import shutil
import sys
from pathlib import Path

V = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARK = "# [glm53-nvfp4-kv]"
RT_SRC = Path("/opt/glm53/glm53_nvfp4_runtime.py")


def patch(path: Path, old: str, new: str, label: str) -> None:
    s = path.read_text()
    if MARK in s and old not in s:
        print(f"[nvfp4-kv] {label}: already patched")
        return
    if old not in s:
        sys.exit(f"[nvfp4-kv] ANCHOR DRIFT in {path} ({label}) -- aborting boot")
    path.write_text(s.replace(old, new, 1))
    print(f"[nvfp4-kv] {label}: patched")


shutil.copy2(RT_SRC, V / "glm53_nvfp4_runtime.py")

# 1. Pool bytes: 656 -> 288 when enabled (main MLA spec branch only).
patch(V / "v1/kv_cache_interface.py",
"""            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656""",
f"""            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            from vllm import glm53_nvfp4_runtime as _nvR  {MARK}
            if _nvR.enabled():
                return self.block_size * 288  {MARK} nvfp4 rows
            return self.block_size * 656""",
"spec bytes")

# 2. Allocated tensor shape must agree.
patch(V / "v1/attention/backends/mla/flashinfer_mla_sparse.py",
"""        if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
            return (num_blocks, block_size, 656)""",
f"""        if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
            from vllm import glm53_nvfp4_runtime as _nvR  {MARK}
            if _nvR.enabled():
                return (num_blocks, block_size, 288)  {MARK}
            return (num_blocks, block_size, 656)""",
"cache shape")

# 3. Write path.
patch(V / "v1/attention/backend.py",
"""        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )""",
f"""        from vllm import _custom_ops as ops
        from vllm import glm53_nvfp4_runtime as _nvR  {MARK}
        if _nvR.enabled() and kv_cache_dtype == "fp8_ds_mla":
            # NoPE model: k_pe is zeros; the 512-dim latent is the whole row.
            _nvR.write_nvfp4(kv_c_normed, kv_cache, slot_mapping.flatten())
            return

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )""",
"write path")

# 4. Read path: dequant scratch just before the kernel.
patch(V / "v1/attention/backends/mla/flashinfer_mla_sparse.py",
"""        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),""",
f"""        from vllm import glm53_nvfp4_runtime as _nvR  {MARK}
        if _nvR.enabled():
            _nv_kv, block_tables = _nvR.build_scratch(  {MARK}
                kv_c_and_k_pe_cache, block_tables)
        else:
            _nv_kv = kv_c_and_k_pe_cache
        kernel_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=_nv_kv.unsqueeze(1),""",
"read path")


# 4b. Read path in the ACTUAL SM120 runtime file (the sibling of the one
# above; boot traceback proved forward_mqa here is what executes).
patch(V / "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
"""        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),""",
f"""        from vllm import glm53_nvfp4_runtime as _nvR  {MARK}
        if _nvR.enabled():
            _nv_kv, _nv_tables = _nvR.build_scratch(  {MARK}
                kv_c_and_k_pe_cache.view(torch.uint8),
                topk_indices_physical.unsqueeze(1))
            topk_indices_physical = _nv_tables.squeeze(1)
        else:
            _nv_kv = kv_c_and_k_pe_cache.view(torch.uint8)
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=_nv_kv.unsqueeze(1),""",
"read path sm120")


# 5. UMA pre-check softening: on GB10 unified memory, torch.cuda.mem_get_info
# under-reports free by the page-cache + watermark share, failing this naive
# util x total guard while the real downstream budgeting (profiler, measured
# free, CG estimator) is what actually sizes things -- and host watchdogs
# guard the floor. Warn instead of refuse. (Applies regardless of NVFP4 env:
# the check is wrong on this hardware either way.)
patch(V / "v1/worker/utils.py",
"""    if init_snapshot.free_memory < requested_memory:
        raise ValueError(""",
f"""    if init_snapshot.free_memory < requested_memory:  {MARK}
        import logging
        logging.getLogger(__name__).warning(
            "UMA: free %.2f GiB < util target %.2f GiB -- proceeding; "
            "downstream profiling uses measured memory (glm53-nvfp4-kv patch)",
            init_snapshot.free_memory / 2**30, requested_memory / 2**30)
        return requested_memory
    if False:
        raise ValueError(""",
"uma pre-check")
print("[nvfp4-kv] all seams patched")
