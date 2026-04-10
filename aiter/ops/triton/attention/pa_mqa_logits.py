# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import math
from functools import lru_cache

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource as ASTSource

from aiter import dtypes
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits import (
    _deepgemm_fp8_paged_mqa_logits,
    _deepgemm_fp8_paged_mqa_logits_varctx_schedule,
    _deepgemm_fp8_paged_mqa_logits_ragged_k,
    _deepgemm_fp8_paged_mqa_logits_stage1,
    _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
)
from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version
from aiter.ops.triton.gluon.pa_mqa_logits import (
    _gluon_deepgemm_fp8_paged_mqa_logits,
    _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle,
    _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx,
)

enable_aot_gluon_pa_mqa_logits = (
    os.environ.get("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS", "0") == "1"
)
enable_gluon_pa_mqa_logits = True
enable_jit_gluon_pa_mqa_logits_kernel = not enable_aot_gluon_pa_mqa_logits


def deepgemm_fp8_paged_mqa_logits_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 64,
    SplitKV: int = 5,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": heads,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }

    grid = (batch_size * next_n * config["SplitKV"],)
    _deepgemm_fp8_paged_mqa_logits_ragged_k[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        **config,
    )


def deepgemm_fp8_paged_mqa_logits_stage1_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": 32,
        "ChunkK": 64,
        "HiddenDim": hidden_dim,
        "SplitKV": 5,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * config["SplitKV"]),)
    _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        **config,
    )


def deepgemm_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8 [num_blocks, 1, 1, D+4]
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkQ: int = 64,
    ChunkK: int = 256,
    TotalCuCount: int = 80,
    WavePerEU: int = 2,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, max_blk_len = kv_indices.size()

    TileQCount = batch_size * next_n * (heads // ChunkQ)
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": ChunkQ,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * SplitKV),)
    _deepgemm_fp8_paged_mqa_logits_stage1[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_blk_len,
        waves_per_eu=WavePerEU,
        **config,
    )


@lru_cache(maxsize=None)
def _compile_deepgemm_fp8_paged_mqa_logits(
    ChunkQ,
    ChunkK,
    Preshuffle,
    KVBlockSize,
    HiddenDim,
    is_padded_mode: bool,
    WavePerEU: int = 2,
    VarCtxOpt: bool = False,
):
    gfx_version = get_gfx()
    assert gfx_version == "gfx942" or gfx_version == "gfx950"
    cdna_version = get_cdna_version()
    target = GPUTarget("hip", gfx_version, 64)

    gfx_fp8_pointer = "*fp8e4b8" if gfx_version == "gfx942" else "*fp8e4nv"

    fn_signature = {
        "batch_size": "i32",
        "next_n": "i32",
        "heads_num": "i32",
        "Q_buffer": gfx_fp8_pointer,
        "stride_q_batch": "i32",
        "stride_q_next_n": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": gfx_fp8_pointer,
        "stride_k_seq": "i32",
        "scale_buffer": "*fp32",
        "stride_scale_seq": "i32",
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i32",
        "max_model_len": "i32",
        "max_block_len": "i32",
    }
    if VarCtxOpt:
        fn_signature["safe_chunks_per_cta_ptr"] = "*i32"
    else:
        fn_signature["SplitKV"] = "i32"

    fn_signature["ChunkQ"] = "constexpr"
    fn_signature["ChunkK"] = "constexpr"
    fn_signature["KVBlockSize"] = "constexpr"
    fn_signature["HiddenDim"] = "constexpr"
    fn_signature["CDNA_VERSION"] = "constexpr"

    options = {
        "num_warps": 4,
        "waves_per_eu": WavePerEU,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": gfx_version,
        "backend_name": "hip",
        "warp_size": 64,
        "name": (
            "_gluon_deepgemm_fp8_paged_mqa_logits"
            if not Preshuffle
            else (
                "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx"
                if VarCtxOpt
                else "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle"
            )
        ),
    }

    kv_cache_attr = []
    if is_padded_mode:
        kv_cache_attr.append(["tt.divisibility", 16])

    kernel_fn = (
        _gluon_deepgemm_fp8_paged_mqa_logits
        if not Preshuffle
        else (
            _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx
            if VarCtxOpt
            else _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle
        )
    )
    src = ASTSource(
        fn=kernel_fn,
        signature=fn_signature,
        constexprs={
            "ChunkQ": ChunkQ,
            "ChunkK": ChunkK,
            "KVBlockSize": KVBlockSize,
            "HiddenDim": HiddenDim,
            "CDNA_VERSION": cdna_version,
        },
        attrs={
            (2,): [["tt.divisibility", 16]],  # heads_num
            (3,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # Q_buffer
            (4,): [["tt.divisibility", 16]],  # stride_q_batch
            (5,): [["tt.divisibility", 16]],  # stride_q_next_n
            (6,): [["tt.divisibility", 16]],  # stride_q_heads
            (7,): kv_cache_attr,  # KV_buffer
            (8,): kv_cache_attr,  # stride_k_seq
            (9,): kv_cache_attr,  # scale_buffer
            (10,): kv_cache_attr,  # stride_scale_seq
            (11,): [["tt.pointer_range", 32]],  # context_len_ptr
            (12,): [["tt.pointer_range", 32]],  # kv_indices
            (13,): [
                ["tt.divisibility", 16],
                ["tt.pointer_range", 32],
            ],  # weights
            (14,): [["tt.divisibility", 16]],  # stride_w_batch
            (15,): [["tt.pointer_range", 32]],  # OutLogits_buffer
        },
    )

    if enable_jit_gluon_pa_mqa_logits_kernel:
        kernel = triton.compile(
            src,
            target=target,
            options=options,
        )
    else:
        padded_str = "T" if is_padded_mode and not Preshuffle else "F"
        preshuffle_suffix = "_preshuffle" if Preshuffle else ""
        varctx_suffix = "_varctx" if VarCtxOpt else ""
        kernel_str = f"paged_mqa_logits{preshuffle_suffix}{varctx_suffix}_{ChunkQ}x{ChunkK}x{HiddenDim}_B{KVBlockSize}P{padded_str}W{WavePerEU}"
        metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/paged_mqa_logits/aot/{kernel_str}"
        with AOTMetadataContext(
            kernel_fn.fn.__name__,
            metadata_pth,
        ):
            kernel = triton.compile(
                src,
                target=target,
                options=options,
            )
    return kernel


def deepgemm_fp8_paged_mqa_logits_schedule(
    batch_size,
    next_n,
    context_lens: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 256,
    TotalCuCount: int = 80 if get_gfx() == "gfx942" else 256,
    WavePerEU: int = 2,
):
    assert batch_size < TotalCuCount * WavePerEU // next_n

    max_chunks = math.ceil(max_model_len / ChunkK)
    schedule_waves_per_eu = 4
    grid = (TotalCuCount * schedule_waves_per_eu, 1, 1)
    TryCount = math.ceil(max_chunks / grid[0])
    align_power_of_2_batch = 1 << (batch_size - 1).bit_length()

    safe_chunks_per_cta = torch.empty(
        (1,),
        device="cuda",
        dtype=torch.int32,
    )
    _deepgemm_fp8_paged_mqa_logits_varctx_schedule[grid](
        batch_size,
        context_lens,
        safe_chunks_per_cta,
        TotalCuCount * WavePerEU // next_n,
        ChunkK,
        align_power_of_2_batch,
        TryCount,
        waves_per_eu=schedule_waves_per_eu,
    )
    return safe_chunks_per_cta


def deepgemm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache,
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool = False,
    KVBlockSize: int = 1,
    ChunkK: int = 256,
    TotalCuCount: int = 80 if get_gfx() == "gfx942" else 256,
    WavePerEU: int = 2,
    VarCtxSchedule: torch.Tensor = None,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    num_block, block_Size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    TileQCount = batch_size * next_n
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    assert ChunkK % KVBlockSize == 0 or KVBlockSize % ChunkK == 0
    assert block_Size == KVBlockSize
    if Preshuffle:
        assert (
            KVBlockSize % 16 == 0
        ), f"Preshuffle mode only supports KVBlockSize aligned to 16. Got KVBlockSize={KVBlockSize}"

    kv_cache = kv_cache.view(-1, KVBlockSize * index_dim)
    kv_cache_fp8, kv_cache_scale = (
        kv_cache[..., : KVBlockSize * hidden_dim],
        kv_cache[..., KVBlockSize * hidden_dim :],
    )
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)
    kv_cache_scale = kv_cache_scale.view(torch.float32)

    VarCtxOpt = VarCtxSchedule is not None
    if VarCtxOpt:
        grid = (TotalCuCount * WavePerEU, 1, 1)
    else:
        grid = (batch_size * next_n * SplitKV, 1, 1)

    if enable_gluon_pa_mqa_logits:
        is_padded_mode = kv_cache_fp8.stride(0) % 16 == 0
        kernel = _compile_deepgemm_fp8_paged_mqa_logits(
            ChunkQ=heads,
            ChunkK=ChunkK,
            Preshuffle=Preshuffle,
            KVBlockSize=KVBlockSize,
            HiddenDim=hidden_dim,
            is_padded_mode=is_padded_mode,
            WavePerEU=WavePerEU,
            VarCtxOpt=VarCtxOpt,
        )
        cdna_version = get_cdna_version()
        kernel[grid](
            batch_size,
            next_n,
            heads,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            kv_cache_fp8,
            kv_cache_fp8.stride(0),
            kv_cache_scale,
            kv_cache_scale.stride(0),
            context_lens,
            kv_indices,
            weights,
            weights.stride(0),
            out_logits,
            out_logits.stride(0),
            max_model_len,
            max_block_len,
            SplitKV if not VarCtxOpt else VarCtxSchedule,
            # constexpr
            heads,
            ChunkK,
            KVBlockSize,
            hidden_dim,
            cdna_version,
        )
    else:
        assert KVBlockSize == 1
        assert not Preshuffle, "Preshuffle mode is only supported on gluon kernel."
        kernel = _deepgemm_fp8_paged_mqa_logits[grid](
            batch_size,
            next_n,
            heads,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            kv_cache_fp8,
            kv_cache_fp8.stride(0),
            kv_cache_scale,
            kv_cache_scale.stride(0),
            context_lens,
            kv_indices,
            weights,
            weights.stride(0),
            out_logits,
            out_logits.stride(0),
            max_model_len,
            max_block_len,
            waves_per_eu=WavePerEU,
            ChunkQ=heads,
            ChunkK=ChunkK,
            SplitKV=SplitKV,
            HiddenDim=hidden_dim,
        )
    return triton.runtime.cache.get_cache_manager(kernel.hash).key
