# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""flash_mla_with_kvcache_fwd_w8a8_fp8 baseline（方案 B）。

native：vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache_fp8
    （src: vllm/v1/attention/ops/flashmla.py#L123；内部
    torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8）。
    签名（flashmla.py#L123-135）：
        flash_mla_with_kvcache_fp8(q, k_cache, block_table, cache_seqlens,
            head_dim_v, tile_scheduler_metadata, num_splits,
            softmax_scale=None, causal=False, descale_q=None, descale_k=None)
    tile_scheduler_metadata / num_splits 由
        get_mla_metadata_dense_fp8(cache_seqlens, num_q_tokens_per_head_k, 1)
    产出（flashmla.py#L109；对齐后端 flashmla.py#L177 用法）。native() 返回薄封装
    自动补这两个 metadata，其余参数与后端调用点 flashmla.py#L310 对齐
    （softmax_scale、causal=True、descale_q/descale_k）。

shape 为 vllm 源码推断，非 FlagGems-vllm 基准。
    benchmark 目录无 w8a8_fp8 变体（CSV 第 41 行标注「未找到对应benchmark」）。
    形状按后端解码路径调用点推断：
    - q: bf16 [batch, 1, h_q=128, d_qk=576]（kv_lora_rank 512 + qk_rope 64）；
    - k_cache: fp8_e4m3fn，unsqueeze 出 h_kv=1 头维 → [num_pages, page_block, 1, d_qk]；
      注：真实 fp8 KV cache 采 656 字节自定义编码（见 test_flash_mla_with_kvcache.py
      generate_v32_fp8_kv_cache），此处以 fp8 稠密张量近似，实机若要求专用编码需另调；
    - head_dim_v = kv_lora_rank = 512；
    - descale_q/descale_k: fp32 标量（layer._q_scale/_k_scale.reshape(1)）。
    seqlen 档位沿用稠密解码常见档 [256,512,1024,2048,4096]。
"""

import importlib

import torch

OP_NAME = "flash_mla_with_kvcache_fwd_w8a8_fp8"
DTYPES = [torch.bfloat16]  # q 为 bf16，KV cache 为 fp8
IS_INPLACE = False

_BATCH = 128
_H_Q = 128
_H_KV = 1
_D_QK = 576
_D_V = 512  # kv_lora_rank
_PAGE_BLOCK_SIZE = 64

_SEQLENS = [256, 512, 1024, 2048, 4096]


def _cdiv(a, b):
    return -(-a // b)


def native():
    """解析 flash_mla_with_kvcache_fp8 + get_mla_metadata_dense_fp8；不可用返回 None。"""
    try:
        mod = importlib.import_module("vllm.v1.attention.ops.flashmla")
    except ImportError:
        return None
    fp8_fn = getattr(mod, "flash_mla_with_kvcache_fp8", None)
    get_meta_fp8 = getattr(mod, "get_mla_metadata_dense_fp8", None)
    if not callable(fp8_fn) or not callable(get_meta_fp8):
        return None

    def _wrapper(
        q, k_cache, block_table, cache_seqlens, head_dim_v,
        descale_q=None, descale_k=None, softmax_scale=None, causal=True,
    ):
        num_q_tokens_per_head_k = q.shape[1] * q.shape[2]
        tile_meta, num_splits = get_meta_fp8(
            cache_seqlens, num_q_tokens_per_head_k, 1
        )
        return fp8_fn(
            q, k_cache, block_table, cache_seqlens, head_dim_v,
            tile_meta, num_splits,
            softmax_scale=softmax_scale, causal=causal,
            descale_q=descale_q, descale_k=descale_k,
        )

    return _wrapper


def grid():
    return [{"seqlen": s} for s in _SEQLENS]


def build_inputs(binding, dtype, device):
    seqlen = binding["seqlen"]
    batch, h_q, h_kv = _BATCH, _H_Q, _H_KV
    d_qk, d_v, pbs = _D_QK, _D_V, _PAGE_BLOCK_SIZE

    max_pages_per_seq = _cdiv(seqlen, pbs) + 4
    total_pages = batch * max_pages_per_seq

    q = torch.randn(batch, 1, h_q, d_qk, dtype=dtype, device=device) / 10
    # fp8 KV cache 近似：稠密 fp8_e4m3fn，head 维 =1（对齐后端 unsqueeze(-2)）
    k_cache = (
        torch.randn(total_pages, pbs, h_kv, d_qk, dtype=torch.float16, device=device)
        * 0.1
    ).to(torch.float8_e4m3fn)
    block_table = torch.arange(
        total_pages, dtype=torch.int32, device=device
    ).view(batch, max_pages_per_seq)
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    cache_seqlens[0] = max(seqlen // 2, 1)
    if batch > 1:
        cache_seqlens[-1] = min(seqlen + pbs, total_pages * pbs)
    descale_q = torch.ones(1, dtype=torch.float32, device=device)
    descale_k = torch.ones(1, dtype=torch.float32, device=device)
    return (q, k_cache, block_table, cache_seqlens, d_v), {
        "descale_q": descale_q,
        "descale_k": descale_k,
        "softmax_scale": d_qk ** (-0.5),
        "causal": True,
    }


def key_shape(binding):
    return [_BATCH, _H_Q, _D_QK, binding["seqlen"]]


def config(binding, dtype):
    seqlen = binding["seqlen"]
    batch, h_q, h_kv = _BATCH, _H_Q, _H_KV
    d_qk, d_v, pbs = _D_QK, _D_V, _PAGE_BLOCK_SIZE
    max_pages_per_seq = _cdiv(seqlen, pbs) + 4
    total_pages = batch * max_pages_per_seq
    return {
        "inputs": {
            "q": {"shape": [batch, 1, h_q, d_qk], "dtype": str(dtype)},
            "k_cache": {
                "shape": [total_pages, pbs, h_kv, d_qk],
                "dtype": "torch.float8_e4m3fn",
                "note": "fp8 稠密近似；真实为 656 字节自定义编码",
            },
            "block_table": {"shape": [batch, max_pages_per_seq], "dtype": "torch.int32"},
            "cache_seqlens": {"shape": [batch], "dtype": "torch.int32"},
            "head_dim_v": d_v,
            "descale_q": {"shape": [1], "dtype": "torch.float32"},
            "descale_k": {"shape": [1], "dtype": "torch.float32"},
        },
        "outputs": {
            "out": {"shape": [batch, 1, h_q, d_v], "dtype": str(dtype)},
        },
        "dims": {
            "batch": batch, "h_q": h_q, "h_kv": h_kv,
            "d_qk": d_qk, "d_v": d_v, "page_block_size": pbs,
            "seqlen": seqlen, "quant": "w8a8_fp8", "causal": True,
            "shape_source": "vllm 源码推断（无 benchmark）",
        },
    }
