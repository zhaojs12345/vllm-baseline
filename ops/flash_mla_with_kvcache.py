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

"""flash_mla_with_kvcache baseline（方案 B）。

native：vllm.third_party.flashmla.flash_mla_interface.flash_mla_with_kvcache，
    vLLM 侧从 vllm.v1.attention.ops.flashmla 重导出（flashmla.py#L86-95），
    优先走 vllm.v1.attention.ops.flashmla 解析，退回 third_party。
    CSV 调用方式：meta, _ = get_mla_metadata();
        flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens,
                               head_dim_v, meta, **kwargs)。
    native() 返回薄封装，内部先算 get_mla_metadata 再调 flash_mla_with_kvcache，
    对齐 test_flash_mla_with_kvcache.py 的 _cuda_wrapper。

输入构造复刻 FlagGems-vllm/benchmark/test_flash_mla_with_kvcache.py 的
    make_input()「Dense BF16 decode」分支（topk=0，非 FP8）：
    batch=128, h_q=128, h_kv=1, d_qk=576, d_v=512, page_block_size=64,
    q          = randn(batch, 1, h_q, d_qk) / 10
    max_pages_per_seq = cdiv(seqlen, page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    k_cache    = randn(total_pages, page_block_size, h_kv, d_qk) * 0.1
    block_table= arange(total_pages).view(batch, max_pages_per_seq)
    cache_seqlens = full((batch,), seqlen)，首/尾略调
    kwargs     = {"causal": True}
    只复刻稠密 bf16 路径；FP8 稀疏路径（is_fp8_kvcache/indices/attn_sink 等）
    依赖 656/584 字节自定义 KV cache 编码，交给 fp8 变体算子，不在此基线内。
shape 来源：test_flash_mla_with_kvcache.py Dense 档位 seqlen [256,512,1024,2048,4096]。
"""

import importlib

import torch

OP_NAME = "flash_mla_with_kvcache"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_BATCH = 128
_H_Q = 128
_H_KV = 1
_D_QK = 576
_D_V = 512
_PAGE_BLOCK_SIZE = 64

# Dense BF16 decode 档位（test_flash_mla_with_kvcache.py）
_SEQLENS = [256, 512, 1024, 2048, 4096]


def _cdiv(a, b):
    return -(-a // b)


def native():
    """解析 flash_mla_with_kvcache（先 v1.attention.ops.flashmla，退回 third_party）。"""
    kvcache = None
    get_meta = None
    for path in (
        "vllm.v1.attention.ops.flashmla",
        "vllm.third_party.flashmla.flash_mla_interface",
    ):
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        kvcache = getattr(mod, "flash_mla_with_kvcache", None)
        get_meta = getattr(mod, "get_mla_metadata", None)
        if callable(kvcache) and callable(get_meta):
            break
    if not callable(kvcache) or not callable(get_meta):
        return None

    def _wrapper(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
        num_q_tokens_per_head_k = q.shape[1] * q.shape[2]
        meta, _ = get_meta(cache_seqlens, num_q_tokens_per_head_k, 1)
        return kvcache(
            q, k_cache, block_table, cache_seqlens, head_dim_v, meta, **kwargs
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
    k_cache = (
        torch.randn(total_pages, pbs, h_kv, d_qk, dtype=dtype, device=device) * 0.1
    )
    block_table = torch.arange(
        total_pages, dtype=torch.int32, device=device
    ).view(batch, max_pages_per_seq)
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    # 复刻 benchmark：首序列减半、末序列略增（可重复、无外部状态）
    cache_seqlens[0] = max(seqlen // 2, 1)
    if batch > 1:
        cache_seqlens[-1] = min(seqlen + pbs, total_pages * pbs)
    return (q, k_cache, block_table, cache_seqlens, d_v), {"causal": True}


def key_shape(binding):
    return [_BATCH, _H_Q, _D_QK, binding["seqlen"]]


def config(binding, dtype):
    seqlen = binding["seqlen"]
    batch, h_q, h_kv = _BATCH, _H_Q, _H_KV
    d_qk, d_v, pbs = _D_QK, _D_V, _PAGE_BLOCK_SIZE
    max_pages_per_seq = _cdiv(seqlen, pbs) + 4
    total_pages = batch * max_pages_per_seq
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [batch, 1, h_q, d_qk], "dtype": dt},
            "k_cache": {"shape": [total_pages, pbs, h_kv, d_qk], "dtype": dt},
            "block_table": {"shape": [batch, max_pages_per_seq], "dtype": "torch.int32"},
            "cache_seqlens": {"shape": [batch], "dtype": "torch.int32"},
            "head_dim_v": d_v,
        },
        "outputs": {
            "out": {"shape": [batch, 1, h_q, d_v], "dtype": dt},
        },
        "dims": {
            "batch": batch, "h_q": h_q, "h_kv": h_kv,
            "d_qk": d_qk, "d_v": d_v, "page_block_size": pbs,
            "seqlen": seqlen, "path": "dense_bf16_decode", "causal": True,
        },
    }
