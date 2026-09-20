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

"""flash_attn_varlen_func baseline（方案 B）。

native：vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func(
            q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
            cu_seqlens_k=None, seqused_k=None, q_v=None, dropout_p=0.0,
            softmax_scale=None, causal=False, window_size=None, softcap=0.0,
            alibi_slopes=None, deterministic=False, return_attn_probs=False,
            block_table=None, return_softmax_lse=False, out=None, ...)
    变长（varlen）paged FlashAttention 前向；这里走 paged 路径（k/v 为 KV cache
    分块，配合 block_table）。
    src: vllm/vllm_flash_attn/flash_attn_interface.py#L176

输入构造复刻 FlagGems-vllm/benchmark/test_flash_attn_varlen_func.py 的
flash_attn_varlen_input_fn（Qwen3-1.7B 采样档）：
    query      = randn(cu_query_lens[-1], num_query_heads, head_size)
    key_cache  = randn(num_blocks, block_size, num_kv_heads, head_size)
    value_cache= randn_like(key_cache)
    cu_query_lens = tensor(cu_seq_lens_q, int32)
    seqused_k     = tensor(seqused_k, int32)
    block_tables  = randint(0, num_blocks,
                            (num_seqs, ceil(max_kv_len/block_size)), int32)
    out           = empty_like(query)
    scale = head_size**-0.5, causal=True, window_size=(-1,-1)
调用位置参数顺序与 benchmark 完全一致（20 个位置参数 + 一组尾部 kwargs）。
num_heads=16, num_heads_k=8, head_dim=128, block_size=16, num_blocks=2000。
shape 网格取 benchmark set_shapes 里的 4 组 (cu_seq_lens_q, seqused_k) 采样档。
"""

import importlib

import torch

OP_NAME = "flash_attn_varlen_func"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False  # 主输出返回；out 张量同时被写（paged 路径）

_NUM_HEADS = 16
_NUM_HEADS_K = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 16
_NUM_BLOCKS = 2000

# 复刻 set_shapes：每档 (cu_seq_lens_q, seqused_k)。
_ALL_CU_SEQ_LENS_Q = [
    (0, 512),
    (0, 1, 2, 72),
    tuple(range(0, 45))
    + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
    tuple(range(0, 196)) + (211, 226, 240, 253, 265),
]
_ALL_SEQUSED_K = [
    (512,),
    (1, 1, 70),
    (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
    (2333,)
    + (2331,) * 20
    + (2330,) * 20
    + (2329,) * 14
    + (2328,) * 18
    + (2327,) * 15
    + (2326,) * 17
    + (2325,) * 18
    + (2324,) * 21
    + (2323,) * 22
    + (2322,) * 24
    + (2321,) * 5
    + (2320, 2319, 2318, 2317, 2316),
]


def native():
    """解析 flash_attn_varlen_func；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.vllm_flash_attn.flash_attn_interface"
        )
    except ImportError:
        return None
    op = getattr(mod, "flash_attn_varlen_func", None)
    return op if callable(op) else None


def grid():
    out = []
    for idx, (cu_q, seq_k) in enumerate(zip(_ALL_CU_SEQ_LENS_Q, _ALL_SEQUSED_K)):
        out.append({"shape_idx": idx})
    return out


def _config(binding):
    idx = binding["shape_idx"]
    return _ALL_CU_SEQ_LENS_Q[idx], _ALL_SEQUSED_K[idx]


def build_inputs(binding, dtype, device):
    cu_query_lens, seqused_k = _config(binding)

    num_seqs = len(cu_query_lens) - 1
    max_query_len = max(
        b - a for a, b in zip(cu_query_lens[:-1], cu_query_lens[1:])
    )
    max_kv_len = max(seqused_k)
    window_size = (-1, -1)
    scale = _HEAD_DIM ** -0.5

    query = torch.randn(
        cu_query_lens[-1], _NUM_HEADS, _HEAD_DIM, dtype=dtype, device=device
    )
    out = torch.empty_like(query)
    key_cache = torch.randn(
        _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS_K, _HEAD_DIM,
        dtype=dtype, device=device,
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens_t = torch.tensor(cu_query_lens, dtype=torch.int32, device=device)
    seqused_k_t = torch.tensor(seqused_k, dtype=torch.int32, device=device)

    max_num_blocks_per_seq = (max_kv_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    block_tables = torch.randint(
        0, _NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32, device=device,
    )

    # 位置参数顺序完全对齐 benchmark：
    #   q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k(None),
    #   seqused_k, q_v(None), dropout_p, softmax_scale, causal, window_size,
    #   softcap, alibi_slopes(None), deterministic, return_attn_probs,
    #   block_table, return_softmax_lse, out
    args = (
        query,
        key_cache,
        value_cache,
        max_query_len,
        cu_query_lens_t,
        max_kv_len,
        None,          # cu_seqlens_k
        seqused_k_t,
        None,          # q_v
        0.0,           # dropout_p
        scale,         # softmax_scale
        True,          # causal
        window_size,
        0,             # softcap (soft_cap=None -> 0)
        None,          # alibi_slopes
        False,         # deterministic
        False,         # return_attn_probs
        block_tables,
        False,         # return_softmax_lse
        out,
    )
    kwargs = {
        "scheduler_metadata": None,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "s_aux": None,
        "num_splits": 0,
        "cp_world_size": 1,
        "cp_rank": 0,
        "cp_tot_seqused_k": None,
        "fa_version": 2,
    }
    return args, kwargs


def key_shape(binding):
    cu_query_lens, seqused_k = _config(binding)
    num_seqs = len(cu_query_lens) - 1
    return [cu_query_lens[-1], num_seqs, _NUM_HEADS, _HEAD_DIM, max(seqused_k)]


def config(binding, dtype):
    cu_query_lens, seqused_k = _config(binding)
    num_seqs = len(cu_query_lens) - 1
    total_q = cu_query_lens[-1]
    max_kv_len = max(seqused_k)
    max_num_blocks_per_seq = (max_kv_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [total_q, _NUM_HEADS, _HEAD_DIM], "dtype": dt},
            "k": {"shape": [_NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS_K, _HEAD_DIM],
                  "dtype": dt, "note": "paged KV cache"},
            "v": {"shape": [_NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS_K, _HEAD_DIM],
                  "dtype": dt, "note": "paged KV cache"},
            "cu_seqlens_q": {"shape": [num_seqs + 1], "dtype": "torch.int32"},
            "seqused_k": {"shape": [num_seqs], "dtype": "torch.int32"},
            "block_table": {"shape": [num_seqs, max_num_blocks_per_seq],
                            "dtype": "torch.int32"},
            "max_seqlen_q": {"scalar": max(
                b - a for a, b in zip(cu_query_lens[:-1], cu_query_lens[1:])
            )},
            "max_seqlen_k": {"scalar": max_kv_len},
            "causal": {"scalar": True},
            "window_size": {"scalar": [-1, -1]},
        },
        "outputs": {
            "out": {"shape": [total_q, _NUM_HEADS, _HEAD_DIM], "dtype": dt},
        },
        "dims": {
            "total_q": total_q,
            "num_seqs": num_seqs,
            "num_heads": _NUM_HEADS,
            "num_heads_k": _NUM_HEADS_K,
            "head_dim": _HEAD_DIM,
            "block_size": _BLOCK_SIZE,
            "num_blocks": _NUM_BLOCKS,
            "max_kv_len": max_kv_len,
        },
    }
