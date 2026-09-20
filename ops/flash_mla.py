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

"""flash_mla baseline（方案 B）。

native：vLLM 侧 flash_mla 的实际入口是 FlashMLA 的 KV-cache 稠密解码算子
    flash_mla_with_kvcache，从 vllm.v1.attention.ops.flashmla 重导出（该模块在
    _is_flashmla_available() 为真时从 vllm.third_party.flashmla.flash_mla_interface
    导入 flash_mla_with_kvcache / get_mla_metadata，见 flashmla.py#L86-95）。
    CSV 调用方式：meta, _ = get_mla_metadata(); flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens, head_dim_v, meta, **kwargs)。
    这里 native() 返回一个薄封装：入参与 CUDA 调用一致，内部先算 get_mla_metadata
    再调 flash_mla_with_kvcache（对齐 vllm 后端 flashmla.py#L170 的
    num_q_tokens_per_head_k = max_query_len * num_q_heads、num_heads_k=1 用法）。
    third_party.flashmla 子模块未编译时上游会用 _raise_flashmla_unavailable 占位，
    import 成功但调用抛 RuntimeError；native() 只负责解析 callable，实机不可用由
    采集器 warmup 阶段暴露。

输入构造复刻 FlagGems-vllm/benchmark/test_flash_mla.py 的 flash_mla_kwargs：
    b=128, s_q=1, h_q=128, h_kv=1, d=576, dv=512, block_size=64, causal=True，
    cache_seqlens = [seqlen + 2*i for i in range(b)]，
    max_seqlen_pad = cdiv(max(cache_seqlens), 256) * 256，
    q          = randn([b, s_q, h_q, d])
    block_table= arange(b*max_seqlen_pad//block_size).view(b, max_seqlen_pad//block_size)
    blocked_k  = randn([block_table.numel(), block_size, h_kv, d])（即 k_cache）
shape 来源：FlashMLABenchmark（GenericBenchmark）默认 seqlen 档位
    [1024, 2048, 4096, 8192, 16384]，shape[0] 即 seqlen。
"""

import importlib

import torch

OP_NAME = "flash_mla"
DTYPES = [torch.bfloat16]  # 上游 FlashMLA KV-cache 稠密路径基准仅 bf16
IS_INPLACE = False

# 固定形参（来自 test_flash_mla.py flash_mla_kwargs）
_B = 128
_S_Q = 1
_H_Q = 128
_H_KV = 1
_D = 576
_DV = 512
_BLOCK_SIZE = 64

# GenericBenchmark 默认 seqlen 档位
_SEQLENS = [1024, 2048, 4096, 8192, 16384]


def _cdiv(a, b):
    return -(-a // b)


def native():
    """解析 vllm.v1.attention.ops.flashmla 的 KV-cache 稠密解码入口；不可用返回 None。"""
    try:
        mod = importlib.import_module("vllm.v1.attention.ops.flashmla")
    except ImportError:
        return None
    kvcache = getattr(mod, "flash_mla_with_kvcache", None)
    get_meta = getattr(mod, "get_mla_metadata", None)
    if not callable(kvcache) or not callable(get_meta):
        return None

    def _flash_mla(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
        # q: [b, s_q, h_q, d]；对齐 vllm 后端解码路径的 metadata 计算方式
        num_q_tokens_per_head_k = q.shape[1] * q.shape[2]
        meta, _ = get_meta(cache_seqlens, num_q_tokens_per_head_k, 1)
        return kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, meta, **kwargs)

    return _flash_mla


def grid():
    return [{"seqlen": s} for s in _SEQLENS]


def build_inputs(binding, dtype, device):
    seqlen = binding["seqlen"]
    b, s_q, h_q, h_kv, d, dv, bs = _B, _S_Q, _H_Q, _H_KV, _D, _DV, _BLOCK_SIZE
    max_seqlen = seqlen + 2 * (b - 1)
    max_seqlen_pad = _cdiv(max_seqlen, 256) * 256
    blocks_per_seq = max_seqlen_pad // bs
    num_blocks = b * blocks_per_seq

    q = torch.randn(b, s_q, h_q, d, dtype=dtype, device=device)
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).view(b, blocks_per_seq)
    k_cache = torch.randn(num_blocks, bs, h_kv, d, dtype=dtype, device=device)
    cache_seqlens = torch.tensor(
        [seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device
    )
    return (q, k_cache, block_table, cache_seqlens, dv), {"causal": True}


def key_shape(binding):
    return [_B, _S_Q, _H_Q, _D, binding["seqlen"]]


def config(binding, dtype):
    seqlen = binding["seqlen"]
    b, s_q, h_q, h_kv, d, dv, bs = _B, _S_Q, _H_Q, _H_KV, _D, _DV, _BLOCK_SIZE
    max_seqlen_pad = _cdiv(seqlen + 2 * (b - 1), 256) * 256
    num_blocks = b * (max_seqlen_pad // bs)
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [b, s_q, h_q, d], "dtype": dt},
            "k_cache": {"shape": [num_blocks, bs, h_kv, d], "dtype": dt},
            "block_table": {"shape": [b, max_seqlen_pad // bs], "dtype": "torch.int32"},
            "cache_seqlens": {"shape": [b], "dtype": "torch.int32"},
            "head_dim_v": dv,
        },
        "outputs": {
            "out": {"shape": [b, s_q, h_q, dv], "dtype": dt},
        },
        "dims": {
            "batch": b, "s_q": s_q, "h_q": h_q, "h_kv": h_kv,
            "d": d, "dv": dv, "block_size": bs, "seqlen": seqlen,
            "causal": True,
        },
    }
