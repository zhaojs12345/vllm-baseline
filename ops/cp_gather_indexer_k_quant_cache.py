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

"""cp_gather_indexer_k_quant_cache baseline（从 paged FP8 indexer cache 收集 K 与 scale）。

native：vllm._custom_ops.cp_gather_indexer_k_quant_cache(kv_cache, dst_k,
        dst_scale, block_table, cu_seq_lens) -> ()。按 block_table/cu_seq_lens
        把 paged cache 里的量化 K 与 scale 收集（gather）到连续的 dst_k/dst_scale。
    Python 封装见 vllm/_custom_ops.py#L3177（内部转 torch.ops._C_cache_ops.
    cp_gather_indexer_k_quant_cache）；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L1016。

输入构造复刻 FlagGems-vllm/benchmark/test_cp_gather_indexer_k_quant_cache.py 的
CpGatherIndexerKQuantCacheBenchmark.get_input_iter + make_gather_metadata：
    seq_lens     = full((batch_size,), seq_len) int32
    cu_seq_lens  = zeros(batch_size+1) int32; cu_seq_lens[1:]=cumsum(seq_lens)
    blocks_per_seq = ceil(seq_len/block_size)
    block_table  = arange(batch_size*blocks_per_seq).view(bs, blocks_per_seq) int32
    num_blocks   = block_table.numel(); num_tokens = batch_size*seq_len
    cache_stride = head_dim + head_dim*4 // quant_block_size
    kv_cache     = empty(num_blocks, block_size, cache_stride) uint8（预填 FP8 值）
    dst_k        = empty(num_tokens, head_dim) fp8_e4m3
    dst_scale    = empty(num_tokens, head_dim*4 // quant_block_size) uint8
    调用顺序：(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)

    注：dst_k 用 float8_e4m3fn（benchmark 里 fp8_dtype，随设备取 e4m3fn/e4m3fnuz）；
    冒烟 stub 只看 shape/dtype，不做真实量化填充（fill_cache_with_valid_fp8 略）。

shape 来自 benchmark set_shapes：
    (batch_size, seq_len, block_size, head_dim, quant_block_size) =
    (4,256,16,128,128)/(8,512,16,128,128)/(16,1024,16,512,128)/
    (32,1024,16,512,128)。DTYPES 取 fp16（与 benchmark dtypes 一致，仅用于命名；
    实际 dst_k 为 fp8_e4m3fn）。
"""

import importlib
import math

import torch

OP_NAME = "cp_gather_indexer_k_quant_cache"
DTYPES = [torch.float16]
IS_INPLACE = True  # dst_k / dst_scale 原地写

# benchmark set_shapes：(batch_size, seq_len, block_size, head_dim, quant_block_size)
_SHAPES = [
    (4, 256, 16, 128, 128),
    (8, 512, 16, 128, 128),
    (16, 1024, 16, 512, 128),
    (32, 1024, 16, 512, 128),
]


def native():
    """解析 vllm._custom_ops.cp_gather_indexer_k_quant_cache；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "cp_gather_indexer_k_quant_cache", None)
    return op if callable(op) else None


def grid():
    return [{"batch_size": b, "seq_len": s, "block_size": bs,
             "head_dim": hd, "quant_block_size": qbs}
            for (b, s, bs, hd, qbs) in _SHAPES]


def build_inputs(binding, dtype, device):
    b = binding["batch_size"]
    s = binding["seq_len"]
    bs = binding["block_size"]
    hd = binding["head_dim"]
    qbs = binding["quant_block_size"]

    seq_lens = torch.full((b,), s, dtype=torch.int32, device=device)
    cu_seq_lens = torch.zeros(b + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

    blocks_per_seq = math.ceil(s / bs)
    block_table = torch.arange(
        b * blocks_per_seq, dtype=torch.int32, device=device
    ).view(b, blocks_per_seq)

    num_blocks = b * blocks_per_seq  # = block_table.numel()
    num_tokens = b * s
    cache_stride = hd + hd * 4 // qbs
    kv_cache = torch.empty(num_blocks, bs, cache_stride,
                           dtype=torch.uint8, device=device)
    dst_k = torch.empty(num_tokens, hd, dtype=torch.float8_e4m3fn, device=device)
    dst_scale = torch.empty(num_tokens, hd * 4 // qbs,
                            dtype=torch.uint8, device=device)
    return (kv_cache, dst_k, dst_scale, block_table, cu_seq_lens), {}


def key_shape(binding):
    return [binding["batch_size"], binding["seq_len"],
            binding["block_size"], binding["head_dim"],
            binding["quant_block_size"]]


def config(binding, dtype):
    b = binding["batch_size"]
    s = binding["seq_len"]
    bs = binding["block_size"]
    hd = binding["head_dim"]
    qbs = binding["quant_block_size"]
    blocks_per_seq = math.ceil(s / bs)
    num_blocks = b * blocks_per_seq
    num_tokens = b * s
    cache_stride = hd + hd * 4 // qbs
    return {
        "inputs": {
            "kv_cache": {"shape": [num_blocks, bs, cache_stride],
                         "dtype": "torch.uint8", "note": "paged FP8 K + scale 源"},
            "dst_k": {"shape": [num_tokens, hd], "dtype": "torch.float8_e4m3fn",
                      "note": "原地写回收集的 K"},
            "dst_scale": {"shape": [num_tokens, hd * 4 // qbs],
                          "dtype": "torch.uint8", "note": "原地写回收集的 scale"},
            "block_table": {"shape": [b, blocks_per_seq], "dtype": "torch.int32"},
            "cu_seq_lens": {"shape": [b + 1], "dtype": "torch.int32"},
        },
        "outputs": {
            "dst_k": {"shape": [num_tokens, hd], "dtype": "torch.float8_e4m3fn"},
            "dst_scale": {"shape": [num_tokens, hd * 4 // qbs],
                          "dtype": "torch.uint8"},
        },
        "dims": {"batch_size": b, "seq_len": s, "block_size": bs,
                 "head_dim": hd, "quant_block_size": qbs,
                 "num_blocks": num_blocks, "num_tokens": num_tokens,
                 "cache_stride": cache_stride},
    }
