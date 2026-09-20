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

"""dequantize_and_gather_k_cache baseline（方案 B）。

native：DeepseekV4 分页 K cache 反量化 + gather 的调度入口。
    - benchmark 记载的公开入口是
      vllm.v1.attention.ops.deepseek_v4_ops.dequantize_and_gather_k_cache；
      本地 vllm 检出（76ba32160a）里 **不存在** 该模块。
    - 实际符号在
      vllm/models/deepseek_v4/common/ops/cache_utils.py#L390
      def dequantize_and_gather_k_cache(out, k_cache, seq_lens, gather_lens,
          block_table, block_size, offset, use_fnuz=False) -> None
      （亦经 common/ops/__init__.py 再导出）；内部按平台分派到
      dequantize_and_gather_k_cache_triton(#L342)（NV 上为 Triton kernel）。
      **原地**：结果写入调用方传入的 out 缓冲，返回 None。
    native() 先试公开路径，再回落 __init__ 再导出，最后回落 cache_utils；
    都取不到返回 None。

输入构造复刻
FlagGems-vllm/benchmark/
test_deepseek_v4_attention_dequantize_and_gather_k_cache.py 的
get_input_iter（dtype=bf16，需 fp8e4nv/sm89+）：
    shape = (batch, seq_len, gather_len, dim, nope_dim, rope_dim)
    scale_slots  = (nope_dim+63)//64 + (1 if nope_dim%64==0 else 0)
    block_size   = 64
    token_data_size = nope_dim + rope_dim*2
    block_stride = block_size*token_data_size + block_size*scale_slots
    num_blocks   = batch * ceil(seq_len/block_size)
    out       = empty(batch, gather_len, dim)          bf16
    k_cache   = zeros(num_blocks, block_stride)         uint8
    seq_lens  = full(batch, seq_len)                    int32
    gather_lens = full(batch, gather_len)               int32
    block_table = arange(num_blocks).view(batch, -1)    int32
benchmark 的 vllm adapter 仅把前 7 个位置参数
    (out, k_cache, seq_lens, gather_lens, block_table, block_size, offset=0)
传给 vllm 原生（rope_dim/nope_dim/scale_slots 只用于构造，不入原生调用），
本模块照此传参（use_fnuz 用默认 False）。
shape 取该 benchmark set_shapes 里的 5 组档位。
"""

import importlib

import torch

OP_NAME = "dequantize_and_gather_k_cache"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 结果原地写入 out 缓冲，返回 None

_BLOCK_SIZE = 64
_OFFSET = 0

_CANDIDATES = [
    ("vllm.v1.attention.ops.deepseek_v4_ops",
     "dequantize_and_gather_k_cache"),
    ("vllm.models.deepseek_v4.common.ops",
     "dequantize_and_gather_k_cache"),
    ("vllm.models.deepseek_v4.common.ops.cache_utils",
     "dequantize_and_gather_k_cache"),
]

# benchmark set_shapes：(batch, seq_len, gather_len, dim, nope_dim, rope_dim)
_SHAPES = [
    (1, 512, 128, 512, 448, 64),
    (2, 1024, 256, 512, 448, 64),
    (4, 2048, 512, 512, 448, 64),
    (4, 2048, 2048, 512, 448, 64),
    (8, 4096, 1024, 512, 448, 64),
]


def _scale_slots(nope_dim):
    return (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)


def _num_blocks(batch, seq_len):
    return batch * ((seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE)


def _block_stride(nope_dim, rope_dim):
    token_data_size = nope_dim + rope_dim * 2
    return _BLOCK_SIZE * token_data_size + _BLOCK_SIZE * _scale_slots(nope_dim)


def native():
    """依次尝试公开路径 / 包再导出 / 源码模块。"""
    for module, symbol in _CANDIDATES:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        op = getattr(mod, symbol, None)
        if callable(op):
            return op
    return None


def grid():
    return [
        {
            "batch": b,
            "seq_len": s,
            "gather_len": g,
            "dim": d,
            "nope_dim": nd,
            "rope_dim": rd,
        }
        for (b, s, g, d, nd, rd) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    b = binding["batch"]
    s = binding["seq_len"]
    g = binding["gather_len"]
    d = binding["dim"]
    nd = binding["nope_dim"]
    rd = binding["rope_dim"]
    block_stride = _block_stride(nd, rd)
    num_blocks = _num_blocks(b, s)
    out = torch.empty((b, g, d), dtype=torch.bfloat16, device=device)
    k_cache = torch.zeros((num_blocks, block_stride), dtype=torch.uint8,
                          device=device)
    seq_lens = torch.full((b,), s, dtype=torch.int32, device=device)
    gather_lens = torch.full((b,), g, dtype=torch.int32, device=device)
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).view(b, -1)
    # 仅传原生实际接收的 7 个位置参数（对齐 benchmark adapter）。
    args = (out, k_cache, seq_lens, gather_lens, block_table, _BLOCK_SIZE,
            _OFFSET)
    return args, {}


def key_shape(binding):
    return [
        binding["batch"],
        binding["seq_len"],
        binding["gather_len"],
        binding["dim"],
        binding["nope_dim"],
        binding["rope_dim"],
    ]


def config(binding, dtype):
    b = binding["batch"]
    s = binding["seq_len"]
    g = binding["gather_len"]
    d = binding["dim"]
    nd = binding["nope_dim"]
    rd = binding["rope_dim"]
    block_stride = _block_stride(nd, rd)
    num_blocks = _num_blocks(b, s)
    return {
        "inputs": {
            "out": {"shape": [b, g, d], "dtype": "torch.bfloat16",
                    "note": "原地写回"},
            "k_cache": {"shape": [num_blocks, block_stride],
                        "dtype": "torch.uint8"},
            "seq_lens": {"shape": [b], "dtype": "torch.int32"},
            "gather_lens": {"shape": [b], "dtype": "torch.int32"},
            "block_table": {"shape": [b, num_blocks // b], "dtype": "torch.int32"},
            "block_size": {"scalar": _BLOCK_SIZE},
            "offset": {"scalar": _OFFSET},
        },
        "outputs": {
            "out": {"shape": [b, g, d], "dtype": "torch.bfloat16"},
        },
        "dims": {
            "batch": b,
            "seq_len": s,
            "gather_len": g,
            "dim": d,
            "nope_dim": nd,
            "rope_dim": rd,
            "block_size": _BLOCK_SIZE,
            "scale_slots": _scale_slots(nd),
            "num_blocks": num_blocks,
            "block_stride": block_stride,
        },
    }
