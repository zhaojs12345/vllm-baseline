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

"""top_k_per_row_decode baseline（DeepSeek V4 稀疏注意力 decode 阶段逐行 top-K）。

native：torch.ops._C.top_k_per_row_decode(logits, next_n, seq_lens, indices,
        numRows, stride0, stride1, topK) -> ()。indices 原地写 top-k 索引。
    Python 封装见 vllm/_custom_ops.py#L3155；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L563。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/test_top_k_per_row_decode.py 的
TopKPerRowDecodeBenchmark.get_input_iter（num_rows=1, next_n=1 的 decode 路径）：
    logits    = randn(1, vocab_size) fp32
    next_n    = 1（标量）
    seq_lens  = tensor([vocab_size]) int32
    indices   = zeros(1, top_k) int32                   （输出，原地写）
    num_rows=1, stride0=logits.stride(0), stride1=logits.stride(1), top_k
    contiguous 时 stride0=vocab_size, stride1=1（此处按维度算，兼容离线 stub）。

shape 来自 benchmark set_shapes：(vocab_size, top_k) 档位——
    (129280,1024) DeepSeek V4 生产配置、(32768,512)、(16384,256)、
    (8192,128)、(4096,64)。num_rows 与 next_n 固定为 1。
"""

import importlib

import torch

OP_NAME = "top_k_per_row_decode"
DTYPES = [torch.float32]
IS_INPLACE = True  # indices 原地写

# benchmark set_shapes：(vocab_size, top_k)
_SHAPES = [
    (129280, 1024),
    (32768, 512),
    (16384, 256),
    (8192, 128),
    (4096, 64),
]


def native():
    """解析 torch.ops._C.top_k_per_row_decode；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "top_k_per_row_decode", None)
    return op if callable(op) else None


def grid():
    return [{"vocab_size": v, "top_k": k} for (v, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    v, k = binding["vocab_size"], binding["top_k"]
    logits = torch.randn(1, v, dtype=torch.float32, device=device)
    next_n = 1
    seq_lens = torch.tensor([v], dtype=torch.int32, device=device)
    indices = torch.zeros((1, k), dtype=torch.int32, device=device)
    num_rows = 1
    # contiguous logits：stride0=vocab_size, stride1=1
    stride0, stride1 = v, 1
    return (logits, next_n, seq_lens, indices, num_rows, stride0, stride1, k), {}


def key_shape(binding):
    return [1, binding["vocab_size"], binding["top_k"]]


def config(binding, dtype):
    v, k = binding["vocab_size"], binding["top_k"]
    return {
        "inputs": {
            "logits": {"shape": [1, v], "dtype": "torch.float32"},
            "next_n": {"scalar": 1},
            "seq_lens": {"shape": [1], "dtype": "torch.int32",
                         "note": "每行有效序列长度，此处 = vocab_size"},
            "indices": {"shape": [1, k], "dtype": "torch.int32",
                        "note": "原地写回 top-k 索引"},
            "num_rows": {"scalar": 1},
            "stride0": {"scalar": v},
            "stride1": {"scalar": 1},
            "top_k": {"scalar": k},
        },
        "outputs": {
            "indices": {"shape": [1, k], "dtype": "torch.int32"},
        },
        "dims": {"num_rows": 1, "next_n": 1, "vocab_size": v, "top_k": k},
    }
