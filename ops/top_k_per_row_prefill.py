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

"""top_k_per_row_prefill baseline（DeepSeek V4 稀疏注意力 prefill 阶段逐行 top-K）。

native：torch.ops._C.top_k_per_row_prefill(logits, rowStarts, rowEnds, indices,
        numRows, stride0, stride1, topK) -> ()。indices 原地写 top-k 索引。
    Python 封装见 vllm/_custom_ops.py#L3133；STABLE_TORCH_LIBRARY 绑定签名见
    csrc/libtorch_stable/torch_bindings.cpp#L559。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/test_top_k_per_row_prefill.py 的
TopKPerRowPrefillBenchmark.get_input_iter（全 vocab 区间，row_starts=0,
row_ends=vocab_size 的常见情形）：
    logits      = randn(num_rows, vocab_size) fp32
    row_starts  = zeros(num_rows) int32
    row_ends    = full((num_rows,), vocab_size) int32
    indices     = empty(num_rows, top_k) int32          （输出，原地写）
    num_rows, stride0, stride1, top_k 为标量
    stride0/stride1 取连续张量的 stride：contiguous 时 stride0=vocab_size,
    stride1=1（此处直接按维度算，不调 .stride() 以兼容离线冒烟 stub）。

shape 来自 benchmark set_shapes：DeepSeek V4 生产配置 vocab_size=129280,
top_k=1024，num_rows ∈ {1(decode 单 token), 32(典型 prefill 微批),
64(较大批), 2048(最大序列)}。
"""

import importlib

import torch

OP_NAME = "top_k_per_row_prefill"
DTYPES = [torch.float32]
IS_INPLACE = True  # indices 原地写

# benchmark set_shapes：(num_rows, vocab_size, top_k)
_SHAPES = [
    (1, 129280, 1024),
    (32, 129280, 1024),
    (64, 129280, 1024),
    (2048, 129280, 1024),
]


def native():
    """解析 torch.ops._C.top_k_per_row_prefill；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "top_k_per_row_prefill", None)
    return op if callable(op) else None


def grid():
    return [{"num_rows": r, "vocab_size": v, "top_k": k}
            for (r, v, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    r, v, k = binding["num_rows"], binding["vocab_size"], binding["top_k"]
    logits = torch.randn(r, v, dtype=torch.float32, device=device)
    row_starts = torch.zeros(r, dtype=torch.int32, device=device)
    row_ends = torch.full((r,), v, dtype=torch.int32, device=device)
    indices = torch.empty((r, k), dtype=torch.int32, device=device)
    # contiguous logits：stride0=vocab_size, stride1=1
    stride0, stride1 = v, 1
    return (logits, row_starts, row_ends, indices, r, stride0, stride1, k), {}


def key_shape(binding):
    return [binding["num_rows"], binding["vocab_size"], binding["top_k"]]


def config(binding, dtype):
    r, v, k = binding["num_rows"], binding["vocab_size"], binding["top_k"]
    return {
        "inputs": {
            "logits": {"shape": [r, v], "dtype": "torch.float32"},
            "row_starts": {"shape": [r], "dtype": "torch.int32",
                           "note": "每行有效区间起点，全 vocab 时为 0"},
            "row_ends": {"shape": [r], "dtype": "torch.int32",
                         "note": "每行有效区间终点，全 vocab 时为 vocab_size"},
            "indices": {"shape": [r, k], "dtype": "torch.int32",
                        "note": "原地写回 top-k 索引"},
            "num_rows": {"scalar": r},
            "stride0": {"scalar": v},
            "stride1": {"scalar": 1},
            "top_k": {"scalar": k},
        },
        "outputs": {
            "indices": {"shape": [r, k], "dtype": "torch.int32"},
        },
        "dims": {"num_rows": r, "vocab_size": v, "top_k": k},
    }
