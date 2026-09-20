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

"""persistent_topk baseline（方案 B）。

native：torch.ops._C.persistent_topk(logits, lengths, output, workspace, k,
        max_seq_len) -> ()。output 原地写 top-k 索引。
    绑定签名见 csrc/libtorch_stable/torch_bindings.cpp#L569。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/test_persistent_topk.py 的
PersistentTopKBenchmark.get_input_iter（K=512, STRIDE=262144）：
    logits = full((num_rows, STRIDE), -inf, fp32); logits[:, :seq_len]=randn(...)
    lengths = full((num_rows,), seq_len, int32)
    indices = empty((num_rows, K), int32)                 输出
    workspace = empty(1024*1024, uint8)
    native 调用：persistent_topk(logits, lengths, indices, workspace, K, max_seq_len)

shape 取 benchmark set_shapes 里 decode 路径的档位（未启用被注释掉的 medium 档，
以及 num_rows=1..32 的 262144 大档——那批单个 logits 张量达数百 MB，采集时按需用
--ops/--blacklist 控制）。这里默认只放 decode 档，避免默认全跑时显存压力过大。
"""

import importlib

import torch

OP_NAME = "persistent_topk"
DTYPES = [torch.float32]
IS_INPLACE = True  # indices 原地写

_K = 512
_STRIDE = 262144

# benchmark 的 decode 路径 shapes：(num_rows, seq_len, max_seq_len)。
# 大档 (nr, 262144, 1048576) for nr in 1..32 等单张量数百 MB，未纳入默认网格；
# 需要时可在此追加，或用 --ops 单独跑。
_SHAPES = [
    (1, 4102, 4102),
    (4, 4102, 4102),
    (10, 1055, 1055),
    (12, 4105, 4105),
    (20, 4105, 4105),
    (28, 4109, 4109),
    (1, 8192, 8192),
    (1, 32773, 32773),
]


def native():
    """解析 torch.ops._C.persistent_topk；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "persistent_topk", None)
    return op if callable(op) else None


def grid():
    return [{"num_rows": r, "seq_len": s, "max_seq_len": m}
            for (r, s, m) in _SHAPES]


def build_inputs(binding, dtype, device):
    r, s, m = binding["num_rows"], binding["seq_len"], binding["max_seq_len"]
    logits = torch.full((r, _STRIDE), float("-inf"),
                        dtype=torch.float32, device=device)
    logits[:, :s] = torch.randn(r, s, device=device)
    lengths = torch.full((r,), s, dtype=torch.int32, device=device)
    indices = torch.empty((r, _K), dtype=torch.int32, device=device)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    return (logits, lengths, indices, workspace, _K, m), {}


def key_shape(binding):
    return [binding["num_rows"], binding["seq_len"], binding["max_seq_len"]]


def config(binding, dtype):
    r, s, m = binding["num_rows"], binding["seq_len"], binding["max_seq_len"]
    return {
        "inputs": {
            "logits": {"shape": [r, _STRIDE], "dtype": "torch.float32",
                       "note": f"前 seq_len={s} 有效，其余 -inf"},
            "lengths": {"shape": [r], "dtype": "torch.int32"},
            "output": {"shape": [r, _K], "dtype": "torch.int32",
                       "note": "原地写回（top-k 索引）"},
            "workspace": {"shape": [1024 * 1024], "dtype": "torch.uint8"},
            "k": {"scalar": _K},
            "max_seq_len": {"scalar": m},
        },
        "outputs": {
            "output": {"shape": [r, _K], "dtype": "torch.int32"},
        },
        "dims": {"num_rows": r, "seq_len": s, "max_seq_len": m,
                 "K": _K, "STRIDE": _STRIDE},
    }
