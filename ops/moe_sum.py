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

"""moe_sum baseline（方案 B）。

native：vllm._custom_ops.moe_sum(input, output, topk_ids=None, expert_map=None)
    内部调用 torch.ops._moe_C.moe_sum；把 [num_tokens, topk, hidden] 沿 topk 维
    求和写入 output[num_tokens, hidden]。原地写 output，返回 None。
    src: vllm/_custom_ops.py#L2309

输入构造复刻 FlagGems-vllm/benchmark/test_moe_sum.py 的 _input_fn：
    input  = randn(num_tokens, topk, hidden)
    output = empty(num_tokens, hidden)
benchmark 用 GenericBenchmarkExcluse1D + 默认 shapes（consts.DEFAULT_SHAPES 里的
2D/3D 项），2D 项 (M, N) 会被 _input_fn 补成 (M, 1, N)。这里直接给出等价的
(num_tokens, topk, hidden) 网格。
"""

import importlib

import torch

OP_NAME = "moe_sum"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # 原地写 output

# 复刻默认 shapes：2D (64,64)/(4096,4096) 视作 topk=1；3D (64,512,512) 直接用；
# 另补常见 MoE topk 档位。语义：(num_tokens, topk, hidden)。
_SHAPES = [
    (64, 1, 64),
    (4096, 1, 4096),
    (64, 512, 512),
    (1024, 1, 1024),
    (128, 8, 4096),
    (512, 8, 4096),
]


def native():
    """解析 vllm._custom_ops.moe_sum；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "moe_sum", None)
    return op if callable(op) else None


def grid():
    return [{"num_tokens": t, "topk": k, "hidden": h} for (t, k, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, k, h = binding["num_tokens"], binding["topk"], binding["hidden"]
    inp = torch.randn(t, k, h, dtype=dtype, device=device)
    out = torch.empty(t, h, dtype=dtype, device=device)
    # topk_ids/expert_map 缺省 None，走纯求和路径。
    return (inp, out), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["topk"], binding["hidden"]]


def config(binding, dtype):
    t, k, h = binding["num_tokens"], binding["topk"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [t, k, h], "dtype": dt},
            "output": {"shape": [t, h], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "output": {"shape": [t, h], "dtype": dt},
        },
        "dims": {"num_tokens": t, "topk": k, "hidden": h},
    }
