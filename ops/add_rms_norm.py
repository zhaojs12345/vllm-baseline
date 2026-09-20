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

"""add_rms_norm baseline（方案 B）。

vLLM 侧对应算子带 fused_ 前缀：native 为
    vllm._custom_ops.fused_add_rms_norm(input, residual, weight, eps)
    input/residual 原地写回，返回 None。src: vllm/_custom_ops.py#L322
（与 ops/fused_add_rms_norm.py 同一底层 kernel；本模块区别仅在于复刻
test_add_rms_norm.py 的 2D-only shape 网格。）

输入构造复刻 FlagGems-vllm/benchmark/test_add_rms_norm.py 的 add_rms_norm_input_fn：
    inp1 = randn(M, N); inp2 = randn(M, N); weight = randn(N)
    benchmark 里 yield (inp1, inp2, (N,), weight)，其中 (N,) 是给 torch 参考实现的
    normalized_shape，native 不需要——native 直接吃 (input, residual, weight, eps)。
shape 取 GenericBenchmark2DOnly 默认（core_shapes.yaml），均为 2D (M, N)。
"""

import importlib

import torch

OP_NAME = "add_rms_norm"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # input/residual 原地写回

# GenericBenchmark2DOnly 默认 shapes（core_shapes.yaml）。
_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
]

_EPS = 1.0e-5


def native():
    """解析 vllm._custom_ops.fused_add_rms_norm；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "fused_add_rms_norm", None)
    return op if callable(op) else None


def grid():
    return [{"M": m, "N": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    M, N = binding["M"], binding["N"]
    inp = torch.randn(M, N, dtype=dtype, device=device)
    residual = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    return (inp, residual, weight, _EPS), {}


def key_shape(binding):
    return [binding["M"], binding["N"]]


def config(binding, dtype):
    M, N = binding["M"], binding["N"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [M, N], "dtype": dt, "note": "原地写回"},
            "residual": {"shape": [M, N], "dtype": dt, "note": "原地写回"},
            "weight": {"shape": [N], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "input": {"shape": [M, N], "dtype": dt},
            "residual": {"shape": [M, N], "dtype": dt},
        },
        "dims": {"M": M, "N": N},
    }
