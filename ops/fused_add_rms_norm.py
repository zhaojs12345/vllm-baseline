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

"""fused_add_rms_norm baseline（方案 B）。

native：vllm._custom_ops.fused_add_rms_norm(input, residual, weight, eps)
    input/residual 原地写回，返回 None。

输入构造复刻 FlagGems-vllm/benchmark/test_fused_add_rms_norm.py 的 _input_fn：
    input    = randn(M, N)
    residual = randn(M, N)
    weight   = randn(N)
    eps      = 1e-5
注意 native 签名不含 benchmark 里的 layer_shape 参数（那是 FlagGems 顶层包装层
的入参）；native 直接吃 (input, residual, weight, eps)。
"""

import importlib
import itertools

import torch

OP_NAME = "fused_add_rms_norm"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True

# 采集维度网格：M（token 数）遍历，N（hidden）取常见档位。
_GRID = {
    "M": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    "N": [256, 512],
}

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
    dims = list(_GRID)
    return [dict(zip(dims, combo))
            for combo in itertools.product(*(_GRID[d] for d in dims))]


def build_inputs(binding, dtype, device):
    M, N = binding["M"], binding["N"]
    inp = torch.randn(M, N, dtype=dtype, device=device)
    residual = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    args = (inp, residual, weight, _EPS)
    return args, {}


def key_shape(binding):
    return [binding["M"], binding["N"]]


def config(binding, dtype):
    """真实输入输出 shape 描述（写入 JSON 的 config 字段）。

    input/residual 原地写回，既是输入也是输出。
    """
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
