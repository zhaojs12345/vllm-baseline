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

"""grouped_topk baseline（方案 B）。

native：torch.ops._moe_C.grouped_topk(
            scores, n_group, topk_group, topk,
            renormalize, routed_scaling_factor, bias, scoring_func)
    需先 import vllm._custom_ops 触发 torch.ops._moe_C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/test_grouped_topk.py 的
grouped_topk_input_fn（Deepseek-3.2 配置）：
    scores = randn(T, E)           遍历 dtype
    bias   = randn(E)              固定 fp32
    n_group=8, topk_group=4, topk=8, renormalize=True,
    routed_scaling_factor=1.0, scoring_func=0     均为常量标量
"""

import importlib
import itertools

import torch

OP_NAME = "grouped_topk"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False

# Deepseek-3.2：num_experts/n_group/topk_group/topk 固定，仅 num_tokens 变化。
_GRID = {
    "T": [1, 8, 32, 64, 128, 256, 496, 512, 16384],
    "E": [256],
}

_N_GROUP = 8
_TOPK_GROUP = 4
_TOPK = 8
_RENORMALIZE = True
_ROUTED_SCALING_FACTOR = 1.0
_SCORING_FUNC = 0


def native():
    """解析 torch.ops._moe_C.grouped_topk；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._moe_C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    moe_c = getattr(getattr(torch.ops, "_moe_C", None), "grouped_topk", None)
    return moe_c if callable(moe_c) else None


def grid():
    dims = list(_GRID)
    return [dict(zip(dims, combo))
            for combo in itertools.product(*(_GRID[d] for d in dims))]


def build_inputs(binding, dtype, device):
    T, E = binding["T"], binding["E"]
    scores = torch.randn(T, E, dtype=dtype, device=device)
    bias = torch.randn(E, dtype=torch.float32, device=device)
    args = (
        scores,
        _N_GROUP,
        _TOPK_GROUP,
        _TOPK,
        _RENORMALIZE,
        _ROUTED_SCALING_FACTOR,
        bias,
        _SCORING_FUNC,
    )
    return args, {}


def key_shape(binding):
    return [binding["T"], binding["E"]]
