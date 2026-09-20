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

"""gemma_rms_norm baseline（方案 B）。

native：CustomOp 入口 GemmaRMSNorm（vllm/model_executor/layers/layernorm.py#L132，
    @CustomOp.register("gemma_rms_norm")）。它是 nn.Module/CustomOp 子类，不是纯
    函数，且 __init__(hidden_size, eps=1e-6)（layernorm.py#L142）持有一个大小为
    hidden_size 的 weight 参数——因此每个 hidden 档位需各自的实例。调用实例走
    CustomOp.forward -> forward_cuda（layernorm.py#L162），后者转 forward_native
    （L151），对 residual=None 走 ir.ops.rms_norm(x, weight, eps)，否则走
    ir.ops.fused_add_rms_norm(...)。语义：x * (1 + w) / sqrt(E[x^2]+eps)。
    native() 返回一个闭包：按输入张量最后一维 hidden 惰性构造并缓存
    GemmaRMSNorm(hidden)，再以实例调用触发 kernel；构造只在每个新 hidden 发生一次
    并缓存，重复采集点只跑 kernel。解析不到 vllm/GemmaRMSNorm 时返回 None。

输入构造：本算子在 FlagGems-vllm 无独立 benchmark（core_shapes.yaml 无专属键），
    故 shape 为 vllm 源码推断，非 FlagGems-vllm 基准。按 RMSNorm 常见用法取
    (num_tokens, hidden) 二维网格，hidden 覆盖 Gemma 常见隐藏维档位。
    forward 输入为单张量 x（residual=None，走 rms_norm 路径）；weight 由实例内部
    持有（__init__ 初始化为 zeros，(1+w) 即恒等权重，满足延迟基准所需）。
"""

import importlib

import torch

OP_NAME = "gemma_rms_norm"
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
IS_INPLACE = False  # 返回归一化后的张量

_EPS = 1.0e-6

# shape 为 vllm 源码推断，非 FlagGems-vllm 基准。
# (num_tokens, hidden)；hidden 取 Gemma 常见隐藏维（2048/3072/3584/4096）。
_SHAPES = [
    (1, 2048),
    (16, 2048),
    (256, 2048),
    (4096, 2048),
    (256, 3072),
    (4096, 3072),
    (256, 3584),
    (4096, 3584),
    (256, 4096),
    (4096, 4096),
]


def native():
    """返回按 hidden 惰性构造并缓存 GemmaRMSNorm 的闭包；不可用时返回 None。"""
    try:
        mod = importlib.import_module("vllm.model_executor.layers.layernorm")
    except ImportError:
        return None
    cls = getattr(mod, "GemmaRMSNorm", None)
    if cls is None:
        return None

    cache: dict[tuple, object] = {}

    def _run(x, residual=None):
        hidden = x.shape[-1]
        key = (hidden, x.dtype, x.device)
        inst = cache.get(key)
        if inst is None:
            try:
                inst = cls(hidden, _EPS).to(device=x.device, dtype=x.dtype)
            except Exception:  # noqa: BLE001 — 无运行时上下文时放弃
                return None
            cache[key] = inst
        return inst(x, residual)

    return _run


def grid():
    return [{"num_tokens": t, "hidden": h} for (t, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, h = binding["num_tokens"], binding["hidden"]
    x = torch.randn(t, h, dtype=dtype, device=device)
    # residual 缺省 None，走 ir.ops.rms_norm 单张量路径。
    return (x,), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["hidden"]]


def config(binding, dtype):
    t, h = binding["num_tokens"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [t, h], "dtype": dt},
            "weight": {"shape": [h], "dtype": dt, "note": "实例内部持有 (1+w)"},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [t, h], "dtype": dt},
        },
        "dims": {"num_tokens": t, "hidden": h},
        "shape_source": "vllm 源码推断，非 FlagGems-vllm 基准",
    }
