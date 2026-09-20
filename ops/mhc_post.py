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

"""mhc_post baseline（方案 B）。

native：CustomOp 入口 MHCPostOp（vllm/model_executor/layers/mhc.py#L197，
    @CustomOp.register("mhc_post")）。它是 nn.Module/CustomOp 子类，不是纯函数：
    调用实例走 CustomOp.forward -> _forward_method；在 CUDA 平台 dispatch 到
    forward_cuda（mhc.py#L210），该方法直接调
    torch.ops.vllm.mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
    -> out。因此 native() 实例化 MHCPostOp() 并返回该实例本身（callable），
    op(x, residual, post_layer_mix, comb_res_mix) 即触发 tilelang CUDA kernel。
    MHCPostOp.__init__ 继承自 CustomOp，无形状入参，一个实例可复用于全部 shape。
    tilelang 不可用（HAS_TILELANG_MHC=False）时返回 None 优雅跳过。

输入构造复刻 FlagGems-vllm/benchmark/test_mhc.py 的 MHCPostBenchmark.get_input_iter：
    hc_mult = 4
    x              = randn(N, H)            bfloat16
    residual       = randn(N, hc_mult, H)   bfloat16
    post_layer_mix = randn(N, hc_mult, 1)   float32
    comb_res_mix   = randn(N, hc_mult, hc_mult) float32
    注意 mix 张量恒 float32，仅 x/residual 用 dtype(bf16)。

shape：直接采用 benchmark 的 (N, H) 三档网格。
"""

import importlib

import torch

OP_NAME = "mhc_post"
DTYPES = [torch.bfloat16]  # benchmark 仅 bf16；x/residual 用之，mix 恒 float32
IS_INPLACE = False  # 返回 out 张量

_HC_MULT = 4

# benchmark 的 (N, H) 网格。
_SHAPES = [
    (4096, 1280),
    (4096, 2560),
    (4096, 7168),
]


def native():
    """实例化 MHCPostOp 并返回实例（callable）；不可用时返回 None。

    先 import mhc 模块（会触发自定义算子注册）。若 HAS_TILELANG_MHC 为假，
    forward_cuda 依赖的 torch.ops.vllm.mhc_post_tilelang 不可用，返回 None。
    """
    try:
        mod = importlib.import_module("vllm.model_executor.layers.mhc")
    except ImportError:
        return None
    if not getattr(mod, "HAS_TILELANG_MHC", False):
        return None  # tilelang CUDA kernel 不可用，优雅跳过
    cls = getattr(mod, "MHCPostOp", None)
    if cls is None:
        return None
    try:
        inst = cls()  # __init__ 继承 CustomOp，无形状入参
    except Exception:  # noqa: BLE001 — 无 vLLM 运行时上下文时优雅跳过
        return None
    return inst if callable(inst) else None


def grid():
    return [{"N": n, "H": h} for (n, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    n, h = binding["N"], binding["H"]
    hc_mult = _HC_MULT

    x = torch.randn(n, h, dtype=dtype, device=device)
    residual = torch.randn(n, hc_mult, h, dtype=dtype, device=device)
    post_layer_mix = torch.randn(n, hc_mult, 1, dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(
        n, hc_mult, hc_mult, dtype=torch.float32, device=device
    )

    args = (x, residual, post_layer_mix, comb_res_mix)
    return args, {}


def key_shape(binding):
    return [binding["N"], _HC_MULT, binding["H"]]


def config(binding, dtype):
    n, h = binding["N"], binding["H"]
    hc_mult = _HC_MULT
    dt = str(dtype)
    f32 = str(torch.float32)
    return {
        "inputs": {
            "x": {"shape": [n, h], "dtype": dt},
            "residual": {"shape": [n, hc_mult, h], "dtype": dt},
            "post_layer_mix": {"shape": [n, hc_mult, 1], "dtype": f32},
            "comb_res_mix": {"shape": [n, hc_mult, hc_mult], "dtype": f32},
        },
        "outputs": {
            "out": {"shape": [n, hc_mult, h], "dtype": dt},
        },
        "dims": {"N": n, "H": h, "hc_mult": hc_mult},
    }
