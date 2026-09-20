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

"""mhc_pre baseline（方案 B）。

native：CustomOp 入口 MHCPreOp（vllm/model_executor/layers/mhc.py#L34，
    @CustomOp.register("mhc_pre")）。它是 nn.Module/CustomOp 子类，不是纯函数：
    调用实例走 CustomOp.forward -> _forward_method；在 CUDA 平台 dispatch 到
    forward_cuda（mhc.py#L47），该方法直接调
    torch.ops.vllm.mhc_pre_tilelang(residual, fn, hc_scale, hc_base, rms_eps,
    hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
    n_splits, norm_weight, norm_eps) -> (post_mix, comb_mix, layer_input)。
    因此 native() 实例化 MHCPreOp() 并返回该实例本身（callable），op(*args) 即触发
    tilelang CUDA kernel。MHCPreOp.__init__ 继承自 CustomOp，无形状入参，一个实例
    可复用于全部 shape。tilelang 不可用（HAS_TILELANG_MHC=False）时返回 None 优雅跳过。

输入构造复刻 FlagGems-vllm/benchmark/test_mhc.py 的 MHCPreBenchmark.get_input_iter：
    hc_mult = 4, sinkhorn_repeat = 10, hc_mult3 = hc_mult*2 + hc_mult*hc_mult = 24
    residual = randn(N, hc_mult, hidden_size).bfloat16()   # 归约主张量
    fn       = randn(hc_mult3, hc_mult, hidden_size).flatten(1,2)  # -> (24, hc_mult*hidden) float32
    hc_scale = randn(3) float32
    hc_base  = randn(hc_mult3) float32
    标量：rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
          hc_post_mult_value=1.0, sinkhorn_repeat=10
    （n_splits/norm_weight/norm_eps 走默认值，benchmark 未传。）
    注意 fn/hc_scale/hc_base 在源码里保持 float32，仅 residual 用 dtype(bf16)。

shape：直接采用 benchmark 的 (N, hidden_size) 12 档网格。
"""

import importlib

import torch

OP_NAME = "mhc_pre"
DTYPES = [torch.bfloat16]  # benchmark 仅 bf16；residual 用之，其余张量恒 float32
IS_INPLACE = False  # 返回 (post_mix, comb_mix, layer_input) 三元组

_HC_MULT = 4
_SINKHORN_REPEAT = 10
_RMS_EPS = 1.0e-6
_HC_PRE_EPS = 1.0e-6
_HC_SINKHORN_EPS = 1.0e-6
_HC_POST_MULT_VALUE = 1.0

# benchmark 的 (N, hidden_size) 网格。
_SHAPES = [
    (512, 1280),
    (512, 2560),
    (512, 4096),
    (1024, 1280),
    (1024, 2560),
    (1024, 4096),
    (2048, 1280),
    (2048, 2560),
    (2048, 4096),
    (8192, 1280),
    (8192, 2560),
    (8192, 4096),
]


def native():
    """实例化 MHCPreOp 并返回实例（callable）；不可用时返回 None。

    先 import mhc 模块（会触发自定义算子注册）。若 HAS_TILELANG_MHC 为假，
    forward_cuda 依赖的 torch.ops.vllm.mhc_pre_tilelang 不可用，返回 None。
    """
    try:
        mod = importlib.import_module("vllm.model_executor.layers.mhc")
    except ImportError:
        return None
    if not getattr(mod, "HAS_TILELANG_MHC", False):
        return None  # tilelang CUDA kernel 不可用，优雅跳过
    cls = getattr(mod, "MHCPreOp", None)
    if cls is None:
        return None
    try:
        inst = cls()  # __init__ 继承 CustomOp，无形状入参
    except Exception:  # noqa: BLE001 — 无 vLLM 运行时上下文时优雅跳过
        return None
    return inst if callable(inst) else None


def grid():
    return [{"N": n, "hidden_size": h} for (n, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    n, hidden = binding["N"], binding["hidden_size"]
    hc_mult = _HC_MULT
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # = 24

    residual = torch.randn(n, hc_mult, hidden, dtype=dtype, device=device)
    # fn 源码 shape (hc_mult3, hc_mult, hidden) 再 flatten(1,2) -> (hc_mult3, hc_mult*hidden)
    fn = torch.randn(hc_mult3, hc_mult * hidden, dtype=torch.float32, device=device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device)

    args = (
        residual,
        fn,
        hc_scale,
        hc_base,
        _RMS_EPS,
        _HC_PRE_EPS,
        _HC_SINKHORN_EPS,
        _HC_POST_MULT_VALUE,
        _SINKHORN_REPEAT,
    )
    return args, {}


def key_shape(binding):
    return [binding["N"], _HC_MULT, binding["hidden_size"]]


def config(binding, dtype):
    n, hidden = binding["N"], binding["hidden_size"]
    hc_mult = _HC_MULT
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    dt = str(dtype)
    f32 = str(torch.float32)
    return {
        "inputs": {
            "residual": {"shape": [n, hc_mult, hidden], "dtype": dt},
            "fn": {"shape": [hc_mult3, hc_mult * hidden], "dtype": f32},
            "hc_scale": {"shape": [3], "dtype": f32},
            "hc_base": {"shape": [hc_mult3], "dtype": f32},
            "rms_eps": {"scalar": _RMS_EPS},
            "hc_pre_eps": {"scalar": _HC_PRE_EPS},
            "hc_sinkhorn_eps": {"scalar": _HC_SINKHORN_EPS},
            "hc_post_mult_value": {"scalar": _HC_POST_MULT_VALUE},
            "sinkhorn_repeat": {"scalar": _SINKHORN_REPEAT},
        },
        "outputs": {
            "post_mix": {"note": "sinkhorn 组合权重"},
            "comb_mix": {"note": "组合权重"},
            "layer_input": {"shape": [n, hidden], "dtype": dt},
        },
        "dims": {"N": n, "hidden_size": hidden, "hc_mult": hc_mult},
    }
