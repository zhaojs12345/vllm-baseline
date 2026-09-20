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

"""fp8_fp4_mqa_logits baseline（方案 B）。

native：vllm.utils.deep_gemm.fp8_fp4_mqa_logits(q, kv, weights,
    cu_seqlen_ks, cu_seqlen_ke, clean_logits) -> logits[M, N] fp32
    deep_gemm 惰性绑定 wrapper：调用时先 _lazy_init()，若 _fp8_fp4_mqa_logits_impl
    仍为 None（deep_gemm 未加载/该版本无此符号）则走 _missing 抛错，否则转调
    impl。native() 只 import 并返回该 wrapper；能否真执行取决于运行环境是否装了
    deep_gemm——未加载时采集器在调用处捕获异常并跳过（符合契约）。
    src: vllm/utils/deep_gemm.py#L510

shape 复刻 FlagGems-vllm 基准 benchmark/test_fp8_fp4_mqa_logits.py：
    DeepSeek-V4 production config H=64, D=128；(M, N) 取基准 11 组。
    FP8 路径：q=(q_fp8[M,H,D] e4m3, None)，per-token scale 折进 weights；
    kv=(k_fp8[N,D] e4m3, k_scale[N] fp32)；weights[M,H] fp32；
    cu_seqlen_ks/ke[M] int32；clean_logits=True。
"""

import importlib

import torch

OP_NAME = "fp8_fp4_mqa_logits"
DTYPES = [torch.bfloat16]  # 构造 q/k 的原始 dtype；operands 量化为 fp8
IS_INPLACE = False  # 返回新 logits 张量

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# DeepSeek-V4 production config（复刻基准）
_H = 64
_D = 128

# (M, N) —— 基准 set_shapes 的 11 组
_SHAPES = [
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (4, 2048),
    (4, 4096),
    (64, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 8192),
    (1024, 8192),
]


def native():
    """解析 vllm.utils.deep_gemm.fp8_fp4_mqa_logits wrapper；解析不到返回 None。

    只 import + 取 wrapper；不触发 _lazy_init（那在调用时发生）。
    """
    try:
        mod = importlib.import_module("vllm.utils.deep_gemm")
    except ImportError:
        return None
    op = getattr(mod, "fp8_fp4_mqa_logits", None)
    return op if callable(op) else None


def grid():
    return [{"M": m, "N": n} for (m, n) in _SHAPES]


def _to_fp8(x):
    """clamp 到 e4m3 动态范围后转 fp8（stub 无 finfo 时直接 .to）。"""
    if hasattr(torch, "finfo"):
        finfo = torch.finfo(_FP8_DTYPE)
        x = x.clamp(min=finfo.min, max=finfo.max)
    return x.to(_FP8_DTYPE)


def _k_cast(k_bf16):
    """把 k[N,D] 量化为 (k_fp8[N,D] e4m3, k_scale[N] fp32)。

    真 torch + vllm 可用时走 per_custom_dims_cast_to_fp8((0,)) 复刻基准；
    离线 stub 时给 fp8 占位 + fp32 scale 占位（仅供采集器跟踪 shape/dtype）。
    """
    try:
        from vllm.third_party.deep_gemm.utils import per_custom_dims_cast_to_fp8

        return per_custom_dims_cast_to_fp8(k_bf16, (0,), False)
    except ImportError:
        N = k_bf16.shape[0]
        k_fp8 = _to_fp8(k_bf16)
        k_scale = torch.randn(N, dtype=torch.float32, device=k_bf16.device)
        return k_fp8, k_scale


def build_inputs(binding, dtype, device):
    M = binding["M"]
    N = binding["N"]

    q_bf16 = torch.randn(M, _H, _D, device=device, dtype=dtype)
    k_bf16 = torch.randn(N, _D, device=device, dtype=dtype)
    weights = torch.randn(M, _H, device=device, dtype=torch.float32).abs()

    q_fp8 = _to_fp8(q_bf16)
    k_fp8, k_scale = _k_cast(k_bf16)

    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    kwargs = {
        "q": (q_fp8, None),
        "kv": (k_fp8, k_scale),
        "weights": weights,
        "cu_seqlen_ks": ks,
        "cu_seqlen_ke": ke,
        "clean_logits": True,
    }
    return (), kwargs


def key_shape(binding):
    return [binding["M"], binding["N"]]


def config(binding, dtype):
    M = binding["M"]
    N = binding["N"]
    fp8 = str(_FP8_DTYPE)
    return {
        "inputs": {
            "q_values": {"shape": [M, _H, _D], "dtype": fp8,
                         "note": "FP8 路径 q_scale=None，per-token scale 折进 weights"},
            "k_packed": {"shape": [N, _D], "dtype": fp8},
            "k_scale": {"shape": [N], "dtype": "torch.float32"},
            "weights": {"shape": [M, _H], "dtype": "torch.float32"},
            "cu_seqlen_ks": {"shape": [M], "dtype": "torch.int32"},
            "cu_seqlen_ke": {"shape": [M], "dtype": "torch.int32"},
            "clean_logits": {"scalar": True},
        },
        "outputs": {
            "logits": {"shape": [M, N], "dtype": "torch.float32"},
        },
        "dims": {"M": M, "N": N, "H": _H, "D": _D},
        "note": "shape 复刻 FlagGems-vllm benchmark/test_fp8_fp4_mqa_logits.py",
    }
