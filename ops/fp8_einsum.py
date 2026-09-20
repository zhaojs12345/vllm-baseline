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

"""fp8_einsum baseline（方案 B）。

native：vllm.utils.deep_gemm.fp8_einsum(*args, **kwargs) -> ...
    deep_gemm 惰性绑定的 wrapper：调用时先 _lazy_init()，若 _fp8_einsum_impl
    仍为 None（deep_gemm 未加载/该版本无 fp8_einsum）则走 _missing 抛错，
    否则转调 _fp8_einsum_impl。native() 只负责 import 并返回该 wrapper；
    是否可真正执行取决于运行环境是否装了 deep_gemm——未加载时采集器会在调用处
    捕获异常并跳过（符合契约）。
    src: vllm/utils/deep_gemm.py#L467

无对应 benchmark（FlagGems-vllm/benchmark 下无用例）。
**shape 为 vllm 源码推断，非 FlagGems-vllm 基准**。
推断依据：唯一调用点 vllm/models/deepseek_v4/nvidia/ops/o_proj.py#L66
    fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (wo_a.weight, weight_scale),
               z, recipe=einsum_recipe)
其中（见同文件 fused_inv_rope_fp8_quant / deep_gemm_fp8_o_proj）：
    o_fp8   : [T, G, D]        float8_e4m3fn      D = heads_per_group*head_dim
    wo_a.weight : [G, o_lora_rank, D]  float8_e4m3fn
    z (out) : [T, G, o_lora_rank]      bfloat16
    recipe  : (1, 128, 128) (SM90) / (1, 1, 128) (SM100)
    o_scale / weight_scale 为 block scale（fp32 或 INT32-packed UE8M0，随 arch）
这里取 SM90 语义（recipe=(1,128,128)、fp32 scale）构造一组占位输入。
注意：scale 张量的精确 packed 布局是 deep_gemm 内部约定，此处按 block 粒度给出
fp32 近似占位；真机若 deep_gemm 已加载但拒绝该布局，采集器会跳过该点。
"""

import importlib

import torch

OP_NAME = "fp8_einsum"
DTYPES = [torch.bfloat16]  # 输出 z 的 dtype；operands 为 fp8（在 build_inputs 固定）
IS_INPLACE = True  # 结果写入 out(z)

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# 源码推断维度（DeepSeek-V4 o_proj 语义）：
#   head_dim = nope_dim(448) + rope_dim(64) = 512
#   D = heads_per_group * head_dim
_HEAD_DIM = 512
_QUANT_BLOCK = 128
_RECIPE = (1, 128, 128)  # SM90 语义

# (T, G, heads_per_group, o_lora_rank)
_SHAPES = [
    (1, 1, 8, 2048),
    (16, 1, 8, 2048),
    (64, 1, 8, 2048),
    (128, 1, 8, 2048),
]


def native():
    """解析 vllm.utils.deep_gemm.fp8_einsum wrapper；解析不到返回 None。

    只 import + 取 wrapper；不触发 _lazy_init（那会在调用时发生）。
    """
    try:
        mod = importlib.import_module("vllm.utils.deep_gemm")
    except ImportError:
        return None
    op = getattr(mod, "fp8_einsum", None)
    return op if callable(op) else None


def grid():
    return [
        {"T": t, "G": g, "heads_per_group": hpg, "o_lora_rank": r}
        for (t, g, hpg, r) in _SHAPES
    ]


def _fp8(shape, device):
    t = torch.randn(shape, device=device, dtype=torch.float32)
    if hasattr(torch, "finfo"):
        finfo = torch.finfo(_FP8_DTYPE)
        t = t.clamp(min=finfo.min, max=finfo.max)
    return t.to(_FP8_DTYPE)


def build_inputs(binding, dtype, device):
    T = binding["T"]
    G = binding["G"]
    hpg = binding["heads_per_group"]
    r = binding["o_lora_rank"]
    D = hpg * _HEAD_DIM

    o_fp8 = _fp8((T, G, D), device)
    wo_weight = _fp8((G, r, D), device)
    # block scale（fp32 占位，block 粒度 = quant_group=128）
    o_scale = torch.randn(T, G, D // _QUANT_BLOCK, dtype=torch.float32, device=device)
    weight_scale = torch.randn(
        G, r // _QUANT_BLOCK, D // _QUANT_BLOCK, dtype=torch.float32, device=device
    )
    z = torch.empty(T, G, r, dtype=torch.bfloat16, device=device)

    args = ("bhr,hdr->bhd", (o_fp8, o_scale), (wo_weight, weight_scale), z)
    return args, {"recipe": _RECIPE}


def key_shape(binding):
    return [
        binding["T"],
        binding["G"],
        binding["heads_per_group"],
        binding["o_lora_rank"],
    ]


def config(binding, dtype):
    T = binding["T"]
    G = binding["G"]
    hpg = binding["heads_per_group"]
    r = binding["o_lora_rank"]
    D = hpg * _HEAD_DIM
    fp8 = str(_FP8_DTYPE)
    return {
        "inputs": {
            "subscripts": {"scalar": "bhr,hdr->bhd"},
            "o_fp8": {"shape": [T, G, D], "dtype": fp8},
            "o_scale": {"shape": [T, G, D // _QUANT_BLOCK], "dtype": "torch.float32",
                        "note": "block scale 占位（真布局为 deep_gemm 内部约定）"},
            "wo_weight": {"shape": [G, r, D], "dtype": fp8},
            "weight_scale": {"shape": [G, r // _QUANT_BLOCK, D // _QUANT_BLOCK],
                             "dtype": "torch.float32"},
            "out": {"shape": [T, G, r], "dtype": "torch.bfloat16", "note": "原地写"},
            "recipe": {"scalar": list(_RECIPE)},
        },
        "outputs": {
            "out": {"shape": [T, G, r], "dtype": "torch.bfloat16"},
        },
        "dims": {"T": T, "G": G, "heads_per_group": hpg,
                 "head_dim": _HEAD_DIM, "D": D, "o_lora_rank": r},
        "note": "shape 为 vllm 源码推断，非 FlagGems-vllm 基准",
    }
