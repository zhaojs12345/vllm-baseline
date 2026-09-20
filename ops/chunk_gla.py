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

"""chunk_gla baseline（方案 B）——FLA GLA 分块前向「输出阶段」。

native：vllm.third_party.flash_linear_attention.ops.kda.chunk_gla_fwd_o_gk
    （src: kda.py#L1149）。签名（已回源核对）：
        chunk_gla_fwd_o_gk(q, v, g, A, h, o, scale, cu_seqlens=None,
            chunk_indices=None, chunk_size=FLA_CHUNK_SIZE=64) -> o
    内部启动 chunk_gla_fwd_kernel_o（kda.py#L1042），原地写 o 并返回。

    ⚠ 接口差异说明：FlagGems-vllm 的 test_FLA/test_chunk_gla_perf.py 基准跑的是
    **上游顶层 fla.ops.gla.chunk_gla(q,k,v,g)**（会在内部算出 A/h 等中间量），
    而 vLLM 侧未导出顶层 chunk_gla，只保留了 Triton 输出阶段 chunk_gla_fwd_o_gk
    （见 CSV 第 row）。因此本模块 native 采集的是**输出阶段算子**，需外部预置
    中间量 A（块内注意力）与 h（状态）。这两者的 shape 从 chunk_gla_fwd_kernel_o
    的 block_ptr 推得：
        A: [B, T, H, BT]（p_A 偏移 (bos*H+i_h)*BT、逻辑 (T, BT)）
        h: [B, NT, H, V, K]（p_h 偏移 (i_tg*H+i_h)*K*V、i_tg=i_b*NT+i_t、逻辑 (V, K)）
        o/v: [B, T, H, V]，q/g: [B, T, H, K]，BT=FLA_CHUNK_SIZE=64，NT=cdiv(T,BT)。

shape 来源：(B,T,H,D) 取自 test_chunk_gla_perf.py _SHAPES（K=V=D）；
    A/h/o 的具体张量为 vLLM 输出阶段 kernel block_ptr 推断（**中间量非 benchmark 原样，
    为对齐 vLLM 仅存的输出阶段接口所构造**）。
"""

import importlib

import torch

OP_NAME = "chunk_gla"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True  # 原地写 o 并返回

_BT = 64  # FLA_CHUNK_SIZE

# (B, T, H, D)（test_chunk_gla_perf.py _SHAPES，K=V=D）
_SHAPES = [
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (2, 2048, 16, 512),
    (4, 1024, 8, 512),
    (8, 1024, 8, 64),
]


def _cdiv(a, b):
    return -(-a // b)


def native():
    """解析 FLA kda.chunk_gla_fwd_o_gk；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.third_party.flash_linear_attention.ops.kda"
        )
    except ImportError:
        return None
    op = getattr(mod, "chunk_gla_fwd_o_gk", None)
    return op if callable(op) else None


def grid():
    return [
        {"B": b, "T": t, "H": h, "D": d}
        for (b, t, h, d) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    B, T, H, D = binding["B"], binding["T"], binding["H"], binding["D"]
    K = V = D
    BT = _BT
    NT = _cdiv(T, BT)
    scale = D ** -0.5

    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * (D ** -0.5)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    # g 为对数域门控：benchmark 用 F.logsigmoid(randn)，其取值恒 ≤ 0。
    # 基准采集只关心 kernel 计时/shape，门控具体分布无影响；用 -|randn|
    # 构造恒非正的门控值，避免依赖 torch.nn/logsigmoid，且不会溢出。
    g = (-torch.abs(torch.randn(B, T, H, K, device=device, dtype=torch.float32))).to(dtype)
    # 预置中间量（输出阶段 kernel 消费；shape 由 block_ptr 推得）
    A = torch.randn(B, T, H, BT, device=device, dtype=dtype)
    h = torch.randn(B, NT, H, V, K, device=device, dtype=dtype)
    o = torch.empty(B, T, H, V, device=device, dtype=dtype)  # 原地输出缓冲
    return (q, v, g, A, h, o, scale), {}


def key_shape(binding):
    return [binding["B"], binding["T"], binding["H"], binding["D"]]


def config(binding, dtype):
    B, T, H, D = binding["B"], binding["T"], binding["H"], binding["D"]
    K = V = D
    BT = _BT
    NT = _cdiv(T, BT)
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [B, T, H, K], "dtype": dt},
            "v": {"shape": [B, T, H, V], "dtype": dt},
            "g": {"shape": [B, T, H, K], "dtype": dt},
            "A": {"shape": [B, T, H, BT], "dtype": dt, "note": "预置块内注意力"},
            "h": {"shape": [B, NT, H, V, K], "dtype": dt, "note": "预置状态"},
            "o": {"shape": [B, T, H, V], "dtype": dt, "note": "原地输出"},
            "scale": D ** -0.5,
        },
        "outputs": {"o": {"shape": [B, T, H, V], "dtype": dt}},
        "dims": {"B": B, "T": T, "H": H, "K": K, "V": V,
                 "chunk_size": BT, "NT": NT,
                 "note": "vLLM 仅存输出阶段 chunk_gla_fwd_o_gk；A/h 为推断中间量"},
    }
