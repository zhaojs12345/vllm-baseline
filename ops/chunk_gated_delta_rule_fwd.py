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

"""chunk_gated_delta_rule_fwd baseline（方案 B）——FLA 门控 delta rule 分块前向。

native：vllm.third_party.flash_linear_attention.ops.chunk.chunk_gated_delta_rule_fwd
    （src: chunk.py#L23）。签名（已回源核对）：
        chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state,
            output_final_state, cu_seqlens=None, chunk_indices=None,
            chunk_offsets=None, core_attn_out=None)

输入构造复刻 FlagGems-vllm/benchmark/test_FLA/test_chunk_gated_delta_rule_fwd.py
    ChunkGatedDeltaRuleFwdBenchmark._build_inputs：
    (B, T, H, K, V)：
    q     = randn(B, T, H, K)/sqrt(K)
    k     = randn(B, T, H, K)/sqrt(K)
    v     = randn(B, T, H, V)
    g     = (-rand(B, T, H) * 0.1) → dtype       # fp32 生成后转 dtype
    beta  = rand(B, T, H).sigmoid()
    scale = K**-0.5
    调用 (q, k, v, g, beta, scale, None, True, None)
    （initial_state=None, output_final_state=True, cu_seqlens=None）
shape 来源：test_chunk_gated_delta_rule_fwd.py DEFAULT_SHAPES（B,T,H,K,V 三档）。
"""

import importlib

import torch

OP_NAME = "chunk_gated_delta_rule_fwd"
DTYPES = [torch.bfloat16, torch.float16]  # benchmark DEFAULT_DTYPES
IS_INPLACE = False

# (B, T, H, K, V)（test_chunk_gated_delta_rule_fwd.py DEFAULT_SHAPES）
_SHAPES = [
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
]


def native():
    """解析 FLA chunk.chunk_gated_delta_rule_fwd；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.third_party.flash_linear_attention.ops.chunk"
        )
    except ImportError:
        return None
    op = getattr(mod, "chunk_gated_delta_rule_fwd", None)
    return op if callable(op) else None


def grid():
    return [
        {"B": b, "T": t, "H": h, "K": k, "V": v}
        for (b, t, h, k, v) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    B, T, H, K, V = (
        binding["B"], binding["T"], binding["H"], binding["K"], binding["V"]
    )
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K ** 0.5)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K ** 0.5)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K ** -0.5
    # (q, k, v, g, beta, scale, initial_state=None, output_final_state=True, cu_seqlens=None)
    return (q, k, v, g, beta, scale, None, True, None), {}


def key_shape(binding):
    return [binding["B"], binding["T"], binding["H"], binding["K"], binding["V"]]


def config(binding, dtype):
    B, T, H, K, V = (
        binding["B"], binding["T"], binding["H"], binding["K"], binding["V"]
    )
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [B, T, H, K], "dtype": dt},
            "k": {"shape": [B, T, H, K], "dtype": dt},
            "v": {"shape": [B, T, H, V], "dtype": dt},
            "g": {"shape": [B, T, H], "dtype": dt},
            "beta": {"shape": [B, T, H], "dtype": dt},
            "scale": K ** -0.5,
        },
        "outputs": {
            "core_attn_out": {"shape": [B, T, H, V], "dtype": dt},
            "final_state": {"note": "output_final_state=True"},
        },
        "dims": {"B": B, "T": T, "H": H, "K": K, "V": V},
    }
