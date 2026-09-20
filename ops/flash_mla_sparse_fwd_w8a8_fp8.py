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

"""flash_mla_sparse_fwd_w8a8_fp8 baseline（方案 B）。

native：vllm.v1.attention.ops.flashmla.flash_mla_sparse_fwd
    （vLLM 侧无独立 def，从 vllm.third_party.flashmla.flash_mla_interface 重导出，
    见 flashmla.py#L87-95；调用点 vllm/v1/attention/backends/mla/flashmla_sparse.py#L972）。
    签名（据调用点与 benchmark 复原）：
        flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=None,
                             attn_sink=None, topk_length=None) -> (out, ?, lse)
    调用点 flashmla_sparse.py#L972：
        flash_mla_sparse_fwd(q, kv_c_and_k_pe_cache, topk_indices,
                             softmax_scale, topk_length=topk_length)
    native() 直接返回该 callable（无需 metadata 预处理）。

shape 来源：FlagGems-vllm/benchmark/test_flash_mla_sparse_fwd.py
    FlashmlaSparseBenchmark.make_input_flashmla（该 benchmark 为 bf16 稀疏基准）。
    w8a8_fp8 变体差异（CSV 第 39 行「非 w8a8_fp8 变体」）：KV cache 走 fp8 稀疏路径，
    真实为专用字节编码；此处沿用 benchmark 的张量维度，KV/Q 以 fp8_e4m3fn 近似，
    **量化编码非 FlagGems-vllm 基准原样，为对齐 w8a8_fp8 的推断**。
    档位取 benchmark 第一组（s_q=4096, topk=2048, h_q=128, d_qk=576, have_attn_sink），
    s_kv ∈ [8192, 32768, 65536]（省略 98304/131072 两档超大 s_kv 以控显存）：
    q       = randn(s_q, h_q, d_qk)/10（clamp[-10,10]）
    kv      = randn(s_kv, h_kv=1, d_qk)/10（clamp[-10,10]）
    indices = 每行随机排列 [0,s_kv) 的 topk 个下标，view(s_q, h_kv, topk)
    sm_scale= 0.5, d_v=512, attn_sink=randn(h_q) fp32
"""

import importlib

import torch

OP_NAME = "flash_mla_sparse_fwd_w8a8_fp8"
DTYPES = [torch.bfloat16]  # Q/KV 逻辑 dtype；w8a8_fp8 路径实际以 fp8 存储
IS_INPLACE = False

_S_Q = 4096
_TOPK = 2048
_H_Q = 128
_H_KV = 1
_D_QK = 576
_D_V = 512
_SM_SCALE = 0.5

# benchmark 第一组 s_kv 档位（省略 98304/131072 超大档以控显存）
_S_KVS = [8192, 32768, 65536]


def native():
    """解析 vllm.v1.attention.ops.flashmla.flash_mla_sparse_fwd；不可用返回 None。"""
    try:
        mod = importlib.import_module("vllm.v1.attention.ops.flashmla")
    except ImportError:
        return None
    op = getattr(mod, "flash_mla_sparse_fwd", None)
    return op if callable(op) else None


def grid():
    return [{"s_kv": s} for s in _S_KVS]


def build_inputs(binding, dtype, device):
    s_kv = binding["s_kv"]
    s_q, topk, h_q, h_kv = _S_Q, _TOPK, _H_Q, _H_KV
    d_qk = _D_QK

    # Q/KV 逻辑张量（w8a8_fp8 以 fp8_e4m3fn 近似量化存储）
    q = (torch.randn(s_q, h_q, d_qk, dtype=torch.float16, device=device) / 10).to(
        torch.float8_e4m3fn
    )
    kv = (torch.randn(s_kv, h_kv, d_qk, dtype=torch.float16, device=device) / 10).to(
        torch.float8_e4m3fn
    )
    # 稀疏下标：每 query 采样 topk 个 [0, s_kv) 的位置（randint 近似、可重复）
    indices = torch.randint(
        0, s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=device
    )
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=device)
    return (q, kv, indices, _SM_SCALE), {
        "d_v": _D_V,
        "attn_sink": attn_sink,
    }


def key_shape(binding):
    return [_S_Q, _H_Q, _D_QK, binding["s_kv"], _TOPK]


def config(binding, dtype):
    s_kv = binding["s_kv"]
    return {
        "inputs": {
            "q": {"shape": [_S_Q, _H_Q, _D_QK], "dtype": "torch.float8_e4m3fn"},
            "kv": {"shape": [s_kv, _H_KV, _D_QK], "dtype": "torch.float8_e4m3fn"},
            "indices": {"shape": [_S_Q, _H_KV, _TOPK], "dtype": "torch.int32"},
            "sm_scale": _SM_SCALE,
            "d_v": _D_V,
            "attn_sink": {"shape": [_H_Q], "dtype": "torch.float32"},
        },
        "outputs": {
            "out": {"shape": [_S_Q, _H_Q, _D_V], "dtype": str(dtype)},
        },
        "dims": {
            "s_q": _S_Q, "s_kv": s_kv, "topk": _TOPK, "h_q": _H_Q, "h_kv": _H_KV,
            "d_qk": _D_QK, "d_v": _D_V, "quant": "w8a8_fp8",
            "shape_source": "test_flash_mla_sparse_fwd.py（bf16 基准；fp8 量化为推断）",
        },
    }
