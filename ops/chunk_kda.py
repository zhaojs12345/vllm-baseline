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

"""chunk_kda baseline（方案 B）——FLA KDA 分块前向。

native：vllm.third_party.flash_linear_attention.ops.kda.chunk_kda
    （src: kda.py#L1506）。签名（已回源核对）：
        chunk_kda(q, k, v, g, beta, scale=None, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=False,
            cu_seqlens=None, **kwargs)
    注意：vLLM native chunk_kda 直接收「已成形的 gate g」，而 benchmark 的 gems_op
    （flaggems_vllm.ops.FLA.chunk_kda）走 use_gate_in_kernel 用 raw g + A_log/dt_bias
    在核内算门控。二者接口不同——这里按 **vLLM native 签名** 传参：g 作为门控张量、
    scale/initial_state/output_final_state/use_qk_l2norm_in_kernel/cu_seqlens 显式传，
    不传 gems 专用的 A_log/dt_bias/use_gate_in_kernel 等（native 的 **kwargs 会忽略，
    但为清晰不传）。

输入构造复刻 FlagGems-vllm/benchmark/test_FLA/test_chunk_kda.py 的 _build_inputs：
    H=96, D=128；seq_lens 三档 [[8192], [1300,547,2048,963,271,3063], [1024]*8]。
    T_total = sum(seq_lens)，N = len(seq_lens)，scale = 1/sqrt(D)。
    q = F.normalize(randn(1, T_total, H, D), p=2, dim=-1)
    k = F.normalize(randn(1, T_total, H, D), p=2, dim=-1)
    v = randn(1, T_total, H, D)
    g = randn(1, T_total, H, D)
    beta = randn(1, T_total, H)
    initial_state = arange(N*H*D*D).reshape(N,H,D,D)
    cu_seqlens = [0]+cumsum(seq_lens)（N>1 时；N==1 为 None）
shape 来源：test_chunk_kda.py FLASHKDA_CASES × (H=96, D=128)。
"""

import importlib
import math

import torch

OP_NAME = "chunk_kda"
DTYPES = [torch.bfloat16]  # benchmark DEFAULT_DTYPES
IS_INPLACE = False

_H = 96
_D = 128
# seq_lens 分档（test_chunk_kda.py FLASHKDA_CASES）
_SEQ_LENS_CASES = [
    [8192],
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]


def _l2norm_lastdim(x):
    """沿最后一维做 L2 归一（等价 F.normalize(p=2, dim=-1)）。"""
    denom = (x.float().pow(2).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
    return (x.float() / denom).to(x.dtype)


def native():
    """解析 FLA kda.chunk_kda；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.third_party.flash_linear_attention.ops.kda"
        )
    except ImportError:
        return None
    op = getattr(mod, "chunk_kda", None)
    return op if callable(op) else None


def grid():
    return [{"case": i} for i in range(len(_SEQ_LENS_CASES))]


def build_inputs(binding, dtype, device):
    seq_lens = _SEQ_LENS_CASES[binding["case"]]
    H, D = _H, _D
    T_total = sum(seq_lens)
    N = len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    q = _l2norm_lastdim(
        torch.randn((1, T_total, H, D), dtype=torch.float32, device=device)
    ).to(dtype)
    k = _l2norm_lastdim(
        torch.randn((1, T_total, H, D), dtype=torch.float32, device=device)
    ).to(dtype)
    v = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
    g = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
    beta = torch.randn((1, T_total, H), dtype=dtype, device=device)
    initial_state = (
        torch.arange(N * H * D * D, dtype=torch.float32, device=device)
        .reshape(N, H, D, D)
        .to(dtype)
    )
    cu_seqlens = None
    if N > 1:
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seq_lens).cumsum(dim=0).tolist()),
            dtype=torch.long,
            device=device,
        )
    kwargs = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": cu_seqlens,
    }
    return (q, k, v, g, beta), kwargs


def key_shape(binding):
    seq_lens = _SEQ_LENS_CASES[binding["case"]]
    return [1, sum(seq_lens), _H, _D, len(seq_lens)]


def config(binding, dtype):
    seq_lens = _SEQ_LENS_CASES[binding["case"]]
    H, D = _H, _D
    T_total = sum(seq_lens)
    N = len(seq_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [1, T_total, H, D], "dtype": dt},
            "k": {"shape": [1, T_total, H, D], "dtype": dt},
            "v": {"shape": [1, T_total, H, D], "dtype": dt},
            "g": {"shape": [1, T_total, H, D], "dtype": dt},
            "beta": {"shape": [1, T_total, H], "dtype": dt},
            "initial_state": {"shape": [N, H, D, D], "dtype": dt},
            "scale": 1.0 / math.sqrt(D),
        },
        "outputs": {
            "o": {"shape": [1, T_total, H, D], "dtype": dt},
            "final_state": {"shape": [N, H, D, D], "note": "output_final_state=True"},
        },
        "dims": {"T_total": T_total, "H": H, "D": D, "N_seqs": N,
                 "seq_lens": seq_lens},
    }
