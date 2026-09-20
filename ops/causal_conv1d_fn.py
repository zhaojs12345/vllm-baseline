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

"""causal_conv1d_fn baseline（方案 B）——Mamba Prefill 因果卷积。

native：vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn
    （src: causal_conv1d.py#L481）。签名（已回源核对）：
        causal_conv1d_fn(x, weight, bias, conv_states, query_start_loc,
            cache_indices=None, has_initial_state=None, activation="silu",
            pad_slot_id=PAD_SLOT_ID, ...)
    varlen + 连续批：x 为 2D (dim, cu_seq_len)，conv_states 原地更新。

输入构造复刻 FlagGems-vllm/benchmark/test_causal_conv1d_fn.py 的 _varlen_inputs：
    WIDTH=4, PADDING=3, PAD_SLOT_ID=-1；(batch, seqlen, dim) varlen shape。
    x            = 从宽 buffer 切出的 (dim, seqlen)（channel 维取中段避免对齐巧合）
    weight       = randn(dim, WIDTH)
    bias         = randn(dim)
    conv_states  = randn(batch*10, WIDTH-1, dim).transpose(1,2) → (batch*10, dim, WIDTH-1)
    query_start_loc = 累积 seqlens（padded_batch=batch+PADDING 段）
    cache_indices   = [state_indices(batch)] ++ [PAD_SLOT_ID]*PADDING
    has_initial_state = randint bool (padded_batch,)
    注：benchmark 的 gems_op 是 Ascend 后端；此处按 vLLM native 签名以位置+kwargs
    传入（activation="silu", pad_slot_id=-1）。seqlens 是 benchmark 参考实现私用的
    第 8 个元素，vLLM native 不收，故不传。
shape 来源：test_causal_conv1d_fn.py SHAPES（batch,seqlen,dim）。
"""

import importlib
import itertools

import torch

OP_NAME = "causal_conv1d_fn"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # conv_states 原地更新

_WIDTH = 4
_PADDING = 3
_PAD_SLOT_ID = -1

# (batch, seqlen, dim) varlen shapes（test_causal_conv1d_fn.py SHAPES）
_SHAPES = [
    (4, 8, 64),
    (4, 249, 4096),
    (10, 4096, 4096),
]


def native():
    """解析 vllm mamba causal_conv1d_fn；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.model_executor.layers.mamba.ops.causal_conv1d"
        )
    except ImportError:
        return None
    op = getattr(mod, "causal_conv1d_fn", None)
    return op if callable(op) else None


def grid():
    return [{"batch": b, "seqlen": s, "dim": d} for (b, s, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    batch, seqlen, dim = binding["batch"], binding["seqlen"], binding["dim"]
    width, padding, pad_id = _WIDTH, _PADDING, _PAD_SLOT_ID

    padded_batch = batch + padding
    seq = seqlen // padded_batch
    rem = seqlen - seq * (padded_batch - 1)
    seqlens = [rem] + [seq] * (padded_batch - 1)
    query_start_loc = torch.tensor(
        [0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device=device
    )

    # x: 从宽 buffer 切出 (dim, seqlen)，复刻 benchmark 的错位取法
    x = torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=dtype)
    x = x.transpose(1, 2)[:, 4096 : 4096 + dim, :].squeeze(0)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(
        batch * 10, width - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)
    state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    cache_indices = torch.concat(
        [
            state_indices,
            torch.full((padding,), pad_id, dtype=torch.int32, device=device),
        ],
        dim=-1,
    )
    has_initial_state = torch.randint(
        0, 2, (padded_batch,), dtype=torch.bool, device=device
    )
    args = (x, weight, bias, conv_states, query_start_loc)
    kwargs = {
        "cache_indices": cache_indices,
        "has_initial_state": has_initial_state,
        "activation": "silu",
        "pad_slot_id": pad_id,
    }
    return args, kwargs


def key_shape(binding):
    return [binding["batch"], binding["seqlen"], binding["dim"]]


def config(binding, dtype):
    batch, seqlen, dim = binding["batch"], binding["seqlen"], binding["dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [dim, seqlen], "dtype": dt, "note": "varlen (dim, cu_seq_len)"},
            "weight": {"shape": [dim, _WIDTH], "dtype": dt},
            "bias": {"shape": [dim], "dtype": dt},
            "conv_states": {"shape": [batch * 10, dim, _WIDTH - 1], "dtype": dt,
                            "note": "原地更新"},
            "query_start_loc": {"shape": [batch + _PADDING + 1], "dtype": "torch.int32"},
        },
        "outputs": {"out": {"shape": [dim, seqlen], "dtype": dt}},
        "dims": {"batch": batch, "seqlen": seqlen, "dim": dim,
                 "width": _WIDTH, "activation": "silu"},
    }
