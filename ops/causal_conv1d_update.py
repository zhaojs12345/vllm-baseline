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

"""causal_conv1d_update baseline（方案 B）——Mamba Decode / 状态更新。

native：vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update
    （src: causal_conv1d.py#L1096）。签名（已回源核对）：
        causal_conv1d_update(x, conv_state, weight, bias=None, activation=None,
            conv_state_indices=None, num_accepted_tokens=None,
            query_start_loc=None, max_query_len=-1, ..., out=None)
    x 可为 [batch, dim] / [batch, dim, seqlen] / [num_tokens, dim]；conv_state 原地更新。

输入构造复刻 FlagGems-vllm/benchmark/test_causal_conv1d_update.py 的 _update_inputs：
    WIDTH=4, PADDING=5, PAD_SLOT_ID=-1；(batch, seqlen, dim) decode shape。
    x            = randn(padded_batch, seqlen, dim).transpose(1,2) → (padded_batch, dim, seqlen)
    weight       = randn(dim, WIDTH)
    bias         = randn(dim)
    conv_states  = randn(10*batch, WIDTH-1, dim).transpose(1,2) → (10*batch, dim, WIDTH-1)
    state_indices= randperm(10*batch)[:batch]
    cache_indices= [state_indices] ++ [PAD_SLOT_ID]*PADDING
    native 调用（对齐 benchmark gems_op _causal_conv1d_update_gems）：
        causal_conv1d_update(x, conv_states, weight, bias, activation="silu",
            conv_state_indices=cache_indices, pad_slot_id=-1)
    注：vLLM native 现签名无 pad_slot_id 形参，改用 null_block_id；为兼容按
    inspect.signature 能力探测决定是否传，缺失则省略（不影响 shape 采样）。
shape 来源：test_causal_conv1d_update.py SHAPES（batch,seqlen,dim）。
"""

import importlib
import inspect

import torch

OP_NAME = "causal_conv1d_update"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # conv_states 原地更新

_WIDTH = 4
_PADDING = 5
_PAD_SLOT_ID = -1

# (batch, seqlen, dim) decode shapes（test_causal_conv1d_update.py SHAPES）
_SHAPES = [
    (3, 1, 2048 + 16),
    (64, 1, 4096),
    (64, 3, 4096),
]


def native():
    """解析 vllm mamba causal_conv1d_update；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.model_executor.layers.mamba.ops.causal_conv1d"
        )
    except ImportError:
        return None
    op = getattr(mod, "causal_conv1d_update", None)
    return op if callable(op) else None


def grid():
    return [{"batch": b, "seqlen": s, "dim": d} for (b, s, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    batch, seqlen, dim = binding["batch"], binding["seqlen"], binding["dim"]
    width, padding = _WIDTH, _PADDING
    padded_batch = batch + padding
    total_entries = 10 * batch

    x = torch.randn(
        padded_batch, seqlen, dim, device=device, dtype=dtype
    ).transpose(1, 2)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)
    state_indices = torch.randperm(
        total_entries, dtype=torch.int32, device=device
    )[:batch]
    cache_indices = torch.concat(
        [
            state_indices,
            torch.full((padding,), _PAD_SLOT_ID, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    kwargs = {"activation": "silu", "conv_state_indices": cache_indices}
    # pad_slot_id 在旧签名存在；新签名用 null_block_id。按能力探测决定是否传。
    op = native()
    if op is not None:
        try:
            params = inspect.signature(op).parameters
            if "pad_slot_id" in params:
                kwargs["pad_slot_id"] = _PAD_SLOT_ID
        except (TypeError, ValueError):
            pass
    return (x, conv_states, weight, bias), kwargs


def key_shape(binding):
    return [binding["batch"], binding["seqlen"], binding["dim"]]


def config(binding, dtype):
    batch, seqlen, dim = binding["batch"], binding["seqlen"], binding["dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [batch + _PADDING, dim, seqlen], "dtype": dt},
            "conv_states": {"shape": [10 * batch, dim, _WIDTH - 1], "dtype": dt,
                            "note": "原地更新"},
            "weight": {"shape": [dim, _WIDTH], "dtype": dt},
            "bias": {"shape": [dim], "dtype": dt},
        },
        "outputs": {"out": {"shape": [batch, dim, seqlen], "dtype": dt}},
        "dims": {"batch": batch, "seqlen": seqlen, "dim": dim,
                 "width": _WIDTH, "activation": "silu"},
    }
