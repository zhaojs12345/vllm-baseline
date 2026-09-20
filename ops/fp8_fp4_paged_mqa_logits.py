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

"""fp8_fp4_paged_mqa_logits baseline（方案 B）。

native：vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits(q, kv_cache, weights,
    context_lens, block_tables, schedule_metadata, max_model_len,
    clean_logits, indices=None) -> logits[B*next_n, max_model_len] fp32
    deep_gemm 惰性绑定 wrapper：调用时先 _lazy_init()，impl 为 None 则 _missing
    抛错，否则转调。native() 只 import 并返回 wrapper；能否真执行取决于运行环境
    是否装了 deep_gemm——未加载时采集器在调用处捕获异常并跳过（符合契约）。
    src: vllm/utils/deep_gemm.py#L608

shape 复刻 FlagGems-vllm 基准 benchmark/test_fp8_fp4_paged_mqa_logits.py：
    DeepSeek-V4 NUM_HEADS=64, HEAD_DIM=128, BLOCK_KV=64, MAX_MODEL_LEN=111*1024。
    FP8 路径：q=(q_fp8[B,next_n,H,D] e4m3, None)；
    kv_cache[num_blocks, BLOCK_KV, 1, D+4] uint8（末 4 字节存 dequant scale）；
    weights[B*next_n, H] fp32；context_lens[B,next_n] int32；
    block_tables[B, max_blocks] int32；schedule_metadata 由
    get_paged_mqa_logits_metadata 生成；clean_logits=False。
"""

import importlib

import torch

OP_NAME = "fp8_fp4_paged_mqa_logits"
DTYPES = [torch.float32]  # 基准以 fp32 计时；operands 量化为 fp8/uint8
IS_INPLACE = False  # 返回新 logits 张量

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

# DeepSeek-V4 model parameters（复刻基准）
_NUM_HEADS = 64
_HEAD_DIM = 128
_BLOCK_KV = 64  # KV cache page size
_MAX_MODEL_LEN = 111 * 1024

# (batch_size, next_n, avg_context_len) —— 基准 BENCH_SHAPES
_SHAPES = [
    (256, 1, 1024),
    (256, 1, 2048),
    (256, 1, 4096),
    (256, 1, 8192),
    (256, 2, 8192),
    (128, 1, 16384),
    (64, 1, 32768),
    (32, 1, 65536),
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def native():
    """解析 vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits wrapper；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm.utils.deep_gemm")
    except ImportError:
        return None
    op = getattr(mod, "fp8_fp4_paged_mqa_logits", None)
    return op if callable(op) else None


def grid():
    return [
        {"batch_size": b, "next_n": nn, "avg_context_len": ctx}
        for (b, nn, ctx) in _SHAPES
    ]


def _kv_cache_cast_to_fp8(x):
    """bf16 KV cache[num_blocks, block_size, 1, D] -> uint8[.., D+4]（复刻基准）。

    末 4 字节/(block,pos) 存 float dequant scale。真 torch 走 amax/scale 打包；
    离线 stub 时直接给最终 shape 的 uint8 占位。
    """
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    if not hasattr(torch, "finfo"):  # stub 路径：直接造最终布局占位
        return torch.empty(
            (num_blocks, block_size, num_heads, head_dim + 4),
            device=x.device, dtype=torch.uint8,
        )
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(_FP8_DTYPE)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device, dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)
    x_fp8[:, block_size * head_dim:] = sf.view(num_blocks, block_size).view(
        torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def _schedule_metadata(context_lens, device):
    """schedule_metadata：真 torch+vllm 走 get_paged_mqa_logits_metadata 复刻基准；
    离线 stub 拿不到时给 [slots+1, 2] int32 占位（下游 CUDA kernel 消费）。"""
    try:
        from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

        num_sms = 132  # Hopper 默认；真机可由 cuda 属性取，此处用基准常量
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "get_device_properties"):
            try:
                num_sms = torch.cuda.get_device_properties(0).multi_processor_count
            except Exception:  # noqa: BLE001
                pass
        return get_paged_mqa_logits_metadata(context_lens, _BLOCK_KV, num_sms)
    except ImportError:
        return torch.zeros(num_sms_slots() + 1, 2, dtype=torch.int32, device=device)


def num_sms_slots():
    # 占位 slot 数（stub 分支用；真值由 get_paged_mqa_logits_metadata 决定）
    return 132


def build_inputs(binding, dtype, device):
    batch_size = binding["batch_size"]
    next_n = binding["next_n"]
    avg_kv = binding["avg_context_len"]

    num_total_blocks = _MAX_MODEL_LEN * 3 // _BLOCK_KV

    q_bf16 = torch.randn(
        (batch_size, next_n, _NUM_HEADS, _HEAD_DIM),
        device=device, dtype=torch.bfloat16,
    )
    kv_cache_bf16 = torch.randn(
        (num_total_blocks, _BLOCK_KV, 1, _HEAD_DIM),
        device=device, dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, _NUM_HEADS), device=device, dtype=torch.float32
    )

    base_ctx = torch.full((batch_size,), avg_kv, device=device, dtype=torch.int32)
    base_ctx = base_ctx.clamp(max=_MAX_MODEL_LEN)
    context_lens = base_ctx.unsqueeze(1).expand(-1, next_n).contiguous()

    q_fp8 = q_bf16.to(_FP8_DTYPE)
    kv_fp8 = _kv_cache_cast_to_fp8(kv_cache_bf16)

    num_blocks_per_query = _ceil_div(avg_kv, _BLOCK_KV)
    block_table = torch.zeros(
        (batch_size, num_blocks_per_query), device=device, dtype=torch.int32
    )
    if hasattr(torch, "randperm"):
        pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
        offset = 0
        for i in range(batch_size):
            n = num_blocks_per_query
            if offset + n > num_total_blocks:
                pool = torch.randperm(
                    num_total_blocks, device=device, dtype=torch.int32
                )
                offset = 0
            block_table[i, :n] = pool[offset:offset + n]
            offset += n

    schedule_meta = _schedule_metadata(context_lens, device)

    kwargs = {
        "q": (q_fp8, None),
        "kv_cache": kv_fp8,
        "weights": weights,
        "context_lens": context_lens,
        "block_tables": block_table,
        "schedule_metadata": schedule_meta,
        "max_model_len": _MAX_MODEL_LEN,
        "clean_logits": False,
    }
    return (), kwargs


def key_shape(binding):
    return [
        binding["batch_size"],
        binding["next_n"],
        binding["avg_context_len"],
    ]


def config(binding, dtype):
    batch_size = binding["batch_size"]
    next_n = binding["next_n"]
    avg_kv = binding["avg_context_len"]
    M = batch_size * next_n
    fp8 = str(_FP8_DTYPE)
    num_blocks_per_query = _ceil_div(avg_kv, _BLOCK_KV)
    return {
        "inputs": {
            "q_values": {"shape": [batch_size, next_n, _NUM_HEADS, _HEAD_DIM],
                         "dtype": fp8, "note": "FP8 路径 q_scale=None"},
            "kv_cache": {"shape": ["num_blocks", _BLOCK_KV, 1, _HEAD_DIM + 4],
                         "dtype": "torch.uint8",
                         "note": "末 4 字节/(block,pos) 存 float dequant scale"},
            "weights": {"shape": [M, _NUM_HEADS], "dtype": "torch.float32"},
            "context_lens": {"shape": [batch_size, next_n], "dtype": "torch.int32"},
            "block_tables": {"shape": [batch_size, num_blocks_per_query],
                             "dtype": "torch.int32"},
            "schedule_metadata": {"shape": ["slots+1", 2], "dtype": "torch.int32",
                                  "note": "get_paged_mqa_logits_metadata 生成"},
            "max_model_len": {"scalar": _MAX_MODEL_LEN},
            "clean_logits": {"scalar": False},
        },
        "outputs": {
            "logits": {"shape": [M, _MAX_MODEL_LEN], "dtype": "torch.float32"},
        },
        "dims": {"batch_size": batch_size, "next_n": next_n,
                 "avg_context_len": avg_kv, "H": _NUM_HEADS, "D": _HEAD_DIM,
                 "BLOCK_KV": _BLOCK_KV, "MAX_MODEL_LEN": _MAX_MODEL_LEN},
        "note": "shape 复刻 FlagGems-vllm benchmark/test_fp8_fp4_paged_mqa_logits.py",
    }
