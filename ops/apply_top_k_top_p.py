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

"""apply_top_k_top_p baseline（方案 B）——采样阶段 top-k/top-p 掩码。

native：vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p
    （src: topk_topp_sampler.py#L349）。签名（已回源核对）：
        apply_top_k_top_p(logits, k, p) -> logits
    分派逻辑：非 CPU 且 HAS_TRITON 且 logits.shape[0] >= 8 → apply_top_k_top_p_triton；
    否则 pytorch 排序回退。为触发 Triton 路径，batch 取 >= 8。

shape 为 vllm 源码推断，非 FlagGems-vllm 基准。
    benchmark 目录无对应用例（CSV「未找到对应benchmark」）。按采样调用点维度推断：
    logits: [batch, vocab_size]（fp32，采样前 logits）；
    k: int32 [batch]（每序列 top-k）；p: fp32 [batch]（每序列 top-p，(0,1]）。
    vocab 取常见 LLM 词表档 [32000, 129280(DeepSeek-V3), 151936(Qwen)]，
    batch ∈ [8, 64, 256]（>=8 命中 triton 路径）。
"""

import importlib

import torch

OP_NAME = "apply_top_k_top_p"
DTYPES = [torch.float32]  # logits 采样普遍 fp32
IS_INPLACE = False  # pytorch 回退可能原地改 logits，triton 路径返回新张量；按非原地采样

# (batch, vocab_size) —— batch>=8 命中 triton 路径
_BATCHES = [8, 64, 256]
_VOCABS = [32000, 129280, 151936]


def native():
    """解析 topk_topp_sampler.apply_top_k_top_p；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    except ImportError:
        return None
    op = getattr(mod, "apply_top_k_top_p", None)
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "vocab": v}
        for v in _VOCABS
        for b in _BATCHES
    ]


def build_inputs(binding, dtype, device):
    batch, vocab = binding["batch"], binding["vocab"]
    logits = torch.randn(batch, vocab, dtype=dtype, device=device)
    # top-k：每序列 [1, vocab] 的整数；top-p：每序列 (0,1] 的概率阈值
    k = torch.randint(1, min(vocab, 1024) + 1, (batch,), dtype=torch.int32, device=device)
    p = torch.rand(batch, dtype=torch.float32, device=device).clamp_(min=0.1, max=1.0)
    return (logits, k, p), {}


def key_shape(binding):
    return [binding["batch"], binding["vocab"]]


def config(binding, dtype):
    batch, vocab = binding["batch"], binding["vocab"]
    dt = str(dtype)
    return {
        "inputs": {
            "logits": {"shape": [batch, vocab], "dtype": dt},
            "k": {"shape": [batch], "dtype": "torch.int32"},
            "p": {"shape": [batch], "dtype": "torch.float32"},
        },
        "outputs": {"logits": {"shape": [batch, vocab], "dtype": dt}},
        "dims": {"batch": batch, "vocab": vocab,
                 "path": "triton (batch>=8)",
                 "shape_source": "vllm 源码推断（无 benchmark）"},
    }
