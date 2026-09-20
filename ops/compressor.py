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

"""compressor baseline（方案 B）——DeepSeek-V4 KV/score 压缩模块。

接口：vllm.models.deepseek_v4.compressor.DeepseekCompressor
    （src: vllm/models/deepseek_v4/compressor.py#L193/#L309）。
    调用形态 DeepseekCompressor(vllm_config, compress_ratio, hidden_size,
        head_dim, ...).forward(kv_score, positions, rotary_emb)。

native() 返回 None（有意跳过采集）——原因：
    1. 非单一算子：DeepseekCompressor 是 nn.Module，forward 内部依次启动
       save_partial_states / compress_norm_rope_store_* 等多个 kernel，且依赖
       get_forward_context() 提供的 attn_metadata（CompressorMetadata：
       token_to_req_indices / slot_mapping / block_table / block_size / state_cache）；
       非 forward_context 时 forward 直接 return（compressor.py#L325）。
    2. 构造需真实 VllmConfig（含 hf_config、scheduler_config、model_config），
       无法在 baseline 采集侧脱离引擎独立构造。
    CSV 亦标注「未找到对应benchmark」「仅在 fused_indexer_q_rope_quant.py 出现」。
    如需单核基准，应针对其内部 kernel（save_partial_states / 压缩存储 kernel）
    单独建模，而非整个压缩模块。

保留本模块用于契约完整性与文档留痕；grid/build_inputs 仅提供 forward 输入的
shape 语义（kv_score: [num_tokens, 2*coff*head_dim]，positions: [num_tokens]），
不会被采集器实际调用（native() is None 时整算子跳过）。
"""

import importlib

import torch

OP_NAME = "compressor"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # forward 将压缩结果写入 state_cache（原地）

HEAD_DIM = 128
COMPRESS_RATIO = 4
COFF = 2  # overlap = (compress_ratio == 4) -> coff = 1 + 1 = 2
_NUM_TOKENS = [512, 2048, 8192]


def native():
    """返回 None：DeepseekCompressor 为压缩模块而非单一 callable，见模块 docstring。

    仍做一次 import 探测以确认符号存在（存在与否都返回 None，采集器据此跳过）。
    """
    try:
        mod = importlib.import_module("vllm.models.deepseek_v4.compressor")
        _ = getattr(mod, "DeepseekCompressor", None)
    except ImportError:
        pass
    return None


def grid():
    return [{"num_tokens": n} for n in _NUM_TOKENS]


def build_inputs(binding, dtype, device):
    n = binding["num_tokens"]
    # forward(kv_score, positions, rotary_emb)
    kv_score = torch.randn(n, 2 * COFF * HEAD_DIM, dtype=dtype, device=device)
    positions = torch.arange(n, dtype=torch.int64, device=device)
    # rotary_emb 为模块对象；此处占位 None（native() is None，不会真正调用）。
    return (kv_score, positions, None), {}


def key_shape(binding):
    return [binding["num_tokens"], 2 * COFF * HEAD_DIM]


def config(binding, dtype):
    n = binding["num_tokens"]
    return {
        "inputs": {
            "kv_score": {"shape": [n, 2 * COFF * HEAD_DIM], "dtype": str(dtype)},
            "positions": {"shape": [n], "dtype": "torch.int64"},
        },
        "dims": {"num_tokens": n, "head_dim": HEAD_DIM, "coff": COFF,
                 "compress_ratio": COMPRESS_RATIO,
                 "resolvable": False,
                 "skip_reason": "DeepseekCompressor 为多 kernel nn.Module，"
                                "依赖 forward_context 与 VllmConfig，无法独立采集",
                 "shape_source": "vllm 源码推断（无 benchmark）"},
    }
