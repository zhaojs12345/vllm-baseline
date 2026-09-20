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

"""compute_global_topk_indices_and_lens baseline（方案 B）。

native：DeepseekV4 局部 topk 索引→全局 KV slot 映射 + 有效计数的 Triton
    wrapper。
    - benchmark 记载的公开入口是
      vllm.v1.attention.ops.deepseek_v4_ops
          .compute_global_topk_indices_and_lens；
      本地 vllm 检出（76ba32160a）里 **不存在** 该模块。
    - 实际符号在
      vllm/models/deepseek_v4/common/ops/cache_utils.py#L435
      def compute_global_topk_indices_and_lens(topk_indices,
          token_to_req_indices, block_table, block_size, is_valid_token)
          -> tuple[Tensor, Tensor]
      （亦经 common/ops/__init__.py 再导出）。
      非原地：新分配 global_topk_indices(empty_like) 与 topk_lens。
    native() 先试公开路径，再回落 __init__ 再导出，最后回落 cache_utils；
    都取不到返回 None。

输入构造复刻
FlagGems-vllm/benchmark/
test_deepseek_v4_attention_compute_global_topk_indices_and_lens.py 的
get_input_iter（dtype=int32）：
    shape = (num_tokens, topk, num_reqs, blocks_per_req, block_size)
    topk_indices = randint(-1, blocks_per_req*block_size, (num_tokens, topk))
    token_to_req_indices = arange(num_tokens) % num_reqs           int32
    block_table = arange(num_reqs*blocks_per_req).view(num_reqs, blocks_per_req)
    is_valid_token = ones(num_tokens)                              int32
    传参顺序：(topk_indices, token_to_req_indices, block_table, block_size,
              is_valid_token)
shape 取该 benchmark set_shapes 里的 6 组档位。
"""

import importlib

import torch

OP_NAME = "compute_global_topk_indices_and_lens"
DTYPES = [torch.int32]
IS_INPLACE = False  # 新分配 global_topk_indices / topk_lens

_CANDIDATES = [
    ("vllm.v1.attention.ops.deepseek_v4_ops",
     "compute_global_topk_indices_and_lens"),
    ("vllm.models.deepseek_v4.common.ops",
     "compute_global_topk_indices_and_lens"),
    ("vllm.models.deepseek_v4.common.ops.cache_utils",
     "compute_global_topk_indices_and_lens"),
]

# benchmark set_shapes：(num_tokens, topk, num_reqs, blocks_per_req, block_size)
_SHAPES = [
    (5, 4, 2, 4, 64),
    (128, 32, 1, 64, 64),
    (512, 64, 2, 128, 64),
    (4096, 128, 1, 640, 64),
    (4096, 128, 4, 640, 64),
    (8192, 128, 8, 1280, 64),
]


def native():
    """依次尝试公开路径 / 包再导出 / 源码模块。"""
    for module, symbol in _CANDIDATES:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        op = getattr(mod, symbol, None)
        if callable(op):
            return op
    return None


def grid():
    return [
        {
            "num_tokens": nt,
            "topk": tk,
            "num_reqs": nr,
            "blocks_per_req": bpr,
            "block_size": bs,
        }
        for (nt, tk, nr, bpr, bs) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    nt = binding["num_tokens"]
    tk = binding["topk"]
    nr = binding["num_reqs"]
    bpr = binding["blocks_per_req"]
    bs = binding["block_size"]
    topk_indices = torch.randint(
        -1, bpr * bs, (nt, tk), dtype=torch.int32, device=device
    )
    token_to_req_indices = (
        torch.arange(nt, dtype=torch.int32, device=device) % nr
    )
    block_table = torch.arange(
        nr * bpr, dtype=torch.int32, device=device
    ).view(nr, bpr)
    is_valid_token = torch.ones((nt,), dtype=torch.int32, device=device)
    args = (topk_indices, token_to_req_indices, block_table, bs, is_valid_token)
    return args, {}


def key_shape(binding):
    return [
        binding["num_tokens"],
        binding["topk"],
        binding["num_reqs"],
        binding["blocks_per_req"],
        binding["block_size"],
    ]


def config(binding, dtype):
    nt = binding["num_tokens"]
    tk = binding["topk"]
    nr = binding["num_reqs"]
    bpr = binding["blocks_per_req"]
    bs = binding["block_size"]
    return {
        "inputs": {
            "topk_indices": {"shape": [nt, tk], "dtype": "torch.int32"},
            "token_to_req_indices": {"shape": [nt], "dtype": "torch.int32"},
            "block_table": {"shape": [nr, bpr], "dtype": "torch.int32"},
            "block_size": {"scalar": bs},
            "is_valid_token": {"shape": [nt], "dtype": "torch.int32"},
        },
        "outputs": {
            "global_topk_indices": {"shape": [nt, tk], "dtype": "torch.int32"},
            "topk_lens": {"shape": [nt], "dtype": "torch.int32"},
        },
        "dims": {
            "num_tokens": nt,
            "topk": tk,
            "num_reqs": nr,
            "blocks_per_req": bpr,
            "block_size": bs,
        },
    }
