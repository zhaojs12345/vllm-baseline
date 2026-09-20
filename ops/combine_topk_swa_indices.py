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

"""combine_topk_swa_indices baseline（方案 B）。

native：DeepseekV4 稀疏预填充 topk+SWA 索引合并的 Triton wrapper。
    - benchmark 记载的公开入口是
      vllm.v1.attention.ops.deepseek_v4_ops.combine_topk_swa_indices；
      本地 vllm 检出（76ba32160a）里 **不存在** 该模块。
    - 实际符号在
      vllm/models/deepseek_v4/common/ops/cache_utils.py#L528
      def combine_topk_swa_indices(topk_indices, query_start_loc, seq_lens,
          gather_lens, window_size, compress_ratio, topk, M, N, out=None)
          -> tuple[Tensor, Tensor]
      （亦经 vllm/models/deepseek_v4/common/ops/__init__.py 再导出）。
      非原地：out=None 时新分配 combined_indices/combined_lens。
    native() 先试公开路径，再回落 __init__ 再导出路径，最后回落 cache_utils；
    都取不到返回 None。

输入构造复刻
FlagGems-vllm/benchmark/test_deepseek_v4_attention_combine_topk_swa_indices.py
的 get_input_iter（dtype=int32）：
    shape 元组 = (query_lens, seq_lens_values, gather_lens_values,
                  topk, window_size, compress_ratio, M, N)
    num_tokens   = sum(query_lens)
    topk_indices = randint(-1, max(N,1), (num_tokens, topk))  int32
    query_start_loc = cumulative([0]+query_lens)              int32
    seq_lens     = tensor(seq_lens_values)                    int32
    gather_lens  = tensor(gather_lens_values)                 int32
    传参顺序：(topk_indices, query_start_loc, seq_lens, gather_lens,
              window_size, compress_ratio, topk, M, N)
shape 取该 benchmark set_shapes 里的 7 组档位。
"""

import importlib

import torch

OP_NAME = "combine_topk_swa_indices"
DTYPES = [torch.int32]
IS_INPLACE = False  # out=None → 新分配 combined_indices/combined_lens

# 依次尝试：公开路径 → 包再导出 → 实际源码模块。
_CANDIDATES = [
    ("vllm.v1.attention.ops.deepseek_v4_ops", "combine_topk_swa_indices"),
    ("vllm.models.deepseek_v4.common.ops", "combine_topk_swa_indices"),
    ("vllm.models.deepseek_v4.common.ops.cache_utils",
     "combine_topk_swa_indices"),
]

# benchmark set_shapes：
# (query_lens, seq_lens, gather_lens, topk, window_size, compress_ratio, M, N)
_SHAPES = [
    ([3, 2], [6, 4], [4, 3], 4, 4, 2, 20, 8),
    ([128], [512], [256], 32, 128, 4, 42240, 40960),
    ([512, 256], [2048, 1024], [1024, 512], 64, 256, 4, 45056, 40960),
    ([4096], [4096], [4096], 128, 256, 4, 45056, 40960),
    ([1024, 1024], [8192, 4096], [2048, 1024], 128, 256, 4, 45056, 40960),
    ([128], [4096], [512], 32, 256, 128, 5632, 1280),
    ([4096], [4096], [4096], 128, 256, 128, 8448, 1280),
]


def native():
    """依次尝试公开路径 / 包再导出 / 源码模块，取出 combine_topk_swa_indices。"""
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
    out = []
    for (ql, sl, gl, topk, ws, cr, M, N) in _SHAPES:
        out.append({
            "query_lens": list(ql),
            "seq_lens_values": list(sl),
            "gather_lens_values": list(gl),
            "topk": topk,
            "window_size": ws,
            "compress_ratio": cr,
            "M": M,
            "N": N,
        })
    return out


def build_inputs(binding, dtype, device):
    query_lens = binding["query_lens"]
    topk = binding["topk"]
    N = binding["N"]
    num_tokens = sum(query_lens)
    topk_indices = torch.randint(
        -1, max(N, 1), (num_tokens, topk), dtype=torch.int32, device=device
    )
    query_start_values = [0]
    for query_len in query_lens:
        query_start_values.append(query_start_values[-1] + query_len)
    query_start_loc = torch.tensor(
        query_start_values, dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor(
        binding["seq_lens_values"], dtype=torch.int32, device=device
    )
    gather_lens = torch.tensor(
        binding["gather_lens_values"], dtype=torch.int32, device=device
    )
    args = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        binding["window_size"],
        binding["compress_ratio"],
        topk,
        binding["M"],
        N,
    )
    return args, {}


def key_shape(binding):
    num_tokens = sum(binding["query_lens"])
    return [
        num_tokens,
        binding["topk"],
        binding["window_size"],
        binding["compress_ratio"],
        binding["M"],
        binding["N"],
    ]


def config(binding, dtype):
    num_tokens = sum(binding["query_lens"])
    topk = binding["topk"]
    ws = binding["window_size"]
    num_reqs = len(binding["query_lens"])
    # combined_topk = ceil((topk + window_size)/128)*128（源码
    # _SPARSE_PREFILL_TOPK_ALIGNMENT=128）。
    align = 128
    combined_topk = (topk + ws + align - 1) // align * align
    return {
        "inputs": {
            "topk_indices": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "query_start_loc": {"shape": [num_reqs + 1], "dtype": "torch.int32"},
            "seq_lens": {"shape": [num_reqs], "dtype": "torch.int32"},
            "gather_lens": {"shape": [num_reqs], "dtype": "torch.int32"},
            "window_size": {"scalar": ws},
            "compress_ratio": {"scalar": binding["compress_ratio"]},
            "topk": {"scalar": topk},
            "M": {"scalar": binding["M"]},
            "N": {"scalar": binding["N"]},
        },
        "outputs": {
            "combined_indices": {"shape": [num_tokens, combined_topk],
                                 "dtype": "torch.int32"},
            "combined_lens": {"shape": [num_tokens], "dtype": "torch.int32"},
        },
        "dims": {
            "num_tokens": num_tokens,
            "num_reqs": num_reqs,
            "topk": topk,
            "window_size": ws,
            "combined_topk": combined_topk,
        },
    }
