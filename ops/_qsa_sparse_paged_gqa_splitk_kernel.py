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

"""_qsa_sparse_paged_gqa_splitk_kernel baseline（方案 B）——QSA 稀疏分页 GQA split-K 内核。

接口：vllm.models.qwen4_exp.nvidia.ops.qsa._qsa_sparse_paged_gqa_splitk_kernel
    （src: vllm/models/qwen4_exp/nvidia/ops/qsa.py#L13）。

native() 返回 None（有意跳过采集）——原因：
    该符号是 @triton.jit 装饰的内部 kernel，只能经 kernel[grid](ptr...) 启动
    （launch 点 qsa.py#L496），不是可直接 native(*args) 调用的 Python 接口。
    其入参为裸指针 + stride + tl.constexpr（TOPK/PAGE_SIZE/GROUP_SIZE/HEAD_DIM/
    NUM_QUERY_HEADS/NUM_SPLITS/NUM_TILES/BLOCK_M/BLOCK_N 等），需由上层封装
    qsa_sparse_paged_attention 组织 partial_output/partial_lse split-K 缓冲后启动。
    CSV 亦标注「未找到对应benchmark」「@triton.jit 内部 kernel，非公开 Python 接口」。
    如需采集该 kernel，应针对其公开封装 qsa_sparse_paged_attention 建模。

保留本模块用于契约完整性与文档留痕；native() is None 时采集器整算子跳过。
grid/build_inputs 仅记录 split-K 启动的维度语义占位。
"""

import importlib

import torch

OP_NAME = "_qsa_sparse_paged_gqa_splitk_kernel"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # 写 partial_output_ptr / partial_lse_ptr / output_ptr

HEAD_DIM = 128
NUM_QUERY_HEADS = 64
# (num_rows, topk, page_size) —— 仅文档占位维度
_CASES = [
    (16, 2048, 64),
    (64, 2048, 64),
]


def native():
    """返回 None：@triton.jit 内部 kernel，无可直调 Python 接口，见模块 docstring。"""
    try:
        mod = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
        _ = getattr(mod, "_qsa_sparse_paged_gqa_splitk_kernel", None)
    except ImportError:
        pass
    return None


def grid():
    return [{"num_rows": r, "topk": t, "page_size": p} for (r, t, p) in _CASES]


def build_inputs(binding, dtype, device):
    # native() is None，采集器不会调用；此处仅构造 q 占位以保契约可跑通。
    r = binding["num_rows"]
    q = torch.randn(r, NUM_QUERY_HEADS, HEAD_DIM, dtype=dtype, device=device)
    return (q,), {}


def key_shape(binding):
    return [binding["num_rows"], binding["topk"], binding["page_size"],
            NUM_QUERY_HEADS, HEAD_DIM]


def config(binding, dtype):
    return {
        "dims": {"num_rows": binding["num_rows"], "topk": binding["topk"],
                 "page_size": binding["page_size"],
                 "query_heads": NUM_QUERY_HEADS, "head_dim": HEAD_DIM,
                 "resolvable": False,
                 "skip_reason": "@triton.jit 内部 kernel，仅 kernel[grid] 启动，"
                                "无可直调 Python 接口；应针对封装 "
                                "qsa_sparse_paged_attention 采集",
                 "shape_source": "vllm 源码推断（无 benchmark）"},
    }
