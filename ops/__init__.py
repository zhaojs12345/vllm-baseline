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

"""方案 B（自定义 ops）：每个算子一个模块，自带 native 调用坐标、shape 网格、
输入构造与主键 shape。采集器（tools/collect_baseline_nvidia.py）按下述契约驱动。

每个算子模块须导出：

    OP_NAME : str
        算子名（写入 JSON / 报告命名 / 与验收看板算子名对应）。
    DTYPES : list[torch.dtype]
        遍历的数据类型。
    IS_INPLACE : bool
        native 是否原地写回输入（原地算子在 NCU 采样前需重建实参，
        避免 warmup 的原地累积污染数值）。
    native() -> callable | None
        解析并返回 NV 原生 callable；解析不到返回 None（该算子无基准，跳过）。
    grid() -> list[dict]
        采集点绑定列表（已展开的维度网格），每个 dict 是一组维度取值。
    build_inputs(binding, dtype, device) -> (args, kwargs)
        按对应 benchmark 的 input_fn 逻辑构造 native 调用实参。
        native 以 op(*args, **kwargs) 调用。
    key_shape(binding) -> list[int]
        该采集点在 JSON 里的主键 shape（对应 benchmark 的 shape）。

注意：build_inputs 会在采集器主进程与 NCU 子进程中各调用一次（子进程按
OP_NAME import 本模块重建输入），因此它必须是无副作用、可重复调用的纯构造。
"""
