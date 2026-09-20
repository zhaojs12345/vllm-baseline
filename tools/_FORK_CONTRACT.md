# ops 模块生成契约（fork 必读）

给 vllm-baseline 的 ops/ 目录新增算子基准模块。每个算子一个文件 `ops/<OP_NAME>.py`。

## 必须遵守
1. **只写你被分配的文件名**，不要碰 ops/ 下别的文件、不要改 CSV、不要改采集器、不要改 tools/ 里已有文件。
2. 每个模块导出契约字段：`OP_NAME`(str)、`DTYPES`(list[torch.dtype])、`IS_INPLACE`(bool)、
   `native()->callable|None`、`grid()->list[dict]`、`build_inputs(binding,dtype,device)->(args,kwargs)`、
   `key_shape(binding)->list|str`；复杂算子再加 `config(binding,dtype)->dict`。
3. 文件顶部放和 ops/moe_sum.py 一样的 Apache License 头 + 模块 docstring，docstring 里写清 native 调用坐标、
   输入构造来源（哪个 benchmark 的 input_fn）、以及 shape 来源。
4. **native() 必须回源码核对**：去 `/Users/baai/Downloads/baai_repo/vllm` 确认 import 路径与符号真实存在、
   签名与调用参数一致。解析不到就让 native() 返回 None（采集器会优雅跳过）。禁止编造不存在的符号。
5. **shape 来源**：
   - 有 benchmark 的：复刻 `/Users/baai/Downloads/baai_repo/FlagGems-vllm/benchmark/test_<...>.py` 的
     input_fn 逻辑与 set_shapes/core_shapes.yaml 里的 shape，docstring 注明来源文件。
   - 无 benchmark 的：从 vllm 源码调用点的张量维度推一组合理 shape，docstring **明确写“shape 为 vllm 源码推断，
     非 FlagGems-vllm 基准”**，不要谎称来自 benchmark。
6. build_inputs 必须无副作用、可重复调用（会在主进程和 NCU 子进程各调一次）。
7. 写完每个文件后跑冒烟：`cd /Users/baai/Downloads/vllm-baseline && python3 tools/_smoke_ops.py <OP_NAME>`，
   必须通过（stub 环境只验证 grid/build_inputs/key_shape/config 能跑通、字段齐全，不验证数值）。
8. 若某算子实在无法落地（如 @triton.jit 内部 kernel 需 kernel[grid](...) 启动、无公开 Python 可调用入口），
   仍建文件：native() 尽力解析该 callable；确实拿不到就返回 None 并在 docstring 说明原因，让它被跳过。
   不要为了“能跑”而编造假接口。

## 参考已完成模块（同款风格）
ops/moe_sum.py, ops/moe_align_block_size.py, ops/topk_softplus_sqrt.py,
ops/persistent_topk.py, ops/grouped_topk.py, ops/per_token_group_quant_fp8.py

## 报告
完成后回一条消息：每个算子一行——OP_NAME | native 是否核实到源码(文件#行) | shape 来源(benchmark 名 / 源码推断) |
冒烟结果 | 存疑点。存疑或未落地的重点标出。
