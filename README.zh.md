# vllm-baseline

**语言：** 中文 · [English](README.md)

采集 NVIDIA 原生算子（vLLM / CUDA kernel）的性能 baseline，供国产后端做跨平台性能验收。字段口径对齐《算子后端缺失性能基准方案参考.md》。

## 目录结构

| 路径 | 说明 |
|---|---|
| `tools/collect_baseline_nvidia.py` | 采集主程序：测 latency + 用 Nsight Compute（ncu）抓计算量/访存量/利用率 |
| `tools/baseline_shape.yaml` | 声明式算子的 shape 配置（简单算子，纯位置参数） |
| `ops/` | 自定义算子模块（复杂算子：多张量、约束张量、量化等），每个算子一个文件 |
| `tools/hardware_specs.py` | 硬件峰值算力/带宽参数 |

## 运行

```bash
# 完整采集（latency + NCU），输出 JSON
python tools/collect_baseline_nvidia.py --output op_perf_baseline.json

# 只测 latency，跳过 NCU
python tools/collect_baseline_nvidia.py --no-ncu

# 指定 ncu 报告目录（.ncu-rep 可用 ncu-ui 打开）
python tools/collect_baseline_nvidia.py --report-dir ncu_reports
```

依赖：CUDA GPU、`vllm`、`triton`、`ncu`(Nsight Compute)。采集器优先扫描 `ops/`，其余算子回落到 `baseline_shape.yaml`；两者同名时以 `ops/` 为准。

## 新增算子

在 `ops/` 下新建一个模块，导出契约字段（`OP_NAME` / `DTYPES` / `IS_INPLACE` / `native()` / `grid()` / `build_inputs()` / `key_shape()`，复杂算子可选 `config()`）。完整契约见 `ops/__init__.py`，可参考 `ops/fused_add_rms_norm.py`、`ops/grouped_topk.py`。

## 输出格式

`{算子: {native_api, shapes: {shape: {dtype: 记录}}}}`。每条记录含（对齐参考文档符号）：

| 字段 | 含义 | 单位 |
|---|---|---|
| `T_us` | kernel 运行时间 T | us |
| `F_cuda` / `F_tensor` | CUDA / Tensor Core 实际计算量 | FLOP |
| `B_mem` | Global Memory 实际访存量 | Byte |
| `U_cuda` / `U_tensor` / `U_mem` | 三维硬件利用率 | %（0–100）|
| `U_bottle_neck` / `bottle_neck_unit` | 瓶颈侧利用率 / 瓶颈单元 | % / `cuda`\|`tensor`\|`mem` |
| `num_kernels` / `per_kernel` | 一次调用的 kernel 数 / 逐 kernel 明细 | — |
| `config` | 复杂算子的真实输入输出 shape | — |

另含一组同值冗余的兼容旧字段（`flops_*`、`util_*`、`bottleneck` 等）。
