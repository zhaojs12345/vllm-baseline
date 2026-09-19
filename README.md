# vllm-baseline

**Language:** English · [中文](README.zh.md)

Collects performance baselines for NVIDIA native operators (vLLM / CUDA kernels), used for cross-platform acceptance of domestic backends. Field conventions follow `算子后端缺失性能基准方案参考.md`.

## Layout

| Path | Description |
|---|---|
| `tools/collect_baseline_nvidia.py` | Collector: measures latency + captures FLOPs/bytes/utilization via Nsight Compute (ncu) |
| `tools/baseline_shape.yaml` | Shape config for declarative ops (simple, positional args only) |
| `ops/` | Custom op modules (complex ops: multi-tensor, constraint tensors, quantization…), one file per op |
| `tools/hardware_specs.py` | Hardware peak FLOPS / bandwidth specs |

## Run

```bash
# Full collection (latency + NCU), writes JSON
python tools/collect_baseline_nvidia.py --output op_perf_baseline.json

# Latency only, skip NCU
python tools/collect_baseline_nvidia.py --no-ncu

# Custom ncu report dir (.ncu-rep opens in ncu-ui)
python tools/collect_baseline_nvidia.py --report-dir ncu_reports
```

Requires: CUDA GPU, `vllm`, `triton`, `ncu` (Nsight Compute). The collector scans `ops/` first, falling back to `baseline_shape.yaml`; on name clashes, `ops/` wins.

## Adding an operator

Create a module under `ops/` exporting the contract fields (`OP_NAME` / `DTYPES` / `IS_INPLACE` / `native()` / `grid()` / `build_inputs()` / `key_shape()`, plus optional `config()` for complex ops). See `ops/__init__.py` for the full contract, and `ops/fused_add_rms_norm.py` / `ops/grouped_topk.py` as references.

## Output format

`{op: {native_api, shapes: {shape: {dtype: record}}}}`. Each record contains (aligned with the reference doc's symbols):

| Field | Meaning | Unit |
|---|---|---|
| `T_us` | kernel runtime T | us |
| `F_cuda` / `F_tensor` | CUDA / Tensor Core actual compute | FLOP |
| `B_mem` | Global Memory actual traffic | Byte |
| `U_cuda` / `U_tensor` / `U_mem` | per-unit hardware utilization | % (0–100) |
| `U_bottle_neck` / `bottle_neck_unit` | bottleneck utilization / unit | % / `cuda`\|`tensor`\|`mem` |
| `num_kernels` / `per_kernel` | kernels per call / per-kernel breakdown | — |
| `config` | real input/output shapes (complex ops) | — |

Plus a set of redundant legacy-compatible fields (`flops_*`, `util_*`, `bottleneck`, …) with identical values.
