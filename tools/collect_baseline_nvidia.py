#!/usr/bin/env python3
"""NVIDIA baseline 采集脚本。

在 NVIDIA 卡上调用原生 CUDA kernel，用 NCU 抓《普通算子后端缺失性能基准
方案》所需的三维实测数据：CUDA Core / Tensor Core / Global Memory 各自的
实际工作量与硬件利用率，取利用率最高者为瓶颈单元，供跨平台按 80% 达标线
对比。每条记录的字段见 profile_with_ncu 的返回值。
"""

import argparse
import csv
import importlib
import io
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton
import yaml

# ---- NCU metric 名（H800/Hopper 已核对，换架构前用 `ncu --query-metrics` 复核）----

# CUDA Core 计算量 F_c：FP 指令计数，ffma/hfma 每条计 2 FLOP。
FLOP_CUDA_CORE = {
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 2,
}

# Tensor Core 计算量 F_t：counter 名随源 dtype 而变，按 dtype 取（未知则记 0）。
FLOP_TENSOR_CORE = {
    "torch.float16": "sm__ops_path_tensor_src_fp16.sum",
    "torch.bfloat16": "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "torch.float8_e4m3fn": "sm__ops_path_tensor_src_fp8.sum",
    "torch.float8_e5m2": "sm__ops_path_tensor_src_fp8.sum",
    "torch.float64": "sm__ops_path_tensor_src_fp64.sum",
}

# 访存量 M、三维利用率 U_c/U_t/U_m、整体 SM 吞吐（均为 % of peak，elapsed 口径）。
MEMORY_BYTES = "dram__bytes.sum"
UTIL = {
    "cuda_core": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_core": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "memory_bw": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}
SM_UTIL = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
# 每 kernel GPU 执行时长：仅用于把 do_bench 测得的端到端 T 按比例拆到各 kernel
# （多 kernel 复杂算子的 per_kernel 明细），比例口径与单位无关，故不做单位换算。
DURATION = "gpu__time_duration.sum"
# ncu raw csv 里的 kernel 名列。
KERNEL_NAME_COL = "Kernel Name"
# ncu 单位行里字节系列单位 → base 字节乘子。ncu 用十进制（1 Kbyte = 1000 byte，
# 已由 H800 实测数据反推核对）。只对字节量归一；us/%/inst 等保持原值不换算
# （T 另由 do_bench 测；util 本就是 %；duration 仅用作比例）。
_BYTE_UNIT_MULT = {
    "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
}

# 采集哪些算子、每个算子的 native 调用坐标 / shape 网格 / 输入构造，全部由
# 这份 yaml 驱动（方案B），逐步添加算子只改 yaml、不改本脚本。
CONFIG_PATH = Path(__file__).with_name("baseline_shape.yaml")

# 方案B（自定义 ops）：每个算子一个 Python 模块，自带 native()/grid()/
# build_inputs()/key_shape()（契约见 ops/__init__.py）。采集器优先扫描此目录，
# 复杂算子（元组入参、前置 metadata、量化预处理、约束张量等）用 Python 表达，
# yaml 声明式路径保留给纯位置参数的简单算子直到全部迁完。
OPS_DIR = Path(__file__).resolve().parent.parent / "ops"

def resolve_native_op(module, symbol):
    """解析一个 NV 原生 callable；解析不到返回 None（该算子无基准，跳过）。

    benchmark/ 里真调 NV 原生的算子分三族，差别只在符号从哪来，本函数统一处理：
      族A  torch.ops._C.<symbol> / torch.ops._moe_C.<symbol>
           —— 需先 import vllm._custom_ops 触发 torch.ops 命名空间注册
      族B  vllm._custom_ops.<symbol>          （python wrapper）
      族C  vllm.v1.attention.ops.* / vllm.third_party.* 等内部模块.<symbol>

    三族都归约成「按 module 路径 import，再 getattr(symbol)」。族A 只是
    module 恰好是 torch.ops._C 这个已注册命名空间的特例。

    module: 模块路径字符串，如 "torch.ops._C" / "vllm._custom_ops" /
            "vllm.v1.attention.ops.deepseek_v4_ops"
    symbol: 该模块下的算子名。
    """
    # 触发 torch.ops._C / _moe_C 等命名空间注册（族A 依赖，其他族无害）。
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        pass

    # 先按真实模块 import（族B/C）；torch.ops._C 这类是 torch 动态生成的命名空间
    # 对象、并非真模块，import_module 会失败，退回从 torch 起逐段 getattr。
    try:
        mod = importlib.import_module(module)
    except (ImportError, ModuleNotFoundError):
        mod = _resolve_dotted_attr(module)
        if mod is None:
            return None
    op = getattr(mod, symbol, None)
    return op if callable(op) else None


def _resolve_dotted_attr(dotted):
    """把 "torch.ops._C" 这类点号路径按「import 顶层包 + 逐段 getattr」解析成对象。"""
    head, *rest = dotted.split(".")
    try:
        obj = importlib.import_module(head)
    except ImportError:
        return None
    for part in rest:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _expand_grid(grid):
    """把 {M:[...], N:[...]} 笛卡尔积展开成 [{M:m, N:n}, ...] 绑定列表。"""
    dims = list(grid)
    return [dict(zip(dims, combo))
            for combo in itertools.product(*(grid[d] for d in dims))]


def _resolve_dim(token, binding):
    """把一个 shape 维度 token 解析成具体 int：int 直接用；字符串支持
    维度名（如 "N"）与「系数*维度名」（如 "2*N"）。"""
    if isinstance(token, int):
        return token
    token = str(token).strip()
    if "*" in token:
        coeff, name = token.split("*", 1)
        return int(coeff) * binding[name.strip()]
    return binding[token] if token in binding else int(token)


def _resolve_inputs(input_specs, binding, dtype_str):
    """把 yaml 的 inputs 规范 + 一组维度绑定，解析成可实例化的具体描述。
    张量 -> {"tensor": [具体int形状], "dtype": ..., "fill": ..., low/high?}；
    标量 -> {"scalar": v}。tensor 的 dtype 缺省用算子遍历的 dtype_str，
    fill 缺省 randn。"""
    resolved = []
    for spec in input_specs:
        if "tensor" in spec:
            item = {
                "tensor": [_resolve_dim(t, binding) for t in spec["tensor"]],
                "dtype": spec.get("dtype", dtype_str),
                "fill": spec.get("fill", "randn"),
            }
            if item["fill"] == "randint":
                item["low"] = spec.get("low", 0)
                item["high"] = spec["high"]
            resolved.append(item)
        elif "scalar" in spec:
            resolved.append({"scalar": spec["scalar"]})
        else:
            raise ValueError(f"输入项须含 tensor 或 scalar: {spec}")
    return resolved


def _instantiate_args(resolved):
    """按具体描述在 CUDA 上实例化实参列表（按 fill 造张量，标量原样）。"""
    args = []
    for item in resolved:
        if "scalar" in item:
            args.append(item["scalar"])
            continue
        shape, fill = item["tensor"], item["fill"]
        dtype = getattr(torch, item["dtype"].split(".")[-1])
        if fill == "randn":
            args.append(torch.randn(shape, dtype=dtype, device="cuda"))
        elif fill == "rand":
            args.append(torch.rand(shape, dtype=dtype, device="cuda"))
        elif fill == "zeros":
            args.append(torch.zeros(shape, dtype=dtype, device="cuda"))
        elif fill == "empty":
            args.append(torch.empty(shape, dtype=dtype, device="cuda"))
        elif fill == "randint":
            args.append(torch.randint(item["low"], item["high"], tuple(shape),
                                      dtype=dtype, device="cuda"))
        else:
            raise ValueError(f"未知 fill: {fill}")
    return args


def _key_shape(op_cfg, binding):
    """算子在 JSON 里的主键 shape：优先 key_dims，否则取第一个 tensor 输入形状。"""
    if "key_dims" in op_cfg:
        return [_resolve_dim(d, binding) for d in op_cfg["key_dims"]]
    for spec in op_cfg["inputs"]:
        if "tensor" in spec:
            return [_resolve_dim(t, binding) for t in spec["tensor"]]
    raise ValueError("算子缺少 tensor 输入且未指定 key_dims，无法确定主键 shape")


def _empty_result(report_path=None):
    """NCU 不可用/失败时的占位结果，字段与成功时一致。

    字段命名对齐《算子后端缺失性能基准方案参考.md》的符号列表：
      F_cuda / F_tensor  实际计算量 (FLOP)
      B_mem              实际访存量 (Byte)
      U_cuda / U_tensor / U_mem / U_bottle_neck  硬件利用率 (%, 0-100)
      bottle_neck_unit   瓶颈单元（cuda / tensor / mem / null）
    kernel 运行时间 T_us 由 do_bench 在采集侧填入（此处不含）。
    """
    return {
        # —— 文档口径字段 ——
        "F_cuda": 0.0, "F_tensor": 0.0, "B_mem": 0.0,
        "U_cuda": 0.0, "U_tensor": 0.0, "U_mem": 0.0,
        "U_bottle_neck": 0.0, "bottle_neck_unit": None,
        "num_kernels": 0, "per_kernel": [],
        # —— 兼容旧字段（同值冗余，供既有下游读取） ——
        "flops_cuda_core": 0.0, "flops_tensor_core": 0.0, "memory_bytes": 0.0,
        "util_cuda_core": 0.0, "util_tensor_core": 0.0, "util_memory_bw": 0.0,
        "bottleneck": None, "bottleneck_util": 0.0,
        "cuda_flops": 0.0, "sm_utilization": 0.0,
        "report_path": report_path,
    }


def _build_profile_script(module, symbol, resolved_inputs, warmup):
    """临时脚本：解析 native callable，warmup 若干次后把最后一次目标调用包在
    NVTX range "kiro_target" 里，供 NCU 用 --nvtx-include 精确锁定这次调用发出的
    kernel（不受 randn 等辅助 kernel 的 launch 序号干扰）。

    脚本自包含（子进程独立运行），把本模块的 resolver 逻辑内联进去；输入按
    resolved_inputs 的具体描述用 randn/标量重建。对原地算子，被采样的那次调用
    前重新构造实参，避免 warmup 的原地累积污染数值。
    """
    return f"""import importlib
import torch


def _resolve(module, symbol):
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        pass
    try:
        mod = importlib.import_module(module)
    except (ImportError, ModuleNotFoundError):
        head, *rest = module.split(".")
        mod = importlib.import_module(head)
        for part in rest:
            mod = getattr(mod, part)
    return getattr(mod, symbol)


def _make_args():
    specs = {resolved_inputs!r}
    args = []
    for it in specs:
        if "scalar" in it:
            args.append(it["scalar"])
            continue
        dt = getattr(torch, it["dtype"].split(".")[-1])
        shape, fill = it["tensor"], it.get("fill", "randn")
        if fill == "randn":
            args.append(torch.randn(shape, dtype=dt, device="cuda"))
        elif fill == "rand":
            args.append(torch.rand(shape, dtype=dt, device="cuda"))
        elif fill == "zeros":
            args.append(torch.zeros(shape, dtype=dt, device="cuda"))
        elif fill == "empty":
            args.append(torch.empty(shape, dtype=dt, device="cuda"))
        elif fill == "randint":
            args.append(torch.randint(it["low"], it["high"], tuple(shape),
                                      dtype=dt, device="cuda"))
    return args


op = _resolve({module!r}, {symbol!r})
args = _make_args()
for _ in range({warmup}):
    op(*args)
torch.cuda.synchronize()
# 重建实参（避免原地算子的 warmup 累积污染），随后仅把这一次目标调用圈进
# NVTX range，NCU 用 --nvtx-include "kiro_target/" 精确采样它发出的 kernel。
args = _make_args()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("kiro_target")
op(*args)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
"""


def _build_profile_script_ops(ops_dir, op_module, binding, dtype_str, warmup):
    """临时脚本（方案B ops 路径）：子进程把 ops 目录加进 sys.path，import 该
    算子模块，用 build_inputs 重建实参、native 解析 callable，warmup 后把最后
    一次目标调用圈进 NVTX range "kiro_target" 供 NCU 精确锁定。

    脚本自包含（子进程独立运行）。对原地算子，被采样的那次调用前用 build_inputs
    重新构造实参，避免 warmup 的原地累积污染数值——与 yaml 路径一致。
    """
    return f"""import sys
import torch

sys.path.insert(0, {str(ops_dir)!r})
import {op_module} as opmod

dtype = {dtype_str}
binding = {binding!r}
op = opmod.native()
if op is None:
    raise RuntimeError("native 解析失败: {op_module}")


def _make_args():
    return opmod.build_inputs(binding, dtype, "cuda")


args, kwargs = _make_args()
for _ in range({warmup}):
    op(*args, **kwargs)
torch.cuda.synchronize()
# 重建实参（避免原地算子的 warmup 累积污染），随后仅把这一次目标调用圈进
# NVTX range，NCU 用 --nvtx-include "kiro_target/" 精确采样它发出的 kernel。
args, kwargs = _make_args()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("kiro_target")
op(*args, **kwargs)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
"""


def _parse_ncu_csv(csv_text, wanted):
    """解析 `ncu --import --page raw --csv`（宽表：每 launch 一行，metric 为列）。

    返回 {metric: [每 kernel 数值, ...]}，跳过无法解析为数字的单位行/空格。
    额外返回 KERNEL_NAME_COL -> [每 kernel 名, ...]（字符串，用于 per_kernel 明细）；
    每个 metric 的列表按 launch 顺序与 kernel 名一一对应。
    """
    rows = [r for r in csv.reader(io.StringIO(csv_text)) if r]
    if not rows:
        return {}
    header = rows[0]
    col = {m: header.index(m) for m in wanted if m in header}
    name_idx = header.index(KERNEL_NAME_COL) if KERNEL_NAME_COL in header else None

    # ncu raw CSV 在表头下紧跟一行“单位行”：各数值列写单位串（byte/Kbyte/us/
    # %/inst…），Kernel Name 列为空。它不是真实 kernel——若当成 kernel，会在
    # per_kernel 里造出空名全 0 记录，还会给 mean_pct 多算一个 0 行、稀释利用率。
    # 这里显式识别并消费它：①得到每列单位→字节系列（Kbyte/Mbyte…，ncu 用十进制
    # 1000 进制）归一到 base 字节，使 B_mem 单位为 Byte（对齐参考文档）；②数据行
    # 从单位行之后开始。无单位行的 ncu 版本则乘子全 1、数据从第 1 行起。
    mult = {m: 1.0 for m in col}
    data_start = 1
    if len(rows) > 1 and name_idx is not None:
        unit_row = rows[1]
        unit_name = (unit_row[name_idx].strip()
                     if name_idx < len(unit_row) else "")
        if not unit_name:  # 确是单位行（Kernel Name 空）
            for m, idx in col.items():
                unit = unit_row[idx].strip() if idx < len(unit_row) else ""
                mult[m] = _BYTE_UNIT_MULT.get(unit, 1.0)
            data_start = 2

    out = {m: [] for m in col}
    names = []
    for row in rows[data_start:]:
        # 防御：跳过任何 Kernel Name 为空的非 kernel 行。
        if name_idx is not None:
            name = row[name_idx].strip() if name_idx < len(row) else ""
            if not name:
                continue
        else:
            name = ""
        # 数值单元无法解析为数字（空格等）时记 0，保持各 metric 列表与 kernel
        # 行数严格对齐——per_kernel 按下标取值依赖这个对齐。按单位乘子归一。
        for m, idx in col.items():
            val = 0.0
            if idx < len(row):
                cell = row[idx].replace(",", "").strip()
                try:
                    val = float(cell) * mult[m]
                except ValueError:
                    val = 0.0
            out[m].append(val)
        names.append(name)
    out[KERNEL_NAME_COL] = names
    return out


def _build_per_kernel(agg, tensor_metric):
    """由 _parse_ncu_csv 的逐 kernel 数组构建 per_kernel 明细列表。

    agg[metric] 是按 launch 顺序排列、与 agg[KERNEL_NAME_COL] 一一对应的每 kernel
    值。这里为每个 kernel 提取：名、F_cuda（加权 FP 指令）、F_tensor、B_mem、
    三维利用率（%），以及 GPU duration 占比 dur_frac（供采集侧把端到端 T 按此拆分）。
    单 kernel 算子会得到长度为 1 的列表。
    """
    names = agg.get(KERNEL_NAME_COL, [])
    n = len(names)
    if n == 0:
        return []

    def col(m):
        vals = agg.get(m, [])
        return [vals[i] if i < len(vals) else 0.0 for i in range(n)]

    dur = col(DURATION)
    dur_sum = sum(dur)
    cuda_cols = {m: col(m) for m in FLOP_CUDA_CORE}
    tensor_col = col(tensor_metric) if tensor_metric else [0.0] * n
    mem_col = col(MEMORY_BYTES)
    ucuda_col = col(UTIL["cuda_core"])
    utensor_col = col(UTIL["tensor_core"])
    umem_col = col(UTIL["memory_bw"])

    per_kernel = []
    for i in range(n):
        f_cuda = sum(w * cuda_cols[m][i] for m, w in FLOP_CUDA_CORE.items())
        per_kernel.append({
            "name": names[i],
            "F_cuda": f_cuda,
            "F_tensor": tensor_col[i],
            "B_mem": mem_col[i],
            "U_cuda": ucuda_col[i],      # 已是 % of peak（0-100）
            "U_tensor": utensor_col[i],
            "U_mem": umem_col[i],
            "dur_frac": (dur[i] / dur_sum) if dur_sum else 0.0,
        })
    return per_kernel


def profile_with_ncu(op_name, profile_script, shape, dtype_str,
                     report_dir=None):
    """NCU 分析目标算子，返回三维工作量/利用率/瓶颈。

    先 `ncu --export` 存 .ncu-rep（可用 ncu-ui 核对），再 `ncu --import --csv`
    解析。返回字段见 _empty_result；NCU 不可用或失败时各值为 0。
    profile_script 是自包含的临时脚本文本（yaml 路径由 _build_profile_script、
    ops 路径由 _build_profile_script_ops 生成），把目标调用圈进 NVTX range
    "kiro_target" 供精确采样。shape/dtype_str 仅用于命名报告与选 Tensor Core
    metric。
    """
    tensor_metric = FLOP_TENSOR_CORE.get(dtype_str)
    metrics = list(FLOP_CUDA_CORE) + [MEMORY_BYTES, *UTIL.values(), SM_UTIL,
                                      DURATION]
    if tensor_metric:
        metrics.append(tensor_metric)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(profile_script)
        script_path = f.name

    report_dir = Path(report_dir) if report_dir else Path(tempfile.gettempdir())
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{op_name}_{'x'.join(map(str, shape))}_{dtype_str.split('.')[-1]}"
    report_base = report_dir / tag

    def _err_msg(cp):
        # ncu 的 ==ERROR== 常写到 stdout；stderr 为空时回退取 stdout，避免空报错。
        return ((cp.stderr or "").strip() or (cp.stdout or "").strip())[:300]

    try:
        # 用 NVTX range 锁定目标调用，不再靠全局 launch 序号（randn 等辅助 kernel
        # 会打乱序号）。--nvtx-include 只采 "kiro_target" range 内发出的 kernel。
        export_cmd = [
            "ncu", "--metrics", ",".join(metrics),
            "--nvtx", "--nvtx-include", "kiro_target/",
            "--force-overwrite", "--export", str(report_base),
            sys.executable, script_path,
        ]
        try:
            proc = subprocess.run(export_cmd, capture_output=True, text=True,
                                  timeout=600)
        except FileNotFoundError:
            print("  ✗ 未找到 ncu 命令，跳过 NCU profiling")
            return _empty_result()
        except subprocess.TimeoutExpired:
            print(f"  ✗ NCU 超时: {shape} {dtype_str}")
            return _empty_result()
        if proc.returncode != 0:
            print(f"  ✗ NCU profiling 失败 (code={proc.returncode}): "
                  f"{_err_msg(proc)}")
            return _empty_result()

        # ncu 版本不同后缀不同（.ncu-rep 或压缩的 .ncu-repz），取实际生成的文件。
        candidates = sorted(report_dir.glob(f"{tag}.ncu-rep*"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"  ✗ NCU 未生成报告文件: {report_base}.ncu-rep[z]")
            return _empty_result()
        report_path = str(candidates[0])

        imp = subprocess.run(
            ["ncu", "--import", report_path, "--csv", "--page", "raw"],
            capture_output=True, text=True, timeout=120)
        if imp.returncode != 0:
            print(f"  ✗ NCU import 失败 (code={imp.returncode}): "
                  f"{_err_msg(imp)}")
            return _empty_result(report_path)

        agg = _parse_ncu_csv(imp.stdout, metrics)

        def total(m):
            return sum(agg.get(m, []))

        def mean_pct(m):
            vals = agg.get(m, [])
            return sum(vals) / len(vals) / 100.0 if vals else 0.0

        flops_cuda = sum(w * total(m) for m, w in FLOP_CUDA_CORE.items())
        flops_tensor = total(tensor_metric) if tensor_metric else 0.0
        mem_bytes = total(MEMORY_BYTES)
        # util：0-1 口径（旧字段沿用）；bottleneck 取三维最大者。
        util = {dim: mean_pct(name) for dim, name in UTIL.items()}
        bottleneck = max(util, key=util.get)
        if util[bottleneck] == 0.0:  # 三维全 0（解析失败/metric 不匹配）
            bottleneck = None

        # 文档口径：U_* 用百分比（0-100），瓶颈单元名映射为 cuda/tensor/mem。
        _unit = {"cuda_core": "cuda", "tensor_core": "tensor",
                 "memory_bw": "mem"}
        U_cuda = util["cuda_core"] * 100.0
        U_tensor = util["tensor_core"] * 100.0
        U_mem = util["memory_bw"] * 100.0
        U_bottle = util[bottleneck] * 100.0 if bottleneck else 0.0

        # per_kernel 明细：一次 native 调用可能启动多个 kernel（MoE 的
        # gather/gemm/scatter 等）。汇总口径下 bottle_neck_unit 会被便宜的小
        # kernel 带偏，故逐 kernel 保留 F/B/U，便于按耗时加权或逐 kernel 判瓶颈。
        # T_us 由采集侧的 do_bench 端到端测得后按 GPU duration 比例拆到各 kernel。
        per_kernel = _build_per_kernel(agg, tensor_metric)

        return {
            # —— 文档口径字段（对齐 参考.md 符号列表）——
            "F_cuda": flops_cuda,            # F_cuda  (FLOP)
            "F_tensor": flops_tensor,        # F_tensor(FLOP)
            "B_mem": mem_bytes,              # B_mem   (Byte)
            "U_cuda": U_cuda,                # U_cuda  (%)
            "U_tensor": U_tensor,            # U_tensor(%)
            "U_mem": U_mem,                  # U_mem   (%)
            "U_bottle_neck": U_bottle,       # U_bottle_neck (%)
            "bottle_neck_unit": _unit.get(bottleneck),  # cuda/tensor/mem/None
            "num_kernels": len(per_kernel),  # NVTX range 内 kernel 数
            "per_kernel": per_kernel,        # 逐 kernel 明细（多 kernel 算子）
            # —— 兼容旧字段（同值冗余）——
            "flops_cuda_core": flops_cuda,
            "flops_tensor_core": flops_tensor,
            "memory_bytes": mem_bytes,
            "util_cuda_core": util["cuda_core"],
            "util_tensor_core": util["tensor_core"],
            "util_memory_bw": util["memory_bw"],
            "bottleneck": bottleneck,
            "bottleneck_util": util[bottleneck] if bottleneck else 0.0,
            "cuda_flops": flops_cuda,
            "sm_utilization": mean_pct(SM_UTIL),
            "report_path": report_path,
        }
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def _finalize_record(latency_ms, ncu, config=None):
    """组装单条 shape×dtype 记录：填 T_us（文档口径 us），把端到端 T 按各
    kernel 的 GPU duration 占比拆进 per_kernel[i]["T_us"]，可选附 config
    （复杂算子的真实输入输出 shape 描述）。ncu 为 profile_with_ncu 的返回。"""
    T_us = latency_ms * 1000.0

    # 把端到端 T 按 dur_frac 拆到各 kernel（多 kernel 算子）；dur_frac 缺省则
    # 均分。就地补上每 kernel 的 T_us，并移除中间量 dur_frac。
    per_kernel = ncu.get("per_kernel", [])
    n = len(per_kernel)
    for k in per_kernel:
        frac = k.pop("dur_frac", (1.0 / n) if n else 0.0)
        k["T_us"] = T_us * frac

    record = {}
    if config is not None:
        record["config"] = config
    record["T_us"] = T_us
    record["latency_ms"] = latency_ms
    record.update(ncu)
    return record


def _collect_one_op(op_name, op_cfg, ncu_enabled, report_dir):
    """按 yaml 配置采集单个算子；native 解析不到则返回 None（跳过）。"""
    module = op_cfg["native"]["module"]
    symbol = op_cfg["native"]["symbol"]
    op = resolve_native_op(module, symbol)
    if op is None:
        print(f"\n跳过算子 {op_name}: 无法解析 native {module}.{symbol}")
        return None

    dtypes = op_cfg["dtypes"]
    bindings = _expand_grid(op_cfg["grid"])

    # 预检：解析到的 callable 可能只注册了 schema、没有 CUDA kernel（如本环境把
    # vLLM 原生 kernel 换成了 Triton），真调才暴露 NotImplementedError。用最小的
    # 那个 shape 试调一次，没实现就整体跳过，避免逐点崩掉。
    probe = _instantiate_args(_resolve_inputs(op_cfg["inputs"], bindings[0],
                                              dtypes[0]))
    try:
        op(*probe)
        torch.cuda.synchronize()
    except NotImplementedError as e:
        print(f"\n跳过算子 {op_name}: native {module}.{symbol} 无 CUDA 实现 "
              f"（{str(e)[:80]}）")
        return None

    shapes = {}
    print(f"\n采集算子: {op_name}  (native {module}.{symbol})")
    for binding in bindings:
        shape = _key_shape(op_cfg, binding)
        per_dtype = shapes.setdefault(str(shape), {})
        for dtype_str in dtypes:
            resolved = _resolve_inputs(op_cfg["inputs"], binding, dtype_str)
            args = _instantiate_args(resolved)
            latency_ms = triton.testing.do_bench(lambda: op(*args),
                                                 warmup=25, rep=100)
            if ncu_enabled:
                script = _build_profile_script(module, symbol, resolved,
                                               warmup=3)
                ncu = profile_with_ncu(op_name, script, shape, dtype_str,
                                       report_dir=report_dir)
            else:
                ncu = _empty_result()
            per_dtype[dtype_str] = _finalize_record(latency_ms, ncu)

            bn = ncu["bottleneck"]
            suffix = (f"  瓶颈={bn} ({ncu['bottleneck_util']*100:.1f}%)"
                      if bn else "")
            print(f"  {shape} {dtype_str}: {latency_ms:.4f} ms{suffix}")

    return {"native_api": f"{module}.{symbol}", "shapes": shapes}


def _discover_ops(ops_dir=OPS_DIR):
    """扫描 ops/ 目录下的算子模块，import 并按契约校验后返回 [(name, module)]。

    校验缺字段的模块直接跳过并告警（避免半成品模块把整轮采集带崩）。
    """
    if not ops_dir.is_dir():
        return []
    # 让 `import <op>` 能找到 ops/ 下的模块（子进程脚本也用同一目录）。
    if str(ops_dir) not in sys.path:
        sys.path.insert(0, str(ops_dir))
    required = ("OP_NAME", "DTYPES", "IS_INPLACE",
                "native", "grid", "build_inputs", "key_shape")
    discovered = []
    for path in sorted(ops_dir.glob("*.py")):
        if path.stem == "__init__":
            continue
        try:
            mod = importlib.import_module(path.stem)
        except Exception as e:  # noqa: BLE001 - 单个模块坏了不该拖垮整轮
            print(f"\n跳过 ops 模块 {path.stem}: import 失败（{str(e)[:80]}）")
            continue
        missing = [a for a in required if not hasattr(mod, a)]
        if missing:
            print(f"\n跳过 ops 模块 {path.stem}: 缺少契约字段 {missing}")
            continue
        discovered.append((mod.OP_NAME, mod))
    return discovered


def _collect_one_op_ops(op_module, ncu_enabled, report_dir):
    """采集单个方案B（自定义 ops）算子；native 解析不到则返回 None（跳过）。"""
    op_name = op_module.OP_NAME
    mod_name = op_module.__name__
    op = op_module.native()
    if op is None:
        print(f"\n跳过算子 {op_name}: ops 模块 native() 解析不到 callable")
        return None

    dtypes = op_module.DTYPES
    bindings = op_module.grid()

    # 预检：callable 可能只注册了 schema、没有 CUDA kernel，真调才暴露
    # NotImplementedError。用第一个 shape/dtype 试调一次，没实现就整体跳过。
    probe_args, probe_kwargs = op_module.build_inputs(
        bindings[0], dtypes[0], "cuda")
    try:
        op(*probe_args, **probe_kwargs)
        torch.cuda.synchronize()
    except NotImplementedError as e:
        print(f"\n跳过算子 {op_name}: native 无 CUDA 实现（{str(e)[:80]}）")
        return None

    shapes = {}
    print(f"\n采集算子: {op_name}  (ops 模块 {mod_name})")
    for binding in bindings:
        shape = op_module.key_shape(binding)
        per_dtype = shapes.setdefault(str(shape), {})
        for dtype in dtypes:
            dtype_str = str(dtype)
            args, kwargs = op_module.build_inputs(binding, dtype, "cuda")
            latency_ms = triton.testing.do_bench(
                lambda: op(*args, **kwargs), warmup=25, rep=100)
            if ncu_enabled:
                script = _build_profile_script_ops(
                    OPS_DIR, mod_name, binding, dtype_str, warmup=3)
                ncu = profile_with_ncu(op_name, script, shape, dtype_str,
                                       report_dir=report_dir)
            else:
                ncu = _empty_result()
            # config：复杂算子可选导出 config(binding, dtype) 描述真实输入输出
            # shape；未导出则不写该字段（简单算子 shape 键已够表达）。
            cfg = None
            if hasattr(op_module, "config"):
                cfg = op_module.config(binding, dtype)
            per_dtype[dtype_str] = _finalize_record(latency_ms, ncu, config=cfg)

            bn = ncu["bottleneck"]
            suffix = (f"  瓶颈={bn} ({ncu['bottleneck_util']*100:.1f}%)"
                      if bn else "")
            print(f"  {shape} {dtype_str}: {latency_ms:.4f} ms{suffix}")

    return {"native_api": f"ops.{mod_name}", "shapes": shapes}


def collect_baseline(output_path, config_path=CONFIG_PATH, ncu_enabled=True,
                     report_dir=None):
    """采集 baseline 数据并写入 JSON。

    先跑方案B（ops/ 下的自定义算子模块），再跑 yaml 声明式路径（跳过已被 ops
    覆盖的同名算子，ops 优先）。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备")
    device_name = torch.cuda.get_device_name()
    if "NVIDIA" not in device_name.upper():
        print(f"警告: 当前设备 {device_name} 可能不是 NVIDIA 硬件")
    print(f"采集设备: {device_name}")

    results = {}

    # 方案B：ops/ 下的自定义算子模块优先。
    ops_modules = _discover_ops()
    for op_name, op_module in ops_modules:
        entry = _collect_one_op_ops(op_module, ncu_enabled, report_dir)
        if entry is not None:
            results[op_name] = entry

    # yaml 声明式路径：跳过已被 ops 覆盖的同名算子（ops 优先）。
    covered = set(results)
    config = yaml.safe_load(Path(config_path).read_text()) or {}
    for op_name, op_cfg in config.items():
        if op_name in covered:
            print(f"\n跳过 yaml 算子 {op_name}: 已由 ops/ 模块采集")
            continue
        entry = _collect_one_op(op_name, op_cfg, ncu_enabled, report_dir)
        if entry is not None:
            results[op_name] = entry

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Baseline 数据已写入: {output_path}  (共 {len(results)} 个算子)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="采集 NVIDIA 原生 kernel 的 baseline 数据")
    parser.add_argument("--output", default="op_perf_baseline.json",
                        help="输出文件路径")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="算子采集配置 yaml（默认同目录 baseline_shape.yaml）")
    parser.add_argument("--no-ncu", action="store_true",
                        help="跳过 NCU profiling（只测 latency）")
    parser.add_argument("--report-dir", default="ncu_reports",
                        help="NCU .ncu-rep 报告存放目录（可用 ncu-ui 打开）")
    args = parser.parse_args()
    collect_baseline(args.output, config_path=args.config,
                     ncu_enabled=not args.no_ncu, report_dir=args.report_dir)
