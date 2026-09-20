#!/usr/bin/env python3
"""离线冒烟测试 ops/ 模块（mac 无 torch/CUDA 时用）。

用一个最小 torch stub 顶替真 torch，逐个 import ops/ 下模块并校验契约字段、
跑通 grid()/build_inputs()/key_shape()/config()（纯逻辑，不做真实 CUDA 计算）。
仅验证“模块能被采集器正确驱动”，不验证数值正确性——后者需在真卡上跑采集器。

用法： python3 tools/_smoke_ops.py [op_name ...]
不带参数则测全部；带名字则只测指定模块（文件名去 .py）。
"""

import sys
import types
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parent.parent / "ops"


# ---- 最小 torch stub：够 build_inputs 构造“张量占位”并读 dtype/属性即可 ----
class _Dtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


class _FakeTensor:
    """记录 shape/dtype 的假张量；支持链式 .to()/.contiguous() 等常见调用。"""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)

    def to(self, *a, **k):
        return self

    def contiguous(self):
        return self

    def view(self, *shape):
        return _FakeTensor(shape, self.dtype)

    def reshape(self, *shape):
        return _FakeTensor(shape, self.dtype)

    def transpose(self, *a):
        return self

    def unsqueeze(self, *a):
        return self

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __getitem__(self, k):
        return self

    def __setitem__(self, k, v):
        return None

    # 未显式实现的方法（.long()/.int()/.float()/.clone()/.flatten()/.cpu()...）
    # 一律返回保形自身的可调用占位；仅用于冒烟期跟踪 shape/dtype。
    def __getattr__(self, name):
        return lambda *a, **k: self

    # 逐元素算术：形状不变，返回自身占位即可（冒烟只关心 shape/dtype 流转）
    def _binop(self, other):
        return self

    __add__ = __radd__ = _binop
    __sub__ = __rsub__ = _binop
    __mul__ = __rmul__ = _binop
    __truediv__ = __rtruediv__ = _binop
    __floordiv__ = __rfloordiv__ = _binop
    __mod__ = __rmod__ = _binop
    __pow__ = __rpow__ = _binop
    __neg__ = lambda self: self

    def __repr__(self):
        return f"FakeTensor(shape={self.shape}, dtype={self.dtype})"


def _make_torch_stub():
    t = types.ModuleType("torch")
    for name in ("float16", "bfloat16", "float32", "float64",
                 "int8", "uint8", "int16", "int32", "int64", "bool",
                 "float8_e4m3fn", "float8_e5m2", "complex64"):
        setattr(t, name, _Dtype(name))

    def _tensor_factory(*a, **k):
        # 支持 torch.randn(2,3,...) / torch.randn([2,3]) / torch.randn((2,3))
        dtype = k.get("dtype", t.float32)
        if len(a) == 1 and isinstance(a[0], (list, tuple)):
            shape = tuple(a[0])
        else:
            shape = tuple(x for x in a if isinstance(x, int))
        return _FakeTensor(shape, dtype)

    for fn in ("randn", "rand", "zeros", "ones", "empty", "full",
               "randint", "arange", "tensor", "empty_like", "zeros_like",
               "ones_like", "randn_like", "randperm"):
        setattr(t, fn, _tensor_factory)

    # 逐元素/规约类：保形返回第一个张量参数（冒烟只跟踪 shape/dtype 流转）
    def _passthrough(*a, **k):
        for x in a:
            if isinstance(x, _FakeTensor):
                return x
        return _FakeTensor((), t.float32)

    for fn in ("cumsum", "cumprod", "cat", "concat", "stack",
               "sigmoid", "silu", "relu", "softmax", "clamp",
               "sum", "mean", "max", "min", "abs", "sqrt", "exp",
               "log", "log1p", "log2", "logsigmoid", "softplus",
               "neg", "reciprocal", "rsqrt", "pow", "square", "tanh"):
        setattr(t, fn, _passthrough)

    # topk/sort 返回 (values, indices) 元组：均保形返回第一个张量参数
    def _pair(*a, **k):
        base = next((x for x in a if isinstance(x, _FakeTensor)),
                    _FakeTensor((), t.float32))
        return base, base

    t.topk = _pair
    t.sort = _pair
    t.max = _passthrough
    t.min = _passthrough

    # finfo/iinfo：返回带 min/max/eps/tiny 的占位（冒烟只需属性可取）
    _info = types.SimpleNamespace(min=-1e30, max=1e30, eps=1e-7, tiny=1e-30,
                                  bits=8, dtype="stub")
    t.finfo = lambda *a, **k: _info
    t.iinfo = lambda *a, **k: types.SimpleNamespace(min=-(2**31), max=2**31 - 1, bits=32)

    t.manual_seed = lambda *a, **k: None
    t.Tensor = _FakeTensor
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda *a, **k: (9, 0),
        get_device_name=lambda *a, **k: "STUB",
        manual_seed=lambda *a, **k: None,
    )
    t.cuda = cuda
    return t


def main(argv):
    sys.modules["torch"] = _make_torch_stub()
    sys.path.insert(0, str(OPS_DIR))

    required = ("OP_NAME", "DTYPES", "IS_INPLACE",
                "native", "grid", "build_inputs", "key_shape")
    targets = argv or sorted(p.stem for p in OPS_DIR.glob("*.py")
                             if p.stem != "__init__")
    import importlib

    ok, bad = [], []
    for stem in targets:
        try:
            mod = importlib.import_module(stem)
        except Exception as e:  # noqa: BLE001
            bad.append((stem, f"import 失败: {e}"))
            continue
        missing = [a for a in required if not hasattr(mod, a)]
        if missing:
            bad.append((stem, f"缺契约字段 {missing}"))
            continue
        try:
            grid = mod.grid()
            assert isinstance(grid, list) and grid, "grid() 应返回非空 list[dict]"
            dtype = mod.DTYPES[0]
            b0 = grid[0]
            args, kwargs = mod.build_inputs(b0, dtype, "cuda")
            assert isinstance(args, tuple), "build_inputs 首元应为 tuple(args)"
            assert isinstance(kwargs, dict), "build_inputs 次元应为 dict(kwargs)"
            ks = mod.key_shape(b0)
            assert isinstance(ks, (list, str)), "key_shape 应返回 list 或 str"
            if hasattr(mod, "config"):
                cfg = mod.config(b0, dtype)
                assert isinstance(cfg, dict), "config 应返回 dict"
            ok.append((stem, f"{len(grid)} 采集点, key={ks!r}"))
        except Exception as e:  # noqa: BLE001
            bad.append((stem, f"逻辑冒烟失败: {type(e).__name__}: {e}"))

    print(f"\n== 冒烟结果：{len(ok)} 通过 / {len(bad)} 失败 ==")
    for stem, info in ok:
        print(f"  ✓ {stem}: {info}")
    for stem, info in bad:
        print(f"  ✗ {stem}: {info}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
