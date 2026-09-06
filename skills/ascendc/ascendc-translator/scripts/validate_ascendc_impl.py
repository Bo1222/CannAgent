#!/usr/bin/env python3
"""AscendC 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测四种退化类型：
  Type 1: 无 AscendC kernel 扩展导入（纯 PyTorch）
  Type 2: 有扩展导入但 forward() 未调用 kernel 函数
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口
  Type 4: forward() 中存在逐元素 Python for 循环（标量写法退化）

用法:
    python validate_ascendc_impl.py <file_path> [--pybind-file <path>] [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor", "arange", "linspace",
    # 类型 / 设备
    "as_tensor",
}
ALLOWED_TORCH_CALLS = {f"torch.{name}" for name in ALLOWED_TORCH_FUNCS}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    # 布局操作（不执行计算）
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda",
    "item", "tolist",
    # buffer 分配
    "new_empty", "new_empty_strided", "new_zeros", "new_ones", "new_full",
    # 原地标记
    "requires_grad_", "zero_",
    # 切片相关
    "index_select",
    # 设备检查
    "is_npu", "is_cuda",
}

# 已知的占位符导入名称（表示扩展模块未正确配置）
PLACEHOLDER_IMPORT_NAMES = {
    "TORCH_EXTENSION_NAME",
}

PYBIND_MODULE_PATTERN = re.compile(
    r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
)

SCALAR_ANNOTATIONS = {"str", "int", "float", "bool", "bytes", "None"}
TENSOR_METADATA_ATTRIBUTES = {
    "shape", "stride", "ndim", "dtype", "device", "layout", "requires_grad",
}
TENSOR_METADATA_METHODS = {
    "size", "stride", "numel", "dim", "element_size", "storage_offset",
    "is_contiguous", "data_ptr", "item", "tolist", "is_npu", "is_cuda",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _qualified_name(node):
    """Return a dotted name for Name/Attribute nodes when statically resolvable."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          _ext.run_kernel -> ('_ext', 'run_kernel')
          my_func -> (None, 'my_func')
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        # 处理 torch.nn.functional.relu 形式
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def extract_pybind_module_names(pybind_file):
    """Read literal extension module names from a task's pybind source."""
    path = Path(pybind_file)
    if not path.is_file():
        raise ValueError(f"pybind source does not exist: {path}")
    names = set(PYBIND_MODULE_PATTERN.findall(path.read_text(encoding="utf-8")))
    names.difference_update(PLACEHOLDER_IMPORT_NAMES)
    if not names:
        raise ValueError(f"no literal PYBIND11_MODULE name found in {path}")
    return names


def _default_pybind_file(filepath):
    if not filepath or filepath == "<unknown>":
        return None
    wrapper = Path(filepath)
    return wrapper.parent / "kernel" / "pybind11.cpp"


def collect_import_sources(tree):
    """Map names bound by imports to their original module or symbol path."""
    sources = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    sources[alias.asname] = alias.name
                else:
                    root = alias.name.split(".", 1)[0]
                    sources[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                used_name = alias.asname or alias.name
                sources[used_name] = f"{node.module}.{alias.name}"
    return sources


def resolve_import_source(node, import_sources):
    """Resolve an expression's dotted name through import aliases."""
    qualified = _qualified_name(node)
    if not qualified:
        return None
    root, *rest = qualified.split(".")
    imported = import_sources.get(root)
    if not imported:
        return qualified
    return ".".join([imported, *rest])


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def find_ascendc_extension_imports(tree, expected_module_names):
    """查找所有 AscendC 扩展模块的导入信息。

    扩展身份来自 pybind11.cpp 的 PYBIND11_MODULE 声明，而不是名称正则。
    支持 import module [as alias] 和 from package import module [as alias]。

    返回 dict: {alias_or_name: {"name": str, "alias": str|None,
                                  "line": int, "is_placeholder": bool,
                                  "import_style": str}}
    """
    extensions = {}
    expected_module_names = set(expected_module_names or ())

    for node in ast.walk(tree):
        # --- import xxx_ext [as alias] ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                actual_name = alias.name
                used_name = alias.asname if alias.asname else alias.name
                is_placeholder = actual_name in PLACEHOLDER_IMPORT_NAMES
                if is_placeholder or actual_name in expected_module_names:
                    extensions[used_name] = {
                        "name": actual_name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "is_placeholder": is_placeholder,
                        "import_style": "import",
                    }

        # --- from xxx import yyy [as alias] ---
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                actual_name = alias.name
                used_name = alias.asname if alias.asname else alias.name
                is_placeholder = actual_name in PLACEHOLDER_IMPORT_NAMES
                full_name = f"{node.module}.{actual_name}" if node.module else actual_name
                if is_placeholder or actual_name in expected_module_names or full_name in expected_module_names:
                    extensions[used_name] = {
                        "name": full_name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "is_placeholder": is_placeholder,
                        "import_style": "from_import",
                    }

    return extensions


def find_model_forward(tree):
    """找到 ModelNew 或 Model 类的 forward 方法节点。

    优先查找 ModelNew，若不存在则查找 Model。
    """
    model_new_forward = None
    model_forward = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "ModelNew":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "forward":
                            model_new_forward = item
            elif node.name == "Model":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "forward":
                            model_forward = item

    return model_new_forward or model_forward, "ModelNew" if model_new_forward else "Model"


def find_reachable_functions(tree, forward_node, class_name):
    """Return forward plus local/module helper functions reachable from it."""
    module_functions = {
        node.name: node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_methods = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_methods = {
                child.name: child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            break

    reachable = []
    queue = [forward_node]
    visited = set()
    while queue:
        function = queue.pop(0)
        identity = id(function)
        if identity in visited:
            continue
        visited.add(identity)
        reachable.append(function)
        for child in ast.walk(function):
            if not isinstance(child, ast.Call):
                continue
            resolved = _resolve_call_name(child)
            if not resolved:
                continue
            qualifier, attr = resolved
            if qualifier is None and attr in module_functions:
                queue.append(module_functions[attr])
            elif qualifier == "self" and attr in class_methods:
                queue.append(class_methods[attr])
    return reachable


def find_torch_module_attributes(tree, class_name, import_sources):
    """Find self attributes initialized from torch.nn module constructors."""
    attributes = set()
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.name != "__init__":
                continue
            for child in ast.walk(method):
                if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                value = child.value
                if not isinstance(value, ast.Call):
                    continue
                source = resolve_import_source(value.func, import_sources)
                if not source or not source.startswith("torch.nn."):
                    continue
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        attributes.add(target.attr)
    return attributes


def check_kernel_calls(function_nodes, ext_names):
    """Find direct AscendC extension calls in forward and reachable helpers."""
    called = []
    for function in function_nodes:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            resolved = _resolve_call_name(node)
            if resolved is None:
                continue
            qualifier, attr = resolved
            if qualifier in ext_names:
                called.append(
                    {
                        "call": f"{qualifier}.{attr}",
                        "line": node.lineno,
                        "function": function.name,
                    }
                )
    return called


def _is_scalar_annotation(annotation, import_sources):
    source = resolve_import_source(annotation, import_sources)
    if not source:
        return False
    return source in SCALAR_ANNOTATIONS or source.split(".")[-1] in SCALAR_ANNOTATIONS


def _is_scalar_literal(node):
    return isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (str, int, float, bool, bytes))
    )


class FunctionProvenance:
    """Lightweight value provenance for one reachable Python function."""

    def __init__(
        self,
        function,
        import_sources,
        ext_names,
        *,
        initial_kinds=None,
        assume_unannotated_tensors=True,
    ):
        self.function = function
        self.import_sources = import_sources
        self.ext_names = set(ext_names)
        self.kinds = dict(initial_kinds or {})
        arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        if function.args.vararg:
            arguments.append(function.args.vararg)
        if function.args.kwarg:
            arguments.append(function.args.kwarg)
        positional = [*function.args.posonlyargs, *function.args.args]
        scalar_defaults = {
            argument.arg
            for argument, default in zip(
                positional[-len(function.args.defaults):] if function.args.defaults else [],
                function.args.defaults,
            )
            if _is_scalar_literal(default)
        }
        scalar_defaults.update(
            argument.arg
            for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
            if _is_scalar_literal(default)
        )
        for argument in arguments:
            if argument.arg == "self":
                continue
            if _is_scalar_annotation(argument.annotation, import_sources) or argument.arg in scalar_defaults:
                self.kinds[argument.arg] = "scalar"
            elif assume_unannotated_tensors and argument.arg not in self.kinds:
                self.kinds[argument.arg] = "tensor"
        self._propagate_assignments()

    def expression_kind(self, node):
        if node is None:
            return "unknown"
        if isinstance(node, ast.Name):
            if node.id in self.kinds:
                return self.kinds[node.id]
            source = self.import_sources.get(node.id)
            if source:
                if source == "torch" or source.startswith("torch."):
                    return "torch"
                return "external"
            return "unknown"
        if isinstance(node, ast.Constant):
            return "scalar"
        if isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return "metadata"
        if isinstance(node, ast.Attribute):
            base = self.expression_kind(node.value)
            if base == "tensor" and node.attr in TENSOR_METADATA_ATTRIBUTES:
                return "metadata"
            return base
        if isinstance(node, ast.Subscript):
            base = self.expression_kind(node.value)
            return "tensor" if base == "tensor" else base
        if isinstance(node, ast.Call):
            qualified = _qualified_name(node.func)
            root = qualified.split(".", 1)[0] if qualified else None
            if root in self.ext_names:
                return "tensor"
            source = resolve_import_source(node.func, self.import_sources)
            if source and (source == "torch" or source.startswith("torch.")):
                return "tensor"
            if isinstance(node.func, ast.Attribute):
                receiver = self.expression_kind(node.func.value)
                if receiver == "tensor":
                    if node.func.attr in TENSOR_METADATA_METHODS:
                        return "metadata"
                    return "tensor"
            return "unknown"
        if isinstance(node, ast.BinOp):
            kinds = {self.expression_kind(node.left), self.expression_kind(node.right)}
            return "tensor" if "tensor" in kinds else "scalar"
        if isinstance(node, ast.UnaryOp):
            return self.expression_kind(node.operand)
        if isinstance(node, ast.IfExp):
            kinds = {self.expression_kind(node.body), self.expression_kind(node.orelse)}
            return "tensor" if "tensor" in kinds else "unknown"
        return "unknown"

    def _set_target_kind(self, target, kind):
        changed = False
        if isinstance(target, ast.Name):
            if self.kinds.get(target.id) != kind:
                self.kinds[target.id] = kind
                changed = True
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                changed = self._set_target_kind(element, kind) or changed
        return changed

    def _propagate_assignments(self):
        assignments = []
        for node in ast.walk(self.function):
            if isinstance(node, ast.Assign):
                assignments.append((node.targets, node.value))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                assignments.append(([node.target], node.value))
            elif isinstance(node, ast.For):
                assignments.append(([node.target], node.iter))
        for _ in range(len(assignments) + 1):
            changed = False
            for targets, value in assignments:
                kind = self.expression_kind(value)
                if kind == "unknown":
                    continue
                for target in targets:
                    changed = self._set_target_kind(target, kind) or changed
            if not changed:
                break


def build_function_provenance(function_nodes, forward_node, import_sources, ext_names):
    """Propagate value kinds from forward call arguments into reachable helpers."""
    by_name = {function.name: function for function in function_nodes}
    initial_by_id = {id(forward_node): {}}
    provenance_by_id = {}

    def merge_kind(current, incoming):
        if not current:
            return incoming
        if current == incoming:
            return current
        if "tensor" in (current, incoming):
            return "tensor"
        if "metadata" in (current, incoming):
            return "metadata"
        return current

    for _ in range(len(function_nodes) + 1):
        changed = False
        for function in function_nodes:
            is_forward = function is forward_node
            provenance = FunctionProvenance(
                function,
                import_sources,
                ext_names,
                initial_kinds=initial_by_id.get(id(function)),
                assume_unannotated_tensors=is_forward,
            )
            provenance_by_id[id(function)] = provenance
            for call in ast.walk(function):
                if not isinstance(call, ast.Call):
                    continue
                resolved = _resolve_call_name(call)
                if not resolved:
                    continue
                qualifier, attr = resolved
                if qualifier not in (None, "self") or attr not in by_name:
                    continue
                target = by_name[attr]
                target_args = [*target.args.posonlyargs, *target.args.args]
                target_args = [argument for argument in target_args if argument.arg != "self"]
                incoming = dict(initial_by_id.get(id(target), {}))
                for argument, value in zip(target_args, call.args):
                    kind = provenance.expression_kind(value)
                    if kind != "unknown":
                        incoming[argument.arg] = merge_kind(incoming.get(argument.arg), kind)
                keyword_args = {argument.arg: argument for argument in target_args}
                for keyword in call.keywords:
                    if keyword.arg not in keyword_args:
                        continue
                    kind = provenance.expression_kind(keyword.value)
                    if kind != "unknown":
                        incoming[keyword.arg] = merge_kind(incoming.get(keyword.arg), kind)
                if incoming != initial_by_id.get(id(target), {}):
                    initial_by_id[id(target)] = incoming
                    changed = True
        if not changed:
            break
    return provenance_by_id


def check_forbidden_torch_ops(
    function_nodes,
    import_sources,
    ext_names=None,
    torch_module_attributes=None,
    provenance_by_id=None,
):
    """Reject only calls/operators proven to originate from torch or Tensor values."""
    violations = []
    ext_names = set(ext_names or ())
    torch_module_attributes = set(torch_module_attributes or ())
    provenance_by_id = provenance_by_id or {}

    for function in function_nodes:
        provenance = provenance_by_id.get(id(function)) or FunctionProvenance(
            function, import_sources, ext_names
        )
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                qualified = _qualified_name(node.func)
                root = qualified.split(".", 1)[0] if qualified else None
                if root in ext_names:
                    continue

                source = resolve_import_source(node.func, import_sources)
                if source and (source == "torch" or source.startswith("torch.")):
                    leaf = source.rsplit(".", 1)[-1]
                    if source in ALLOWED_TORCH_CALLS:
                        continue
                    violations.append({
                        "line": node.lineno,
                        "call": qualified or leaf,
                        "source": source,
                        "reason": f"{source} 是 PyTorch 计算操作，必须在 AscendC kernel 中实现",
                    })
                    continue

                if isinstance(node.func, ast.Attribute):
                    receiver_kind = provenance.expression_kind(node.func.value)
                    attr = node.func.attr
                    if receiver_kind == "tensor":
                        if attr in ALLOWED_TENSOR_METHODS:
                            continue
                        receiver = _qualified_name(node.func.value) or "<tensor>"
                        violations.append({
                            "line": node.lineno,
                            "call": f"{receiver}.{attr}()",
                            "source": "Tensor method",
                            "reason": f"Tensor.{attr} 是计算操作，必须在 AscendC kernel 中实现",
                        })
                        continue
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                        and attr in torch_module_attributes
                    ):
                        violations.append({
                            "line": node.lineno,
                            "call": f"self.{attr}(...)",
                            "source": "torch.nn.Module",
                            "reason": f"self.{attr} 是 torch.nn.Module，核心计算必须在 AscendC kernel 中实现",
                        })
                continue

            if isinstance(node, ast.BinOp):
                operand_kinds = {
                    provenance.expression_kind(node.left),
                    provenance.expression_kind(node.right),
                }
                if "tensor" not in operand_kinds:
                    continue
                operator = {
                    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                    ast.FloorDiv: "//", ast.Pow: "**", ast.Mod: "%", ast.MatMult: "@",
                }.get(type(node.op), type(node.op).__name__)
                violations.append({
                    "line": node.lineno,
                    "call": operator,
                    "source": "Tensor operator",
                    "reason": f"Tensor {operator} 运算必须在 AscendC kernel 中实现",
                })
            elif isinstance(node, ast.UnaryOp) and provenance.expression_kind(node.operand) == "tensor":
                violations.append({
                    "line": node.lineno,
                    "call": type(node.op).__name__,
                    "source": "Tensor operator",
                    "reason": "Tensor 一元运算必须在 AscendC kernel 中实现",
                })
            elif isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                if not any(provenance.expression_kind(item) == "tensor" for item in operands):
                    continue
                violations.append({
                    "line": node.lineno,
                    "call": "comparison",
                    "source": "Tensor operator",
                    "reason": "Tensor 比较运算必须在 AscendC kernel 中实现",
                })

    return violations


def check_for_loops_over_tensors(
    function_nodes, import_sources, ext_names, provenance_by_id=None
):
    """检查 forward 中是否存在用于计算的逐元素 Python for 循环（标量写法退化信号）。

    典型退化模式：
      for n in range(N):
          for c in range(C):
              x_nc = tensor[n, c]
              result = x_nc * weight + bias  # 逐元素计算
              output[n, c] = result.sum()    # 计算归约

    以下不视为退化：
      - 数据准备循环（仅做简单赋值 / 索引映射，无计算操作）

    返回违规列表 [{"line": N, "loop_var": str, "reason": str}, ...]
    """
    violations = []
    provenance_by_id = provenance_by_id or {}
    for function in function_nodes:
        provenance = provenance_by_id.get(id(function)) or FunctionProvenance(
            function, import_sources, ext_names
        )
        for node in ast.walk(function):
            if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Call):
                continue
            resolved = _resolve_call_name(node.iter)
            if not resolved or resolved != (None, "range"):
                continue
            loop_var = node.target.id if isinstance(node.target, ast.Name) else ""
            has_tensor_indexing = _loop_has_tensor_indexing(node, loop_var, provenance)
            has_computation = _loop_has_computation(
                node, provenance, import_sources, set(ext_names)
            )
            if has_tensor_indexing and has_computation:
                violations.append({
                    "line": node.lineno,
                    "loop_var": loop_var,
                    "function": function.name,
                    "reason": (
                        f"for {loop_var} in range(...) 循环中存在 tensor 索引 + 计算操作，"
                        "这是逐元素标量写法，必须使用 AscendC kernel 的向量化操作替代"
                    ),
                })

    return violations


def _loop_has_tensor_indexing(for_node, loop_var, provenance):
    """检查 for 循环体中是否存在使用循环变量的 tensor 索引。"""
    if not loop_var:
        return False
    for child in ast.walk(for_node):
        if isinstance(child, ast.Subscript):
            if provenance.expression_kind(child.value) != "tensor":
                continue
            for sub_node in ast.walk(child.slice):
                if isinstance(sub_node, ast.Name) and sub_node.id == loop_var:
                    return True
    return False


def _loop_has_computation(for_node, provenance, import_sources, ext_names):
    """检查 for 循环体中是否包含实际的计算操作。

    计算操作包括：
    - 禁止的 tensor 方法（.sum(), .mul(), ...）
    - torch.xxx 计算调用
    - F.xxx 计算调用
    - BinOp 算术运算符（+, -, *, /, ** 等，作用于 tensor 时）
    - @ 矩阵乘法运算符
    """
    for child in ast.walk(for_node):
        if isinstance(child, ast.BinOp):
            if "tensor" in {
                provenance.expression_kind(child.left),
                provenance.expression_kind(child.right),
            }:
                return True
        if isinstance(child, ast.UnaryOp) and provenance.expression_kind(child.operand) == "tensor":
            return True
        if isinstance(child, ast.Compare):
            operands = [child.left, *child.comparators]
            if any(provenance.expression_kind(item) == "tensor" for item in operands):
                return True
        if isinstance(child, ast.Call):
            qualified = _qualified_name(child.func)
            root = qualified.split(".", 1)[0] if qualified else None
            if root in ext_names:
                return True
            source = resolve_import_source(child.func, import_sources)
            if source and (source == "torch" or source.startswith("torch.")):
                if source not in ALLOWED_TORCH_CALLS:
                    return True
            if isinstance(child.func, ast.Attribute):
                if (
                    provenance.expression_kind(child.func.value) == "tensor"
                    and child.func.attr not in ALLOWED_TENSOR_METHODS
                ):
                    return True

    return False


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def validate(code, filepath="<unknown>", expected_extension_modules=None, pybind_file=None):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = {
        "valid": False,
        "filepath": filepath,
        "checks": {
            "ascendc_ext_imported": {
                "passed": False, "extensions": [], "expected_modules": [], "error": None,
            },
            "kernel_called_from_forward": {
                "passed": False, "called": [], "error": None,
            },
            "no_forbidden_torch_ops": {
                "passed": False, "violations": [], "error": None,
            },
            "no_scalar_for_loops": {
                "passed": False, "violations": [], "error": None,
            },
        },
        "regression_type": None,
        "suggestion": "",
    }

    # --- 解析 ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["ascendc_ext_imported"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    # --- Check 1: AscendC 扩展导入存在性 ---
    discovery_error = None
    if expected_extension_modules is None:
        selected_pybind = Path(pybind_file) if pybind_file else _default_pybind_file(filepath)
        try:
            expected_extension_modules = extract_pybind_module_names(selected_pybind) if selected_pybind else set()
        except ValueError as error:
            expected_extension_modules = set()
            discovery_error = str(error)
    expected_extension_modules = set(expected_extension_modules or ())
    result["checks"]["ascendc_ext_imported"]["expected_modules"] = sorted(expected_extension_modules)
    extensions = find_ascendc_extension_imports(tree, expected_extension_modules)
    ext_names = set(extensions.keys())

    result["checks"]["ascendc_ext_imported"]["extensions"] = [
        {
            "used_name": k,
            "actual_name": v["name"],
            "line": v["line"],
            "is_placeholder": v["is_placeholder"],
            "import_style": v["import_style"],
        }
        for k, v in extensions.items()
    ]

    if not ext_names:
        if discovery_error:
            error_text = f"无法从 pybind 源确认 AscendC 扩展: {discovery_error}"
        elif not expected_extension_modules:
            error_text = "没有提供可验证的 PYBIND11_MODULE 模块名"
        else:
            error_text = (
                "Python 未导入 pybind 声明的 AscendC 扩展模块: "
                f"expected={sorted(expected_extension_modules)}"
            )
        result["checks"]["ascendc_ext_imported"]["error"] = error_text
        result["regression_type"] = 1
        result["suggestion"] = (
            "确保 kernel/pybind11.cpp 使用字面量 PYBIND11_MODULE(module_name, m)，"
            "并在 model_new_ascendc.py 中导入完全相同的 module_name 后调用其函数。"
        )
        return result

    # 检查是否全部为占位符导入
    placeholder_exts = [k for k, v in extensions.items() if v["is_placeholder"]]
    if len(placeholder_exts) == len(extensions):
        result["checks"]["ascendc_ext_imported"]["error"] = (
            f"扩展导入使用了占位符名称 {placeholder_exts}（如 TORCH_EXTENSION_NAME），"
            "扩展模块未正确配置"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "扩展模块导入使用了占位符名称（如 import TORCH_EXTENSION_NAME），"
            "这表示 AscendC kernel 未正确编译或配置。"
            "请确保使用 NpuExtension 编译 kernel 并使用正确的模块名导入。"
        )
        return result

    # 过滤掉占位符，只保留有效的扩展名
    valid_ext_names = {k for k, v in extensions.items() if not v["is_placeholder"]}

    result["checks"]["ascendc_ext_imported"]["passed"] = True

    # --- Check 2: forward 是否调用 kernel ---
    forward_node, class_name = find_model_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 或 Model.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew（或 Model）类或 forward 方法。"
        return result

    reachable_functions = find_reachable_functions(tree, forward_node, class_name)
    called = check_kernel_calls(reachable_functions, valid_ext_names)
    result["checks"]["kernel_called_from_forward"]["called"] = called

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"已导入扩展模块 {list(valid_ext_names)} 但 {class_name}.forward() "
            f"未调用任何扩展函数"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已导入 AscendC 扩展模块 {list(valid_ext_names)} 但 "
            f"{class_name}.forward() 及其本地 helper 中未调用。"
            "forward() 必须直接或通过可静态追踪的 helper 调用扩展函数。"
        )
        return result

    result["checks"]["kernel_called_from_forward"]["passed"] = True

    # --- Check 3: 禁止的 torch 操作 ---
    import_sources = collect_import_sources(tree)
    torch_module_attributes = find_torch_module_attributes(tree, class_name, import_sources)
    provenance_by_id = build_function_provenance(
        reachable_functions, forward_node, import_sources, valid_ext_names
    )
    violations = check_forbidden_torch_ops(
        reachable_functions,
        import_sources,
        valid_ext_names,
        torch_module_attributes,
        provenance_by_id,
    )
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if violations:
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"forward() 中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
        )
        violation_details = "; ".join(
            f"第{v['line']}行 {v['call']}" for v in violations[:5]
        )
        result["regression_type"] = 3
        result["suggestion"] = (
            f"forward() 调用了 AscendC kernel 但仍使用 PyTorch 进行部分计算: "
            f"{violation_details}。"
            "所有核心计算必须在 AscendC kernel 中完成，"
            "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
        )
        return result

    result["checks"]["no_forbidden_torch_ops"]["passed"] = True

    # --- Check 4: 标量 for 循环退化 ---
    loop_violations = check_for_loops_over_tensors(
        reachable_functions, import_sources, valid_ext_names, provenance_by_id
    )
    result["checks"]["no_scalar_for_loops"]["violations"] = loop_violations

    if loop_violations:
        result["checks"]["no_scalar_for_loops"]["error"] = (
            f"forward() 中发现 {len(loop_violations)} 处逐元素 Python for 循环"
        )
        loop_details = "; ".join(
            f"第{v['line']}行 for {v['loop_var']} in range(...)"
            for v in loop_violations[:5]
        )
        result["regression_type"] = 4
        result["suggestion"] = (
            f"forward() 中存在逐元素 Python for 循环: {loop_details}。"
            "不能用标量逐元素写法，必须使用 AscendC kernel 的向量化 / 块级操作。"
        )
        return result

    result["checks"]["no_scalar_for_loops"]["passed"] = True

    # --- 全部通过 ---
    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="检查 AscendC 生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument(
        "--pybind-file",
        default=None,
        help="pybind11.cpp 路径；默认使用 Python 文件同目录下的 kernel/pybind11.cpp",
    )
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        if args.json:
            print(json.dumps({"valid": False, "error": f"文件不存在: {args.file}"}))
        else:
            print(f"[ERROR] 文件不存在: {args.file}")
        sys.exit(1)

    result = validate(code, filepath=args.file, pybind_file=args.pybind_file)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["valid"]:
            exts = result["checks"]["ascendc_ext_imported"]["extensions"]
            called = result["checks"]["kernel_called_from_forward"]["called"]
            print("[PASS] AscendC 实现验证通过")
            print(f"  - 导入 {len(exts)} 个扩展模块: "
                  f"{', '.join(e['used_name'] for e in exts)}")
            print(f"  - forward() 调用: "
                  f"{', '.join(c['call'] for c in called)}")
            print("  - forward() 中无禁止的 PyTorch 计算操作")
            print("  - forward() 中无逐元素 Python for 循环")
        else:
            rtype = result["regression_type"]
            type_desc = {
                1: "无 AscendC 扩展导入（纯 PyTorch / 占位符导入）",
                2: "有扩展导入但 forward() 未调用 kernel",
                3: "部分计算仍使用 PyTorch（需全部移入 AscendC kernel）",
                4: "存在逐元素 Python for 循环（需使用向量化操作）",
            }
            print(f"[FAIL] 检测到 PyTorch 退化 — Type {rtype}: "
                  f"{type_desc.get(rtype, '未知')}")

            for check_name, check_result in result["checks"].items():
                if check_result["passed"]:
                    status = "PASS"
                elif check_result["error"] or check_result.get("violations"):
                    status = "FAIL"
                else:
                    status = "SKIP"
                print(f"  [{status}] {check_name}")
                if check_result["error"]:
                    print(f"         {check_result['error']}")

            # 显示 torch 操作违规详情
            torch_violations = result["checks"]["no_forbidden_torch_ops"]["violations"]
            if torch_violations:
                print("  torch 操作违规详情:")
                for v in torch_violations:
                    print(f"    第 {v['line']} 行: {v['call']} — {v['reason']}")

            # 显示 for 循环违规详情
            loop_violations = result["checks"]["no_scalar_for_loops"]["violations"]
            if loop_violations:
                print("  for 循环违规详情:")
                for v in loop_violations:
                    print(f"    第 {v['line']} 行: for {v['loop_var']} in range(...)"
                          f" — {v['reason']}")

            print(f"\n  修复建议: {result['suggestion']}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
