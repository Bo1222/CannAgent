from __future__ import annotations

import ast
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "templates" / "ascendc_direct"
OP_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PLACEHOLDER_PATTERN = re.compile(r"@[A-Z][A-Z0-9_]*@")

NAME_PLACEHOLDERS = {
    "@OP_NAME@",
    "@OP_NAME_PASCAL@",
    "@OP_JSON_FILENAME@",
}
ABI_PLACEHOLDERS = {
    "@OP_SCHEMA@",
    "@CPP_RETURN_TYPE@",
    "@CPP_ARGUMENTS@",
    "@PY_FORWARD_SIGNATURE@",
    "@PY_RETURN_ANNOTATION@",
}


class ProjectGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class AbiParameter:
    name: str
    schema_type: str
    cpp_type: str
    python_text: str
    default_text: str | None = None
    keyword_only: bool = False


@dataclass(frozen=True)
class OperatorAbi:
    parameters: tuple[AbiParameter, ...]
    schema_return: str
    cpp_return: str
    python_return: str

    def schema(self, op_name: str) -> str:
        fields: list[str] = []
        inserted_keyword_marker = False
        for parameter in self.parameters:
            if parameter.keyword_only and not inserted_keyword_marker:
                fields.append("*")
                inserted_keyword_marker = True
            field = f"{parameter.schema_type} {parameter.name}"
            if parameter.default_text is not None:
                field += f"={parameter.default_text}"
            fields.append(field)
        return f"{op_name}({', '.join(fields)}) -> {self.schema_return}"

    @property
    def cpp_arguments(self) -> str:
        return ", ".join(f"{item.cpp_type} {item.name}" for item in self.parameters)

    @property
    def python_signature(self) -> str:
        fields: list[str] = []
        inserted_keyword_marker = False
        for parameter in self.parameters:
            if parameter.keyword_only and not inserted_keyword_marker:
                fields.append("*")
                inserted_keyword_marker = True
            fields.append(parameter.python_text)
        return ", ".join(fields)


@dataclass(frozen=True)
class ProjectSpec:
    op_name: str
    op_name_pascal: str
    op_file: Path
    op_json: Path
    op_json_filename: str
    abi: OperatorAbi

    @property
    def editable_paths(self) -> tuple[str, ...]:
        return (
            f"op_kernel/{self.op_name}_kernel.asc",
            f"op_kernel/{self.op_name}_tiling.h",
            f"op_host/{self.op_name}.asc",
            f"op_extension/{self.op_name}_torch.cpp",
            "model_new_ascendc.py",
        )

    @property
    def fixed_paths(self) -> tuple[str, ...]:
        return (
            "CMakeLists.txt",
            "op_extension/ops.h",
            "op_extension/register.cpp",
            "op_host/data_utils.h",
            "model.py",
            self.op_json_filename,
        )


_SCALAR_TYPES = {
    "int": ("int", "int64_t"),
    "float": ("float", "double"),
    "bool": ("bool", "bool"),
    "str": ("str", "const std::string&"),
}
_TENSOR_NAMES = {"Tensor", "torch.Tensor"}
_NONE_NAMES = {"None", "NoneType", "types.NoneType"}


def validate_op_name(value: str) -> str:
    if not isinstance(value, str) or not OP_NAME_PATTERN.fullmatch(value):
        raise ProjectGenerationError(
            "op-name must be a C/C++ identifier containing only letters, digits, and underscores"
        )
    if value in {".", ".."}:
        raise ProjectGenerationError("op-name is not a safe file name")
    return value


def pascal_case(value: str) -> str:
    converted = "".join(part[:1].upper() + part[1:] for part in value.split("_") if part)
    if not converted:
        raise ProjectGenerationError("op-name cannot be converted to a PascalCase identifier")
    if converted[0].isdigit():
        converted = "Op" + converted
    return converted


def _annotation_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _annotation_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _subscript_parts(node: ast.Subscript) -> list[ast.expr]:
    value = node.slice
    return list(value.elts) if isinstance(value, ast.Tuple) else [value]


def _is_none_annotation(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant)
        and node.value is None
        or _annotation_name(node) in _NONE_NAMES
    )


def _type_mapping(node: ast.expr, *, is_return: bool = False) -> tuple[str, str]:
    name = _annotation_name(node)
    if name in _TENSOR_NAMES:
        return "Tensor", "at::Tensor" if is_return else "const at::Tensor&"
    if name in _SCALAR_TYPES:
        schema, cpp = _SCALAR_TYPES[name]
        return schema, cpp
    if isinstance(node, ast.Constant) and node.value is None or name in _NONE_NAMES:
        return "None", "void"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        inner = (
            node.right
            if _is_none_annotation(node.left)
            else node.left
            if _is_none_annotation(node.right)
            else None
        )
        if inner is None:
            raise ProjectGenerationError("only T | None unions are supported in Model.forward")
        schema, cpp = _type_mapping(inner, is_return=is_return)
        if schema == "None":
            raise ProjectGenerationError("None | None is not a valid operator type")
        optional_cpp = f"c10::optional<{cpp.replace('const ', '').rstrip('&').strip()}>"
        if not is_return:
            optional_cpp = f"const {optional_cpp}&"
        return schema + "?", optional_cpp
    if isinstance(node, ast.Subscript):
        container = _annotation_name(node.value)
        parts = _subscript_parts(node)
        if container in {"Optional", "typing.Optional"} and len(parts) == 1:
            schema, cpp = _type_mapping(parts[0], is_return=is_return)
            optional_cpp = f"c10::optional<{cpp.replace('const ', '').rstrip('&').strip()}>"
            if not is_return:
                optional_cpp = f"const {optional_cpp}&"
            return schema + "?", optional_cpp
        if container in {"list", "List", "typing.List", "Sequence", "typing.Sequence"} and len(parts) == 1:
            schema, _ = _type_mapping(parts[0], is_return=False)
            cpp_lists = (
                {
                    "Tensor": "std::vector<at::Tensor>",
                    "int": "std::vector<int64_t>",
                    "float": "std::vector<double>",
                    "bool": "std::vector<bool>",
                    "str": "std::vector<std::string>",
                }
                if is_return
                else {
                    "Tensor": "at::TensorList",
                    "int": "at::IntArrayRef",
                    "float": "c10::ArrayRef<double>",
                    "bool": "c10::ArrayRef<bool>",
                    "str": "c10::ArrayRef<std::string>",
                }
            )
            if schema not in cpp_lists:
                raise ProjectGenerationError(f"unsupported list element type: {ast.unparse(parts[0])}")
            return schema + "[]", cpp_lists[schema]
        if container in {"tuple", "Tuple", "typing.Tuple"}:
            if len(parts) == 2 and isinstance(parts[1], ast.Constant) and parts[1].value is Ellipsis:
                schema, _ = _type_mapping(parts[0], is_return=False)
                if schema != "int":
                    raise ProjectGenerationError("only tuple[int, ...] is supported as an input")
                return "int[]", "at::IntArrayRef"
            if not is_return:
                raise ProjectGenerationError("fixed-length tuple inputs are not supported")
            mapped = [_type_mapping(part, is_return=True) for part in parts]
            if not mapped or any(schema != "Tensor" for schema, _ in mapped):
                raise ProjectGenerationError("only tuple returns containing Tensor values are supported")
            return (
                "(" + ", ".join(schema for schema, _ in mapped) + ")",
                "std::tuple<" + ", ".join(cpp for _, cpp in mapped) + ">",
            )
    raise ProjectGenerationError(f"unsupported Model.forward annotation: {ast.unparse(node)}")


def _schema_default(node: ast.expr) -> str:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise ProjectGenerationError(
            f"Model.forward default must be a literal: {ast.unparse(node)}"
        ) from error
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)) and all(isinstance(item, (int, float, bool)) for item in value):
        return "[" + ", ".join(repr(item) for item in value) + "]"
    raise ProjectGenerationError(f"unsupported Model.forward default: {value!r}")


def parse_operator_abi(op_file: Path) -> OperatorAbi:
    try:
        source = op_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ProjectGenerationError(f"cannot read op-file as UTF-8: {op_file}") from error
    try:
        tree = ast.parse(source, filename=str(op_file))
    except SyntaxError as error:
        raise ProjectGenerationError(f"op-file is not valid Python: {error}") from error
    models = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"]
    if len(models) != 1:
        raise ProjectGenerationError("op-file must define exactly one top-level class Model")
    forwards = [
        node
        for node in models[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
    ]
    if len(forwards) != 1 or isinstance(forwards[0], ast.AsyncFunctionDef):
        raise ProjectGenerationError("class Model must define exactly one synchronous forward method")
    forward = forwards[0]
    arguments = forward.args
    if arguments.vararg or arguments.kwarg:
        raise ProjectGenerationError("Model.forward cannot use *args or **kwargs")
    if arguments.posonlyargs:
        raise ProjectGenerationError("Model.forward positional-only parameters are not supported")
    positional = [*arguments.posonlyargs, *arguments.args]
    if not positional or positional[0].arg != "self":
        raise ProjectGenerationError("Model.forward must use self as its first parameter")
    positional = positional[1:]
    positional_defaults: dict[str, ast.expr] = {}
    if arguments.defaults:
        for arg, default in zip(positional[-len(arguments.defaults) :], arguments.defaults):
            positional_defaults[arg.arg] = default
    records: list[AbiParameter] = []

    def add_parameter(arg: ast.arg, default: ast.expr | None, *, keyword_only: bool) -> None:
        if arg.annotation is None:
            raise ProjectGenerationError(f"Model.forward parameter {arg.arg!r} requires a type annotation")
        schema_type, cpp_type = _type_mapping(arg.annotation)
        default_text = _schema_default(default) if default is not None else None
        if default_text == "None" and not schema_type.endswith("?"):
            schema_type += "?"
            base_cpp = cpp_type.replace("const ", "").rstrip("&").strip()
            cpp_type = f"const c10::optional<{base_cpp}>&"
        annotation_text = ast.unparse(arg.annotation)
        python_text = f"{arg.arg}: {annotation_text}"
        if default is not None:
            python_text += f" = {ast.unparse(default)}"
        records.append(
            AbiParameter(
                name=arg.arg,
                schema_type=schema_type,
                cpp_type=cpp_type,
                python_text=python_text,
                default_text=default_text,
                keyword_only=keyword_only,
            )
        )

    for arg in positional:
        add_parameter(arg, positional_defaults.get(arg.arg), keyword_only=False)
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        add_parameter(arg, default, keyword_only=True)
    if not any(item.schema_type.startswith("Tensor") for item in records):
        raise ProjectGenerationError("Model.forward must contain at least one Tensor parameter")
    if forward.returns is None:
        raise ProjectGenerationError("Model.forward requires an explicit return annotation")
    schema_return, cpp_return = _type_mapping(forward.returns, is_return=True)
    if schema_return == "None":
        raise ProjectGenerationError("Model.forward must return at least one Tensor")
    return OperatorAbi(
        tuple(records),
        schema_return,
        cpp_return,
        ast.unparse(forward.returns),
    )


def validate_json_or_jsonl(path: Path) -> None:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ProjectGenerationError(f"cannot read op-json as UTF-8: {path}") from error
    if not text.strip():
        raise ProjectGenerationError("op-json cannot be empty")
    try:
        json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    parsed = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            raise ProjectGenerationError(
                f"op-json is neither JSON nor JSONL; invalid line {line_number}: {error.msg}"
            ) from error
        parsed += 1
    if parsed == 0:
        raise ProjectGenerationError("op-json cannot be empty")


def build_project_spec(op_name: str, op_file: Path | str, op_json: Path | str) -> ProjectSpec:
    name = validate_op_name(op_name)
    source = Path(op_file).expanduser().resolve()
    cases = Path(op_json).expanduser().resolve()
    if not source.is_file():
        raise ProjectGenerationError(f"op-file does not exist: {source}")
    if not cases.is_file():
        raise ProjectGenerationError(f"op-json does not exist: {cases}")
    if cases.suffix.lower() != ".json" or cases.name in {".", ".."}:
        raise ProjectGenerationError("op-json must have a safe .json file name")
    validate_json_or_jsonl(cases)
    return ProjectSpec(
        op_name=name,
        op_name_pascal=pascal_case(name),
        op_file=source,
        op_json=cases,
        op_json_filename=cases.name,
        abi=parse_operator_abi(source),
    )


def _render(template: Path, replacements: dict[str, str], *, names_only: bool = False) -> str:
    text = template.read_text(encoding="utf-8")
    present = set(PLACEHOLDER_PATTERN.findall(text))
    allowed = NAME_PLACEHOLDERS if names_only else NAME_PLACEHOLDERS | ABI_PLACEHOLDERS
    unexpected = sorted(present - allowed)
    if unexpected:
        raise ProjectGenerationError(f"template contains unsupported placeholders: {unexpected}")
    for placeholder in sorted(present):
        if placeholder not in replacements:
            raise ProjectGenerationError(f"template replacement is missing: {placeholder}")
        text = text.replace(placeholder, replacements[placeholder])
    remaining = sorted(set(PLACEHOLDER_PATTERN.findall(text)))
    if remaining:
        raise ProjectGenerationError(f"rendered template contains placeholders: {remaining}")
    return text


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _populate_project(root: Path, spec: ProjectSpec, template_root: Path) -> None:
    cpp_schema = spec.abi.schema(spec.op_name).replace("\\", "\\\\").replace('"', '\\"')
    replacements = {
        "@OP_NAME@": spec.op_name,
        "@OP_NAME_PASCAL@": spec.op_name_pascal,
        "@OP_JSON_FILENAME@": spec.op_json_filename,
        "@OP_SCHEMA@": cpp_schema,
        "@CPP_RETURN_TYPE@": spec.abi.cpp_return,
        "@CPP_ARGUMENTS@": spec.abi.cpp_arguments,
        "@PY_FORWARD_SIGNATURE@": spec.abi.python_signature,
        "@PY_RETURN_ANNOTATION@": spec.abi.python_return,
    }
    files = {
        "CMakeLists.txt": ("CMakeLists.txt.in", True),
        f"op_extension/{spec.op_name}_torch.cpp": ("op_extension/op_torch.cpp.in", False),
        "op_extension/ops.h": ("op_extension/ops.h.in", False),
        "op_extension/register.cpp": ("op_extension/register.cpp.in", False),
        f"op_host/{spec.op_name}.asc": ("op_host/op.asc.in", False),
        "op_host/data_utils.h": ("op_host/data_utils.h.in", True),
        f"op_kernel/{spec.op_name}_kernel.asc": ("op_kernel/op_kernel.asc.in", False),
        f"op_kernel/{spec.op_name}_tiling.h": ("op_kernel/op_tiling.h.in", False),
        "model_new_ascendc.py": ("model_new_ascendc.py.in", False),
    }
    for output_name, (template_name, names_only) in files.items():
        template = template_root / template_name
        if not template.is_file():
            raise ProjectGenerationError(f"required template does not exist: {template}")
        _write_text(root / output_name, _render(template, replacements, names_only=names_only))
    (root / "scripts").mkdir(parents=True, exist_ok=False)
    shutil.copyfile(spec.op_file, root / "model.py")
    shutil.copyfile(spec.op_json, root / spec.op_json_filename)


def create_project(
    *,
    op_name: str,
    op_file: Path | str,
    op_json: Path | str,
    output: Path | str,
    template_root: Path = TEMPLATE_ROOT,
) -> ProjectSpec:
    spec = build_project_spec(op_name, op_file, op_json)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ProjectGenerationError(f"output exists and is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ProjectGenerationError(f"output directory is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        _populate_project(temporary, spec, template_root.resolve())
        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return spec
