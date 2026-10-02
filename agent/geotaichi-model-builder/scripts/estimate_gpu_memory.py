#!/usr/bin/env python3
"""Estimate the Taichi GPU pool reserved by a GeoTaichi Python script.

The target script is parsed, never imported or executed.  The estimate follows
``geotaichi.init``/``taichi.init`` configuration, including simple variables,
environment defaults, arithmetic, and the small environment helper functions
used by maintained examples.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import build_error, build_ok, emit


TAICHI_DEFAULT_DEVICE_MEMORY_GIB = 1.0
GPU_ARCHES = {"gpu", "cuda", "metal", "vulkan", "opengl", "dx11"}
CPU_ARCHES = {"cpu", "x64", "arm64"}
UNKNOWN = object()


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


@dataclass
class InitCall:
    owner: str
    node: ast.Call
    scopes: list[dict[str, Any]]


class SafeEvaluator:
    """Evaluate the small, side-effect-free expression subset used in examples."""

    _binary_operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_operations = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }
    _comparisons = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda left, right: left in right,
        ast.NotIn: lambda left, right: left not in right,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
    }

    def __init__(
        self,
        definitions: list[dict[str, Any]],
        functions: dict[str, ast.FunctionDef],
        environment: dict[str, str],
        target_platform: str,
        target_machine: str,
    ):
        self.definitions = definitions
        self.functions = functions
        self.environment = environment
        self.target_platform = target_platform
        self.target_machine = target_machine
        self._resolving: set[tuple[int, str]] = set()
        self._function_stack: set[str] = set()

    def with_scope(self, scope: dict[str, Any]) -> "SafeEvaluator":
        return SafeEvaluator(
            [*self.definitions, scope],
            self.functions,
            self.environment,
            self.target_platform,
            self.target_machine,
        )

    def evaluate(self, node: ast.AST | None) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "__name__":
                return "__main__"
            if node.id in {"True", "False", "None"}:
                return {"True": True, "False": False, "None": None}[node.id]
            for scope_index in range(len(self.definitions) - 1, -1, -1):
                scope = self.definitions[scope_index]
                if node.id not in scope:
                    continue
                marker = (scope_index, node.id)
                if marker in self._resolving:
                    return UNKNOWN
                self._resolving.add(marker)
                value = scope[node.id]
                result = self.evaluate(value) if isinstance(value, ast.AST) else value
                self._resolving.remove(marker)
                return result
            return UNKNOWN
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.evaluate(item) for item in node.elts]
            if UNKNOWN in values:
                return UNKNOWN
            if isinstance(node, ast.Tuple):
                return tuple(values)
            if isinstance(node, ast.Set):
                return set(values)
            return values
        if isinstance(node, ast.Dict):
            keys = [self.evaluate(item) for item in node.keys]
            values = [self.evaluate(item) for item in node.values]
            if UNKNOWN in keys or UNKNOWN in values:
                return UNKNOWN
            return dict(zip(keys, values))
        if isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand)
            operation = self._unary_operations.get(type(node.op))
            return self._apply(operation, operand)
        if isinstance(node, ast.BinOp):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            operation = self._binary_operations.get(type(node.op))
            return self._apply(operation, left, right)
        if isinstance(node, ast.BoolOp):
            values = [self.evaluate(item) for item in node.values]
            if UNKNOWN in values:
                return UNKNOWN
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self.evaluate(node.left)
            if left is UNKNOWN:
                return UNKNOWN
            for operation_node, comparator_node in zip(node.ops, node.comparators):
                right = self.evaluate(comparator_node)
                operation = self._comparisons.get(type(operation_node))
                if right is UNKNOWN or operation is None:
                    return UNKNOWN
                try:
                    if not operation(left, right):
                        return False
                except (TypeError, ValueError):
                    return UNKNOWN
                left = right
            return True
        if isinstance(node, ast.IfExp):
            condition = self.evaluate(node.test)
            if condition is UNKNOWN:
                return UNKNOWN
            return self.evaluate(node.body if condition else node.orelse)
        if isinstance(node, ast.Subscript):
            if _dotted_name(node.value) == "os.environ":
                key = self.evaluate(node.slice)
                return self.environment.get(str(key), UNKNOWN) if key is not UNKNOWN else UNKNOWN
            value = self.evaluate(node.value)
            index = self.evaluate(node.slice)
            return self._apply(operator.getitem, value, index)
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for item in node.values:
                if isinstance(item, ast.Constant):
                    parts.append(str(item.value))
                elif isinstance(item, ast.FormattedValue):
                    value = self.evaluate(item.value)
                    if value is UNKNOWN:
                        return UNKNOWN
                    parts.append(str(value))
            return "".join(parts)
        if isinstance(node, ast.Attribute):
            name = _dotted_name(node)
            if name and name.startswith("ti."):
                return name.split(".")[-1]
            value = self.evaluate(node.value)
            if value is UNKNOWN:
                return UNKNOWN
            if node.attr in ("real", "imag"):
                return getattr(value, node.attr, UNKNOWN)
            return UNKNOWN
        if isinstance(node, ast.Call):
            return self._evaluate_call(node)
        return UNKNOWN

    @staticmethod
    def _apply(operation, *values):
        if operation is None or any(value is UNKNOWN for value in values):
            return UNKNOWN
        try:
            return operation(*values)
        except (ArithmeticError, TypeError, ValueError, KeyError, IndexError):
            return UNKNOWN

    def _evaluate_call(self, node: ast.Call) -> Any:
        name = _dotted_name(node.func)
        args = [self.evaluate(item) for item in node.args]
        kwargs = {item.arg: self.evaluate(item.value) for item in node.keywords if item.arg}
        if name in ("float", "int", "str", "bool", "abs", "round", "min", "max"):
            functions = {
                "float": float,
                "int": int,
                "str": str,
                "bool": bool,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
            }
            return self._apply(lambda *values: functions[name](*values, **kwargs), *args)
        if name in ("os.environ.get", "os.getenv"):
            if not args or args[0] is UNKNOWN:
                return UNKNOWN
            key = str(args[0])
            default = args[1] if len(args) > 1 else kwargs.get("default", None)
            return self.environment.get(key, default)
        if name == "platform.system":
            return self.target_platform
        if name == "platform.machine":
            return self.target_machine
        if name == "os.path.exists" and args and args[0] is not UNKNOWN:
            if self.target_platform != platform.system():
                return UNKNOWN
            return Path(str(args[0])).exists()
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("lower", "upper", "strip"):
            value = self.evaluate(node.func.value)
            if value is UNKNOWN:
                return UNKNOWN
            return self._apply(getattr(str(value), node.func.attr), *args)
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            return self._evaluate_user_function(node.func.id, node)
        return UNKNOWN

    def _evaluate_user_function(self, name: str, call: ast.Call) -> Any:
        if name in self._function_stack:
            return UNKNOWN
        function = self.functions[name]
        parameters = list(function.args.posonlyargs) + list(function.args.args)
        defaults = [None] * (len(parameters) - len(function.args.defaults)) + list(function.args.defaults)
        local_scope: dict[str, Any] = {}
        for parameter, default in zip(parameters, defaults):
            if default is not None:
                local_scope[parameter.arg] = default
        for parameter, value in zip(parameters, call.args):
            local_scope[parameter.arg] = value
        for keyword in call.keywords:
            if keyword.arg:
                local_scope[keyword.arg] = keyword.value
        evaluator = self.with_scope(local_scope)
        evaluator._function_stack = {*self._function_stack, name}
        for statement in function.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                evaluator._record_assignment(statement, local_scope)
            elif isinstance(statement, ast.If):
                condition = evaluator.evaluate(statement.test)
                if condition is UNKNOWN and any(
                    isinstance(item, ast.Return) for item in (*statement.body, *statement.orelse)
                ):
                    return UNKNOWN
                branch = statement.body if condition is True else statement.orelse if condition is False else []
                for branch_statement in branch:
                    if isinstance(branch_statement, ast.Return):
                        return evaluator.evaluate(branch_statement.value)
                    if isinstance(branch_statement, (ast.Assign, ast.AnnAssign)):
                        evaluator._record_assignment(branch_statement, local_scope)
            elif isinstance(statement, ast.Return):
                return evaluator.evaluate(statement.value)
        return UNKNOWN

    def _record_assignment(self, statement: ast.Assign | ast.AnnAssign, scope: dict[str, Any]) -> None:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if isinstance(target, ast.Name):
                scope[target.id] = statement.value


class ScriptAnalyzer:
    def __init__(
        self,
        tree: ast.Module,
        environment: dict[str, str],
        target_platform: str,
        target_machine: str,
    ):
        self.tree = tree
        self.environment = environment
        self.target_platform = target_platform
        self.target_machine = target_machine
        self.functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.module_scope: dict[str, Any] = {}
        self.geotaichi_init_names: set[str] = set()
        self.taichi_init_names: set[str] = set()
        self.module_aliases: dict[str, str] = {}
        self.init_calls: list[InitCall] = []
        self._active_functions: set[str] = set()
        self._collect_imports()

    def _collect_imports(self) -> None:
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    if alias.name in ("geotaichi", "src", "taichi"):
                        self.module_aliases[local] = "taichi" if alias.name == "taichi" else "geotaichi"
            elif isinstance(node, ast.ImportFrom):
                if node.module in ("geotaichi", "src"):
                    for alias in node.names:
                        if alias.name == "*":
                            self.geotaichi_init_names.add("init")
                        elif alias.name == "init":
                            self.geotaichi_init_names.add(alias.asname or alias.name)
                elif node.module == "taichi":
                    for alias in node.names:
                        if alias.name == "init":
                            self.taichi_init_names.add(alias.asname or alias.name)

    def analyze(self) -> list[InitCall]:
        self._scan_statements(self.tree.body, [self.module_scope])
        unique: dict[tuple[int, int, str], InitCall] = {}
        for item in self.init_calls:
            key = (item.node.lineno, item.node.col_offset, item.owner)
            unique[key] = item
        return list(unique.values())

    def _evaluator(self, scopes: list[dict[str, Any]]) -> SafeEvaluator:
        return SafeEvaluator(
            scopes,
            self.functions,
            self.environment,
            self.target_platform,
            self.target_machine,
        )

    def _scan_statements(self, statements: list[ast.stmt], scopes: list[dict[str, Any]]) -> None:
        scope = scopes[-1]
        evaluator = self._evaluator(scopes)
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                self._record_assignment(statement, scope, evaluator)
                self._scan_expression(statement.value, scopes)
                continue
            if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                previous = evaluator.evaluate(ast.Name(id=statement.target.id))
                value = evaluator.evaluate(statement.value)
                operation = SafeEvaluator._binary_operations.get(type(statement.op))
                result = SafeEvaluator._apply(operation, previous, value)
                scope[statement.target.id] = result
                continue
            if isinstance(statement, ast.Expr):
                self._scan_expression(statement.value, scopes)
                continue
            if isinstance(statement, ast.If):
                condition = evaluator.evaluate(statement.test)
                if condition is True:
                    self._scan_statements(statement.body, scopes)
                elif condition is False:
                    self._scan_statements(statement.orelse, scopes)
                else:
                    self._scan_statements(statement.body, [*scopes[:-1], dict(scope)])
                    self._scan_statements(statement.orelse, [*scopes[:-1], dict(scope)])
                continue
            if isinstance(statement, (ast.For, ast.While, ast.With, ast.AsyncWith)):
                self._scan_statements(statement.body, scopes)
                self._scan_statements(statement.orelse, scopes)
                continue
            if isinstance(statement, ast.Try):
                self._scan_statements(statement.body, scopes)
                self._scan_statements(statement.orelse, scopes)
                for handler in statement.handlers:
                    self._scan_statements(handler.body, scopes)
                self._scan_statements(statement.finalbody, scopes)

    def _record_assignment(
        self,
        statement: ast.Assign | ast.AnnAssign,
        scope: dict[str, Any],
        evaluator: SafeEvaluator,
    ) -> None:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if isinstance(target, ast.Name):
                scope[target.id] = statement.value
            elif isinstance(target, ast.Subscript) and _dotted_name(target.value) == "os.environ":
                key = evaluator.evaluate(target.slice)
                value = evaluator.evaluate(statement.value)
                if key is not UNKNOWN and value is not UNKNOWN:
                    self.environment[str(key)] = str(value)

    def _scan_expression(self, expression: ast.AST, scopes: list[dict[str, Any]]) -> None:
        if isinstance(expression, ast.Lambda):
            return
        if not isinstance(expression, ast.Call):
            for child in ast.iter_child_nodes(expression):
                if isinstance(child, ast.expr):
                    self._scan_expression(child, scopes)
            return
        for argument in expression.args:
            self._scan_expression(argument, scopes)
        for keyword in expression.keywords:
            self._scan_expression(keyword.value, scopes)
        owner = self._init_owner(expression)
        if owner:
            self.init_calls.append(InitCall(owner, expression, [dict(scope) for scope in scopes]))
        elif isinstance(expression.func, ast.Name) and expression.func.id in self.functions:
            self._scan_function(expression.func.id, expression, scopes)

    def _scan_function(self, name: str, call: ast.Call, parent_scopes: list[dict[str, Any]]) -> None:
        if name in self._active_functions:
            return
        function = self.functions[name]
        parameters = list(function.args.posonlyargs) + list(function.args.args)
        defaults = [None] * (len(parameters) - len(function.args.defaults)) + list(function.args.defaults)
        scope: dict[str, Any] = {}
        for parameter, default in zip(parameters, defaults):
            if default is not None:
                scope[parameter.arg] = default
        for parameter, value in zip(parameters, call.args):
            scope[parameter.arg] = value
        for keyword in call.keywords:
            if keyword.arg:
                scope[keyword.arg] = keyword.value
        self._active_functions.add(name)
        self._scan_statements(function.body, [*parent_scopes, scope])
        self._active_functions.remove(name)

    def _init_owner(self, call: ast.Call) -> str | None:
        if isinstance(call.func, ast.Name):
            if call.func.id in self.geotaichi_init_names:
                return "geotaichi"
            if call.func.id in self.taichi_init_names:
                return "taichi"
            return None
        name = _dotted_name(call.func)
        if not name or not name.endswith(".init"):
            return None
        root = name.split(".")[0]
        return self.module_aliases.get(root)


def _keyword(call: ast.Call, name: str, positional_index: int | None = None) -> ast.AST | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    if positional_index is not None and len(call.args) > positional_index:
        return call.args[positional_index]
    return None


def _number(value: Any) -> float | None:
    if value is UNKNOWN or value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_arch(value: Any, owner: str) -> str | None:
    if value is UNKNOWN:
        return None
    if value is None:
        return "cpu" if owner == "taichi" else "gpu"
    return str(value).strip().lower().split(".")[-1]


def _detect_nvidia_memory() -> tuple[float | None, float | None, str | None]:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            divisor = float(1024**3)
            return info.total / divisor, info.free / divisor, str(name)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None, None, None


def _evaluate_init_call(
    item: InitCall,
    functions: dict[str, ast.FunctionDef],
    environment: dict[str, str],
    target_platform: str,
    target_machine: str,
    gpu_total_gib: float | None,
    gpu_free_gib: float | None,
) -> dict[str, Any]:
    call = item.node
    evaluator = SafeEvaluator(
        item.scopes,
        functions,
        environment,
        target_platform,
        target_machine,
    )
    arch_index = 1 if item.owner == "geotaichi" else 0
    arch_node = _keyword(call, "arch", arch_index)
    arch_value = evaluator.evaluate(arch_node)
    if arch_node is None:
        arch_value = evaluator.environment.get("TI_ARCH", None)
    requested_arch = _normalize_arch(arch_value, item.owner)
    effective_arch = requested_arch
    notes: list[str] = []
    if (
        item.owner == "geotaichi"
        and requested_arch == "cpu"
        and target_platform == "Darwin"
        and target_machine == "arm64"
        and evaluator.environment.get("GEOTAICHI_FORCE_CPU", "0") != "1"
    ):
        effective_arch = "metal"
        notes.append("GeoTaichi maps arch='cpu' to Metal on Apple Silicon unless GEOTAICHI_FORCE_CPU=1.")

    gib_index = 6 if item.owner == "geotaichi" else None
    fraction_index = 7 if item.owner == "geotaichi" else None
    gib_node = _keyword(call, "device_memory_GB", gib_index)
    fraction_node = _keyword(call, "device_memory_fraction", fraction_index)
    gib_value = evaluator.evaluate(gib_node) if gib_node is not None else None
    fraction_value = evaluator.evaluate(fraction_node) if fraction_node is not None else None
    source = "script"
    if gib_node is None and fraction_node is None:
        if "TI_DEVICE_MEMORY_GB" in evaluator.environment:
            gib_value = evaluator.environment["TI_DEVICE_MEMORY_GB"]
            source = "TI_DEVICE_MEMORY_GB"
        elif "TI_DEVICE_MEMORY_FRACTION" in evaluator.environment:
            fraction_value = evaluator.environment["TI_DEVICE_MEMORY_FRACTION"]
            source = "TI_DEVICE_MEMORY_FRACTION"
        else:
            gib_value = TAICHI_DEFAULT_DEVICE_MEMORY_GIB
            source = "Taichi 1.7 default"

    requested_gib = _number(gib_value)
    fraction = _number(fraction_value)
    if requested_gib is not None and fraction is not None:
        if item.owner == "geotaichi":
            notes.append("Both pool controls are set; geotaichi.init gives device_memory_GB precedence.")
            fraction = None
        else:
            notes.append("Both Taichi pool controls are set; configure only one to avoid backend-dependent precedence.")
    estimated_gib: float | None = None
    confidence = "exact-configured-pool"
    if effective_arch in CPU_ARCHES:
        estimated_gib = 0.0
        confidence = "exact-no-gpu-backend"
        if gib_node is not None or fraction_node is not None:
            notes.append("Device-memory settings do not reserve GPU memory on a CPU backend.")
    elif effective_arch in GPU_ARCHES:
        if item.owner == "geotaichi" and target_platform == "Darwin":
            estimated_gib = TAICHI_DEFAULT_DEVICE_MEMORY_GIB
            confidence = "configured-Taichi-default"
            if gib_node is not None or fraction_node is not None:
                notes.append(
                    "The current GeoTaichi Darwin branch does not forward device_memory_GB/device_memory_fraction "
                    "to ti.init, so Taichi's default pool is used."
                )
        elif fraction is not None:
            if gpu_total_gib is None:
                confidence = "requires-gpu-total"
                notes.append("Provide --gpu-total-gib, or run on an NVML-visible GPU, to convert the fraction to GiB.")
            else:
                estimated_gib = fraction * gpu_total_gib
                confidence = "exact-fraction-of-reported-total"
        elif requested_gib is not None:
            estimated_gib = requested_gib
            if item.owner == "geotaichi" and target_platform != "Darwin" and gpu_free_gib is not None:
                capped = min(requested_gib, round(gpu_free_gib, 2))
                if capped != requested_gib:
                    estimated_gib = capped
                    confidence = "current-launch-cap"
                    notes.append(
                        "GeoTaichi caps device_memory_GB to currently free GPU memory; "
                        "the configured request is larger."
                    )
        else:
            confidence = "unresolved-expression"
            notes.append("The memory expression could not be resolved without executing the model script.")
    else:
        confidence = "unresolved-backend"
        notes.append("The backend expression could not be resolved statically.")

    return {
        "line": call.lineno,
        "initializer": f"{item.owner}.init",
        "requested_arch": requested_arch,
        "effective_arch": effective_arch,
        "device_memory_GB": requested_gib,
        "device_memory_fraction": fraction,
        "configured_source": source,
        "estimated_preallocated_pool_gib": estimated_gib,
        "confidence": confidence,
        "notes": notes,
    }


def estimate_script(
    script: str | Path,
    *,
    environment: dict[str, str] | None = None,
    target_platform: str | None = None,
    target_machine: str | None = None,
    gpu_total_gib: float | None = None,
    gpu_free_gib: float | None = None,
) -> dict[str, Any]:
    path = Path(script).resolve()
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    environment = dict(os.environ if environment is None else environment)
    target_platform = target_platform or platform.system()
    target_machine = target_machine or platform.machine()
    analyzer = ScriptAnalyzer(tree, environment, target_platform, target_machine)
    calls = analyzer.analyze()
    estimates = [
        _evaluate_init_call(
            item,
            analyzer.functions,
            environment,
            target_platform,
            target_machine,
            gpu_total_gib,
            gpu_free_gib,
        )
        for item in calls
    ]
    values = [
        item["estimated_preallocated_pool_gib"]
        for item in estimates
        if item["estimated_preallocated_pool_gib"] is not None
    ]
    summary_value = values[0] if len(estimates) == 1 and len(values) == 1 else None
    warnings: list[str] = []
    if not estimates:
        warnings.append("No imported geotaichi.init or taichi.init call was found on a statically reachable path.")
    elif len(estimates) > 1:
        warnings.append("Multiple reachable init calls were found; they are alternatives, not memory values to sum.")
    return {
        "script": str(path),
        "target": {
            "platform": target_platform,
            "machine": target_machine,
            "gpu_total_gib": gpu_total_gib,
            "gpu_free_gib": gpu_free_gib,
        },
        "estimated_preallocated_pool_gib": summary_value,
        "init_calls": estimates,
        "warnings": warnings,
        "scope": {
            "included": "Taichi preallocated device-memory pool configured by init",
            "excluded": [
                "CUDA/graphics driver context and kernel-module overhead",
                "non-Taichi allocations made by external CUDA libraries",
                "host RAM",
                "active-field byte accounting inside the reserved pool",
            ],
        },
    }


def _environment_overrides(entries: list[str]) -> dict[str, str]:
    result = dict(os.environ)
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--env expects NAME=VALUE, received {entry!r}")
        name, value = entry.split("=", 1)
        if not name:
            raise ValueError("--env variable name cannot be empty")
        result[name] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="GeoTaichi Python model script")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a script environment value",
    )
    parser.add_argument("--platform", choices=("current", "linux", "darwin", "windows"), default="current")
    parser.add_argument("--machine", help="Target machine name, for example x86_64 or arm64")
    parser.add_argument("--gpu-total-gib", type=float, help="Target GPU total capacity for device_memory_fraction")
    parser.add_argument(
        "--gpu-free-gib",
        type=float,
        help="Currently free target GPU memory for GeoTaichi's launch cap",
    )
    parser.add_argument("--no-nvml", action="store_true", help="Do not auto-detect current NVIDIA GPU memory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        environment = _environment_overrides(args.env)
        platform_names = {
            "linux": "Linux",
            "darwin": "Darwin",
            "windows": "Windows",
        }
        target_platform = platform.system() if args.platform == "current" else platform_names[args.platform]
        target_machine = args.machine or platform.machine()
        gpu_total_gib = args.gpu_total_gib
        gpu_free_gib = args.gpu_free_gib
        detected_name = None
        if (
            not args.no_nvml
            and target_platform == platform.system()
            and (gpu_total_gib is None or gpu_free_gib is None)
        ):
            detected_total, detected_free, detected_name = _detect_nvidia_memory()
            gpu_total_gib = gpu_total_gib if gpu_total_gib is not None else detected_total
            gpu_free_gib = gpu_free_gib if gpu_free_gib is not None else detected_free
        result = estimate_script(
            args.script,
            environment=environment,
            target_platform=target_platform,
            target_machine=target_machine,
            gpu_total_gib=gpu_total_gib,
            gpu_free_gib=gpu_free_gib,
        )
        if detected_name:
            result["target"]["detected_gpu"] = detected_name
        if not result["init_calls"]:
            emit(build_error("init_not_found", result["warnings"][0], result))
            return 1
        emit(build_ok(result))
        return 0
    except FileNotFoundError as exc:
        emit(build_error("input_not_found", str(exc)))
        return 1
    except SyntaxError as exc:
        emit(build_error("python_syntax_error", str(exc), {"line": exc.lineno, "offset": exc.offset}))
        return 1
    except Exception as exc:
        emit(build_error("gpu_memory_estimation_error", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
