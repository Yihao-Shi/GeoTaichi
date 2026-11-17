#!/usr/bin/env python3
"""Build the post-paper solver inventory and conservative guard-audit ledger.

The report is intentionally conservative: static analysis may nominate a guard
for hoisting or one-time binding, but it never authorizes deletion.  A manual
decision, a behavior-invariant test, and a regression remain mandatory before
source code is changed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SOLVER_ROOTS = (
    "src/mpm",
    "src/dem",
    "src/mpdem",
    "src/fem",
    "src/fedem",
    "src/fempm",
    "src/iga",
    "src/igampm",
)
HISTORICAL_REF = "6f07377ecbd9e2bb43140b32083edeed439797c0"
LEDGER_COLUMNS = (
    "solver",
    "file",
    "class",
    "function",
    "kind",
    "role",
    "line",
    "callers",
    "caller_resolution",
    "configured_paths",
    "preconditions",
    "mutable_inputs",
    "device_behavior",
    "guards",
    "frequency",
    "classification",
    "proof",
    "regression",
)

GUARD_DECISION_COLUMNS = (
    "file",
    "class",
    "function",
    "guard_line",
    "final_decision",
    "configured_caller_proof",
    "behavior_invariant_test",
    "regression",
    "notes",
)

STYLE_AUDIT_COLUMNS = (
    "solver",
    "file",
    "origin",
    "line_count",
    "top_level_classes",
    "responsibility_roles",
    "wildcard_imports",
    "mutable_defaults",
    "empty_pass_bodies",
    "broad_exception_without_reraise",
    "duplicate_method_definitions",
    "nonconforming_declarations",
    "new_mechanical_violations",
    "status",
    "notes",
)

RESPONSIBILITY_SUFFIXES = (
    "Assembler",
    "Engine",
    "Manager",
    "Operator",
    "Projector",
    "Recorder",
    "Solver",
    "State",
)

REVIEWED_MIXED_ROLE_MODULES = {
    "src/dem/structs/BaseStruct.py": "Taichi structure catalog; classes are field layouts without lifecycle ownership",
    "src/mpm/structs/GridNode.py": "Taichi grid-node layout variants; no manager/engine lifecycle is mixed in",
}

# The pre-June solver vocabulary intentionally contains UpperCamel public
# recorder/generator verbs and mathematical symbols. New names may extend
# those exact families, but arbitrary UpperCamel implementation methods remain
# violations.
HISTORICAL_UPPERCASE_FUNCTION_PREFIXES = (
    "Generate",
    "Lattice",
    "LSparticle",
    "Mean",
    "Monitor",
    "Visualize",
)
REVIEWED_MATHEMATICAL_OR_FACADE_FUNCTIONS = {
    "D2distance_div_Dpoint2",
    "Ddistance_div_Dpoint",
    "MPM",
    "PSD",
    "Psi",
    "RodriguesRotationMatrix",
}


@dataclass(frozen=True)
class FunctionRecord:
    solver: str
    file: str
    module: str
    class_name: str
    function: str
    qualified_name: str
    kind: str
    role: str
    line: int
    parameters: tuple[str, ...]
    calls: tuple[str, ...]
    guards: tuple[str, ...]
    self_writes: tuple[str, ...]
    device_behavior: tuple[str, ...]
    frequency: str
    classification: str
    regression: str


class FunctionBodyScanner(ast.NodeVisitor):
    """Inspect one function body without leaking nested scopes into its row."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: set[str] = set()
        self.guards: list[str] = []
        self.self_writes: set[str] = set()
        self.device: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        if name:
            self.calls.add(name)
            if name.endswith(("to_numpy", "to_torch", "to_ndarray")):
                self.device.add("device_to_host")
            if name.endswith(("from_numpy", "from_torch", "from_ndarray")):
                self.device.add("host_to_device")
            if name.startswith("ti.") and name.split(".")[-1].endswith("field"):
                self.device.add("field_allocation")
            if name.endswith("fill"):
                self.device.add("field_fill")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.guards.append(f"L{node.lineno}:if {compact_source(self.source, node.test)} -> {first_consequence(node)}")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.guards.append(f"L{node.lineno}:assert {compact_source(self.source, node.test)}")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.guards.append(f"L{node.lineno}:try handlers={len(node.handlers)}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_targets((node.target,))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_targets((node.target,))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def _record_targets(self, targets: Iterable[ast.AST]) -> None:
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                self.self_writes.add(target.attr)


def run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def compact_source(source: str, node: ast.AST) -> str:
    text = ast.get_source_segment(source, node) or ast.dump(node, include_attributes=False)
    return " ".join(text.split())


def first_consequence(node: ast.If) -> str:
    for child in node.body:
        if isinstance(child, ast.Raise):
            return "raise"
        if isinstance(child, ast.Return):
            return "return"
        if isinstance(child, ast.Continue):
            return "continue"
        if isinstance(child, ast.Break):
            return "break"
    return "branch"


def infer_role(path: Path, function: str) -> str:
    lowered = function.lower()
    parts = {part.lower() for part in path.parts}
    filename = path.name.lower()
    if "checkpoint" in filename or "restart" in lowered:
        return "restart"
    if (
        "recorder" in filename
        or "postplot" in filename
        or any(token in lowered for token in ("write", "save", "output", "record"))
    ):
        return "output"
    if filename.startswith("main"):
        return "public_facade"
    if "simulation.py" == filename or lowered.startswith(("set_", "add_", "configure", "initialize", "activate")):
        return "setup"
    if filename.endswith("base.py") and lowered in {"run", "solve", "simulation", "visualize"}:
        return "loop"
    if "generator" in parts or "generator" in filename:
        return "generation"
    if "kernel" in filename:
        return "device"
    if "engines" in parts or filename == "engine.py":
        return "engine"
    if "contact" in parts or "neighbor" in parts or "contact" in filename or "neighbor" in filename:
        return "contact_search"
    return "state_or_helper"


def infer_frequency(path: Path, function: str, role: str) -> str:
    lowered = function.lower()
    if role in {"setup", "generation", "public_facade"}:
        return "construction"
    if role == "restart":
        return "restart_or_checkpoint"
    if role == "output":
        return "output_interval"
    if any(
        token in lowered for token in ("substep", "compute", "integration", "resolve", "p2g", "g2p", "force", "advance")
    ):
        return "every_step_candidate"
    if any(token in lowered for token in ("rebuild", "verlet", "broad", "cull")):
        return "neighbor_rebuild_interval"
    if any(token in lowered for token in ("diagnostic", "energy", "jacobian", "residual")):
        return "diagnostic_interval_candidate"
    if role == "device":
        return "caller_defined_device_interval"
    return "caller_defined"


def infer_classification(
    path: Path,
    function: str,
    role: str,
    frequency: str,
    guards: Iterable[str],
    calls: Iterable[str],
) -> str:
    text = " ".join((*guards, *calls, function)).lower()
    correctness_tokens = (
        "overflow",
        "capacity",
        "finite",
        "nan",
        "jacobian",
        "penetration",
        "topology",
        "fingerprint",
        "schema",
        "dtype",
        "shape",
        "line_search",
        "barrier",
        "ccd",
        "watertight",
        "orientation",
        "volume",
    )
    if any(token in text for token in correctness_tokens):
        return "retain_or_schedule_correctness"
    if role in {"public_facade", "setup", "restart", "generation"}:
        return "retain_at_boundary_or_hoist"
    hot = frequency == "every_step_candidate"
    if hot and ("getattr" in text or " is none" in text or ' == "' in text or " == '" in text):
        return "bind_or_specialize_candidate"
    if hot and guards:
        return "hot_guard_manual_review"
    return "retain_pending_function_proof"


def infer_regression(role: str, frequency: str, text: str) -> str:
    lowered = text.lower()
    tests = []
    if role in {"public_facade", "setup"}:
        tests.append("construction_contract")
    if role == "restart" or "history" in lowered:
        tests.append("restart_equivalence")
    if "overflow" in lowered or "capacity" in lowered:
        tests.append("capacity_stress")
    if "contact" in lowered or "neighbor" in lowered or "cull" in lowered:
        tests.append("contact_continuity")
    if "implicit" in lowered or "ccd" in lowered or "barrier" in lowered:
        tests.append("implicit_path_equivalence")
    if frequency == "every_step_candidate":
        tests.append("numerical_and_hot_path")
    if role == "output":
        tests.append("output_schedule")
    return ",".join(dict.fromkeys(tests)) or "targeted_unit_regression"


def function_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = {dotted_name(decorator) for decorator in node.decorator_list}
    if any(name.endswith("ti.kernel") or name == "ti.kernel" for name in decorators):
        return "taichi_kernel"
    if any(name.endswith("ti.func") or name == "ti.func" for name in decorators):
        return "taichi_func"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_python"
    return "python"


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, solver: str, path: Path, source: str) -> None:
        self.solver = solver
        self.path = path
        self.source = source
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.records: list[FunctionRecord] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        declared_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg is not None:
            declared_args.append(node.args.vararg)
        if node.args.kwarg is not None:
            declared_args.append(node.args.kwarg)
        parameters = tuple(arg.arg for arg in declared_args if arg.arg not in {"self", "cls"})
        scanner = FunctionBodyScanner(self.source)
        for statement in node.body:
            scanner.visit(statement)
        calls = scanner.calls
        guards = scanner.guards
        self_writes = scanner.self_writes
        device = scanner.device
        kind = function_kind(node)
        if kind == "taichi_kernel":
            device.add("kernel_launch_when_called")
        elif kind == "taichi_func":
            device.add("inlined_device_helper")
        role = infer_role(self.path, node.name)
        frequency = infer_frequency(self.path, node.name, role)
        classification = infer_classification(self.path, node.name, role, frequency, guards, calls)
        text = " ".join((*calls, *guards, node.name, self.path.as_posix()))
        class_name = ".".join(self.class_stack)
        nested = ".".join((*self.function_stack, node.name))
        module = self.path.with_suffix("").as_posix().replace("/", ".")
        qualified = ".".join(part for part in (module, class_name, nested) if part)
        self.records.append(
            FunctionRecord(
                solver=self.solver,
                file=self.path.as_posix(),
                module=module,
                class_name=class_name,
                function=nested,
                qualified_name=qualified,
                kind=kind,
                role=role,
                line=node.lineno,
                parameters=parameters,
                calls=tuple(sorted(calls)),
                guards=tuple(guards),
                self_writes=tuple(sorted(self_writes)),
                device_behavior=tuple(sorted(device)),
                frequency=frequency,
                classification=classification,
                regression=infer_regression(role, frequency, text),
            )
        )
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()


def solver_files(repo: Path) -> list[tuple[str, Path]]:
    files = []
    for root_name in SOLVER_ROOTS:
        root = repo / root_name
        for path in sorted(root.rglob("*.py")):
            if path.name.startswith("._"):
                continue
            files.append((Path(root_name).name, path.relative_to(repo)))
    return files


def _function_has_accessor_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Attribute) and decorator.attr in {"setter", "deleter"}
        for decorator in node.decorator_list
    )


def mechanical_style_issues(
    source: str,
    filename: str = "<string>",
    allowed_historical_declarations: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, list[object]]:
    """Return objective style hazards without treating historical taste as law."""
    tree = ast.parse(source, filename=filename)
    issues: dict[str, list[object]] = {
        "wildcard_imports": [],
        "mutable_defaults": [],
        "empty_pass_bodies": [],
        "broad_exception_without_reraise": [],
        "duplicate_method_definitions": [],
        "nonconforming_declarations": [],
    }
    module_name = Path(filename).stem
    if filename != "<string>" and not (
        re.fullmatch(r"__init__", module_name)
        or re.fullmatch(r"_?[A-Z][A-Za-z0-9]*", module_name)
        or re.fullmatch(r"_?[a-z][a-z0-9_]*", module_name)
        or re.fullmatch(r"main[A-Z][A-Za-z0-9]*", module_name)
        or ("module", module_name) in allowed_historical_declarations
    ):
        issues["nonconforming_declarations"].append(f"module:{module_name}:L1")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            issues["wildcard_imports"].append(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not (
                re.fullmatch(r"__[a-z][a-z0-9_]*__", node.name)
                or re.fullmatch(r"_?[a-z][A-Za-z0-9_]*", node.name)
                or ("function", node.name) in allowed_historical_declarations
                or node.name.startswith(HISTORICAL_UPPERCASE_FUNCTION_PREFIXES)
                or node.name in REVIEWED_MATHEMATICAL_OR_FACADE_FUNCTIONS
            ):
                issues["nonconforming_declarations"].append(f"function:{node.name}:L{node.lineno}")
            defaults = [*node.args.defaults, *(default for default in node.args.kw_defaults if default is not None)]
            for default in defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues["mutable_defaults"].append(f"{node.name}:L{default.lineno}")
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                issues["empty_pass_bodies"].append(f"{node.name}:L{node.lineno}")
        elif (
            isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id == "BaseException"
        ):
            if not any(isinstance(child, ast.Raise) and child.exc is None for child in ast.walk(node)):
                issues["broad_exception_without_reraise"].append(node.lineno)
        elif isinstance(node, ast.ClassDef):
            if not (
                re.fullmatch(r"_?[A-Z][A-Za-z0-9]*", node.name)
                or ("class", node.name) in allowed_historical_declarations
            ):
                issues["nonconforming_declarations"].append(f"class:{node.name}:L{node.lineno}")
            methods: dict[str, int] = {}
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if child.name in methods and not _function_has_accessor_decorator(child):
                    issues["duplicate_method_definitions"].append(
                        f"{node.name}.{child.name}:L{methods[child.name]}/L{child.lineno}"
                    )
                methods[child.name] = child.lineno
    return issues


def _issue_lines(issue: object) -> tuple[int, ...]:
    if isinstance(issue, int):
        return (issue,)
    return tuple(int(line) for line in re.findall(r"L(\d+)", str(issue)))


def _canonical_issue(issue: object) -> str:
    if isinstance(issue, int):
        return "line_only_issue"
    return re.sub(r"L\d+", "L#", str(issue))


def _added_current_lines(repo: Path, historical_ref: str, relative: Path, line_count: int) -> set[int]:
    """Return current-worktree lines introduced after the historical reference."""
    try:
        run_git(repo, "show", f"{historical_ref}:{relative.as_posix()}")
    except subprocess.CalledProcessError:
        return set(range(1, line_count + 1))

    diff = run_git(
        repo,
        "diff",
        "--unified=0",
        "--no-ext-diff",
        historical_ref,
        "--",
        relative.as_posix(),
    )
    added: set[int] = set()
    for match in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE):
        start = int(match.group(1))
        count = int(match.group(2) or 1)
        added.update(range(start, start + count))
    return added


def _historical_declarations(repo: Path, historical_ref: str) -> frozenset[tuple[str, str]]:
    paths = run_git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        historical_ref,
        "--",
        "src/mpm",
        "src/dem",
        "src/mpdem",
    )
    declarations: set[tuple[str, str]] = set()
    for path_text in paths.splitlines():
        if not path_text.endswith(".py") or Path(path_text).name.startswith("._"):
            continue
        declarations.add(("module", Path(path_text).stem))
        tree = ast.parse(run_git(repo, "show", f"{historical_ref}:{path_text}"), filename=path_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                declarations.add(("class", node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declarations.add(("function", node.name))
    return frozenset(declarations)


def _responsibility_role(node: ast.ClassDef) -> str:
    class_name = node.name
    base_names = {dotted_name(base) for base in node.bases}
    if (
        class_name.endswith(("Adapter", "Entry", "Error", "Handle", "Property", "Result", "Table"))
        or class_name.startswith("_")
        or any(name.endswith(("Error", "Exception")) for name in base_names)
    ):
        return "helper"
    for suffix in RESPONSIBILITY_SUFFIXES:
        if class_name.endswith(suffix) or class_name.endswith(suffix + "Mixin"):
            return suffix.lower()
    if "Contact" in class_name:
        return "contact"
    if "Boundary" in class_name:
        return "boundary"
    return "domain"


def style_audit_rows(repo: Path, modules: list[dict[str, object]], historical_ref: str) -> list[dict[str, object]]:
    rows = []
    issue_names = STYLE_AUDIT_COLUMNS[6:12]
    historical_declarations = _historical_declarations(repo, historical_ref)
    for module in modules:
        relative = Path(str(module["file"]))
        source = (repo / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative.as_posix())
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        current_issues = mechanical_style_issues(
            source,
            relative.as_posix(),
            historical_declarations,
        )
        try:
            historical_source = run_git(repo, "show", f"{historical_ref}:{relative.as_posix()}")
        except subprocess.CalledProcessError:
            origin = "post_june_module"
            historical_issues = {name: [] for name in issue_names}
        else:
            origin = "historical_module"
            historical_issues = mechanical_style_issues(
                historical_source,
                f"{historical_ref}:{relative.as_posix()}",
                historical_declarations,
            )
        added_lines = _added_current_lines(repo, historical_ref, relative, len(source.splitlines()))
        new_violations = 0
        for name in issue_names:
            historical_counts = Counter(_canonical_issue(issue) for issue in historical_issues[name])
            for issue in current_issues[name]:
                canonical = _canonical_issue(issue)
                if historical_counts[canonical] > 0:
                    historical_counts[canonical] -= 1
                elif any(line in added_lines for line in _issue_lines(issue)):
                    new_violations += 1
        class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        roles = sorted({_responsibility_role(node) for node in class_nodes} - {"helper"})
        mixed_roles = len(roles) > 1
        reviewed_mixed_role = REVIEWED_MIXED_ROLE_MODULES.get(relative.as_posix())
        notes = []
        if mixed_roles and not reviewed_mixed_role:
            notes.append("manual mixed-responsibility review required")
        elif reviewed_mixed_role:
            notes.append(reviewed_mixed_role)
        if len(source.splitlines()) >= 1200:
            notes.append("large module reviewed by mathematical/lifecycle responsibility, not size alone")
        rows.append(
            {
                "solver": module["solver"],
                "file": relative.as_posix(),
                "origin": origin,
                "line_count": len(source.splitlines()),
                "top_level_classes": ",".join(classes),
                "responsibility_roles": ",".join(roles) or "module_functions",
                **{name: " | ".join(map(str, current_issues[name])) for name in issue_names},
                "new_mechanical_violations": new_violations,
                "status": "violation" if new_violations or (mixed_roles and not reviewed_mixed_role) else "aligned",
                "notes": "; ".join(notes),
            }
        )
    return rows


def collect_current(repo: Path) -> tuple[list[FunctionRecord], list[dict[str, object]]]:
    records: list[FunctionRecord] = []
    modules: list[dict[str, object]] = []
    for solver, relative in solver_files(repo):
        source = (repo / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative.as_posix())
        collector = FunctionCollector(solver, relative, source)
        collector.visit(tree)
        classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        modules.append(
            {
                "solver": solver,
                "file": relative.as_posix(),
                "classes": classes,
                "functions": len(collector.records),
                "taichi_kernels": sum(record.kind == "taichi_kernel" for record in collector.records),
                "taichi_funcs": sum(record.kind == "taichi_func" for record in collector.records),
            }
        )
        records.extend(collector.records)
    return records, modules


def collect_historical_inventory(repo: Path, ref: str) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    paths = run_git(repo, "ls-tree", "-r", "--name-only", ref, "--", "src/mpm", "src/dem", "src/mpdem")
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths.splitlines():
        if path.endswith(".py") and not Path(path).name.startswith("._"):
            grouped[Path(path).parts[1]].append(path)
    for solver, files in sorted(grouped.items()):
        counts = Counter(files=len(files), classes=0, functions=0, taichi_kernels=0, taichi_funcs=0)
        for path in files:
            source = run_git(repo, "show", f"{ref}:{path}")
            tree = ast.parse(source, filename=f"{ref}:{path}")
            counts["classes"] += sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    counts["functions"] += 1
                    kind = function_kind(node)
                    counts["taichi_kernels"] += kind == "taichi_kernel"
                    counts["taichi_funcs"] += kind == "taichi_func"
        inventory[solver] = dict(counts)
    return inventory


def caller_map(records: list[FunctionRecord]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    by_name: dict[str, list[FunctionRecord]] = defaultdict(list)
    by_module_name: dict[tuple[str, str], list[FunctionRecord]] = defaultdict(list)
    for record in records:
        simple = record.function.split(".")[-1]
        by_name[simple].append(record)
        by_module_name[(record.module, simple)].append(record)

    callers: dict[str, set[str]] = defaultdict(set)
    resolution: dict[str, set[str]] = defaultdict(set)
    for caller in records:
        for call in caller.calls:
            simple = call.split(".")[-1]
            local = by_module_name.get((caller.module, simple), [])
            candidates = local or by_name.get(simple, [])
            if not candidates:
                continue
            label = "module_exact" if local else ("global_unique" if len(candidates) == 1 else "name_overapproximation")
            for candidate in candidates:
                callers[candidate.qualified_name].add(caller.qualified_name)
                resolution[candidate.qualified_name].add(label)
    return callers, resolution


def configured_path(record: FunctionRecord) -> str:
    if record.role == "public_facade":
        return "public configure->allocate->generate->bind->run"
    if record.role == "loop":
        return "configured Base loop"
    if record.role == "engine":
        return "bound engine path; verify choose_engine/manage_function"
    if record.role == "contact_search":
        return "configured contact/search implementation"
    if record.role == "restart":
        return "checkpoint save/load boundary"
    if record.role == "output":
        return "scheduled recorder path"
    return "direct callers listed; dynamic reachability requires invariant test"


def ledger_rows(records: list[FunctionRecord]) -> list[dict[str, object]]:
    callers, resolution = caller_map(records)
    rows = []
    for record in records:
        direct_callers = sorted(callers.get(record.qualified_name, set()))
        modes = sorted(resolution.get(record.qualified_name, set()))
        preconditions = (
            f"parameters={','.join(record.parameters) or '-'}; "
            f"caller_count={len(direct_callers)}; setup contract must establish required fields/capacities"
        )
        mutable = [*record.parameters]
        if record.self_writes:
            mutable.append("self_writes=" + ",".join(record.self_writes))
        proof = (
            "static direct-caller overapproximation recorded; "
            + (
                "ambiguous dynamic/method dispatch requires runtime invariant test"
                if "name_overapproximation" in modes
                else "no ambiguous name match observed"
            )
            + "; no guard deletion authorized by this generated row"
        )
        rows.append(
            {
                "solver": record.solver,
                "file": record.file,
                "class": record.class_name,
                "function": record.function,
                "kind": record.kind,
                "role": record.role,
                "line": record.line,
                "callers": " | ".join(direct_callers),
                "caller_resolution": ",".join(modes) or "no_static_internal_caller",
                "configured_paths": configured_path(record),
                "preconditions": preconditions,
                "mutable_inputs": ",".join(mutable) or "none",
                "device_behavior": ",".join(record.device_behavior) or "none_detected",
                "guards": " | ".join(record.guards),
                "frequency": record.frequency,
                "classification": record.classification,
                "proof": proof,
                "regression": record.regression,
            }
        )
    return rows


def write_reports(
    repo: Path,
    output: Path,
    records: list[FunctionRecord],
    modules: list[dict[str, object]],
    historical: dict[str, dict[str, int]],
    historical_ref: str,
    revision: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "guard_decisions.csv"
    existing_decisions = []
    if decisions_path.exists():
        with decisions_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == list(GUARD_DECISION_COLUMNS):
                existing_decisions = list(reader)

    rows = ledger_rows(records)
    current_lines = {(record.file, record.class_name, record.function): str(record.line) for record in records}
    for decision in existing_decisions:
        key = (decision["file"], decision["class"], decision["function"])
        if key in current_lines:
            decision["guard_line"] = current_lines[key]
    with (output / "function_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    style_rows = style_audit_rows(repo, modules, historical_ref)
    with (output / "style_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=STYLE_AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(style_rows)

    current_summary: dict[str, Counter[str]] = defaultdict(Counter)
    for module in modules:
        counter = current_summary[str(module["solver"])]
        counter["files"] += 1
        counter["classes"] += int(module["classes"])
        counter["functions"] += int(module["functions"])
        counter["taichi_kernels"] += int(module["taichi_kernels"])
        counter["taichi_funcs"] += int(module["taichi_funcs"])
    inventory = {
        "schema_version": 1,
        "git_revision": revision,
        "solver_roots": SOLVER_ROOTS,
        "current": {solver: dict(counts) for solver, counts in sorted(current_summary.items())},
        "historical_reference": historical_ref,
        "historical": historical,
        "modules": modules,
    }
    (output / "module_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    classifications = Counter(record.classification for record in records)
    roles = Counter(record.role for record in records)
    guarded = sum(bool(record.guards) for record in records)
    style_violations = sum(row["status"] == "violation" for row in style_rows)
    candidate_keys = {
        (record.file, record.class_name, record.function)
        for record in records
        if record.classification == "bind_or_specialize_candidate"
    }
    decision_keys = {
        (row["file"], row["class"], row["function"]) for row in existing_decisions if row["final_decision"]
    }
    reviewed_candidates = len(candidate_keys & decision_keys)
    lines = [
        "# Post-paper generated solver audit summary",
        "",
        f"- Git revision: `{revision}`",
        f"- Solver Python files: {len(modules)}",
        f"- Function/method/kernel rows: {len(records)}",
        f"- Rows containing at least one guard: {guarded}",
        f"- Modules with unresolved new mechanical/mixed-role violations: {style_violations}",
        f"- One-time-binding candidates with explicit decisions: {reviewed_candidates}/{len(candidate_keys)}",
        "- Generated classifications are conservative nominations; none authorizes deletion.",
        "",
        "## Current inventory",
        "",
        "| Solver | Files | Classes | Functions | ti.kernel | ti.func |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for solver, counts in sorted(current_summary.items()):
        lines.append(
            f"| {solver} | {counts['files']} | {counts['classes']} | {counts['functions']} | "
            f"{counts['taichi_kernels']} | {counts['taichi_funcs']} |"
        )
    lines.extend(("", "## Role counts", ""))
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(roles.items()))
    lines.extend(("", "## Classification counts", ""))
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(classifications.items()))
    lines.extend(
        (
            "",
            "## Required manual continuation",
            "",
            "For each source edit, copy the affected generated row into `guard_decisions.csv`, "
            "replace the nomination with a final decision, cite the behavior-invariant test and "
            "configured caller proof, and name the regression actually run.",
        )
    )
    (output / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with decisions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GUARD_DECISION_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_decisions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/llm_assist/postpaper/solver_audit"),
    )
    parser.add_argument("--historical-ref", default=HISTORICAL_REF)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    records, modules = collect_current(repo)
    historical = collect_historical_inventory(repo, args.historical_ref)
    revision = run_git(repo, "rev-parse", "HEAD").strip()
    write_reports(repo, output, records, modules, historical, args.historical_ref, revision)
    print(f"wrote {len(records)} ledger rows for {len(modules)} files to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
