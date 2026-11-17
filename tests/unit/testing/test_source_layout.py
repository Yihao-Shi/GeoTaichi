"""Keep runtime modules free of validation entry points and stale APIs."""

import ast
import re
import symtable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
DEPRECATED_PATTERNS = (
    r"\bnp\.(?:float|int|bool|complex|object|str|unicode|long)\b",
    r"\bnumpy\.(?:float|int|bool|complex|object|str|unicode|long)\b",
    r"\bcollections\.(?:Mapping|MutableMapping|Sequence|Iterable)\b",
    r"\b(?:time\.clock|inspect\.getargspec)\b",
    r"\bti\.(?:ext_arr|block_dim|classkernel|async_mode|static_print)\b",
    r"(?<!ti)\.atomic_(?:add|sub|max|min|and|or|xor)\s*\(",
)
SDF_FACTORY_MODULES = {
    "SDFs2D.py",
    "SDFs3D.py",
    "arbitrarily.py",
    "pointcloud.py",
    "polyhedron.py",
    "polysuperellipsoid.py",
    "polysuperquadrics.py",
    "surfacefunction.py",
    "text.py",
}


@pytest.fixture(scope="module", autouse=True)
def _isolated_taichi_test_module():
    """This structural test neither imports nor initializes Taichi."""
    yield


def _trees():
    for path in sorted(SRC.rglob("*.py")):
        if path.name.startswith("._"):
            continue
        source = path.read_text(encoding="utf-8")
        yield path, ast.parse(source)


def _is_ti_decorator(node, name):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ti"
        and node.attr == name
    )


def _is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    names = {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
    strings = {
        item.value for item in ast.walk(node.test) if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    return "__name__" in names and "__main__" in strings


def test_src_has_no_executable_validation_entrypoints():
    offenders = []
    for path, tree in _trees():
        for node in tree.body:
            if _is_main_guard(node):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_modules_do_not_define_top_level_kernels_alongside_classes():
    offenders = []
    for path, tree in _trees():
        if not any(isinstance(node, ast.ClassDef) for node in tree.body):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and any(
                _is_ti_decorator(decorator, "kernel") for decorator in node.decorator_list
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert offenders == []


def test_src_has_no_known_removed_python_or_taichi_apis():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name.startswith("._"):
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in DEPRECATED_PATTERNS:
            if re.search(pattern, source):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_linalg_contains_only_host_functions():
    tree = ast.parse((SRC / "utils/linalg.py").read_text())
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "taichi" for alias in node.names)
        or isinstance(node, ast.ImportFrom)
        and node.module == "taichi"
        for node in ast.walk(tree)
    )


def test_device_functions_have_no_python_imports_or_staticmethod_wrapping():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not any(
                _is_ti_decorator(decorator, name) for decorator in node.decorator_list for name in ("func", "kernel")
            ):
                continue
            if any(isinstance(item, (ast.Import, ast.ImportFrom)) for item in ast.walk(node)) or any(
                isinstance(decorator, ast.Name) and decorator.id == "staticmethod" for decorator in node.decorator_list
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert offenders == []


@pytest.mark.required
def test_src_file_and_package_names_follow_project_conventions():
    offenders = []
    for path, _ in _trees():
        relative = path.relative_to(SRC)
        if not re.fullmatch(r"(?:[A-Z][A-Za-z0-9]*|[a-z][a-z0-9]*(?:_[a-z0-9]+)*|main[A-Z]+|__init__)", path.stem):
            offenders.append(str(relative))
        for package in relative.parts[:-1]:
            if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", package):
                offenders.append(str(relative))
    assert offenders == []


@pytest.mark.required
def test_src_classes_use_pascal_case_except_public_sdf_factories():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if re.fullmatch(r"[A-Z][A-Za-z0-9]*", node.name):
                continue
            if (
                path.parent == SRC / "sdf"
                and path.name in SDF_FACTORY_MODULES
                and re.fullmatch(r"[a-z][a-z0-9_]*", node.name)
            ):
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert offenders == []


@pytest.mark.required
def test_src_ordinary_locals_use_snake_case_without_changing_arguments():
    offenders = []
    for path, _ in _trees():
        pending = [symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")]
        while pending:
            table = pending.pop()
            pending.extend(table.get_children())
            if table.get_type() != "function":
                continue
            for symbol in table.get_symbols():
                name = symbol.get_name()
                if not (symbol.is_local() and symbol.is_assigned()) or (
                    symbol.is_parameter() or symbol.is_imported() or symbol.is_namespace()
                ):
                    continue
                if len(name) <= 3 or re.match(r"d\d*[A-Z]|dd[A-Z]", name):
                    continue
                if re.search(r"[a-z][A-Z]|^[A-Z][a-z]", name):
                    offenders.append(f"{path.relative_to(ROOT)}:{table.get_lineno()}:{name}")
    assert offenders == []


@pytest.mark.required
def test_src_absolute_imports_resolve_to_existing_modules():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                if not module.startswith("src."):
                    continue
                target = ROOT.joinpath(*module.split("."))
                if not target.is_dir() and not target.with_suffix(".py").is_file():
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}")
    assert offenders == []


@pytest.mark.required
def test_src_callable_names_follow_snake_case_including_device_functions():
    constructors = {"DEM", "MPM", "IGA", "FEM", "DEMPM", "MPDEM", "FEDEM", "FEMPM", "IGAMPM"}
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            if node in tree.body and node.name in constructors and path.name == "__init__.py":
                continue
            if any(
                isinstance(decorator, ast.Name)
                and decorator.id == "property"
                or isinstance(decorator, ast.Attribute)
                and decorator.attr in ("setter", "deleter")
                for decorator in node.decorator_list
            ):
                continue
            if not re.fullmatch(r"_*[a-z][a-z0-9]*(?:_[a-z0-9]+)*", node.name):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert offenders == []


@pytest.mark.required
def test_src_device_arguments_use_snake_case_except_mathematical_notation():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not any(
                _is_ti_decorator(decorator, name)
                for decorator in node.decorator_list
                for name in ("func", "kernel", "pyfunc")
            ):
                continue
            for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                name = argument.arg
                if len(name) <= 3 or re.match(r"d\d*[A-Z]|dd[A-Z]", name):
                    continue
                if re.search(r"[a-z][A-Z]|^[A-Z][a-z]", name):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    assert offenders == []


@pytest.mark.required
def test_src_module_variables_and_constants_follow_naming_conventions():
    offenders = []
    for path, tree in _trees():
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if path.name == "GlobalVariable.py" and name in {"Taylor", "Affine"}:
                    continue
                if name.startswith("__") and name.endswith("__"):
                    continue
                if re.fullmatch(r"_*[A-Z][A-Z0-9_]*|_*[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name):
                    continue
                if re.fullmatch(r"[A-Z][A-Za-z0-9]*", name) and isinstance(node.value, (ast.Name, ast.Subscript)):
                    continue
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    assert offenders == []
