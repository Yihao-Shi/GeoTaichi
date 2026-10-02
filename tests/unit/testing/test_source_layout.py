"""Keep runtime modules free of validation entry points and stale APIs."""

import ast
import re
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
