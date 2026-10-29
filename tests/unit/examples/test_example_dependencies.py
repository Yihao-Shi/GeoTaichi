"""Independent cases may import their own parameters and postprocessing."""

import ast
from pathlib import Path

import pytest


@pytest.mark.filterwarnings("ignore:invalid escape sequence:DeprecationWarning")
def test_examples_do_not_import_other_cases():
    root = Path(__file__).resolve().parents[3]
    violations = []
    for path in sorted((root / "examples").rglob("*.py")):
        if "OutputData" in path.parts or path.name.startswith("._"):
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        case = Path(*parts[: parts.index("draw")]) if "draw" in parts else relative.parent
        for node in ast.walk(ast.parse(path.read_text(), filename=str(relative))):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                if module.startswith("examples.") and not Path(*module.split(".")).is_relative_to(case):
                    violations.append(f"{relative}:{node.lineno}: {module}")
    assert not violations, "Imports from other cases:\n" + "\n".join(violations)
