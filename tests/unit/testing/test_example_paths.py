import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
PERSONAL_UNIX_PREFIXES = ("/home/", "/media/", "/Users/", "/Volumes/")


def _string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _python_examples():
    return (path for path in EXAMPLES_ROOT.rglob("*.py") if not path.name.startswith("._"))


def test_examples_do_not_embed_developer_machine_paths():
    violations = []
    for path in _python_examples():
        for line, value in _string_literals(path):
            if value.startswith(PERSONAL_UNIX_PREFIXES) or WINDOWS_ABSOLUTE_PATH.match(value):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line}: {value!r}")

    assert not violations, "Developer-machine paths found:\n" + "\n".join(violations)


def test_examples_do_not_use_misspelled_assets_directory():
    violations = []
    for path in _python_examples():
        for line, value in _string_literals(path):
            if "/asserts/" in value or "\\asserts\\" in value:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line}: {value!r}")

    assert not violations, "Use assets/, not asserts/:\n" + "\n".join(violations)
