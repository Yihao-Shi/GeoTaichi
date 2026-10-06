import ast
import re
import subprocess
import sys
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


def test_offline_postprocessing_lives_in_draw():
    names = {
        "evaluate",
        "evaluate_metrics",
        "write_metrics",
        "postprocess_consolidation",
        "postprocess_landslide",
        "plot_terzaghi_profiles",
        "average_pressure_profile",
        "_pressure_profile",
        "validate_laplace",
    }
    violations = []
    for path in _python_examples():
        if "draw" in path.relative_to(EXAMPLES_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.name}")
    assert not violations, "Move offline evaluation into the case's draw/ directory:\n" + "\n".join(violations)


def test_evaluation_modules_load_without_simulation_runtime(tmp_path):
    paths = sorted(
        str(path)
        for path in EXAMPLES_ROOT.rglob("evaluate*.py")
        if "draw" in path.relative_to(EXAMPLES_ROOT).parts and not path.name.startswith("._")
    )
    assert paths
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys\n"
                "for path in sys.argv[1:]:\n"
                "    runpy.run_path(path, run_name='offline_example_check')\n"
                "assert 'taichi' not in sys.modules, 'Offline evaluation imported the simulation runtime'\n"
            ),
            *paths,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
