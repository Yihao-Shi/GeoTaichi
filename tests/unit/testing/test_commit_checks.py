"""Commit gates select affected law suites and test the staged content."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.testing.run_commit_checks import affected_law_partitions

pytestmark = [pytest.mark.required, pytest.mark.cpu]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def git(repo, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True)


@pytest.fixture
def commit_repo(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for relative in (".githooks/commit-msg", "tests/testing/run_commit_checks.py"):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, target)
    (repo / "tests/testing/run_partition.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "assert os.environ['GEOTAICHI_REAL_DTYPE'] == 'float64'\n"
        "assert Path('example.py').read_text() == 'value = 1\\n'\n"
        "assert not Path('untracked.py').exists()\n"
        "assert Path('tests/data/fixture.dat').read_bytes() == b'staged fixture'\n"
        "print('CHECKED ' + sys.argv[1], flush=True)\n"
        "failure = json.loads(Path('failure.json').read_text())\n"
        "raise SystemExit(7 if failure == sys.argv[1] else 0)\n",
        encoding="utf-8",
    )
    (repo / "example.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "failure.json").write_text('""', encoding="utf-8")
    stage_change(repo, "tests/data/fixture.dat", "staged fixture")
    stage_change(repo, "src/physics_model/consititutive_model/Old.py")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Commit gate test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "baseline",
    )
    (repo / "example.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "untracked.py").write_text("value = 3\n", encoding="utf-8")
    return repo


def stage_change(repo, relative, content="value = 4\n"):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    git(repo, "add", "--", relative)


def run_hook(repo, message):
    message_file = repo / ".git/COMMIT_EDITMSG"
    message_file.write_text(message, encoding="utf-8")
    return subprocess.run(
        ["bash", ".githooks/commit-msg", str(message_file)],
        cwd=repo,
        env={**os.environ, "GEOTAICHI_PYTHON": sys.executable},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("message", "extended"),
    [("Small fix\n", False), ("Fix contact [full test]\n", True), ("Small fix\n# [full test]\n", False)],
)
def test_message_selects_basic_or_extended_checks_on_staged_content(commit_repo, message, extended):
    result = run_hook(commit_repo, message)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKED pre-commit" in result.stdout
    assert ("CHECKED commit-validation" in result.stdout) is extended
    assert git(commit_repo, "show", ":example.py") == "value = 1\n"
    assert (commit_repo / "example.py").read_text() == "value = 2\n"
    assert (commit_repo / "untracked.py").exists()


@pytest.mark.parametrize("partition", ["pre-commit", "commit-validation"])
def test_failed_checks_reject_the_commit(commit_repo, partition):
    (commit_repo / "failure.json").write_text(f'"{partition}"', encoding="utf-8")
    git(commit_repo, "add", "failure.json")

    result = run_hook(commit_repo, "Fix contact [full test]\n")

    assert result.returncode == 7, result.stdout + result.stderr
    assert f"CHECKED {partition}" in result.stdout
    if partition == "pre-commit":
        assert "CHECKED commit-validation" not in result.stdout


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["README.md", "src/mpm/README.md", "tools/render.py"], ()),
        (["src/physics_model/consititutive_model/finite_strain/NeoHookean.py"], ("commit-materials",)),
        (["tests/helpers/material_cases.py"], ("commit-materials",)),
        (["src/fem/MaterialManager.py"], ("commit-materials", "commit-contact")),
        (["src/mpm/MaterialManager.py"], ("commit-materials", "commit-contact")),
        (["tests/unit/physics_model/materials/test_linear_elastic.py"], ("commit-materials",)),
        (["src/physics_model/contact_model/LinearModel.py"], ("commit-contact",)),
        (
            ["src/dem/contact/HertzMindlinModel.py", "tests/data/ipc_toolkit/friction/cube_cube.tar.gz"],
            ("commit-contact",),
        ),
        (["tests/unit/mpm/test_grid_wall_friction.py"], ("commit-contact",)),
        (["src/utils/TypeDefination.py"], ("commit-materials", "commit-contact")),
        (["src/mpm/engines/direct/MPMSolver.py"], ("commit-materials", "commit-contact")),
        (["pyproject.toml", "requirements.txt", "tests/conftest.py"], ("commit-materials", "commit-contact")),
        (
            ["src/physics_model/consititutive_model/Old.py", "src/dem/contact/New.py"],
            ("commit-materials", "commit-contact"),
        ),
    ],
)
def test_law_selection_includes_direct_and_shared_dependencies(paths, expected):
    assert affected_law_partitions(paths) == expected


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("notes.md", ()),
        ("src/physics_model/consititutive_model/New.py", ("commit-materials",)),
        ("src/dem/contact/New.py", ("commit-contact",)),
        ("src/utils/shared.py", ("commit-materials", "commit-contact")),
    ],
)
def test_full_message_runs_only_affected_law_suites(commit_repo, relative, expected):
    stage_change(commit_repo, relative)
    result = run_hook(commit_repo, "Validate [full test]\n")
    assert result.returncode == 0, result.stdout + result.stderr
    for partition in ("commit-materials", "commit-contact"):
        assert (f"CHECKED {partition}" in result.stdout) == (partition in expected)


def test_normal_message_keeps_law_change_in_the_fast_gate(commit_repo):
    stage_change(commit_repo, "src/dem/contact/New.py")
    result = run_hook(commit_repo, "Small change\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKED commit-validation" not in result.stdout
    assert "CHECKED commit-contact" not in result.stdout


def test_validation_precision_does_not_inherit_simulation_settings(commit_repo, monkeypatch):
    monkeypatch.setenv("GEOTAICHI_REAL_DTYPE", "float32")
    result = run_hook(commit_repo, "Small change\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.environ["GEOTAICHI_REAL_DTYPE"] == "float32"


@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_removed_or_moved_material_still_triggers_material_checks(commit_repo, operation):
    original = "src/physics_model/consititutive_model/Old.py"
    if operation == "delete":
        git(commit_repo, "rm", "--", original)
    else:
        (commit_repo / "tools").mkdir()
        git(commit_repo, "mv", "--", original, "tools/moved.py")
    result = run_hook(commit_repo, "Validate [full test]\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKED commit-materials" in result.stdout
    assert "CHECKED commit-contact" not in result.stdout


@pytest.mark.parametrize("partition", ["commit-materials", "commit-contact"])
def test_failed_law_suite_blocks_the_commit(commit_repo, partition):
    stage_change(commit_repo, "src/utils/shared.py")
    stage_change(commit_repo, "failure.json", f'"{partition}"')
    result = run_hook(commit_repo, "Validate [full test]\n")
    assert result.returncode == 7, result.stdout + result.stderr
    assert f"CHECKED {partition}" in result.stdout
