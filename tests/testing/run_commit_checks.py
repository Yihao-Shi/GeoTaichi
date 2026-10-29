#!/usr/bin/env python3
"""Check staged content; [full test] also validates affected constitutive laws."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def git_output(*arguments, **kwargs):
    return subprocess.check_output(["git", *arguments], cwd=REPOSITORY_ROOT, **kwargs)


def affected_law_partitions(changed_paths):
    affected = set()
    for relative in changed_paths:
        path = PurePosixPath(relative)
        if path.suffix.lower() in (".md", ".rst"):
            continue
        if relative.startswith("tests/data/ipc_toolkit/"):
            affected.add("commit-contact")
        elif relative.startswith(("src/physics_model/consititutive_model/", "tests/unit/physics_model/materials/")) or (
            relative.startswith("tests/") and "material" in relative.lower()
        ):
            affected.add("commit-materials")
        elif (
            relative.startswith(
                (
                    "src/physics_model/contact_model/",
                    "src/dem/contact/",
                    "src/fedem/contact/",
                    "src/fem/contact/",
                    "src/contact_detection/",
                )
            )
            or relative.startswith("tests/")
            and any(word in relative.lower() for word in ("contact", "friction", "barrier", "hertz", "rolling"))
        ):
            affected.add("commit-contact")
        elif relative.startswith(("src/", "geotaichi/", "tests/", "third_party/", ".githooks/")) or (
            relative in ("pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", ".gitignore", ".gitattributes")
            or path.name.startswith(("requirements", "environment", "geotaichi_env"))
            or path.name in ("uv.lock", "poetry.lock", "Pipfile", "Pipfile.lock")
        ):
            # Unknown shared code may affect either law family; prefer an extra run to a missed dependency.
            affected.update(("commit-materials", "commit-contact"))
    return tuple(name for name in ("commit-materials", "commit-contact") if name in affected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message_file", type=Path)
    options = parser.parse_args(argv)
    message = git_output("stripspace", "--strip-comments", input=options.message_file.read_bytes())
    full = b"[full test]" in message
    tree = git_output("write-tree")
    started = time.monotonic()
    partitions = ("pre-commit", "commit-validation") if full else ("pre-commit",)
    if full:
        # shortcut: this diff covers index vs HEAD; validate inherited changes explicitly when amending commits.
        changed = git_output("diff", "--cached", "--name-only", "--no-renames", "--diff-filter=ACDMRT", "-z")
        affected = affected_law_partitions(os.fsdecode(path) for path in changed.split(b"\0") if path)
        partitions += affected
        for name in ("commit-materials", "commit-contact"):
            if name not in affected:
                print(f"commit checks: skip {name} (no relevant staged change)", flush=True)
    print(f"commit checks: {', '.join(partitions)} ({sys.executable})", flush=True)

    with tempfile.TemporaryDirectory(prefix="geotaichi-commit-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        # Keep bundled test data alongside Python/config files so law oracles cannot depend on untracked fixtures.
        paths = git_output(
            "ls-files",
            "-z",
            "--",
            "*.py",
            "*.json",
            "*.toml",
            "tests/data",
            "tests/regression/.gitkeep",
            ".githooks/pre-commit",
            ".githooks/commit-msg",
        )
        checkout = ["git", "checkout-index", f"--prefix={source}{os.sep}", "-z", "--stdin"]
        subprocess.run(checkout, input=paths, cwd=REPOSITORY_ROOT, check=True)
        git_dir = os.fsdecode(git_output("rev-parse", "--absolute-git-dir")).strip()
        (source / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
        environment = {
            **os.environ,
            "GEOTAICHI_TEST_ARCH": "cpu",
            "GEOTAICHI_FORCE_CPU": "1",
            "GEOTAICHI_REAL_DTYPE": "float64",
            "GEOTAICHI_REPO_ROOT": str(source),
        }
        environment.pop("PYTHONPATH", None)
        if "GIT_INDEX_FILE" in environment:
            environment["GIT_INDEX_FILE"] = str(Path(environment["GIT_INDEX_FILE"]).resolve())
        for partition in partitions:
            command = [sys.executable, "tests/testing/run_partition.py", partition, "-q", "--taichi-arch", "cpu"]
            timeout = 60 if partition in ("pre-commit", "commit-validation") else None
            result = subprocess.run(command, cwd=source, env=environment, timeout=timeout)
            if result.returncode:
                return result.returncode

    if git_output("write-tree") != tree:
        print("commit checks: the index changed during testing; retry the commit", file=sys.stderr)
        return 1
    print(f"commit checks passed in {time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError) as error:
        print(f"commit checks failed: {error}", file=sys.stderr)
        raise SystemExit(1)
