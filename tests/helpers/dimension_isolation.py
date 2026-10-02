"""Run dimension-specialized pytest nodes in a clean Python interpreter.

Several legacy Taichi kernels use module-level dimension constants in type
annotations.  Those annotations are evaluated when Python imports the module,
so resetting Taichi cannot turn a module imported in 2D into its 3D variant.
Tests marked with ``pytest.mark.isolated_dimension(3)`` therefore execute their
own node id in a child pytest process.  The child fixes every GeoTaichi
dimension source before pytest imports the selected test module.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as element_tree

import pytest

_CHILD_NODE_ENV = "GEOTAICHI_ISOLATED_PYTEST_NODE"
_CHILD_DIMENSION_ENV = "GEOTAICHI_ISOLATED_DIMENSION"


def _marker_dimension(marker) -> int:
    if marker.args and "dimension" in marker.kwargs:
        raise pytest.UsageError(
            "isolated_dimension accepts the dimension either positionally " "or by keyword, not both"
        )
    if len(marker.args) > 1:
        raise pytest.UsageError("isolated_dimension accepts exactly one dimension")
    value = marker.args[0] if marker.args else marker.kwargs.get("dimension")
    try:
        dimension = int(value)
    except (TypeError, ValueError) as error:
        raise pytest.UsageError("isolated_dimension requires dimension=2 or dimension=3") from error
    if dimension not in (2, 3) or dimension != value:
        raise pytest.UsageError("isolated_dimension requires dimension=2 or dimension=3")
    return dimension


def configure_isolated_dimension_from_environment() -> None:
    """Fix all dimension globals before pytest imports any test module.

    The ordinary parent process is the canonical 2D test process.  Tests that
    require 3D use ``isolated_dimension(3)`` and override that value through
    the child environment.  Establishing the 2D default here is essential:
    classic MPM kernels read ``GlobalVariable.DIMENSION`` in decorators at
    import time, so a later ``ti.reset()`` cannot repair a module first
    imported with GeoTaichi's production default of 3D.
    """

    raw_dimension = os.environ.get(_CHILD_DIMENSION_ENV)
    child_node = os.environ.get(_CHILD_NODE_ENV)
    if raw_dimension is None and child_node is None:
        dimension = 2
    else:
        if raw_dimension is None or child_node is None:
            raise RuntimeError(f"{_CHILD_NODE_ENV} and {_CHILD_DIMENSION_ENV} must be set together")

        try:
            dimension = int(raw_dimension)
        except ValueError as error:
            raise RuntimeError(f"{_CHILD_DIMENSION_ENV} must be 2 or 3, got {raw_dimension!r}") from error
        if dimension not in (2, 3):
            raise RuntimeError(f"{_CHILD_DIMENSION_ENV} must be 2 or 3, got {dimension!r}")

    # The classic MPM API reads GlobalVariable.DIMENSION in Taichi function
    # annotations, while the newer IGA, direct-MPM, and IGA-MPM paths use
    # their package-local config modules.  Set all of them before importing
    # any solver module.
    import src.utils.GlobalVariable as global_variable
    import src.iga.config as iga_config
    import src.igampm.config as igampm_config
    import src.mpm.config as mpm_config

    global_variable.DIMENSION = dimension
    iga_config.set_dimension(dimension)
    mpm_config.set_dimension(dimension)
    igampm_config.set_dimension(dimension)


def _pytest_option(config, option: str):
    try:
        return config.getoption(option)
    except (AttributeError, ValueError):
        return None


def _isolated_command(item, report_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        # Each child owns this existing unique temporary directory; the
        # repository's shared --basetemp would delete another child's files.
        f"--basetemp={report_path.parent / 'pytest'}",
        f"--junitxml={report_path}",
        item.nodeid,
    ]
    arch = _pytest_option(item.config, "--taichi-arch")
    fp = _pytest_option(item.config, "--taichi-fp")
    if arch:
        command.extend(["--taichi-arch", str(arch)])
    if fp:
        command.extend(["--taichi-fp", str(fp)])
    if _pytest_option(item.config, "--run-benchmarks"):
        command.append("--run-benchmarks")
    return command


def pytest_pyfunc_call(pyfuncitem):
    """Execute marked Python tests once, in a dimension-clean child pytest."""

    marker = pyfuncitem.get_closest_marker("isolated_dimension")
    if marker is None:
        return None

    dimension = _marker_dimension(marker)
    child_node = os.environ.get(_CHILD_NODE_ENV)
    if child_node is not None:
        if child_node != pyfuncitem.nodeid:
            pytest.fail(
                "dimension-isolated child selected an unexpected node: "
                f"expected {child_node!r}, got {pyfuncitem.nodeid!r}",
                pytrace=False,
            )
        configured = os.environ.get(_CHILD_DIMENSION_ENV)
        if configured != str(dimension):
            pytest.fail(
                "dimension-isolated child has inconsistent dimension: "
                f"marker={dimension}, environment={configured!r}",
                pytrace=False,
            )
        # Returning None lets pytest's normal Python hook invoke the real test
        # with its fixtures in the dimension-clean child process.
        return None

    repository_root = Path(str(pyfuncitem.config.rootpath)).resolve()
    environment = os.environ.copy()
    environment[_CHILD_NODE_ENV] = pyfuncitem.nodeid
    environment[_CHILD_DIMENSION_ENV] = str(dimension)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repository_root) if not python_path else str(repository_root) + os.pathsep + python_path
    )
    with tempfile.TemporaryDirectory(prefix="geotaichi-isolated-pytest-") as temporary_directory:
        report_path = Path(temporary_directory) / "report.xml"
        command = _isolated_command(pyfuncitem, report_path)
        completed = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        child_report = element_tree.parse(report_path).getroot() if report_path.is_file() else None
    if completed.returncode != 0:
        rendered_command = " ".join(command)
        pytest.fail(
            "dimension-isolated pytest child failed\n"
            f"dimension: {dimension}\n"
            f"node: {pyfuncitem.nodeid}\n"
            f"command: {rendered_command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
            pytrace=False,
        )
    if child_report is None:
        pytest.fail(
            "dimension-isolated pytest child returned success without a "
            "JUnit result\n"
            f"node: {pyfuncitem.nodeid}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
            pytrace=False,
        )

    test_cases = list(child_report.iter("testcase"))
    if len(test_cases) != 1:
        pytest.fail(
            "dimension-isolated pytest child did not report exactly one "
            f"testcase (reported {len(test_cases)})\n"
            f"node: {pyfuncitem.nodeid}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
            pytrace=False,
        )
    skipped = test_cases[0].find("skipped")
    if skipped is not None:
        reason = skipped.get("message") or (skipped.text or "").strip()
        pytest.skip("dimension-isolated child skipped the selected node" + (f": {reason}" if reason else ""))
    return True
