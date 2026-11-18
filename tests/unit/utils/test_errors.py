"""Shared error ownership and concrete runtime failure contracts."""

from pathlib import Path
import subprocess
import sys

import pytest
import taichi as ti

from src.fem.cloth.ClothAssembler import ClothAssembler
from src.utils.errors import RuntimeNotInitializedError

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial]


def test_error_module_import_has_no_solver_or_taichi_dependency():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.utils.errors; "
            "assert 'taichi' not in sys.modules; "
            "assert not any(name.startswith(('src.fem', 'src.iga', 'src.mpm', 'src.dem')) for name in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )


def test_cloth_reports_uninitialized_runtime_with_specific_error():
    ti.reset()
    with pytest.raises(RuntimeNotInitializedError, match="initialized runtime"):
        ClothAssembler(None, None)
