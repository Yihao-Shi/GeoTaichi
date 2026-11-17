"""Moving-piston continuation must preserve forcing time and both phase states."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mpm, pytest.mark.cpu, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "examples/mmpm/TwoPhaseWavemaker3D/two_layer_two_phase_wavemaker_3d.py"


def test_wavemaker_fresh_process_restart_matches_uninterrupted(tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    common = [
        sys.executable,
        str(ENTRY),
        "--arch",
        "cpu",
        "--dx",
        ".04",
        "--dt",
        ".0001",
        "--time",
        ".0004",
        "--save-interval",
        ".0002",
        "--fluid-ppc",
        "1",
        "--solid-ppc",
        "1",
        "--ramp-time",
        ".001",
        "--wave-velocity",
        ".01",
        "--no-post",
    ]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        GEOTAICHI_REAL_DTYPE="float64",
        TI_CPU_MAX_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )

    def run(output, extra=()):
        log = tmp_path / (output.name + ".log")
        with log.open("w") as stream:
            result = subprocess.run(
                [*common, "--output", str(output), *extra],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        assert result.returncode == 0, log.read_text()[-8000:]

    run(full)
    resumed.mkdir()
    shutil.copy2(full / "configuration.json", resumed / "configuration.json")
    for folder, prefix in [("particles", "MPMParticle"), ("grids", "MPMGrid")]:
        (resumed / folder).mkdir()
        for frame in [0, 1]:
            name = f"{prefix}{frame:06d}.npz"
            shutil.copy2(full / folder / name, resumed / folder / name)
    run(resumed, ["--restart-from", "1", "--restart-path", str(full)])
    with (
        np.load(full / "particles/MPMParticle000002.npz") as left,
        np.load(resumed / "particles/MPMParticle000002.npz") as right,
    ):
        assert int(right["current_step"]) == int(left["current_step"]) == 4
        assert float(right["t_current"]) == float(left["t_current"]) == 0.0004
        for name in [
            "position",
            "velocity",
            "solid_velocity",
            "fluid_velocity",
            "mass",
            "solid_mass",
            "fluid_mass",
            "volume",
            "porosity",
            "pressure",
            "stress",
            "solid_velocity_gradient",
            "fluid_velocity_gradient",
        ]:
            np.testing.assert_allclose(right[name], left[name], rtol=1e-8, atol=1e-9, err_msg=name)
        assert np.max(np.abs(left["fluid_velocity"])) > 1e-6
