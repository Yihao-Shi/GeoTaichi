"""Detached underresolved particles need not become resolved pressure cells."""

import numpy as np
import json
from pathlib import Path
import subprocess
import sys
import pytest
import taichi as ti
from geotaichi import init
from src.mpm.engines.EngineKernel import kernel_find_fluid_domain_by_volume
from src.mpm.structs.Particle import ParticleCloudIncompressible2D, ParticleCloudIncompressible3D


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("attached", [False, True])
def test_pressure_support_distinguishes_detached_spray_and_attached_surface(dimension, attached):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    n = 8
    cnum = ti.Vector([n] * dimension)
    particles = (ParticleCloudIncompressible2D if dimension == 2 else ParticleCloudIncompressible3D).field(shape=1)
    particles.active[0] = 1
    particles.materialID[0] = 1
    particles.x[0] = [0.025] * dimension
    types = ti.field(int, shape=(n,) * dimension, offset=(-1,) * dimension)
    fraction = ti.field(float, shape=n**dimension)
    own = sum(2 * n**d for d in range(dimension))
    fraction[own] = 0.04
    if attached:
        fraction[sum(3 * n**d for d in range(dimension))] = 0.9
    kernel_find_fluid_domain_by_volume(1, 1, cnum, ti.Vector([100.0] * dimension), 0.1, fraction, types, particles)
    assert types[(2,) * dimension] == int(attached)
    assert particles.active[0] == 1
    np.testing.assert_allclose(particles.x.to_numpy()[0], [0.025] * dimension)


@pytest.mark.parametrize("solver", ["PCG", "MGPCG"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_detached_spray_pressure_preserves_total_particle_momentum(tmp_path, solver, dimension):
    root = Path(__file__).resolve().parents[3]
    run = subprocess.run(
        [
            sys.executable,
            str(root / "tests/helpers/probe_single_phase_spray_momentum.py"),
            "--solver",
            solver,
            "--dimension",
            str(dimension),
            "--output",
            str(tmp_path),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert run.returncode == 0, (run.stdout + run.stderr)[-6000:]
    rows = json.loads((tmp_path / "momentum_audit.json").read_text())["rows"]
    for row in rows:
        assert row["finite"] and row["active_particles"] == 2
        np.testing.assert_allclose(row["final_particle_momentum"], row["initial_particle_momentum"], rtol=0, atol=1e-12)
