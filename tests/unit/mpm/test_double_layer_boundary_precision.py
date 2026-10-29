"""Decimal grid boundaries must use the configured precision before ti.init."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("dimension", [2, 3])
def test_decimal_domain_wall_clamps_boundary_node_in_fresh_process(dimension):
    result = subprocess.run(
        [sys.executable, "-m", "tests.unit.mpm.test_double_layer_boundary_precision", str(dimension)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "GEOTAICHI_REAL_DTYPE": "float64"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    import numpy as np
    import taichi as ti
    from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
        kernel_enforce_double_layer_solid_plane_nodes2d,
        kernel_enforce_double_layer_solid_plane_nodes3d,
    )
    from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D

    dimension = int(sys.argv[1])
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=2, offline_cache=False)
    vector = ti.types.vector(dimension, ti.f64)
    fields = ti.types.struct(ms=ti.f64, momentums=vector, momentum=vector, forces=vector, force=vector)
    node = fields.field(shape=(41**dimension, 1))
    index = 40 + 20 * 41 + (20 * 41**2 if dimension == 3 else 0)
    node[index, 0].ms = 1.0
    for name in ["momentums", "momentum", "forces", "force"]:
        getattr(node, name)[index, 0] = [1.0] * dimension
    clamp = (
        kernel_enforce_double_layer_solid_plane_nodes2d
        if dimension == 2
        else kernel_enforce_double_layer_solid_plane_nodes3d
    )
    clamp(
        1e-12,
        ti.Vector([0.005] * dimension),
        ti.Vector([41] * dimension),
        ti.Vector([0.0] * dimension),
        ti.Vector([0.2] * dimension),
        ti.Vector([0.2] + [0.0] * (dimension - 1)),
        ti.Vector([1.0] + [0.0] * (dimension - 1)),
        node,
    )
    for name in ["momentums", "momentum", "forces", "force"]:
        np.testing.assert_array_equal(getattr(node, name)[index, 0], [0.0] + [1.0] * (dimension - 1))
    production = (NodeTwoPhase2D if dimension == 2 else NodeTwoPhase).field(shape=(1, 1))
    for name in ["ms", "mf", "porosity", "pressure"]:
        assert getattr(production, name).to_numpy().dtype == np.float64, name
