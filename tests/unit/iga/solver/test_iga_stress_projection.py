"""IGA visualization stress must be a volume-normalized nodal projection."""

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

import src.iga.config as iga_config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


def test_uniform_deformation_has_uniform_projected_stress(tmp_path):
    from src.iga import Cube, ExplicitIGA, Primitives

    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    iga_config.set_dimension(3)
    try:
        cube = Cube()
        cube.set_parameters(start_point=[0.0, 0.0, 0.0], size=[1.0, 0.5, 0.2])
        cube.generate_knot_u(degree=2, num_ctrlpts=3)
        cube.generate_knot_v(degree=2, num_ctrlpts=3)
        cube.generate_knot_w(degree=2, num_ctrlpts=3)
        cube.generate_ctrlpts()
        cube.generate_weights()
        primitives = Primitives()
        primitives.append(cube, "block")
        primitives.finialize()
        engine = ExplicitIGA(
            primitives=primitives,
            young_modulus=1.0e4,
            poisson_ratio=0.3,
            density=1000.0,
            gravity=[0.0, 0.0, 0.0],
            degree=[2, 2, 2],
            dt=1.0e-4,
            step=1,
            path=str(tmp_path),
        )
        engine.precompute()
        engine.visualize_stress()
        assert np.count_nonzero(engine.patch.stress.to_numpy()) == 0

        current = engine.patch.control_points.to_numpy()
        current[:, 0] *= 1.02
        engine.patch.control_points.from_numpy(current)
        engine.visualize_stress()
        stress = engine.patch.stress.to_numpy()
        assert np.min(stress) > 0.0
        assert np.allclose(stress, stress[0], rtol=1.0e-10, atol=1.0e-10)
    finally:
        ti.reset()
