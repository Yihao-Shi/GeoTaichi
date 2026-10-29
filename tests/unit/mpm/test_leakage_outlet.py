import json
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d import boundaries
from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d_parameters import represented_crack_bounds
from examples.mmpm.GranularWaterLeakage3D.draw.evaluate_granular_water_leakage_3d import write_metrics
from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_enforce_double_layer_solid_plane_nodes3d


def test_outlet_retains_original_geometry_and_validates_custom_width():
    assert represented_crack_bounds(0.005) == pytest.approx((0.18, 0.185))
    assert represented_crack_bounds(0.005, 0.08) == pytest.approx((0.14, 0.22))
    for width in (0, float("nan"), 0.007, 0.005, 0.3):
        with pytest.raises(ValueError):
            represented_crack_bounds(0.005, width)


def test_wider_outlet_has_free_solid_grid_nodes_under_native_wall_constraints():
    ti.init(arch=ti.cpu, cpu_max_num_threads=2, default_fp=ti.f64, offline_cache=False)
    try:
        dims = np.array([73, 33, 73])
        node = ti.types.struct(
            ms=ti.f64,
            momentums=ti.types.vector(3, ti.f64),
            momentum=ti.types.vector(3, ti.f64),
            forces=ti.types.vector(3, ti.f64),
            force=ti.types.vector(3, ti.f64),
        ).field(shape=(int(np.prod(dims)), 1))
        center = 36 + 73 * 16 + 73 * 33 * 25
        for width, expected in ((None, 0), (0.08, -1)):
            node.ms.fill(1)
            node.momentums.fill(-1)
            for wall in boundaries(0.005, width)[8:10]:
                normal = np.array(wall["Norm"])
                start, end = np.array(wall["StartPoint"]), np.array(wall["EndPoint"])
                kernel_enforce_double_layer_solid_plane_nodes3d(
                    1e-12,
                    ti.Vector([0.005] * 3),
                    ti.Vector(dims),
                    ti.Vector(start + np.minimum(normal, 0) * 0.005),
                    ti.Vector(end + np.maximum(normal, 0) * 0.005),
                    ti.Vector(start),
                    ti.Vector(normal),
                    node,
                )
            assert node.momentums[center, 0][2] == expected
    finally:
        ti.reset()


def test_discharge_gate_requires_majority_of_initial_solid_mass(tmp_path):
    (tmp_path / "particles").mkdir()
    args = SimpleNamespace(dx=0.005, dt=2e-5, time=6, crack_width=0.08, minimum_solid_discharge=0.6, strict=False)
    for discharged in (False, True):
        for frame, time in enumerate((0.0, 6.0)):
            position = np.array([[0.18, 0.08, 0.2], [0.18, 0.08, 0.16], [0.1, 0.08, 0.16]])
            if frame:
                position[0, 2] = 0.02
                if discharged:
                    position[1, 2] = 0.03
            np.savez(
                tmp_path / f"particles/MPMParticle{frame:06d}.npz",
                t_current=time,
                active=[1, 1, 1],
                phase=[2, 1, 1],
                position=position,
                fluid_velocity=np.full((3, 3), 0.1),
                solid_velocity=np.full((3, 3), 0.1),
                pressure=np.zeros(3),
                porosity=np.full(3, 0.4),
                volume=np.ones(3),
                solid_mass=[0, 3, 1],
                fluid_mass=[1, 0, 0],
            )
        write_metrics(tmp_path, 1, 2, args)
        metrics = json.loads((tmp_path / "metrics.json").read_text())
        assert metrics["passed"] is discharged
        assert metrics["solid_mass_conserved"]
        assert metrics["final_discharged_solid_mass_fraction"] == (0.75 if discharged else 0)
