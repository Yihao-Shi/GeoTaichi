import json
from types import SimpleNamespace

import numpy as np

from examples.mmpm.SubmarineLandslide.submarine_landslide_3d.draw.evaluate_submarine_landslide_3d import write_metrics
from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d_parameters import DOMAIN as LEAKAGE_DOMAIN
from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d import boundaries as leakage_boundaries
from examples.mmpm.GranularWaterLeakage3D.draw.evaluate_granular_water_leakage_3d import (
    write_metrics as write_leakage_metrics,
)
from examples.mmpm.TwoPhaseLSDEMCoupling.sphere_impact_submerged_bed_3d.draw.evaluate_sphere_impact_submerged_bed_3d import (
    write_metrics as write_sphere_impact_metrics,
)


def test_landslide_metrics_require_conserved_downslope_motion(tmp_path):
    particle_dir = tmp_path / "particles"
    particle_dir.mkdir()
    phase = np.array([2, 2, 1, 1], dtype=np.uint8)
    active = np.ones(4, dtype=np.uint8)
    initial = np.array([[1.0, 0.03, 1.59], [2.0, 0.05, 1.58], [3.10, 0.03, 1.20], [3.50, 0.05, 1.30]])
    final = initial.copy()
    final[:2, 2] += [0.01, -0.01]
    final[2:, 0] -= 0.05
    final[2:, 2] -= 0.05
    for frame, (time, position) in enumerate(((0.0, initial), (0.8, final))):
        solid_velocity = np.zeros((4, 3))
        solid_velocity[2:, [0, 2]] = -0.2
        np.savez(
            particle_dir / f"MPMParticle{frame:06d}.npz",
            t_current=time,
            active=active,
            phase=phase,
            position=position,
            fluid_velocity=np.zeros((4, 3)),
            solid_velocity=solid_velocity,
            pressure=np.zeros(4),
            porosity=np.full(4, 0.38),
        )

    args = SimpleNamespace(dx=0.01, dt=5.0e-5, time=0.8, thickness=0.08, wall_cells=1, strict=True)
    write_metrics(tmp_path, 2, 2, args)

    assert json.loads((tmp_path / "metrics.json").read_text())["passed"]


def test_sphere_impact_metrics_require_entry_deceleration_and_coupling_force(tmp_path):
    particle_dir = tmp_path / "particles"
    particle_dir.mkdir()
    directions = []
    for polar in range(4):
        z = -0.75 + 0.5 * polar
        radial = np.sqrt(1.0 - z * z)
        for azimuth in range(8):
            angle = 2.0 * np.pi * (azimuth + 0.5) / 8.0
            directions.append([radial * np.cos(angle), radial * np.sin(angle), z])
    fluid = np.array([0.1, 0.1, 0.09]) + 0.015 * np.asarray(directions)
    position = np.vstack((fluid, [0.05, 0.15, 0.05], [0.15, 0.05, 0.08]))
    phase = np.r_[np.full(len(fluid), 2, dtype=np.uint8), np.ones(2, dtype=np.uint8)]
    active = np.ones(len(position), dtype=np.uint8)
    for frame, (time, center_z, vertical_velocity, force) in enumerate(
        ((0.0, 0.1625, -4.429446918, 0.0), (0.12, 0.09, -0.2, 100.0))
    ):
        np.savez(
            particle_dir / f"MPMParticle{frame:06d}.npz",
            t_current=time,
            active=active,
            phase=phase,
            position=position,
            fluid_velocity=np.zeros((len(position), 3)),
            solid_velocity=np.zeros((len(position), 3)),
            pressure=np.zeros(len(position)),
        )
        np.savez(
            particle_dir / f"LSDEMRigid{frame:06d}.npz",
            t_current=time,
            mass_center=np.array([[0.1, 0.1, center_z]]),
            velocity=np.array([[0.0, 0.0, vertical_velocity]]),
            contact_force=np.array([[0.0, 0.0, force]]),
        )

    args = SimpleNamespace(dx=0.005, dt=1.0e-5, time=0.12, drop_height=1.0, ppc=2, strict=True)
    write_sphere_impact_metrics(tmp_path, len(fluid), 2, args)

    assert json.loads((tmp_path / "metrics.json").read_text())["passed"]


def test_leakage_metrics_require_water_to_reach_receiver(tmp_path):
    particle_dir = tmp_path / "particles"
    particle_dir.mkdir()
    phase = np.array([2, 2, 1, 1], dtype=np.uint8)
    active = np.ones(4, dtype=np.uint8)
    initial = np.array([[0.10, 0.08, 0.20], [0.20, 0.08, 0.20], [0.10, 0.08, 0.15], [0.20, 0.08, 0.15]])
    final = initial.copy()
    final[0] = [0.18, 0.08, 0.06]
    for frame, (time, position) in enumerate(((0.0, initial), (6.0, final))):
        fluid_velocity = np.zeros((4, 3))
        fluid_velocity[0, 2] = -0.2 if frame else 0.0
        np.savez(
            particle_dir / f"MPMParticle{frame:06d}.npz",
            t_current=time,
            active=active,
            phase=phase,
            position=position,
            fluid_velocity=fluid_velocity,
            solid_velocity=np.zeros((4, 3)),
            pressure=np.zeros(4),
            porosity=np.full(4, 0.40),
        )

    args = SimpleNamespace(dx=0.005, dt=2.0e-5, time=6.0, strict=True)
    write_leakage_metrics(tmp_path, 2, 2, args)

    assert json.loads((tmp_path / "metrics.json").read_text())["passed"]


def test_leakage_upper_tank_side_walls_cover_air_padding():
    walls = leakage_boundaries(0.005)
    physical_top = 0.325
    assert all(wall["BoundaryType"] == "SolidCell" for wall in walls[:4])
    assert all(wall["EndPoint"][2] == physical_top for wall in walls[:4])
    assert all(wall["BoundaryType"] == "SolidPlaneCell" for wall in walls[4:8])
    assert all(wall["StartPoint"][2] == physical_top for wall in walls[4:8])
    assert all(wall["EndPoint"][2] == LEAKAGE_DOMAIN[2] for wall in walls[4:8])
    assert walls[8]["EndPoint"][2] == walls[8]["StartPoint"][2]
    assert walls[9]["EndPoint"][2] == walls[9]["StartPoint"][2]
