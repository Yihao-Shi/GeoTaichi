from __future__ import annotations

import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


def test_unilateral_dashpot_cutoff_preserves_the_spring_work_split() -> None:
    """A tensile trial is clipped without silently deleting spring energy."""
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    previous_track_energy = GlobalVariable.TRACKENERGY
    GlobalVariable.TRACKENERGY = True
    try:
        prop = LinearSurfaceProperty.field(shape=1)
        normal_force = ti.field(dtype=ti.f64, shape=())
        elastic_energy = ti.field(dtype=ti.f64, shape=())
        damping_power = ti.field(dtype=ti.f64, shape=())

        @ti.kernel
        def evaluate():
            force, elastic, damping = prop[0]._normal_force(
                100.0,
                1.0,
                1.0,
                -0.01,
                1.0,
            )
            normal_force[None] = force
            elastic_energy[None] = elastic
            damping_power[None] = damping

        evaluate()

        # The 1 N compressed spring is exactly balanced by the capped
        # dashpot.  Total force remains zero, while its stored energy and
        # unloading work remain visible to the physical energy ledger.
        assert normal_force[None] == pytest.approx(0.0)
        assert elastic_energy[None] == pytest.approx(0.5 * 100.0 * 0.01**2)
        assert damping_power[None] == pytest.approx(-1.0)
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        ti.reset()


def test_coulomb_return_map_records_unloading_plastic_slip() -> None:
    """A shrinking Coulomb limit dissipates stored shear energy at zero rate."""
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    previous_track_energy = GlobalVariable.TRACKENERGY
    GlobalVariable.TRACKENERGY = True
    try:
        prop = LinearSurfaceProperty.field(shape=1)
        dt = ti.field(dtype=ti.f64, shape=())
        force = ti.Vector.field(3, dtype=ti.f64, shape=())
        overlap = ti.Vector.field(3, dtype=ti.f64, shape=())
        friction = ti.field(dtype=ti.f64, shape=())

        @ti.kernel
        def evaluate():
            tangential_force, current, _, _, friction_energy = prop[
                0
            ]._tangential_force(
                100.0,
                0.0,
                1.0,
                1.0,
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 1.0]),
                ti.Vector([0.01, 0.0, 0.0]),
                dt,
            )
            force[None] = tangential_force
            overlap[None] = current
            friction[None] = friction_energy

        prop[0].add_surface_property(
            100.0,
            100.0,
            0.0,
            0.0,
            0.3,
            0.3,
            0.0,
            0.0,
            0.0,
        )
        dt[None] = 0.1
        evaluate()

        assert force[None][0] == pytest.approx(-0.3)
        assert overlap[None][0] == pytest.approx(0.003)
        # Plastic slip is 0.01 - 0.003 m and the opposing Coulomb force is
        # -0.3 N.  The DEM ledger stores dissipative work with a negative sign.
        assert friction[None] == pytest.approx(-0.3 * 0.007)
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        ti.reset()
