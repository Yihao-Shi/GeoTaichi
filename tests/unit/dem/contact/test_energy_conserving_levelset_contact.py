from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.physics_model.contact_model.EnergyConservingModel import PenaltyProperty
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty
from src.mpm.soft_particle.ContactKernel import (
    assemble_lsmpm_wall_contact_force_,
    soft_soft_work_conjugate_penetration_state_,
)


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


def test_penalty_unilateral_cutoff_preserves_spring_work_split() -> None:
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
        prop = PenaltyProperty.field(shape=1)
        normal_force = ti.field(dtype=ti.f64, shape=())
        elastic_energy = ti.field(dtype=ti.f64, shape=())
        damping_power = ti.field(dtype=ti.f64, shape=())

        @ti.kernel
        def evaluate():
            force, elastic, damping = prop[0]._normal_force(
                100.0, 1.0, 1.0, -0.01, 1.0
            )
            normal_force[None] = force
            elastic_energy[None] = elastic
            damping_power[None] = damping

        prop[0].add_surface_property(100.0, 100.0, 2.0, 0.3, 1.0, 0.0)
        evaluate()

        assert normal_force[None] == pytest.approx(0.0)
        assert elastic_energy[None] == pytest.approx(0.5 * 100.0 * 0.01**2)
        assert damping_power[None] == pytest.approx(-1.0)
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        ti.reset()


def test_penalty_coulomb_return_map_records_only_plastic_slip() -> None:
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
        prop = PenaltyProperty.field(shape=1)
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
                ti.Vector([0.0, 0.0, 0.0]),
                1.0,
                ti.Vector([0.0, 0.0, 1.0]),
                ti.Vector([0.01, 0.0, 0.0]),
                dt,
            )
            force[None] = tangential_force
            overlap[None] = current
            friction[None] = friction_energy

        prop[0].add_surface_property(100.0, 100.0, 2.0, 0.3, 0.0, 0.0)
        dt[None] = 0.1
        evaluate()

        assert force[None][0] == pytest.approx(-0.3)
        assert overlap[None][0] == pytest.approx(0.003)
        assert friction[None] == pytest.approx(-0.3 * 0.007)
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        ti.reset()


def test_work_conjugate_soft_soft_activation_starts_from_zero_energy() -> None:
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
        prop = PenaltyProperty.field(shape=1)
        dt = ti.field(dtype=ti.f64, shape=())
        stored = ti.field(dtype=ti.f64, shape=())
        active = ti.field(dtype=ti.i32, shape=())
        gap_rate = ti.field(dtype=ti.f64, shape=())
        state = ti.Vector.field(2, dtype=ti.f64, shape=())
        force = ti.Vector.field(3, dtype=ti.f64, shape=())
        assemble = ti.field(dtype=ti.i32, shape=())
        dt[None] = 1.0e-4

        @ti.kernel
        def evaluate():
            penetration, next_penetration, assemble_contact = (
                soft_soft_work_conjugate_penetration_state_(
                    stored[None], active[None], gap_rate[None], dt[None]
                )
            )
            state[None] = ti.Vector([penetration, next_penetration])
            assemble[None] = ti.cast(assemble_contact, ti.i32)
            normal, _, _ = prop[0]._force_assemble_work_conjugate(
                1.0,
                1.0,
                penetration,
                next_penetration,
                0.4,
                ti.Vector([1.25, 0.0, 0.0]),
                ti.Vector([gap_rate[None] / 1.25, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                dt,
            )
            force[None] = normal

        prop[0].add_surface_property(
            1200.0, 600.0, 2.0, 0.0, 0.0, 0.0
        )

        # A newly detected or reactivated query ignores a finite stale overlap
        # and stores only the penetration generated by this step's trace work.
        stored[None] = 0.02
        active[None] = 0
        gap_rate[None] = -2.0
        evaluate()
        np.testing.assert_allclose(
            state[None], [0.0, 2.0e-4], rtol=0.0, atol=1.0e-15
        )
        assert int(assemble[None]) == 1
        current_energy = 0.0
        next_energy = 0.5 * (1200.0 * 0.4) * (2.0e-4) ** 2
        contact_work = float(force[None][0]) * (-2.0 / 1.25) * dt[None]
        assert contact_work + next_energy - current_energy == pytest.approx(
            0.0, abs=1.0e-15
        )

        # Opening cannot recreate a spring from a lagging negative SDF.
        active[None] = 0
        gap_rate[None] = 2.0
        evaluate()
        np.testing.assert_allclose(state[None], [0.0, 0.0], atol=0.0)
        assert int(assemble[None]) == 0

        # An inherited active history continues from its stored penetration.
        stored[None] = 0.02
        active[None] = 1
        gap_rate[None] = 2.0
        evaluate()
        np.testing.assert_allclose(
            state[None], [0.02, 0.0198], rtol=0.0, atol=1.0e-15
        )
        assert int(assemble[None]) == 1
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        ti.reset()


def test_trilinear_levelset_gradient_uses_physical_coordinates() -> None:
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    try:
        from src.dem.structs.BaseStruct import BoundingBox, LevelSetGrid
        from src.utils.ScalarFunction import linearize3D

        box = BoundingBox.field(shape=1)
        grid = LevelSetGrid.field(shape=8)
        gradient = ti.Vector.field(3, dtype=ti.f64, shape=())
        spacing = 0.03

        @ti.kernel
        def initialize():
            box[0]._set_bounding_box(
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([spacing, spacing, spacing]),
            )
            box[0]._add_grid(
                0, spacing, ti.Vector([2, 2, 2]), 1.0, 0
            )
            for i, j, k in ti.ndrange(2, 2, 2):
                node = linearize3D(i, j, k, ti.Vector([2, 2, 2]))
                position = spacing * ti.Vector([i, j, k])
                grid[node].distance_field = (
                    (2.0 / 3.0) * position[0]
                    - (1.0 / 3.0) * position[1]
                    + (2.0 / 3.0) * position[2]
                    - 0.01
                )

        @ti.kernel
        def sample():
            gradient[None] = box[0].calculate_gradient(
                spacing * ti.Vector([0.37, 0.42, 0.61]), grid
            )

        initialize()
        sample()
        np.testing.assert_allclose(
            gradient[None],
            [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        assert np.linalg.norm(gradient[None]) == pytest.approx(
            1.0, rel=1.0e-13, abs=1.0e-13
        )
    finally:
        ti.reset()


def test_quadratic_levelset_penalty_force_is_potential_gradient() -> None:
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
        prop = PenaltyProperty.field(shape=1)
        dt = ti.field(dtype=ti.f64, shape=())
        gap = ti.field(dtype=ti.f64, shape=())
        force = ti.Vector.field(3, dtype=ti.f64, shape=())
        tangential_force = ti.Vector.field(3, dtype=ti.f64, shape=())
        dt[None] = 1.0e-4

        @ti.kernel
        def evaluate():
            prop[0].elastic_energy = 0.0
            normal, tangential, _ = prop[0]._force_assemble(
                1.0,
                1.0,
                gap[None],
                0.4,
                ti.Vector([1.25, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                dt,
            )
            force[None] = normal
            tangential_force[None] = tangential

        prop[0].add_surface_property(1200.0, 600.0, 2.0, 0.0, 0.0, 0.0)
        gap[None] = -0.02
        evaluate()

        effective_coefficient = 0.4 * 1.25
        expected_force = 1200.0 * effective_coefficient * 0.02
        expected_energy = 0.5 * 1200.0 * 0.4 * 0.02**2
        np.testing.assert_allclose(
            force[None], [expected_force, 0.0, 0.0], rtol=1.0e-13, atol=1.0e-13
        )
        np.testing.assert_allclose(
            tangential_force[None], np.zeros(3), rtol=0.0, atol=1.0e-15
        )
        assert float(prop.elastic_energy.to_numpy()[0]) == pytest.approx(
            expected_energy, rel=5.0e-7, abs=5.0e-8
        )

        epsilon = 1.0e-6
        gap[None] = -0.02 + 1.25 * epsilon
        evaluate()
        energy_plus = float(prop.elastic_energy.to_numpy()[0])
        gap[None] = -0.02 - 1.25 * epsilon
        evaluate()
        energy_minus = float(prop.elastic_energy.to_numpy()[0])
        force_from_potential = -(energy_plus - energy_minus) / (2.0 * epsilon)
        gap[None] = -0.02
        evaluate()
        assert float(force[None][0]) == pytest.approx(
            force_from_potential, rel=1.0e-10, abs=1.0e-10
        )
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        ti.reset()


def test_quadratic_levelset_penalty_matches_linear_for_unit_sdf_gradient() -> None:
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    previous_track_energy = GlobalVariable.TRACKENERGY
    previous_adaptive_stiffness = GlobalVariable.ADAPTIVESTIFF
    GlobalVariable.TRACKENERGY = True
    GlobalVariable.ADAPTIVESTIFF = False

    try:
        penalty = PenaltyProperty.field(shape=1)
        linear = LinearSurfaceProperty.field(shape=1)
        dt = ti.field(dtype=ti.f64, shape=())
        penalty_force = ti.Vector.field(3, dtype=ti.f64, shape=())
        linear_force = ti.Vector.field(3, dtype=ti.f64, shape=())
        dt[None] = 1.0e-4

        @ti.kernel
        def evaluate():
            p_normal, p_tangential, _ = penalty[0]._force_assemble(
                1.0,
                0.1,
                -0.02,
                0.4,
                ti.Vector([1.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                dt,
            )
            l_normal, l_tangential, _, _ = linear[0]._force_assemble(
                1.0,
                0.1,
                -0.02,
                0.4,
                0.1,
                ti.Vector([1.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                dt,
            )
            penalty_force[None] = p_normal + p_tangential
            linear_force[None] = l_normal + l_tangential

        penalty[0].add_surface_property(
            1200.0, 600.0, 2.0, 0.0, 0.0, 0.0
        )
        linear[0].add_surface_property(
            1200.0, 600.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        evaluate()
        np.testing.assert_allclose(
            penalty_force[None], linear_force[None], rtol=1.0e-13, atol=1.0e-13
        )
        assert float(penalty.elastic_energy.to_numpy()[0]) == pytest.approx(
            float(linear.elastic_energy.to_numpy()[0]),
            rel=1.0e-13,
            abs=1.0e-13,
        )
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        GlobalVariable.ADAPTIVESTIFF = previous_adaptive_stiffness
        ti.reset()


def test_lsmpm_wall_dispatch_uses_energy_conserving_signature() -> None:
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
        prop = PenaltyProperty.field(shape=1)
        dt = ti.field(dtype=ti.f64, shape=())
        force = ti.Vector.field(3, dtype=ti.f64, shape=())
        momentum = ti.Vector.field(3, dtype=ti.f64, shape=())
        dt[None] = 1.0e-4

        @ti.kernel
        def evaluate():
            normal, tangential, couple, _ = assemble_lsmpm_wall_contact_force_(
                prop,
                0,
                0,
                1.0,
                0.1,
                -0.02,
                0.4,
                0.1,
                ti.Vector([1.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                dt,
            )
            force[None] = normal + tangential
            momentum[None] = couple

        prop[0].add_surface_property(1200.0, 600.0, 2.0, 0.0, 0.0, 0.0)
        evaluate()
        np.testing.assert_allclose(
            force[None], [9.6, 0.0, 0.0], rtol=1.0e-13, atol=1.0e-13
        )
        np.testing.assert_allclose(
            momentum[None], np.zeros(3), rtol=0.0, atol=1.0e-15
        )
        expected_energy = 0.5 * 1200.0 * 0.4 * 0.02**2
        assert float(prop.elastic_energy.to_numpy()[0]) == pytest.approx(
            expected_energy, rel=5.0e-7, abs=5.0e-8
        )
    finally:
        GlobalVariable.TRACKENERGY = previous_track_energy
        ti.reset()
