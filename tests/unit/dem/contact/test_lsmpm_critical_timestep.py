import math

import pytest
import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.dem.BaseKernel import find_lsmpm_contact_min_mass_
from src.dem.contact.ContactKernel import (
    kernel_find_max_barrier_stiffness_lsmpm,
    kernel_find_max_hertz_stiffness_lsmpm,
    kernel_find_max_penalty_stiffness_lsmpm,
    kernel_find_max_stiffness_lsmpm,
)
from src.physics_model.contact_model.BarrierModel import BarrierProperty
from src.physics_model.contact_model.EnergyConservingModel import PenaltyProperty
from src.physics_model.contact_model.HertzMindlinModel import (
    HertzMindlinSurfaceProperty,
)
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


@ti.dataclass
class RigidFixture:
    m: ti.f64
    equi_r: ti.f64
    materialID: ti.i32
    is_soft: ti.i32
    start_node: ti.i32
    end_node: ti.i32

    @ti.func
    def _get_mass(self):
        return self.m

    @ti.func
    def _get_vertice_number(self):
        return self.end_node - self.start_node

    @ti.func
    def global_node_to_local(self, global_node):
        return global_node - self.start_node


@ti.dataclass
class BoxFixture:
    reference_surface_area: ti.f64
    scale: ti.f64


@ti.dataclass
class VertexFixture:
    parameter: ti.f64


def test_lsmpm_critical_mass_and_stiffness_match_contact_quadrature() -> None:
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    previous_adaptive = GlobalVariable.ADAPTIVESTIFF
    GlobalVariable.ADAPTIVESTIFF = False

    try:
        rigid = RigidFixture.field(shape=2)
        box = BoxFixture.field(shape=2)
        vertice = VertexFixture.field(shape=1)
        surface = ti.field(dtype=ti.i32, shape=1)
        linear = LinearSurfaceProperty.field(shape=1)
        penalty = PenaltyProperty.field(shape=1)
        hertz = HertzMindlinSurfaceProperty.field(shape=1)
        barrier = BarrierProperty.field(shape=1)

        rigid[0].m = 10.0
        rigid[0].equi_r = 0.4
        rigid[0].materialID = 0
        rigid[0].is_soft = 1
        rigid[0].start_node = 0
        rigid[0].end_node = 5
        rigid[1].m = 1.5
        rigid[1].equi_r = 0.2
        rigid[1].materialID = 0
        rigid[1].is_soft = 0
        rigid[1].start_node = 0
        rigid[1].end_node = 1
        box[0].reference_surface_area = 6.0
        box[0].scale = 0.5
        box[1].reference_surface_area = 1.0
        box[1].scale = 1.0
        vertice[0].parameter = 0.2
        surface[0] = 0
        linear[0].add_surface_property(
            120.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        penalty[0].add_surface_property(120.0, 80.0, 2.0, 0.0, 0.0, 0.0)
        hertz[0].add_surface_property(100.0, 30.0, 0.0, 0.0, 0.0, 0.0)
        barrier[0].add_surface_property(50.0, 0.04, 1.0, 0.0, 0.0, 0.0)

        contact_mass = find_lsmpm_contact_min_mass_(2, rigid)
        linear_stiffness = kernel_find_max_stiffness_lsmpm(
            1, 1, rigid, surface, vertice, box, linear
        )
        penalty_stiffness = kernel_find_max_penalty_stiffness_lsmpm(
            1, 1, 0.1, 1.2, rigid, surface, vertice, box, penalty
        )
        hertz_stiffness = kernel_find_max_hertz_stiffness_lsmpm(
            1, 1, 0.1, rigid, surface, vertice, box, hertz
        )
        barrier_stiffness = kernel_find_max_barrier_stiffness_lsmpm(
            1, 1, 1.2, rigid, surface, vertice, box, barrier
        )

        # A_i = 6 * 0.5^2 * 0.2 = 0.3; M_s = 10 / 5 = 2.
        assert float(contact_mass) == pytest.approx(1.5, rel=1.0e-6)
        assert float(linear_stiffness) == pytest.approx(36.0, rel=1.0e-6)
        assert float(penalty_stiffness) == pytest.approx(43.2, rel=1.0e-6)
        assert float(hertz_stiffness) == pytest.approx(14.4, rel=1.0e-6)
        ratio = 0.04 / 0.4
        expected_barrier = -50.0 * 0.3 * 1.2 * (
            2.0 * math.log(ratio)
            + ((ratio - 1.0) * (3.0 * ratio + 1.0)) / ratio**2
        )
        assert float(barrier_stiffness) == pytest.approx(
            expected_barrier, rel=1.0e-6
        )
    finally:
        GlobalVariable.ADAPTIVESTIFF = previous_adaptive
        ti.reset()
