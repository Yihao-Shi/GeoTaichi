from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator
from src.mpdem.engines.DirectAffineIPCAssembler import DirectAffineIPCAssembler
from src.physics_model.contact_model.ipc.IPC import ipc_toolkit_barrier_distance2_offset_terms


@ti.dataclass
class Particle:
    bodyID: int


@ti.kernel
def offset_barrier_energy(distance: float, dmin: float) -> float:
    active = (0.02 + 2.0 * dmin) * 0.02
    energy, first, second = ipc_toolkit_barrier_distance2_offset_terms(
        distance * distance, 0.02, dmin, 1.0 / (active * active), 0
    )
    return 0.02 * energy


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("dmin", [0.0, 0.01])
def test_mixed_friction_normal_force_matches_contact_energy_derivative(taichi_runtime, dimension, dmin):
    assembler = object.__new__(DirectAffineIPCAssembler)
    assembler.real_type = ti.f64
    assembler.surface_count = assembler.mpm_body_count = assembler.fem_body_count = 1
    particle = Particle.field(shape=1)
    surface_id = ti.field(int, shape=1)
    measure = ti.field(float, shape=1)
    measure[0] = 1.0
    assembler.mpm = SimpleNamespace(particle=particle, surface_id=surface_id, surface_measure=measure)
    assembler.fem_node_body = ti.field(int, shape=3)
    assembler.positions = ti.Vector.field(3, float, shape=4)
    distance = dmin + 0.001
    positions = np.array([[0.25, 0.25, distance], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if dimension == 2:
        positions[0] = [0.25, distance, 0.0]
    assembler.positions.from_numpy(positions)
    assembler.pair_active = ti.field(int, shape=(1, 1))
    assembler.pair_active[0, 0] = 1
    for name, value in (("pair_dmin", dmin), ("pair_dhat", 0.02), ("pair_kappa", 1.0)):
        field = ti.field(float, shape=(1, 1))
        field[0, 0] = value
        setattr(assembler, name, field)
    size = dimension + 1
    candidates = ti.Vector.field(size, int, shape=1)
    candidates[0] = list(range(size))
    frozen = ti.Vector.field(size, int, shape=1)
    weights = ti.Vector.field(size, float, shape=1)
    normals = ti.Vector.field(3, float, shape=1)
    force = ti.field(float, shape=1)
    getattr(assembler, "_freeze_friction_" + str(dimension) + "d")(1, candidates, frozen, weights, normals, force)
    epsilon = 1e-8
    expected = -(offset_barrier_energy(distance + epsilon, dmin) - offset_barrier_energy(distance - epsilon, dmin)) / (
        2 * epsilon
    )
    assert force[0] == pytest.approx(expected, rel=1e-8)


@ti.data_oriented
class PlanarContact(TaichiAffineBodyOperator):
    def __init__(self):
        self.body_num = 2
        self.control_num = 8
        self.max_body_vertex_count = 1
        self.fully_implicit = False
        self.epsv = 0.001
        for name in ("y", "hat_y", "grad", "levelset_friction_y"):
            setattr(self, name, ti.Vector.field(3, float, shape=8))
        self.x = ti.Vector.field(3, float, shape=1)
        self.basis = ti.field(float, shape=(1, 4))
        self.basis[0, 0] = 1.0
        for name in ("active_body_pair_count", "levelset_friction_contacts", "levelset_active_contacts"):
            setattr(self, name, ti.field(int, shape=()))
        self.active_body_pair_count[None] = 1
        self.active_body_pair_list = ti.field(int, shape=1)
        self.active_body_pair_list[0] = 1
        for name in ("body_vertex_count", "body_vertex_start", "body_contact_type"):
            setattr(self, name, ti.field(int, shape=2))
        self.body_vertex_count[0] = 1
        self.body_contact_type.fill(1)
        self.body_scale = ti.field(float, shape=2)
        self.body_scale.fill(1.0)
        for name in ("node_area", "friction_scale"):
            setattr(self, name, ti.field(float, shape=1))
        self.node_area[0] = 2.0
        self.friction_scale[0] = 1.0
        for name in ("scale_device", "dt_device", "energy", "levelset_minimum_gap"):
            setattr(self, name, ti.field(float, shape=()))
        self.scale_device[None] = self.dt_device[None] = 1.0

    @ti.func
    def _sample_affine_levelset(self, body, coordinate):
        return coordinate[0], ti.Vector([1.0, 0.0, 0.0]), ti.Matrix.zero(float, 3, 3), True

    @ti.func
    def _body_pair_allowed(self, body_i, body_j):
        return True

    @ti.func
    def _pp_dhat(self, body_i, body_j):
        return 0.1

    @ti.func
    def _pp_mu(self, body_i, body_j):
        return 1.0

    @ti.func
    def _pp_kappa(self, body_i, body_j):
        return 1.0


@pytest.mark.parametrize("stretch", [0.5, 1.0, 2.0])
def test_levelset_sliding_friction_obeys_coulomb_limit_after_affine_deformation(taichi_runtime, stretch):
    operator = PlanarContact()
    target = np.vstack((np.zeros(3), stretch * np.eye(3)))
    point = stretch * np.array([0.3, 0.25, 0.25])
    source = np.vstack((point, point + np.eye(3)))
    frozen = np.vstack((source, target))
    current = frozen.copy()
    current[:4, 1] += 0.01
    operator.levelset_friction_y.from_numpy(frozen)
    operator.hat_y.from_numpy(frozen)
    operator.y.from_numpy(current)
    operator.x[0] = current[0]
    operator._assemble_levelset_contacts(False, 0)
    normal_force = operator.grad.to_numpy()[:4].sum(axis=0)
    operator.grad.fill(0.0)
    operator._assemble_levelset_lagged_friction(False, 0, False)
    friction_force = operator.grad.to_numpy()[:4].sum(axis=0)
    assert np.linalg.norm(friction_force) == pytest.approx(np.linalg.norm(normal_force), rel=1e-12)
