from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from tests.helpers.fem_contact_reference import ReferenceFEMContactAssembler as FEMContactAssembler
from src.fedem.contact.AffineIPCAssembler import FEMAffineIPCAssembler
from src.fedem.DifferentiableFEMAffine import DifferentiableFEMAffine
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_grad,
    point_triangle_distance_grad,
)
from src.physics_model.contact_model.ipc.IPC import ipc_toolkit_barrier_distance2_offset_terms


@ti.kernel
def _offset_barrier_energy(distance: ti.f64, dmin: ti.f64) -> ti.f64:
    active = (0.02 + 2.0 * dmin) * 0.02
    energy, first, second = ipc_toolkit_barrier_distance2_offset_terms(
        distance * distance, 0.02, dmin, 1.0 / (active * active), 0
    )
    return 0.02 * energy


def _scalar(value=0.0, shape=1, dtype=float):
    field = ti.field(dtype, shape=shape)
    field.fill(value)
    return field


def _positions(kind, distance):
    if kind == "pt":
        return np.array([[0.25, 0.25, distance], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -0.5, distance], [0.5, 0.5, distance]])


def _frames(assembler, kind, load_name):
    setattr(assembler, f"friction_{kind}_candidate", ti.Vector.field(4, int, shape=1))
    setattr(assembler, f"friction_{kind}_weight", ti.Vector.field(4, float, shape=1))
    setattr(assembler, f"friction_{kind}_normal", ti.Vector.field(3, float, shape=1))
    setattr(assembler, load_name, _scalar())


def _energy_normal_force(distance, dmin):
    step = 1.0e-8
    return -(_offset_barrier_energy(distance + step, dmin) - _offset_barrier_energy(distance - step, dmin)) / (2 * step)


@pytest.mark.parametrize("kind", ["pt", "ee"])
@pytest.mark.parametrize("dmin", [0.0, 0.01])
def test_fem_host_and_device_friction_load_matches_barrier_energy(taichi_runtime, kind, dmin):
    assembler = object.__new__(FEMContactAssembler)
    assembler.real_type = ti.f64
    assembler.contact = SimpleNamespace(model="IPC")
    assembler.dmin, assembler.dhat, assembler.kappa = dmin, 0.02, 1.0
    distance = dmin + 0.001
    data = _positions(kind, distance)
    assembler.reference_positions = data.copy()
    positions = ti.Vector.field(3, float, shape=4)
    positions.from_numpy(data)
    candidates = ti.Vector.field(4, int, shape=1)
    candidates[0] = [0, 1, 2, 3]
    assembler.collision_culling = SimpleNamespace(reference_position=positions)
    setattr(assembler.collision_culling, "point_triangle" if kind == "pt" else "edge_edge", candidates)
    setattr(
        assembler.collision_culling, "point_triangle_measure" if kind == "pt" else "edge_edge_measure", _scalar(1.0)
    )
    _frames(assembler, kind, f"friction_{kind}_normal_force")

    getattr(assembler, f"_freeze_{kind}_friction_device")(positions, 1)
    expected = _energy_normal_force(distance, dmin)
    assert getattr(assembler, f"friction_{kind}_normal_force")[0] == pytest.approx(expected, rel=1.0e-8)
    # Host candidate measures already contain dhat.
    host = assembler._frozen_normal_forces(data, np.array([[0, 1, 2, 3]]), np.array([0.02]), kind)
    assert host[0] == pytest.approx(expected, rel=1.0e-8)


@ti.kernel
def _cache_distance(assembler: ti.template(), kind: ti.template()):
    if ti.static(kind == "pt"):
        stencil = assembler.culling.point_triangle[0]
        distance2, gradient, feature = point_triangle_distance_grad(
            assembler.position[stencil[0]],
            assembler.position[stencil[1]],
            assembler.position[stencil[2]],
            assembler.position[stencil[3]],
        )
        assembler.pt_distance2[0] = distance2
        assembler.pt_distance_type[0] = feature
        for component in ti.static(range(12)):
            assembler.pt_gradient[0, component] = gradient[component]
    else:
        stencil = assembler.culling.edge_edge[0]
        distance2, gradient, feature = edge_edge_distance_grad(
            assembler.position[stencil[0]],
            assembler.position[stencil[1]],
            assembler.position[stencil[2]],
            assembler.position[stencil[3]],
        )
        assembler.ee_distance2[0] = distance2
        assembler.ee_distance_type[0] = feature
        for component in ti.static(range(12)):
            assembler.ee_gradient[0, component] = gradient[component]


def _fem_affine_contact(kind, dmin):
    assembler = object.__new__(FEMAffineIPCAssembler)
    assembler.real_type = ti.f64
    assembler.affine_body_count = assembler.fem_body_count = 1
    assembler.fem_node_count = 3 if kind == "pt" else 2
    assembler.affine = SimpleNamespace(node2body=_scalar(shape=2, dtype=int))
    assembler.fem_node_body = _scalar(shape=assembler.fem_node_count, dtype=int)
    data = _positions(kind, dmin + 0.001)
    order = [1, 2, 3, 0] if kind == "pt" else [2, 3, 0, 1]
    assembler.position = ti.Vector.field(3, float, shape=4)
    assembler.position.from_numpy(data[order])
    candidates = ti.Vector.field(4, int, shape=1)
    candidates[0] = np.argsort(order)
    assembler.culling = SimpleNamespace(reference_position=assembler.position)
    setattr(assembler.culling, "point_triangle" if kind == "pt" else "edge_edge", candidates)
    setattr(assembler.culling, "point_triangle_measure" if kind == "pt" else "edge_edge_measure", _scalar(1.0))
    for name, value in (("dmin", dmin), ("dhat", 0.02), ("kappa", 1.0), ("friction", 0.6), ("epsv", 0.001)):
        setattr(assembler, f"pair_{name}", _scalar(value, shape=(1, 1)))
    assembler.pair_active = _scalar(1, shape=(1, 1), dtype=int)
    setattr(assembler, f"{kind}_distance_type", _scalar(dtype=int))
    setattr(assembler, f"{kind}_distance2", _scalar())
    setattr(assembler, f"{kind}_gradient", _scalar(shape=(1, 12)))
    _frames(assembler, kind, f"friction_{kind}_coefficient")
    setattr(assembler, f"friction_{kind}_epsv", _scalar())
    assembler.friction_active_count = _scalar(shape=(), dtype=int)
    return assembler, data, order


def _freeze_affine(assembler, kind):
    _cache_distance(assembler, kind)
    getattr(assembler, "_freeze_point_triangle_friction" if kind == "pt" else "_freeze_edge_edge_friction")(1)
    return float(getattr(assembler, f"friction_{kind}_coefficient")[0])


@pytest.mark.parametrize("kind", ["pt", "ee"])
@pytest.mark.parametrize("dmin", [0.0, 0.01])
def test_fem_affine_friction_load_matches_barrier_energy(taichi_runtime, kind, dmin):
    assembler, _, _ = _fem_affine_contact(kind, dmin)
    coefficient = _freeze_affine(assembler, kind)
    assert coefficient == pytest.approx(0.6 * _energy_normal_force(dmin + 0.001, dmin), rel=1.0e-8)


@ti.data_oriented
class _AdjointProbe(DifferentiableFEMAffine):
    def __init__(self, assembler, kind, base_coefficient):
        self.engine = SimpleNamespace(contact=assembler)
        self.dt = 0.01
        self.kind = 0 if kind == "pt" else 1
        self.base_distance2 = float(getattr(assembler, f"{kind}_distance2")[0])
        self.base_coefficient = base_coefficient
        self.stencil = getattr(assembler.culling, "point_triangle" if kind == "pt" else "edge_edge")

    @ti.kernel
    def contraction(
        self,
        lagged: ti.types.matrix(4, 3, ti.f64),
        current: ti.types.matrix(4, 3, ti.f64),
        adjoint: ti.types.matrix(4, 3, ti.f64),
    ) -> ti.f64:
        return self._mixed_lagged_contraction(
            lagged,
            current,
            adjoint,
            self.stencil[0],
            0,
            self.kind,
            self.base_distance2,
            self.base_coefficient,
        )


@pytest.mark.parametrize("kind", ["pt", "ee"])
def test_fem_affine_lagged_adjoint_reconstructs_perturbed_friction_load(taichi_runtime, kind):
    assembler, lagged, order = _fem_affine_contact(kind, 0.01)
    probe = _AdjointProbe(assembler, kind, _freeze_affine(assembler, kind))
    source_count = 1 if kind == "pt" else 2
    lagged[:source_count, 2] += 3.0e-5
    current = lagged.copy()
    current[:source_count, 0] += 0.002
    adjoint = np.zeros((4, 3))
    adjoint[:source_count, 0] = 1.0
    contraction = probe.contraction(ti.Matrix(lagged), ti.Matrix(current), ti.Matrix(adjoint))
    assembler.position.from_numpy(lagged[order])
    expected = _freeze_affine(assembler, kind)
    assert contraction == pytest.approx(expected, rel=1.0e-10)
