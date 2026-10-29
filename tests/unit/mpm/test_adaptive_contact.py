"""Numerical contact checks independent of a full MPM solve."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.Contact import DEMContact, GeoContact
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.engines.EngineKernel import kernel_calc_geocontact, kernel_calc_demcontact_2D

pytestmark = pytest.mark.isolated_dimension(2)


@ti.dataclass
class ContactNode:
    m: float
    momentum: ti.types.vector(2, float)
    grad_domain: ti.types.vector(2, float)
    contact_pos: ti.types.vector(2, float)
    contact_length: float
    contact_force: ti.types.vector(2, float)

    @ti.func
    def _update_contact_force(self, force):
        self.contact_force += force


class Element:
    adaptive = True


def test_geocontact_rejects_a_silently_ignored_cutoff_keyword():
    with pytest.raises(ValueError, match="cut_off"):
        GeoContact({"cutoff": 0.8})
    assert GeoContact({"cut_off": 0.8}).cut_off == 0.8


@pytest.mark.parametrize("rigid_body", [0, 1])
def test_geocontact_scales_rigid_geometry_with_its_own_support(taichi_runtime, rigid_body):
    nodes = ContactNode.field(shape=(1, 2))
    dt = ti.field(float, shape=())
    dt[None] = 1e-5
    rigid = ti.field(int, shape=2)
    rigid[rigid_body] = 1
    soil_body = 1 - rigid_body
    element = Element()
    element.fine_grid_size = ti.Vector([0.1, 0.1])
    element.compact_to_logical = ti.field(int, shape=1)
    element.compact_to_logical[0] = 9904  # Physical (0.6, 9.8).
    for body in (0, 1):
        nodes[0, body].m = 1.0
        nodes[0, body].contact_pos = [0.6, 9.8]
    nodes[0, soil_body].contact_length = 0.1
    # A rigid GIMP stencil contains two thirds fine and one third coarse mass.
    rigid_length = (2 * 0.1 + 0.2) / 3
    nodes[0, rigid_body].contact_length = rigid_length
    nodes[0, rigid_body].contact_pos = [0.6, 9.94580833824823]
    nodes[0, rigid_body].grad_domain = [0.0, -1.0]
    nodes[0, soil_body].momentum = [0.0, 0.001]
    kernel_calc_geocontact(
        1e-10,
        0.0,
        0.78,
        2.5,
        1.5,
        ti.Vector([101, 121]),
        ti.Vector([0.2, 0.2]),
        dt,
        rigid,
        nodes,
        element,
        None,
    )
    distance = abs(2 * (0.14580833824823 / (1.25 * rigid_length)) ** 0.58 - 1)
    factor = (1 - 0.78 * distance**2.5) / (1 + 0.78 * distance**2.5)
    expected = np.array([0.0, factor * 0.001 / 1e-5])
    assert factor > 0
    np.testing.assert_allclose(nodes.contact_force.to_numpy()[0, rigid_body], expected, rtol=1e-5)
    np.testing.assert_allclose(nodes.contact_force.to_numpy().sum(axis=1), 0.0, atol=1e-8)
    nodes.contact_force.fill(0)
    nodes[0, rigid_body].contact_pos = [0.6, 9.97]
    kernel_calc_geocontact(
        1e-10,
        0.0,
        0.78,
        2.5,
        1.5,
        ti.Vector([101, 121]),
        ti.Vector([0.2, 0.2]),
        dt,
        rigid,
        nodes,
        element,
        None,
    )
    np.testing.assert_array_equal(nodes.contact_force.to_numpy(), 0.0)


def test_geocontact_uses_compact_coordinates_local_length_and_sticking_force(taichi_runtime):
    nodes = ContactNode.field(shape=(2, 2))
    dt = ti.field(float, shape=())
    dt[None] = 0.01
    rigid = ti.field(int, shape=2)
    rigid[1] = 1
    element = Element()
    element.fine_grid_size = ti.Vector([0.05, 0.05])
    element.compact_to_logical = ti.field(int, shape=2)
    element.compact_to_logical[0] = 12  # Physical (0.1, 0.1), unrelated to compact id 0.
    for body in (0, 1):
        nodes[0, body].m = 1.0
        nodes[0, body].contact_length = 0.05
    nodes[0, 0].contact_pos = [0.08, 0.1]
    nodes[0, 1].contact_pos = [0.12, 0.1]
    nodes[0, 1].grad_domain = [-1.0, 0.0]
    nodes[0, 0].momentum = [1.0, 0.01]
    kernel_calc_geocontact(
        1e-10, 0.3, 0.5, 2.0, 1.0, ti.Vector([5, 5]), ti.Vector([0.1, 0.1]), dt, rigid, nodes, element, None
    )
    dist = abs(2.0 * (0.02 / (1.25 * 0.05)) ** 0.58 - 1.0)
    factor = (1.0 - 0.5 * dist**2) / (1.0 + 0.5 * dist**2)
    np.testing.assert_allclose(nodes.contact_force.to_numpy()[0, 1], [factor / 0.01, 1.0], rtol=1e-6)
    np.testing.assert_allclose(nodes.contact_force.to_numpy().sum(axis=1), 0.0, atol=1e-8)
    nodes.contact_force.fill(0)
    nodes[0, 1].contact_pos = [0.14, 0.1]
    kernel_calc_geocontact(
        1e-10, 0.3, 0.5, 2.0, 1.0, ti.Vector([5, 5]), ti.Vector([0.1, 0.1]), dt, rigid, nodes, element, None
    )
    np.testing.assert_array_equal(nodes.contact_force.to_numpy(), 0.0)


def test_demcontact_refined_radius_and_tangential_history(taichi_runtime):
    particle = ti.Struct.field(
        dict(
            active=int,
            x=ti.types.vector(2, float),
            v=ti.types.vector(2, float),
            m=float,
            bodyID=int,
            contact_traction=ti.types.vector(2, float),
        ),
        shape=2,
    )
    particle[0].active = 1
    particle[0].x = [0.0, 0.06]
    particle[0].m = 1.0
    mapping = ti.field(int, shape=2)
    size = ti.field(int, shape=2)
    size[0] = 1
    lnid = ti.field(int, shape=2)
    shape = ti.field(float, shape=2)
    shape[0] = 1.0
    nodes = ContactNode.field(shape=(2, 1))
    polygon = ti.Vector.field(2, float, shape=4)
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
    polygon.from_numpy(vertices)
    element = Element()
    element.coarse_grid_size = ti.Vector([0.1, 0.1])
    element.particle_refined = ti.field(int, shape=2)
    material = Element()
    material.kn, material.kt, material.friction = 100.0, 50.0, 0.3
    dt = ti.field(float, shape=())
    dt[None] = 0.001

    def contact():
        nodes.contact_force.fill(0)
        kernel_calc_demcontact_2D(
            1,
            0,
            1,
            ti.Vector([2, 2]),
            ti.Vector([0.1, 0.1]),
            ti.Vector([0.0, 0.0]),
            particle,
            mapping,
            material,
            dt,
            polygon,
            lnid,
            shape,
            size,
            nodes,
            element,
        )

    contact()
    np.testing.assert_allclose(particle[0].contact_traction, [0.0, 1.5], atol=1e-6)
    element.particle_refined[0] = 1
    contact()
    np.testing.assert_array_equal(particle[0].contact_traction, [0.0, 0.0])
    particle[0].x = [0.0, 0.02]
    particle[0].contact_traction = [0.1, 999.0]
    contact()
    np.testing.assert_allclose(particle[0].contact_traction, [0.1, 1.75], atol=1e-6)
    np.testing.assert_array_equal(polygon.to_numpy(), vertices)


@pytest.mark.parametrize(
    "parameters",
    [
        {"materialID": 1, "stiffness": [100.0, 50.0]},
        {"params": {"materialID": 1, "stiffness": [100.0, 50.0]}},
        {"params": [{"materialID": 1}, {"materialID": 2}]},
        {"first": {"materialID": 1}, "second": {"materialID": 2}},
    ],
)
def test_dem_contact_parameters_are_flat(parameters):
    contact = DEMContact(parameters)
    assert all(isinstance(value, dict) for value in contact.contact_phys)
    assert contact.contact_phys[0]["materialID"] == 1


def test_adaptive_contact_validation(taichi_runtime):
    sims = Simulation()
    sims.dimension, sims.solver_type, sims.contact_detection = 2, "Explicit", "MPMContact"
    scene = myScene()
    result = scene.configure_adaptive_grid_for_solver(sims, {"RefineRatio": 1.0})
    assert result["HangingConstraintMode"] == "ShapeFunction"
    assert result["HangingShapeCapacityFactor"] >= 4
    for options in ({"HangingConstraintMode": "Penalty"}, {"BridgingDomain": True}):
        with pytest.raises(RuntimeError, match="ShapeFunction"):
            scene.configure_adaptive_grid_for_solver(sims, options)


@pytest.mark.parametrize("total_lagrangian", [False, True])
def test_geocontact_accepts_a_rigid_body_beyond_first_pair(taichi_runtime, total_lagrangian):
    from types import SimpleNamespace
    from src.mpm.engines.Engine import Engine
    from src.mpm.engines.TLExplicitEngine import TLExplicitEngine

    engine_type = TLExplicitEngine if total_lagrangian else Engine
    engine = object.__new__(engine_type)
    rigid = ti.field(int, shape=3)
    scene = SimpleNamespace(is_rigid=rigid)
    sims = SimpleNamespace(contact_detection="GeoContact")
    with pytest.raises(RuntimeError, match="soil-structure"):
        engine.valid_contact(sims, scene)
    rigid[2] = 1
    engine.valid_contact(sims, scene)


@pytest.mark.parametrize("levels", [[2, 1], [1, 2]])
def test_migrating_split_particles_preserve_finest_level(taichi_runtime, levels):
    from src.mpm.elements.AdaptiveQuadrilateralKernel import mark_refined_cells_migrated_particles_2d

    particles = ti.Struct.field(dict(active=int, materialID=int, x=ti.types.vector(2, float)), shape=2)
    refined = ti.field(ti.u8, shape=2)
    current = ti.field(ti.u8, shape=4)
    seeds = ti.field(ti.u8, shape=4)
    for i, level in enumerate(levels):
        particles[i].active = 1
        particles[i].materialID = i + 1
        particles[i].x = [0.75, 0.75]
        refined[i] = level
    mark_refined_cells_migrated_particles_2d(
        ti.Vector([2.0, 2.0]), ti.Vector([2, 2]), 2, particles, current, refined, seeds
    )
    assert seeds[3] == 2
