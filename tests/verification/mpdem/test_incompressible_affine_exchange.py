from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GV
from src.mpdem.fluid_dynamics.IncompressibleCoupling import IncompressibleAffineBodyCoupling
from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.engines.EngineKernel import kernel_correct_velocity_coupled_3d
from src.mpm.structs.StaggeredGrid import StaggeredGrid

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


@ti.data_oriented
class Fluid:
    density = 1.0
    viscosity = 0.0
    atmospheric_pressure = 0.0


@ti.data_oriented
class Cube:
    body_num = 1
    vertex_num = 8
    levelset_contact = True

    def __init__(self, stretch):
        self.y = ti.Vector.field(3, float, shape=4)
        self.y.from_numpy(np.vstack((np.zeros(3), stretch * np.eye(3))) + 1.0)
        self.velocity_y = ti.Vector.field(3, float, shape=4)
        self.body_mass = ti.field(float, shape=1)
        self.body_mass[0] = 1.0
        self.volume = ti.field(float, shape=1)
        self.volume[0] = 1.0
        self.body_scale = ti.field(float, shape=1)
        self.body_scale[0] = 1.0
        self.external_generalized_force = ti.Vector.field(3, float, shape=4)
        self.x = ti.Vector.field(3, float, shape=8)
        self.node2body = ti.field(int, shape=8)

    @ti.func
    def _affine_levelset_matrix(self, body):
        return ti.Matrix.cols([self.y[1] - self.y[0], self.y[2] - self.y[0], self.y[3] - self.y[0]])

    @ti.func
    def _sample_affine_levelset(self, body, point):
        q = ti.abs(point) - 0.5
        value = ti.max(q, 0.0).norm() + ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0)
        return value, ti.Vector.zero(float, 3), ti.Matrix.zero(float, 3, 3), True

    def _reconstruct_vertices(self):
        controls = self.y.to_numpy()
        vertices = np.array(list(np.ndindex(2, 2, 2)), dtype=float) - 0.5
        material = vertices + 0.25
        basis = np.column_stack((1 - material.sum(axis=1), material))
        self.x.from_numpy(basis @ controls)

    def device_clear_external_generalized_force(self):
        self.external_generalized_force.fill(0.0)


@ti.data_oriented
class GridOccupancyCoupling(IncompressibleAffineBodyCoupling):
    @ti.func
    def _cell_body_fraction(self, cell, grid_size, body):
        return self.solid_fraction[cell]


def build(stretch=2.0, active=3, coupling_type=IncompressibleAffineBodyCoupling):
    GV.DIMENSION = 3
    cnum = ti.Vector([active + 2] * 3)
    shape = tuple(active + 2 for _ in range(3))
    node = StaggeredGrid(SimpleNamespace(dimension=3, coupling=False), list(shape), 1)
    cell = SimpleNamespace()
    for name in ("pressure", "surface_tension", "fluid_sdf"):
        setattr(cell, name, ti.field(float, shape=shape, offset=(-1, -1, -1)))
    cell.type = ti.field(int, shape=shape, offset=(-1, -1, -1))
    cell.type.fill(2)
    for index in np.ndindex(active, active, active):
        cell.type[index] = 1
    scene = SimpleNamespace(
        node=node, element=SimpleNamespace(cnum=cnum, ghost_cell=1, grid_size=ti.Vector([1.0, 1.0, 1.0]), cell=cell)
    )
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    sims = SimpleNamespace(dimension=3, dt=dt, set_fluid_level_set=lambda value: None)
    viscous_delta = [ti.field(float, shape=field.shape, offset=(-1, -1, -1)) for field in node.velocity]
    engine = SimpleNamespace(
        configure_runtime_functions=lambda *args: None,
        get_single_fluid_mat_props=lambda scene: Fluid(),
        use_pressure_free_surface_theta=False,
        coupled_viscous_delta=viscous_delta,
    )
    cube = Cube(stretch)
    coupling = coupling_type(sims, scene, engine, SimpleNamespace(operator=cube))
    coupling.attach()
    return coupling, cube, sims, scene


@pytest.mark.parametrize("stretch", [1.0, 2.0])
def test_affine_ibm_density_uses_current_volume(taichi_runtime, stretch):
    coupling, cube, sims, scene = build(stretch)
    coupling.update_solid_cells(sims, scene)
    density = coupling.solid_density.to_numpy()
    occupied = coupling.solid_fraction.to_numpy() > 1e-12
    assert occupied.any()
    np.testing.assert_allclose(density[occupied], 1.0 / stretch**3, atol=1e-12)


@pytest.mark.parametrize("first_edge", [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
def test_affine_ibm_rejects_reflected_or_singular_frames(taichi_runtime, first_edge):
    coupling, cube, sims, scene = build()
    cube.y[1] = cube.y[0] + np.asarray(first_edge)
    with pytest.raises(RuntimeError, match="positive determinant"):
        coupling.update_solid_cells(sims, scene)


def test_uniform_pressure_has_affine_virtual_work_and_zero_resultant(taichi_runtime):
    coupling, cube, sims, scene = build()
    coupling.update_solid_cells(sims, scene)
    scene.element.cell.pressure.fill(2.0)
    coupling.capture_viscous_stress(sims, scene)
    coupling.accumulate_pressure_force(sims, scene)
    generalized = cube.external_generalized_force.to_numpy()
    occupied_volume = coupling.solid_fraction.to_numpy()[1:-1, 1:-1, 1:-1].sum()
    expected_moment = -2.0 * occupied_volume * 0.5 * np.eye(3)
    np.testing.assert_allclose(generalized[1:].T, expected_moment, atol=1e-12)
    np.testing.assert_allclose(generalized.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(cube.y.to_numpy(), generalized).sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("stretch", [1.0, 2.0, 0.8])
@pytest.mark.parametrize("fraction", [0.5, 1.0e-7])
def test_pressure_projection_and_affine_feedback_conserve_total_impulse(taichi_runtime, stretch, fraction):
    coupling, cube, sims, scene = build(stretch, active=5, coupling_type=GridOccupancyCoupling)
    coupling.body_bbox_min[0] = [-1.0, -1.0, -1.0]
    coupling.body_bbox_max[0] = [6.0, 6.0, 6.0]
    coupling.solid_fraction[2, 2, 2] = fraction
    coupling.solid_fraction_sum[2, 2, 2] = fraction
    coupling.solid_density[2, 2, 2] = 1.0 / stretch**3
    scene.element.cell.pressure[3, 2, 2] = 2.0
    dummy = coupling.solid_fraction
    kernel_correct_velocity_coupled_3d(
        1,
        scene.element.cnum,
        scene.element.grid_size,
        sims.dt,
        Fluid(),
        scene.element.cell.pressure,
        scene.element.cell.surface_tension,
        scene.element.cell.type,
        scene.element.cell.fluid_sdf,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        coupling.solid_fraction,
        coupling.solid_density,
        2,
        False,
        False,
        scene.node,
    )
    coupling.accumulate_pressure_force(sims, scene)
    fluid_impulse = np.zeros(3)
    for index in np.ndindex(5, 5, 5):
        for d in range(3):
            right = list(index)
            right[d] += 1
            velocity = 0.5 * (scene.node.velocity[d][index] + scene.node.velocity[d][tuple(right)])
            fluid_impulse[d] += (1.0 - coupling.solid_fraction[index]) * velocity
    body_impulse = cube.external_generalized_force.to_numpy().sum(axis=0) * sims.dt[None]
    np.testing.assert_allclose(fluid_impulse + body_impulse, 0.0, atol=1e-12)


def test_affine_viscous_stress_is_captured_before_velocity_changes(taichi_runtime):
    coupling, cube, sims, scene = build()
    coupling.update_solid_cells(sims, scene)
    for index in np.ndindex(scene.node.velocity[0].shape):
        face = tuple(component - 1 for component in index)
        scene.node.velocity[0][face] = 2.5 * (face[1] + 0.5)
    coupling._capture_viscous_stress(scene.element.cnum - 2, scene.element.grid_size, 0.4, scene.node)
    for velocity in scene.node.velocity:
        velocity.fill(0.0)
    coupling.accumulate_pressure_force(sims, scene)
    stress = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    occupied_volume = coupling.solid_fraction.to_numpy()[1:-1, 1:-1, 1:-1].sum()
    generalized = cube.external_generalized_force.to_numpy()
    np.testing.assert_allclose(generalized[1:].T, 0.5 * occupied_volume * stress, atol=1e-12)
    np.testing.assert_allclose(generalized.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(cube.y.to_numpy(), generalized).sum(axis=0), 0.0, atol=1e-12)


def test_affine_viscous_feedback_uses_applied_mac_increment(taichi_runtime):
    coupling, cube, sims, scene = build()
    coupling.update_solid_cells(sims, scene)
    coupling.mengine.coupled_viscous_delta[0].fill(0.2)
    coupling.accumulate_pressure_force(sims, scene)
    occupied_volume = coupling.solid_fraction.to_numpy()[1:-1, 1:-1, 1:-1].sum()
    expected_force = [occupied_volume * 0.125 * 0.2 / sims.dt[None], 0.0, 0.0]
    np.testing.assert_allclose(cube.external_generalized_force.to_numpy().sum(axis=0), expected_force, atol=1e-12)


@pytest.mark.parametrize("backend", ["pcg", "mgpcg"])
def test_affine_feedback_occurs_before_g2p_and_density_reclassification(backend):
    engine = IncompressibleEngine.__new__(IncompressibleEngine)
    events = []
    names = (
        "reset_solid_face_velocity_step",
        "update_external_cut_cell_boundary",
        "update_external_ibm_fields",
        "validate_ibm_cut_cell_geometry",
        "compute_nodal_kinematics",
        "run_pressure_fluid_level_set",
        "run_cut_cell",
        "compute_grid_velcity_gravity",
        "run_coupled_viscosity",
        "run_external_fluid_particle_coupling",
        "run_ibm_source",
        "enforce_mac_boundary_step",
        "prepare_poisson_equations_step",
        "prepare_poisson_equations_mgpcg_step",
        "require_pressure_solver_success",
        "apply_pressures_step",
        "apply_pressures_mgpcg_step",
        "enforce_mac_boundary_after_pressure_step",
        "accumulate_external_fluid_force",
        "compute_particle_kinematics",
        "run_density_projection",
        "run_particle_shifting",
        "enforce_particle_boundary",
    )
    for name in names:
        setattr(engine, name, lambda *args, _name=name, **kwargs: events.append(_name))
    engine.find_active_nodes = lambda *args: 1
    engine.poisson_solver = SimpleNamespace(solve=lambda *args, **kwargs: True)
    for name in ("operator", "right_hand_vector", "unknow_vector", "diag_A"):
        setattr(engine, name, None)
    timer = SimpleNamespace(begin=lambda *args: None, end=lambda *args: None)
    sims = SimpleNamespace(timer=timer, residual_tolerance=1e-8, linear_solver_relative_tolerance=1e-8, iter_max=10)
    getattr(engine, "fdm_discretization_" + backend)(sims, SimpleNamespace())
    feedback = events.index("accumulate_external_fluid_force")
    assert (
        events.index("enforce_mac_boundary_after_pressure_step")
        < feedback
        < events.index("compute_particle_kinematics")
    )
    assert feedback < events.index("run_density_projection")
    assert events.count("accumulate_external_fluid_force") == 1
