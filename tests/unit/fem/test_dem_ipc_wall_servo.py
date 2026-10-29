"""DEM facet servo feedback stays on device and uses physical IPC reactions."""

from types import SimpleNamespace
import numpy as np
import pytest
import taichi as ti

from src.dem.engines.IPCWallBoundary import IPCWallBoundary
from src.dem.structs.BaseStruct import FacetFamily, ServoWall
from src.fem.contact.ContactModel import FEMContact
from src.fem.contact.DEMWallContact import DEMWallContactAssembler
from src.fem.engines.SparseMatrix import FEMTripletContribution
from src.fem.generator import FEMMesh

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fem,
    pytest.mark.ipc,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


def boundary():
    wall = FacetFamily.field(shape=2)
    for index, vertices in enumerate(([[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1], [0, 0, 1]])):
        wall[index].add_wall_geometry(
            0, *[ti.Vector(p) for p in vertices], ti.Vector([1.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        )
    servo = ServoWall.field(shape=1)
    servo[0].add_servo_wall(0, 2, 0.8, 0.1, 100.0, 0.02)
    scene = SimpleNamespace(wall=wall, servo=servo, wallNum=np.array([2]), servoNum=np.array([1]))
    sims = SimpleNamespace(wall_type=1, servo_type="StiffnessControl", servo_status="On", delta=0.01)
    return IPCWallBoundary(scene, sims)


def test_wall_servo_combines_fem_abd_loads_limits_speed_and_rolls_back(taichi_runtime, monkeypatch):
    b = boundary()
    b.abd_force.from_numpy(np.array([[-10.0, 0, 0], [-15.0, 0, 0]]))
    b.fem_force.from_numpy(np.array([[-10.0, 0, 0], [-15.0, 0, 0]]))
    b.abd_stiffness.fill(100.0)
    with monkeypatch.context() as patch:
        patch.setattr(type(b.abd_force), "to_numpy", lambda *_: pytest.fail("servo feedback read back a device vector"))
        b.update_servo_device(0.01, 1)
        b.begin_step_device(0.01)
        b.apply_motion_device()
        b.rollback_device()
    assert b.scene.servo[0].area == pytest.approx(1.0)
    assert b.scene.servo[0].current_force == pytest.approx(50.0)
    assert b.scene.servo[0].gain == pytest.approx(0.4)
    np.testing.assert_allclose(b.velocity.to_numpy()[:, 0], 0.02, rtol=1e-6)
    np.testing.assert_array_equal(b.vertex1.to_numpy(), b.start1.to_numpy())
    b.fem_force.fill(-100.0)
    b.update_servo_device(0.01, 1)
    assert b.scene.servo[0].pre_velocity == pytest.approx(-0.02, rel=1e-6)


@pytest.mark.parametrize("model, friction", [("IPC", 0.0), ("IPC", 0.4)])
def test_fem_facet_force_matches_energy_gradient_and_wall_reaction(taichi_runtime, model, friction):
    b = boundary()
    points = np.array([[0.01, 0.25, 0.25], [0.08, 0.25, 0.25], [0.08, 0.35, 0.25], [0.08, 0.25, 0.35]])
    mesh = FEMMesh(points, np.array([[0, 1, 2, 3]], dtype=np.int32), "TET4")
    contact = FEMContact(
        model=model,
        self_contact=False,
        dhat=0.1,
        kappa=100.0,
        friction_coefficient=friction,
        project_pd=False,
    )
    assembler = DEMWallContactAssembler(mesh, SimpleNamespace(), contact, b)
    positions = ti.Vector.field(3, ti.f64, shape=4)
    force = ti.Vector.field(3, ti.f64, shape=4)
    positions.from_numpy(points)
    assembler.begin_step_device(positions, 0.01)
    if friction:
        points = points + np.array([0.0, 0.001, 0.0])

    def evaluate(value):
        positions.from_numpy(value)
        force.fill(0.0)
        assembler.assemble_device(positions, force, FEMTripletContribution(12))
        return assembler._last_energy, force.to_numpy()

    energy, gradient = evaluate(points)
    delta = np.zeros_like(points)
    delta[0, 0] = 1e-7
    fd = (evaluate(points + delta)[0] - evaluate(points - delta)[0]) / 2e-7
    assert fd == pytest.approx(gradient[0, 0], rel=1e-6)
    evaluate(points)
    np.testing.assert_allclose(b.fem_force.to_numpy().sum(axis=0), gradient.sum(axis=0), rtol=1e-12)
    assert energy > 0 and gradient[0, 0] < 0
    assert b.fem_stiffness.to_numpy().sum() > 0
    diagonal = ti.Vector.field(9, ti.f64, shape=4)
    positions.from_numpy(points)
    assembler._copy_stiffness_positions(positions)
    assembler._scatter_walls_device(type("Target", (), {"diag": diagonal})(), 0, True)
    hessian = diagonal.to_numpy()[0].reshape(3, 3)
    for component in range(3):
        perturbation = np.zeros_like(points)
        perturbation[0, component] = 1e-7
        fd_column = (evaluate(points + perturbation)[1][0] - evaluate(points - perturbation)[1][0]) / 2e-7
        np.testing.assert_allclose(hessian[:, component], fd_column, rtol=1e-5, atol=1e-6)


def test_public_fem_wall_path_uses_device_newton_and_servo(taichi_runtime, tmp_path):
    from src.fem import FEM

    b = boundary()
    b.scene.ipc_wall_boundary = b
    dem = SimpleNamespace(scene=b.scene, sims=b.sims)
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    points = np.array([[0.01, 0.25, 0.25], [0.08, 0.25, 0.25], [0.08, 0.35, 0.25], [0.08, 0.25, 0.35]])
    fem.add_mesh(FEMMesh(points, np.array([[0, 1, 2, 3]], dtype=np.int32), "TET4"))
    fem.add_material("StVK", density=1000.0, young_modulus=1e4, poisson_ratio=0.3)
    fem.add_contact("IPC", dem_walls=dem, self_contact=False, dhat=0.1, kappa=100.0)
    fem.set_solver(dt=0.001, steps=2, assemble_type="HashTriplet", linear_solver="PCG")
    engine = fem.build()
    before = b.vertex1.to_numpy().copy()
    result = engine.run(steps=2)
    assert result.converged
    assert engine.step_count == 2
    assert b.scene.servo[0].current_force > 0
    assert np.any(b.vertex1.to_numpy() != before)
    np.testing.assert_allclose(b.scene.wall.vertice1.to_numpy(), b.vertex1.to_numpy(), atol=1e-8)
