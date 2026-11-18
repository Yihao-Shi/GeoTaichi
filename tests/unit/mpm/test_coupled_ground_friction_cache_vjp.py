"""Ground cache VJP, including transfer curvature and the zero-mu limit."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [
    pytest.mark.unit,
    pytest.mark.mpdem,
    pytest.mark.ipc,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


@ti.kernel
def _force_contraction(ipc: ti.template(), point: ti.types.vector(3, ti.f64)) -> ti.f64:
    gap = ipc.ground.distance(0, point)
    _, first, _ = ipc._normal_terms(gap)
    tangent = ipc.ground.tangent_operator(0)
    velocity = tangent @ (ipc.mpm.p_temp[0] - ipc.hat_x[0]) / ipc.mpm.dt
    adjoint = ti.Vector.zero(ti.f64, 3)
    for local in range(2):
        for c in ti.static(range(3)):
            adjoint[c] += ipc.mpm.shape[0, local] * ipc.mpm.incre_resolution[3 * local + c]
    return (
        ipc.mpm.surface_measure[0] * first * ipc.friction.grad_term(velocity.norm()) * adjoint.dot(tangent @ velocity)
    )


def test_coupled_ground_cache_and_zero_coefficient_vjp():
    from src.mpm.generator.Ground import Ground
    from src.mpm.soft_particle.IPCMPM import IPCMPM
    from src.physics_model.contact_model.ipc.Barrier import Barrier
    from src.physics_model.contact_model.ipc.Friction import Friction

    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=1)
    ipc = object.__new__(IPCMPM)
    mpm = SimpleNamespace(dt=0.001)
    ipc.mpm = mpm
    ipc.ground = Ground()
    ipc.ground.append([0, 0, 0], [0, 0, 1])
    ipc.ground.finalize()
    ipc.barrier = Barrier(dhat=0.06, dmin=0.001, kappa=1e4)
    ipc.friction = Friction(mu=0.2, epsv=0.1)
    mpm.particle = ti.Struct.field({"x": ti.types.vector(3, ti.f64)}, shape=1)
    mpm.surface_id = ti.field(ti.i32, shape=1)
    mpm.surface_measure = ti.field(ti.f64, shape=1)
    mpm.offset = ti.field(ti.i32, shape=1)
    mpm.LnID = ti.field(ti.i32, shape=(1, 2))
    mpm.node2dof = ti.field(ti.i32, shape=2)
    mpm.shape = ti.field(ti.f64, shape=(1, 2))
    mpm.dshape = ti.Vector.field(3, ti.f64, shape=(1, 2))
    mpm.incre_resolution = ti.field(ti.f64, shape=6)
    mpm.p_temp = ti.Vector.field(3, ti.f64, shape=1)
    ipc.hat_x = ti.Vector.field(3, ti.f64, shape=1)
    ipc.particle_position_vjp = ti.Vector.field(3, ti.f64, shape=1)
    ipc.friction_parameter_vjp = ti.Vector.field(4, ti.f64, shape=())
    ipc.gfrictionNum = ti.field(ti.i32, shape=1)
    ipc.gfriction = ti.Struct.field({"surfaceID": ti.i32, "wallID": ti.i32}, shape=1)
    tape = ti.field(ti.f64, shape=(1, 6))
    rhs = ti.field(ti.f64, shape=6)
    point = np.array([0.4, 0.5, 0.04])
    weights = np.array([0.4, 0.6])
    shape_gradient = np.array([[1.0, -0.3, 0.7], [-1.0, 0.3, -0.7]])
    displacement = np.array([[1e-4, -2e-4, 3e-4], [-2e-4, 1e-4, -1e-4]])
    rng = np.random.default_rng(831)
    point_direction, grid_direction = rng.normal(size=3), rng.normal(size=(2, 3))
    mpm.particle[0].x = point
    mpm.surface_measure[0], mpm.offset[0], ipc.gfrictionNum[0] = 0.01, 2, 1
    mpm.LnID.from_numpy(np.array([[0, 1]], dtype=np.int32))
    mpm.node2dof.from_numpy(np.array([1, 2], dtype=np.int32))
    mpm.shape.from_numpy(weights.reshape(1, 2))
    mpm.dshape.from_numpy(shape_gradient.reshape(1, 2, 3))
    mpm.incre_resolution.from_numpy(rng.normal(size=6))
    tape.from_numpy(displacement.reshape(1, 6))
    ipc.hat_x[0] = point
    for slip in (0.0, 1e-5, 1e-3):
        mpm.p_temp[0] = point + [slip, 0.3 * slip, 0.0]
        cache_point = point + weights @ displacement
        for mu in (0.2, 0.0):
            ipc.friction.mu[0] = mu
            rhs.fill(0.0)
            ipc.particle_position_vjp.fill(0.0)
            ipc.friction_parameter_vjp.fill(0.0)
            ipc.pullback_ground_friction_cache(tape, 0, rhs, 0)
            analytic = np.dot(ipc.particle_position_vjp.to_numpy()[0], point_direction) + np.dot(
                rhs.to_numpy(), grid_direction.ravel()
            )
            for step in (1e-6, 2e-7):

                def objective(shift):
                    source = point + shift * point_direction
                    shape = weights + shift * shape_gradient @ point_direction
                    position = source + shape @ (displacement + shift * grid_direction)
                    return mu * _force_contraction(ipc, position)

                numerical = (objective(step) - objective(-step)) / (2 * step)
                assert analytic == pytest.approx(numerical, rel=2e-6, abs=1e-10)
            if mu == 0:
                assert ipc.friction_parameter_vjp[None][0] == pytest.approx(
                    _force_contraction(ipc, cache_point), rel=1e-12, abs=1e-12
                )
    ti.reset()
