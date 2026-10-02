import numpy as np
import taichi as ti

from src.dem.structs.Geometry import _GeometryDeviceState


@ti.dataclass
class _PatchWall:
    vertice1: ti.types.vector(3, float)
    vertice2: ti.types.vector(3, float)
    vertice3: ti.types.vector(3, float)
    norm: ti.types.vector(3, float)
    v: ti.types.vector(3, float)
    verletDisp: ti.types.vector(3, float)
    contact_force: ti.types.vector(3, float)
    contact_torque: ti.types.vector(3, float)

    @ti.func
    def _get_center(self):
        return (self.vertice1 + self.vertice2 + self.vertice3) / 3.0


@ti.kernel
def _initialize_wall(wall: ti.template()):
    wall[0].vertice1 = ti.Vector([0.0, 0.0, 0.0])
    wall[0].vertice2 = ti.Vector([1.0, 0.0, 0.0])
    wall[0].vertice3 = ti.Vector([0.0, 1.0, 0.0])
    wall[0].norm = ti.Vector([0.0, 0.0, 1.0])
    wall[0].v = ti.Vector.zero(float, 3)
    wall[0].verletDisp = ti.Vector.zero(float, 3)
    wall[0].contact_force = ti.Vector.zero(float, 3)
    wall[0].contact_torque = ti.Vector.zero(float, 3)


def test_patch_rigid_body_step_keeps_resultant_and_state_on_device(
    taichi_runtime,
):
    wall = _PatchWall.field(shape=1)
    _initialize_wall(wall)
    vertices = np.asarray([[0.0, 0.0, 0.0]])
    state = _GeometryDeviceState(
        mass=2.0,
        mass_center=[0.0, 0.0, 0.0],
        orientation=np.eye(3),
        velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        inertia_body=np.eye(3),
        inertia_body_inv=np.eye(3),
        rotate_center=[0.0, 0.0, 0.0],
        external_force=[2.0, 0.0, 0.0],
        external_torque=[0.0, 0.0, 0.0],
        local_v1=vertices,
        local_v2=np.asarray([[1.0, 0.0, 0.0]]),
        local_v3=np.asarray([[0.0, 1.0, 0.0]]),
        local_norm=np.asarray([[0.0, 0.0, 1.0]]),
    )

    state.step_rigid(0, 1, 0.1, wall)

    np.testing.assert_allclose(state.velocity[None], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(state.mass_center[None], [0.01, 0.0, 0.0])
    np.testing.assert_allclose(wall.vertice1.to_numpy()[0], [0.01, 0.0, 0.0])
    np.testing.assert_allclose(wall.v.to_numpy()[0], [0.1, 0.0, 0.0])
