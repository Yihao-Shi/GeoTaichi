import numpy as np
import pytest


def _make_projection_fields(ti, local_points, local_velocities, *, scale, center,
                            body_velocity, angular_velocity, quaternion, gnum,
                            xmin_unscaled, spacing_unscaled):
    point_count = local_points.shape[0]
    grid_count = int(np.prod(gnum))
    vec3 = ti.types.vector(3, ti.f64)
    vec3i = ti.types.vector(3, ti.i32)
    vec4 = ti.types.vector(4, ti.f64)
    soft_type = ti.types.struct(
        bodyID=ti.i32,
        startIndex=ti.i32,
        gridStart=ti.i32,
        gridNum=ti.i32,
        mpmGridStart=ti.i32,
        templatePointStart=ti.i32,
        scale=ti.f64,
        v=vec3,
        referenceRotation=ti.types.matrix(3, 3, ti.f64),
    )
    point_type = ti.types.struct(
        active=ti.i32,
        bodyID=ti.i32,
        x=vec3,
        v=vec3,
        m=ti.f64,
        F=ti.types.matrix(3, 3, ti.f64),
    )
    soft_grid_type = ti.types.struct(m=ti.f64, v=vec3)
    rigid_type = ti.types.struct(
        softID=ti.i32,
        mass_center=vec3,
        q=vec4,
        w=vec3,
    )
    box_type = ti.types.struct(
        xmin=vec3,
        xmax=vec3,
        gnum=vec3i,
        grid_space=ti.f64,
    )

    soft = soft_type.field(shape=1)
    point = point_type.field(shape=point_count)
    soft_grid = soft_grid_type.field(shape=3)
    shape_node = ti.field(ti.i32, shape=(point_count, 3))
    dshape = ti.Vector.field(3, ti.f64, shape=(point_count, 3))
    shape_count = ti.field(ti.i32, shape=point_count)
    rigid = rigid_type.field(shape=1)
    box = box_type.field(shape=1)
    velocity = ti.Vector.field(3, ti.f64, shape=grid_count)
    weight = ti.field(ti.f64, shape=grid_count)
    max_support_loss = ti.field(ti.f64, shape=())

    qx, qy, qz, qw = quaternion
    rotation = np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )
    relative = scale * (local_points @ rotation.T)
    global_points = center + relative
    global_velocities = (
        body_velocity
        + np.cross(np.broadcast_to(angular_velocity, relative.shape), relative)
        + scale * (local_velocities @ rotation.T)
    )

    soft.bodyID[0] = 0
    soft.startIndex[0] = 0
    soft.gridStart[0] = 0
    soft.gridNum[0] = grid_count
    soft.mpmGridStart[0] = 0
    soft.templatePointStart[0] = 0
    soft.scale[0] = scale
    soft.v[0] = body_velocity
    soft.referenceRotation[0] = rotation
    rigid.softID[0] = 0
    rigid.mass_center[0] = center
    rigid.q[0] = quaternion
    rigid.w[0] = angular_velocity
    box.xmin[0] = scale * xmin_unscaled
    box.xmax[0] = scale * (
        xmin_unscaled + (np.asarray(gnum) - 1) * spacing_unscaled
    )
    box.gnum[0] = gnum
    box.grid_space[0] = scale * spacing_unscaled
    affine_fit = np.linalg.lstsq(
        np.column_stack((local_points, np.ones(point_count))),
        local_velocities,
        rcond=None,
    )[0]
    local_gradient = affine_fit[:3].T
    frame_spin = np.array(
        [
            [0.0, -angular_velocity[2], angular_velocity[1]],
            [angular_velocity[2], 0.0, -angular_velocity[0]],
            [-angular_velocity[1], angular_velocity[0], 0.0],
        ],
        dtype=np.float64,
    )
    global_gradient = (
        frame_spin + rotation @ local_gradient @ rotation.T
    )
    template_f_rate = scale * global_gradient @ rotation
    for node in range(3):
        soft_grid.m[node] = 1.0
        soft_grid.v[node] = template_f_rate[:, node]
    for p in range(point_count):
        point.active[p] = 1
        point.bodyID[p] = 0
        point.x[p] = global_points[p]
        point.v[p] = global_velocities[p]
        point.m[p] = 1.0
        point.F[p] = np.eye(3)
        shape_count[p] = 3
        for node in range(3):
            shape_node[p, node] = node
            basis = np.zeros(3)
            basis[node] = 1.0
            dshape[p, node] = basis

    return (
        soft,
        point,
        rigid,
        box,
        velocity,
        weight,
        max_support_loss,
        soft_grid,
        shape_node,
        dshape,
        shape_count,
    )


def _run_projection(ti, fields, shape_function_type):
    from src.mpm.soft_particle.SoftBodyKernel import (
        normalize_soft_levelset_velocity_projection_,
        project_current_soft_points_to_levelset_,
        reset_soft_levelset_velocity_projection_,
    )

    (
        soft,
        point,
        rigid,
        box,
        velocity,
        weight,
        max_support_loss,
        soft_grid,
        shape_node,
        dshape,
        shape_count,
    ) = fields
    point_count = point.shape[0]
    grid_count = velocity.shape[0]
    reset_soft_levelset_velocity_projection_(
        1, grid_count, soft, velocity, weight
    )
    project_current_soft_points_to_levelset_(
        point_count,
        shape_function_type,
        soft,
        point,
        soft_grid,
        shape_node,
        dshape,
        shape_count,
        velocity,
        weight,
        max_support_loss,
        rigid,
        box,
    )
    normalize_soft_levelset_velocity_projection_(
        1, grid_count, soft, velocity, weight
    )
    ti.sync()


def test_current_point_projection_removes_rigid_frame_velocity(
    taichi_runtime,
):
    ti = taichi_runtime
    coordinates = np.array(
        [
            [x, y, z]
            for z in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for x in (-1.0, 0.0, 1.0)
        ],
        dtype=np.float64,
    )
    angle = np.deg2rad(37.0)
    fields = _make_projection_fields(
        ti,
        coordinates,
        np.zeros_like(coordinates),
        scale=1.7,
        center=np.array([2.5, -1.25, 0.8]),
        body_velocity=np.array([0.7, -0.2, 1.1]),
        angular_velocity=np.array([0.3, -0.4, 0.8]),
        quaternion=np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]),
        gnum=np.array([7, 7, 7], dtype=np.int32),
        xmin_unscaled=np.array([-3.0, -3.0, -3.0]),
        spacing_unscaled=1.0,
    )

    _run_projection(ti, fields, shape_function_type=1)
    velocity = fields[4].to_numpy()
    weight = fields[5].to_numpy()
    active = weight > 1.0e-14

    assert np.count_nonzero(active) > 0
    assert np.max(np.linalg.norm(velocity[active], axis=1)) < 2.0e-14
    assert float(fields[6][None]) < 2.0e-14


@pytest.mark.parametrize("shape_function_type", [0, 1, 2])
def test_current_point_projection_reproduces_affine_velocity_interior(
    taichi_runtime,
    shape_function_type,
):
    ti = taichi_runtime
    coordinates = np.array(
        [
            [x, y, z]
            for z in range(-3, 4)
            for y in range(-3, 4)
            for x in range(-3, 4)
        ],
        dtype=np.float64,
    )
    gradient = np.array(
        [
            [0.17, -0.08, 0.05],
            [0.03, -0.11, 0.09],
            [-0.04, 0.06, 0.13],
        ],
        dtype=np.float64,
    )
    local_velocity = coordinates @ gradient.T
    angle = np.deg2rad(23.0)
    fields = _make_projection_fields(
        ti,
        coordinates,
        local_velocity,
        scale=1.25,
        center=np.array([-1.0, 0.5, 2.0]),
        body_velocity=np.array([0.2, -0.3, 0.4]),
        angular_velocity=np.array([0.12, -0.07, 0.19]),
        quaternion=np.array(
            [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
        ),
        gnum=np.array([9, 9, 9], dtype=np.int32),
        xmin_unscaled=np.array([-4.0, -4.0, -4.0]),
        spacing_unscaled=1.0,
    )

    _run_projection(ti, fields, shape_function_type)
    velocity = fields[4].to_numpy().reshape(9, 9, 9, 3)
    weight = fields[5].to_numpy().reshape(9, 9, 9)
    kk, jj, ii = np.indices((9, 9, 9))
    local_nodes = np.stack((ii - 4, jj - 4, kk - 4), axis=-1)
    expected = local_nodes @ gradient.T
    active = weight > 1.0e-14
    np.testing.assert_allclose(
        velocity[active],
        expected[active],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.count_nonzero(active) > 27
    assert float(fields[6][None]) < 2.0e-14


def test_velocity_extension_crosses_nonmonotone_transported_sdf(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.SoftBodyKernel import (
        extend_soft_levelset_velocity_projection_,
    )

    vec3 = ti.types.vector(3, ti.f64)
    vec3i = ti.types.vector(3, ti.i32)
    soft_type = ti.types.struct(
        bodyID=ti.i32,
        gridStart=ti.i32,
        gridNum=ti.i32,
        scale=ti.f64,
    )
    box_type = ti.types.struct(
        gnum=vec3i,
        grid_space=ti.f64,
    )
    grid_type = ti.types.struct(distance_field=ti.f64)

    gnum = np.array([5, 5, 5], dtype=np.int32)
    grid_count = int(np.prod(gnum))
    soft = soft_type.field(shape=1)
    box = box_type.field(shape=1)
    grid = grid_type.field(shape=grid_count)
    velocity = ti.Vector.field(3, ti.f64, shape=grid_count)
    weight = ti.field(ti.f64, shape=grid_count)

    soft.bodyID[0] = 0
    soft.gridStart[0] = 0
    soft.gridNum[0] = grid_count
    soft.scale[0] = 1.0
    box.gnum[0] = gnum
    box.grid_space[0] = 1.0
    grid.distance_field.fill(10.0)
    velocity.fill(0.0)
    weight.fill(0.0)

    target = 2 + 2 * gnum[0] + 2 * gnum[0] * gnum[1]
    seed = 1 + 2 * gnum[0] + 2 * gnum[0] * gnum[1]
    grid[target].distance_field = 0.0
    grid[seed].distance_field = 2.5
    velocity[seed] = [0.3, -0.2, 0.1]
    weight[seed] = 1.0

    extend_soft_levelset_velocity_projection_(
        1,
        grid_count,
        0,
        soft,
        velocity,
        weight,
        box,
    )
    ti.sync()

    np.testing.assert_allclose(
        np.asarray(velocity[target]),
        np.array([0.3, -0.2, 0.1]),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert float(weight[target]) > 0.0


def test_velocity_extension_keeps_outer_sdf_boundary_fixed(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.SoftBodyKernel import (
        extend_soft_levelset_velocity_projection_,
    )

    vec3 = ti.types.vector(3, ti.f64)
    vec3i = ti.types.vector(3, ti.i32)
    soft_type = ti.types.struct(
        bodyID=ti.i32,
        gridStart=ti.i32,
        gridNum=ti.i32,
        scale=ti.f64,
    )
    box_type = ti.types.struct(
        gnum=vec3i,
        grid_space=ti.f64,
    )

    gnum = np.array([5, 5, 5], dtype=np.int32)
    grid_count = int(np.prod(gnum))
    soft = soft_type.field(shape=1)
    box = box_type.field(shape=1)
    velocity = ti.Vector.field(3, ti.f64, shape=grid_count)
    weight = ti.field(ti.f64, shape=grid_count)

    soft.bodyID[0] = 0
    soft.gridStart[0] = 0
    soft.gridNum[0] = grid_count
    soft.scale[0] = 1.0
    box.gnum[0] = gnum
    box.grid_space[0] = 1.0
    velocity.fill(0.0)
    weight.fill(0.0)

    boundary = 0 + 2 * gnum[0] + 2 * gnum[0] * gnum[1]
    seed = 1 + 2 * gnum[0] + 2 * gnum[0] * gnum[1]
    velocity[seed] = [0.3, -0.2, 0.1]
    weight[seed] = 1.0

    extend_soft_levelset_velocity_projection_(
        1,
        grid_count,
        0,
        soft,
        velocity,
        weight,
        box,
    )
    ti.sync()

    np.testing.assert_allclose(
        np.asarray(velocity[boundary]),
        np.zeros(3),
        rtol=0.0,
        atol=0.0,
    )
    assert float(weight[boundary]) == 0.0
