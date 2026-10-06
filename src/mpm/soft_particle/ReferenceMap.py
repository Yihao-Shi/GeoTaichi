"""Total inverse-map reconstruction from material-point positions and F.

Adaptation of reference-SDF pullback: affine particle patches replace k-NN
displacement interpolation. No previous-step SDF enters reconstruction.
"""

import taichi as ti

from src.mpm.soft_particle.SoftBodyKernel import (
    audit_soft_levelset_velocity_projection_,
    normalize_soft_levelset_velocity_projection_,
    reset_soft_levelset_velocity_projection_,
    scatter_soft_point_to_sdf_node_,
)
from src.utils.constants import Threshold
from src.utils.Quaternion import SetToRotate
from src.utils.ScalarFunction import linearize3D


@ti.kernel
def initialize_soft_reference_sdf_(
    soft_num: int, soft: ti.template(), grid: ti.template(), reference: ti.template(), initialized: ti.template()
):
    # Per-body flags also handle bodies inserted after simulation starts.
    for sb in range(soft_num):
        if initialized[sb] == 0:
            for local in range(soft[sb].gridNum):
                node = soft[sb].gridStart + local
                reference[node] = grid[node].distance_field
            initialized[sb] = 1


@ti.kernel
def project_soft_inverse_map_(
    point_num: int,
    shape_type: ti.template(),
    soft: ti.template(),
    points: ti.template(),
    rigid: ti.template(),
    box: ti.template(),
    displacement: ti.template(),
    derivative: ti.template(),
    weight: ti.template(),
    support_loss: ti.template(),
) -> int:
    invalid = 0
    support_loss[None] = 0.0
    for p in range(point_num):
        if points[p].active == 1:
            body = points[p].bodyID
            sb = rigid[body].softID
            scale = soft[sb].scale
            R = SetToRotate(rigid[body].q)
            X = soft[sb].referenceRotation.transpose() @ (points[p].x0 - soft[sb].mass_center0) / scale
            x = R.transpose() @ (points[p].x - rigid[body].mass_center) / scale
            determinant = points[p].F.determinant()
            if not (determinant > Threshold) or ti.math.isinf(determinant):
                ti.atomic_add(invalid, 1)
            else:
                B = soft[sb].referenceRotation.transpose() @ points[p].F.inverse() @ R
                D = B - ti.Matrix.identity(float, 3)
                xmin = box[body].xmin / scale
                h = box[body].grid_space / scale
                offset = 0.0
                width = 2
                if ti.static(shape_type == 1):
                    offset = 0.5
                    width = 3
                elif ti.static(shape_type == 2):
                    offset = 1.0
                    width = 4
                base = ti.cast(ti.floor((x - xmin) / h - offset), ti.i32)
                total = 0.0
                for i, j, k in ti.ndrange(width, width, width):
                    ijk = base + ti.Vector([i, j, k])
                    w = scatter_soft_point_to_sdf_node_(
                        shape_type,
                        sb,
                        body,
                        ijk,
                        x,
                        X - x,
                        D,
                        points[p].m,
                        xmin,
                        h,
                        1.0 / h,
                        soft,
                        displacement,
                        weight,
                        box,
                    )
                    if w > Threshold:
                        node = soft[sb].gridStart + linearize3D(ijk[0], ijk[1], ijk[2], box[body].gnum)
                        ti.atomic_add(derivative[node], points[p].m * w * D)
                    total += w
                ti.atomic_max(support_loss[None], ti.abs(1.0 - total))
    return invalid


@ti.kernel
def normalize_soft_inverse_derivative_(
    soft_num: int, max_grid: int, soft: ti.template(), derivative: ti.template(), weight: ti.template()
):
    for sb, local in ti.ndrange(soft_num, max_grid):
        if local < soft[sb].gridNum:
            node = soft[sb].gridStart + local
            if weight[node] > Threshold:
                derivative[node] /= weight[node]


@ti.kernel
def extend_soft_inverse_map_(
    soft_num: int,
    max_grid: int,
    soft: ti.template(),
    box: ti.template(),
    displacement: ti.template(),
    derivative: ti.template(),
    weight: ti.template(),
):
    for sb, local in ti.ndrange(soft_num, max_grid):
        if local < soft[sb].gridNum:
            body = soft[sb].bodyID
            gnum = box[body].gnum
            h = box[body].grid_space / soft[sb].scale
            ijk = ti.Vector([local % gnum[0], (local // gnum[0]) % gnum[1], local // (gnum[0] * gnum[1])])
            node = soft[sb].gridStart + local
            if weight[node] <= Threshold:
                total = 0.0
                value = ti.Vector.zero(float, 3)
                gradient = ti.Matrix.zero(float, 3, 3)
                count = 0
                for axis, side in ti.static(ti.ndrange(3, 2)):
                    neighbor = ijk
                    direction = -1 if side == 0 else 1
                    neighbor[axis] += direction
                    if 0 <= neighbor[axis] < gnum[axis]:
                        other = soft[sb].gridStart + linearize3D(neighbor[0], neighbor[1], neighbor[2], gnum)
                        w = weight[other]
                        if w > Threshold:
                            # Affine extrapolation preserves rigid/affine maps in the exterior halo.
                            delta = ti.Vector.zero(float, 3)
                            delta[axis] = -direction * h
                            value += w * (displacement[other] + derivative[other] @ delta)
                            gradient += w * derivative[other]
                            total += w
                            count += 1
                if total > Threshold:
                    displacement[node] = value / total
                    derivative[node] = gradient / total
                    # Negative marks this wave's nodes: readers only use positive
                    # previous-wave weights, avoiding order-dependent extension.
                    weight[node] = -total / count


@ti.kernel
def activate_soft_inverse_extension_(soft_num: int, max_grid: int, soft: ti.template(), weight: ti.template()):
    for sb, local in ti.ndrange(soft_num, max_grid):
        if local < soft[sb].gridNum:
            node = soft[sb].gridStart + local
            weight[node] = ti.abs(weight[node])


@ti.func
def sample_soft_reference_sdf_(X, body_box, reference):
    h = body_box.grid_space / body_box.scale
    xmin = body_box.xmin / body_box.scale
    xmax = body_box.xmax / body_box.scale
    q = ti.min(ti.max(X, xmin), xmax)
    base = ti.min(ti.max(ti.cast(ti.floor((q - xmin) / h), ti.i32), 0), body_box.gnum - 2)
    f = (q - xmin) / h - ti.cast(base, float)
    phi = 0.0
    gradient = ti.Vector.zero(float, 3)
    for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
        w = ti.Vector([f[0] if i else 1 - f[0], f[1] if j else 1 - f[1], f[2] if k else 1 - f[2]])
        node = body_box.startGrid + linearize3D(base[0] + i, base[1] + j, base[2] + k, body_box.gnum)
        value = reference[node]
        phi += w[0] * w[1] * w[2] * value
        gradient += (
            value
            / h
            * ti.Vector(
                [(1 if i else -1) * w[1] * w[2], (1 if j else -1) * w[0] * w[2], (1 if k else -1) * w[0] * w[1]]
            )
        )
    return phi, gradient


@ti.kernel
def reconstruct_soft_reference_sdf_(
    soft_num: int,
    max_grid: int,
    monitor_band: float,
    soft: ti.template(),
    grid: ti.template(),
    box: ti.template(),
    reference: ti.template(),
    displacement: ti.template(),
    derivative: ti.template(),
    weight: ti.template(),
) -> float:
    excess = 0.0
    for sb, local in ti.ndrange(soft_num, max_grid):
        if local < soft[sb].gridNum:
            body = soft[sb].bodyID
            gnum = box[body].gnum
            h = box[body].grid_space / soft[sb].scale
            xmin = box[body].xmin / soft[sb].scale
            xmax = box[body].xmax / soft[sb].scale
            ijk = ti.Vector([local % gnum[0], (local // gnum[0]) % gnum[1], local // (gnum[0] * gnum[1])])
            node = soft[sb].gridStart + local
            x = xmin + ti.cast(ijk, float) * h
            X = x + displacement[node]
            phi, n0 = sample_soft_reference_sdf_(X, box[body], reference)
            B = ti.Matrix.identity(float, 3) + derivative[node]
            if weight[node] > Threshold:
                if B.determinant() <= Threshold and phi <= monitor_band * h:
                    ti.atomic_max(excess, 1.0e30)
                if B.determinant() > Threshold and n0.norm() > Threshold:
                    # Local metric correction; exact for affine motion of a plane.
                    stretch = (B.transpose() @ (n0 / n0.norm())).norm()
                    phi /= ti.max(stretch, Threshold)
                if ti.abs(phi) <= monitor_band * h:
                    for d in ti.static(range(3)):
                        ti.atomic_max(excess, ti.max(xmin[d] - X[d], X[d] - xmax[d]) / h)
            else:
                # Unsupported far field is exterior, not the undeformed body.
                # The support audit below rejects cuts through the negative region.
                phi = ti.max(ti.abs(phi), (monitor_band + 1) * h)
            grid[node].distance_field = phi
    return excess


@ti.kernel
def audit_soft_inverse_halo_(
    soft_num: int, max_grid: int, soft: ti.template(), grid: ti.template(), box: ti.template(), weight: ti.template()
) -> int:
    missing = 0
    for sb, local in ti.ndrange(soft_num, max_grid):
        if local < soft[sb].gridNum:
            body = soft[sb].bodyID
            gnum = box[body].gnum
            node = soft[sb].gridStart + local
            if weight[node] <= Threshold:
                ijk = ti.Vector([local % gnum[0], (local // gnum[0]) % gnum[1], local // (gnum[0] * gnum[1])])
                for axis, side in ti.static(ti.ndrange(3, 2)):
                    other = ijk
                    other[axis] += -1 if side == 0 else 1
                    if 0 <= other[axis] < gnum[axis]:
                        index = soft[sb].gridStart + linearize3D(other[0], other[1], other[2], gnum)
                        if grid[index].distance_field < 0:
                            ti.atomic_add(missing, 1)
    return missing


def reconstruct_soft_levelset(
    soft_num,
    max_grid,
    point_num,
    shape_type,
    monitor_band,
    extension_iterations,
    soft,
    grid,
    points,
    displacement,
    derivative,
    weight,
    band_nodes,
    uncovered_nodes,
    support_loss,
    rigid,
    box,
    reference,
):
    reset_soft_levelset_velocity_projection_(soft_num, max_grid, soft, displacement, weight)
    derivative.fill(0)
    invalid = project_soft_inverse_map_(
        point_num, shape_type, soft, points, rigid, box, displacement, derivative, weight, support_loss
    )
    if invalid:
        raise RuntimeError(
            f"LSMPM reference-map reconstruction encountered {invalid} nonfinite or nonpositive particle F determinants"
        )
    normalize_soft_levelset_velocity_projection_(soft_num, max_grid, soft, displacement, weight)
    normalize_soft_inverse_derivative_(soft_num, max_grid, soft, derivative, weight)
    for _ in range(max(int(extension_iterations), 0)):
        extend_soft_inverse_map_(soft_num, max_grid, soft, box, displacement, derivative, weight)
        activate_soft_inverse_extension_(soft_num, max_grid, soft, weight)
    excess = float(
        reconstruct_soft_reference_sdf_(
            soft_num, max_grid, monitor_band, soft, grid, box, reference, displacement, derivative, weight
        )
    )
    if excess >= 1.0e30:
        raise RuntimeError(
            "LSMPM reference map has missing support or a nonpositive inverse Jacobian near the interface"
        )
    if audit_soft_inverse_halo_(soft_num, max_grid, soft, grid, box, weight):
        raise RuntimeError(
            "LSMPM reference-map support halo cuts the material interior; increase projection extension iterations"
        )
    audit_soft_levelset_velocity_projection_(
        soft_num, max_grid, monitor_band, soft, grid, weight, band_nodes, uncovered_nodes, box
    )
    return excess


def advect_soft_levelset_reference_map_(
    softNum,
    maxGridNum,
    pointNum,
    shape_function_type,
    projection_monitor_band,
    projection_extension_iterations,
    dt,
    soft,
    grid,
    material_point,
    soft_grid,
    soft_shape_node,
    soft_dshape,
    soft_shape_count,
    levelset_velocity,
    projection_weight,
    projection_band_nodes,
    projection_uncovered_nodes,
    projection_max_support_loss,
    rigid,
    box,
    *,
    reference,
    derivative,
):
    return reconstruct_soft_levelset(
        softNum,
        maxGridNum,
        pointNum,
        shape_function_type,
        projection_monitor_band,
        projection_extension_iterations,
        soft,
        grid,
        material_point,
        levelset_velocity,
        derivative,
        projection_weight,
        projection_band_nodes,
        projection_uncovered_nodes,
        projection_max_support_loss,
        rigid,
        box,
        reference,
    )
