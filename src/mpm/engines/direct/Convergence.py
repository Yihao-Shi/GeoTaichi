"""Physical correction measure shared by direct MPM and IPC couplings."""

import taichi as ti


@ti.func
def particle_correction_measure(
    mpm: ti.template(), correction: ti.template(), particle_id, active_dof, base_dof, stride, dim: ti.template()
):
    displacement = ti.Vector.zero(ti.f64, dim)
    gradient = ti.Matrix.zero(ti.f64, dim, dim)
    for local_id in range(mpm.offset[particle_id]):
        grid_id = mpm.LnID[particle_id, local_id]
        block = mpm.node2dof[grid_id] - 1
        if 0 <= dim * block and dim * (block + 1) <= active_dof:
            delta = ti.Vector([correction[base_dof + stride * block + d] for d in ti.static(range(dim))])
            displacement += mpm.shape[particle_id, local_id] * delta
            gradient += delta.outer_product(mpm.dshape[particle_id, local_id])
    spacing = mpm.body[mpm.particle[particle_id].bodyID].grid_size
    measure = 0.0
    for d in ti.static(range(dim)):
        measure = ti.max(measure, ti.abs(displacement[d]))
        for axis in ti.static(range(dim)):
            measure = ti.max(measure, spacing * ti.abs(gradient[d, axis]))
    if ti.static(mpm.is_axisymmetric):
        radius = mpm.particle[particle_id].x[0] - ti.static(mpm.axis_offset)
        measure = ti.max(measure, spacing * ti.abs(displacement[0]) / ti.max(radius, 1.0e-30))
    return measure


@ti.kernel
def mpm_correction_measure(
    mpm: ti.template(), correction: ti.template(), active_dof: ti.i32, dim: ti.template()
) -> ti.f64:
    measure = 0.0
    for particle_id in range(mpm.particleNum[0]):
        ti.atomic_max(measure, particle_correction_measure(mpm, correction, particle_id, active_dof, 0, dim, dim))
    return measure
