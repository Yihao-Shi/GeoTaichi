"""Uniform porosity must stay uniform at walls without filling a free surface."""

import itertools
import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    _mac_base_1d,
    _mac_weight_1d,
    _mac_bspline_boundary_type,
    _boundary_face_control_volume_scale,
    kernel_extend_double_layer_face_support_at_solid_walls,
)


@ti.kernel
def scatter(
    count: int,
    dx: ti.template(),
    particle: ti.template(),
    faces: ti.template(),
    lengths: ti.template(),
    shape_type: int,
    support: int,
):
    dim = ti.static(len(faces))
    volume = 1.0
    for d in ti.static(range(dim)):
        volume *= dx[d]
    for p in range(count):
        for axis in ti.static(range(dim)):
            base = ti.Vector.zero(int, dim)
            for d in ti.static(range(dim)):
                shift = 0.0 if d == axis else 0.5
                base[d] = _mac_base_1d(particle[p].x[d], 1 / dx[d], shift, lengths[0][d], shape_type)
            for flat in range(support**dim):
                target = base + ti.Vector.zero(int, dim)
                local = flat
                valid = True
                for d in ti.static(range(dim)):
                    target[d] += local % support
                    local //= support
                    valid = valid and 0 <= target[d] < faces[axis].shape[d]
                if valid:
                    w = 1.0
                    for d in ti.static(range(dim)):
                        shift = 0.0 if d == axis else 0.5
                        w *= _mac_weight_1d(
                            particle[p].x[d],
                            (target[d] + shift) * dx[d],
                            1 / dx[d],
                            lengths[0][d],
                            _mac_bspline_boundary_type(target[d], faces[axis].shape[d]) if d == axis else 0,
                            shape_type,
                        )
                    faces[axis][target] -= (
                        w
                        * particle[p].vol
                        * (1 - particle[p].porosity)
                        / volume
                        * _boundary_face_control_volume_scale(target[axis], faces[axis].shape[axis])
                    )


def setup(dimension, positions, shape_type, kinds):
    particle = ti.Struct.field(
        {
            "active": ti.i32,
            "materialID": ti.i32,
            "phase": ti.i32,
            "bodyID": ti.i32,
            "x": ti.types.vector(dimension, ti.f64),
            "vol": ti.f64,
            "porosity": ti.f64,
        },
        shape=len(positions),
    )
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(1)
    particle.bodyID.fill(0)
    particle.x.from_numpy(np.asarray(positions, dtype=np.float64))
    particle.vol.fill(0.1**dimension / 2**dimension)
    particle.porosity.fill(0.3)
    kind = ti.field(ti.i32, shape=kinds.shape)
    kind.from_numpy(kinds)
    faces = tuple(
        ti.field(ti.f64, shape=tuple(n + (d == a) for d, n in enumerate(kinds.shape))) for a in range(dimension)
    )
    for f in faces:
        f.fill(1.0)
    lengths = ti.Vector.field(dimension, ti.f64, shape=1)
    lengths.fill([0.0, 0.025, 0.05, 0.1][shape_type])
    support = [2, 3, 3, 4][shape_type]
    return particle, kind, faces, lengths, support


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("shape_type", [0, 1, 2, 3])
@pytest.mark.parametrize("corner", [False, True])
def test_uniform_wall_face_porosity(taichi_runtime, dimension, shape_type, corner):
    size, lo, hi = 12, 3, 9
    positions = np.array(list(itertools.product(np.arange(lo + 0.25, hi, 0.5), repeat=dimension))) * 0.1
    kinds = np.zeros((size,) * dimension, dtype=np.int32)
    kinds[tuple(slice(lo, hi) for _ in range(dimension))] = 1
    for axis in range(dimension) if corner else [0]:
        index = [slice(None)] * dimension
        index[axis] = slice(0, lo)
        kinds[tuple(index)] = 2
        if axis != dimension - 1:
            index[axis] = slice(hi, None)
            kinds[tuple(index)] = 2
    particle, kind, faces, lengths, support = setup(dimension, positions, shape_type, kinds)
    dx = ti.Vector([0.1] * dimension)
    scatter(len(positions), dx, particle, faces, lengths, shape_type, support)
    free_index = tuple(hi if d == dimension - 1 else 6 for d in range(dimension))
    free_before = float(faces[-1][free_index])
    wall_index = tuple(lo if d == 0 else 6 for d in range(dimension))
    assert abs(float(faces[0][wall_index]) - 0.3) > 0.01
    kernel_extend_double_layer_face_support_at_solid_walls(
        len(positions), dx, particle, kind, faces, lengths, shape_type, support
    )
    for axis, face in enumerate(faces):
        values = face.to_numpy()
        for index in itertools.product(range(lo, 7), repeat=dimension):
            if not corner and any(index[d] != 6 for d in range(1, dimension)):
                continue
            np.testing.assert_allclose(values[index], 0.3, atol=2e-12, rtol=0, err_msg=f"axis={axis}, index={index}")
    assert abs(float(faces[-1][free_index]) - free_before) < 2e-12
    assert free_before > 0.3


@pytest.mark.parametrize("dimension", [2, 3])
def test_wall_images_do_not_cross_a_thin_wall(taichi_runtime, dimension):
    kinds = np.ones((12,) * dimension, dtype=np.int32)
    kinds[5] = 2
    positions = [[0.475] + [0.625] * (dimension - 1)]
    particle, kind, faces, lengths, support = setup(dimension, positions, 2, kinds)
    kernel_extend_double_layer_face_support_at_solid_walls(
        1, ti.Vector([0.1] * dimension), particle, kind, faces, lengths, 2, support
    )
    for f in faces:
        np.testing.assert_array_equal(f.to_numpy()[6:], 1.0)
