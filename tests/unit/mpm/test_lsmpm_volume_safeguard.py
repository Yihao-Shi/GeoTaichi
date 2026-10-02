import numpy as np


def _make_scalar_fields(ti):
    soft_type = ti.types.struct(bodyID=ti.i32)
    box_type = ti.types.struct(grid_space=ti.f64, scale=ti.f64)
    soft = soft_type.field(shape=1)
    box = box_type.field(shape=1)
    fields = [ti.field(ti.f64, shape=1) for _ in range(8)]
    fields.append(ti.field(ti.f64, shape=()))
    soft.bodyID[0] = 0
    box.grid_space[0] = 1.0
    box.scale[0] = 1.0
    return soft, box, fields


def test_volume_safeguard_brackets_and_accepts_interior_newton_step(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.LevelSet import (
        compute_soft_levelset_safeguarded_shift_,
        initialize_soft_levelset_volume_bracket_,
    )

    soft, box, fields = _make_scalar_fields(ti)
    (
        target,
        current,
        interface,
        lower,
        upper,
        trial,
        shift,
        error,
        max_error,
    ) = fields
    target[0] = 1.0
    current[0] = 1.1
    interface[0] = 1.0

    initialize_soft_levelset_volume_bracket_(
        1,
        1.0e-9,
        0.25,
        soft,
        box,
        target,
        current,
        lower,
        upper,
        trial,
        shift,
    )
    ti.sync()
    np.testing.assert_allclose(
        [lower[0], upper[0], trial[0], shift[0]],
        [0.0, 0.25, 0.25, 0.25],
        rtol=0.0,
        atol=0.0,
    )

    current[0] = 0.9
    compute_soft_levelset_safeguarded_shift_(
        1,
        1.0e-9,
        target,
        current,
        interface,
        lower,
        upper,
        trial,
        shift,
        error,
        max_error,
    )
    ti.sync()
    np.testing.assert_allclose(
        [lower[0], upper[0], trial[0], shift[0]],
        [0.0, 0.25, 0.15, -0.10],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_volume_safeguard_falls_back_to_bracket_midpoint(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.LevelSet import (
        compute_soft_levelset_safeguarded_shift_,
    )

    _, _, fields = _make_scalar_fields(ti)
    (
        target,
        current,
        interface,
        lower,
        upper,
        trial,
        shift,
        error,
        max_error,
    ) = fields
    target[0] = 1.0
    current[0] = 0.9
    interface[0] = 0.01
    lower[0] = 0.0
    upper[0] = 0.25
    trial[0] = 0.25

    compute_soft_levelset_safeguarded_shift_(
        1,
        1.0e-9,
        target,
        current,
        interface,
        lower,
        upper,
        trial,
        shift,
        error,
        max_error,
    )
    ti.sync()
    np.testing.assert_allclose(
        [lower[0], upper[0], trial[0], shift[0]],
        [0.0, 0.25, 0.125, -0.125],
        rtol=0.0,
        atol=1.0e-14,
    )
