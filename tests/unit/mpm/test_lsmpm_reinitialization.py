import numpy as np


def _make_levelset_fields(ti, phi, spacing):
    vec3 = ti.types.vector(3, ti.f64)
    vec3i = ti.types.vector(3, ti.i32)
    soft_type = ti.types.struct(
        bodyID=ti.i32,
        gridStart=ti.i32,
        gridNum=ti.i32,
    )
    grid_type = ti.types.struct(
        distance_field=ti.f64,
        distance_field_temp=ti.f64,
        distance_field0=ti.f64,
    )
    box_type = ti.types.struct(
        gnum=vec3i,
        grid_space=ti.f64,
        scale=ti.f64,
        xmin=vec3,
        xmax=vec3,
    )
    flat = np.ascontiguousarray(phi.ravel(), dtype=np.float64)
    soft = soft_type.field(shape=1)
    grid = grid_type.field(shape=flat.size)
    box = box_type.field(shape=1)
    soft.bodyID[0] = 0
    soft.gridStart[0] = 0
    soft.gridNum[0] = flat.size
    grid.distance_field.from_numpy(flat)
    grid.distance_field_temp.from_numpy(flat)
    grid.distance_field0.from_numpy(flat)
    box.gnum[0] = np.array(phi.shape[::-1], dtype=np.int32)
    box.grid_space[0] = spacing
    box.scale[0] = 1.0
    box.xmin[0] = np.zeros(3)
    box.xmax[0] = spacing * (np.asarray(phi.shape[::-1]) - 1)
    return soft, grid, box


def _row_crossings(field, coordinates):
    roots = {}
    for row_id, row in enumerate(field):
        crossed = np.flatnonzero(row[:-1] * row[1:] < 0.0)
        if crossed.size != 2:
            continue
        values = row[crossed]
        next_values = row[crossed + 1]
        fraction = np.abs(values) / (np.abs(values) + np.abs(next_values))
        roots[row_id] = coordinates[crossed] + fraction * (
            coordinates[crossed + 1] - coordinates[crossed]
        )
    return roots


def _line_crossings(field, coordinates):
    crossed = np.flatnonzero(field[:-1] * field[1:] < 0.0)
    values = field[crossed]
    next_values = field[crossed + 1]
    fraction = np.abs(values) / (np.abs(values) + np.abs(next_values))
    return coordinates[crossed] + fraction * (
        coordinates[crossed + 1] - coordinates[crossed]
    )


def _eikonal_mean(phi, spacing, band_cells):
    dz, dy, dx = np.gradient(phi, spacing, edge_order=1)
    error = np.abs(np.sqrt(dx * dx + dy * dy + dz * dz) - 1.0)
    mask = np.abs(phi) <= band_cells * spacing
    mask[[0, -1], :, :] = False
    mask[:, [0, -1], :] = False
    mask[:, :, [0, -1]] = False
    return float(np.mean(error[mask]))


def test_subcell_redistancing_does_not_accumulate_square_interface_motion(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.LevelSet import (
        reinitialize_soft_levelset_step_,
        store_soft_levelset_reinit_reference_,
    )

    spacing = 0.06
    x = np.arange(-1.2, 1.2 + 0.5 * spacing, spacing)
    y = x.copy()
    z = np.arange(-0.3, 0.3 + 0.5 * spacing, spacing)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    angle = np.deg2rad(23.0)
    xr = np.cos(angle) * xx + np.sin(angle) * yy
    yr = -np.sin(angle) * xx + np.cos(angle) * yy
    qx = np.abs(xr) - 0.5
    qy = np.abs(yr) - 0.5
    qz = np.abs(zz) - 0.2
    outside = np.sqrt(
        np.maximum(qx, 0.0) ** 2
        + np.maximum(qy, 0.0) ** 2
        + np.maximum(qz, 0.0) ** 2
    )
    phi = outside + np.minimum(np.maximum.reduce((qx, qy, qz)), 0.0)
    soft, grid, box = _make_levelset_fields(ti, phi, spacing)
    initial = _row_crossings(phi[len(z) // 2], x)

    for _ in range(128):
        store_soft_levelset_reinit_reference_(1, phi.size, soft, grid)
        reinitialize_soft_levelset_step_(
            1, phi.size, 8.0, 0.1, soft, grid, box
        )
    ti.sync()

    final_phi = grid.distance_field.to_numpy().reshape(phi.shape)
    final = _row_crossings(final_phi[len(z) // 2], x)
    shared_rows = sorted(set(initial) & set(final))
    assert len(shared_rows) >= 12
    assert set(initial) == set(final)
    drift = max(
        np.max(np.abs(final[row] - initial[row])) for row in shared_rows
    )
    # The sharp corner limits this test to first-order interface accuracy, but
    # repeated maintenance must remain subcell rather than accumulate an O(h)
    # displacement at every event.
    assert drift < 0.50 * spacing


def test_interface_locked_redistancing_reduces_eikonal_error_without_moving_roots(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.LevelSet import (
        reinitialize_soft_levelset_step_,
        store_soft_levelset_reinit_reference_,
    )

    spacing = 0.05
    coordinates = np.arange(-0.8, 0.8 + 0.5 * spacing, spacing)
    zz, yy, xx = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    )
    normal = np.array([1.0, 0.35, -0.2], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    signed_distance = (
        normal[0] * xx + normal[1] * yy + normal[2] * zz - 0.037
    )
    phi = signed_distance * (
        1.0 + 0.75 * np.tanh(signed_distance / (2.0 * spacing))
    )
    soft, grid, box = _make_levelset_fields(ti, phi, spacing)
    center = len(coordinates) // 2
    initial_roots = _line_crossings(
        phi[center, center], coordinates
    )
    initial_error = _eikonal_mean(phi, spacing, 4.0)

    store_soft_levelset_reinit_reference_(1, phi.size, soft, grid)
    for _ in range(48):
        reinitialize_soft_levelset_step_(
            1, phi.size, 8.0, 0.1, soft, grid, box
        )
    ti.sync()

    final_phi = grid.distance_field.to_numpy().reshape(phi.shape)
    final_roots = _line_crossings(
        final_phi[center, center], coordinates
    )
    final_error = _eikonal_mean(final_phi, spacing, 4.0)
    assert initial_roots.size == final_roots.size == 1
    assert np.max(np.abs(final_roots - initial_roots)) < 1.0e-12
    assert final_error < 0.50 * initial_error


def test_weno5_ssprk3_advects_a_linear_sdf_and_subcycles_large_cfl(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.SoftBodyKernel import (
        advect_projected_soft_levelset_weno5_,
    )

    spacing = 0.05
    x_coordinates = np.arange(-2.0, 2.0 + 0.5 * spacing, spacing)
    transverse_coordinates = np.arange(
        -0.15, 0.15 + 0.5 * spacing, spacing
    )
    zz, yy, xx = np.meshgrid(
        transverse_coordinates,
        transverse_coordinates,
        x_coordinates,
        indexing="ij",
    )
    phi = xx - 0.037
    soft, grid, box = _make_levelset_fields(ti, phi, spacing)
    velocity = ti.Vector.field(3, ti.f64, shape=phi.size)
    projection_weight = ti.field(ti.f64, shape=phi.size)
    velocity.fill((0.4, 0.0, 0.0))
    projection_weight.fill(1.0)

    # The requested step has C=0.8 and must therefore be split into four
    # C<=0.2 SSP-RK3 updates. A linear level set is transported exactly.
    dt = 0.8 * spacing / 0.4
    advect_projected_soft_levelset_weno5_(
        1,
        phi.size,
        dt,
        soft,
        grid,
        velocity,
        projection_weight,
        box,
        maximum_cfl=0.2,
    )
    ti.sync()

    transported = grid.distance_field.to_numpy().reshape(phi.shape)
    expected = phi - 0.4 * dt
    # Fixed outer nodes can affect three cells per RK stage. Four substeps
    # contain twelve stages, so exclude that complete numerical domain of
    # dependence when checking the exact linear solution.
    interior = np.s_[3:-3, 3:-3, 37:-37]
    assert np.max(np.abs(transported[interior] - expected[interior])) < 1.0e-11
