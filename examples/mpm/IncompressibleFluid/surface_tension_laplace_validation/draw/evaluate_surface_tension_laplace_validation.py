"""Evaluation and postprocessing for surface_tension_laplace_validation."""

import numpy as np


def validate_laplace(mpm, dim, center, radius, sigma, tolerance):
    cell_type = mpm.scene.element.cell.type.to_numpy()
    pressure = mpm.scene.element.cell.pressure.to_numpy()
    surface_tension = mpm.scene.element.cell.surface_tension.to_numpy()
    grid_size = np.array(mpm.scene.element.grid_size, dtype=float)[:dim]
    ghost_cell = int(mpm.scene.element.ghost_cell)
    min_dx = float(np.min(grid_size))

    indices = np.indices(cell_type.shape).reshape(dim, -1).T
    cell_indices = indices - ghost_cell
    centers = (cell_indices + 0.5) * grid_size
    distance = np.linalg.norm(centers - np.array(center, dtype=float), axis=1).reshape(cell_type.shape)

    interface_air = (cell_type == 0) & (np.abs(surface_tension) > 0.0)
    interior = (cell_type == 1) & (distance < radius - 2.0 * min_dx)
    if not np.any(interior):
        interior = cell_type == 1

    if not np.any(interface_air):
        print(f"[{dim}D] FAIL: no air-interface cells received surface tension.")
        return False

    theory_jump = (dim - 1) * sigma / radius
    mean_jump = float(np.mean(surface_tension[interface_air]))
    mean_pressure = float(np.mean(pressure[interior])) if np.any(interior) else float("nan")
    rel_error = abs(mean_jump - theory_jump) / max(abs(theory_jump), 1e-12)

    print(f"[{dim}D] sampled interface air cells = {int(np.count_nonzero(interface_air))}")
    print(
        f"[{dim}D] surface-tension pressure jump = {mean_jump:.8g}, theory = {theory_jump:.8g}, rel_error = {rel_error:.6g}"
    )
    print(f"[{dim}D] mean interior pressure after projection = {mean_pressure:.8g}")
    if rel_error <= tolerance:
        print(f"[{dim}D] PASS")
        return True
    print(f"[{dim}D] FAIL: relative pressure-jump error exceeds {tolerance}")
    return False
