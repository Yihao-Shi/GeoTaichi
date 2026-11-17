"""Case-independent fluid-grid diagnostics and piston-wave references."""

import math

import numpy as np


def deep_air_particle_mask(cell_type, particle_cell):
    occupied_type = cell_type[tuple(particle_cell.T)]
    air_cell = particle_cell[occupied_type == 0]
    detached = np.zeros(len(particle_cell), dtype=bool)
    if len(air_cell) == 0:
        return detached
    adjacent_fluid = np.zeros(len(air_cell), dtype=bool)
    upper = np.asarray(cell_type.shape) - 1
    for offset in np.ndindex(*([3] * cell_type.ndim)):
        neighbor = np.minimum(np.maximum(air_cell + np.asarray(offset) - 1, 0), upper)
        adjacent_fluid |= cell_type[tuple(neighbor.T)] == 1
    detached[occupied_type == 0] = ~adjacent_fluid
    return detached


def nonzero_spray_pressure_count(pressure, spray, atmospheric_pressure=0.0):
    # Detached samples without grid support are zero; samples of interface
    # air may equal the configured atmospheric pressure.
    pressure = np.asarray(pressure)
    tolerance = 64.0 * np.finfo(pressure.dtype).eps * max(1.0, float(np.max(np.abs(pressure))))
    deviation = np.minimum(np.abs(pressure[spray]), np.abs(pressure[spray] - atmospheric_pressure))
    return int(np.count_nonzero(deviation > tolerance))


def saved_grid_spacing(grid):
    dims = tuple(np.asarray(grid["dims"], dtype=np.int64))
    coords = np.asarray(grid["coords"])
    if len(dims) != coords.shape[1] or math.prod(dims) != len(coords) or min(dims) < 2:
        raise ValueError(f"invalid saved grid coordinates: dims={dims}, coords={coords.shape}")
    lattice = coords.reshape(*dims, len(dims))
    origin = lattice[(0,) * len(dims)]
    spacing = []
    for axis in range(len(dims)):
        neighbor = [0] * len(dims)
        neighbor[axis] = 1
        spacing.append(abs(float(lattice[tuple(neighbor)][axis] - origin[axis])))
    if min(spacing) <= 0.0:
        raise ValueError(f"invalid saved grid spacing: {spacing}")
    return np.asarray(spacing)


def enclosed_air_count(cell_type, fluid_sdf=None, exterior_faces=None):
    air = cell_type == 0
    diagonal_edges = []
    if fluid_sdf is not None:
        phi = np.asarray(fluid_sdf)
        if air.ndim not in (2, 3) or phi.shape != air.shape or not np.isfinite(phi).all():
            raise ValueError("fluid_sdf connectivity requires a finite 2-D or 3-D field matching cell_type")
        # A bilinear face saddle decides each diagonal connection; solids remain barriers.
        for axis1 in range(air.ndim):
            for axis2 in range(axis1 + 1, air.ndim):
                corners = []
                for first, second in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    index = [slice(None)] * air.ndim
                    index[axis1] = slice(first, None if first else -1)
                    index[axis2] = slice(second, None if second else -1)
                    corners.append(tuple(index))
                a, b, c, d = corners
                ad = (
                    air[a]
                    & air[d]
                    & (cell_type[b] == 1)
                    & (cell_type[c] == 1)
                    & (phi[a] > 0)
                    & (phi[d] > 0)
                    & (phi[b] < 0)
                    & (phi[c] < 0)
                    & (phi[a] * phi[d] > phi[b] * phi[c])
                )
                bc = (
                    air[b]
                    & air[c]
                    & (cell_type[a] == 1)
                    & (cell_type[d] == 1)
                    & (phi[b] > 0)
                    & (phi[c] > 0)
                    & (phi[a] < 0)
                    & (phi[d] < 0)
                    & (phi[b] * phi[c] > phi[a] * phi[d])
                )
                diagonal_edges.extend(((a, d, ad), (b, c, bc)))
    exterior = np.zeros_like(air)
    if exterior_faces is None:
        exterior_faces = [(axis, side) for axis in range(air.ndim) for side in (0, -1)]
    for axis, index in exterior_faces:
        face = [slice(None)] * air.ndim
        face[axis] = index
        exterior[tuple(face)] = air[tuple(face)]

    while True:
        connected = exterior.copy()
        for axis in range(air.ndim):
            lower = [slice(None)] * air.ndim
            upper = [slice(None)] * air.ndim
            lower[axis] = slice(1, None)
            upper[axis] = slice(None, -1)
            connected[tuple(lower)] |= exterior[tuple(upper)]
            connected[tuple(upper)] |= exterior[tuple(lower)]
        for first, second, mask in diagonal_edges:
            connected[first] |= mask & exterior[second]
            connected[second] |= mask & exterior[first]
        connected &= air
        if np.array_equal(connected, exterior):
            return int(np.count_nonzero(air & ~exterior))
        exterior = connected


def linear_piston_wave(frequency, depth, velocity_amplitude, gravity=9.81):
    omega = 2.0 * math.pi * frequency
    wavenumber = omega * omega / gravity
    for _ in range(20):
        kh = wavenumber * depth
        residual = gravity * wavenumber * math.tanh(kh) - omega * omega
        derivative = gravity * (math.tanh(kh) + kh / math.cosh(kh) ** 2)
        correction = residual / derivative
        wavenumber -= correction
        if abs(correction) <= 1.0e-14 * max(1.0, wavenumber):
            break
    kh = wavenumber * depth
    transfer = 4.0 * math.sinh(kh) ** 2 / (math.sinh(2.0 * kh) + 2.0 * kh)
    phase_speed = omega / wavenumber
    group_speed = 0.5 * phase_speed * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    return wavenumber, transfer * velocity_amplitude / omega, group_speed


def piston_displacement(time, frequency, velocity_amplitude, ramp_time):
    omega = 2.0 * math.pi * frequency
    ramp = 1.0 if time >= ramp_time else 0.5 * (1.0 - math.cos(math.pi * time / ramp_time))
    return ramp * velocity_amplitude / omega * math.sin(omega * time)


def harmonic_fit(time, values, frequency):
    omega_time = 2.0 * math.pi * frequency * time
    design = np.column_stack((np.sin(omega_time), np.cos(omega_time), np.ones_like(time)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    residual = float(np.sum((values - fitted) ** 2))
    total = float(np.sum((values - np.mean(values)) ** 2))
    return (
        float(np.hypot(coefficients[0], coefficients[1])),
        float(coefficients[2]),
        1.0 - residual / max(total, 1.0e-30),
    )


def deep_air_particle_count(cell_type, particle_cell):
    return int(np.count_nonzero(deep_air_particle_mask(cell_type, particle_cell)))


def sdf_surface_gauge(cell_type, fluid_sdf, dx, gauge_x, half_width=None):
    cell_type, fluid_sdf = np.squeeze(cell_type), np.squeeze(fluid_sdf)
    x_cells = np.flatnonzero(np.abs((np.arange(cell_type.shape[0]) + 0.5) * dx - gauge_x) <= (half_width or dx))
    surfaces = []
    for i in x_cells:
        for j in range(1, cell_type.shape[1] - 1):
            crossings = np.flatnonzero((cell_type[i, j, :-1] == 1) & (cell_type[i, j, 1:] == 0))
            if crossings.size:
                k = int(crossings[-1])
                phi_fluid, phi_air = float(fluid_sdf[i, j, k]), float(fluid_sdf[i, j, k + 1])
                theta = (
                    float(np.clip(phi_fluid / (phi_fluid - phi_air), 0.01, 1.0)) if phi_fluid < 0.0 < phi_air else 0.5
                )
                surfaces.append((k + 0.5 + theta) * dx)
    return float(np.median(surfaces)) if surfaces else float("nan")


def sdf_mean_surface(cell_type, fluid_sdf, dx, piston_position=None):
    cell_type, fluid_sdf = np.squeeze(cell_type), np.squeeze(fluid_sdf)
    surfaces = []
    for i in range(1, cell_type.shape[0] - 1):
        if piston_position is not None and i * dx < piston_position:
            continue
        for j in range(1, cell_type.shape[1] - 1):
            crossings = np.flatnonzero((cell_type[i, j, :-1] == 1) & (cell_type[i, j, 1:] == 0))
            if crossings.size:
                k = int(crossings[-1])
                phi_fluid, phi_air = float(fluid_sdf[i, j, k]), float(fluid_sdf[i, j, k + 1])
                theta = (
                    float(np.clip(phi_fluid / (phi_fluid - phi_air), 0.01, 1.0)) if phi_fluid < 0.0 < phi_air else 0.5
                )
                surfaces.append((k + 0.5 + theta) * dx)
    return float(np.mean(surfaces)) if surfaces else float("nan")


def sdf_surface_profile_metrics(cell_type, fluid_sdf, dx, piston_position=None):
    cell_type, fluid_sdf = np.squeeze(cell_type), np.squeeze(fluid_sdf)
    surface = np.full(cell_type.shape[:2], np.nan)
    for i in range(1, cell_type.shape[0] - 1):
        for j in range(1, cell_type.shape[1] - 1):
            crossings = np.flatnonzero((cell_type[i, j, :-1] == 1) & (cell_type[i, j, 1:] == 0))
            if crossings.size:
                k = int(crossings[-1])
                phi_fluid, phi_air = float(fluid_sdf[i, j, k]), float(fluid_sdf[i, j, k + 1])
                theta = (
                    float(np.clip(phi_fluid / (phi_fluid - phi_air), 0.01, 1.0)) if phi_fluid < 0.0 < phi_air else 0.5
                )
                surface[i, j] = (k + 0.5 + theta) * dx
    x_margin = 2 if surface.shape[0] > 6 else 1
    y_margin = 2 if surface.shape[1] > 6 else 1
    # A column intersecting the solid piston has no physical free-water surface.
    x_start = x_margin if piston_position is None else max(x_margin, math.ceil(piston_position / dx))
    interior = surface[x_start:-x_margin, y_margin:-y_margin]
    if not np.isfinite(interior).any():
        return float("nan"), float("nan"), float("nan")
    interior = interior[np.any(np.isfinite(interior), axis=1)]
    centerline = np.nanmedian(interior, axis=1)
    cross_rms = float(np.sqrt(np.nanmean((interior - centerline[:, None]) ** 2)))
    centerline = centerline[np.isfinite(centerline)]
    longitudinal_rms = (
        float(np.sqrt(np.mean(np.diff(centerline, n=2) ** 2) / 6.0)) if len(centerline) >= 3 else float("nan")
    )
    return cross_rms, longitudinal_rms, float(np.ptp(centerline))
