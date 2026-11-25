"""Initial soil state and grid constraints shared by the excavation and pile examples."""

import numpy as np


def box_points(minimum, size, counts):
    spacing = np.asarray(size) / counts
    axes = [low + (np.arange(n) + 0.5) * dx for low, n, dx in zip(minimum, counts, spacing)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return points, float(spacing.prod())


def geostatic_state(depth, material, gravity):
    """Return K0, principal Kirchhoff stresses and elastic Hencky deformation."""
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Every soil point must have an overlying free surface")
    k0 = 1.0 - np.sin(np.deg2rad(material["FrictionAngle"]))
    stress = -(material["density"] * gravity * depth[:, None]) * np.array([k0, k0, 1.0])
    young, poisson = material["young_modulus"], material["poisson_ratio"]
    log_stretch = ((1 + poisson) * stress - poisson * stress.sum(axis=1, keepdims=True)) / young
    return k0, stress, np.exp(log_stretch)[:, :, None] * np.eye(3)


def soil_boundary_dofs(lo, hi, h, container=None):
    """Fixed bottom and normal-only side supports, with an optional inner container."""
    shape = np.ceil((np.asarray(hi) - lo) / h).astype(int) + 1
    nx, ny, nz = shape
    z, y, x = np.ogrid[:nz, :ny, :nx]
    coords = [lo[0] + h * x, lo[1] + h * y, lo[2] + h * z]
    nodes = np.arange(nx * ny * nz, dtype=np.int32).reshape(nz, ny, nx)
    tol = 1e-9 * h
    floor = coords[2] <= tol
    fixed = []
    for axis, coord in enumerate(coords):
        mask = floor | (coord <= lo[axis] + tol) | (coord >= lo[axis] + h * (shape[axis] - 1) - tol)
        if container is not None and axis < 2:
            minimum, maximum, rim = container
            cross = 1 - axis
            within = (
                (coords[cross] >= minimum[cross] - h - tol)
                & (coords[cross] <= maximum[cross] + h + tol)
                & (coords[2] <= rim + tol)
            )
            sides = ((coord >= minimum[axis] - h - tol) & (coord <= minimum[axis] + tol)) | (
                (coord >= maximum[axis] - tol) & (coord <= maximum[axis] + h + tol)
            )
            mask = mask | (within & sides)
        fixed.append(3 * nodes[np.broadcast_to(mask, nodes.shape)] + axis)
    return fixed


def add_soil_boundary(mpm, lo, hi, h, container=None):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    fixed = soil_boundary_dofs(lo, hi, h, container)
    boundary = DirichletBoundary()
    boundary.append(fixed, np.zeros(sum(map(len, fixed))))
    mpm.add_boundary_condition(dirichlet=boundary)
