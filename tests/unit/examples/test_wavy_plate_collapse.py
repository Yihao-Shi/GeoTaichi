"""Independent FEM/IGA examples keep equivalent geometry and soil state."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline

from examples.fempm.wavy_plate_collapse import wavy_plate_collapse as fem
from examples.igampm.cpt_dp import cpt_dp as cpt
from examples.igampm.flexible_barrier import flexible_barrier as barrier
from examples.igampm.wavy_plate_collapse import wavy_plate_collapse as iga
from examples.igampm.wavy_plate_collapse.wavy_plate_collapse import (
    parameters,
    plate_control_points,
    plate_mesh,
    plate_section,
    soil_particles,
    wave,
)


def test_independent_wave_geometry_mass_and_fixed_half():
    p = parameters(
        SimpleNamespace(
            method="igampm",
            amplitude=0.06,
            spacing=0.1,
            dt=0.001,
            time=1.0,
            save_interval=0.02,
            cycles=15,
            samples_per_period=8,
            width=3.0,
            height=3.0,
            soil_height=2.0,
            height_segments=12,
            contact_capacity=32768,
        )
    )
    for module in (fem, iga, cpt, barrier):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("examples")
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("examples", "runpy", "importlib")) for alias in node.names)
    fem_p = fem.parameters(SimpleNamespace(**p))
    iga_p = iga.parameters(SimpleNamespace(**p))
    assert fem_p["method"] == "fempm" and iga_p["method"] == "igampm"
    assert p["gravity"] == [0.0, 0.0, -9.81]
    dr = cpt.GRID_SIZE / cpt.PARTICLES_PER_CELL
    radius = (np.arange(round(cpt.SOIL_SIZE[0] / dr)) + 0.5) * dr
    top = np.column_stack((radius, np.full(len(radius), cpt.SOIL_SIZE[1] - 0.5 * dr)))
    particle_ids, annular_area = cpt.axisymmetric_surface_pressure_particles(top, cpt.GRID_SIZE)
    assert len(particle_ids) == len(radius)
    np.testing.assert_allclose(annular_area.sum(), np.pi * cpt.SOIL_SIZE[0] ** 2, rtol=1e-14)
    assert cpt.dp_direct_material(dilation_angle=0)["DilationAngle"] == 0
    for left, right in zip(fem.soil_particles(fem_p), iga.soil_particles(iga_p)):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(fem.plate_mesh(fem_p)[:2], iga.plate_mesh(iga_p)[:2]):
        np.testing.assert_array_equal(left, right)
    solid, cells, quality = plate_mesh(p)
    points, _, _ = plate_section(p)
    assert quality["minimum_quality"] > 0.02 and np.all(cells >= 0) and cells.max() < len(solid)
    t = solid[cells]
    assert np.all(np.linalg.det(np.stack([t[:, i] - t[:, 0] for i in (1, 2, 3)], axis=-1)) > 0)
    control, knots, fixed = plate_control_points(p)
    nz = p["height_segments"] + 1
    ny = len(control) // (2 * nz)
    y = np.linspace(0, p["width"], 2401)
    basis = BSpline(knots, np.eye(ny), 3)(y / p["width"])
    evaluated = np.einsum("ij,jk->ik", basis, control.reshape(nz, ny, 2, 3)[0, :, 0, :2])
    np.testing.assert_allclose(evaluated[:, 1], y + p["y_origin"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(evaluated[:, 0], wave(p)(y), atol=1e-12, rtol=0)
    assert np.count_nonzero(np.diff(np.sign(np.diff(evaluated[:, 0])))) == 29
    assert np.ptp(evaluated[:, 0]) >= 2 * p["amplitude"] - 1e-12
    z_knots = np.r_[0.0, np.linspace(0.0, 1.0, nz), 1.0]
    z = np.linspace(0, 1, 101)
    z_basis = BSpline(z_knots, np.eye(nz), 1)(z)
    free_layers = np.setdiff1d(np.arange(nz), fixed // (2 * ny))
    np.testing.assert_allclose(z_basis[z >= 0.5][:, free_layers], 0, atol=1e-14)
    assert np.max(z_basis[z < 0.5][:, free_layers]) > 0.99
    assert np.ptp(solid[:, 2]) == p["height"]
    assert len(np.unique(solid[:, 2])) == nz
    assert np.all(control[fixed, 2] >= p["floor"] + 0.5 * p["height"] - 1e-12)
    soil, volumes, measures = soil_particles(p)
    local_y = soil[:, 1] - p["y_origin"]
    np.testing.assert_array_less(soil[:, 0], wave(p)(local_y))
    np.testing.assert_array_less(soil[:, 0], np.interp(local_y, points[::3, 1] - p["y_origin"], points[::3, 0]))
    assert np.all(volumes > 0) and np.all(measures > 0)
    # Particle row quadrature integrates the declared initial soil area.
    row_y = np.unique(local_y)
    right = np.minimum(wave(p)(row_y), np.interp(row_y, points[::3, 1] - p["y_origin"], points[::3, 0]))
    expected = (right - 0.1 * p["spacing"] / p["ppc"] - p["soil_left"]).sum() * p["width"] / len(row_y)
    np.testing.assert_allclose(volumes.sum(), expected * p["soil_height"], atol=1e-12)
