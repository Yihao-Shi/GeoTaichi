import os
import sys

import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

import src.mpm.config as config

config.set_dimension(3)

from src.mpm.generator.Body import Body
from src.mpm.boundaries.BoundaryCondition import DirichletBoundary, NeumannBoundary


def terzaghi_series(depth_from_top, time_value, height, load, cv, terms=200):
    series = np.zeros_like(depth_from_top)
    for m in range(terms):
        n = 2 * m + 1
        coeff = 4.0 / (n * np.pi)
        series += coeff * np.sin(n * np.pi * depth_from_top / (2.0 * height)) * np.exp(-(n * n) * np.pi * np.pi * cv * time_value / (4.0 * height * height))
    return load * series


def average_pressure_profile(positions, pressure, height, n_bins=40):
    depth = height - positions[:, 2]
    bins = np.linspace(0.0, height, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int32)
    ids = np.clip(np.digitize(depth, bins) - 1, 0, n_bins - 1)
    for pid, bid in enumerate(ids):
        values[bid] += pressure[pid]
        counts[bid] += 1
    mask = counts > 0
    values[mask] /= counts[mask]
    return centers, values, mask


if __name__ == "__main__":
    init(dim=3, arch="cpu", default_fp="float64", debug=False, log=False)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    domain_width = 0.6
    domain_depth = 0.6
    domain_height = 12.0
    width = 0.1
    depth_y = 0.1
    height = 10.0
    start_x = 0.1
    start_y = 0.1
    start_z = 0.0
    dx = 0.1
    dt = 100.0
    total_step = 600
    ppc = 2

    young_modulus = 1.5e3
    poisson_ratio = 0.25
    density = 1.0
    fluid_density = 1.0
    mobility = 1.0e-6
    surcharge = 1.0
    material_model = "neoHookean"

    domain = [domain_width, domain_depth, domain_height]
    body = Body()
    body.add_cube(
        [start_x, start_y, start_z],
        [start_x + width, start_y + depth_y, start_z + height],
        dx,
        ppc=ppc,
        init_v=[0.0, 0.0, 0.0],
    )

    n_grid_x = int(round(domain_width / dx)) + 1
    n_grid_y = int(round(domain_depth / dx)) + 1
    n_grid_z = int(round(domain_height / dx)) + 1
    xs = np.arange(n_grid_x, dtype=np.float64) * dx
    ys = np.arange(n_grid_y, dtype=np.float64) * dx
    zs = np.arange(n_grid_z, dtype=np.float64) * dx
    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing="ij")
    coords = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    component = 4
    dirichlet = DirichletBoundary()
    dbc_ids = []
    dbc_vals = []

    all_nodes = np.arange(coords.shape[0], dtype=np.int32)
    dbc_ids.append(list(component * all_nodes + 0))
    dbc_vals.extend([0.0] * len(all_nodes))
    dbc_ids.append(list(component * all_nodes + 1))
    dbc_vals.extend([0.0] * len(all_nodes))

    bottom = np.where(np.isclose(coords[:, 2], 0.0))[0]
    dbc_ids.append(list(component * bottom + 2))
    dbc_vals.extend([0.0] * len(bottom))

    top = np.where(
        np.isclose(coords[:, 2], height)
        & (coords[:, 0] >= start_x - 1.0e-12)
        & (coords[:, 0] <= start_x + width + 1.0e-12)
        & (coords[:, 1] >= start_y - 1.0e-12)
        & (coords[:, 1] <= start_y + depth_y + 1.0e-12)
    )[0]
    dbc_ids.append(list(component * top + 3))
    dbc_vals.extend([0.0] * len(top))
    dirichlet.append(dbc_ids, dbc_vals)

    neumann = NeumannBoundary()
    top_force_ids = []
    top_force_vals = []
    for node in top:
        x, y, _ = coords[node]
        wx = 0.5 if np.isclose(x, start_x) or np.isclose(x, start_x + width) else 1.0
        wy = 0.5 if np.isclose(y, start_y) or np.isclose(y, start_y + depth_y) else 1.0
        top_force_ids.append(component * node + 2)
        top_force_vals.append(-surcharge * dx * dx * wx * wy)
    neumann.append([top_force_ids], top_force_vals)

    out_dir = os.path.join(script_dir, "Terzaghi1D")
    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        static_twophase=True,
        solver_type="Implicit",
        configuration="ULMPM",
        domain=domain,
        gravity=[0.0, 0.0, 0.0],
    )
    mpm.add_body(body)
    mpm.add_boundary_condition(dirichlet=dirichlet, neumann=neumann)
    mpm.add_material(
        model=material_model,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        fluid_density=fluid_density,
        mobility=mobility,
    )
    mpm.add_element({"ElementSize": dx, "ShapeFunction": "gimp"})
    mpm.set_solver({
        "dt": dt,
        "step": total_step,
        "interval": 1,
        "residual": 1.0e-10,
        "max_iters": 20,
        "line_search": True,
        "ppd": ppc,
        "visualize": True,
        "path": out_dir,
    })
    mpm.add_engine()
    solver = mpm.enginer

    solver.initial_simulation()
    saved_profiles = []
    sample_steps = [55, 111, 222, 333, 555, total_step]
    constrained_modulus = young_modulus * (1.0 - poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    cv = constrained_modulus * mobility

    for step in range(total_step):
        solver.substep(verbose=True)
        if step + 1 in sample_steps:
            solver.record()
            pos = solver.particle.x.to_numpy()[: solver.particleNum.to_numpy()[0]]
            pressure = solver.particle_pressure.to_numpy()[: solver.particleNum.to_numpy()[0]]
            depth, numerical, mask = average_pressure_profile(pos, pressure, height)
            analytical = terzaghi_series(depth, (step + 1) * dt, height, surcharge, cv)
            abs_err = np.linalg.norm(numerical[mask] - analytical[mask])
            rel = abs_err / max(np.linalg.norm(analytical[mask]), 1.0e-12)
            saved_profiles.append(np.column_stack([depth, numerical, analytical]))
            print(f"step={step + 1:3d}, time={dt * (step + 1):8.4f}, abs_profile_error={abs_err:8.3e}, relative_profile_error={rel:8.3e}")

    if saved_profiles:
        np.savez(
            os.path.join(out_dir, "terzaghi_profiles.npz"),
            profiles=np.array(saved_profiles, dtype=object),
            steps=np.array(sample_steps, dtype=np.int32),
            dt=dt,
            cv=cv,
            height=height,
            surcharge=surcharge,
        )
