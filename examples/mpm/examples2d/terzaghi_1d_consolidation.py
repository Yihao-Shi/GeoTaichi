import os
import sys

import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

import src.mpm.config as config

config.set_dimension(2)

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
    depth = height - positions[:, 1]
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


def save_profiles(out_dir, saved_steps, saved_profiles, dt, cv, height, surcharge):
    if not saved_profiles:
        return
    np.savez(
        os.path.join(out_dir, "terzaghi_profiles.npz"),
        profiles=np.array(saved_profiles, dtype=object),
        steps=np.array(saved_steps, dtype=np.int32),
        dt=dt,
        cv=cv,
        height=height,
        surcharge=surcharge,
    )


def load_saved_profiles(out_dir):
    profile_path = os.path.join(out_dir, "terzaghi_profiles.npz")
    if not os.path.exists(profile_path):
        return [], []
    data = np.load(profile_path, allow_pickle=True)
    return list(data["steps"].astype(np.int32)), [np.asarray(profile, dtype=float) for profile in data["profiles"]]


def save_checkpoint(path, solver, step):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = solver.get_solver_state()
    state["step"] = np.array(step, dtype=np.int32)
    np.savez(path, **state)


def load_checkpoint(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


if __name__ == "__main__":
    cpu_threads = os.environ.get("GT_TAICHI_THREADS")
    init(
        dim=2,
        arch="cpu",
        default_fp="float64",
        debug=False,
        log=False,
        cpu_max_num_threads=int(cpu_threads) if cpu_threads is not None else 0,
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))

    domain_width = 0.6
    domain_height = 12.0
    width = 0.1
    height = 10.0
    start_x = 0.1
    start_y = 0.0
    dx = 0.1
    dt = 100.0
    total_step = int(os.environ.get("GT_TERZAGHI_STEPS", "600"))
    record_interval = int(os.environ.get("GT_TERZAGHI_RECORD_INTERVAL", "0"))
    if record_interval < 0:
        raise ValueError("GT_TERZAGHI_RECORD_INTERVAL must be non-negative")
    ppc = 2

    young_modulus = 1.5e3
    poisson_ratio = 0.25
    density = 1.0
    fluid_density = 1.0
    mobility = 1.0e-6
    surcharge = 1.0
    material_model = "neoHookean"

    domain = [domain_width, domain_height]
    body = Body()
    body.add_rectangle([start_x, start_y], [start_x + width, start_y + height], dx, ppc=ppc, init_v=[0.0, 0.0])

    n_grid_x = int(round(domain_width / dx)) + 1
    n_grid_y = int(round(domain_height / dx)) + 1
    xs = np.arange(n_grid_x, dtype=np.float64) * dx
    ys = np.arange(n_grid_y, dtype=np.float64) * dx
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    component = 3
    dirichlet = DirichletBoundary()
    dbc_ids = []
    dbc_vals = []

    x_fix = np.arange(coords.shape[0], dtype=np.int32)
    dbc_ids.append(list(component * x_fix + 0))
    dbc_vals.extend([0.0] * len(x_fix))

    bottom = np.where(np.isclose(coords[:, 1], 0.0))[0]
    dbc_ids.append(list(component * bottom + 1))
    dbc_vals.extend([0.0] * len(bottom))

    top = np.where(
        np.isclose(coords[:, 1], height)
        & (coords[:, 0] >= start_x - 1.0e-12)
        & (coords[:, 0] <= start_x + width + 1.0e-12)
    )[0]
    dbc_ids.append(list(component * top + 2))
    dbc_vals.extend([0.0] * len(top))
    dirichlet.append(dbc_ids, dbc_vals)

    neumann = NeumannBoundary()
    top_sorted = top[np.argsort(coords[top, 0])]
    top_force_ids = []
    top_force_vals = []
    edge_force = -0.5 * surcharge * dx
    interior_force = -surcharge * dx
    for i, node in enumerate(top_sorted):
        top_force_ids.append(component * node + 1)
        top_force_vals.append(edge_force if i == 0 or i == len(top_sorted) - 1 else interior_force)
    neumann.append([top_force_ids], top_force_vals)

    out_dir = os.environ.get("GT_TERZAGHI_OUTPUT_PATH", os.path.join(script_dir, "Terzaghi1D"))
    os.makedirs(out_dir, exist_ok=True)
    visualize = os.environ.get("GT_TERZAGHI_VISUALIZE", "1") == "1"
    verbose = os.environ.get("GT_TERZAGHI_VERBOSE", "1") == "1"
    restart_file = os.environ.get("GT_TERZAGHI_RESTART_FILE")
    checkpoint_file = os.environ.get("GT_TERZAGHI_CHECKPOINT_FILE")
    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        static_twophase=True,
        solver_type="Implicit",
        configuration="ULMPM",
        domain=domain,
        gravity=[0.0, 0.0],
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
        "max_iters": 8,
        "line_search": True,
        "ppd": ppc,
        "visualize": visualize,
        "path": out_dir,
    })
    mpm.add_engine()
    solver = mpm.enginer

    solver.initial_simulation()
    saved_steps, saved_profiles = load_saved_profiles(out_dir) if restart_file else ([], [])
    start_step = int(os.environ.get("GT_TERZAGHI_START_STEP", "0"))
    if restart_file:
        state = load_checkpoint(restart_file)
        solver.set_solver_state(state)
        if "GT_TERZAGHI_START_STEP" not in os.environ and "step" in state:
            start_step = int(np.asarray(state["step"]))
    sample_steps = sorted({step for step in [55, 111, 222, 333, 555, total_step] if step <= total_step})
    cv = (young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)) + young_modulus / (1.0 + poisson_ratio)) * mobility

    # Preserve the compact six-profile default, while allowing a gallery run
    # to record a smooth pressure-dissipation animation at a fixed cadence.
    if record_interval > 0 and start_step == 0:
        solver.record()

    for step in range(start_step, total_step):
        solver.substep(verbose=verbose)
        is_profile_step = step + 1 in sample_steps
        is_gallery_frame = record_interval > 0 and (step + 1) % record_interval == 0
        if is_profile_step or is_gallery_frame:
            solver.record()
        if is_profile_step:
            pos = solver.particle.x.to_numpy()[: solver.particleNum.to_numpy()[0]]
            pressure = solver.particle_pressure.to_numpy()[: solver.particleNum.to_numpy()[0]]
            depth, numerical, mask = average_pressure_profile(pos, pressure, height)
            analytical = terzaghi_series(depth, (step + 1) * dt, height, surcharge, cv)
            abs_err = np.linalg.norm(numerical[mask] - analytical[mask])
            rel = abs_err / max(np.linalg.norm(analytical[mask]), 1.0e-12)
            saved_profiles.append(np.column_stack([depth, numerical, analytical]))
            saved_steps.append(step + 1)
            save_profiles(out_dir, saved_steps, saved_profiles, dt, cv, height, surcharge)
            print(f"step={step + 1:3d}, time={dt * (step + 1):8.4f}, abs_profile_error={abs_err:8.3e}, relative_profile_error={rel:8.3e}")

    save_checkpoint(checkpoint_file, solver, total_step)
    ti.reset()
