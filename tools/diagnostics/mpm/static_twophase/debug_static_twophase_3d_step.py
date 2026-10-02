import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import src.mpm.config as config

config.set_dimension(3)

from src.mpm.generator.Body import Body
from src.mpm.engines.direct.StaticTwoPhaseULMPM import StaticTwoPhaseULMPM
from src.mpm.boundaries.BoundaryCondition import DirichletBoundary, NeumannBoundary


def build_solver():
    domain = [0.6, 0.6, 12.0]
    width = 0.1
    depth_y = 0.1
    height = 10.0
    start_x = 0.1
    start_y = 0.1
    start_z = 0.0
    dx = 0.1
    dt = 100.0
    ppc = 2

    body = Body()
    body.add_cube(
        [start_x, start_y, start_z],
        [start_x + width, start_y + depth_y, start_z + height],
        dx,
        ppc=ppc,
        init_v=[0.0, 0.0, 0.0],
    )

    n_grid_x = int(round(domain[0] / dx)) + 1
    n_grid_y = int(round(domain[1] / dx)) + 1
    n_grid_z = int(round(domain[2] / dx)) + 1
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
        top_force_vals.append(-dx * dx * wx * wy)
    neumann.append([top_force_ids], top_force_vals)

    return StaticTwoPhaseULMPM(
        domain=domain,
        dx=dx,
        dt=dt,
        step=1,
        interval=1,
        bodies=body,
        dirichlet=dirichlet,
        neumann=neumann,
        gravity=[0.0, 0.0, 0.0],
        material="neoHookean",
        young_modulus=1.5e3,
        poisson_ratio=0.25,
        density=1.0,
        fluid_density=1.0,
        mobility=1.0e-6,
        residual=1.0e-10,
        max_iters=20,
        line_search=True,
        ppd=ppc,
        shape_function="gimp",
        visualize=False,
        path="/tmp/static_twophase_debug_3d",
    )


if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False)
    solver = build_solver()
    solver.initial_simulation()
    steps = int(os.environ.get("DEBUG_STEPS", "1"))
    particle_num = solver.particleNum.to_numpy()[0]

    for step in range(steps):
        solver.refresh_active_dofs()
        if step == 0:
            print("active_dof", solver.active_dof)
            print("supported_nodes", int(solver.supported_node.to_numpy().sum()))

        solver.reset_step_solution()
        solver.assemble_pressure_projection()
        solver.apply_pressure_projection_to_solution()

        p0 = solver.old_solution.to_numpy()[: solver.active_dof]
        print(f"step {step + 1} projection_finite", np.isfinite(p0).all(), "projection_maxabs", np.max(np.abs(p0)) if p0.size else 0.0)

        solver.solve_current_step(verbose=True)
        j = solver.J_new.to_numpy()[:particle_num]
        p = solver.particle_pressure_new.to_numpy()[:particle_num]
        s = solver.stress_new.to_numpy()[:particle_num]
        print(f"step {step + 1} J finite/min/max", np.isfinite(j).all(), np.min(j), np.max(j))
        print(f"step {step + 1} p finite/min/max", np.isfinite(p).all(), np.min(p), np.max(p))
        print(f"step {step + 1} stress finite", np.isfinite(s).all())
        if (not np.isfinite(j).all()) or (not np.isfinite(p).all()) or (not np.isfinite(s).all()):
            break
        solver.commit_step()
