import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import src.mpm.config as config

config.set_dimension(2)

from src.mpm.generator.Body import Body
from src.mpm.engines.direct.StaticTwoPhaseULMPM import StaticTwoPhaseULMPM
from src.mpm.boundaries.BoundaryCondition import DirichletBoundary, NeumannBoundary


def top_strip_nodes(coords, y_top, x_min, x_max):
    return np.where(
        np.isclose(coords[:, 1], y_top)
        & (coords[:, 0] >= x_min - 1.0e-12)
        & (coords[:, 0] <= x_max + 1.0e-12)
    )[0]


def build_solver():
    domain_width = 6.0
    domain_height = 3.0
    soil_width = 5.0
    soil_height = 2.5
    soil_start = np.array([0.5, 0.0], dtype=np.float64)
    soil_end = soil_start + np.array([soil_width, soil_height], dtype=np.float64)
    footing_width = 1.0
    footing_center_x = 0.5 * (soil_start[0] + soil_end[0])
    footing_x_min = footing_center_x - 0.5 * footing_width
    footing_x_max = footing_center_x + 0.5 * footing_width

    dx = 0.1
    ppc = 2
    dt = 1.0

    body = Body()
    body.add_rectangle(
        soil_start.tolist(),
        soil_end.tolist(),
        dx,
        ppc=ppc,
        init_v=[0.0, 0.0],
    )

    n_grid_x = int(round(domain_width / dx)) + 1
    n_grid_y = int(round(domain_height / dx)) + 1
    xs = np.arange(n_grid_x, dtype=np.float64) * dx
    ys = np.arange(n_grid_y, dtype=np.float64) * dx
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    component = 3
    bottom = np.where(np.isclose(coords[:, 1], 0.0))[0]
    left = np.where(np.isclose(coords[:, 0], soil_start[0]))[0]
    right = np.where(np.isclose(coords[:, 0], soil_end[0]))[0]
    top_all = np.where(
        np.isclose(coords[:, 1], soil_end[1])
        & (coords[:, 0] >= soil_start[0] - 1.0e-12)
        & (coords[:, 0] <= soil_end[0] + 1.0e-12)
    )[0]
    top_strip = top_strip_nodes(coords, soil_end[1], footing_x_min, footing_x_max)

    dirichlet = DirichletBoundary()
    dbc_ids = []
    dbc_vals = []
    dbc_ids.append(list(component * bottom + 1))
    dbc_vals.extend([0.0] * len(bottom))
    side_nodes = np.unique(np.concatenate([left, right]))
    dbc_ids.append(list(component * side_nodes + 0))
    dbc_vals.extend([0.0] * len(side_nodes))
    dbc_ids.append(list(component * top_all + 2))
    dbc_vals.extend([0.0] * len(top_all))
    dirichlet.append(dbc_ids, dbc_vals)

    strip_sorted = top_strip[np.argsort(coords[top_strip, 0])]
    strip_force_ids = []
    strip_force_weights = []
    for i, node in enumerate(strip_sorted):
        strip_force_ids.append(component * node + 1)
        strip_force_weights.append(0.5 * dx if i == 0 or i == len(strip_sorted) - 1 else dx)

    neumann = NeumannBoundary()
    neumann.append([strip_force_ids], [0.0] * len(strip_force_ids))
    solver = StaticTwoPhaseULMPM(
        domain=[domain_width, domain_height],
        dx=dx,
        dt=dt,
        step=2,
        interval=1,
        bodies=body,
        dirichlet=dirichlet,
        neumann=neumann,
        gravity=[0.0, 0.0],
        material="druckerPrager",
        young_modulus=2.0e4,
        poisson_ratio=0.3,
        density=2.0,
        fluid_density=1.0,
        porosity=0.4,
        mobility=1.0e-7,
        friction_angle=25.0,
        dilation_angle=0.0,
        cohesion=0.0,
        shape_factor=0.0,
        residual=1.0e-4,
        rhs_tolerance_abs=5.0e-2,
        rhs_tolerance_rel=1.0e-1,
        max_iters=20,
        line_search=True,
        line_search_max_backtrack=10,
        ppd=ppc,
        shape_function="gimp",
        visualize=False,
        path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "StripFootingDruckerPragerDiag"),
    )
    solver.initial_simulation()
    return solver, np.array(strip_force_weights, dtype=np.float64)


def prepare_second_step_entrance(solver, strip_force_weights):
    solver.neumann.value.from_numpy(strip_force_weights * -0.5)
    ok = solver.substep(verbose=True)
    print(
        f"step1 accepted={ok}, delta_inf={solver.last_delta_inf:.6e}, "
        f"rhs_inf={solver.last_rhs_inf:.6e}, rhs_target={solver.last_rhs_target:.6e}"
    )
    solver.refresh_active_dofs()
    solver.reset_step_solution()
    solver.assemble_pressure_projection()
    solver.apply_pressure_projection_to_solution()
    solver.neumann.value.from_numpy(strip_force_weights * -1.0)


def block_norms(K, component):
    n_node = K.shape[0] // component
    for row_comp in range(component):
        for col_comp in range(component):
            rows = np.arange(row_comp, component * n_node, component)
            cols = np.arange(col_comp, component * n_node, component)
            block = K[rows][:, cols]
            print(
                f"block({row_comp},{col_comp}) nnz={block.nnz:7d} "
                f"maxabs={np.max(np.abs(block.data)) if block.nnz > 0 else 0.0:.6e}"
            )


def diagnose_matrix(K, r):
    row_nnz = K.getnnz(axis=1)
    col_nnz = K.getnnz(axis=0)
    zero_rows = np.where(row_nnz == 0)[0]
    zero_cols = np.where(col_nnz == 0)[0]
    print(f"matrix shape={K.shape}, nnz={K.nnz}, rhs_inf={np.linalg.norm(r, np.inf):.6e}")
    print(f"zero_rows={zero_rows.size}, zero_cols={zero_cols.size}")
    if zero_rows.size > 0:
        print("zero_row_ids", zero_rows[:40])
    if zero_cols.size > 0:
        print("zero_col_ids", zero_cols[:40])
    block_norms(K, 3)


def compare_direct_solves(K, r):
    rhs = -r
    for reg in (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
        A = K if reg == 0.0 else K + reg * eye(K.shape[0], format="csr")
        x = spsolve(A, rhs)
        step_inf = np.linalg.norm(x, np.inf)
        lin_res = np.linalg.norm(A @ x - rhs, np.inf)
        print(
            f"direct reg={reg:.1e} step_inf={step_inf:.6e} "
            f"lin_res_inf={lin_res:.6e}"
        )


if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False)
    solver, strip_force_weights = build_solver()
    prepare_second_step_entrance(solver, strip_force_weights)
    K, r = solver.assemble_current_system_arrays()
    diagnose_matrix(K, r)
    compare_direct_solves(K, r)
