import os
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import taichi as ti

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


def make_case(
    case_name,
    linear_solver,
    regularization,
    preconditioner,
    target_strip_load,
    initial_load_increment,
    min_load_increment,
    max_accepted_steps,
):
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

    max_accepted_delta_inf = 1.0e2

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

    case_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "solver_compare_outputs",
        case_name,
    )
    os.makedirs(case_dir, exist_ok=True)
    for old_vtu in glob(os.path.join(case_dir, "vtks", "particles*.vtu")):
        os.remove(old_vtu)

    neumann = NeumannBoundary()
    neumann.append([strip_force_ids], [0.0] * len(strip_force_ids))
    solver = StaticTwoPhaseULMPM(
        domain=[domain_width, domain_height],
        dx=dx,
        dt=dt,
        step=max_accepted_steps,
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
        require_both_convergence_checks=True,
        linear_solver=linear_solver,
        direct_regularization=regularization,
        iterative_regularization=regularization,
        iterative_preconditioner=preconditioner,
        iterative_rtol=1.0e-8,
        iterative_atol=1.0e-10,
        iterative_maxiter=500,
        ppd=ppc,
        shape_function="gimp",
        visualize=False,
        path=case_dir,
    )
    return (
        solver,
        np.array(strip_force_weights, dtype=np.float64),
        target_strip_load,
        initial_load_increment,
        min_load_increment,
        max_accepted_steps,
        max_accepted_delta_inf,
    )


def run_case(
    case_name,
    linear_solver,
    regularization,
    preconditioner,
    target_strip_load,
    initial_load_increment,
    min_load_increment,
    max_accepted_steps,
):
    (
        solver,
        strip_force_weights,
        target_strip_load,
        initial_load_increment,
        min_load_increment,
        max_accepted_steps,
        max_accepted_delta_inf,
    ) = make_case(
        case_name,
        linear_solver,
        regularization,
        preconditioner,
        target_strip_load,
        initial_load_increment,
        min_load_increment,
        max_accepted_steps,
    )

    solver.initial_simulation()
    current_load = 0.0
    load_increment = initial_load_increment
    accepted_steps = 0
    attempted_steps = 0
    failure_reason = ""
    max_newton_iters = 0

    while current_load > target_strip_load + 1.0e-12 and accepted_steps < max_accepted_steps:
        attempted_steps += 1
        step_state = solver.get_solver_state()
        target_load = max(current_load + load_increment, target_strip_load)
        solver.neumann.value.from_numpy(strip_force_weights * target_load)
        try:
            converged = solver.substep(verbose=False)
        except Exception as exc:
            converged = False
            failure_reason = f"{type(exc).__name__}"
        accepted = converged and solver.last_delta_inf < max_accepted_delta_inf

        if not accepted:
            solver.set_solver_state(step_state)
            if abs(load_increment) <= min_load_increment + 1.0e-12:
                if failure_reason == "":
                    failure_reason = "reject_at_min_increment"
                break
            load_increment *= 0.5
            continue

        failure_reason = ""
        current_load = target_load
        accepted_steps += 1
        max_newton_iters = max(max_newton_iters, solver.last_iterations)
        if solver.last_iterations <= 2 and abs(load_increment) < abs(initial_load_increment):
            load_increment = max(2.0 * load_increment, initial_load_increment)

    return {
        "case": case_name,
        "linear_solver": linear_solver,
        "preconditioner": preconditioner,
        "regularization": regularization,
        "accepted_steps": accepted_steps,
        "attempted_steps": attempted_steps,
        "final_load": current_load,
        "reached_target": current_load <= target_strip_load + 1.0e-12,
        "max_newton_iters": max_newton_iters,
        "last_delta_inf": solver.last_delta_inf,
        "last_rhs_inf": solver.last_rhs_inf,
        "failure_reason": failure_reason,
    }


if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False)
    all_cases = [
        ("direct_reg_1e6", "direct", 1.0e-6, "none"),
        ("gmres_diag_1e10", "gmres", 1.0e-10, "diag"),
        ("gmres_block_1e10", "gmres", 1.0e-10, "block_diag"),
        ("gmres_schur_1e10", "gmres", 1.0e-10, "schur_lu"),
        ("gmres_ilu_1e10", "gmres", 1.0e-10, "ilu"),
        ("minres_diag_1e10", "minres", 1.0e-10, "diag"),
    ]
    target_strip_load = float(os.environ.get("TARGET_STRIP_LOAD", "-1.0"))
    initial_load_increment = float(os.environ.get("INITIAL_LOAD_INCREMENT", "-0.5"))
    min_load_increment = float(os.environ.get("MIN_LOAD_INCREMENT", "0.03125"))
    max_accepted_steps = int(os.environ.get("MAX_ACCEPTED_STEPS", "32"))
    case_filter = os.environ.get("COMPARE_CASES", "")
    allowed = {name.strip() for name in case_filter.split(",") if name.strip()}
    cases = [case for case in all_cases if not allowed or case[0] in allowed]
    results = []
    for case_name, linear_solver, regularization, preconditioner in cases:
        result = run_case(
            case_name,
            linear_solver,
            regularization,
            preconditioner,
            target_strip_load,
            initial_load_increment,
            min_load_increment,
            max_accepted_steps,
        )
        results.append(result)
        print(
            f"{case_name:18s} solver={linear_solver:6s} prec={preconditioner:4s} "
            f"reg={regularization:.1e} reached={result['reached_target']} "
            f"final_load={result['final_load']:.6e} accepted={result['accepted_steps']:2d} "
            f"attempted={result['attempted_steps']:2d} max_newton={result['max_newton_iters']:2d} "
            f"failure={result['failure_reason'] or '-'}",
            flush=True,
        )
