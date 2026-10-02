import os
import sys

from glob import glob

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


def build_grid_coords(domain_width, domain_height, dx):
    n_grid_x = int(round(domain_width / dx)) + 1
    n_grid_y = int(round(domain_height / dx)) + 1
    xs = np.arange(n_grid_x, dtype=np.float64) * dx
    ys = np.arange(n_grid_y, dtype=np.float64) * dx
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def build_mechanical_dirichlet(coords, component):
    bottom = np.where(np.isclose(coords[:, 1], 0.0))[0]
    symmetry = np.where(np.isclose(coords[:, 0], 0.0))[0]

    dbc_ids = []
    dbc_vals = []

    # Rigid bottom.
    dbc_ids.append(list(component * bottom + 0))
    dbc_vals.extend([0.0] * len(bottom))
    dbc_ids.append(list(component * bottom + 1))
    dbc_vals.extend([0.0] * len(bottom))

    # Symmetry plane at x = 0.
    symmetry_only = np.setdiff1d(symmetry, bottom, assume_unique=False)
    dbc_ids.append(list(component * symmetry_only + 0))
    dbc_vals.extend([0.0] * len(symmetry_only))

    dirichlet = DirichletBoundary()
    dirichlet.append(dbc_ids, dbc_vals)
    return dirichlet


def add_pressure_dirichlet(dirichlet, coords, component, domain_width, domain_height):
    top = np.where(np.isclose(coords[:, 1], domain_height))[0]
    right = np.where(np.isclose(coords[:, 0], domain_width))[0]
    drained_nodes = np.unique(np.concatenate([top, right]))
    dirichlet.append([list(component * drained_nodes + 2)], [0.0] * drained_nodes.shape[0])


def max_abs_pressure(state):
    if state["pressure"].size == 0:
        return 0.0
    return float(np.max(np.abs(state["pressure"])))


def mean_top_settlement(state, initial_position, domain_height, dx):
    mask = initial_position[:, 1] >= domain_height - 1.5 * dx
    if not np.any(mask):
        return 0.0
    return float(np.mean(initial_position[mask, 1] - state["position"][mask, 1]))


def clear_vtu_files(path):
    os.makedirs(path, exist_ok=True)
    for old_file in glob(os.path.join(path, "vtks", "particles*.vtu")):
        os.remove(old_file)


def build_solver(
    body,
    dirichlet,
    neumann,
    domain_width,
    domain_height,
    dx,
    dt,
    total_step,
    ppc,
    gravity,
    young_modulus,
    poisson_ratio,
    solid_density,
    fluid_density,
    porosity,
    mobility,
    out_dir,
):
    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        static_twophase=True,
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[domain_width, domain_height],
        gravity=gravity,
    )
    mpm.add_body(body)
    mpm.add_boundary_condition(dirichlet=dirichlet, neumann=neumann)
    mpm.add_material(
        model="neoHookean",
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=solid_density,
        fluid_density=fluid_density,
        porosity=porosity,
        mobility=mobility,
    )
    mpm.add_element({"ElementSize": dx, "ShapeFunction": "gimp"})
    mpm.set_solver({
        "dt": dt,
        "step": total_step,
        "interval": 1,
        "residual": 1.0e-8,
        "rhs_tolerance_abs": 1.0e-8,
        "rhs_tolerance_rel": 1.0e-6,
        "max_iters": 40,
        "line_search": True,
        "line_search_max_backtrack": 10,
        "require_both_convergence_checks": False,
        "linear_solver": "direct",
        "direct_regularization": 1.0e-8,
        "ppd": ppc,
        "visualize": True,
        "path": out_dir,
    })
    mpm.add_engine()
    return mpm.enginer


if __name__ == "__main__":
    init(dim=2, arch="cpu", default_fp="float64", debug=False, log=False)

    discretizations = {
        "coarse": {"dx": 0.125, "ppc": 5},
        "fine": {"dx": 0.0625, "ppc": 4},
    }
    case_key = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    if case_key not in discretizations:
        raise ValueError(f"Unknown discretization '{case_key}'. Choose from {sorted(discretizations)}.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "SelfWeightConsolidationSection53", case_key)
    consolidation_dir = os.path.join(root_dir, "consolidation")
    clear_vtu_files(consolidation_dir)
    history_path = os.path.join(root_dir, "consolidation_history.csv")
    if os.path.exists(history_path):
        os.remove(history_path)

    domain_width = 2.0
    domain_height = 2.0
    dx = discretizations[case_key]["dx"]
    ppc = discretizations[case_key]["ppc"]

    bulk_modulus = 15.0
    poisson_ratio = 0.3
    young_modulus = 3.0 * bulk_modulus * (1.0 - 2.0 * poisson_ratio)
    permeability0 = 1.0e-14
    fluid_viscosity = 1.0e-6
    mobility = permeability0 / fluid_viscosity
    porosity = 0.5
    solid_density = 2.6
    fluid_density = 1.0
    gravity = [0.0, -9.81]

    body = Body()
    body.add_rectangle([0.0, 0.0], [domain_width, domain_height], dx, ppc=ppc, init_v=[0.0, 0.0])
    coords = build_grid_coords(domain_width, domain_height, dx)
    component = 3

    first_body = next(iter(body.bodies.values()))
    initial_position = first_body["points"].copy()

    drained_dirichlet = build_mechanical_dirichlet(coords, component)
    add_pressure_dirichlet(drained_dirichlet, coords, component, domain_width, domain_height)
    consolidation_solver = build_solver(
        body=body,
        dirichlet=drained_dirichlet,
        neumann=None,
        domain_width=domain_width,
        domain_height=domain_height,
        dx=dx,
        dt=0.1,
        total_step=80,
        ppc=ppc,
        gravity=gravity,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        solid_density=solid_density,
        fluid_density=fluid_density,
        porosity=porosity,
        mobility=mobility,
        out_dir=consolidation_dir,
    )
    consolidation_solver.initial_simulation()

    time_value = 0.0
    dt_value = 0.1
    growth = 1.2
    pressure_tol = 1.0e-3
    history = []

    for step in range(1, 81):
        consolidation_solver.dt = dt_value
        converged = consolidation_solver.substep(verbose=True)
        if not converged:
            raise RuntimeError(
                f"Consolidation step {step} failed: "
                f"delta_inf={consolidation_solver.last_delta_inf:.3e}, "
                f"rhs_inf={consolidation_solver.last_rhs_inf:.3e}, "
                f"rhs_target={consolidation_solver.last_rhs_target:.3e}"
            )
        consolidation_solver.record()
        time_value += dt_value

        state = consolidation_solver.get_particle_state()
        top_settlement = mean_top_settlement(state, initial_position, domain_height, dx)
        pmax = max_abs_pressure(state)
        history.append(
            [
                step,
                time_value,
                dt_value,
                pmax,
                float(np.mean(state["pressure"])),
                top_settlement,
                float(np.min(state["volume_ratio"])),
                consolidation_solver.last_iterations,
                consolidation_solver.last_delta_inf,
                consolidation_solver.last_rhs_inf,
                consolidation_solver.last_rhs_target,
            ]
        )
        print(
            f"step={step:03d}, time={time_value:12.6e}, dt={dt_value:12.6e}, "
            f"max|p|={pmax:12.6e}, settlement={top_settlement:12.6e}"
        )
        if step >= 5 and pmax < pressure_tol:
            break
        dt_value *= growth

    history = np.asarray(history, dtype=np.float64)
    np.savetxt(
        history_path,
        history,
        delimiter=",",
        header=(
            "step,time_s,dt_s,max_abs_pressure,mean_pressure,mean_top_settlement,"
            "min_J,newton_iterations,delta_inf,rhs_inf,rhs_target"
        ),
        comments="",
    )

    print(f"saved consolidation vtu directory: {consolidation_dir}")
    print(f"saved history: {history_path}")
    print(
        "assumptions: mobility is kept constant at k0 / mu_f because the current solver "
        "does not update permeability with porosity as in Eq. (14), and the run starts "
        "directly from the drained self-weight consolidation stage."
    )
