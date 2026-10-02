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

config.set_dimension(3)

from src.mpm.generator.Body import Body
from src.mpm.boundaries.BoundaryCondition import DirichletBoundary, NeumannBoundary


def top_patch_nodes(coords, z_top, x_min, x_max, y_min, y_max):
    return np.where(
        np.isclose(coords[:, 2], z_top)
        & (coords[:, 0] >= x_min - 1.0e-12)
        & (coords[:, 0] <= x_max + 1.0e-12)
        & (coords[:, 1] >= y_min - 1.0e-12)
        & (coords[:, 1] <= y_max + 1.0e-12)
    )[0]


def top_patch_weights(sample_xy, x_min, x_max, y_min, y_max):
    weights = np.ones(sample_xy.shape[0], dtype=np.float64)
    on_x_edge = np.isclose(sample_xy[:, 0], x_min) | np.isclose(sample_xy[:, 0], x_max)
    on_y_edge = np.isclose(sample_xy[:, 1], y_min) | np.isclose(sample_xy[:, 1], y_max)
    weights[on_x_edge] *= 0.5
    weights[on_y_edge] *= 0.5
    return weights


def build_surface_columns(initial_position, sample_xy, soil_top, dx):
    columns = []
    for x, y in sample_xy:
        mask = (
            (np.abs(initial_position[:, 0] - x) <= 0.55 * dx)
            & (np.abs(initial_position[:, 1] - y) <= 0.55 * dx)
            & (initial_position[:, 2] >= soil_top - 1.5 * dx)
        )
        ids = np.where(mask)[0]
        if ids.size == 0:
            dist2 = (initial_position[:, 0] - x) ** 2 + (initial_position[:, 1] - y) ** 2
            ids = np.array([int(np.argmin(dist2))], dtype=np.int32)
        columns.append(ids)
    return columns


def compute_penalty_contact(columns, state, footing_z, node_area_weights, penalty_stiffness, contact_offsets):
    surface_z = np.empty(len(columns), dtype=np.float64)
    nodal_force = np.zeros(len(columns), dtype=np.float64)
    for i, ids in enumerate(columns):
        z_now = float(np.max(state["position"][ids, 2]))
        surface_z[i] = z_now
        penetration = max(z_now + contact_offsets[i] - footing_z, 0.0)
        nodal_force[i] = -penalty_stiffness * penetration * node_area_weights[i]
    return surface_z, nodal_force


if __name__ == "__main__":
    init(dim=3, arch="cpu", default_fp="float64", debug=False, log=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "FootingNeoHookeanPenalty3D")
    os.makedirs(out_dir, exist_ok=True)
    for old_file in glob(os.path.join(out_dir, "vtks", "particles*.vtu")):
        os.remove(old_file)
    history_path = os.path.join(out_dir, "load_settlement_history.csv")
    if os.path.exists(history_path):
        os.remove(history_path)

    domain = [5.0, 5.0, 5.0]
    soil_start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    soil_end = np.array(domain, dtype=np.float64)
    footing_size = 1.0
    footing_x_min = 0.5 * (domain[0] - footing_size)
    footing_x_max = footing_x_min + footing_size
    footing_y_min = 0.5 * (domain[1] - footing_size)
    footing_y_max = footing_y_min + footing_size

    dx = 0.25
    ppc = 4
    dt = 0.1
    total_increments = 25
    target_settlement = 0.5
    disp_increment = target_settlement / total_increments
    min_disp_increment = disp_increment / 8.0
    max_accepted_steps = total_increments

    young_modulus = 1.0e7
    poisson_ratio = 0.3
    density = 2000.0
    fluid_density = 1000.0
    porosity = 0.4
    # Assumption: solver mobility corresponds to k / mu_f in SI units.
    mobility = 1.0e-11
    penalty_factor = 1000.0
    penalty_stiffness = penalty_factor * young_modulus / dx

    body = Body()
    body.add_cube(
        soil_start.tolist(),
        soil_end.tolist(),
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
    bottom = np.where(np.isclose(coords[:, 2], soil_start[2]))[0]
    x_sides = np.where(np.isclose(coords[:, 0], soil_start[0]) | np.isclose(coords[:, 0], soil_end[0]))[0]
    y_sides = np.where(np.isclose(coords[:, 1], soil_start[1]) | np.isclose(coords[:, 1], soil_end[1]))[0]
    top_all = np.where(np.isclose(coords[:, 2], soil_end[2]))[0]
    top_patch = top_patch_nodes(coords, soil_end[2], footing_x_min, footing_x_max, footing_y_min, footing_y_max)

    dirichlet = DirichletBoundary()
    dbc_ids = []
    dbc_vals = []
    dbc_ids.append(list(component * bottom + 2))
    dbc_vals.extend([0.0] * len(bottom))
    dbc_ids.append(list(component * np.unique(np.concatenate([x_sides, bottom])) + 0))
    dbc_vals.extend([0.0] * len(np.unique(np.concatenate([x_sides, bottom]))))
    dbc_ids.append(list(component * np.unique(np.concatenate([y_sides, bottom])) + 1))
    dbc_vals.extend([0.0] * len(np.unique(np.concatenate([y_sides, bottom]))))
    dbc_ids.append(list(component * top_all + 3))
    dbc_vals.extend([0.0] * len(top_all))
    dirichlet.append(dbc_ids, dbc_vals)

    patch_uz_dofs = list(component * top_patch + 2)
    neumann = NeumannBoundary()
    neumann.append([patch_uz_dofs], [0.0] * len(patch_uz_dofs))

    case_dir = os.path.join(out_dir, "case_history")
    os.makedirs(case_dir, exist_ok=True)

    nx = int((soil_end[0] - soil_start[0]) / dx)
    ny = int((soil_end[1] - soil_start[1]) / dx)
    nz = int((soil_end[2] - soil_start[2]) / dx)
    particles_per_cell = ppc**3
    n_particles = nx * ny * nz * particles_per_cell
    max_support = 27
    estimated_triplets = n_particles * (max_support**2) * (component**2)
    if estimated_triplets > 2.0e8:
        raise RuntimeError(
            "This 3D footing case is too large for the current CoordinateSparseMatrix assembly. "
            f"Estimated stiffness triplets={estimated_triplets:.3e}. "
            "This is a data-structure limit of the current solver, not a contact convergence result."
        )

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
        model="neoHookean",
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        fluid_density=fluid_density,
        porosity=porosity,
        mobility=mobility,
    )
    mpm.add_element({"ElementSize": dx, "ShapeFunction": "gimp"})
    mpm.set_solver(
        {
            "dt": dt,
            "step": max_accepted_steps,
            "interval": 1,
            "residual": 1.0e-4,
            "rhs_tolerance_abs": 1.0e-2,
            "rhs_tolerance_rel": 1.0e-2,
            "max_iters": 30,
            "line_search": True,
            "line_search_max_backtrack": 10,
            "require_both_convergence_checks": False,
            "linear_solver": "direct",
            "direct_regularization": 1.0e-6,
            "ppd": ppc,
            "visualize": True,
            "path": case_dir,
        }
    )
    mpm.add_engine()
    solver = mpm.enginer
    solver.initial_simulation()
    initial_state = solver.get_particle_state()
    initial_position = initial_state["position"].copy()
    sample_xy = coords[top_patch, :2]
    node_area_weights = (
        top_patch_weights(sample_xy, footing_x_min, footing_x_max, footing_y_min, footing_y_max) * dx * dx
    )
    surface_columns = build_surface_columns(initial_position, sample_xy, soil_end[2], dx)
    contact_offsets = np.array(
        [soil_end[2] - float(np.max(initial_position[ids, 2])) for ids in surface_columns],
        dtype=np.float64,
    )
    neumann_values = solver.neumann.value.to_numpy()

    current_settlement = 0.0
    accepted_steps = 0
    attempted_steps = 0
    history = []

    while current_settlement < target_settlement - 1.0e-12 and accepted_steps < max_accepted_steps:
        attempted_steps += 1
        step_state = solver.get_solver_state()
        state_before = solver.get_particle_state()
        target_increment = min(abs(disp_increment), target_settlement - current_settlement)
        footing_z = soil_end[2] - (current_settlement + target_increment)

        surface_z, contact_force = compute_penalty_contact(
            surface_columns,
            state_before,
            footing_z,
            node_area_weights,
            penalty_stiffness,
            contact_offsets,
        )
        neumann_values[:] = contact_force
        solver.neumann.value.from_numpy(neumann_values)

        solver.refresh_active_dofs()
        solver.reset_step_solution()
        solver.assemble_pressure_projection()
        solver.apply_pressure_projection_to_solution()
        solver.solve_current_step(verbose=True)

        accepted = solver.last_converged and solver.last_delta_inf < 1.0e2
        if not accepted:
            solver.set_solver_state(step_state)
            neumann_values[:] = 0.0
            solver.neumann.value.from_numpy(neumann_values)
            if target_increment <= min_disp_increment + 1.0e-12:
                print(
                    f"stop: rejected target_increment={target_increment:.6e}, "
                    f"delta_inf={solver.last_delta_inf:.6e}, rhs_inf={solver.last_rhs_inf:.6e}"
                )
                break
            disp_increment *= 0.5
            print(
                f"reject: target_increment={target_increment:.6e}, "
                f"delta_inf={solver.last_delta_inf:.6e}, rhs_inf={solver.last_rhs_inf:.6e}, "
                f"halve increment to {disp_increment:.6e}"
            )
            continue

        solver.commit_step()
        solver.record()
        current_settlement += target_increment
        accepted_steps += 1

        max_penetration = float(np.max(np.maximum(surface_z + contact_offsets - footing_z, 0.0)))
        applied_load = float(-np.sum(contact_force))
        history.append(
            [
                accepted_steps,
                attempted_steps,
                current_settlement,
                target_increment,
                applied_load,
                max_penetration,
                solver.last_iterations,
                solver.last_delta_inf,
                solver.last_rhs_inf,
                solver.last_rhs_target,
            ]
        )
        print(
            f"accept: step={accepted_steps:02d}, settlement={current_settlement:.6e}, "
            f"inc={target_increment:.6e}, load={applied_load:.6e}, max_pen={max_penetration:.6e}, "
            f"iters={solver.last_iterations}, rhs_inf={solver.last_rhs_inf:.6e}, "
            f"rhs_target={solver.last_rhs_target:.6e}"
        )

    history = np.array(history, dtype=np.float64)
    np.savetxt(
        history_path,
        history,
        delimiter=",",
        header=(
            "accepted_step,attempted_step,target_settlement,settlement_increment,"
            "compressive_contact_load,max_penetration,newton_iterations,delta_inf,rhs_inf,rhs_target"
        ),
        comments="",
    )
    print(f"saved history: {history_path}")
    print("note: this 3D case omits moving-mesh drainage and true monolithic penalty contact.")
