"""FEM--MPM CPT with explicit DEM contact or monolithic IPC.

The IPC route is a two-dimensional axisymmetric meridian calculation with
revolved material/contact measures. Drucker--Prager substitutes for the
standard SDMC soil because it is the available implicit coupling material.
The legacy explicit route remains a thin three-dimensional capability check.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact", choices=("explicit", "ipc"), default="explicit")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--time", type=float)
    parser.add_argument("--save-interval", type=float)
    parser.add_argument("--resolution-scale", type=float)
    parser.add_argument("--settling-time", type=float, default=0.0)
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData" / "cpt_dp"))
    arguments = parser.parse_args()
    implicit = arguments.contact == "ipc"
    if arguments.dt is None:
        arguments.dt = 5.0e-4 if implicit else 1.0e-5
    if arguments.time is None:
        arguments.time = 10.0 if implicit else 0.12
    if arguments.save_interval is None:
        arguments.save_interval = 0.2 if implicit else 0.02
    if arguments.resolution_scale is None:
        arguments.resolution_scale = 1.0
    if not math.isfinite(arguments.settling_time) or arguments.settling_time < 0.0:
        parser.error("--settling-time must be finite and non-negative")
    return arguments


def main():
    arguments = parse_arguments()
    # Shared FEM/MPM/IPC scalar fields are declared while implementation
    # modules import, so select their dtype before importing geotaichi.
    os.environ["GEOTAICHI_REAL_DTYPE"] = arguments.default_fp

    import geotaichi as gt

    from examples.mpm.Contact.CPT2D.coupled_cpt import (
        configure_direct_axisymmetric_mpm,
        configure_fem_axisymmetric_penetrator,
        configure_fem_penetrator,
        configure_native_explicit_mpm,
        explicit_contact_parameters,
        axisymmetric_initial_deformation_gradient,
        ipc_contact_parameters,
        validate_run_parameters,
    )

    validate_run_parameters(
        arguments.dt,
        arguments.time,
        arguments.save_interval,
        arguments.resolution_scale,
    )
    implicit = arguments.contact == "ipc"
    output_path = Path(arguments.output_dir)
    total_time = arguments.time + (arguments.settling_time if implicit else 0.0)
    gt.init(
        dim=2 if implicit else 3,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
    )

    fem = gt.FEM(log=True)
    mpm = gt.MPM(log=True)
    if implicit:
        top_nodes = configure_fem_axisymmetric_penetrator(fem, arguments.settling_time)
        grid_size, particle_count = configure_direct_axisymmetric_mpm(
            mpm,
            output_path,
            arguments.dt,
            total_time,
            arguments.save_interval,
            arguments.resolution_scale,
        )
        model = gt.FEMPM(fem=fem, mpm=mpm, log=True)
    else:
        configure_fem_penetrator(fem, implicit)
        # Explicit FEM--MPM selects native Lagrangian coupling before the MPM
        # particle fields are allocated.
        model = gt.FEMPM(fem=fem, mpm=mpm, log=True)
        grid_size = configure_native_explicit_mpm(
            mpm,
            output_path,
            arguments.dt,
            arguments.time,
            arguments.save_interval,
            arguments.resolution_scale,
        )
        particle_count = int(mpm.sims.max_particle_num)

    model.set_configuration(
        domain=list(mpm.sims.domain),
        gravity=list(mpm.sims.gravity),
        search="BVH",
        axisymmetric=implicit,
        axis_offset=0.0,
        log=True,
    )
    solver = {
        "Timestep": arguments.dt,
        "SimulationTime": total_time,
        "SaveInterval": arguments.save_interval,
        "SavePath": str(output_path),
    }
    if implicit:
        solver.update(
            assemble_type="HashTriplet",
            linear_solver="PCG",
            project_pd=True,
            max_iterations=64,
            residual_tolerance=1.0e-3,
            linear_solver_tolerance=1.0e-6,
            linear_solver_relative_tolerance=1.0e-7,
            linear_solver_max_iters=5_000,
            enable_step_retry=True,
            step_retry_max_retries=4,
            step_retry_reduction=0.5,
            step_retry_minimum_timestep=arguments.dt / 16.0,
            quasi_static=True,
            contact_all_mpm_particles=True,
            scale=1.0,
        )
    model.set_solver(solver, log=True)
    model.add_surface(body_ids=[0])
    model.memory_allocate(
        {
            "max_particle_number": particle_count,
            "contact_coordination_number": 4 if implicit else 8,
            "max_contact_pairs": max(32_768, 4 * particle_count),
            "max_point_triangle_pairs": 1 if implicit else max(32_768, 4 * particle_count),
            "max_point_edge_pairs": max(32_768, 4 * particle_count) if implicit else 1,
            "max_facet_cell_pairs": 1 if implicit else 8_192,
        }
    )
    if implicit:
        model.choose_contact_model(
            "IPC",
            **ipc_contact_parameters(grid_size),
            friction_mode="lagged",
            friction_iterations=1,
            project_pd=True,
        )
    else:
        model.choose_contact_model("Linear")
        model.add_property(
            MPMmaterial=1,
            FEMbody=0,
            property=explicit_contact_parameters("fempm"),
        )
    if implicit:
        history = []
        model.add_essentials()
        initial_deformation = np.repeat(
            axisymmetric_initial_deformation_gradient()[None, :, :],
            particle_count,
            axis=0,
        )
        model.mpm.enginer.F0.from_numpy(initial_deformation)

        def sample_cpt(engine):
            reaction = engine.fem.state.reaction.to_numpy()
            positions = engine.fem.state.position.to_numpy()
            history.append(
                (
                    float(engine.time),
                    2.5 - float(np.max(positions[top_nodes, 1])),
                    float(np.min(positions[:, 1])),
                    float(np.max(positions[:, 1])),
                    float(np.sum(reaction[top_nodes, 1])),
                    int(engine.contact.diagnostics()["active_contacts"]),
                )
            )

        result = model.run(
            verbose=True,
            postprocessing=(sample_cpt,),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "cpt_history.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time", "penetration", "pile_tip", "pile_top", "axial_reaction", "active_contacts"))
            writer.writerows(history)
        final_positions = model.enginer.fem.state.position.to_numpy()
        final_top = float(np.max(final_positions[top_nodes, 1]))
        summary = {
            "axisymmetric": True,
            "grid_size": grid_size,
            "particle_count": particle_count,
            "settling_time": arguments.settling_time,
            "penetration_time": arguments.time,
            "target_penetration": 0.1 * arguments.time,
            "completed_time": float(model.enginer.time),
            "final_pile_tip": float(np.min(final_positions[:, 1])),
            "final_pile_top": final_top,
            "fully_inserted": final_top <= 1.5 + 1.0e-6,
            "maximum_axial_reaction": max((abs(row[4]) for row in history), default=0.0),
            "maximum_active_contacts": max((row[5] for row in history), default=0),
            "converged": bool(result["converged"]),
        }
        (output_path / "cpt_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if not summary["fully_inserted"]:
            raise RuntimeError(f"FEM pile stopped above the soil surface: top={final_top:.6e} m")
    else:
        model.run(verbose=True, steps=int(math.ceil(arguments.time / arguments.dt)))


if __name__ == "__main__":
    main()
