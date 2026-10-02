"""IGA--MPM CPT with explicit DEM contact or monolithic IPC.

The IPC route uses the physical axisymmetric meridian, a refined NURBS pile,
and DP soil.  The legacy explicit route remains a thin 3-D capability check.
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
    return arguments


def main():
    arguments = parse_arguments()
    os.environ["GEOTAICHI_REAL_DTYPE"] = arguments.default_fp

    import geotaichi as gt

    from examples.mpm.Contact.CPT2D.coupled_cpt import (
        axisymmetric_initial_deformation_gradient,
        configure_direct_axisymmetric_mpm,
        configure_iga_axisymmetric_penetrator,
        configure_iga_penetrator,
        configure_native_explicit_mpm,
        explicit_contact_parameters,
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
    step_count = int(math.ceil(arguments.time / arguments.dt))
    output_interval = max(1, int(round(arguments.save_interval / arguments.dt)))
    gt.init(
        dim=2 if implicit else 3,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
    )

    iga = gt.IGA(log=True)
    mpm = gt.MPM(log=True)
    if implicit:
        grid_size, particle_count = configure_direct_axisymmetric_mpm(
            mpm,
            output_path,
            arguments.dt,
            arguments.time,
            arguments.save_interval,
            arguments.resolution_scale,
        )
        contact_parameters = ipc_contact_parameters(grid_size)
        coupling = gt.IGAMPM(
            iga=iga,
            mpm=mpm,
            log=True,
            contact_model="IPC",
            activate_friction=False,
            contact_all_mpm_particles=True,
            contact_surface_include=[(0, 0), (0, 1)],
            monolithic_max_iterations=35,
            monolithic_tolerance=5.0e-4,
            monolithic_linear_solver_tolerance=1.0e-8,
            monolithic_linear_solver_max_iters=30_000,
            project_pd=True,
            **contact_parameters,
        )
    else:
        coupling = gt.IGAMPM(
            iga=iga,
            mpm=mpm,
            log=True,
            contact_model="Linear",
            contact_surface_include=[(0, 0), (0, 4), (0, 5)],
        )
        grid_size = configure_native_explicit_mpm(
            mpm,
            output_path,
            arguments.dt,
            arguments.time,
            arguments.save_interval,
            arguments.resolution_scale,
        )
    if implicit:
        top_control_points = configure_iga_axisymmetric_penetrator(
            iga, arguments.dt, step_count, output_interval, output_path
        )
    else:
        configure_iga_penetrator(
            iga,
            implicit,
            arguments.dt,
            step_count,
            output_interval,
            output_path,
        )
    coupling.set_configuration(
        dimension=2 if implicit else 3,
        coupling_scheme="IGAMPM",
        contact_model="IPC" if implicit else "Linear",
        activate_friction=False,
        axisymmetric=implicit,
        axis_offset=0.0,
    )
    if not implicit:
        coupling.add_property(
            MPMmaterial=1,
            IGAbody=0,
            property=explicit_contact_parameters("igampm"),
        )
    if not implicit:
        coupling.run(steps=step_count, verbose=True, record=True)
        return

    engine = coupling.build()
    engine.mpm.F0.from_numpy(
        np.repeat(
            axisymmetric_initial_deformation_gradient()[None, :, :],
            particle_count,
            axis=0,
        )
    )
    history = []

    def sample_cpt(coupled_engine):
        control_points = coupled_engine.iga.patch.control_points.to_numpy()
        history.append(
            (
                float(coupled_engine.time),
                float(2.5 - np.max(control_points[top_control_points, 1])),
                float(np.min(control_points[:, 1])),
                float(np.max(control_points[:, 1])),
                int(coupled_engine.curr_barrier_contact_num),
            )
        )

    result = coupling.run(
        steps=step_count,
        verbose=True,
        record=True,
        postprocessing=(sample_cpt,),
    )
    final_control_points = engine.iga.patch.control_points.to_numpy()
    final_top = float(np.max(final_control_points[top_control_points, 1]))
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "cpt_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "penetration", "pile_tip", "pile_top", "active_contacts"))
        writer.writerows(history)
    summary = {
        "axisymmetric": True,
        "grid_size": grid_size,
        "particle_count": particle_count,
        "penetration_time": arguments.time,
        "target_penetration": 0.1 * arguments.time,
        "completed_time": float(engine.time),
        "final_pile_tip": float(np.min(final_control_points[:, 1])),
        "final_pile_top": final_top,
        "fully_inserted": final_top <= 1.5 + 1.0e-6,
        "maximum_active_contacts": max((row[4] for row in history), default=0),
        "converged": bool(result["converged"]),
    }
    (output_path / "cpt_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["fully_inserted"]:
        raise RuntimeError(f"IGA pile stopped above the soil surface: top={final_top:.6e} m")


if __name__ == "__main__":
    main()
