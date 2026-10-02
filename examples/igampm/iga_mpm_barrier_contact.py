"""I04: implicit soft MPM block sliding on a deformable NURBS ramp.

A Direct ULMPM block arrives obliquely on the upper face of a curved,
end-clamped IGA beam.  Both subsystems advance in one monolithic IPC
Newton solve with contact-aware CCD/line search, so the synchronized output
shows compression, surface deflection, and sliding without penetration.

``--material`` selects ``DruckerPrager`` (default), ``NeoHookean``, or
``VonMises`` for the Direct ULMPM block. The IGA ramp remains elastic.

The block dimensions and discretization are independently configurable for
resolution studies. Defaults write ``--steps + 1`` aligned frames as
``GraphicIGA*.vtu`` and ``GraphicMPMParticle*.vtu``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geotaichi import IGAMPM, init
from src.iga import Cube, DirichletBoundary, Primitives
from src.igampm.GalleryRecorder import IGAMPMGalleryRecorder


def _ramp_offset(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.16 * (1.0 - x / 2.0) + 0.025 * np.sin(0.5 * np.pi * x)


def configure_iga_ramp(iga, output_path, frame_count, interval, dt, degree=2, refinement=2):
    iga.set_configuration(dimension=3, solver_type="Implicit")

    length, width, thickness = 2.0, 1.0, 0.12
    ramp = Cube()
    ramp.set_parameters(start_point=[0.0, 0.0, 0.0], size=[length, width, thickness])
    ramp.generate_knot_u(degree=degree, num_ctrlpts=8 * refinement + 1)
    ramp.generate_knot_v(degree=degree, num_ctrlpts=4 * refinement + 1)
    ramp.generate_knot_w(degree=degree, num_ctrlpts=refinement + 2)
    ramp.generate_ctrlpts()
    ramp.generate_weights()
    # Translate every through-thickness control point by the same smooth
    # profile.  This preserves thickness while making the contact face visibly
    # curved and downhill in +x.
    ramp.control_points[:, 2] += _ramp_offset(ramp.control_points[:, 0])

    primitives = Primitives()
    primitives.append(ramp, "ramp", init_v=[0.0, 0.0, 0.0])
    primitives.finialize()

    end_control_points = np.flatnonzero(
        np.isclose(ramp.control_points[:, 0], 0.0) | np.isclose(ramp.control_points[:, 0], length)
    )
    fixed_dofs = [
        list(3 * end_control_points),
        list(3 * end_control_points + 1),
        list(3 * end_control_points + 2),
    ]
    dirichlet = DirichletBoundary()
    dirichlet.append(fixed_dofs, [0.0] * (3 * len(end_control_points)))

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=[degree, degree, degree])
    iga.add_material(
        young_modulus=6.0e5,
        poisson_ratio=0.32,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
    )
    iga.set_solver(
        dt=dt,
        newmark=[1.0, 0.5, 1.0],
        residual=5.0e-4,
        max_iters=30,
        interval=interval,
        step=frame_count * interval,
        path=str(output_path),
    )


def _soft_block_points(refinement=1, bottom=0.32, size=(0.28, 0.24, 0.14)):
    refinement = int(refinement)
    if refinement <= 0:
        raise ValueError("block refinement must be positive")
    size = np.asarray(size, dtype=np.float64)
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("block size must contain three finite positive lengths")
    grid_size = 0.08 / refinement

    def sample_axis(start, length):
        return np.linspace(start, start + length, int(np.ceil(length / grid_size)) + 1)

    x = sample_axis(0.32, size[0])
    y = sample_axis(0.38, size[1])
    bottom = float(bottom)
    z = sample_axis(bottom, size[2])
    points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    indices = np.arange(points.shape[0]).reshape(len(x), len(y), len(z))
    boundary_mask = np.zeros(indices.shape, dtype=bool)
    boundary_mask[[0, -1], :, :] = True
    boundary_mask[:, [0, -1], :] = True
    boundary_mask[:, :, [0, -1]] = True
    boundary = indices[boundary_mask].astype(np.int32)

    # Tensor-product trapezoidal weights retain points on the true contact
    # boundary while making mass exactly independent of refinement.
    wx = np.full(len(x), (x[-1] - x[0]) / (len(x) - 1))
    wy = np.full(len(y), (y[-1] - y[0]) / (len(y) - 1))
    wz = np.full(len(z), (z[-1] - z[0]) / (len(z) - 1))
    wx[[0, -1]] *= 0.5
    wy[[0, -1]] *= 0.5
    wz[[0, -1]] *= 0.5
    volume = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).reshape(-1)

    surface_measure = np.zeros(boundary_mask.shape, dtype=np.float64)
    surface_measure[[0, -1], :, :] += wy[None, :, None] * wz[None, None, :]
    surface_measure[:, [0, -1], :] += wx[:, None, None] * wz[None, None, :]
    surface_measure[:, :, [0, -1]] += wx[:, None, None] * wy[None, :, None]
    return points, boundary, volume, surface_measure[boundary_mask]


def configure_mpm_block(
    mpm,
    output_path,
    frame_count,
    interval,
    dt,
    material_name="NeoHookean",
    block_refinement=1,
    cohesion=1500.0,
    gravity=4.0,
    initial_velocity=(1.4, 0.0, -0.35),
    block_bottom=0.32,
    block_size=(0.28, 0.24, 0.14),
):
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[2.3, 1.1, 0.8],
        gravity=[0.0, 0.0, -float(gravity)],
        background_damping=0.015,
        visualize=False,
    )

    points, boundary, volume, surface_measure = _soft_block_points(
        block_refinement, bottom=block_bottom, size=block_size
    )
    grid_size = 0.08 / int(block_refinement)
    body = mpm.create_body()
    contact_measure = np.power(volume, 2.0 / 3.0)
    contact_measure[boundary] = surface_measure
    body.add_particles(
        points,
        volume=volume,
        init_v=list(initial_velocity),
        name="soft_block",
        grid_size=grid_size,
        xmin=[0.0, 0.15, 0.0],
        xmax=[2.2, 0.85, 0.75],
        boundary_ids=boundary,
        surface_measure=contact_measure,
    )
    mpm.add_body(body)
    material_key = str(material_name).strip().replace("-", "").replace("_", "").lower()
    material_parameters = dict(
        young_modulus=1.5e5,
        poisson_ratio=0.32,
        density=900.0,
    )
    if material_key in ("neohookean", "neo"):
        material_parameters["model"] = "NeoHookean"
    elif material_key in ("druckerprager", "dp"):
        material_parameters.update(
            model="DruckerPrager",
            Cohesion=float(cohesion),
            FrictionAngle=30.0,
            DilationAngle=30.0,
            dpType="Circumscribed",
        )
    elif material_key in ("vonmises", "j2"):
        material_parameters.update(
            model="VonMises",
            YieldStress=5000.0,
            HardeningModulus=2.0e4,
        )
    else:
        raise ValueError("IGAMPM example MATERIAL must be NeoHookean, DruckerPrager, " "or VonMises")
    mpm.add_material(**material_parameters)
    mpm.add_element({"ElementSize": grid_size, "ShapeFunction": "linear"})
    mpm.set_solver(
        {
            "dt": dt,
            "newmark": [1.0, 0.5, 1.0],
            "residual": 5.0e-4,
            "max_iters": 30,
            "interval": interval,
            "step": frame_count * interval,
            "scale": 0.55,
            "ccd": True,
            "line_search": True,
            "visualize": False,
            "path": str(output_path),
        },
        log=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output-interval", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--dt", type=float, default=1.25e-3)
    parser.add_argument("--block-refinement", type=int, default=3)
    parser.add_argument("--iga-refinement", type=int, default=1)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--cohesion", type=float, default=250.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--initial-vx", type=float, default=0.0)
    parser.add_argument("--initial-vz", type=float, default=0.0)
    parser.add_argument("--block-bottom", type=float, default=0.275)
    parser.add_argument("--block-length", type=float, default=0.24)
    parser.add_argument("--block-width", type=float, default=0.24)
    parser.add_argument("--block-height", type=float, default=0.36)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--newton-max-iterations", type=int, default=50)
    parser.add_argument("--step-retry-max-retries", type=int, default=5)
    parser.add_argument("--contact-model", choices=("BarrierIPC", "SemiIPC"), default="BarrierIPC")
    parser.add_argument("--assemble-type", choices=("COO", "HashTriplet"), default="HashTriplet")
    parser.add_argument("--minimum-timestep-ratio", type=float, default=32.0)
    parser.add_argument("--linear-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--linear-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--newton-tolerance",
        type=float,
        default=5.0e-3,
        help="maximum coupled correction velocity in m/s",
    )
    parser.add_argument(
        "--material",
        choices=("NeoHookean", "DruckerPrager", "VonMises"),
        default="DruckerPrager",
    )
    parser.add_argument(
        "--friction-mode",
        choices=("lagged", "fully_implicit"),
        default="lagged",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData" / "implicit_dp_barrier_contact"),
    )
    arguments = parser.parse_args()
    if (
        arguments.steps <= 0
        or arguments.output_interval <= 0
        or arguments.resolution <= 0
        or arguments.block_refinement <= 0
        or arguments.iga_refinement <= 0
        or arguments.newton_max_iterations <= 0
        or arguments.step_retry_max_retries < 0
        or not np.isfinite(arguments.minimum_timestep_ratio)
        or arguments.minimum_timestep_ratio < 1.0
        or not np.isfinite(arguments.linear_tolerance)
        or arguments.linear_tolerance < 0.0
        or not np.isfinite(arguments.linear_relative_tolerance)
        or arguments.linear_relative_tolerance < 0.0
        or arguments.linear_tolerance + arguments.linear_relative_tolerance <= 0.0
        or not np.isfinite(arguments.dt)
        or arguments.dt <= 0.0
        or not np.isfinite(arguments.cohesion)
        or arguments.cohesion < 0.0
        or not np.isfinite(arguments.gravity)
        or arguments.gravity < 0.0
        or not np.isfinite(arguments.initial_vx)
        or not np.isfinite(arguments.initial_vz)
        or not np.isfinite(arguments.block_bottom)
        or arguments.block_bottom <= 0.0
        or not np.isfinite(arguments.newton_tolerance)
        or arguments.newton_tolerance <= 0.0
        or any(
            not np.isfinite(length) or length <= 0.0
            for length in (arguments.block_length, arguments.block_width, arguments.block_height)
        )
        or 0.32 + arguments.block_length >= 2.2
        or 0.38 + arguments.block_width >= 0.85
        or arguments.block_bottom + arguments.block_height >= 0.75
    ):
        raise ValueError("invalid positive discretization/time value or finite physical parameter")
    output_path = Path(arguments.output_dir).expanduser().resolve()
    frame_count = arguments.steps
    interval = arguments.output_interval
    resolution = arguments.resolution
    dt = arguments.dt
    block_grid_size = 0.08 / arguments.block_refinement

    init(
        dim=3,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        debug=False,
        log=True,
        random_seed=23,
    )

    coupling = IGAMPM(
        log=True,
        contact_model=arguments.contact_model,
        assemble_type=arguments.assemble_type,
        kappa=4.0e5,
        dhat=0.5 * block_grid_size,
        dmin=0.1 * block_grid_size,
        contact_ccd_safety=0.88,
        friction_mode=arguments.friction_mode,
        monolithic_max_iterations=arguments.newton_max_iterations,
        monolithic_tolerance=arguments.newton_tolerance,
        monolithic_linear_solver_tolerance=arguments.linear_tolerance,
        monolithic_linear_solver_relative_tolerance=arguments.linear_relative_tolerance,
        monolithic_linear_solver_max_iters=5000,
        enable_step_retry=arguments.contact_model == "BarrierIPC",
        step_retry_max_retries=arguments.step_retry_max_retries,
        step_retry_reduction=0.5,
        step_retry_minimum_timestep=dt / arguments.minimum_timestep_ratio,
        contact_all_mpm_particles=True,
        contact_surface_include=[(0, 1)],
    )
    coupling.set_configuration(
        dimension=3,
        coupling_scheme="IGAMPM",
        contact_model=arguments.contact_model,
        activate_friction=False,
    )
    configure_iga_ramp(
        coupling.iga,
        output_path,
        frame_count,
        interval,
        dt,
        degree=arguments.degree,
        refinement=arguments.iga_refinement,
    )
    configure_mpm_block(
        coupling.mpm,
        output_path,
        frame_count,
        interval,
        dt,
        material_name=arguments.material,
        block_refinement=arguments.block_refinement,
        cohesion=arguments.cohesion,
        gravity=arguments.gravity,
        initial_velocity=(arguments.initial_vx, 0.0, arguments.initial_vz),
        block_bottom=arguments.block_bottom,
        block_size=(arguments.block_length, arguments.block_width, arguments.block_height),
    )

    engine = coupling.build()
    recorder = IGAMPMGalleryRecorder(
        engine,
        output_path,
        iga_resolution=(resolution, resolution // 2 + 1, 2 * arguments.iga_refinement + 1),
    )
    recorder.record(0, update_iga_stress=False)
    minimum_distance = np.inf
    maximum_block_compression = 0.0
    maximum_active_contacts = 0
    initial_positions = engine.mpm.particle.x.to_numpy().copy()
    preferred_timestep = dt
    total_retries = 0
    for frame in range(1, frame_count + 1):
        target_time = frame * interval * dt
        accepted_substeps = 0
        frame_retries = 0
        while engine.time < target_time - 1.0e-12 * max(1.0, target_time):
            requested_timestep = min(preferred_timestep, target_time - engine.time)
            engine._set_implicit_timestep(requested_timestep)
            result = engine.implicit_ipc_substep(
                include_friction=False,
                newton_max_iterations=arguments.newton_max_iterations,
                newton_tolerance=arguments.newton_tolerance,
                verbose=arguments.verbose,
            )
            accepted_substeps += 1
            frame_retries += int(result["step_retry"]["retry_count"])
            accepted_timestep = float(result["step_retry"]["accepted_timestep"])
            if accepted_timestep < requested_timestep * (1.0 - 1.0e-12):
                preferred_timestep = accepted_timestep
        engine._set_implicit_timestep(preferred_timestep)
        total_retries += frame_retries
        recorder.record(frame)
        minimum_distance = min(minimum_distance, float(result["minimum_distance"]))
        maximum_active_contacts = max(maximum_active_contacts, int(engine.curr_barrier_contact_num))
        current_positions = engine.mpm.particle.x.to_numpy()
        initial_height = np.ptp(initial_positions[:, 2])
        current_height = np.ptp(current_positions[:, 2])
        maximum_block_compression = max(
            maximum_block_compression,
            1.0 - current_height / initial_height,
        )
        print(
            f"frame={frame:03d}/{frame_count:03d} "
            f"contacts={engine.curr_barrier_contact_num:03d} "
            f"minimum_distance={float(result['minimum_distance']):.6e} "
            f"newton_iterations={int(result['iterations']):02d} "
            f"time={engine.time:.6e} "
            f"substeps={accepted_substeps:03d} "
            f"dt={float(result['step_retry']['accepted_timestep']):.6e} "
            f"retries={frame_retries}",
            flush=True,
        )

    print(f"minimum_distance={minimum_distance:.6e}")
    print(f"maximum_block_compression={maximum_block_compression:.6e}")
    print(f"IGA_VTU={output_path / 'vtks' / 'GraphicIGA*.vtu'}")
    print("MPM_VTU=" f"{output_path / 'vtks' / 'GraphicMPMParticle*.vtu'}")
    summary = {
        "case": "implicit_iga_mpm_barrier_contact",
        "material": arguments.material,
        "contact_model": arguments.contact_model,
        "assemble_type": arguments.assemble_type,
        "frames": frame_count,
        "completed_time": float(engine.time),
        "iga_control_points": int(engine.iga.patch.control_points.shape[0]),
        "mpm_particles": int(engine.mpm.particleNum[0]),
        "minimum_contact_distance": float(minimum_distance),
        "maximum_active_contacts": int(maximum_active_contacts),
        "maximum_block_compression": float(maximum_block_compression),
        "accepted_timestep": float(preferred_timestep),
        "total_retries": int(total_retries),
        "finite": bool(np.isfinite(engine.mpm.particle.x.to_numpy()).all()),
    }
    (output_path / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        not summary["finite"]
        or not summary["maximum_active_contacts"]
        or not np.isclose(summary["completed_time"], frame_count * interval * dt)
    ):
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()
