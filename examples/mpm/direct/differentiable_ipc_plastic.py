"""Large Direct-MPM plastic trajectory adjoint with lagged IPC friction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from geotaichi import MPM, init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--impact-speed", type=float, default=0.05)
    parser.add_argument("--active-grid-scale", type=float, default=0.3)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--material", choices=("DruckerPrager", "VonMises"), default="DruckerPrager")
    parser.add_argument("--cohesion", type=float, default=20.0)
    parser.add_argument("--friction-angle", type=float, default=25.0)
    parser.add_argument("--yield-stress", type=float, default=80.0)
    parser.add_argument("--hardening-modulus", type=float, default=400.0)
    parser.add_argument("--output-dir", default="differentiable_direct_mpm")
    args = parser.parse_args()
    if (
        min(args.nx, args.ny, args.steps) < 1
        or args.max_iters < 1
        or args.dt <= 0.0
        or args.impact_speed < 0.0
        or args.active_grid_scale <= 0.0
        or args.cohesion < 0.0
        or args.friction_angle < 0.0
        or args.yield_stress < 0.0
        or args.hardening_modulus < 0.0
    ):
        parser.error("nx, ny, steps, dt, impact speed, and active-grid scale must be valid")

    init(dim=2, arch=args.arch, default_fp="float64", offline_cache=True, log=False)
    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[1.0, 1.0],
        gravity=[0.0, -9.81],
        ipc=True,
        visualize=False,
        log=False,
    )

    dx = 0.01
    x = 0.18 + (np.arange(args.nx) + 0.5) * (0.64 / args.nx)
    y = 0.03 + (np.arange(args.ny) + 0.5) * (0.32 / args.ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    ix, iy = np.meshgrid(np.arange(args.nx), np.arange(args.ny), indexing="ij")
    boundary = np.flatnonzero((ix == 0) | (ix == args.nx - 1) | (iy == 0) | (iy == args.ny - 1))
    body = mpm.create_body()
    body.add_particles(
        points,
        volume=0.64 * 0.32 / points.shape[0],
        init_v=[0.2, -args.impact_speed],
        boundary_ids=boundary,
        name="plastic_block",
        grid_size=dx,
        xmin=[0.0, 0.0],
        xmax=[1.0, 1.0],
    )
    mpm.add_body(body)
    material = {
        "model": args.material,
        "density": 1500.0,
        "young_modulus": 2.0e5,
        "poisson_ratio": 0.3,
    }
    if args.material == "DruckerPrager":
        material.update(
            Cohesion=args.cohesion,
            FrictionAngle=args.friction_angle,
            DilationAngle=args.friction_angle,
            dpType="Circumscribed",
        )
    else:
        material.update(YieldStress=args.yield_stress, HardeningModulus=args.hardening_modulus)
    mpm.add_material(**material)
    mpm.add_element({"ElementSize": dx, "ShapeFunction": "Linear"})
    ground = mpm.create_ground()
    ground.append([0.0, 0.0], [0.0, 1.0])
    mpm.add_ground(ground)
    mpm.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.steps * args.dt,
            "SaveInterval": args.steps * args.dt,
            "SavePath": args.output_dir,
            "newmark": [0.5, 0.25, 0.5],
            "residual": 1.0e-5,
            "max_iters": args.max_iters,
            "linear_solver_tolerance": 1.0e-9,
            "linear_solver_max_iters": 5000,
            "line_search": True,
            "scale": args.active_grid_scale,
            "enable_step_retry": False,
        }
    )
    mpm.add_contact(
        "BarrierIPC",
        dhat=0.04,
        dmin=1.0e-4,
        kappa=2.0e4,
        mu=0.3,
        epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=1,
        contact_search="LinkedCell",
    )

    trajectory = mpm.differentiable(steps=args.steps)
    simulation = trajectory.simulation
    initial = simulation.mpm.particle.x.to_numpy()[: points.shape[0]].copy()
    started = time.perf_counter()
    for _ in range(args.steps):
        trajectory.step(verbose=False)
    forward_seconds = time.perf_counter() - started
    terminal = simulation.mpm.particle.x.to_numpy()[: points.shape[0]].copy()
    target = initial + np.asarray([0.015, -0.01])
    difference = terminal - target
    started = time.perf_counter()
    gradient = trajectory.backward({"position": difference / points.shape[0]}, verbose=False)
    backward_seconds = time.perf_counter() - started

    material_vjp = np.asarray(gradient["material_parameters"])
    plastic_strain = simulation.mpm.material.equivalent_plastic_strain.to_numpy()[: points.shape[0]]
    plastic_names = (
        ("cohesion", "friction_angle_degrees")
        if args.material == "DruckerPrager"
        else ("yield_stress", "hardening_modulus")
    )
    summary = {
        "case": "differentiable_direct_mpm_plastic_ipc",
        "material": args.material,
        "particles": int(points.shape[0]),
        "steps": args.steps,
        "dt": args.dt,
        "impact_speed": args.impact_speed,
        "active_grid_scale": args.active_grid_scale,
        "max_iters": args.max_iters,
        "cohesion": args.cohesion if args.material == "DruckerPrager" else None,
        "friction_angle_degrees": args.friction_angle if args.material == "DruckerPrager" else None,
        "yield_stress": args.yield_stress if args.material == "VonMises" else None,
        "hardening_modulus": args.hardening_modulus if args.material == "VonMises" else None,
        "loss": float(0.5 * np.mean(np.sum(difference**2, axis=1))),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "finite": bool(
            np.isfinite(material_vjp).all()
            and np.isfinite(gradient["initial_position"]).all()
            and np.isfinite(gradient["initial_velocity"]).all()
        ),
        "material_parameter_vjp": material_vjp.tolist(),
        "plastic_parameter_vjp": dict(zip(plastic_names, material_vjp[2:].tolist())),
        "max_equivalent_plastic_strain": float(np.max(plastic_strain)),
        "plastic_particle_count": int(np.count_nonzero(plastic_strain > 1.0e-12)),
        "friction_coefficient_vjp": float(gradient["friction_coefficient"]),
        "initial_position_vjp_norm": float(np.linalg.norm(gradient["initial_position"])),
        "initial_velocity_vjp_norm": float(np.linalg.norm(gradient["initial_velocity"])),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "differentiable_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["finite"] or not summary["initial_velocity_vjp_norm"]:
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()
