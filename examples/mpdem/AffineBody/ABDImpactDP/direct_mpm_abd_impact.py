#!/usr/bin/env python3
"""Polyhedral ABD impact into an ordinary finite-strain Direct-MPM bed."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MESH = ROOT / "assets/mesh/AffineBody/lowpoly_sphere.obj"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--contact-model", choices=("BarrierIPC", "SemiIPC"), default="BarrierIPC")
    parser.add_argument("--material", choices=("NeoHookean", "DruckerPrager", "VonMises"), default="DruckerPrager")
    parser.add_argument("--dilation-angle", type=float, default=0.0)
    parser.add_argument("--inexact-newton", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spacing", type=float, default=0.025)
    parser.add_argument("--ppc", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--impact-speed", type=float, default=1.5)
    parser.add_argument("--start-height", type=float, default=0.39)
    parser.add_argument("--output-interval", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("OutputData") / "direct_mpm_abd_impact"),
    )
    args = parser.parse_args()
    if (
        min(args.spacing, args.dt, args.impact_speed, args.start_height) <= 0.0
        or min(args.ppc, args.steps, args.output_interval) <= 0
        or not 0.38 <= args.start_height < 0.9
    ):
        parser.error("spacing, ppc, steps, dt, impact speed, and output interval must be positive")
    if not 0.0 <= args.dilation_angle <= 30.0:
        parser.error("dilation-angle must be between 0 and 30 degrees")
    if args.inexact_newton is None:
        args.inexact_newton = args.material == "DruckerPrager" and args.dilation_angle != 30.0
    if args.inexact_newton and (args.material != "DruckerPrager" or args.dilation_angle == 30.0):
        parser.error("inexact-newton requires nonassociated DruckerPrager")

    import geotaichi as gt

    gt.init(arch=args.arch, default_fp="float64", log=True, offline_cache=False)
    coupling = gt.MPDEM(coupling=False, log=True)

    mpm = coupling.mpm
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[1.0, 1.0, 1.0],
        gravity=[0.0, 0.0, -9.81],
        visualize=True,
        ipc=True,
    )
    body = mpm.create_body()
    body.add_cube(
        start=[0.25, 0.25, 0.08],
        end=[0.75, 0.75, 0.30],
        spacing=args.spacing,
        ppc=args.ppc,
        name="ordinary_mpm_bed",
        grid_size=args.spacing,
        xmin=[0.08, 0.08, 0.0],
        xmax=[0.92, 0.92, 0.75],
    )
    mpm.add_body(body)
    material = {
        "model": args.material,
        "density": 1700.0,
        "young_modulus": 2.0e5,
        "poisson_ratio": 0.3,
    }
    if args.material == "DruckerPrager":
        material.update(Cohesion=250.0, FrictionAngle=30.0, DilationAngle=args.dilation_angle, dpType="Circumscribed")
    elif args.material == "VonMises":
        material.update(YieldStress=5.0e3, HardeningModulus=2.0e4)
    mpm.add_material(**material)
    mpm.add_element({"ElementSize": args.spacing, "ShapeFunction": "Linear"})
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    grid_shape = (
        np.ceil((np.asarray([0.92, 0.92, 0.75]) - np.asarray([0.08, 0.08, 0.0])) / args.spacing).astype(int) + 1
    )
    nx, ny, nz = map(int, grid_shape)
    nodes = np.arange(nx * ny * nz, dtype=np.int32).reshape(nz, ny, nx)
    bottom = nodes[0].reshape(-1)
    x_sides = np.concatenate((nodes[:, :, 0].reshape(-1), nodes[:, :, -1].reshape(-1)))
    y_sides = np.concatenate((nodes[:, 0, :].reshape(-1), nodes[:, -1, :].reshape(-1)))
    fixed = [
        list(3 * np.unique(np.concatenate((bottom, x_sides))) + 0),
        list(3 * np.unique(np.concatenate((bottom, y_sides))) + 1),
        list(3 * bottom + 2),
    ]
    boundary = DirichletBoundary()
    boundary.append(fixed, [0.0] * sum(map(len, fixed)))
    mpm.add_boundary_condition(dirichlet=boundary)
    mpm.memory_allocate(
        {"max_material_number": 1, "max_particle_number": body.particle_counter},
        log=True,
    )
    mpm.add_contact(
        args.contact_model,
        dhat=0.6 * args.spacing,
        dmin=0.1 * args.spacing,
        kappa=3.0e5,
        mu=0.25,
        epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=-1,
        contact_search="LinkedCell",
    )

    dem = coupling.dem
    dem.set_configuration(
        domain=[1.0, 1.0, 1.0],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, -9.81],
        visualize=True,
        log=True,
    )
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        coupling_scheme="MPDEM",
        gravity=[0.0, 0.0, -9.81],
        search="BVH",
        visualize=True,
        log=True,
    )
    dem.set_affine_body_parameters(
        contact_model=args.contact_model,
        assemble_type="HashTriplet",
        young_modulus=5.0e5,
        dhat=0.6 * args.spacing,
        barrier_stiffness=3.0e5,
        friction_mode="lagged",
        friction_iterations=-1,
        max_newton_iteration=100,
        newton_tolerance=1.0e-3,
        linear_tolerance=1.0e-8,
        linear_max_iteration=5000,
        line_search_max_iteration=24,
        max_step=0.02,
        ccd=True,
        ccd_type="ccd",
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": 1,
            "max_rigid_template_number": 1,
            "surface_node_number": 12,
            "body_coordination_number": 2,
            "wall_coordination_number": 0,
            "affine_contact_block_capacity": 512,
            "compaction_ratio": [1.0, 1.0],
        },
        log=True,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 7800.0})
    dem.add_template(
        {
            "Name": "impactor",
            "TemplateType": "AffineBody",
            "Object": gt.polyhedron(file=str(MESH)).reset(False),
        }
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {
                "Name": "impactor",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.5, 0.5, args.start_height],
                "BoundingRadius": 0.08,
                "InitialVelocity": [0.0, 0.0, -args.impact_speed],
            },
        }
    )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property={"Dhat": 0.6 * args.spacing, "BarrierStiffness": 3.0e5, "Friction": 0.25},
    )

    coupling.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.steps * args.dt,
            "SaveInterval": args.output_interval * args.dt,
            "SavePath": args.output_dir,
            "inexact_newton": args.inexact_newton,
            "enable_step_retry": args.contact_model == "BarrierIPC",
            "step_retry_max_retries": 3,
            "step_retry_reduction": 0.5,
            "step_retry_minimum_timestep": args.dt / 8.0,
        },
        log=True,
    )
    coupling.memory_allocate(
        {
            "body_coordination_number": 8,
            "max_point_triangle_pairs": 4096,
            "max_point_edge_pairs": 1,
        },
        log=True,
    )
    coupling.choose_contact_model(
        args.contact_model,
        dhat=0.6 * args.spacing,
        dmin=0.1 * args.spacing,
        kappa=3.0e5,
        friction_coefficient=0.25,
        epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=-1,
        project_pd=True,
    )
    coupling.add_ipc_property(
        MPMbody=0,
        AffineBody=0,
        property={
            "dhat": 0.6 * args.spacing,
            "dmin": 0.1 * args.spacing,
            "kappa": 3.0e5,
            "friction_coefficient": 0.25,
            "epsv": 1.0e-3,
        },
    )

    peak_contacts = [0]
    peak_candidates = [0]
    minimum_distance = [np.inf]

    def sample(engine):
        peak_contacts[0] = max(peak_contacts[0], int(engine.last_contact_count))
        peak_candidates[0] = max(peak_candidates[0], int(engine.last_candidate_count))
        minimum_distance[0] = min(minimum_distance[0], float(engine.mixed.diagnostics()["minimum_distance"]))

    result = coupling.run(verbose=False, postprocessing=[sample])
    mpm_position = coupling.mpm.enginer.mpm.particle.x.to_numpy()
    affine_position = coupling.dem.enginer.operator.x.to_numpy()
    summary = {
        "case": "ordinary_direct_mpm_polyhedral_abd_ipc",
        "contact_model": args.contact_model,
        "material": args.material,
        "dilation_angle": args.dilation_angle,
        "inexact_newton": args.inexact_newton,
        "linear_solver": coupling.enginer.matrix.solver,
        "mpm_particles": int(body.particle_counter),
        "affine_bodies": 1,
        "steps": int(result["step"]),
        "completed_time": float(result["time"]),
        "maximum_candidate_contacts": int(peak_candidates[0]),
        "maximum_active_contacts": int(peak_contacts[0]),
        "minimum_contact_distance": float(minimum_distance[0]),
        "mpm_bounds": [mpm_position.min(axis=0).tolist(), mpm_position.max(axis=0).tolist()],
        "affine_bounds": [affine_position.min(axis=0).tolist(), affine_position.max(axis=0).tolist()],
        "converged": bool(result["converged"]),
        "finite": bool(np.isfinite(mpm_position).all() and np.isfinite(affine_position).all()),
    }
    output = Path(args.output_dir)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        not summary["converged"]
        or not summary["finite"]
        or not summary["maximum_active_contacts"]
        or not np.isclose(summary["completed_time"], args.steps * args.dt)
    ):
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()
