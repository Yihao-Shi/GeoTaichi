"""Thin three-dimensional extrusion of the submarine landslide benchmark."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geotaichi import MPM, init
from examples.mmpm.SubmarineLandslide.submarine_landslide_2d import (
    FLUID_DENSITY,
    FRICTION_ANGLE,
    GRAVITY,
    INITIAL_POROSITY,
    K0,
    MAXIMUM_POROSITY,
    REFERENCE_DOI,
    SAND_AREA,
    SAND_LEFT_X,
    SAND_RIGHT_X,
    SAND_TOP,
    SLOPE_NORMAL,
    SLOPE_START_X,
    SOLID_DENSITY,
    TANK_LENGTH,
    WATER_AREA,
    WATER_DEPTH,
)

DOMAIN_HEIGHT = 1.8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--thickness", type=float, default=0.08)
    parser.add_argument("--dt", type=float, default=5.0e-5)
    parser.add_argument("--time", type=float, default=0.8)
    parser.add_argument("--save-interval", type=float, default=0.04)
    parser.add_argument("--solid-ppc", type=int, default=1)
    parser.add_argument("--fluid-ppc", type=int, default=1)
    parser.add_argument("--pressure-iterations", type=int, default=200)
    parser.add_argument("--wall-cells", type=int, default=1)
    parser.add_argument("--device-memory-gb", type=float, default=7.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "OutputData" / "rzadkiewicz_thin_slice_3d",
    )
    parser.add_argument("--save-grid", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    return parser.parse_args()


def validate(args):
    if (
        min(
            args.dx,
            args.thickness,
            args.dt,
            args.time,
            args.save_interval,
            args.device_memory_gb,
        )
        <= 0.0
        or min(args.solid_ppc, args.fluid_ppc, args.pressure_iterations, args.wall_cells) <= 0
    ):
        raise ValueError("all spacings, times, counts, and memory limits must be positive")
    cells = np.asarray([TANK_LENGTH, args.thickness, DOMAIN_HEIGHT]) / args.dx
    if not np.allclose(cells, np.round(cells), atol=1.0e-12, rtol=0.0):
        raise ValueError("dx must divide the tank length, extrusion thickness, and domain height")
    if np.any(np.round(cells).astype(int) % 2):
        raise ValueError("two-level MGPCG requires an even cell count along every axis")
    if round(args.thickness / args.dx) <= 2 * args.wall_cells:
        raise ValueError("the thin slice needs at least one interior cell between its side walls")


def make_region(name, bounds, volume, predicate):
    return {
        "Name": name,
        "Type": "UserDefined",
        "BoundingBoxPoint": bounds[0],
        "BoundingBoxSize": bounds[1],
        "RegionVolume": lambda: volume,
        "RegionFunction": predicate,
    }


def regions(thickness, wall):
    def water(position, particle_radius=0.0):
        bed = ti.max(0.0, position[0] - SLOPE_START_X)
        return (
            0.0 <= position[0] <= TANK_LENGTH
            and wall <= position[1] <= thickness - wall
            and bed <= position[2] <= WATER_DEPTH
        )

    def sand(position, particle_radius=0.0):
        return (
            SAND_LEFT_X <= position[0] <= SAND_RIGHT_X
            and wall <= position[1] <= thickness - wall
            and position[0] - SLOPE_START_X <= position[2] <= SAND_TOP
        )

    return [
        make_region(
            "water",
            ([0.0, wall, 0.0], [TANK_LENGTH, thickness - 2.0 * wall, WATER_DEPTH]),
            WATER_AREA * (thickness - 2.0 * wall),
            water,
        ),
        make_region(
            "sand",
            (
                [SAND_LEFT_X, wall, SAND_LEFT_X - SLOPE_START_X],
                [
                    SAND_RIGHT_X - SAND_LEFT_X,
                    thickness - 2.0 * wall,
                    SAND_TOP - (SAND_LEFT_X - SLOPE_START_X),
                ],
            ),
            SAND_AREA * (thickness - 2.0 * wall),
            sand,
        ),
    ]


def write_metrics(output, expected_fluid, expected_solid, args):
    files = sorted((output / "particles").glob("MPMParticle*.npz"))
    if len(files) < 2:
        raise RuntimeError("the 3D landslide produced fewer than two particle snapshots")
    rows = []
    finite = True
    for file_name in files:
        with np.load(file_name) as data:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            fluid_velocity = data["fluid_velocity"]
            solid_velocity = data["solid_velocity"]
            porosity = data["porosity"]
            finite &= bool(
                np.isfinite(position[active]).all()
                and np.isfinite(fluid_velocity[fluid]).all()
                and np.isfinite(solid_velocity[solid]).all()
                and np.isfinite(data["pressure"][active]).all()
            )
            outside = np.any(
                (position[active] < -1.0e-10 * args.dx)
                | (position[active] > np.array([TANK_LENGTH, args.thickness, DOMAIN_HEIGHT]) + 1.0e-10 * args.dx),
                axis=1,
            )
            solid_position = position[solid]
            rows.append(
                [
                    float(data["t_current"]),
                    int(np.count_nonzero(fluid)),
                    int(np.count_nonzero(solid)),
                    *solid_position.mean(axis=0),
                    float(np.min(solid_position[:, 0])),
                    float(np.linalg.norm(solid_velocity[solid], axis=1).max()),
                    float(position[fluid, 2].max()),
                    int(np.count_nonzero(outside)),
                    float(porosity[solid].min()),
                    float(porosity[solid].max()),
                ]
            )
    rows = np.asarray(rows, dtype=np.float64)
    metrics = {
        "case": "Rzadkiewicz submerged landslide, thin 3D extrusion",
        "reference_doi": REFERENCE_DOI,
        "thickness_m": args.thickness,
        "interior_thickness_m": args.thickness - 2.0 * args.wall_cells * args.dx,
        "snapshots": len(rows),
        "final_time_s": float(rows[-1, 0]),
        "duration_complete": math.isclose(float(rows[-1, 0]), args.time, abs_tol=0.1 * args.dt),
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "particle_conservation": bool(np.all(rows[:, 1] == expected_fluid) and np.all(rows[:, 2] == expected_solid)),
        "finite": finite,
        "maximum_particles_outside_domain": int(rows[:, 9].max()),
        "solid_centroid_leftward_displacement_m": float(rows[0, 3] - rows[-1, 3]),
        "solid_centroid_downward_displacement_m": float(rows[0, 5] - rows[-1, 5]),
        "solid_front_leftward_displacement_m": float(rows[0, 6] - rows[:, 6].min()),
        "maximum_solid_speed_mps": float(rows[:, 7].max()),
        "free_surface_excursion_m": float(np.ptp(rows[:, 8])),
        "minimum_solid_porosity": float(rows[:, 10].min()),
        "maximum_solid_porosity": float(rows[:, 11].max()),
        "solid_stress_cutoff_porosity": MAXIMUM_POROSITY,
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["particle_conservation"]
        and metrics["maximum_particles_outside_domain"] == 0
        and metrics["solid_centroid_leftward_displacement_m"] >= 0.01
        and metrics["solid_centroid_downward_displacement_m"] >= 0.01
        and metrics["solid_front_leftward_displacement_m"] >= 0.01
        and metrics["maximum_solid_speed_mps"] >= 0.05
        and metrics["free_surface_excursion_m"] >= 0.002
        and metrics["minimum_solid_porosity"] > 0.0
        and metrics["maximum_solid_porosity"] <= 1.0 + 1.0e-8
    )
    np.savetxt(
        output / "landslide_3d_diagnostics.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,fluid_particles,solid_particles,solid_centroid_x_m,solid_centroid_y_m,"
            "solid_centroid_z_m,solid_front_x_m,solid_speed_max_mps,fluid_top_z_m,"
            "particles_outside_domain,solid_porosity_min,solid_porosity_max"
        ),
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("3D submarine-landslide validation failed; inspect metrics.json")


@ti.kernel
def initialize_hydrostatic_landslide(particle_count: int, particle: ti.template()):
    for p in range(particle_count):
        if int(particle[p].active) == 1:
            particle[p].pressure = FLUID_DENSITY * GRAVITY * ti.max(WATER_DEPTH - particle[p].x[2], 0.0)
            if int(particle[p].phase) == 1:
                sand_depth = ti.max(SAND_TOP - particle[p].x[2], 0.0)
                vertical = (1.0 - INITIAL_POROSITY) * (SOLID_DENSITY - FLUID_DENSITY) * GRAVITY * sand_depth
                particle[p].stress = ti.Vector([-K0 * vertical, -K0 * vertical, -vertical, 0.0, 0.0, 0.0])


def solid_cell(start, end, normal, wall_cells):
    return {
        "BoundaryType": "SolidCell",
        "StartPoint": start,
        "EndPoint": end,
        "Norm": normal,
        "CellThickness": wall_cells,
    }


def boundaries(thickness, wall_cells):
    top = DOMAIN_HEIGHT - 1.0e-6
    slope_normal = [float(SLOPE_NORMAL[0]), 0.0, float(SLOPE_NORMAL[1])]
    walls = [
        solid_cell([0.0, 0.0, 0.0], [SLOPE_START_X, thickness, 0.0], [0.0, 0.0, -1.0], wall_cells),
        solid_cell([0.0, 0.0, 0.0], [0.0, thickness, top], [-1.0, 0.0, 0.0], wall_cells),
        solid_cell([TANK_LENGTH, 0.0, 0.0], [TANK_LENGTH, thickness, top], [1.0, 0.0, 0.0], wall_cells),
        solid_cell([0.0, 0.0, 0.0], [TANK_LENGTH, 0.0, top], [0.0, -1.0, 0.0], wall_cells),
        solid_cell([0.0, thickness, 0.0], [TANK_LENGTH, thickness, top], [0.0, 1.0, 0.0], wall_cells),
        {
            "BoundaryType": "SolidPlaneCell",
            "StartPoint": [SLOPE_START_X, 0.0, 0.0],
            "EndPoint": [TANK_LENGTH, thickness, WATER_DEPTH],
            "Point": [SLOPE_START_X, 0.0, 0.0],
            "Norm": slope_normal,
        },
    ]
    walls.extend(
        [
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, None, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [SLOPE_START_X, thickness, 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.0, thickness, top],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [TANK_LENGTH, 0.0, 0.0],
                "EndPoint": [TANK_LENGTH, thickness, top],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [TANK_LENGTH, 0.0, top],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [0.0, thickness, 0.0],
                "EndPoint": [TANK_LENGTH, thickness, top],
            },
        ]
    )
    return walls


def run(args):
    validate(args)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    interior_thickness = args.thickness - 2.0 * args.wall_cells * args.dx
    fluid_count = math.ceil(WATER_AREA * interior_thickness * (args.fluid_ppc / args.dx) ** 3)
    solid_count = math.ceil(SAND_AREA * interior_thickness * (args.solid_ppc / args.dx) ** 3)
    capacity = math.ceil(1.15 * (fluid_count + solid_count))
    nx, ny, nz = np.round(np.asarray([TANK_LENGTH, args.thickness, DOMAIN_HEIGHT]) / args.dx).astype(int)
    max_constraints = math.ceil(1.2 * (2 * (nx + 1) * (nz + 1) + 2 * (ny + 1) * (nz + 1) + (nx + 1) * (ny + 1)))
    print(
        f"# Rzadkiewicz 3D thin slice: thickness={args.thickness:g} m, dx={args.dx:g} m, "
        f"estimated particles={fluid_count + solid_count:,}, capacity={capacity:,}"
    )
    metadata = {
        "benchmark": "Rzadkiewicz submerged granular landslide, thin 3-D extrusion",
        "reference_doi": REFERENCE_DOI,
        "extrusion_thickness_m": args.thickness,
        "shared_grid_spacing_m": args.dx,
        "solid_particles_per_axis": args.solid_ppc,
        "fluid_particles_per_axis": args.fluid_ppc,
        "estimated_particle_count": fluid_count + solid_count,
        "duration_s": args.time,
        "timestep_s": args.dt,
        "adaptation": "The plane-strain benchmark is extruded into a thin no-slip channel.",
    }
    (args.output / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    init(
        dim=3,
        arch=args.arch,
        default_fp="float64",
        device_memory_GB=args.device_memory_gb,
        offline_cache=True,
        debug=False,
    )
    mpm = MPM()
    mpm.set_configuration(
        domain=[TANK_LENGTH, args.thickness, DOMAIN_HEIGHT],
        background_damping=0.0,
        gravity=[0.0, 0.0, -GRAVITY],
        alphaPIC=1.0,
        mapping="USL",
        shape_function="QuadBSpline",
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection="Affine",
        delayed_fluid_advection=True,
    )
    mpm.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.time,
            "SaveInterval": args.save_interval,
            "SavePath": str(args.output),
        }
    )
    mpm.set_semi_implicit_solver_parameters(
        {
            "assemble_type": "MatrixFree",
            "pressure_solver": "MGPCG",
            "linear_solver": "MGPCG",
            "max_iteration_number": args.pressure_iterations,
            "residual_tolerance": 1.0e-8,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": capacity,
            "max_constraint_number": {"max_velocity_constraint": max_constraints},
        }
    )
    mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "SolidDensity": SOLID_DENSITY,
            "FluidDensity": FLUID_DENSITY,
            "Porosity": INITIAL_POROSITY,
            "MaximumPorosity": MAXIMUM_POROSITY,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 1.0e-3,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 6.0e-3,
            "DragModel": "Ergun",
            "YoungModulus": 5.0e6,
            "PoissonRatio": 0.30,
            "Cohesion": 0.0,
            "Friction": FRICTION_ANGLE,
            "Dilation": 0.0,
        },
    )
    mpm.add_element({"ElementType": "R8N3D", "ElementSize": [args.dx] * 3})
    mpm.add_region(regions(args.thickness, args.wall_cells * args.dx))
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": args.fluid_ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
                {
                    "RegionName": "sand",
                    "nParticlesPerCell": args.solid_ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
            ]
        }
    )
    initialize_hydrostatic_landslide(int(mpm.scene.particleNum[0]), mpm.scene.particle)
    initial_phase = mpm.scene.particle.phase.to_numpy()[: int(mpm.scene.particleNum[0])]
    expected_fluid = int(np.count_nonzero(initial_phase == 2))
    expected_solid = int(np.count_nonzero(initial_phase == 1))
    mpm.add_boundary_condition(boundaries(args.thickness, args.wall_cells))
    mpm.select_save_data(particle=True, grid=args.save_grid, object=False)
    mpm.run()
    write_metrics(args.output, expected_fluid, expected_solid, args)
    if not args.no_post:
        mpm.postprocessing()


if __name__ == "__main__":
    run(parse_args())
