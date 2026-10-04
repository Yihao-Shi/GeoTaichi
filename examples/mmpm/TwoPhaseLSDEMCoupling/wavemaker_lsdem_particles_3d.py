"""3D two-point wavemaker with fully resolved LevelSet DEM particles.

Fluid points exchange momentum with the LSDEM bodies through IBM.  The
trapezoidal MPM bed uses the ordinary point--level-set contact law.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geotaichi import DEMPM, init, polyhedron
from examples.mmpm.TwoPhaseWavemaker3D.two_layer_two_phase_wavemaker_3d import set_moving_piston_mac_boundary
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d import (
    enforce_moving_piston_particles,
    linear_piston_wave,
    piston_displacement,
    piston_velocity,
    sdf_surface_gauge,
)
from src.utils.SolverRuntime import python_callback


TANK = (1.20, 0.24, 0.36)
WATER_DEPTH = 0.24
SLOPE_HEIGHT = WATER_DEPTH / 3.0
SLOPE_TOE = 0.35
SLOPE_RUN = 0.35
PISTON_MEAN_X = 0.06
SOIL_YOUNG_MODULUS = 12.0e6
SOIL_COHESION = 300.0
SOIL_FRICTION = 26.0
LSDEM_BODY_COUNT = 5


def parse_args():
    parser = argparse.ArgumentParser(description="TwoPhaseDoubleLayer wavemaker with LSDEM particles")
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=5.0e-5)
    parser.add_argument("--time", type=float, default=4.0)
    parser.add_argument("--save-interval", type=float, default=0.02)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--frequency", type=float, default=0.8)
    parser.add_argument("--wave-velocity", type=float, default=0.10)
    parser.add_argument("--ramp-time", type=float, default=0.75)
    parser.add_argument("--alpha-pic", type=float, default=1.0)
    parser.add_argument("--pressure-iterations", type=int, default=1000)
    parser.add_argument("--device-memory", type=float, default=4.0)
    parser.add_argument("--soil-young-modulus", type=float, default=SOIL_YOUNG_MODULUS)
    parser.add_argument("--soil-cohesion", type=float, default=SOIL_COHESION)
    parser.add_argument("--soil-friction", type=float, default=SOIL_FRICTION)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples" / "mmpm" / "TwoPhaseLSDEMCoupling" / "OutputData" / "wavemaker_lsdem_particles_3d",
    )
    return parser.parse_args()


def write_metrics(output, expected_fluid, expected_solid, args):
    mpm_files = sorted((output / "particles").glob("MPMParticle*.npz"))
    rigid_files = sorted((output / "particles").glob("LSDEMRigid*.npz"))
    grids = {path.stem.removeprefix("MPMGrid"): path for path in (output / "grids").glob("MPMGrid*.npz")}
    if len(mpm_files) < 2 or len(rigid_files) < 2:
        raise RuntimeError("coupled wavemaker produced fewer than two MPM or LSDEM snapshots")

    mpm_rows = []
    initial_solid_position = None
    finite = True
    for file_name in mpm_files:
        frame = file_name.stem.removeprefix("MPMParticle")
        if frame not in grids:
            raise RuntimeError(f"missing grid snapshot for MPM frame {frame}")
        with np.load(file_name) as data, np.load(grids[frame]) as grid:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            if initial_solid_position is None:
                initial_solid_position = position[solid].copy()
            solid_displacement = float(np.linalg.norm(position[solid] - initial_solid_position, axis=1).max())
            time = float(data["t_current"])
            surface = (
                WATER_DEPTH
                if time == 0.0
                else sdf_surface_gauge(grid["cell_type"], grid["cell_fluid_sdf"], args.dx, 0.50, 2.0 * args.dx)
            )
            outside = np.any(
                (position[active] < -1.0e-10 * args.dx) | (position[active] > np.asarray(TANK) + 1.0e-10 * args.dx),
                axis=1,
            )
            finite &= bool(
                np.isfinite(position[active]).all()
                and np.isfinite(data["fluid_velocity"][fluid]).all()
                and np.isfinite(data["solid_velocity"][solid]).all()
                and np.isfinite(data["pressure"][active]).all()
                and np.isfinite(surface)
            )
            mpm_rows.append(
                [
                    time,
                    np.count_nonzero(fluid),
                    np.count_nonzero(solid),
                    np.count_nonzero(outside),
                    surface,
                    solid_displacement,
                ]
            )

    rigid_rows = []
    for file_name in rigid_files:
        with np.load(file_name) as data:
            centers = data["mass_center"]
            forces = data["contact_force"]
            finite &= bool(np.isfinite(centers).all() and np.isfinite(forces).all())
            rigid_rows.append([float(data["t_current"]), float(np.linalg.norm(forces, axis=1).max()), len(centers)])
    mpm_rows = np.asarray(mpm_rows, dtype=np.float64)
    rigid_rows = np.asarray(rigid_rows, dtype=np.float64)
    _, expected_amplitude, _ = linear_piston_wave(args.frequency, WATER_DEPTH, args.wave_velocity)
    with np.load(rigid_files[0]) as data:
        initial_centers = data["mass_center"].copy()
    maximum_rigid_displacement = 0.0
    for file_name in rigid_files:
        with np.load(file_name) as data:
            maximum_rigid_displacement = max(
                maximum_rigid_displacement,
                float(np.linalg.norm(data["mass_center"] - initial_centers, axis=1).max()),
            )
    metrics = {
        "case": "3D two-phase two-point semi-implicit MPM--LSDEM wavemaker",
        "coupling": {"fluid_lsdem": "IBM", "solid_lsdem": "ordinary point-level-set contact"},
        "velocity_projection": "Affine",
        "alpha_pic": getattr(args, "alpha_pic", 1.0),
        "final_time_s": float(min(mpm_rows[-1, 0], rigid_rows[-1, 0])),
        "duration_complete": bool(
            math.isclose(mpm_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
            and math.isclose(rigid_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
        ),
        "finite": finite,
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "particle_conservation": bool(
            np.all(mpm_rows[:, 1] == expected_fluid) and np.all(mpm_rows[:, 2] == expected_solid)
        ),
        "maximum_mpm_particles_outside_domain": int(mpm_rows[:, 3].max()),
        "maximum_mpm_solid_displacement_m": float(mpm_rows[:, 5].max()),
        "expected_linear_wave_amplitude_m": expected_amplitude,
        "surface_gauge_excursion_m": float(np.ptp(mpm_rows[:, 4])),
        "maximum_lsdem_displacement_m": maximum_rigid_displacement,
        "maximum_coupling_force_n": float(rigid_rows[:, 1].max()),
        "expected_lsdem_bodies": LSDEM_BODY_COUNT,
        "minimum_lsdem_bodies": int(rigid_rows[:, 2].min()),
        "soil_young_modulus_pa": args.soil_young_modulus,
        "soil_cohesion_pa": args.soil_cohesion,
        "soil_friction_deg": args.soil_friction,
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["particle_conservation"]
        and metrics["maximum_mpm_particles_outside_domain"] == 0
        and metrics["minimum_lsdem_bodies"] == metrics["expected_lsdem_bodies"]
        and metrics["surface_gauge_excursion_m"] >= max(0.75 * args.dx, expected_amplitude)
        and metrics["maximum_mpm_solid_displacement_m"] >= 0.05 * args.dx
        and metrics["maximum_lsdem_displacement_m"] > 1.0e-5
        and metrics["maximum_coupling_force_n"] > 0.0
    )
    np.savetxt(
        output / "coupled_wavemaker_diagnostics.csv",
        mpm_rows,
        delimiter=",",
        header="time_s,fluid_particles,solid_particles,particles_outside_domain,surface_gauge_m,solid_displacement_max_m",
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("coupled wavemaker validation failed; inspect metrics.json")


def make_slope_region(wall):
    def volume():
        plateau = (TANK[0] - wall - SLOPE_TOE - SLOPE_RUN) * SLOPE_HEIGHT
        return (plateau + 0.5 * SLOPE_RUN * SLOPE_HEIGHT) * (TANK[1] - 2.0 * wall)

    def inside(position, particle_radius=0.0):
        top = ti.min(SLOPE_HEIGHT, SLOPE_HEIGHT * (position[0] - SLOPE_TOE) / SLOPE_RUN)
        return (
            SLOPE_TOE <= position[0] <= TANK[0] - wall
            and wall <= position[1] <= TANK[1] - wall
            and wall + 0.0 * particle_radius <= position[2] <= top
        )

    return {
        "Name": "trapezoidal_bed",
        "Type": "UserDefined",
        "BoundingBoxPoint": [SLOPE_TOE, wall, wall],
        "BoundingBoxSize": [TANK[0] - wall - SLOPE_TOE, TANK[1] - 2.0 * wall, SLOPE_HEIGHT - wall],
        "RegionVolume": volume,
        "RegionFunction": inside,
    }


@ti.kernel
def initialize_hydrostatic_state(particle_num: int, friction_angle: float, particle: ti.template()):
    k0 = 1.0 - ti.sin(friction_angle * math.pi / 180.0)
    for p in range(particle_num):
        particle[p].pressure = 1000.0 * 9.81 * ti.max(WATER_DEPTH - particle[p].x[2], 0.0)
        if int(particle[p].phase) == 1:
            bed_top = ti.min(
                SLOPE_HEIGHT,
                ti.max(0.0, SLOPE_HEIGHT * (particle[p].x[0] - SLOPE_TOE) / SLOPE_RUN),
            )
            effective = 0.60 * (2650.0 - 1000.0) * 9.81 * ti.max(bed_top - particle[p].x[2], 0.0)
            particle[p].stress = ti.Vector([-k0 * effective, -k0 * effective, -effective, 0.0, 0.0, 0.0])


def run(args):
    if (
        min(args.dx, args.dt, args.time, args.save_interval, args.frequency, args.ramp_time, args.device_memory) <= 0.0
        or min(args.ppc, args.pressure_iterations) <= 0
    ):
        raise ValueError("grid, time and wave parameters must be positive")
    if not 0.0 <= args.alpha_pic <= 1.0:
        raise ValueError("alpha-pic must lie in [0, 1]")
    if args.soil_young_modulus <= 0.0 or args.soil_cohesion < 0.0 or not 0.0 <= args.soil_friction < 90.0:
        raise ValueError("soil Young's modulus must be positive, cohesion nonnegative, and friction in [0, 90)")
    cells = np.asarray(TANK) / args.dx
    if not np.allclose(cells, np.round(cells), atol=1.0e-12) or np.any(np.round(cells).astype(int) % 2):
        raise ValueError("dx must divide the tank into even cell counts for two-level MGPCG")
    stroke = args.wave_velocity / (2.0 * math.pi * args.frequency)
    if PISTON_MEAN_X - stroke <= args.dx:
        raise ValueError("piston stroke leaves less than one cell of clearance")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    wall = args.dx
    point_spacing = args.dx / args.ppc
    fluid_count = math.ceil(
        (TANK[0] - PISTON_MEAN_X - wall) * (TANK[1] - 2.0 * wall) * (WATER_DEPTH - wall) / point_spacing**3
    )
    slope_area = (TANK[0] - wall - SLOPE_TOE - SLOPE_RUN) * SLOPE_HEIGHT + 0.5 * SLOPE_RUN * SLOPE_HEIGHT
    solid_count = math.ceil(slope_area * (TANK[1] - 2.0 * wall) / point_spacing**3)
    max_particles = math.ceil(1.1 * (fluid_count + solid_count))

    init(
        dim=3,
        arch=args.arch,
        default_fp="float64",
        default_ip="int32",
        device_memory_GB=args.device_memory,
        offline_cache=True,
        debug=False,
        log=False,
    )
    dempm = DEMPM()
    dempm.set_configuration(
        domain=list(TANK),
        coupling_scheme="MPDEM",
        particle_interaction=True,
        wall_interaction=False,
        gravity=[0.0, 0.0, -9.81],
        visualize=True,
    )
    dempm.mpm.set_configuration(
        background_damping=0.03,
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, 0.0, -9.81],
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection="Affine",
        delayed_fluid_advection=True,
        particle_shifting=True,
        free_surface_detection=False,
        visualize=True,
    )
    dempm.mpm.set_semi_implicit_solver_parameters(
        {
            "assemble_type": "MatrixFree",
            "pressure_solver": "MGPCG",
            "linear_solver": "MGPCG",
            "max_iteration_number": args.pressure_iterations,
            "residual_tolerance": 1.0e-7,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    dempm.dem.set_configuration(
        scheme="LSDEM",
        boundary=["Destroy", "Destroy", "Destroy"],
        gravity=[0.0, 0.0, -9.81],
        engine="VelocityVerlet",
        search="LinkedCell",
        visualize=True,
    )
    dempm.set_solver(
        {
            "Timestep": args.dt,
            "DEMTimestep": args.dt,
            "SimulationTime": args.time,
            "SaveInterval": args.save_interval,
            "SavePath": str(output),
            "CFL": 0.4,
        }
    )
    dempm.dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_rigid_body_number": LSDEM_BODY_COUNT,
            "max_rigid_template_number": 2,
            "levelset_grid_number": 800000,
            "surface_node_number": 24000,
            "max_plane_number": 0,
            "body_coordination_number": 12,
            "wall_coordination_number": 0,
            "verlet_distance_multiplier": [0.15, 0.10],
            "point_coordination_number": [6, 2],
            "compaction_ratio": [0.15, 0.15],
        }
    )
    dempm.mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": max_particles,
            "verlet_distance_multiplier": 0.5,
            "max_constraint_number": {"max_velocity_constraint": 50000},
        }
    )
    dempm.memory_allocate(
        {"body_coordination_number": LSDEM_BODY_COUNT, "wall_coordination_number": 0, "compaction_ratio": [0.1, 0.1]}
    )

    dempm.dem.add_attribute(
        materialID=0,
        attribute={"Density": 2200.0, "ForceLocalDamping": 0.05, "TorqueLocalDamping": 0.05},
    )
    dempm.dem.add_template(
        {
            "Name": "tree_trunk",
            "Object": polyhedron(file=str(ROOT / "assets" / "mesh" / "LSDEM" / "tree_trunk.obj")).grids(
                space=0.04, extent=32
            ),
            "WriteFile": False,
        }
    )
    dempm.dem.add_template(
        {
            "Name": "irregular_grain",
            "Object": polyhedron(file=str(ROOT / "assets" / "mesh" / "LSDEM" / "sand.stl")).grids(
                space=0.05, extent=12
            ),
            "WriteFile": False,
        }
    )
    grain_centers = ([0.82, 0.06, 0.098], [0.87, 0.11, 0.099], [0.92, 0.16, 0.098], [0.98, 0.09, 0.100])
    dempm.dem.create_body(
        {
            "GenerateType": "Create",
            "BodyType": "RigidBody",
            "Template": [
                {
                    "Name": "tree_trunk",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [0.74, 0.12, 0.075],
                    "Radius": 0.075,
                    "BodyOrientation": [0.0, 75.0, 8.0],
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixMotion": ["Free", "Free", "Free"],
                },
                *[
                    {
                        "Name": "irregular_grain",
                        "GroupID": 0,
                        "MaterialID": 0,
                        "BodyPoint": center,
                        "Radius": 0.018,
                        "BodyOrientation": [17.0 * index, 29.0 * index, 41.0 * index],
                        "InitialVelocity": [0.0, 0.0, 0.0],
                        "FixMotion": ["Free", "Free", "Free"],
                    }
                    for index, center in enumerate(grain_centers, start=1)
                ],
            ],
        }
    )
    dempm.dem.choose_contact_model("Hertz Mindlin Model", None)
    dempm.dem.add_property(
        materialID1=0,
        materialID2=0,
        property={"ShearModulus": 20.0e6, "Poisson": 0.30, "Friction": 0.35, "Restitution": 0.2},
    )

    dempm.mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "SolidDensity": 2650.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.40,
            "MaximumPorosity": 0.56,
            "FluidBulkModulus": 2.2e9,
            "Permeability": 2.0e-4,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 3.0e-3,
            "DragModel": "Ergun",
            "YoungModulus": args.soil_young_modulus,
            "PoissonRatio": 0.30,
            "Cohesion": args.soil_cohesion,
            "Friction": args.soil_friction,
            "Dilation": 0.0,
        },
    )
    dempm.mpm.add_element({"ElementType": "R8N3D", "ElementSize": [args.dx] * 3})
    dempm.mpm.add_region(
        [
            make_slope_region(wall),
            {
                "Name": "water",
                "Type": "Rectangle",
                "BoundingBoxPoint": [PISTON_MEAN_X, wall, wall],
                "BoundingBoxSize": [TANK[0] - PISTON_MEAN_X - wall, TANK[1] - 2.0 * wall, WATER_DEPTH - wall],
            },
        ]
    )
    dempm.mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "trapezoidal_bed",
                    "nParticlesPerCell": args.ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
                {
                    "RegionName": "water",
                    "nParticlesPerCell": args.ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
            ]
        }
    )
    initialize_hydrostatic_state(int(dempm.mpm.scene.particleNum[0]), args.soil_friction, dempm.mpm.scene.particle)
    initial_phase = dempm.mpm.scene.particle.phase.to_numpy()[: int(dempm.mpm.scene.particleNum[0])]
    expected_fluid = int(np.count_nonzero(initial_phase == 2))
    expected_solid = int(np.count_nonzero(initial_phase == 1))

    walls = [
        ([wall, wall, wall], [TANK[0] - wall, wall, TANK[2]], [0.0, -1.0, 0.0]),
        ([wall, TANK[1] - wall, wall], [TANK[0] - wall, TANK[1] - wall, TANK[2]], [0.0, 1.0, 0.0]),
        ([wall, wall, wall], [TANK[0] - wall, TANK[1] - wall, wall], [0.0, 0.0, -1.0]),
        ([TANK[0] - wall, wall, wall], [TANK[0] - wall, TANK[1] - wall, TANK[2]], [1.0, 0.0, 0.0]),
    ]
    dempm.mpm.add_boundary_condition(
        [
            {"BoundaryType": "SolidCell", "StartPoint": a, "EndPoint": b, "Norm": n, "CellThickness": 1}
            for a, b, n in walls
        ]
    )

    def update_wavemaker(sims, scene):
        end_time = min(float(sims.current_time + sims.delta), args.time)
        velocity = piston_velocity(
            end_time - 0.5 * float(sims.delta), args.frequency, args.wave_velocity, args.ramp_time
        )
        position = PISTON_MEAN_X + piston_displacement(end_time, args.frequency, args.wave_velocity, args.ramp_time)
        set_moving_piston_mac_boundary(
            velocity,
            position,
            TANK[1],
            TANK[2],
            scene.element.grid_size,
            dempm.mpm.enginer.cell_type,
            dempm.mpm.enginer.solid_velocity_x,
        )

    @python_callback
    def constrain_piston_particles():
        end_time = min(float(dempm.mpm.sims.current_time + dempm.mpm.sims.delta), args.time)
        velocity = piston_velocity(
            end_time - 0.5 * float(dempm.mpm.sims.delta), args.frequency, args.wave_velocity, args.ramp_time
        )
        position = PISTON_MEAN_X + piston_displacement(end_time, args.frequency, args.wave_velocity, args.ramp_time)
        enforce_moving_piston_particles(
            int(dempm.mpm.scene.particleNum[0]), position, velocity, args.dx, dempm.mpm.scene.particle
        )

    dempm.mpm.select_save_data(particle=True, grid=True, object=False)
    dempm.dem.select_save_data(surface=True, grid=True, bounding=True)
    dempm.choose_contact_model("Hertz Mindlin Model", None)
    dempm.add_property(
        DEMmaterial=0,
        MPMmaterial=1,
        property={"ShearModulus": 30.0e6, "Poisson": 0.30, "Friction": 0.35, "Restitution": 0.2},
    )
    dempm.select_save_data(particle_particle_contact=True)
    dempm.run(
        mpm_mac_boundary_function=update_wavemaker,
        function=constrain_piston_particles,
    )
    write_metrics(output, expected_fluid, expected_solid, args)
    if not args.no_post:
        dempm.postprocessing(scheme="LSDEM")


if __name__ == "__main__":
    run(parse_args())
