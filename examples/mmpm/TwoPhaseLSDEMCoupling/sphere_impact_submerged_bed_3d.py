"""Section 5.2: LSDEM sphere impacting a saturated granular bed.

The 0.2 m diameter vessel is represented by 32 tangent solid-cell planes.
The air drop is replaced by the equivalent impact velocity sqrt(2 g h0), so
the simulation starts when the sphere first touches the free surface.
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


DOMAIN = (0.20, 0.20, 0.20)
BED_HEIGHT = 0.10
WATER_DEPTH = 0.05
WATER_SURFACE = BED_HEIGHT + WATER_DEPTH
SPHERE_DIAMETER = 0.025
CONTAINER_RADIUS = 0.10
CYLINDER_SIDES = 32
REFERENCE_PARTICLE_SPACING = 0.001
TEFLON_YOUNG_MODULUS = 500.0e6
TEFLON_POISSON_RATIO = 0.46
GLASS_YOUNG_MODULUS = 20.0e6
GLASS_POISSON_RATIO = 0.30
# Reduced Teflon--glass Hertz constants, converted to this API's
# ``ShearModulus``/``Poisson`` input convention.
CONTACT_MODULUS = 14_723_203.769140163
CONTACT_POISSON = 0.3068786808009423


def parse_args():
    parser = argparse.ArgumentParser(description="3D two-point MPM--LSDEM submerged-bed impact")
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=0.005)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--time", type=float, default=0.12)
    parser.add_argument("--save-interval", type=float, default=0.002)
    parser.add_argument("--drop-height", type=float, default=1.0)
    parser.add_argument("--ppc", type=int, default=1)
    parser.add_argument("--pressure-iterations", type=int, default=300)
    parser.add_argument("--device-memory", type=float, default=4.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples" / "mmpm" / "TwoPhaseLSDEMCoupling" / "OutputData" / "sphere_impact_section_5_2",
    )
    return parser.parse_args()


def cylinder_region(name, wall, top):
    radius = CONTAINER_RADIUS - wall

    def inside(position, particle_radius=0.0):
        radial_x = position[0] - 0.5 * DOMAIN[0]
        radial_y = position[1] - 0.5 * DOMAIN[1]
        return radial_x * radial_x + radial_y * radial_y <= radius * radius and wall <= position[2] <= top

    return {
        "Name": name,
        "Type": "UserDefined",
        "BoundingBoxPoint": [wall, wall, wall],
        "BoundingBoxSize": [DOMAIN[0] - 2.0 * wall, DOMAIN[1] - 2.0 * wall, top - wall],
        "RegionVolume": lambda: math.pi * radius * radius * (top - wall),
        "RegionFunction": inside,
    }


def cylindrical_boundaries(wall):
    boundaries = [
        {
            "BoundaryType": "SolidCell",
            "StartPoint": [0.0, 0.0, wall],
            "EndPoint": [DOMAIN[0], DOMAIN[1], wall],
            "Norm": [0.0, 0.0, -1.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, None, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [DOMAIN[0], DOMAIN[1], 2.0 * wall],
        },
    ]
    for side in range(CYLINDER_SIDES):
        angle = 2.0 * math.pi * side / CYLINDER_SIDES
        normal = [math.cos(angle), math.sin(angle), 0.0]
        boundaries.append(
            {
                "BoundaryType": "SolidPlaneCell",
                "StartPoint": [0.0, 0.0, wall],
                "EndPoint": list(DOMAIN),
                "Point": [
                    0.5 * DOMAIN[0] + CONTAINER_RADIUS * normal[0],
                    0.5 * DOMAIN[1] + CONTAINER_RADIUS * normal[1],
                    wall,
                ],
                "Norm": normal,
            }
        )
    return boundaries


def write_metrics(output, expected_fluid, expected_solid, args):
    mpm_files = sorted((output / "particles").glob("MPMParticle*.npz"))
    rigid_files = sorted((output / "particles").glob("LSDEMRigid*.npz"))
    if len(mpm_files) < 2 or len(rigid_files) < 2:
        raise RuntimeError("sphere-impact run produced fewer than two MPM or LSDEM snapshots")
    mpm_rows = []
    finite = True
    for file_name in mpm_files:
        with np.load(file_name) as data:
            active = data["active"] > 0
            phase = data["phase"]
            fluid = active & (phase == 2)
            solid = active & (phase == 1)
            position = data["position"]
            finite &= bool(
                np.isfinite(position[active]).all()
                and np.isfinite(data["fluid_velocity"][fluid]).all()
                and np.isfinite(data["solid_velocity"][solid]).all()
                and np.isfinite(data["pressure"][active]).all()
            )
            outside = np.any(
                (position[active] < -1.0e-10 * args.dx) | (position[active] > np.asarray(DOMAIN) + 1.0e-10 * args.dx),
                axis=1,
            )
            radial = np.linalg.norm(position[active, :2] - 0.5 * np.asarray(DOMAIN[:2]), axis=1)
            mpm_rows.append(
                [
                    float(data["t_current"]),
                    int(np.count_nonzero(fluid)),
                    int(np.count_nonzero(solid)),
                    int(np.count_nonzero(outside)),
                    int(np.count_nonzero(radial > CONTAINER_RADIUS + 1.0e-10 * args.dx)),
                    float(np.linalg.norm(data["fluid_velocity"][fluid], axis=1).max()),
                    float(np.linalg.norm(data["solid_velocity"][solid], axis=1).max()),
                ]
            )
    rigid_rows = []
    for file_name in rigid_files:
        with np.load(file_name) as data:
            center = data["mass_center"][0]
            velocity = data["velocity"][0]
            force = data["contact_force"][0]
            finite &= bool(np.isfinite(center).all() and np.isfinite(velocity).all() and np.isfinite(force).all())
            rigid_rows.append([float(data["t_current"]), *center, *velocity, float(np.linalg.norm(force))])
    mpm_rows = np.asarray(mpm_rows, dtype=np.float64)
    rigid_rows = np.asarray(rigid_rows, dtype=np.float64)
    impact_speed = math.sqrt(2.0 * 9.81 * args.drop_height)
    radius = 0.5 * SPHERE_DIAMETER
    initial_center_z = WATER_SURFACE + radius
    minimum_center_z = float(rigid_rows[:, 3].min())
    metrics = {
        "case": "Section 5.2 LSDEM sphere impact into a saturated granular bed",
        "drop_height_m": args.drop_height,
        "equivalent_impact_speed_mps": impact_speed,
        "container_diameter_m": 2.0 * CONTAINER_RADIUS,
        "cylindrical_wall_planes": CYLINDER_SIDES,
        "reference_particle_spacing_m": REFERENCE_PARTICLE_SPACING,
        "mpm_particle_spacing_m": args.dx / args.ppc,
        "teflon_young_modulus_pa": TEFLON_YOUNG_MODULUS,
        "teflon_poisson_ratio": TEFLON_POISSON_RATIO,
        "glass_young_modulus_pa": GLASS_YOUNG_MODULUS,
        "glass_poisson_ratio": GLASS_POISSON_RATIO,
        "snapshots_mpm": len(mpm_rows),
        "snapshots_lsdem": len(rigid_rows),
        "final_time_s": float(min(mpm_rows[-1, 0], rigid_rows[-1, 0])),
        "duration_complete": bool(
            math.isclose(mpm_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
            and math.isclose(rigid_rows[-1, 0], args.time, abs_tol=0.1 * args.dt)
        ),
        "expected_fluid_particles": expected_fluid,
        "expected_solid_particles": expected_solid,
        "particle_conservation": bool(
            np.all(mpm_rows[:, 1] == expected_fluid) and np.all(mpm_rows[:, 2] == expected_solid)
        ),
        "finite": finite,
        "maximum_mpm_particles_outside_domain": int(mpm_rows[:, 3].max()),
        "maximum_mpm_particles_outside_cylinder": int(mpm_rows[:, 4].max()),
        "maximum_fluid_speed_mps": float(mpm_rows[:, 5].max()),
        "maximum_solid_speed_mps": float(mpm_rows[:, 6].max()),
        "sphere_minimum_center_z_m": minimum_center_z,
        "sphere_water_entry_displacement_m": initial_center_z - minimum_center_z,
        "sphere_maximum_bed_penetration_m": max(0.0, BED_HEIGHT - (minimum_center_z - radius)),
        "sphere_final_vertical_velocity_mps": float(rigid_rows[-1, 6]),
        "maximum_coupling_force_n": float(rigid_rows[:, 7].max()),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["particle_conservation"]
        and metrics["maximum_mpm_particles_outside_domain"] == 0
        and metrics["maximum_mpm_particles_outside_cylinder"] == 0
        and metrics["sphere_water_entry_displacement_m"] >= 0.02
        and metrics["sphere_maximum_bed_penetration_m"] >= 0.002
        and metrics["sphere_minimum_center_z_m"] - radius >= 0.5 * args.dx
        and abs(metrics["sphere_final_vertical_velocity_mps"]) < 0.9 * impact_speed
        and metrics["maximum_coupling_force_n"] > 0.0
    )
    np.savetxt(
        output / "sphere_impact_trajectory.csv",
        rigid_rows,
        delimiter=",",
        header="time_s,center_x_m,center_y_m,center_z_m,velocity_x_mps,velocity_y_mps,velocity_z_mps,coupling_force_n",
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("Section 5.2 sphere-impact validation failed; inspect metrics.json")


@ti.kernel
def initialize_hydrostatic_state(particle_num: int, particle: ti.template()):
    k0 = 1.0 - ti.sin(26.0 * math.pi / 180.0)
    for p in range(particle_num):
        particle[p].pressure = 1000.0 * 9.81 * ti.max(WATER_SURFACE - particle[p].x[2], 0.0)
        if int(particle[p].phase) == 1:
            effective = (1.0 - 0.36) * (2500.0 - 1000.0) * 9.81 * ti.max(BED_HEIGHT - particle[p].x[2], 0.0)
            particle[p].stress = ti.Vector([-k0 * effective, -k0 * effective, -effective, 0.0, 0.0, 0.0])


def run(args):
    if min(args.dx, args.dt, args.time, args.save_interval, args.drop_height) <= 0.0 or args.ppc <= 0:
        raise ValueError("dx, dt, time, save interval, drop height and ppc must be positive")
    cells = np.asarray(DOMAIN) / args.dx
    if not np.allclose(cells, np.round(cells), atol=1.0e-12) or np.any(np.round(cells).astype(int) % 2):
        raise ValueError("dx must divide all domain lengths into even cell counts for two-level MGPCG")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    wall = args.dx
    point_spacing = args.dx / args.ppc
    cross_section = math.pi * (CONTAINER_RADIUS - wall) ** 2
    fluid_count = math.ceil(cross_section * (WATER_SURFACE - wall) / point_spacing**3)
    solid_count = math.ceil(cross_section * (BED_HEIGHT - wall) / point_spacing**3)
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
        domain=list(DOMAIN),
        coupling_scheme="MPDEM",
        particle_interaction=True,
        wall_interaction=False,
        gravity=[0.0, 0.0, -9.81],
        visualize=True,
    )
    dempm.mpm.set_configuration(
        background_damping=0.02,
        alphaPIC=0.05,
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
            "bottom_smoothing": 10,
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
            "max_rigid_body_number": 1,
            "max_rigid_template_number": 1,
            "levelset_grid_number": 300000,
            "surface_node_number": 5000,
            "max_plane_number": 0,
            "body_coordination_number": 0,
            "wall_coordination_number": 0,
            "verlet_distance_multiplier": [0.15, 0.10],
            "point_coordination_number": [4, 2],
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
        {"body_coordination_number": 1, "wall_coordination_number": 0, "compaction_ratio": [0.1, 0.1]}
    )

    dempm.dem.add_attribute(
        materialID=0,
        attribute={
            "Density": 2200.0,
            "YoungModulus": TEFLON_YOUNG_MODULUS,
            "PoissonRatio": TEFLON_POISSON_RATIO,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    dempm.dem.add_template(
        {
            "Name": "teflon_sphere",
            "Object": polyhedron(file=str(ROOT / "assets" / "mesh" / "LSDEM" / "sphere.stl")).grids(
                space=0.05, extent=10
            ),
            "WriteFile": False,
        }
    )
    dempm.dem.create_body(
        {
            "GenerateType": "Create",
            "BodyType": "RigidBody",
            "Template": [
                {
                    "Name": "teflon_sphere",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [0.10, 0.10, WATER_SURFACE + 0.5 * SPHERE_DIAMETER],
                    "Radius": 0.5 * SPHERE_DIAMETER,
                    "BodyOrientation": "constant",
                    "InitialVelocity": [0.0, 0.0, -math.sqrt(2.0 * 9.81 * args.drop_height)],
                    "FixMotion": ["Free", "Free", "Free"],
                }
            ],
        }
    )
    dempm.dem.choose_contact_model(None, None)

    dempm.mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "SolidDensity": 2500.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.36,
            "MaximumPorosity": 0.52,
            "FluidBulkModulus": 2.2e9,
            "Permeability": 1.6e-10,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 5.0e-4,
            "DragModel": "Ergun",
            "YoungModulus": GLASS_YOUNG_MODULUS,
            "PoissonRatio": GLASS_POISSON_RATIO,
            "Cohesion": 0.0,
            "Friction": 26.0,
            "Dilation": 0.0,
        },
    )
    dempm.mpm.add_element({"ElementType": "R8N3D", "ElementSize": [args.dx] * 3})
    dempm.mpm.add_region(
        [cylinder_region("granular_bed", wall, BED_HEIGHT), cylinder_region("water", wall, WATER_SURFACE)]
    )
    # Solid first is intentional: ordinary MPDEM contact uses the coupled
    # prefix, while all following fluid points are handled exclusively by IBM.
    dempm.mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "granular_bed",
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
    initialize_hydrostatic_state(int(dempm.mpm.scene.particleNum[0]), dempm.mpm.scene.particle)
    initial_phase = dempm.mpm.scene.particle.phase.to_numpy()[: int(dempm.mpm.scene.particleNum[0])]
    expected_fluid = int(np.count_nonzero(initial_phase == 2))
    expected_solid = int(np.count_nonzero(initial_phase == 1))

    dempm.mpm.add_boundary_condition(cylindrical_boundaries(wall))

    dempm.mpm.select_save_data(particle=True, grid=True, object=False)
    dempm.dem.select_save_data(surface=True, grid=True, bounding=True)
    dempm.choose_contact_model("Hertz Mindlin Model", None)
    dempm.add_property(
        DEMmaterial=0,
        MPMmaterial=1,
        property={
            "ShearModulus": CONTACT_MODULUS,
            "Poisson": CONTACT_POISSON,
            "Friction": math.tan(math.radians(26.0)),
            "Restitution": 0.1,
        },
    )
    dempm.select_save_data(particle_particle_contact=True)
    dempm.run()
    write_metrics(output, expected_fluid, expected_solid, args)
    if not args.no_post:
        dempm.postprocessing(scheme="LSDEM")


if __name__ == "__main__":
    run(parse_args())
