"""Self-contained IGA--MPM CPT with explicit DEM contact or monolithic IPC.

The IPC route uses the physical axisymmetric meridian, a refined NURBS pile,
and DP soil.  The legacy explicit route remains a thin 3-D capability check.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


DOMAIN = (0.6, 2.508)

SOIL_ORIGIN = (0.0, 0.0)

SOIL_SIZE = (0.6, 1.5)

GRID_SIZE = 0.006

PARTICLES_PER_CELL = 2

GRAVITY = (0.0, -9.8)

BACKGROUND_DAMPING = 0.05

ALPHA_PIC = 0.05

MAPPING = "USF"

SHAPE_FUNCTION = "GIMP"

STABILIZATION = "B-Bar Method"

SURFACE_ORIGIN = (0.0, SOIL_SIZE[1] - 0.5 * GRID_SIZE)

SURFACE_SIZE = (SOIL_SIZE[0], 0.5 * GRID_SIZE)

MAX_MATERIAL_NUMBER = 1

MAX_PARTICLE_NUMBER = 800_000

MAX_VELOCITY_CONSTRAINT = 134_474

MAX_PARTICLE_TRACTION_CONSTRAINT = 134_474

TIMESTEP = 1.0e-5

SIMULATION_TIME = 10.0

SAVE_INTERVAL = 0.2

PILE_SPEED = 0.1

SURFACE_PRESSURE = 150.0e3

INITIAL_STRESS = (-75.0e3, -150.0e3, -75.0e3, 0.0, 0.0, 0.0)

PILE_PROFILE = (
    (0.001, 1.5000),
    (0.018, 1.5312),
    (0.018, 2.5000),
    (0.001, 2.5000),
)

SOIL_MATERIAL = {
    "MaterialID": 1,
    "Density": 1600.0,
    "YoungModulus": 60.0e6,
    "PoissonRatio": 0.30,
    "e0": 0.62,
    "e_Tao": 0.90,
    "lambda_c": 0.119,
    "ksi": 0.23,
    "nd": 1.70,
    "nf": 2.68,
    "fai_c": 30.0,
    "Cohesion": 3000.0,
}

MPM_DEM_CONTACT = {
    "stiffness": (1.0e5, 1.0e5),
    "friction": 0.0,
}

FEMPM_EXPLICIT_CONTACT = {
    "NormalStiffness": 1.0e7,
    "TangentialStiffness": 1.0e7,
    "Friction": 0.0,
    "NormalViscousDamping": 0.05,
    "TangentialViscousDamping": 0.05,
}

IGAMPM_EXPLICIT_CONTACT = {
    "NormalStiffness": 1.0e7,
    "TangentialStiffness": 1.0e7,
    "StaticFriction": 0.0,
    "DynamicFriction": 0.0,
    "NormalViscousDamping": 0.05,
    "TangentialViscousDamping": 0.05,
}

IPC_CONTACT = {
    # IPC's activation distance is dmin + dhat. Keep it within half a
    # particle spacing so the initially separated pile does not preload the
    # soil through the barrier before prescribed penetration starts.
    "dhat": (0.5 / PARTICLES_PER_CELL - 0.1) * GRID_SIZE,
    "dmin": 0.1 * GRID_SIZE,
    "kappa": 60.0e6,
    "friction_coefficient": 0.0,
    "epsv": 1.0e-4,
}

PENETRATOR_MATERIAL = {
    "density": 7850.0,
    "young_modulus": 200.0e9,
    "poisson_ratio": 0.30,
}

COUPLED_SLICE_THICKNESS = 8.0 * GRID_SIZE

COUPLED_DOMAIN = (DOMAIN[0], COUPLED_SLICE_THICKNESS, DOMAIN[1])

COUPLED_SOIL_SIZE = (SOIL_SIZE[0], COUPLED_SLICE_THICKNESS, SOIL_SIZE[1])

COUPLED_GRAVITY = (GRAVITY[0], 0.0, GRAVITY[1])

COUPLED_INITIAL_STRESS = (
    INITIAL_STRESS[0],
    INITIAL_STRESS[2],
    INITIAL_STRESS[1],
    0.0,
    0.0,
    0.0,
)

COUPLED_DP_MATERIAL = {
    "density": SOIL_MATERIAL["Density"],
    "young_modulus": SOIL_MATERIAL["YoungModulus"],
    "poisson_ratio": SOIL_MATERIAL["PoissonRatio"],
    "friction_angle": SOIL_MATERIAL["fai_c"],
    "dilation_angle": SOIL_MATERIAL["fai_c"],
    "cohesion": SOIL_MATERIAL["Cohesion"],
}


def validate_run_parameters(dt, simulation_time, save_interval, resolution_scale):
    values = {
        "dt": dt,
        "simulation_time": simulation_time,
        "save_interval": save_interval,
        "resolution_scale": resolution_scale,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def realized_grid_size(resolution_scale):
    return GRID_SIZE * float(resolution_scale)


def native_particle_capacity(grid_size):
    cells = np.ceil(np.asarray(COUPLED_SOIL_SIZE) / float(grid_size)).astype(int)
    particle_count = int(np.prod(cells)) * PARTICLES_PER_CELL**3
    ten_percent_headroom = (particle_count + 9) // 10
    return max(10_000, particle_count + ten_percent_headroom)


def dp_native_material():
    material = COUPLED_DP_MATERIAL
    return {
        "MaterialID": 1,
        "Density": material["density"],
        "YoungModulus": material["young_modulus"],
        "PoissonRatio": material["poisson_ratio"],
        "Friction": material["friction_angle"],
        "Dilation": material["dilation_angle"],
        "Cohesion": material["cohesion"],
        "Tensile": 0.0,
    }


def dp_direct_material(dilation_angle=None, state_dependent=False):
    material = COUPLED_DP_MATERIAL
    if state_dependent:
        return {
            "model": "StateDependentDruckerPrager",
            "density": material["density"],
            "young_modulus": material["young_modulus"],
            "poisson_ratio": material["poisson_ratio"],
            **{key: SOIL_MATERIAL[key] for key in ("e0", "e_Tao", "lambda_c", "ksi", "nd", "nf", "fai_c", "Cohesion")},
            "dpType": "MiddleCircumscribed",
        }
    return {
        "model": "DruckerPrager",
        "density": material["density"],
        "young_modulus": material["young_modulus"],
        "poisson_ratio": material["poisson_ratio"],
        "FrictionAngle": material["friction_angle"],
        "DilationAngle": material["dilation_angle"] if dilation_angle is None else dilation_angle,
        "Cohesion": material["cohesion"],
        "dpType": "Circumscribed",
    }


def configure_native_explicit_mpm(mpm, output_path, dt, simulation_time, save_interval, resolution_scale=1.0):
    """Configure the native explicit 3-D thin-slice CPT soil."""
    validate_run_parameters(dt, simulation_time, save_interval, resolution_scale)
    grid_size = realized_grid_size(resolution_scale)
    surface_band = 0.5 * grid_size
    particle_capacity = native_particle_capacity(grid_size)

    mpm.set_configuration(
        domain=list(COUPLED_DOMAIN),
        gravity=list(COUPLED_GRAVITY),
        background_damping=BACKGROUND_DAMPING,
        alphaPIC=ALPHA_PIC,
        mapping=MAPPING,
        shape_function=SHAPE_FUNCTION,
        stabilize=STABILIZATION,
        configuration="ULMPM",
        solver_type="Explicit",
        material_type="Solid",
        visualize=False,
        log=True,
    )
    mpm.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": simulation_time,
            "SaveInterval": save_interval,
            "SavePath": str(output_path),
        },
        log=True,
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": particle_capacity,
            "max_constraint_number": {
                "max_velocity_constraint": 400_000,
                "max_reflection_constraint": 200_000,
                "max_particle_traction_constraint": 400_000,
            },
        },
        log=True,
    )
    mpm.add_material(model="DruckerPrager", material=dp_native_material())
    mpm.add_element(
        {
            "ElementType": "R8N3D",
            "ElementSize": [grid_size] * 3,
        }
    )
    mpm.add_region(
        [
            {
                "Name": "cpt_soil",
                "Type": "Rectangle",
                "BoundingBoxPoint": [0.0, 0.0, 0.0],
                "BoundingBoxSize": list(COUPLED_SOIL_SIZE),
            },
            {
                "Name": "cpt_surface",
                "Type": "Rectangle",
                "BoundingBoxPoint": [
                    0.0,
                    0.0,
                    COUPLED_SOIL_SIZE[2] - surface_band,
                ],
                "BoundingBoxSize": [
                    COUPLED_SOIL_SIZE[0],
                    COUPLED_SOIL_SIZE[1],
                    surface_band,
                ],
            },
        ]
    )
    mpm.add_body(
        {
            "Template": {
                "RegionName": "cpt_soil",
                "nParticlesPerCell": PARTICLES_PER_CELL,
                "BodyID": 0,
                "MaterialID": 1,
                "ParticleStress": {"InternalStress": list(COUPLED_INITIAL_STRESS)},
                "Traction": [
                    {
                        "Pressure": [0.0, 0.0, -SURFACE_PRESSURE],
                        "RegionName": "cpt_surface",
                    }
                ],
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    x_max, y_max, z_max = COUPLED_DOMAIN
    mpm.add_boundary_condition(
        [
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, 0.0, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [x_max, y_max, 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.0, y_max, z_max],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [x_max, 0.0, 0.0],
                "EndPoint": [x_max, y_max, z_max],
            },
            {
                "BoundaryType": "ReflectionConstraint",
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [x_max, 0.0, z_max],
                "Norm": [0.0, -1.0, 0.0],
            },
            {
                "BoundaryType": "ReflectionConstraint",
                "StartPoint": [0.0, y_max, 0.0],
                "EndPoint": [x_max, y_max, z_max],
                "Norm": [0.0, 1.0, 0.0],
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    return grid_size


def axisymmetric_particle_count(grid_size, particles_per_cell=PARTICLES_PER_CELL):
    cells = np.floor(
        np.asarray(SOIL_SIZE) / float(grid_size)
        + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.asarray(SOIL_SIZE) / float(grid_size))
    ).astype(int)
    return int(np.prod(cells)) * int(particles_per_cell) ** 2


def axisymmetric_surface_pressure_particles(points, grid_size):
    """Return material-point IDs and reference annular areas for top pressure.

    Each point represents a reference annulus of radial width dx/ppc. The
    forces keep that reference area and follow the selected particle IDs;
    current shape functions transfer them to the grid at every step.
    """
    points = np.asarray(points, dtype=np.float64)
    particle_spacing = float(grid_size) / PARTICLES_PER_CELL
    top_height = SOIL_ORIGIN[1] + SOIL_SIZE[1] - 0.5 * particle_spacing
    particle_ids = np.flatnonzero(np.isclose(points[:, 1], top_height, rtol=0.0, atol=1.0e-12)).astype(np.int32)
    surface_area = 2.0 * np.pi * points[particle_ids, 0] * particle_spacing
    return particle_ids, surface_area


def axisymmetric_initial_deformation_gradient(points=None):
    """Return the preload, including self weight when particle points are given.

    The no-argument form retains the historical uniform Kirchhoff preload.
    The CPT drivers pass their points to initialize the actual Cauchy stress
    profile, with the lateral self-weight coefficient of the saved reference.
    """
    material = COUPLED_DP_MATERIAL
    young, poisson = material["young_modulus"], material["poisson_ratio"]
    stress = np.asarray(INITIAL_STRESS[:3], dtype=np.float64)
    if points is None:
        log_stretch = ((1.0 + poisson) * stress - poisson * np.sum(stress)) / young
        return np.diag(np.exp(log_stretch))
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("axisymmetric preload points must be a finite (N, 2) array")
    depth = np.maximum(0.0, SOIL_ORIGIN[1] + SOIL_SIZE[1] - points[:, 1])
    overburden = material["density"] * abs(GRAVITY[1]) * depth
    k0 = poisson / (1.0 - poisson)
    stress = stress[None, :] - overburden[:, None] * np.array([k0, 1.0, k0])
    log_stretch = ((1.0 + poisson) * stress - poisson * np.sum(stress, axis=1)[:, None]) / young
    # Hencky produces Kirchhoff stress; solve J=exp(J tr(epsilon_sigma))
    # so the requested preload is Cauchy stress rather than tau=J sigma.
    jacobian = np.ones(points.shape[0])
    for _ in range(12):
        jacobian = np.exp(jacobian * np.sum(log_stretch, axis=1))
    deformation = np.zeros((points.shape[0], 3, 3))
    for axis in range(3):
        deformation[:, axis, axis] = np.exp(jacobian * log_stretch[:, axis])
    return deformation


def _direct_axisymmetric_boundaries(grid_size):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    grid_num = np.ceil(np.asarray(DOMAIN) / float(grid_size)).astype(np.int32) + 1
    nr, nz = map(int, grid_num)
    nodes = np.arange(nr * nz, dtype=np.int32).reshape(nz, nr)
    bottom = nodes[0, :]
    radial_fixed = np.unique(np.concatenate((bottom, nodes[:, 0], nodes[:, -1])))
    dirichlet = DirichletBoundary()
    dirichlet.append(
        [list(2 * radial_fixed), list(2 * bottom + 1)],
        [0.0] * (radial_fixed.size + bottom.size),
    )

    return dirichlet


def configure_direct_axisymmetric_mpm(
    mpm,
    output_path,
    dt,
    simulation_time,
    save_interval,
    resolution_scale=1.0,
    dilation_angle=None,
    state_dependent=False,
):
    """Configure the Direct implicit axisymmetric DP soil for IPC coupling."""
    validate_run_parameters(dt, simulation_time, save_interval, resolution_scale)
    grid_size = realized_grid_size(resolution_scale)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=list(DOMAIN),
        axisymmetric=True,
        axis_offset=0.0,
        gravity=list(GRAVITY),
        background_damping=BACKGROUND_DAMPING,
        alphaPIC=ALPHA_PIC,
        visualize=True,
        log=True,
    )
    body = mpm.create_body()
    body.add_rectangle(
        SOIL_ORIGIN,
        SOIL_SIZE,
        grid_size,
        PARTICLES_PER_CELL,
        init_v=[0.0, 0.0],
        name="cpt_soil",
        grid_size=grid_size,
        xmin=[0.0, 0.0],
        xmax=list(DOMAIN),
    )
    particle_count = int(body.particle_counter)
    expected_count = axisymmetric_particle_count(grid_size)
    if particle_count != expected_count:
        raise RuntimeError(f"axisymmetric CPT generated {particle_count} particles; expected {expected_count}")
    mpm.add_body(body)
    mpm.memory_allocate({"max_particle_number": particle_count}, log=False)
    mpm.add_boundary_condition(dirichlet=_direct_axisymmetric_boundaries(grid_size))
    mpm.add_material(**dp_direct_material(dilation_angle, state_dependent))
    mpm.add_element({"ElementSize": grid_size, "ShapeFunction": "QuadBSpline"})
    step_count = int(math.ceil(simulation_time / dt))
    output_interval = max(1, int(round(save_interval / dt)))
    mpm.set_solver(
        {
            "dt": dt,
            "step": step_count,
            "interval": output_interval,
            "path": str(output_path),
            "newmark": [1.0, 0.5, 1.0],
            "residual": 5.0e-4,
            "max_iters": 35,
            "scale": 1.0,
            "ccd": True,
            "project_pd": True,
        }
    )
    mpm.add_engine()
    particle_ids, surface_area = axisymmetric_surface_pressure_particles(body.bodies["cpt_soil"]["points"], grid_size)
    mpm.enginer.init_particle_pressure(particle_ids, [0.0, -SURFACE_PRESSURE], surface_area)
    return grid_size, particle_count


def configure_iga_axisymmetric_penetrator(iga, dt, step_count, output_interval, output_path):
    """Configure a refined axisymmetric NURBS representation of the CPT pile."""
    from src.iga import DirichletBoundary, Primitives, Rectangle

    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    pile = Rectangle()
    pile.set_parameters(
        start_point=profile[0],
        size=[profile[1, 0] - profile[0, 0], profile[2, 1] - profile[0, 1]],
    )
    pile.generate_knot_u(degree=2, num_ctrlpts=3)
    pile.generate_knot_v(degree=2, num_ctrlpts=34)
    pile.generate_ctrlpts()
    pile.generate_weights()
    normalized_height = (pile.control_points[:, 1] - profile[0, 1]) / (profile[2, 1] - profile[0, 1])
    bottom_offset = (pile.control_points[:, 0] - profile[0, 0]) * (
        (profile[1, 1] - profile[0, 1]) / (profile[1, 0] - profile[0, 0])
    )
    pile.control_points[:, 1] += (1.0 - normalized_height) * bottom_offset

    primitives = Primitives()
    primitives.append(pile, "cpt_penetrator", init_v=[0.0, -PILE_SPEED])
    primitives.finialize()
    iga.set_configuration(
        dimension=2,
        solver_type="Implicit",
        axisymmetric=True,
        axis_offset=0.0,
    )
    iga.add_primitives(primitives)
    all_control_points = np.arange(pile.control_points.shape[0], dtype=np.int32)
    boundary = DirichletBoundary()
    boundary.append_velocity(
        [list(2 * all_control_points), list(2 * all_control_points + 1)],
        [0.0] * all_control_points.size + [-PILE_SPEED] * all_control_points.size,
    )
    iga.add_boundary_condition(dirichlet=boundary)
    iga.add_element(degree=[2, 2])
    iga.add_material(
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
        gravity=list(GRAVITY),
    )
    iga.set_solver(
        dt=dt,
        step=step_count,
        interval=output_interval,
        path=str(output_path),
        newmark=[1.0, 0.5, 1.0],
        residual=5.0e-4,
        max_iters=35,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
    )
    top = np.flatnonzero(np.isclose(pile.control_points[:, 1], profile[2, 1]))
    return top


def configure_iga_penetrator(iga, implicit, dt, step_count, output_interval, output_path):
    from src.iga import Cube, DirichletBoundary, Primitives

    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    x_min, z_min = profile[0]
    x_max = float(np.max(profile[:, 0]))
    z_max = float(np.max(profile[:, 1]))
    iga.set_configuration(dimension=3, solver_type="Implicit" if implicit else "Explicit")
    pile = Cube()
    pile.set_parameters(
        start_point=[x_min, 0.0, z_min],
        size=[x_max - x_min, COUPLED_SOIL_SIZE[1], z_max - z_min],
    )
    pile.generate_knot_u(degree=2, num_ctrlpts=3)
    pile.generate_knot_v(degree=2, num_ctrlpts=3)
    pile.generate_knot_w(degree=2, num_ctrlpts=4)
    pile.generate_ctrlpts()
    pile.generate_weights()
    normalized_height = (pile.control_points[:, 2] - z_min) / (z_max - z_min)
    bottom_offset = (
        (pile.control_points[:, 0] - x_min) * (profile[1, 1] - profile[0, 1]) / (profile[1, 0] - profile[0, 0])
    )
    pile.control_points[:, 2] += (1.0 - normalized_height) * bottom_offset

    primitives = Primitives()
    primitives.append(pile, "cpt_penetrator", init_v=[0.0, 0.0, -PILE_SPEED])
    primitives.finialize()
    iga.add_primitives(primitives)
    all_control_points = np.arange(pile.control_points.shape[0], dtype=np.int32)
    fixed_dofs = [list(3 * all_control_points), list(3 * all_control_points + 1)]
    boundary = DirichletBoundary()
    boundary.append(fixed_dofs, [0.0] * (2 * all_control_points.size))
    iga.add_boundary_condition(dirichlet=boundary)
    iga.add_element(degree=[2, 2, 2])
    iga.add_material(
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
        gravity=list(COUPLED_GRAVITY),
    )
    iga.set_solver(
        dt=dt,
        step=step_count,
        interval=output_interval,
        path=str(output_path),
        newmark=[1.0, 0.5, 1.0],
        residual=5.0e-4,
        max_iters=35,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
    )
    return pile.control_points.shape[0]


def explicit_contact_parameters(family):
    if str(family).lower() == "fempm":
        return dict(FEMPM_EXPLICIT_CONTACT)
    if str(family).lower() == "igampm":
        return dict(IGAMPM_EXPLICIT_CONTACT)
    raise ValueError("family must be FEMPM or IGAMPM")


def ipc_contact_parameters(grid_size):
    scale = float(grid_size) / GRID_SIZE
    return {
        "dhat": scale * IPC_CONTACT["dhat"],
        "dmin": scale * IPC_CONTACT["dmin"],
        "kappa": IPC_CONTACT["kappa"],
        "friction_coefficient": IPC_CONTACT["friction_coefficient"],
        "epsv": IPC_CONTACT["epsv"],
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact", choices=("explicit", "ipc"), default="explicit")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument(
        "--dilation-angle", type=float, help="IPC DP dilation angle in degrees; default equals friction"
    )
    parser.add_argument("--inexact-newton", action="store_true", help="adapt linear tolerance for nonassociated IPC DP")
    parser.add_argument(
        "--state-dependent", action="store_true", help="use finite-strain state-dependent DP for IPC soil"
    )
    parser.add_argument("--dt", type=float)
    parser.add_argument("--time", type=float)
    parser.add_argument("--save-interval", type=float)
    parser.add_argument("--resolution-scale", type=float)
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData"))
    parser.add_argument("--resume", type=Path, help="IPC latest_state.npz to restore")
    arguments = parser.parse_args()
    implicit = arguments.contact == "ipc"
    if arguments.resume is not None and not implicit:
        parser.error("--resume requires --contact ipc")
    if arguments.dilation_angle is not None and not implicit:
        parser.error("--dilation-angle is supported by the IPC route")
    if arguments.inexact_newton and not implicit:
        parser.error("--inexact-newton requires --contact ipc")
    if arguments.state_dependent and (not implicit or arguments.dilation_angle is not None):
        parser.error("--state-dependent requires --contact ipc and evolves dilation without --dilation-angle")
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

    validate_run_parameters(
        arguments.dt,
        arguments.time,
        arguments.save_interval,
        arguments.resolution_scale,
    )
    implicit = arguments.contact == "ipc"
    output_path = Path(arguments.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing_run = (output_path / "step_diagnostics.jsonl").exists() or (output_path / "cpt_summary.json").exists()
    if existing_run and (arguments.resume is None or arguments.resume.resolve().parent != output_path.resolve()):
        raise FileExistsError(f"refusing to overwrite existing run: {output_path}")
    if arguments.resume is not None:
        previous = json.loads((arguments.resume.parent / "parameters.json").read_text())
        for key in ("contact", "default_fp", "dilation_angle", "resolution_scale", "save_interval"):
            if previous[key] != getattr(arguments, key):
                raise ValueError(f"checkpoint configuration mismatch: {key}")
        if previous.get("state_dependent", False) != arguments.state_dependent:
            raise ValueError("checkpoint configuration mismatch: state_dependent")
        if previous.get("shape_function", "Linear") != "QuadBSpline":
            raise ValueError("checkpoint configuration mismatch: shape_function; restart QuadBSpline from frame 0")
    if not existing_run:
        (output_path / "parameters.json").write_text(
            json.dumps(
                {**vars(arguments), "shape_function": "QuadBSpline" if implicit else SHAPE_FUNCTION},
                default=str,
                indent=2,
            )
            + "\n"
        )
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
            dilation_angle=arguments.dilation_angle,
            state_dependent=arguments.state_dependent,
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
            compact_contact_slots=True,
            # Retain all-particle broad phase, reserving blocks only for active contacts.
            barrier_nnz=min(4096, 2 * particle_count) * (3 + 9) ** 2,
            friction_nnz=1,
            # Bound Newton iterations for associated PCG or nonassociated BiCGSTAB.
            monolithic_max_iterations=100,
            monolithic_tolerance=5.0e-4,
            monolithic_linear_solver_tolerance=1.0e-8,
            monolithic_linear_solver_relative_tolerance=1.0e-7,
            monolithic_linear_solver_max_iters=30_000,
            monolithic_inexact_newton=arguments.inexact_newton,
            project_pd=True,
            # kappa is a pressure in this case, as in FEM--MPM; normalize
            # the squared-distance barrier before multiplying by area.
            use_physical_barrier=True,
            enable_step_retry=True,
            step_retry_max_retries=4,
            step_retry_reduction=0.5,
            step_retry_minimum_timestep=arguments.dt / 16,
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
        axisymmetric_initial_deformation_gradient(engine.mpm.particle.x.to_numpy()[:particle_count])
    )
    history = []
    next_save = [arguments.save_interval]
    started = time.monotonic()

    state = {
        "particle_position": engine.mpm.particle.x,
        "particle_velocity": engine.mpm.particle.v,
        "particle_acceleration": engine.mpm.particle.a,
        "deformation": engine.mpm.F0,
        "plastic_inverse": engine.mpm.material.plastic_deformation_inverse,
        "equivalent_plastic_strain": engine.mpm.material.equivalent_plastic_strain,
        "volumetric_plastic_strain": engine.mpm.material.volumetric_plastic_strain,
        "grid_velocity": engine.mpm.grid.v,
        "grid_acceleration": engine.mpm.grid.a,
        "pile_position": engine.iga.patch.control_points,
        "pile_velocity": engine.iga.patch.velocitys,
        "pile_acceleration": engine.iga.patch.accelerations,
    }
    if arguments.state_dependent:
        state.update(
            void_ratio=engine.mpm.material.void_ratio,
            committed_jacobian=engine.mpm.material.committed_jacobian,
            state_pressure=engine.mpm.material.state_pressure,
        )
    engine._initialize_implicit_ipc_state()
    if arguments.resume is not None:
        with np.load(arguments.resume, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            output_counts = saved["output_counts"].copy() if "output_counts" in saved else None
            if not 0 <= metadata["time"] < arguments.time:
                raise ValueError("checkpoint time must precede the requested end time")
            for name, field in state.items():
                values = saved[name]
                if values.shape != field.to_numpy().shape or not np.all(np.isfinite(values)):
                    raise ValueError(f"invalid checkpoint field: {name}")
            for name in ("deformation", "plastic_inverse"):
                if np.any(np.linalg.det(saved[name]) <= 0.0):
                    raise ValueError(f"checkpoint requires positive determinants: {name}")
            if arguments.state_dependent:
                if (
                    np.any(saved["committed_jacobian"] <= 0.0)
                    or np.any(saved["state_pressure"] < 1000.0)
                    or np.any((saved["void_ratio"] < 0.1) | (saved["void_ratio"] > 1.5))
                ):
                    raise ValueError("invalid state-dependent DP checkpoint history")
            for name, field in state.items():
                field.from_numpy(saved[name])
            if "history" in saved:
                history = saved["history"].tolist()
        # ULMPM rebuilds nodal mass/momentum from the restored particles. A
        # legacy checkpoint has no grid mass mask, so discard all old nodal
        # accumulators before the first P2G instead of retaining stale values.
        engine.mpm.grid.m.fill(0.0)
        engine.mpm.grid.v.fill(0.0)
        engine.mpm.grid.a.fill(0.0)
        engine.time = float(metadata["time"])
        engine.implicit_step_index = int(metadata["step"])
        for child in (engine.iga, engine.mpm):
            child.time, child.step_count = engine.time, engine.implicit_step_index
        next_save[0] = (math.floor((engine.time + 1e-12) / arguments.save_interval) + 1) * arguments.save_interval
        if existing_run:
            frame_count = math.ceil((engine.time - 1e-12) / arguments.save_interval) + 1
            if output_counts is not None:
                if output_counts.shape != (2,) or output_counts[0] != output_counts[1]:
                    raise ValueError("checkpoint IGA/MPM frame counts do not match")
                frame_count = int(output_counts[0])
            if len(list((output_path / "vtks").glob("GraphicMPMParticle*.vtu"))) != frame_count:
                raise ValueError("saved frames do not match checkpoint time; resume into an empty output directory")
            engine.iga.output_count = engine.mpm.output_count = frame_count
            mpm.sims.current_print = frame_count
            coupling._last_implicit_recorded_step = engine.implicit_step_index
            diagnostics_path = output_path / "step_diagnostics.jsonl"
            lines = diagnostics_path.read_text().splitlines(keepends=True)
            retained = [line for line in lines if json.loads(line)["step"] <= engine.implicit_step_index]
            if retained != lines:
                diagnostics_path.replace(output_path / "step_diagnostics_before_resume.jsonl")
                diagnostics_path.write_text("".join(retained))
        print(f"Resuming CPT at t={engine.time:.9g}, step={engine.implicit_step_index}", flush=True)

    def checkpoint():
        arrays = {name: field.to_numpy() for name, field in state.items()}
        arrays["metadata"] = np.array(json.dumps(engine.diagnostics_snapshot()))
        arrays["history"] = np.asarray(history, dtype=np.float64).reshape(-1, 5)
        arrays["output_counts"] = np.array([engine.iga.output_count, engine.mpm.output_count], dtype=np.int64)
        temporary = output_path / "latest_state.tmp.npz"
        np.savez(temporary, **arrays)
        temporary.replace(output_path / "latest_state.npz")

    def sample_cpt(coupled_engine):
        if not coupled_engine.last_step_record.get("converged", False):
            raise RuntimeError("an unconverged CPT step was accepted")
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
        snapshot = coupled_engine.diagnostics_snapshot()
        snapshot["last_step"] = coupled_engine.last_step_record
        snapshot["elapsed_seconds"] = time.monotonic() - started
        with (output_path / "step_diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        if coupled_engine.time >= next_save[0] - 1e-12:
            coupling._record_implicit_frame(coupled_engine)
            checkpoint()
            while next_save[0] <= coupled_engine.time + 1e-12:
                next_save[0] += arguments.save_interval

    if coupling._last_implicit_recorded_step != engine.implicit_step_index:
        coupling._record_implicit_frame(engine)
    if not existing_run:
        checkpoint()
    while engine.time < arguments.time - 1e-12:
        engine._set_implicit_timestep(min(arguments.dt, arguments.time - engine.time, next_save[0] - engine.time))
        result = coupling.run(steps=1, verbose=False, record=False, postprocessing=(sample_cpt,))
    if coupling._last_implicit_recorded_step != engine.implicit_step_index:
        coupling._record_implicit_frame(engine)
    checkpoint()
    final_control_points = engine.iga.patch.control_points.to_numpy()
    final_top = float(np.max(final_control_points[top_control_points, 1]))
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "cpt_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "penetration", "pile_tip", "pile_top", "active_contacts"))
        writer.writerows(history)
    summary = {
        "axisymmetric": True,
        "shape_function": "QuadBSpline",
        "grid_size": grid_size,
        "particle_count": particle_count,
        "friction_angle_deg": math.degrees(engine.mpm.material.friction_angle),
        "dilation_angle_deg": math.degrees(engine.mpm.material.dilation_angle),
        "inexact_newton": engine.monolithic_inexact_newton,
        "penetration_time": arguments.time,
        "target_penetration": 0.1 * arguments.time,
        "completed_time": float(engine.time),
        "final_pile_tip": float(np.min(final_control_points[:, 1])),
        "final_pile_top": final_top,
        "fully_inserted": final_top <= 1.5 + 1.0e-6,
        "maximum_active_contacts": max((row[4] for row in history), default=0),
        "history_start_time": history[0][0] if history else None,
        "converged": bool(result["converged"]),
    }
    if arguments.state_dependent:
        summary.pop("friction_angle_deg")
        summary.pop("dilation_angle_deg")
        void_ratio = engine.mpm.material.void_ratio.to_numpy()
        summary.update(
            soil_model="StateDependentDruckerPrager",
            critical_friction_angle_deg=math.degrees(engine.mpm.material.friction_angle),
            void_ratio_range=[float(void_ratio.min()), float(void_ratio.max())],
        )
    (output_path / "cpt_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not math.isclose(engine.time, arguments.time, abs_tol=1e-10, rel_tol=0):
        raise RuntimeError("CPT did not reach the requested physical end time")
    if arguments.time >= 10.0 and not summary["fully_inserted"]:
        raise RuntimeError(f"IGA pile stopped above the soil surface: top={final_top:.6e} m")


if __name__ == "__main__":
    main()
