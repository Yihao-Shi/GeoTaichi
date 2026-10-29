"""Self-contained FEM--MPM CPT with explicit DEM contact or monolithic IPC.

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


def dp_direct_material(dilation_angle=None):
    material = COUPLED_DP_MATERIAL
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

    ratios = np.asarray(DOMAIN) / float(grid_size)
    grid_num = np.floor(ratios + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(ratios))).astype(np.int32) + 1
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
    mpm.add_material(**dp_direct_material(dilation_angle))
    mpm.add_element({"ElementSize": grid_size, "ShapeFunction": "Linear"})
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


def penetrator_vertices():
    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    front = np.column_stack((profile[:, 0], np.zeros(4), profile[:, 1]))
    back = front.copy()
    back[:, 1] = COUPLED_SOIL_SIZE[1]
    return np.vstack((front, back))


def penetrator_tetrahedra():
    # Five positive-volume tetrahedra forming the extruded standard profile.
    return np.asarray(
        [
            [0, 3, 1, 4],
            [1, 3, 2, 6],
            [1, 3, 6, 4],
            [1, 5, 4, 6],
            [3, 6, 4, 7],
        ],
        dtype=np.int32,
    )


def configure_fem_penetrator(fem, implicit):
    from src.fem import DirichletBoundary, FEMMesh

    points = penetrator_vertices()
    fem.set_configuration(
        dimension=3,
        solver_type="Implicit" if implicit else "Explicit",
        backend="taichi",
    )
    fem.add_mesh(FEMMesh(points, penetrator_tetrahedra(), "TET4"))
    fem.add_material(
        model="NeoHookean",
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
    )
    top = np.flatnonzero(np.isclose(points[:, 2], np.max(points[:, 2])))

    def driven_displacement(time, coordinates):
        displacement = np.zeros_like(coordinates)
        displacement[:, 2] = -PILE_SPEED * float(time)
        return displacement

    fem.add_boundary_condition(dirichlet=DirichletBoundary().add(top, "all", driven_displacement))
    return points.shape[0]


def configure_fem_axisymmetric_penetrator(fem, penetration_start=0.0):
    from src.fem import DirichletBoundary, FEMMesh

    points, cells = axisymmetric_penetrator_mesh()
    fem.set_configuration(
        dimension=2,
        solver_type="Implicit",
        axisymmetric=True,
        axis_offset=0.0,
        backend="taichi",
    )
    fem.add_mesh(FEMMesh(points, cells, "TRI3"))
    fem.add_material(
        model="NeoHookean",
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
    )
    top = np.flatnonzero(np.isclose(points[:, 1], np.max(points[:, 1])))

    def driven_displacement(time, coordinates):
        displacement = np.zeros((coordinates.shape[0], 3), dtype=np.float64)
        displacement[:, 1] = -PILE_SPEED * max(float(time) - penetration_start, 0.0)
        return displacement

    fem.add_boundary_condition(dirichlet=DirichletBoundary().add(top, "all", driven_displacement))
    return top


def axisymmetric_penetrator_mesh(radial_nodes=4, axial_nodes=34):
    """Structured TRI3 mesh of the standard CPT meridian profile."""
    radial_nodes = int(radial_nodes)
    axial_nodes = int(axial_nodes)
    if radial_nodes < 2 or axial_nodes < 2:
        raise ValueError("axisymmetric penetrator mesh needs at least 2 nodes per direction")
    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    radii = np.linspace(profile[0, 0], profile[1, 0], radial_nodes)
    bottom = np.linspace(profile[0, 1], profile[1, 1], radial_nodes)
    points = np.asarray(
        [
            [radius, bottom[i] + eta * (profile[2, 1] - bottom[i])]
            for eta in np.linspace(0.0, 1.0, axial_nodes)
            for i, radius in enumerate(radii)
        ],
        dtype=np.float64,
    )
    cells = []
    for row in range(axial_nodes - 1):
        for column in range(radial_nodes - 1):
            lower_left = row * radial_nodes + column
            lower_right = lower_left + 1
            upper_left = lower_left + radial_nodes
            upper_right = upper_left + 1
            cells.extend(([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left]))
    return points, np.asarray(cells, dtype=np.int32)


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
    parser.add_argument("--dt", type=float)
    parser.add_argument("--time", type=float)
    parser.add_argument("--save-interval", type=float)
    parser.add_argument("--resolution-scale", type=float)
    parser.add_argument("--settling-time", type=float, default=0.0)
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData"))
    arguments = parser.parse_args()
    implicit = arguments.contact == "ipc"
    if arguments.dilation_angle is not None and not implicit:
        parser.error("--dilation-angle is supported by the IPC route")
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
            dilation_angle=arguments.dilation_angle,
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
            # Driving displacement scales the initial eliminated residual
            # with dt; keep its force tolerance comparable in timestep studies.
            residual_tolerance=1.0e-3 * min(1.0, 5.0e-4 / arguments.dt),
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
        initial_deformation = axisymmetric_initial_deformation_gradient(
            model.mpm.enginer.particle.x.to_numpy()[:particle_count]
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
        if arguments.time >= 10.0 and not summary["fully_inserted"]:
            raise RuntimeError(f"FEM pile stopped above the soil surface: top={final_top:.6e} m")
    else:
        model.run(verbose=True, steps=int(math.ceil(arguments.time / arguments.dt)))


if __name__ == "__main__":
    main()
