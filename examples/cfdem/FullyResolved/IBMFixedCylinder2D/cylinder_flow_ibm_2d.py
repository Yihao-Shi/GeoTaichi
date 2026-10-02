import json
import math
import os
import sys

from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

ARCH = os.environ.get("GEOTAICHI_ARCH", "cpu")
DT = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_DT", "2.0e-4"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_TIME", "5.0e-2"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_SAVE_INTERVAL", "5.0e-3"))
SAVE_PATH = os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_SAVE_PATH", "cylinder_flow_ibm_2d")
DX = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_DX", "0.02"))
PPC = int(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_PPC", "3"))
DOMAIN = [1.0, 0.4]
INFLOW_VELOCITY = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_VELOCITY", "0.20"))
DYNAMIC_VISCOSITY = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_VISCOSITY", "1.0e-3"))
DRIVE_ACCELERATION = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_ACCELERATION", "0.0"))
TRANSVERSE_PERTURBATION = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_PERTURBATION", "0.0"))
PARTICLE_SHIFTING = os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_PARTICLE_SHIFTING", "1") != "0"
DENSITY_PROJECTION = os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_DENSITY_PROJECTION", "1") != "0"
CYLINDER_RADIUS = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_RADIUS", "0.05"))
IBM_SOLID_DENSITY = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_SOLID_DENSITY", "1.0"))
PERIODIC_X = os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_PERIODIC_X", "1") != "0"
VALIDATE = os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_VALIDATE", "0") != "0"
VELOCITY_PROJECTION = os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_VELOCITY_PROJECTION", "Affine")
ALPHA_PIC = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_ALPHA_PIC", "0.5"))
LINEAR_SOLVER = os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_LINEAR_SOLVER", "MGPCG")
MULTILEVEL = int(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_MULTILEVEL", "3"))
RESIDUAL_TOLERANCE = float(os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_RESIDUAL_TOLERANCE", "1.0e-10"))
CYLINDER_CENTER = [0.35, 0.20]
allocation_cells = [math.ceil(length / DX) for length in DOMAIN]
if LINEAR_SOLVER == "MGPCG":
    multiple = 2 ** (MULTILEVEL - 1)
    allocation_cells = [multiple * math.ceil(count / multiple) for count in allocation_cells]
MAX_PARTICLES = int(
    os.environ.get("GEOTAICHI_CYLINDER_IBM_2D_MAX_PARTICLES", str(math.prod(allocation_cells) * PPC**2))
)


@ti.kernel
def kernel_update_circular_ibm_sdf(
    center: ti.types.vector(2, float),
    radius: float,
    solid_density_value: float,
    grid_size: ti.types.vector(2, float),
    solid_sdf: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
    solid_velocity_cell: ti.template(),
):
    band = 0.75 * ti.min(grid_size[0], grid_size[1])
    for I in ti.grouped(solid_fraction):
        position = (I.cast(float) + 0.5) * grid_size
        phi = (position - center).norm() - radius
        ti.atomic_min(solid_sdf[I], phi)
        solid_fraction[I] = ti.min(1.0, ti.max(0.0, 0.5 - 0.5 * phi / band))
        solid_density[I] = solid_density_value
        solid_velocity_cell[I] = ti.Vector([0.0, 0.0])


def update_cylinder_geometry(sims, scene):
    del sims
    mpm.enginer.ensure_ibm_source_fields(scene)
    kernel_update_circular_ibm_sdf(
        ti.Vector(CYLINDER_CENTER),
        CYLINDER_RADIUS,
        IBM_SOLID_DENSITY,
        scene.element.grid_size,
        scene.element.cell.solid_sdf,
        mpm.enginer.ibm_solid_fraction,
        mpm.enginer.ibm_solid_density,
        mpm.enginer.ibm_solid_velocity_cell,
    )


@ti.kernel
def seed_transverse_perturbation(particle_num: int, amplitude: float, particle: ti.template()):
    for np in range(particle_num):
        x, y = particle[np].x
        particle[np].v[1] += amplitude * ti.sin(2.0 * math.pi * x / DOMAIN[0]) * ti.sin(math.pi * y / DOMAIN[1])


init(dim=2, arch=ARCH, cpu_max_num_threads=4, device_memory_GB=2, debug=False, kernel_profiler=False)

mpm = MPM()

print("# 2D incompressible MPM: cylinder flow with a fixed SDF cut-cell boundary")
print(f"# cylinder IBM: center = {CYLINDER_CENTER}, radius = {CYLINDER_RADIUS}")
print(f"# nominal Reynolds number = {2.0 * CYLINDER_RADIUS * INFLOW_VELOCITY / DYNAMIC_VISCOSITY}")
print("# boundary model: SDF cut-cell impermeability plus IBM no-slip forcing")
print(f"# x particle boundary: {'periodic' if PERIODIC_X else 'destroy'}")

mpm.set_configuration(
    domain=DOMAIN,
    boundary=["Period" if PERIODIC_X else None, None],
    background_damping=0.0,
    alphaPIC=ALPHA_PIC,
    mapping="USL",
    shape_function="QuadBSpline",
    gravity=[DRIVE_ACCELERATION, 0.0],
    material_type="Fluid",
    velocity_projection=VELOCITY_PROJECTION,
    solver_type="Implicit",
    discretization="FDM",
    fluid_level_set=False,
    fluid_domain_volume_fraction=0.10,
    solid_sdf_cut_cell=True,
    solid_cut_cell_min_fraction=0.05,
    particle_shifting=PARTICLE_SHIFTING,
    density_projection=DENSITY_PROJECTION,
    density_projection_interior_only=True,
    visualize=True,
)

mpm.set_implicit_solver_parameters(
    linear_solver=LINEAR_SOLVER,
    multilevel=MULTILEVEL,
    pre_and_post_smoothing=2,
    bottom_smoothing=20,
    max_iteration_number=120,
    residual_tolerance=RESIDUAL_TOLERANCE,
)

mpm.set_solver(
    {"Timestep": DT, "SimulationTime": SIMULATION_TIME, "SaveInterval": SAVE_INTERVAL, "SavePath": SAVE_PATH}
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": MAX_PARTICLES,
        "verlet_distance_multiplier": 1.0,
        "max_constraint_number": {},
    }
)

mpm.add_material(
    model="Newtonian",
    material={
        "MaterialID": 1,
        "Density": 1.0,
        "Modulus": 2.0e3,
        "Viscosity": DYNAMIC_VISCOSITY,
        "ElementLength": DX,
        "cL": 1.5,
        "cQ": 2.0,
        "atmospheric_pressure": 0.0,
        "SurfaceTension": 0.0,
    },
)

mpm.add_element(element={"ElementType": "Staggered", "ElementSize": [DX, DX], "GhostCell": 1})

mpm.add_region(
    region=[
        {
            "Name": "channel",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": DOMAIN,
            "rotate2D": 0.0,
        }
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "channel",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [INFLOW_VELOCITY, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        ]
    }
)

if TRANSVERSE_PERTURBATION:
    seed_transverse_perturbation(int(mpm.scene.particleNum[0]), TRANSVERSE_PERTURBATION, mpm.scene.particle)

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, -1.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN[0], 0.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 1.0],
            "StartPoint": [0.0, DOMAIN[1]],
            "EndPoint": [DOMAIN[0], DOMAIN[1]],
            "CellThickness": 1,
        },
    ]
)

mpm.select_save_data(particle=True, grid=True)

mpm.run(gravity_field=False, cut_cell_function=update_cylinder_geometry, ibm_field_function=update_cylinder_geometry)

if VALIDATE:
    from src.mpm.engines.EngineKernel import interpolate_solid_sdf, mac_cell_center_velocity

    @ti.kernel
    def measure_no_slip_error(
        active_cnum: ti.types.vector(2, int), solid_fraction: ti.template(), node: ti.template()
    ) -> float:
        error2 = 0.0
        weight = 0.0
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            fraction = ti.max(0.0, ti.min(1.0, solid_fraction[I]))
            if fraction > 0.5:
                velocity = mac_cell_center_velocity(I, node)
                error2 += fraction * velocity.dot(velocity)
                weight += fraction
        return ti.sqrt(error2 / ti.max(weight, 1.0e-12))

    @ti.kernel
    def count_discrete_sdf_penetrations(
        particle_num: int,
        ghost_cell: int,
        cnum: ti.types.vector(2, int),
        grid_size: ti.types.vector(2, float),
        solid_sdf: ti.template(),
        particle: ti.template(),
    ) -> int:
        count = 0
        tolerance = 1.0e-6 * ti.min(grid_size[0], grid_size[1])
        for np in range(particle_num):
            if int(particle[np].active) == 1:
                phi = interpolate_solid_sdf(particle[np].x, ghost_cell, cnum, grid_size, solid_sdf)
                if phi < -tolerance:
                    count += 1
        return count

    active_cnum = mpm.scene.element.cnum - 2 * mpm.scene.element.ghost_cell
    actual_grid_size = np.array([float(mpm.scene.element.grid_size[d]) for d in range(2)])
    velocity_error = float(measure_no_slip_error(active_cnum, mpm.enginer.ibm_solid_fraction, mpm.scene.node))
    ghost = int(mpm.scene.element.ghost_cell)
    solid_sdf = mpm.scene.element.cell.solid_sdf.to_numpy()
    active_sdf = solid_sdf[ghost:-ghost, ghost:-ghost]
    mapped_area = float(np.sum(mpm.enginer.ibm_solid_fraction.to_numpy()) * np.prod(actual_grid_size))
    expected_area = math.pi * CYLINDER_RADIUS * CYLINDER_RADIUS
    area_error = abs(mapped_area - expected_area) / expected_area
    particle_count = int(mpm.scene.particleNum[0])
    position = mpm.scene.particle.x.to_numpy()[:particle_count]
    velocity = mpm.scene.particle.v.to_numpy()[:particle_count]
    particle_sdf = np.linalg.norm(position - np.asarray(CYLINDER_CENTER), axis=1) - CYLINDER_RADIUS
    particles_inside_analytic_geometry = int(np.count_nonzero(particle_sdf < -1.0e-10))
    particles_inside_sdf = int(
        count_discrete_sdf_penetrations(
            particle_count,
            mpm.scene.element.ghost_cell,
            mpm.scene.element.cnum,
            mpm.scene.element.grid_size,
            mpm.scene.element.cell.solid_sdf,
            mpm.scene.particle,
        )
    )
    finite_state = bool(np.isfinite(position).all() and np.isfinite(velocity).all())
    position_in_domain = bool(np.all(position >= 0.0) and np.all(position <= np.asarray(DOMAIN)))
    max_particle_speed = float(np.max(np.linalg.norm(velocity, axis=1)))
    active_fluid_cells = int(np.count_nonzero(mpm.scene.element.cell.type.to_numpy() == 1))
    expected_fluid_cells = math.prod(int(active_cnum[d]) for d in range(2))
    pressure_solver = mpm.enginer.poisson_solver
    metrics = {
        "case": "fixed-cylinder-ibm-2d",
        "fluid_solver": "implicit-incompressible-fdm",
        "cells_per_diameter": 2.0 * CYLINDER_RADIUS / min(actual_grid_size),
        "mapped_area": mapped_area,
        "expected_area": expected_area,
        "area_relative_error": area_error,
        "no_slip_velocity_l2": velocity_error,
        "finite_particle_state": finite_state,
        "particle_position_in_domain": position_in_domain,
        "max_particle_speed": max_particle_speed,
        "negative_sdf_cells": int(np.count_nonzero(active_sdf < 0.0)),
        "particles_inside_sdf": particles_inside_sdf,
        "particles_inside_analytic_geometry": particles_inside_analytic_geometry,
        "minimum_analytic_particle_sdf": float(np.min(particle_sdf)),
        "maximum_analytic_penetration_tolerance": 0.1 * min(actual_grid_size),
        "active_fluid_cells": active_fluid_cells,
        "expected_fluid_cells": expected_fluid_cells,
        "pressure_iterations": int(getattr(pressure_solver, "last_iterations", -1)),
        "pressure_initial_residual": float(
            getattr(pressure_solver, "initial_residual", getattr(pressure_solver, "last_initial_residual", math.nan))
        ),
        "pressure_final_residual": float(
            getattr(pressure_solver, "final_residual", getattr(pressure_solver, "last_residual", math.nan))
        ),
        "passed": (
            area_error <= 0.05
            and velocity_error <= 0.1 * abs(INFLOW_VELOCITY)
            and np.isfinite(active_sdf).all()
            and np.any(active_sdf < 0.0)
            and particles_inside_sdf == 0
            and np.min(particle_sdf) >= -0.1 * min(actual_grid_size)
            and finite_state
            and position_in_domain
            and max_particle_speed <= 2.0 * abs(INFLOW_VELOCITY)
            and active_fluid_cells >= 0.95 * expected_fluid_cells
        ),
    }
    metrics_path = Path(SAVE_PATH) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    if not metrics["passed"]:
        raise SystemExit("fixed-cylinder IBM validation criteria failed")

if os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "1":
    mpm.postprocessing()
