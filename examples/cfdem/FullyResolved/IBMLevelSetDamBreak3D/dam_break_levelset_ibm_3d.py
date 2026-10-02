"""3D incompressible MPM dam break carrying ten buoyant LSDEM bodies."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

from geotaichi import DEMPM, init, polyhedron  # noqa: E402
from src.utils.SolverRuntime import python_callback  # noqa: E402

PREFIX = "GT_IBM_DAM3D_"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(PREFIX + name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(PREFIX + name, default))


ARCH = os.environ.get(PREFIX + "ARCH", "gpu")
OUTPUT = Path(os.environ.get(PREFIX + "OUTPUT", Path(__file__).resolve().parent / "OutputData")).expanduser().resolve()
DT = env_float("DT", 2.0e-4)
SIMULATION_TIME = env_float("TIME", 1.2)
SAVE_INTERVAL = env_float("SAVE_INTERVAL", 0.06)
DX = env_float("DX", 0.004)
PPC = env_int("PPC", 2)
BODY_DENSITY = env_float("BODY_DENSITY", 800.0)
LINEAR_SOLVER = os.environ.get(PREFIX + "LINEAR_SOLVER", "MGPCG")
SKIP_POSTPROCESS = os.environ.get(PREFIX + "SKIP_POSTPROCESS", "0") == "1"
STRICT = os.environ.get(PREFIX + "STRICT", "0") == "1"
SOLID_VOLUME_RELATIVE_TOLERANCE = 0.05

DOMAIN = [0.480, 0.192, 0.288]
WATER_SIZE = [0.228, 0.180, 0.216]
DOMAIN_CELLS = [round(length / DX) for length in DOMAIN]
WATER_CELLS = [round(length / DX) for length in WATER_SIZE]
WATER_ORIGIN = [DX, DX, DX]
RADIUS = 0.024
BODY_COUNT = 10
MAX_MPM_PARTICLES = math.prod(WATER_CELLS) * PPC**3
MESH_FILE = Path(os.environ.get(PREFIX + "MESH", ROOT / "assets/mesh/LSDEM/sand_middle.stl")).expanduser()
BODY_COORDINATION = max(BODY_COUNT - 1, 1)
WALL_COORDINATION = 6

if not all(math.isclose(count * DX, length, abs_tol=1.0e-12) for count, length in zip(DOMAIN_CELLS, DOMAIN)):
    raise ValueError("DX must divide every physical domain dimension")
if not all(math.isclose(count * DX, length, abs_tol=1.0e-12) for count, length in zip(WATER_CELLS, WATER_SIZE)):
    raise ValueError("DX must divide every physical water-column dimension")
if any(count % 2 ** (3 - 1) for count in DOMAIN_CELLS):
    raise ValueError("the default three-level MGPCG grid must be divisible by four")
if min(DT, SIMULATION_TIME, SAVE_INTERVAL, DX) <= 0.0 or PPC <= 0:
    raise ValueError("time, spacing, and particles per cell must be positive")
if BODY_DENSITY >= 1000.0:
    raise ValueError("the LSDEM density must be lower than the 1000 kg/m^3 water density")

centers = [
    [0.048, 0.048, 0.055],
    [0.118, 0.048, 0.055],
    [0.188, 0.048, 0.055],
    [0.048, 0.132, 0.055],
    [0.118, 0.132, 0.055],
    [0.188, 0.132, 0.055],
    [0.048, 0.048, 0.195],
    [0.118, 0.048, 0.195],
    [0.048, 0.132, 0.195],
    [0.118, 0.132, 0.195],
]
if len(centers) != BODY_COUNT:
    raise AssertionError("the center list must match the configured LSDEM body count")

print("# 3D incompressible MPM dam break + ten buoyant LSDEM IBM bodies")
print(f"# domain={DOMAIN}, cells={DOMAIN_CELLS}, dx={DX:g}, ppc={PPC}")
print(f"# body density={BODY_DENSITY:g} kg/m^3, D/dx={2.0 * RADIUS / DX:g}")
print(f"# LSDEM mesh={MESH_FILE.name}")
print(f"# exact initial MPM capacity={MAX_MPM_PARTICLES}")

init(
    dim=3,
    arch=ARCH,
    default_fp="float64",
    default_ip="int32",
    device_memory_GB=env_float("DEVICE_MEMORY_GB", 8.0),
    offline_cache=True,
    debug=False,
    kernel_profiler=False,
    log=False,
)

if not MESH_FILE.is_file():
    raise FileNotFoundError(MESH_FILE)
body_shape = polyhedron(file=str(MESH_FILE))
body_shape.grids(space=env_float("LEVELSET_GRID_SPACE_RATIO", 0.05) * body_shape.eqradius, extent=3)
local_radius = float(np.max(np.linalg.norm(body_shape.mesh.vertices - body_shape.mesh.center_mass, axis=1)))
grid_half_count = math.ceil(local_radius / body_shape.grid.grid_space) + body_shape.grid.extent
# Principal-axis alignment can rotate the local box; the realized vertex radius
# is the smallest orientation-independent bound for every generated SDF grid.
LEVELSET_GRID_NUMBER = (2 * grid_half_count + 1) ** 3
SURFACE_NODE_NUMBER = int(body_shape.mesh.vertices.shape[0])

dempm = DEMPM()
dempm.set_configuration(
    domain=DOMAIN,
    coupling_scheme="CFDEM",
    cfdem_resolution="FullyResolved",
    particle_interaction=True,
    wall_interaction=False,
    CFD_coupling_domain=[3, 6],
    gravity=[0.0, 0.0, -9.81],
    visualize=True,
)
dempm.mpm.set_configuration(
    dimension=3,
    background_damping=0.0,
    alphaPIC=0.5,
    mapping="USL",
    shape_function="QuadBSpline",
    gravity=[0.0, 0.0, -9.81],
    material_type="Fluid",
    velocity_projection="Affine",
    solver_type="Implicit",
    discretization="FDM",
    fluid_level_set=True,
    fluid_domain_volume_fraction=0.2,
    particle_shifting=True,
    density_projection=True,
    density_projection_interior_only=True,
    visualize=True,
)
implicit = {
    "linear_solver": LINEAR_SOLVER,
    "max_iteration_number": env_int("MAX_ITERATIONS", 200),
    "residual_tolerance": env_float("RESIDUAL_TOLERANCE", 1.0e-7),
}
if LINEAR_SOLVER == "MGPCG":
    implicit.update(multilevel=3, pre_and_post_smoothing=2, bottom_smoothing=20)
dempm.mpm.set_implicit_solver_parameters(**implicit)
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
        "Timestep": DT,
        "SimulationTime": SIMULATION_TIME,
        "SaveInterval": SAVE_INTERVAL,
        "SavePath": str(OUTPUT),
        "CFL": 0.5,
    }
)

dempm.dem.memory_allocate(
    memory={
        "max_material_number": 2,
        "max_rigid_body_number": BODY_COUNT,
        "max_rigid_template_number": 1,
        "levelset_grid_number": LEVELSET_GRID_NUMBER,
        "surface_node_number": SURFACE_NODE_NUMBER,
        "max_plane_number": 6,
        "body_coordination_number": BODY_COORDINATION,
        "wall_coordination_number": WALL_COORDINATION,
        "verlet_distance_multiplier": [0.15, 0.1],
        "point_coordination_number": [4, 2],
        "compaction_ratio": [0.15, 0.15],
    }
)
dempm.mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": MAX_MPM_PARTICLES,
        "verlet_distance_multiplier": 0.5,
        "max_constraint_number": {},
    }
)
dempm.memory_allocate(
    memory={
        # Non-overlapping LSDEM bodies can cover at most two carrier points at contact.
        "body_coordination_number": 2,
        "wall_coordination_number": 0,
        "compaction_ratio": [0.2, 0.1],
    }
)

for material_id, density in ((0, BODY_DENSITY), (1, 2500.0)):
    dempm.dem.add_attribute(
        materialID=material_id,
        attribute={
            "Density": density,
            "ForceLocalDamping": 0.01,
            "TorqueLocalDamping": 0.01,
        },
    )

dempm.dem.add_template(template={"Name": "irregular_grain", "Object": body_shape, "WriteFile": False})
dempm.dem.create_body(
    body={
        "GenerateType": "Create",
        "BodyType": "RigidBody",
        "Template": [
            {
                "Name": "irregular_grain",
                "GroupID": body_id,
                "MaterialID": 0,
                "BodyPoint": center,
                "Radius": RADIUS,
                "BodyOrientation": [17.0 * body_id % 180.0, 29.0 * body_id % 180.0, 41.0 * body_id % 180.0],
                "InitialVelocity": [0.0, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
            }
            for body_id, center in enumerate(centers)
        ],
    }
)

dempm.dem.choose_contact_model(
    particle_particle_contact_model="Linear Model",
    particle_wall_contact_model="Linear Model",
)
contact = {
    "NormalStiffness": env_float("CONTACT_STIFFNESS", 2000.0),
    "TangentialStiffness": env_float("TANGENTIAL_STIFFNESS", 1000.0),
    "Friction": 0.1,
    "NormalViscousDamping": 0.1,
    "TangentialViscousDamping": 0.05,
}
for pair in ((0, 0), (0, 1)):
    dempm.dem.add_property(materialID1=pair[0], materialID2=pair[1], property=contact)

wall_specs = [
    ([0.0, 0.5 * DOMAIN[1], 0.5 * DOMAIN[2]], [1.0, 0.0, 0.0]),
    ([DOMAIN[0], 0.5 * DOMAIN[1], 0.5 * DOMAIN[2]], [-1.0, 0.0, 0.0]),
    ([0.5 * DOMAIN[0], 0.0, 0.5 * DOMAIN[2]], [0.0, 1.0, 0.0]),
    ([0.5 * DOMAIN[0], DOMAIN[1], 0.5 * DOMAIN[2]], [0.0, -1.0, 0.0]),
    ([0.5 * DOMAIN[0], 0.5 * DOMAIN[1], 0.0], [0.0, 0.0, 1.0]),
    ([0.5 * DOMAIN[0], 0.5 * DOMAIN[1], DOMAIN[2]], [0.0, 0.0, -1.0]),
]
dempm.dem.add_wall(
    body=[
        {"WallType": "Plane", "MaterialID": 1, "WallCenter": point, "OuterNormal": normal}
        for point, normal in wall_specs
    ]
)

dempm.mpm.add_material(
    model="Newtonian",
    material={
        "MaterialID": 1,
        "Density": 1000.0,
        "Modulus": 2.0e6,
        "Viscosity": 1.0e-3,
        "ElementLength": DX,
        "cL": 1.5,
        "cQ": 2.0,
        "atmospheric_pressure": 0.0,
        "SurfaceTension": 0.0,
    },
)
dempm.mpm.add_element(element={"ElementType": "Staggered", "ElementSize": [DX, DX, DX], "GhostCell": 1})
dempm.mpm.add_region(
    region={
        "Name": "water_column",
        "Type": "Rectangle",
        "BoundingBoxPoint": WATER_ORIGIN,
        "BoundingBoxSize": WATER_SIZE,
    }
)
dempm.mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "water_column",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        ]
    }
)
# The volume-fraction IBM keeps fictitious fluid carriers inside immersed
# bodies; Eq. (19)/(21) constrains their grid velocity to rigid motion.
dempm.add_body(check_overlap=False)

solid_cells = [
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, DOMAIN[1], DOMAIN[2]]),
    ([1.0, 0.0, 0.0], [DOMAIN[0], 0.0, 0.0], [DOMAIN[0], DOMAIN[1], DOMAIN[2]]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [DOMAIN[0], 0.0, DOMAIN[2]]),
    ([0.0, 1.0, 0.0], [0.0, DOMAIN[1], 0.0], [DOMAIN[0], DOMAIN[1], DOMAIN[2]]),
    ([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [DOMAIN[0], DOMAIN[1], 0.0]),
    ([0.0, 0.0, 1.0], [0.0, 0.0, DOMAIN[2]], DOMAIN),
]
dempm.mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "Norm": normal,
            "StartPoint": start,
            "EndPoint": end,
            "CellThickness": 1,
        }
        for normal, start, end in solid_cells
    ]
)
dempm.mpm.select_save_data(particle=True, grid=True)
dempm.dem.select_save_data(surface=True, grid=True, bounding=True, wall=False)
dempm.select_save_data()
dempm.choose_contact_model(None, None)

history_time = [0.0]
history_center = [dempm.dem.scene.rigid.mass_center.to_numpy()[:BODY_COUNT].copy()]
history_velocity = [dempm.dem.scene.rigid.v.to_numpy()[:BODY_COUNT].copy()]


@python_callback
def record_trajectory():
    if int(dempm.sims.current_step) % env_int("SAMPLE_EVERY", 20):
        return
    history_time.append(float(dempm.sims.current_time + dempm.sims.delta))
    history_center.append(dempm.dem.scene.rigid.mass_center.to_numpy()[:BODY_COUNT].copy())
    history_velocity.append(dempm.dem.scene.rigid.v.to_numpy()[:BODY_COUNT].copy())


dempm.run(mpm_gravity_field=True, function=record_trajectory)

final_time = float(dempm.sims.current_time)
if history_time[-1] < final_time:
    history_time.append(final_time)
    history_center.append(dempm.dem.scene.rigid.mass_center.to_numpy()[:BODY_COUNT].copy())
    history_velocity.append(dempm.dem.scene.rigid.v.to_numpy()[:BODY_COUNT].copy())

times = np.asarray(history_time)
body_centers = np.asarray(history_center)
body_velocities = np.asarray(history_velocity)
np.savez(OUTPUT / "trajectory.npz", time=times, center=body_centers, velocity=body_velocities)

coupler = dempm.enginer.incompressible_coupler
mapped_volume = float(np.sum(coupler.solid_fraction.to_numpy()) * DX**3)
expected_volume = BODY_COUNT * 4.0 * math.pi * RADIUS**3 / 3.0
solid_volume_error = abs(mapped_volume - expected_volume) / expected_volume
vertical_displacement = body_centers[-1, :, 2] - body_centers[0, :, 2]
upward_velocity_count = int(np.count_nonzero(body_velocities[-1, :, 2] > 0.0))
maximum_upward_velocity = float(np.max(body_velocities[:, :, 2]))
maximum_upward_displacement = float(np.max(body_centers[:, :, 2] - body_centers[0, :, 2]))
finite = bool(np.isfinite(body_centers).all() and np.isfinite(body_velocities).all())
ghost = int(dempm.mpm.scene.element.ghost_cell)
cell_type = np.squeeze(dempm.mpm.scene.element.cell.type.to_numpy())
cell_pressure = np.squeeze(dempm.mpm.scene.element.cell.pressure.to_numpy())
active_slice = tuple(slice(ghost, ghost + count) for count in DOMAIN_CELLS)
active_cell_type = cell_type[active_slice]
fluid_cells = active_cell_type == 1
enclosed_air = active_cell_type == 0
for axis in range(3):
    enclosed_air &= np.roll(fluid_cells, 1, axis) & np.roll(fluid_cells, -1, axis)
    lower = [slice(None)] * 3
    lower[axis] = 0
    enclosed_air[tuple(lower)] = False
    upper = [slice(None)] * 3
    upper[axis] = -1
    enclosed_air[tuple(upper)] = False
particle_count = int(dempm.mpm.scene.particleNum[0])
particle_active = dempm.mpm.scene.particle.active.to_numpy()[:particle_count] > 0
particle_position = dempm.mpm.scene.particle.x.to_numpy()[:particle_count][particle_active]
domain = np.asarray(DOMAIN)
particle_tolerance = 1.0e-10 * DX
particle_inside = np.all(
    (particle_position >= -particle_tolerance) & (particle_position <= domain + particle_tolerance), axis=1
)
particle_cell = np.floor(np.minimum(np.maximum(particle_position, 0.0), np.nextafter(domain, 0.0)) / DX).astype(
    np.int64
)
particle_storage_cell = particle_cell[particle_inside] + ghost
occupied_cell_type = cell_type[tuple(particle_storage_cell.T)]
# Cell labels are built before the step's final G2P, so particles in newly
# occupied AIR cells are diagnostic rather than a same-time strict criterion.
duration_complete = math.isclose(times[-1], SIMULATION_TIME, rel_tol=0.0, abs_tol=0.1 * DT)
minimum_resolution_met = 2.0 * RADIUS / DX >= 10.0
metrics = {
    "case": "3D incompressible MPM dam break with ten buoyant LSDEM IBM bodies",
    "body_count": BODY_COUNT,
    "body_density_kg_m3": BODY_DENSITY,
    "fluid_density_kg_m3": 1000.0,
    "density_ratio": BODY_DENSITY / 1000.0,
    "levelset_mesh": MESH_FILE.name,
    "cell_counts": DOMAIN_CELLS,
    "cell_size_m": DX,
    "cells_per_body_diameter": 2.0 * RADIUS / DX,
    "initial_mpm_capacity": MAX_MPM_PARTICLES,
    "computational_mpm_particles": int(dempm.mpm.scene.particleNum[0]),
    "fictitious_ibm_carriers_retained": True,
    "levelset_grid_capacity_per_template": LEVELSET_GRID_NUMBER,
    "surface_nodes_per_body": SURFACE_NODE_NUMBER,
    "body_coordination_number": BODY_COORDINATION,
    "wall_coordination_number": WALL_COORDINATION,
    "requested_timestep_s": DT,
    "requested_time_s": SIMULATION_TIME,
    "timestep_s": float(np.median(np.diff(times)) / env_int("SAMPLE_EVERY", 20)),
    "final_time_s": float(times[-1]),
    "duration_complete": duration_complete,
    "minimum_resolution_met": minimum_resolution_met,
    "mean_vertical_displacement_m": float(np.mean(vertical_displacement)),
    "net_upward_displacement_body_count": int(np.count_nonzero(vertical_displacement > 0.0)),
    "upward_velocity_body_count": upward_velocity_count,
    "maximum_upward_velocity_m_s": maximum_upward_velocity,
    "maximum_upward_displacement_m": maximum_upward_displacement,
    "buoyant_response_observed": bool(upward_velocity_count > 0 or np.any(vertical_displacement > 0.0)),
    "mapped_solid_volume_relative_error": solid_volume_error,
    "solid_volume_relative_tolerance": SOLID_VOLUME_RELATIVE_TOLERANCE,
    "strict_enclosed_air_cells": int(np.count_nonzero(enclosed_air)),
    "particles_outside_domain": int(np.count_nonzero(~particle_inside)),
    "particles_in_air_cells": int(np.count_nonzero(occupied_cell_type == 0)),
    "particles_in_solid_cells": int(np.count_nonzero(occupied_cell_type == 2)),
    "finite_fluid_pressure": bool(np.isfinite(cell_pressure[active_slice][fluid_cells]).all()),
    "finite": finite,
}
metrics["passed"] = bool(
    finite
    and duration_complete
    and minimum_resolution_met
    and metrics["computational_mpm_particles"] == MAX_MPM_PARTICLES
    and solid_volume_error <= SOLID_VOLUME_RELATIVE_TOLERANCE
    and metrics["strict_enclosed_air_cells"] == 0
    and metrics["particles_outside_domain"] == 0
    and metrics["particles_in_solid_cells"] == 0
    and metrics["finite_fluid_pressure"]
    and metrics["buoyant_response_observed"]
)
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, sort_keys=True))

if not SKIP_POSTPROCESS:
    dempm.postprocessing(scheme="LSDEM")
if STRICT and not metrics["passed"]:
    raise RuntimeError(f"3D IBM validation failed: {metrics}")
