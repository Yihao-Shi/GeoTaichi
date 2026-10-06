import os
import sys
import math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


# Struct precision is selected at import, independently of ti.init(default_fp).
os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
from geotaichi import *
from src.utils.SolverRuntime import python_callback

ARCH = os.environ.get("GEOTAICHI_ARCH", "cpu")
DT = float(os.environ.get("GEOTAICHI_LID_2D_DT", "0.002"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_LID_2D_TIME", "30.0"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_LID_2D_SAVE_INTERVAL", "0.5"))
SAVE_PATH = os.environ.get(
    "GEOTAICHI_LID_2D_SAVE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "OutputData/lid_driven_cavity_2d"),
)
DX = float(os.environ.get("GEOTAICHI_LID_2D_DX", str(1.0 / 64)))
PPC = int(os.environ.get("GEOTAICHI_LID_2D_PPC", "3"))
MAX_PARTICLES = int(os.environ.get("GEOTAICHI_LID_2D_MAX_PARTICLES", str((4 * math.ceil(1.0 / DX / 4)) ** 2 * PPC**2)))
LID_VELOCITY = float(os.environ.get("GEOTAICHI_LID_2D_VELOCITY", "1.0"))
VISCOSITY = float(os.environ.get("GEOTAICHI_LID_2D_VISCOSITY", "0.01"))
DOMAIN = [1.0, 1.0]
LID_Y = DOMAIN[1]


@ti.kernel
def kernel_set_lid_solid_velocity(
    lid_y: float,
    lid_length: float,
    lid_velocity: float,
    grid_size: ti.types.vector(2, float),
    solid_velocity_x: ti.template(),
):
    for I in ti.grouped(solid_velocity_x):
        y = (I[1] + 0.5) * grid_size[1]
        if y >= lid_y and 0.0 < I[0] * grid_size[0] < lid_length:
            solid_velocity_x[I] = lid_velocity


def set_moving_lid_velocity(sims, scene):
    mpm.enginer.ensure_solid_cut_cell_fields(sims, scene)
    kernel_set_lid_solid_velocity(
        LID_Y, DOMAIN[0], LID_VELOCITY, scene.element.grid_size, mpm.enginer.solid_face_velocity[0]
    )


init(
    dim=2,
    arch=ARCH,
    cpu_max_num_threads=int(os.environ.get("GEOTAICHI_CPU_THREADS", "4")),
    device_memory_GB=2,
    debug=False,
    kernel_profiler=True,
)

mpm = MPM()

print("# 2D incompressible MPM: lid-driven cavity")
print(f"# moving lid: y = {LID_Y}, velocity = [{LID_VELOCITY}, 0.0]")

mpm.set_configuration(
    domain=DOMAIN,
    background_damping=0.0,
    alphaPIC=0.5,
    mapping="USL",
    shape_function="QuadBSpline",
    gravity=[0.0, 0.0],
    material_type="Fluid",
    velocity_projection="Affine",
    solver_type="Implicit",
    discretization="FDM",
    fluid_level_set=False,
    fluid_domain_volume_fraction=0.10,
    solid_sdf_cut_cell=True,
    fluid_wall_no_slip=True,
    solid_cut_cell_min_fraction=0.05,
    particle_shifting=os.environ.get("GEOTAICHI_LID_2D_SHIFTING", "1") == "1",
    density_projection=os.environ.get("GEOTAICHI_LID_2D_DENSITY_PROJECTION", "1") == "1",
    density_projection_interior_only=True,
    visualize=False,
)

mpm.set_implicit_solver_parameters(
    linear_solver="MGPCG",
    multilevel=3,
    pre_and_post_smoothing=2,
    bottom_smoothing=20,
    max_iteration_number=120,
    residual_tolerance=1.0e-8,
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
        "Viscosity": VISCOSITY,
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
            "Name": "fluid",
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
                "RegionName": "fluid",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        ]
    }
)

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
            "Norm": [-1.0, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.0, DOMAIN[1]],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [1.0, 0.0],
            "StartPoint": [DOMAIN[0], 0.0],
            "EndPoint": [DOMAIN[0], DOMAIN[1]],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 1.0],
            "StartPoint": [0.0, LID_Y],
            "EndPoint": [DOMAIN[0], LID_Y],
            "CellThickness": 1,
        },
    ]
)

mpm.select_save_data(particle=True, grid=True)

phase_counts = ti.field(int, shape=2)


@ti.kernel
def record_phase_counts(cell_type: ti.template()):
    air = 0
    for I in ti.grouped(cell_type):
        # Staggered cell fields use physical indices (ghosts are negative).
        if 0 <= I[0] < ti.static(round(DOMAIN[0] / DX)) and 0 <= I[1] < ti.static(round(DOMAIN[1] / DX)):
            air += int(cell_type[I] == 0)
    ti.atomic_max(phase_counts[0], air)
    phase_counts[1] += int(air > 0)


@python_callback
def record_mac_flow():
    record_phase_counts(mpm.scene.element.cell.type)
    step = int(mpm.sims.current_step) + 1
    if step % max(1, round(SAVE_INTERVAL / DT)) and mpm.sims.current_time + mpm.sims.delta < SIMULATION_TIME - 1e-12:
        return
    np.savez(
        os.path.join(SAVE_PATH, f"mac_flow_{step:06d}.npz"),
        time=mpm.sims.current_time + mpm.sims.delta,
        velocity_x=mpm.scene.node.velocity[0].to_numpy(),
        velocity_y=mpm.scene.node.velocity[1].to_numpy(),
        cell_type=mpm.scene.element.cell.type.to_numpy(),
        grid_size=np.asarray(mpm.scene.element.grid_size),
        ghost_cell=mpm.scene.element.ghost_cell,
        lid_velocity=LID_VELOCITY,
        viscosity=VISCOSITY,
        domain=DOMAIN,
        cell_volume_fraction=mpm.scene.element.cell_volumefrac.to_numpy(),
        shift_node_volume=(
            mpm.enginer.shifting_node_volume.to_numpy() if mpm.enginer.shifting_node_volume is not None else np.empty(0)
        ),
        max_interior_air_cells_over_steps=int(phase_counts[0]),
        steps_with_interior_air=int(phase_counts[1]),
    )


mpm.run(gravity_field=False, cut_cell_function=set_moving_lid_velocity, function=record_mac_flow)

if os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "1":
    mpm.postprocessing()
