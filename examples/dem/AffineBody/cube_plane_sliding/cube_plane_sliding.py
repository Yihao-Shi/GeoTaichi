"""An affine-body cube sliding on a horizontal IPC plane."""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geotaichi import DEM, init, polyhedron, python_callback


arch = os.environ.get("GT_ARCH", "cpu" if sys.platform == "darwin" else "gpu")
if sys.platform == "darwin" and arch == "cpu":
    os.environ["GEOTAICHI_FORCE_CPU"] = "1"
init(arch=arch, log=False, debug=False, offline_cache=False)

scale = 0.25
dem = DEM(log=False)
dem.set_configuration(
    domain=[1.5, 1.0, 0.8],
    scheme="AffineBody",
    search="LinkedCell",
    gravity=[0.0, 0.0, -9.81],
    visualize=False,
    log=False,
)
dem.set_affine_body_parameters(
    assemble_type="MatrixFree",
    young_modulus=5.0e5,
    friction_epsv=1.0e-4,
    friction_iterations=-1,
    friction_max_iterations=20,
    friction_tolerance=1.0e-7,
    newton_tolerance=1.0e-4,
    max_newton_iteration=20,
    line_search_max_iteration=50,
    max_step=0.02,
    ccd_type="accd",
)
dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_affine_body_number": 1,
        "surface_node_number": 16,
        "max_plane_number": 1,
        "body_coordination_number": 4,
        "wall_coordination_number": 4,
        "compaction_ratio": [1.0, 1.0],
    },
    log=False,
)
dem.set_solver(
    {
        "Timestep": 1.0e-3,
        "SimulationTime": 0.3,
        "SaveInterval": 0.02,
        "SavePath": "AffineCubePlaneSliding",
    },
    log=False,
)
dem.add_attribute(
    materialID=0,
    attribute={"Density": 1000.0, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0},
)
dem.add_template(
    template={
        "Name": "cube",
        "TemplateType": "AffineBody",
        "Object": polyhedron(file=str(ROOT / "assets/mesh/AffineBody/cube.obj")),
    }
)
dem.create_body(
    body={
        "BodyType": "AffineBody",
        "Template": {
            "Name": "cube",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": [0.35, 0.5, 0.5 * scale + 0.002],
            "ScaleFactor": scale,
            "InitialVelocity": [1.0, 0.0, 0.0],
            "Friction": 0.25,
        },
    }
)
dem.add_wall(
    {
        "WallType": "Plane",
        "MaterialID": 0,
        "WallCenter": np.array([0.0, 0.0, 0.0]),
        "OuterNormal": np.array([0.0, 0.0, 1.0]),
    }
)
dem.add_property(
    materialID1=0,
    materialID2=0,
    property={"Dhat": 0.03, "BarrierStiffness": 8.0e5, "Friction": 0.25},
    dType="all",
)


@python_callback
def report_convergence():
    e = dem.enginer
    print(
        f"step={dem.sims.current_step + 1:04d} "
        f"newton_ok={e.last_inner_converged} "
        f"newton_it={e.last_newton_iterations} "
        f"newton_res={e.last_inner_residual:.3e} "
        f"friction_ok={e.last_friction_converged} "
        f"friction_it={e.last_friction_iterations} "
        f"friction_res={e.last_friction_residual:.3e} "
        f"linear_ok={e.last_linear_converged} "
        f"linear_it={e.last_linear_iterations} "
        f"linear_res={e.last_linear_residual:.3e} "
        f"ls_ok={e.step_line_search_converged} "
        f"ls_calls={e.step_line_search_calls} "
        f"ls_alpha_min={e.step_line_search_min_alpha:.3e} "
        f"ls_backtracks_max={e.step_line_search_max_backtracks}"
    )


dem.run(function=report_convergence)
