import os
import sys

from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

arch = os.environ.get("GT_ARCH") or os.environ.get("GEOTAICHI_TEST_ARCH")
if arch and arch.lower() == "cuda":
    arch = "gpu"
init_kwargs = {
    "dim": 2,
    "device_memory_GB": float(os.environ.get("GT_DEVICE_MEMORY_GB", "2")),
    "default_fp": os.environ.get("GT_DEFAULT_FP", "float32"),
}
if arch:
    init_kwargs["arch"] = arch
init(**init_kwargs)

mpm = MPM()

domain = [
    float(os.environ.get("GT_DOMAIN_X", "6.0")),
    float(os.environ.get("GT_DOMAIN_Y", "5.0")),
]
element_size = float(os.environ.get("GT_ELEMENT_SIZE", "0.025"))
column_size = [
    float(os.environ.get("GT_COLUMN_WIDTH", "2.0")),
    float(os.environ.get("GT_COLUMN_HEIGHT", "1.0")),
]

mpm.set_configuration(domain=domain,
                      background_damping=0.00,
                      gravity=[0., -9.8],
                      alphaPIC=0.005,
                      shape_function="QuadBSpline",
                      solver_type="Implicit",
                      )

mpm.set_implicit_solver_parameters(
    quasi_static=False,
    assemble_type=os.environ.get("GT_IMPLICIT_ASSEMBLE_TYPE", "MatrixFree"),
    max_iteration_number=int(os.environ.get("GT_IMPLICIT_MAX_ITER", "50")),
    displacement_tolerance=float(os.environ.get("GT_DISPLACEMENT_TOL", "1e-4")),
    residual_tolerance=float(os.environ.get("GT_RESIDUAL_TOL", "1e-10")),
)

mpm.set_solver(solver={
                           "Timestep":                   float(os.environ.get("GT_TIMESTEP", "1e-4")),
                           "SimulationTime":             float(os.environ.get("GT_SIMULATION_TIME", "5")),
                           "SaveInterval":               float(os.environ.get("GT_SAVE_INTERVAL", "1e-1")),
                           "SavePath":                   os.environ.get("GT_SAVE_PATH", "OutputData")
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    float(os.environ.get("GT_MAX_PARTICLES", "5.12e5")),
                                "max_constraint_number":  {
                                                               "max_displacement_constraint": int(os.environ.get("GT_MAX_DISPLACEMENT_CONSTRAINTS", "83000"))
                                                          },
                                "dof_multiplier":         int(os.environ.get("GT_DOF_MULTIPLIER", "2")),
                            })

mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":      1,
                               "Density":         2500,
                               "YoungModulus":    8.6e5,
                               "PoissionRatio":   0.3,
                               "Friction":        19,
                               "Cohesion":        10,
                               "Dilation":        0
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               [element_size, element_size]
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [0., 0.],
                            "BoundingBoxSize": column_size,
                            
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0, 0],
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   }
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "DisplacementConstraint",
                                             "Displacement":   [0., 0],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [domain[0], 0.]
                                        },

                                        {
                                             "BoundaryType":   "DisplacementConstraint",
                                             "Displacement":   [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., domain[1]]
                                        },
                                    ])

mpm.select_save_data()

mpm.run(gravity_field=lambda points: column_size[1] - points[:,1])

if os.environ.get("GT_SKIP_POSTPROCESSING", "0") != "1":
    mpm.postprocessing()
