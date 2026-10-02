import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import argparse, math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-w', type=float, default=1)
parser.add_argument('-suffix', type=int, default=0)
parser.add_argument('-t', type=float, default=6.0)
parser.add_argument('-st', type=float, default=0.3)
parser.add_argument('-path', type=str, default=None)
parser.add_argument('-rt', '--refine-threshold', type=float, default=0.1)
parser.add_argument('-ri', '--refine-interval', type=int, default=1000)
parser.add_argument('-rr', '--refine-ratio', type=float, default=0.08)
parser.add_argument('-buffer', type=int, default=2)
parser.add_argument('--particle-split-batch', type=int, default=8192)
parser.add_argument('-penalty', type=float, default=1.0)
parser.add_argument('-pit', '--penalty-iterations', type=int, default=1)
parser.add_argument('--penalty-beta', type=float, default=10.0)
parser.add_argument('--mapping', choices=['USL', 'USF', 'MUSL'], default='USL')
parser.add_argument(
    '--shape-function',
    choices=['Linear', 'GIMP', 'QuadBSpline', 'CubicBSpline'],
    default='QuadBSpline',
)
parser.add_argument('--no-refine-particles', action='store_true')
parser.add_argument('--bridging-domain', action='store_true')
parser.add_argument('--bridging-cells', type=int, default=1)
parser.add_argument(
    '--hanging-constraint-mode',
    choices=['Penalty', 'ShapeFunction'],
    default='Penalty',
)
parser.add_argument('--hanging-shape-capacity-factor', type=int, default=3)
parser.add_argument('--max-particle-number', type=float, default=1500000)
parser.add_argument('--device-memory-gb', type=float, default=5.0)
parser.add_argument('--split-interior-only', dest='split_interior_only', action='store_true', default=True)
parser.add_argument('--split-boundary-particles', dest='split_interior_only', action='store_false')
parser.add_argument('--split-interior-volume-fraction', type=float, default=0.95)
parser.add_argument('--save-grid', action='store_true')
parser.add_argument('--uniform-grid', action='store_true')
args = parser.parse_args()

scale = args.w

from geotaichi import *

init(debug=False, arch='gpu', device_memory_GB=args.device_memory_gb, random_seed=0)

mpm = MPM()

mpm.set_configuration(domain=[50., scale * 50., 20.],
                      background_damping=0.,
                      alphaPIC=0.00, 
                      mapping=args.mapping,
                      shape_function=args.shape_function,
                      gravity=[0., 0., -9.8],
                      sparse_grid=False,
                      #velocity_projection="Affine",
                      #stabilize="B-Bar Method",
                      #pressure_smooth=True
                      )

path = ('Flat' if args.uniform_grid else 'FlatAdaptive') + str(args.suffix)
mpm.set_solver({
                      "Timestep":         3e-4,
                      "SimulationTime":   args.t,
                      "SaveInterval":     args.st,
                      "SavePath":         path
                 }) 
                 
mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           int(args.max_particle_number * args.w),
                                "verlet_distance_multiplier":    0.1,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   526702
                                                          }
                            })  

if args.path is not None and os.path.isfile(args.path):
    mpm.add_material(model="DruckerPrager",
                     material={
                                   "MaterialID":   1,
                                   "MaterialFile": args.path
                     })
else:
    mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "Density":              1800.,
                               "YoungModulus":                  1e8,
                               "PossionRatio":                 0.3,
                               "Friction":                20,
                               "Cohesion":                      6700,
                               "Dilation":                      9,
                               "dpType":                         "Inscribed"
                 })

element = {
              "ElementType": "R8N3D",
              "ElementSize": [1.0, 1.0, 1.0],
          }

if not args.uniform_grid:
    element["AdaptiveGrid"] = {
        "RefineThreshold": args.refine_threshold,
        "RefineInterval": args.refine_interval,
        "RefineRatio": args.refine_ratio,
        "RefineParticles": not args.no_refine_particles,
        "ParticleSplitBatch": args.particle_split_batch,
        "BufferCells": args.buffer,
        "PenaltyBeta": args.penalty_beta,
        "Penalty": args.penalty,
        "PenaltyIterations": args.penalty_iterations,
        "BridgingDomain": args.bridging_domain,
        "BridgingCells": args.bridging_cells,
        "HangingConstraintMode": args.hanging_constraint_mode,
        "HangingShapeCapacityFactor": args.hanging_shape_capacity_factor,
        "SplitInteriorOnly": args.split_interior_only,
        "SplitInteriorVolumeFraction": args.split_interior_volume_fraction,
    }
mpm.add_element(element=element)


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0.0, 0.0, 0.0],
                            "BoundingBoxSize": [50., scale * 50, 6],
                            
                      }])
                      
mpm.add_region(region=[{
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0.0, 0.0, 6.0],
                            "BoundingBoxSize": [20, scale * 50, 10.],
                            
                      }])
                      
mpm.add_region(region=[{
                            "Name": "region3",
                            "Type": "TrianglarPrism",
                            "BoundingBoxPoint": [20.0, 0.0, 6.0],
                            "BoundingBoxSize": [10., scale * 50., 10.],
                            
                      }])

mpm.add_body(body={
                       "WriteFile": False,
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":   [0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":   [0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   {
                                       "RegionName":         "region3",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":   [0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   
mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [50., 0.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0, scale * 50.0, 0],
                                        "EndPoint":       [50., scale * 50.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., scale * 50.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., 0., 0.],
                                        "StartPoint":     [50., 0.0, 0],
                                        "EndPoint":       [50., scale * 50.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., 0., 0.],
                                        "StartPoint":     [0., 0.0, 0.],
                                        "EndPoint":       [50., scale * 50.0, 0.],
                                    }])

mpm.select_save_data(grid=args.save_grid)

def get_gravity(points):
    return np.where(points[:, 0]<20.0, 16.-points[:, 2],
                   np.where(points[:, 0]<30.0, 36.0-points[:, 0]-points[:, 2],
                            6.-points[:, 2]))
                            
mpm.run(mpm_gravity_field=get_gravity)

mpm.postprocessing()
