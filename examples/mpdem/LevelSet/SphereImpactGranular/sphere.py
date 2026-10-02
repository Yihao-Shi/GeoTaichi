import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-rho', type=float, default=700.)
parser.add_argument('-mu', type=float, default=0.5)
parser.add_argument('-l', type=float, default=0.2)
parser.add_argument('-model', type=int, default=0)
args = parser.parse_args()

from geotaichi import *

init(device_memory_GB=6.0, debug=False)

#rho = 2200
#mu = 0.5
theta = math.atan(args.mu) / math.pi * 180.
#h = 0.2 #+ 0.0125

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([0.2, 0.2, 0.1]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=False,
                        visualize=True)

dempm.mpm.set_configuration( 
                      background_damping=0.00,
                      alphaPIC=0.0,
                      mapping="USF", 
                      shape_function="GIMP",
                      gravity=ti.Vector([0., 0., -9.8]),
                      material_type="Solid",
                      #velocity_projection="Affine",
                      #sparse_grid=True
                      )

dempm.dem.set_configuration(
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
dempm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   0.1,
                      "SaveInterval":     0.1,
                      "SavePath":         f"Rho_{args.rho}_Mu_{args.mu}_H_{args.l}_{args.model}"
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 1,
                                "levelset_grid_number": 166375,
                                "surface_node_number": 338,
                                "max_plane_number": 0,
                                "body_coordination_number":   0,
                                "wall_coordination_number":   0,
                                "verlet_distance_multiplier": [0.15, 0.1],
                                "point_coordination_number":  [3, 2], 
                                "compaction_ratio":           [0.3, 0.3, 0.15, 0.15],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           2400000,
                                "verlet_distance_multiplier":    0.4,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   197642,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    1,
                                  "wall_coordination_number":    0,
                                  "compaction_ratio": [0.02, 0.1]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            args.rho,
                                "ForceLocalDamping":  0.0,
                                "TorqueLocalDamping": 0.0
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sphere.stl').grids(space=0.05, extent=7),
                                "WriteFile":          False}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.1, 0.1, 0.0725],
                                  "Radius": 0.0125,
                                  "BodyOrientation": "constant",
                                  "InitialVelocity": [0., 0., -math.sqrt(2 * args.l * 9.8)]
                                  }
                                ]})

dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                               particle_wall_contact_model=None)        
                           

dempm.dem.select_save_data(surface=True)

if args.model==0:
     dempm.mpm.add_material(model="GranularMaterial",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1510,
                               "YoungModulus":                  2e6,
                               "PoissionRatio":                 0.3,
                               "StaticFriction":                      theta,
                               "DynamicFriction":                      theta,
                               "AverageDiameter":                      0.001,
                               "InertialNumber":                       0.03,
                               "eps":                          0.01
                 })
else:
    dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "RateDependent":        True,
                               "Density":              1510.,
                               "YoungModulus":                  2e6,
                               "PossionRatio":                 0.3,
                               "StaticFriction":                theta,
                               "DynamicFriction":               theta,
                               "AverageDiameter":               0.001,
                               "InertialNumber":                0.03,
                               "eps":                           0.01
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.004, 0.004, 0.004])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.2, 0.2, 0.06]),
                            
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   

dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., -1., 0.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0.2, 0.0, 0.1],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0, 0.2, 0],
                                        "EndPoint":       [0.2, 0.2, 0.1],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 0.2, 0.1],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [0.2, 0.0, 0],
                                        "EndPoint":       [0.2, 0.2, 0.1],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0.2, 0.2, 0.],
                                    }])


dempm.mpm.select_save_data(particle=False)

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            9e2,
                                 "TangentialStiffness":        6e2,
                                 "Friction":                   0.3,
                                 "NormalViscousDamping":       0.0,
                                 "TangentialViscousDamping":   0.0
                            })

dempm.run(mpm_gravity_field=True)
