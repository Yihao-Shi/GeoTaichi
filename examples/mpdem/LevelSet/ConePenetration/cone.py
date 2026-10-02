import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=5.0, debug=False)

h = 0.0

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([0.1, 0.1, 0.15]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.00,
                      alphaPIC=0.002,
                      mapping="USF", 
                      shape_function="QuadBSpline",
                      gravity=ti.Vector([0., 0., -9.8]),
                      material_type="Solid",
                      velocity_projection="Affine",
                      #sparse_grid=True
                      )

dempm.dem.set_configuration(
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
dempm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   0.3,
                      "SaveInterval":     0.015,
                      "SavePath":         f"Height{h}"
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 1,
                                "levelset_grid_number": 166375,
                                "surface_node_number": 1325,
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
                                "Density":            59500,
                                "ForceLocalDamping":  0.0,
                                "TorqueLocalDamping": 0.0
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/cone.stl').grids(space=0.002, extent=7).reset(False),
                                "WriteFile":          False}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.05, 0.05, 0.1165],
                                  "ScaleFactor": 1.0,
                                  "BodyOrientation": "constant",
                                  "InitialVelocity": [0., 0., -math.sqrt(2 * h * 9.8)]
                                  }
                                ]})

dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                               particle_wall_contact_model=None)        
                           

dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(model="GranularMaterial",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1630,
                               "YoungModulus":                  2e6,
                               "PoissionRatio":                 0.3,
                               "StaticFriction":                      35,
                               "DynamicFriction":                      38,
                               "AverageDiameter":                      0.0028,
                               "InertialNumber":                       0.0015,
                               "eps":                          0.001
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.0025, 0.0025, 0.0025])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.1]),
                            
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
                                        "EndPoint":       [0.1, 0.0, 0.15],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0, 0.1, 0],
                                        "EndPoint":       [0.1, 0.1, 0.15],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 0.1, 0.15],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [0.1, 0.0, 0],
                                        "EndPoint":       [0.1, 0.1, 0.15],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0.1, 0.1, 0.],
                                    }])


dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            9e3,
                                 "TangentialStiffness":        6e3,
                                 "Friction":                   0.7,
                                 "NormalViscousDamping":       0.0,
                                 "TangentialViscousDamping":   0.0
                            })

dempm.run(mpm_gravity_field=True)

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
