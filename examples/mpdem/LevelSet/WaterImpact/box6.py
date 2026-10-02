import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=int, default=0)
args = parser.parse_args()
start_file = args.f
restart = False if start_file == 0 else True

init(device_memory_GB=3.7)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([8., 0.7, 1.]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.01,
                      alphaPIC=0.001, 
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      gravity=ti.Vector([0., 0., -9.8]),
                      material_type="Fluid",
                      #velocity_projection="Affine",
                      stabilize="Displacement F-Bar Method"
                      )

dempm.dem.set_configuration(
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
dempm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   2.,
                      "SaveInterval":     0.04,
                      "SavePath":         "box6"
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 6,
                                "levelset_grid_number": 205379,
                                "surface_node_number": 4322,
                                "max_plane_number": 1,
                                "body_coordination_number":   3,
                                "wall_coordination_number":   3,
                                "verlet_distance_multiplier": [0.15, 0.1],
                                "point_coordination_number":  [3, 2], 
                                "compaction_ratio":           [0.3, 0.3, 0.15, 0.15],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           980000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   541681,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    6,
                                  "wall_coordination_number":    0,
                                  "compaction_ratio": [0.01, 0.15]
                             })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           980000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   541681,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    3,
                                  "wall_coordination_number":    0,
                                  "compaction_ratio": [0.01, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            800,
                                "ForceLocalDamping":  0.2,
                                "TorqueLocalDamping": 0.1
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/box.stl').grids(space=0.05, extent=5),
                                "WriteFile":          False}) 

dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")
                            
dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               633981403.1,
                            "Poisson":                    0.3,
                            "Friction":                   0.1,
                            "Restitution":                0.6
                           })           
                           
dempm.dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "ShearModulus":               1250104175,
                            "Poisson":                    0.3,
                            "Friction":                   0.35,
                            "Restitution":                0.8
                           })  
                           

if restart:
    dempm.dem.read_restart(file_number=start_file, file_path="box6", 
                           particle=True, wall=True, ppcontact=True, pwcontact=True, is_continue=True)
else:
    dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5.275, 0.15, 0.075],
                                  "ScaleFactor": 0.15,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5.275, 0.35, 0.075],
                                  "ScaleFactor": 0.15,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5.275, 0.55, 0.075],
                                  "ScaleFactor": 0.15,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5.275, 0.25, 0.225],
                                  "ScaleFactor": 0.15,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5.275, 0.45, 0.225],
                                  "ScaleFactor": 0.15,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5.275, 0.35, 0.375],
                                  "ScaleFactor": 0.15,
                                  "BodyOrientation": "constant"
                                  }
                                ]})

    dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([4., 0.35, 0.0]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })

dempm.dem.select_save_data(particle=True, surface=True, bounding=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dempm.mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              5e6,
                               "Viscosity":            1e-3,
                               "ElementLength":        0.02,
                               "cL":                   1.5,
                               "cQ":                   2
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.035, 0.035, 0.035])
                        })


if restart:
    dempm.mpm.read_restart(file_number=start_file, file_path="box6", is_continue=True)
else:
    dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([3.5, 0.7, 0.4]),
                            
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
                                        "EndPoint":       [8., 0.0, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0, 0.7, 0],
                                        "EndPoint":       [8., 0.7, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 0.7, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [8., 0.0, 0],
                                        "EndPoint":       [8., 0.7, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [8., 0.7, 0.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., 1.],
                                        "StartPoint":     [0, 0.0, 1.],
                                        "EndPoint":       [8., 0.7, 1.],
                                    }])


dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Fluid Particle",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               2e4,
                                 "NormalViscousDamping":          0.2
                            }, dType='particle-particle')

dempm.run(mpm_gravity_field=True)

dempm.mpm.postprocessing(start_file=0, end_file=51, total_displacement=True, smooth_setting={'smooth_rad': 0.035, 'lower_bound': [0., 0., 0.], 'upper_bound': [8., 0.7, 1.]})

dempm.dem.postprocessing()


