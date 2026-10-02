import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=4, debug=False)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([2.5, 0.6, 0.5]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.005,
                      alphaPIC=0.001, 
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=ti.Vector([0., 0., -9.8]))

dempm.dem.set_configuration(
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
dempm.set_solver({
                      "Timestep":         2.5e-4,
                      "SimulationTime":   4.,
                      "SaveInterval":     0.25,
                      "CFL":              1.,
                      "SavePath":         'OutputData'
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 3,
                                "max_rigid_body_number": 15,
                                "levelset_grid_number": 257853,
                                 "max_rigid_template_number": 2,
                                "surface_node_number": 4902,
                                "max_plane_number": 6,
                                "body_coordination_number":   6,
                                "wall_coordination_number":   4,
                                "verlet_distance_multiplier": [0.15, 0.2],
                                "point_coordination_number":  [3, 2], 
                                "compaction_ratio":           [0.5, 0.5, 0.5, 0.5],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           1920000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   147135,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    6,
                                  "wall_coordination_number":    3,
                                  "compaction_ratio": [0.2, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            850,
                                "ForceLocalDamping":  0.15,
                                "TorqueLocalDamping": 0.05
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/L_plow.obj').grids(space=0.003, extent=8),
                                "SurfaceResolution":   4902}) 

dempm.dem.add_template(template={
                                "Name":               "clump2",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/Stone.obj').grids(space=12, extent=6),
                                "SurfaceResolution":   4902}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.6, 0.2, 0.25],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.8, 0.2, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 0.2, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.2, 0.2, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.4, 0.2, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.6, 0.4, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.8, 0.4, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 0.4, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.2, 0.4, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.4, 0.4, 0.24],
                                  "ScaleFactor": 0.0002
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [2.15, 0.2, 0.175],
                                  "BodyOrientation": [89.5, -44.5, 0.5],
                                  "ScaleFactor": 1.,
                                  "FixMotion": ["Fix", "Fix", "Fix"]
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [2.15, 0.4, 0.175],
                                  "BodyOrientation": [89.5, -44.5, 0.5],
                                  "ScaleFactor": 1.,
                                  "FixMotion": ["Fix", "Fix", "Fix"]
                                  }
                                ]})
                 
dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model=None)
                            
dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e7,
                            "TangentialStiffness":        1e7,
                            "Friction":                   0.25,
                            "NormalViscousDamping":       0.15,
                            "TangentialViscousDamping":   0.15
                           })  
                  
dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(model="GranularMaterial",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1700,
                               "YoungModulus":                  1e6,
                               "PoissionRatio":                 0.3,
                               "StaticFriction":                0.25,
                               "DynamicFriction":               0.38,
                               "AverageDiameter":               0.001,
                               "InertialNumber":                0.03,
                               "eps":                           0.01
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.01, 0.01, 0.01])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([2.0, 0.6, 0.2]),
                            
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
                                        "EndPoint":       [2.5, 0.0, 0.5],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0, 0.6, 0],
                                        "EndPoint":       [2.5, 0.6, 0.5],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 0.6, 0.5],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [2.0, 0.0, 0],
                                        "EndPoint":       [2.0, 0.6, 0.5],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [2.5, 0.6, 0.],
                                    }])

dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            9e3,
                                 "TangentialStiffness":        6e3,
                                 "Friction":                   0.25,
                                 "NormalViscousDamping":       0.15,
                                 "TangentialViscousDamping":   0.15
                            })

dempm.dem.scene.rigid[10].v = [-0.48, 0., 0.]
dempm.dem.scene.rigid[11].v = [-0.48, 0., 0.]

dempm.run(mpm_gravity_field=True)

dempm.mpm.postprocessing()

#dempm.dem.postprocessing()
