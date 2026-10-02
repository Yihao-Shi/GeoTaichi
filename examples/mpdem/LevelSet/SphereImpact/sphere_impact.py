import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([10, 10, 10]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.00,
                      alphaPIC=0.00, 
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      gravity=[0., 0., 0.],
                      particle_shifting=True)

dempm.dem.set_configuration(
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      

dempm.set_solver({
                      "Timestep":         2e-5,
                      "SimulationTime":   0.5505,
                      "SaveInterval":     0.05,
                      "SavePath":         'OutputData'
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_rigid_body_number": 1,
                                "levelset_grid_number": 85193,
                                "surface_node_number": 338,
                                "max_plane_number": 0,
                                "body_coordination_number":   0,
                                "wall_coordination_number":   0,
                                "verlet_distance_multiplier": [0.0, 0.0],
                                "point_coordination_number":  [0, 0], 
                                "compaction_ratio":           [0.0, 0., 0., 0.],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           256000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    1,
                                  "wall_coordination_number":    0,
                                  "compaction_ratio": [1.0, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand.stl').grids(space=5, extent=3),
                                "WriteFile":          False}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5.71, 5., 5.],
                                  "Radius": 0.5,
                                  "FixMotion":  ["Free", "Free", "Free"],
                                  "InitialVelocity":      [-0.1, -0.1, 0]
                                  },
                                ]})
                 
dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                         particle_wall_contact_model=None)
                           

dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1300,
                               "YoungModulus":                  2e5,
                               "PoissionRatio":                 0.4,
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([3.99, 4.5, 4.5]),
                            "BoundingBoxSize": ti.Vector([1., 1., 1.]),
                            
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0.1, 0.1, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   
dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            9e3,
                                 "TangentialStiffness":        6e3,
                                 "Friction":                   0.,
                                 "NormalViscousDamping":       0.,
                                 "TangentialViscousDamping":   0.
                            })

dempm.run()

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
