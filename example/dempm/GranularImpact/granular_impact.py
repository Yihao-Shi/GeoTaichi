try:
    from geotaichi import *
except:
    import os
    import sys
    current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(current_file_path)
    from geotaichi import *

init(device_memory_GB=4)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([0.5, 0.1, 0.3]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.05,
                      alphaPIC=0.002, 
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=ti.Vector([0., 0., -9.8]))

dempm.dem.set_configuration(
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   0.401,
                      "SaveInterval":     0.02,
                      "SavePath":         'OutputData'
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 750,
                                "max_sphere_number": 0,
                                "max_clump_number": 6,
                                "max_plane_number": 6,
                                "comptaction_ratio": 0.8,
                                "verlet_distance_multiplier":  0.15,
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           200000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   12322
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    162,
                                  "wall_coordination_number":    3,
                                  "comptaction_ratio": 1.0
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            850,
                                "ForceLocalDamping":  0.25,
                                "TorqueLocalDamping": 0.7
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
'''
dempm.dem.add_clump_template(template={
                                 "Name": "clump1",
                                 "NSphere": 27,
                                 "Pebble": [{
                                             "Position": ti.Vector([-1.0, -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            
                                            {
                                             "Position": ti.Vector([-1.0, 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            
                                            
                                            {
                                             "Position": ti.Vector([-1.0, -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            
                                            {
                                             "Position": ti.Vector([-1.0, 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            
                                            
                                            {
                                             "Position": ti.Vector([-1.0, -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            
                                            {
                                             "Position": ti.Vector([-1.0, 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, 1.0]),
                                             "Radius": 0.5
                                            }]
                                 }
                      )
'''                           
dempm.dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 125,
                                 "Pebble": [{
                                             "Position": ti.Vector([-1.0, -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, -0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0.5, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, -1.0]),
                                             "Radius": 0.5
                                            },
                                            
                                            
                                            {
                                             "Position": ti.Vector([-1.0, -1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, -0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0., -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0., -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0.5, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, -0.5]),
                                             "Radius": 0.5
                                            },
                                            
                                            
                                            {
                                             "Position": ti.Vector([-1.0, -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, -0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0.5, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, 0.]),
                                             "Radius": 0.5
                                            },
                                            
                                            
                                            {
                                             "Position": ti.Vector([-1.0, -1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, -0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0., 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0., 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0.5, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, 0.5]),
                                             "Radius": 0.5
                                            },
                                            
                                            
                                            {
                                             "Position": ti.Vector([-1.0, -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, -0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, -0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., -0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, -0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, -0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0., 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 0.5, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-1.0, 1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([-0.5, 1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0., 1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 1.0, 1.0]),
                                             "Radius": 0.5
                                            },
                                            {
                                             "Position": ti.Vector([1.0, 1.0, 1.0]),
                                             "Radius": 0.5
                                            }]
                                 }
                      )

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "Clump",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.02, 0.015],
                                  "Radius": 1.0*1.7837811482282802/100,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.05, 0.015],
                                  "Radius": 1.0*1.7837811482282802/100,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.08, 0.015],
                                  "Radius": 1.0*1.7837811482282802/100,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.035, 0.045],
                                  "Radius": 1.0*1.7837811482282802/100,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.065, 0.045],
                                  "Radius": 1.0*1.7837811482282802/100,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.05, 0.075],
                                  "Radius": 1.0*1.7837811482282802/100,
                                  "BodyOrientation": "constant"
                                  }
                                ]})
                 
dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            2e4,
                            "TangentialStiffness":        1e6,
                            "Friction":                   0.18,
                            "NormalViscousDamping":       0.15,
                            "TangentialViscousDamping":   0.55
                           })           
                           
dempm.dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            2e4,
                            "TangentialStiffness":        1e6,
                            "Friction":                   0.7,
                            "NormalViscousDamping":       0.25,
                            "TangentialViscousDamping":   0.65
                           })  

                           
                           
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.05, 0.0]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.05, 0.3]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 0.05, 0.15]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.5, 0.05, 0.15]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
                  
dempm.dem.select_save_data(clump=True)

dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1350,
                               "YoungModulus":                  8e5,
                               "PoissionRatio":                 0.25,
                               "Friction":                      32,
                               "Dilation":                      0.0,
                               "Cohesion":                      0.0,
                               "Tensile":                       0.0
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.005, 0.005, 0.005])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.2]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   
dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0.5, 0.0, 0.3]
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0, 0.1, 0],
                                        "EndPoint":       [0.5, 0.1, 0.3]
                                    }])

dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            3e6,
                                 "TangentialStiffness":        8e6,
                                 "Friction":                   0.18,
                                 "NormalViscousDamping":       0.45,
                                 "TangentialViscousDamping":   0.45
                            })
                            
dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            5e4,
                                 "TangentialStiffness":        1e5,
                                 "Friction":                   0.7,
                                 "NormalViscousDamping":       0.3,
                                 "TangentialViscousDamping":   0.3
                            })
                            

dempm.run()

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
