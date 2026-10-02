import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=4)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([10.2, 0.3, 12.2]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.02,
                      alphaPIC=0.002, 
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=ti.Vector([0., 0., 0.]),
                      stabilize="B-Bar Method")

dempm.dem.set_configuration(
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")

dempm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   5,
                      "SaveInterval":     0.25,
                      "SavePath":         'OutputData'
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_rigid_body_number": 1,
                                "levelset_grid_number": 3375,
                                "surface_node_number": 272,
                                "max_plane_number": 0,
                                "body_coordination_number":   1,
                                "wall_coordination_number":   1,
                                "verlet_distance_multiplier": [0.1, 0.2],
                                "point_coordination_number":  [1, 1], 
                                "compaction_ratio":           [0.3, 0.3, 0.15, 0.15],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    652800,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     103716
                                                          },
                                "verlet_distance_multiplier":    1.
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    1,
                                  "wall_coordination_number":    1,
                                  "compaction_ratio": [0.2, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            850,
                                "ForceLocalDamping":  0.15,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/box.stl').grids(space=0.1, extent=2),
                                "WriteFile":          True}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.6, 0.15, 10.6],
                                  "ScaleFactor": 1.0,
                                  "BodyOrientation": "constant",
                                  "InitialVelocity": [0., 0., 0.],
                                  "FixMotion":  ["Fix", "Fix", "Fix"]
                                  }
                                ]})
                 
dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                         particle_wall_contact_model=None)
                  
dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "YoungModulus":         1e5,
                               "PoissonRatio":         0.49,
                               "Cohesion":             1000,
                               "Friction":             0.,
                               "Dilation":             0.
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.1, 0.1, 0.1]),
                            "BoundingBoxSize": ti.Vector([10., 0.1, 10.]),
                            
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  4,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "Traction":       [],
                                       "InitialVelocity":ti.Vector([0., 0., 0.]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   
dempm.mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [10.2, 0.3, 0.1],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [0., None, None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.1, 0.3, 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [0., None, None],
                                             "StartPoint":     [10.1, 0., 0.],
                                             "EndPoint":       [10.2, 0.3, 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [None, 0., None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [10.2, 0.1, 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [None, 0., None],
                                             "StartPoint":     [0., 0.2, 0.],
                                             "EndPoint":       [10.2, 0.3, 12.2]
                                        }
                                    ])

dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            1e5,
                                 "TangentialStiffness":        8e4,
                                 "Friction":                   0.4,
                                 "NormalViscousDamping":       0.2,
                                 "TangentialViscousDamping":   0.15
                            })

def stepwise():
    ramp = dempm.sims.time
    vel0 = 0.1
    deltat = dempm.sims.CurrentTime[None]
    vel = 0.
    dempm.dem.scene.rigid[0].v=[0., 0., -(deltat / ramp) * vel0]
    dempm.sims.CurrentTime[None] += dempm.sims.dt[None]

dempm.run(function=stepwise)

dempm.modify_parameters(SimulationTime=17.5, SaveInterval=0.25)

dempm.run()

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
