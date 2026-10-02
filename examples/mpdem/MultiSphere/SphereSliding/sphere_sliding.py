import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(debug=True)

dempm = DEMPM()

dempm.set_configuration(domain=[15., 6., 5.],
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=False,
                        enhanced_coupling=True)

dempm.dem.set_configuration(
                      gravity=ti.Vector([0., 0., -6.929646456]),
                      engine="SymplecticEuler",
                      search="LinkedCell")
                      
       
dempm.mpm.set_configuration(
                      background_damping=0., 
                      alphaPIC=0.005, 
                      mapping="USF", 
                      shape_function="Linear",
                      gravity=ti.Vector([6.929646456, 0., -6.929646456]))

dempm.set_solver({
                      "Timestep":         5e-05,
                      "SimulationTime":   3.,
                      "SaveInterval":     0.1,
                      "SavePath":         "OutputData/mu=0.1"
                 })
                             
dempm.dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_particle_number": 1,
                                "max_sphere_number": 1,
                                "max_clump_number": 0,
                                "verlet_distance_multiplier": 0.
                            })            

dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           33552
                            })

dempm.memory_allocate(memory={
                                  "body_coordination_number":    80,
                                  "wall_coordination_number":    6,
                             })


dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": [0., 0., 0.],
                               "InitialAngularVelocity": [0., 0., 0.],
                               "BodyPoint": [2.0, 3.0, 2.6],
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "Radius": 1.6,
                               "BodyOrientation": "uniform"}]})
 
                  
dempm.dem.select_save_data()

dempm.mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         1e5,
                               "PoissionRatio":        0.3
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               [0.5, 0.5, 0.5]
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0., 0., 0.],
                            "BoundingBoxSize": [15., 6., 1.],
                            
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  1,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":   [0, 0, 0],
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

dempm.mpm.select_save_data()


dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            1e6,
                                 "TangentialStiffness":        1e6,
                                 "Friction":                   0.,
                                 "NormalViscousDamping":       0.02,
                                 "TangentialViscousDamping":   0.
                            })

dempm.run()

dempm.contactor.physpp.surfaceProps[1].ndratio = 0.
dempm.contactor.physpp.surfaceProps[1].mu = 0.1
dempm.dem.scene.material[0].fdamp = 0.

dempm.modify_parameters(SimulationTime=5)

dempm.dem.sims.set_gravity([6.929646456, 0., -6.929646456])

dempm.run()

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
