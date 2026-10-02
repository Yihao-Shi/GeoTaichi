import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([15., 6., 5.]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., -6.929646456]),
                      search="LinkedCell",
                      track_energy=True)

dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_particle_number": 1,
                                "max_sphere_number": 1,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 0,
                                "max_facet_number": 2,
                                "body_coordination_number":   16,
                                "wall_coordination_number":   6,
                                "verlet_distance_multiplier": 0.1,
                                "compaction_ratio":   [1., 1.]
                            })    

dem.set_solver({
                "Timestep":         5e-4,
                "SimulationTime":   3.,
                "SaveInterval":     0.1
               })               

dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.
                            })
                           

dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0., 0., 0.]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([2.0, 3.0, 1.0-0.0031506888]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "Radius": 1.,
                               "BodyOrientation": "uniform"}]})
                          
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e8,
                            "TangentialStiffness":        1e8,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.2,
                            "TangentialViscousDamping":   0.2
                           })         
                           
dem.add_wall(body={
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0., 0., 0.]),
                                    "vertice2": ti.Vector([15., 0., 0.]),
                                    "vertice3": ti.Vector([15., 6., 0.]),
                                    "vertice4": ti.Vector([0., 6., 0.])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.])
                  })
                  
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.run()

#dem.contactor.physpw.surfaceProps[0].ndratio = 0.
dem.scene.material[0].fdamp = 0.
dem.sims.gravity = ti.Vector([6.929646456, 0., -6.929646456])

dem.modify_parameters(SimulationTime=5.,SaveInterval=0.1)

dem.run()

dem.postprocessing()
    
