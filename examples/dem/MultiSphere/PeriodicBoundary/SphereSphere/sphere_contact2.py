import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='cpu')

dem = DEM()

dem.set_configuration(domain=ti.Vector([15., 6., 5.]),
                      boundary=["Period", "Period", "Period"],
                      gravity=[0., 0., 0.],
                      search="HierarchicalLinkedCell",
                      track_energy=True)

dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_particle_number": 2,
                                "max_sphere_number": 2,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 0,
                                "max_facet_number": 0,
                            "body_coordination_number": [32, 150],
                            "wall_coordination_number": 8,
                            "hierarchical_level": 2,
                            "hierarchical_size": [0.11, 1.1],
                            "compaction_ratio": [1., 1.],
                            })    

dem.set_solver({
                "Timestep":         5e-4,
                "SimulationTime":   10.,
                "SaveInterval":     0.1,
                "SavePath":         "OutputData2"
               })               

dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
                           

dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([3., 0., 0.]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([13.0, 3.0, 1.0]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "Radius": 1.,
                               "BodyOrientation": "uniform"},
                               {
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([-3., 0., 0.]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([2.0, 3.0, 1.0]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "Radius": 0.1,
                               "BodyOrientation": "uniform"},
                               ]})
                          
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e8,
                            "TangentialStiffness":        1e8,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.,
                            "TangentialViscousDamping":   0.
                           })         
                  
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.run()
