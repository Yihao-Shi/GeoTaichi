import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu')

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.00036,0.00036,0.00055]),
                      boundary=["Period", "Period", None],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 13021,
                                "max_sphere_number": 13021,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 1,
                                "max_facet_number": 12,
                                "body_coordination_number":   32,
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.2,
                                "wall_per_cell":              12
                            })    

dem.set_solver({
                "Timestep":         9.8e-10,
                "SimulationTime":   1.6e-2,
                "SaveInterval":     1.6e-4,
                "SavePath":         "OutputData"
               })     

dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            7158,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
               
dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               5.95e10,
                            "Poisson":                    0.25,
                            "StaticFriction":             0.4,
                            "DynamicFriction":            0.42,
                            "Restitution":                0.5
                           },
                dType="particle-particle")
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            4.25e5,
                            "TangentialStiffness":        4.25e5,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           },
                dType="particle-wall")


'''
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            4.25e4,
                            "TangentialStiffness":        4.25e4,
                            "Friction":                   0.2,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            4.25e5,
                            "TangentialStiffness":        4.25e5,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
'''     
                           
dem.read_restart(file_number=40, file_path="Generation", particle=True, sphere=True, wall=True, servo=True, ppcontact=True, pwcontact=True, is_continue=False)
                           
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

def consol_ss():
    ti.loop_config(serialize=True)
    for i in range(1):
        force1 = dem.scene.servo[0].get_geometry_force(dem.scene.wall)
        dem.scene.servo[0].update_area(1.296e-7)
        dem.scene.servo[0].update_current_force(ti.abs(-force1[2]))
        
        
dem.servo_switch(status="GainControl")

dem.scene.servo[0].gain = 2.
dem.scene.servo[0].max_velocity = 0.1

dem.run(callback=consol_ss)

dem.postprocessing(read_path="OutputData", write_path="OutputData")
    
