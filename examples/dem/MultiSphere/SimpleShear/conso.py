import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.9, 0.6, 0.3]),
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 78369,
                                "max_sphere_number": 78369,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 1,
                                "max_facet_number": 24,
                                "body_coordination_number":   28,
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.1,
                                "wall_per_cell":              12
                            })    

dem.set_solver({
                "Timestep":         1e-5,
                "SimulationTime":   2.,
                "SaveInterval":     0.1,
                "SavePath":         "50kPa/consolidation"
               })               

dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2580,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.7
                            })
                            
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            8e4,
                            "TangentialStiffness":        5.4e4,
                            "Friction":                   0.5,
                            "RollingFriction":            0.5, 
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            2e5,
                            "TangentialStiffness":        1.3e5,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
                           
dem.read_restart(file_number=20, file_path="Generation", particle=True, sphere=True, wall=True, servo=True, ppcontact=True, pwcontact=True, is_continue=False)
                           
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

def consol_ss():
    ti.loop_config(serialize=True)
    for tid in range(1):
        force = dem.scene.servo[tid].get_geometry_force(dem.scene.wall)
        dem.scene.servo[tid].update_area(0.09)
        dem.scene.servo[tid].update_current_force(ti.abs(-force[2]))

dem.servo_switch()

dem.run(callback=consol_ss)

dem.postprocessing(read_path="50kPa/consolidation", write_path="50kPa/consolidation/vtks")
    
