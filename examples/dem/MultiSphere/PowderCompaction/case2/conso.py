import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(debug=False)

dem = DEM()

dem.set_configuration(domain=[0.00036,0.00036,0.00054],
                      boundary=["Period", "Period", None],
                      gravity=[0.,0.,0.],
                      engine="SymplecticEuler",
                      search="HierarchicalLinkedCell",
                      search_direction="Up")

dem.set_solver({
                "Timestep":         1.7e-10,
                "SimulationTime":   1.4e-4,
                "SaveInterval":     1e-5,
                "SavePath":         "OutputData"
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_particle_number": 14000000,
                            "max_sphere_number": 14000000,
                            "max_clump_number": 0,
                            "max_facet_number": 4,
                            "max_servo_wall_number": 1,
                            "verlet_distance_multiplier": 0.1,
                            "body_coordination_number": [26, 24],
                            "wall_coordination_number": 2,
                            "hierarchical_level": 2,
                            "hierarchical_size": [0.6e-06, 8.6e-06],
                            "compaction_ratio": [0.2, 0.05],
                            "wall_per_cell": [2, 2]
                            }, log=True) 

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            7158,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })
                            
dem.add_attribute(materialID=1,
                  attribute={
                            "Density":            26500,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })
                           
                           
dem.read_restart(file_number=10, file_path="OutputData", particle=True, sphere=True, wall=True, servo=True, ppcontact=True, pwcontact=True, is_continue=True)

dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Hertz Mindlin Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               5.95e10,
                            "Poisson":                    0.25,
                            "StaticFriction":             0.4,
                            "DynamicFriction":            0.42,
                            "Restitution":                0.5
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "ShearModulus":               5.95e10,
                            "Poisson":                    0.25,
                            "StaticFriction":             0.0,
                            "DynamicFriction":            0.0,
                            "Restitution":                0.5
                           })
                           
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=False, particle_wall_contact=False)


def consol_ss():
    ti.loop_config(serialize=True)
    for i in range(1):
        force = dem.scene.servo[0].get_geometry_force(dem.scene.wall)
        dem.scene.servo[0].update_area(1.296e-7)
        dem.scene.servo[0].update_current_force(ti.abs(-force[2]))
        #print(ti.abs(-force[2])/1.296e-7)
        
dem.servo_switch(status="GainControl")

# dem.scene.servo[0].gain = 10.


dem.run(callback=consol_ss)

dem.postprocessing(read_path="OutputData", write_path="OutputData/vtks")
    
