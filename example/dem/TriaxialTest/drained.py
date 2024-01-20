from src import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.05, 0.05, 0.027]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 28482,
                                "max_sphere_number": 28482,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 6,
                                "max_facet_number": 12,
                                "body_coordination_number":   16,
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.1,
                                "wall_per_cell":              12
                            })  

dem.set_solver({
                "Timestep":         5e-7,
                "SimulationTime":   6,
                "SaveInterval":     0.25,
                "SavePath":         "Dense/CD"
               })              

dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.7
                            })
                          
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            4.25e4,
                            "TangentialStiffness":        4.25e4,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            4.25e4,
                            "TangentialStiffness":        4.25e4,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.read_restart(file_number=20, file_path="Dense/consolidation", particle=True, sphere=True, wall=True, servo=True, ppcontact=True, pwcontact=True, is_continue=False)
                  
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.servo_switch()

dem.scene.servo[0].active = 0
dem.scene.servo[1].active = 0
dem.scene.wall[0].v = ti.Vector([0.,0.,0.])
dem.scene.wall[1].v = ti.Vector([0.,0.,0.])
dem.scene.wall[2].v = ti.Vector([0.,0.,0.])
dem.scene.wall[3].v = ti.Vector([0.,0.,0.])

@ti.kernel
def chunk():
    ramp = dem.sims.time
    vel0 = 0.0005
    deltat = dem.sims.CurrentTime[None]
    vel = 0.
    if deltat < ramp:
        vel = (deltat / ramp) * vel0

    dem.scene.wall[0].v = ti.Vector([0.,0.,0.5*vel])
    dem.scene.wall[1].v = ti.Vector([0.,0.,0.5*vel])
    dem.scene.wall[2].v = ti.Vector([0.,0.,-0.5*vel])
    dem.scene.wall[3].v = ti.Vector([0.,0.,-0.5*vel])
    
    dem.sims.CurrentTime[None] += dem.sims.dt[None]


position = ti.Vector.field(3, float, 6, layout=ti.Layout.SOA)
force = ti.Vector.field(3, float, 6, layout=ti.Layout.SOA)
@ti.kernel
def get_gain():
    ti.loop_config(parallelize=16, block_dim=16)
    for tid in range(6):
        position[tid] = dem.scene.servo[tid].get_geometry_center(dem.scene.wall)
        force[tid] = dem.scene.servo[tid].get_geometry_force(dem.scene.wall)

    down_wall_position = position[0][2]
    up_wall_position = position[1][2]
    left_wall_position = position[2][0]
    right_wall_position = position[3][0]
    front_wall_position = position[4][1]
    back_wall_position = position[5][1]
    
    width = right_wall_position - left_wall_position
    depth = back_wall_position - front_wall_position
    height = up_wall_position - down_wall_position
    
    dem.scene.servo[0].update_area(width*depth)
    dem.scene.servo[1].update_area(width*depth)
    dem.scene.servo[2].update_area(height*depth)
    dem.scene.servo[3].update_area(height*depth)
    dem.scene.servo[4].update_area(width*height)
    dem.scene.servo[5].update_area(width*height)
    
    dem.scene.servo[2].update_current_force(-force[2][0])
    dem.scene.servo[3].update_current_force(force[3][0])
    dem.scene.servo[4].update_current_force(-force[4][1])
    dem.scene.servo[5].update_current_force(force[5][1])
    

dem.run(callback=get_gain, function=chunk)

dem.modify_parameters(SimulationTime=22.25, SaveInterval=0.25)

dem.run(callback=get_gain)

dem.postprocessing(read_path="Dense/CD", write_path="Dense/CD/vtks")
    
