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
                                "verlet_distance_multiplier": 0.2,
                                "wall_per_cell":              12
                            })    

dem.set_solver({
                "Timestep":         5e-7,
                "SimulationTime":   28.25,
                "SaveInterval":     0.25,
                "SavePath":         "Medium/CU"
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
                           
dem.read_restart(file_number=25, file_path="Medium/consolidation", particle=True, sphere=True, wall=True, servo=True, ppcontact=True, pwcontact=True, is_continue=False)
                  
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.servo_switch(status="Off")

dem.scene.servo[0].active = 0
dem.scene.servo[1].active = 0
dem.scene.servo[2].active = 0
dem.scene.servo[3].active = 0
dem.scene.servo[4].active = 0
dem.scene.servo[5].active = 0

dem.scene.wall[0].v = ti.Vector([0.,0.,0.])
dem.scene.wall[1].v = ti.Vector([0.,0.,0.])
dem.scene.wall[2].v = ti.Vector([0.,0.,0.])
dem.scene.wall[3].v = ti.Vector([0.,0.,0.])
dem.scene.wall[4].v = ti.Vector([0.,0.,0.])
dem.scene.wall[5].v = ti.Vector([0.,0.,0.])
dem.scene.wall[6].v = ti.Vector([0.,0.,0.])
dem.scene.wall[7].v = ti.Vector([0.,0.,0.])
dem.scene.wall[8].v = ti.Vector([0.,0.,0.])
dem.scene.wall[9].v = ti.Vector([0.,0.,0.])
dem.scene.wall[10].v = ti.Vector([0.,0.,0.])
dem.scene.wall[11].v = ti.Vector([0.,0.,0.])

position = ti.Vector.field(3, float, 6, layout=ti.Layout.SOA)
@ti.kernel
def initial_fill():
    ti.loop_config(parallelize=16, block_dim=16)
    for tid in range(6):
        position[tid] = dem.scene.servo[tid].get_geometry_center(dem.scene.wall)
initial_fill()

down_wall_position = position[0][2]
up_wall_position = position[1][2]
left_wall_position = position[2][0]
right_wall_position = position[3][0]
front_wall_position = position[4][1]
back_wall_position = position[5][1]

width = right_wall_position - left_wall_position
depth = back_wall_position - front_wall_position
height = up_wall_position - down_wall_position
Init_boxVolume = width * depth * height


@ti.kernel
def chunk():
    ramp = 6
    vel0 = 0.0005
    deltat = dem.sims.CurrentTime[None]
    vel = vel0
    if deltat < ramp:
        vel = (deltat / ramp) * vel0
    
    dem.scene.wall[0].v = ti.Vector([0.,0.,0.5*vel])
    dem.scene.wall[1].v = ti.Vector([0.,0.,0.5*vel])
    dem.scene.wall[2].v = ti.Vector([0.,0.,-0.5*vel])
    dem.scene.wall[3].v = ti.Vector([0.,0.,-0.5*vel])
    dem.sims.CurrentTime[None] += dem.sims.dt[None]


    ti.loop_config(parallelize=16, block_dim=16)
    for tid in range(6):
        position[tid] = dem.scene.servo[tid].get_geometry_center(dem.scene.wall)

    down_wall_position = position[0][2]
    up_wall_position = position[1][2]
    left_wall_position = position[2][0]
    right_wall_position = position[3][0]
    front_wall_position = position[4][1]
    back_wall_position = position[5][1]

    width = right_wall_position - left_wall_position
    depth = back_wall_position - front_wall_position
    height = up_wall_position - down_wall_position

    S = Init_boxVolume / height
    delta_d = 0.5 * (S / width - depth)
    delta_w = S / (depth + delta_d) - width

    dem.scene.servo[2].move(-0.5 * delta_w, dem.scene.wall)
    dem.scene.servo[3].move(0.5 * delta_w, dem.scene.wall)
    dem.scene.servo[4].move(-0.5 * delta_d, dem.scene.wall)
    dem.scene.servo[5].move(0.5 * delta_d, dem.scene.wall)


dem.run(function=chunk)

dem.postprocessing(read_path="Medium/CU")
    
