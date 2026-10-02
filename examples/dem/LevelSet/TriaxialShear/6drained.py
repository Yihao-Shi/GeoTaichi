import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


def read_Z0(read_num):
    data = np.load(load_path+'/walls/DEMWall{0:06d}.npz'.format(read_num), allow_pickle=True)
    down_position = (data["point1"][0][2]+data["point2"][0][2]+data["point3"][0][2]+data["point1"][1][2]+data["point2"][1][2]+data["point3"][1][2])/6.
    up_position = (data["point1"][2][2]+data["point2"][2][2]+data["point3"][2][2]+data["point1"][3][2]+data["point2"][3][2]+data["point3"][3][2])/6.
    print(up_position-down_position)
    return up_position-down_position

scale_fac0 = 1.001891414401244518
scale_fac1 = 1.247387418619341659
scale_fac2 = 1.554954183054590544

scale_fac = scale_fac1

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Run the drained LSDEM triaxial stage from a restart.")
parser.add_argument("--template-mesh", required=True, help="STL mesh for Template1.")
parser.add_argument("--restart-dir", default=str(CASE_DIR / "OutputData" / "Generation_last"))
parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData" / "Drained_test"))
parser.add_argument("--restart-frame", type=int, default=10)
arguments = parser.parse_args()
load_path = str(Path(arguments.restart_dir).expanduser().resolve())
save_path = str(Path(arguments.output_dir).expanduser().resolve())
template_mesh = str(Path(arguments.template_mesh).expanduser().resolve())
read_num = arguments.restart_frame

z0 =read_Z0(read_num)
I = 1 # 应变率

vel = I * z0



from geotaichi import *

init(arch='gpu', log=True, debug=False, device_memory_GB=27, kernel_profiler=False)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([0.1, 0.1, 0.1]),
                        boundary=["Destroy", "Destroy", "Destroy"],
                        gravity=ti.Vector([0., 0., 0.]),
                        scheme="LSDEM",
                        search="HierarchicalLinkedCell",
                        visualize=False)

lsdem.set_solver({
                "Timestep":         1.e-7,
                "CFL":              0.5,
                "SimulationTime":   0.250,
                "SaveInterval":     0.005,
                "SavePath":         save_path
               })  

lsdem.memory_allocate(memory={
                                 "max_material_number":        2,
                                 "max_rigid_body_number":      13310,
                                 "levelset_grid_number":       54567,
                                 "surface_node_number":        1502,
                                 "max_facet_number":           12,
                                 "max_servo_wall_number":      6,
                                 "hierarchical_level": 2,
                                 "hierarchical_size": [0.00035*scale_fac, 0.00024*scale_fac],
                                 "body_coordination_number":   [40, 45],
                                 "wall_coordination_number":   6,
                                 'wall_per_cell':              12,
                                 "verlet_distance_multiplier": [0.15, 0.1],
                                 "point_coordination_number":  [4, 4], 
                                 "compaction_ratio":           [0.3, 0.1, 0.2, 0.1]
                             })  


lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.7
                            })

lsdem.add_template(template={
                                "Name": "Template1",
                                "Object": polyhedron(file=template_mesh).grids(space=8, extent=3),
                                "WriteFile": True})   

lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")


lsdem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "EffectiveModulus":           1.e10,
                                "NormalToShearRatio":         4/3,
                                "Friction":                   0.65,
                                "NormalViscousDamping":       0.00,
                                "TangentialViscousDamping":   0.00
                            })   
                            
lsdem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "EffectiveModulus":           1.e11,
                                "NormalToShearRatio":         1.0,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.00,
                                "TangentialViscousDamping":   0.00
                            })   

lsdem.read_restart(file_number=read_num, file_path=load_path, 
                   particle=True, wall=True, servo=True, ppcontact=True, pwcontact=True, is_continue=False)

lsdem.scene.servo[0].active = 0
lsdem.scene.servo[1].active = 0

lsdem.scene.servo[2].alpha = 1.0
lsdem.scene.servo[3].alpha = 1.0
lsdem.scene.servo[4].alpha = 1.0
lsdem.scene.servo[5].alpha = 1.0

lsdem.scene.wall[0].v = ti.Vector([0.,0.,0.])
lsdem.scene.wall[1].v = ti.Vector([0.,0.,0.])
lsdem.scene.wall[2].v = ti.Vector([0.,0.,0.])
lsdem.scene.wall[3].v = ti.Vector([0.,0.,0.])

def load():
    lsdem.scene.wall[0].v = ti.Vector([0.,0.,0.5*vel])
    lsdem.scene.wall[1].v = ti.Vector([0.,0.,0.5*vel])
    lsdem.scene.wall[2].v = ti.Vector([0.,0.,-0.5*vel])
    lsdem.scene.wall[3].v = ti.Vector([0.,0.,-0.5*vel])

load()

position = ti.Vector.field(3, float, 6, layout=ti.Layout.SOA)
force = ti.Vector.field(3, float, 6, layout=ti.Layout.SOA)

def get_gain():
    ti.loop_config(parallelize=16, block_dim=16)
    for tid in range(6):
        position[tid] = lsdem.scene.servo[tid].get_geometry_center(lsdem.scene.wall)
        force[tid] = lsdem.scene.servo[tid].get_geometry_force(lsdem.scene.wall)

    down_wall_position = position[0][2]
    up_wall_position = position[1][2]
    left_wall_position = position[2][0]
    right_wall_position = position[3][0]
    front_wall_position = position[4][1]
    back_wall_position = position[5][1]
    
    width = right_wall_position - left_wall_position
    depth = back_wall_position - front_wall_position
    height = up_wall_position - down_wall_position
    
    lsdem.scene.servo[0].update_area(width*depth)
    lsdem.scene.servo[1].update_area(width*depth)
    lsdem.scene.servo[2].update_area(height*depth)
    lsdem.scene.servo[3].update_area(height*depth)
    lsdem.scene.servo[4].update_area(width*height)
    lsdem.scene.servo[5].update_area(width*height)
    
    lsdem.scene.servo[2].update_current_force(-force[2][0])
    lsdem.scene.servo[3].update_current_force(force[3][0])
    lsdem.scene.servo[4].update_current_force(-force[4][1])
    lsdem.scene.servo[5].update_current_force(force[5][1])

lsdem.select_save_data(particle=True, surface=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

lsdem.servo_switch(status='On')

lsdem.run(callback=get_gain)
