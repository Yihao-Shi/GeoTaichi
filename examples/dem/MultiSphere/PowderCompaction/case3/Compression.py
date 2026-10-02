import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


load_path = "NonInteraction.txt"

save_path = "DEM_Compression"

from geotaichi import *

def box_wall(box_point=[3.35e-6, 3.35e-6, 3.35e-6], box_size=[2 * 93.3e-6, 2 * 93.3e-6, 2 * 93.3e-6], servo_stress=1e6,
             expand=1.2,
             limitVelocity = 0.1, 
             servo_fac=1):
    
    x0, y0, z0 = box_point
    dx, dy, dz = box_size
    ex = expand - 1.0
    dem.add_wall(body=[
        {
            "WallID": 0,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": [x0 - dx*ex, y0 - dy*ex, z0],
                "vertice2": [x0 + dx*(1 + ex), y0 - dy*ex, z0],
                "vertice3": [x0 + dx*(1 + ex), y0 + dy*(1 + ex), z0],
                "vertice4": [x0 - dx*ex, y0 + dy*(1 + ex), z0]
            },
            "OuterNormal": [0., 0., 1.], 
        },

        {
            "WallID": 1,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": [x0 - dx*ex, y0 - dy*ex, z0 + dz],
                "vertice2": [x0 + dx*(1 + ex), y0 - dy*ex, z0 + dz],
                "vertice3": [x0 + dx*(1 + ex), y0 + dy*(1 + ex), z0 + dz],
                "vertice4": [x0 - dx*ex, y0 + dy*(1 + ex), z0 + dz]
            },
            "OuterNormal": [0., 0., -1.],
            "ControlType":  "Force",
            "TargetStress": servo_stress,
            "Alpha":    servo_fac,
            "LimitVelocity": limitVelocity
        },

        {
            "WallID": 2,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": [x0, y0 - dy*ex, z0 - dz*ex],
                "vertice2": [x0, y0 + dy*(1 + ex), z0 - dz*ex],
                "vertice3": [x0, y0 + dy*(1 + ex), z0 + dz*(1 + ex)],
                "vertice4": [x0, y0 - dy*ex, z0 + dz*(1 + ex)],
            },
            "OuterNormal": [1., 0., 0.],
        },

        {
            "WallID": 3,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": [x0 + dx, y0 - dy*ex, z0 - dz*ex],
                "vertice2": [x0 + dx, y0 + dy*(1 + ex), z0 - dz*ex],
                "vertice3": [x0 + dx, y0 + dy*(1 + ex), z0 + dz*(1 + ex)],
                "vertice4": [x0 + dx, y0 - dy*ex, z0 + dz*(1 + ex)],
            },
            "OuterNormal": [-1., 0., 0.],  
        },

        {
            "WallID": 4,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": [x0 - dx*ex, y0, z0 - dz*ex],
                "vertice2": [x0 + dx*(1 + ex), y0, z0 - dz*ex],
                "vertice3": [x0 + dx*(1 + ex), y0, z0 + dz*(1 + ex)],
                "vertice4": [x0 - dx*ex, y0, z0 + dz*(1 + ex)],
            },
            "OuterNormal": [0., 1., 0.],
        },

        {
            "WallID": 5,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": [x0 - dx*ex, y0 + dy, z0 - dz*ex],
                "vertice2": [x0 + dx*(1 + ex), y0 + dy, z0 - dz*ex],
                "vertice3": [x0 + dx*(1 + ex), y0 + dy, z0 + dz*(1 + ex)],
                "vertice4": [x0 - dx*ex, y0 + dy, z0 + dz*(1 + ex)],
            },
            "OuterNormal": [0., -1., 0.],
        },
    ])

init(arch='gpu', log=True, debug=False, device_memory_GB=3, kernel_profiler=False)

dem = DEM()

dem.set_configuration(domain=[2 * 100.e-6, 2 * 100.e-6, 2 * 100.e-6],
                      boundary=[None, None, None],
                      gravity=[0., 0., 0.],
                      engine="SymplecticEuler",
                      search="HierarchicalLinkedCell", 
                      visualize=True)

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 1818870,
                                "max_sphere_number": 1818870,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 1,
                                "max_facet_number": 12,
                                "hierarchical_level": 2,
                                "hierarchical_size": [1.675102e-06, 1.2896e-05],
                                "body_coordination_number":   [30,50],
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.15,
                                "wall_per_cell":              12, 
                                "compaction_ratio":           [0.15, 0.1]
                            })



dem.set_solver({
                "Timestep":         3e-10,
                "CFL":              4.0,
                "SimulationTime":   0.002,
                "SaveInterval":     0.0002,
                "SavePath":         save_path
               })  

dem.add_attribute(materialID=0,
                    attribute={
                                "Density":            715800,
                                "ForceLocalDamping":  0.0,
                                "TorqueLocalDamping": 0.0
                            })

dem.add_attribute(materialID=1,
                    attribute={
                                "Density":            8000,
                                "ForceLocalDamping":  0.0,
                                "TorqueLocalDamping": 0.0
                            })

dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": load_path,
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": [0.,0.,0.],
                               "InitialAngularVelocity": [0.,0.,0.],
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})

box_wall()

dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                        #    particle_wall_contact_model="Hertz Mindlin Model")
                           particle_wall_contact_model="Linear Model")

   
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               5.95e10,
                            "Poisson":                    0.25,
                            "Friction":                   0.2,
                            "RollingFriction":            0.421,
                            "Restitution":                0.5
                           },
                dType="particle-particle")
                            
# dem.add_property(materialID1=0,
#                    materialID2=1,
#                    property={
#                                 "ShearModulus":               5.95e11,
#                                 "Poisson":                    0.30,
#                                 "Friction":                   0.01,
#                                 "RollingFriction":            0.01,
#                                 "Restitution":                0.5
#                             },
#                 dType="particle-wall")  
    

dem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "EffectiveModulus":           5.95e+10,
                                "NormalToShearRatio":         1.0,
                                "Friction":                   0.01,
                                "NormalViscousDamping":       0.00,
                                "TangentialViscousDamping":   0.00
                            },
                dType="particle-wall")

dem.select_save_data(particle=True, sphere=True, wall=True, particle_particle_contact = True, particle_wall_contact = True)
                
def consol_ss():
    ti.loop_config(serialize=True)
    force = dem.scene.servo[0].get_geometry_force(dem.scene.wall)
    dem.scene.servo[0].update_area(3.481956e-8)
    dem.scene.servo[0].update_current_force(ti.abs(-force[2]))

def postprocess():
    if dem.sims.current_step%3330==0: 
        @ti.kernel
        def _max_vel_(particleNum: int, particle: ti.template()) -> float:
            _max_vel = 0.
            for np in range(particleNum):
                vel = particle[np].v
                ti.atomic_max(_max_vel, vel.norm())
            return _max_vel
        max_vel = _max_vel_(dem.scene.particleNum[0], dem.scene.particle)
        total_force = dem.scene.servo[0].current_force
        top_pos = dem.scene.wall[2].vertice1[2]
        with open('time_series.txt', 'ab') as file:
            np.savetxt(file, np.array([dem.sims.current_time, top_pos, max_vel, total_force]).reshape(1,-1), delimiter=" ")
          
dem.servo_switch(status="On")

dem.run(callback=consol_ss, function=postprocess)
