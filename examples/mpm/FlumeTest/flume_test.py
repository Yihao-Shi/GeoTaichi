import os
import sys

import numpy as np
from math import sin, cos, pi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', type=bool, default=False)
parser.add_argument('-f', type=int, default=10)
args = parser.parse_args()

from geotaichi import *

init(device_memory_GB=6.8, debug=False)

trans = pi / 180.

start_leng = 2.
delta = 0.1

flume_leng = 2.3
flume_wid = 0.15
flume_heig = 0.2
theta = 40.
theta *= trans

spec_leng = 0.18
spec_heig = 0.18
spec_wid = 0.15

restart = args.r

dempm = DEMPM()

dempm.set_configuration(domain=[3., 1.0, 1.7],
                        coupling_scheme="MPDEM",
                        particle_interaction=False,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.0,
                      alphaPIC=0.001,
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      gravity=[0., 0., -9.8],
                      material_type="Solid",
                      #velocity_projection="Affine",
                      sparse_grid=True
                      )

dempm.dem.set_configuration(
                      gravity=[0., 0., -9.8],
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="DEM")
                      
dempm.set_solver({
                      "Timestep":         1e-4,
                      "SimulationTime":   0.5,
                      "CFL":              0.5,
                      "SaveInterval":     0.05
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_particle_number": 0,
                                "max_facet_number": 10,
                                "body_coordination_number":   0,
                                "wall_coordination_number":   0,
                                "verlet_distance_multiplier": 0.15,
                                "compaction_ratio":           [0.3, 0.3],
                                "wall_per_cell":              9
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           2400000,
                                "verlet_distance_multiplier":    0.4,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   1042205,
                                                               "max_friction_constraint":   0,
                                                               
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    0,
                                  "wall_coordination_number":    8,
                                  "compaction_ratio": [0.02, 0.3]
                             })  
                             
dempm.dem.add_wall(body={
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0., 0.425, flume_leng*sin(theta)+delta],
                                    "vertice2": [flume_leng*cos(theta), 0.425, 0.+delta],
                                    "vertice3": [flume_leng*cos(theta)+flume_heig*cos(0.5*pi-theta), 0.425, flume_heig*sin(0.5*pi-theta)+delta],
                                    "vertice4": [flume_heig*cos(0.5*pi-theta), 0.425, flume_leng*sin(theta)+flume_heig*sin(0.5*pi-theta)+delta]
                                   },
                   "OuterNormal": [0., 1., 0.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0., 0.575, flume_leng*sin(theta)+delta],
                                    "vertice2": [flume_leng*cos(theta), 0.575, 0.+delta],
                                    "vertice3": [flume_leng*cos(theta)+flume_heig*cos(0.5*pi-theta), 0.575, flume_heig*sin(0.5*pi-theta)+delta],
                                    "vertice4": [flume_heig*cos(0.5*pi-theta), 0.575, flume_leng*sin(theta)+flume_heig*sin(0.5*pi-theta)+delta]
                                   },
                   "OuterNormal": [0., -1., 0.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0., 0.415, flume_leng*sin(theta)+delta],
                                    "vertice2": [flume_leng*cos(theta), 0.415, 0.005+delta],
                                    "vertice3": [flume_leng*cos(theta), 0.585, 0.005+delta],
                                    "vertice4": [0., 0.585, flume_leng*sin(theta)+delta]
                                   },
                   "OuterNormal": [cos(0.5*pi-theta), 0., sin(0.5*pi-theta)]})
                   
dempm.dem.add_wall(body={
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [flume_leng*cos(theta)-0.5, 0., 0.+delta],
                                    "vertice2": [3., 0., 0.+delta],
                                    "vertice3": [3., 1.0, 0.+delta],
                                    "vertice4": [flume_leng*cos(theta)-0.5, 1.0, 0.+delta]
                                   },
                   "OuterNormal": [0., 0., 1.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [(flume_leng-start_leng)*cos(theta), 0.425, start_leng*sin(theta)+delta],
                                    "vertice2": [(flume_leng-start_leng)*cos(theta) + flume_heig*sin(theta), 0.425, start_leng*sin(theta)+flume_heig*cos(theta)+delta],
                                    "vertice3": [(flume_leng-start_leng)*cos(theta) + flume_heig*sin(theta), 0.575, start_leng*sin(theta)+flume_heig*cos(theta)+delta],
                                    "vertice4": [(flume_leng-start_leng)*cos(theta), 0.575, start_leng*sin(theta)+delta]
                                   },
                   "OuterNormal": [-cos(theta), 0., sin(theta)]})
                   
dempm.dem.set_static_wall()
                   
dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                         particle_wall_contact_model=None)  
                                             
dempm.dem.select_save_data(wall=True)

dempm.mpm.add_material(model="GranularMaterial",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1650,
                               "GrainDensity":                  2500,
                               "YoungModulus":                  1e7,
                               "PoissionRatio":                 0.3,
                               "StaticFriction":                25,
                               "DynamicFriction":               39,
                               "AverageDiameter":               0.0015,
                               "InertialNumber":                0.7,
                               "eps":                           0.01
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               [0.005, 0.005, 0.005]
                        })

Bx = (flume_leng - start_leng - 0.5 * spec_leng) * np.cos(theta) + 0.5 * spec_heig * np.sin(theta) - 0.5 * spec_leng
By = 0.425
Bz = (start_leng + 0.5 * spec_leng) * np.sin(theta) + 0.5 * 1.1 * spec_heig * np.cos(theta) - 0.5 * spec_heig + delta

bbox_point = np.array([Bx, By, Bz])
bbox_size  = np.array([spec_leng, spec_wid, spec_heig])
rotate     = np.array([0., 40., 0.])

if restart:
    dempm.mpm.read_restart(args.f, "OutputData", True)
else:
    dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": bbox_point,
                            "BoundingBoxSize": bbox_size,
                            "rotate":         rotate,
                            
                      }])

    dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [3.0, 0.0, 1.7],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0, 1.0, 0],
                                        "EndPoint":       [3.0, 1.0, 1.7],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., None, None],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 1.0, 1.7],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., None, None],
                                        "StartPoint":     [3.0, 0.0, 0],
                                        "EndPoint":       [3.0, 1.0, 1.7],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, None, 0.],
                                        "StartPoint":     [0., 0.0, 1.7],
                                        "EndPoint":       [3.0, 1.0, 1.7],
                                    }])

dempm.mpm.select_save_data()

if restart:
    dempm.read_restart(args.f, "OutputData", False, True)

dempm.choose_contact_model(particle_particle_contact_model=None,
                           particle_wall_contact_model="Linear Model")

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            1e4,
                                 "TangentialStiffness":        1e4,
                                 "Friction":                   0.48,
                                 "NormalViscousDamping":       0.5,
                                 "TangentialViscousDamping":   0.5
                            }, dType='particle-wall')

dempm.select_save_data(particle_wall_contact=True)

if restart:
    dempm.modify_parameters(SimulationTime=3.5)
    dempm.dem.delete_walls(4)
    dempm.run()
else:
    def vertical_distance_to_rotated_box(
        point_world,
        bbox_point, 
        bbox_size,
        rotate_deg 
    ):
        point_world = np.asarray(point_world, dtype=float)
        bbox_point = np.asarray(bbox_point, dtype=float)
        bbox_size = np.asarray(bbox_size, dtype=float)
        rotate_deg = np.asarray(rotate_deg, dtype=float)

        C_world = bbox_point + 0.5 * bbox_size
        hx, hy, hz = 0.5 * bbox_size

        theta_deg = rotate_deg[1]
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[ cos_t, 0.0,  sin_t],
                    [ 0.0,   1.0,  0.0  ],
                    [-sin_t, 0.0,  cos_t]])

        P_local = R.T @ (point_world - C_world)
        d_world = np.array([0.0, 0.0, 1.0])
        d_local = R.T @ d_world
        
        d_local_norm = np.linalg.norm(d_local)
        if d_local_norm == 0.0:
            return np.inf
        d_local = d_local / d_local_norm

        tmin, tmax = -np.inf, np.inf
        origin = P_local
        direction = d_local

        for i, (o, d, h) in enumerate(zip(origin, direction, [hx, hy, hz])):
            if abs(d) < 1e-8:
                if not (-h <= o <= h):
                    return np.inf
            else:
                t1 = (-h - o) / d
                t2 = ( h - o) / d
                t_near = min(t1, t2)
                t_far  = max(t1, t2)
                tmin = max(tmin, t_near)
                tmax = min(tmax, t_far)
                if tmin > tmax:
                    return np.inf

        if tmax < 0:
            return np.inf

        t_hit = tmin if tmin >= 0 else tmax
        if t_hit < 0:
            return np.inf

        return float(t_hit)
    
    def get_gravity(pos):
        points = np.asarray(pos)
        dists = np.array([
            vertical_distance_to_rotated_box(p, bbox_point, bbox_size, rotate)
            for p in points
        ])
        print(dists, np.isnan(dists))
        return dists


    dempm.run(mpm_gravity_field=get_gravity)

    dempm.modify_parameters(SimulationTime=3.5)
    dempm.dem.delete_walls(4)
    dempm.run()

dempm.mpm.postprocessing()
dempm.dem.postprocessing()
