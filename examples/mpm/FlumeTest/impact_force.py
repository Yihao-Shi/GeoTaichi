import os
import sys

import numpy as np
from math import sin, cos, pi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', type=float, default=45)
parser.add_argument('-f', type=int, default=0)
args = parser.parse_args()

from geotaichi import *

init(device_memory_GB=5, debug=False)


incline_angle=args.a
theta = incline_angle / 180. * pi
flume_leng = 3.0
flume_wid = 0.3
flume_heig = 1.4
baffle_pos = 2.3
baffle_heig = 0.3

spec_leng = 0.5
spec_wid = 0.3
spec_heig = 0.3

restart = False if args.f==0 else True
path = f"ImpactForce{int(incline_angle)}"

dempm = DEMPM()

dempm.set_configuration(domain=[3., 1.0, 1.7],
                        coupling_scheme="MPDEM",
                        particle_interaction=False,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.01,
                      alphaPIC=0.001,
                      mapping="USF", 
                      shape_function="QuadBSpline",
                      gravity=[9.8*sin(theta), 0., -9.8*cos(theta)],
                      material_type="Solid",
                      #velocity_projection="Taylor",
                      #sparse_grid=True
                      )

dempm.dem.set_configuration(
                      gravity=[0., 0., -9.8],
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="DEM")

dempm.set_solver({
                      "Timestep":         1e-4,
                      "SimulationTime":   0.,
                      "CFL":              0.2,
                      "SaveInterval":     0.05,
                      "SavePath":         path,
                      "AdaptiveStep":     50
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 3,
                                "max_particle_number": 0,
                                "max_facet_number": 16,
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
                                  "wall_coordination_number":    16,
                                  "compaction_ratio": [0.02, 0.3]
                             })  
                             
dempm.dem.add_wall(body={
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0., 0., 0.],
                                    "vertice2": [flume_leng, 0., 0.],
                                    "vertice3": [flume_leng, 0., flume_heig],
                                    "vertice4": [0., 0., flume_heig]
                                   },
                   "OuterNormal": [0., 1., 0.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   2,
                   "WallVertice":  {
                                    "vertice1": [0., flume_wid, 0.],
                                    "vertice2": [flume_leng, flume_wid, 0.],
                                    "vertice3": [flume_leng, flume_wid, flume_heig],
                                    "vertice4": [0., flume_wid, flume_heig]
                                   },
                   "OuterNormal": [0., -1., 0.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   2,
                   "WallVertice":  {
                                    "vertice1": [0., 0., 0.],
                                    "vertice2": [flume_leng, 0., 0.],
                                    "vertice3": [flume_leng, flume_wid, 0.],
                                    "vertice4": [0., flume_wid, 0.]
                                   },
                   "OuterNormal": [0., 0., 1.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   2,
                   "WallVertice":  {
                                    "vertice1": [0., 0., 0.],
                                    "vertice2": [0., flume_wid, 0.],
                                    "vertice3": [0., flume_wid, flume_heig],
                                    "vertice4": [0., 0., flume_heig]
                                   },
                   "OuterNormal": [1., 0., 0.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   2,
                   "WallVertice":  {
                                    "vertice1": [flume_leng, 0., 0.],
                                    "vertice2": [flume_leng, flume_wid, 0.],
                                    "vertice3": [flume_leng, flume_wid, flume_heig],
                                    "vertice4": [flume_leng, 0., flume_heig]
                                   },
                   "OuterNormal": [-1., 0., 0.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      5,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   2,
                   "WallVertice":  {
                                    "vertice1": [0., 0., flume_heig],
                                    "vertice2": [flume_leng, 0., flume_heig],
                                    "vertice3": [flume_leng, flume_wid, flume_heig],
                                    "vertice4": [0., flume_wid, flume_heig]
                                   },
                   "OuterNormal": [0., 0., 1.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      6,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   2,
                   "WallVertice":  {
                                    "vertice1": [baffle_pos, 0., 0.],
                                    "vertice2": [baffle_pos, flume_wid, 0.],
                                    "vertice3": [baffle_pos, flume_wid, baffle_heig],
                                    "vertice4": [baffle_pos, 0., baffle_heig]
                                   },
                   "OuterNormal": [-1., 0., 0.]})
                   
dempm.dem.add_wall(body={
                   "WallID":      7,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   2,
                   "WallVertice":  {
                                    "vertice1": [spec_leng, 0., 0.],
                                    "vertice2": [spec_leng, flume_wid, 0.],
                                    "vertice3": [spec_leng, flume_wid, flume_heig],
                                    "vertice4": [spec_leng, 0., flume_heig]
                                   },
                   "OuterNormal": [-1., 0., 0.]})
                   
dempm.dem.set_static_wall()
                   
dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                         particle_wall_contact_model=None)  
                                             
dempm.dem.select_save_data(wall=True)

dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "dpType":               "MiddleCircumscribed",
                               #"RateDependent":        True,
                               "Density":              1379.,
                               "GrainDensity":                  2650,
                               "YoungModulus":                  2778000,
                               "PossionRatio":                 0.2,
                               "StaticFriction":                35,
                               "DynamicFriction":               40,
                               "Cohesion":                      0.,
                               "Tensile":                       20000,
                               "AverageDiameter":               0.0015,
                               "InertialNumber":                0.02,
                               "eps":                           0.001
                 })
                 
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               [0.01, 0.01, 0.01]
                        })

if restart:
    dempm.mpm.read_restart(args.f, path, True)
else:
    dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0., 0., 0.],
                            "BoundingBoxSize": [spec_leng, spec_wid, spec_heig],
                            
                      }])

    dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

dempm.mpm.select_save_data()

if restart:
    dempm.read_restart(args.f, path, False, True)

dempm.choose_contact_model(particle_particle_contact_model=None,
                           particle_wall_contact_model="Linear Model")

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            6e5,
                                 "TangentialStiffness":        4e5,
                                 "Friction":                   0.6,
                                 "NormalViscousDamping":       0.05,
                                 "TangentialViscousDamping":   0.05
                            }, dType='particle-wall')

dempm.add_property(DEMmaterial=2,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            6e5,
                                 "TangentialStiffness":        4e5,
                                 "Friction":                   0.6,
                                 "NormalViscousDamping":       0.05,
                                 "TangentialViscousDamping":   0.05
                            }, dType='particle-wall')

dempm.select_save_data(particle_wall_contact=True)

if not restart:
    dempm.run()

dempm.modify_parameters(SimulationTime=dempm.sims.current_time+2, TimeStep=1e-4, CFL=0.2)
dempm.dem.delete_walls(7)
dempm.run()

dempm.mpm.postprocessing()
dempm.dem.postprocessing()
