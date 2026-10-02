import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
from geotaichi import *

init(arch='gpu', log=False, debug=True)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([0.2, 0.2, 0.4]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., 0.]),
                        track_energy=True)

lsdem.memory_allocate(memory={
                                 "max_material_number": 1,
                                 "max_rigid_body_number": 1,
                                 "levelset_grid_number": 243340,
                                 "surface_node_number": 14554,
                                 "max_sphere_number": 0,
                                 "max_clump_number": 0,
                                 "max_plane_number": 1,
                                 "body_coordination_number":   1,
                                 "wall_coordination_number":   1,
                                 "verlet_distance_multiplier": [0.15, 0.3],
                                 "compaction_ratio":           [0.9, 1.0]
                             })  

lsdem.set_solver({
                "Timestep":         1e-6,
                "SimulationTime":   3e-3,
                "SaveInterval":     3e-5,
                "SavePath":         "CylinderImpact/dynamics/fine"
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
                   
lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":             polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/cylinder_finer.stl').grids(space=0.1, extent=4),
                                "WriteFile":          True}) 

fai = 62/180*math.pi
vector = [-math.cos(fai), 0., math.sin(fai)]
lsdem.create_body(body={
                            "BodyType": "RigidBody",
                            "Template":[{
                                             "Name": "Template1",
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0., 0., -1.]),
                                             "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                                             "BodyPoint": ti.Vector([0.1-math.sqrt(2)/50.*math.cos((45)/180.*math.pi+fai), 0.1, 1.03*math.sqrt(2)/50.*math.sin((45)/180.*math.pi+fai)]),
                                             "ScaleFactor": 0.02,
                                             "BodyOrientation": vector,
                                             "FixMotion": ["Free", "Free", "Free"]
                                        }]
                        })


lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   ti.Vector([0.1, 0.1, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })


'''lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")
                            
lsdem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "NormalStiffness":            1e6,
                                "TangentialStiffness":        1e6,
                                "Friction":                   0.,
                                "NormalViscousDamping":       0.,
                                "TangentialViscousDamping":   0.0
                            })  '''
                            

lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Barrier Model")
                            
lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "Stiffness":                  1e8,
                            "NormalCutOff":               0.001,
                            "StiffnessRatio":             1.,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.,
                            "TangentialViscousDamping":   0.0
                           }, dType="particle-wall")     

lsdem.select_save_data()

lsdem.run()
