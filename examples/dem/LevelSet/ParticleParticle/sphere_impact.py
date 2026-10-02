import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False, debug=False)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([10.,5.,5.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -0.]),
                        track_energy=True)

lsdem.memory_allocate(memory={
                                 "max_material_number": 1,
                                 "max_rigid_body_number": 2,
                                 "levelset_grid_number": 148877,
                                 "surface_node_number": 13000,
                                 "max_sphere_number": 0,
                                 "max_clump_number": 0,
                                 "max_plane_number": 0,
                                 "body_coordination_number":   1,
                                 "wall_coordination_number":   1,
                                 "verlet_distance_multiplier": [0.1, 0.3],
                                 "point_coordination_number":  [1, 1], 
                                 "compaction_ratio":           [0.9, 1.0]
                             })  

lsdem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   1e-1,
                "SaveInterval":     1e-3,
                "SavePath":         "SphereImpact"
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
                   
lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sphere_finer.stl').grids(space=0.025, extent=6),
                                "WriteFile":          True}) 


lsdem.create_body(body={
                            "BodyType": "RigidBody",
                            "Template":[{
                                             "Name": "Template1",
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([1., 0., 0.]),
                                             "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                                             "BodyPoint": ti.Vector([4.485, 2.0, 3.]),
                                             "ScaleFactor": 1.0,
                                             "BodyOrientation": "constant"
                                        },
                                        {
                                             "Name": "Template1",
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([-1., 0., 0.]),
                                             "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                                             "BodyPoint": ti.Vector([5.515, 2.0, 3.]),
                                             "ScaleFactor": 1.0,
                                             "BodyOrientation": "constant"
                                        }]
                        })


lsdem.choose_contact_model(particle_particle_contact_model="Barrier Model",
                           particle_wall_contact_model="Linear Model")

                            
lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "Stiffness":                  2e8,
                            "NormalCutOff":               0.05,
                            "StiffnessRatio":             1.,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.,
                            "TangentialViscousDamping":   0.
                           }, dType="particle-particle")    
                           
'''lsdem.choose_contact_model(particle_particle_contact_model="Energy Conserving Model",
                           particle_wall_contact_model="Linear Model")
                           
lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            2e8,
                            "TangentialStiffness":        2e8,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.,
                            "TangentialViscousDamping":   0.
                           }, dType="particle-particle")   '''

lsdem.select_save_data()

lsdem.run()
