import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([10.,5.,5.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., 0.]),
                        track_energy=True)

lsdem.memory_allocate(memory={
                                 "max_material_number": 1,
                                 "max_rigid_body_number": 1,
                                 "levelset_grid_number": 24389,
                                 "surface_node_number": 13000,
                                 "max_sphere_number": 0,
                                 "max_clump_number": 0,
                                 "max_facet_number": 2,
                                 "body_coordination_number":   150,
                                 "wall_coordination_number":   150,
                                 "verlet_distance_multiplier": [0.15, 0.2],
                                 "compaction_ratio":           [0.9, 1.0]
                             })  

lsdem.set_solver({
                "Timestep":         1e-5,
                "SimulationTime":   0.07,
                "SaveInterval":     0.0007,
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
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sphere_finer.stl').grids(space=0.1,extent=4),
                                "WriteFile":          True}) 


lsdem.create_body(body={
                            "BodyType": "RigidBody",
                            "Template":[{
                                             "Name": "Template1",
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0., 0., -2.]),
                                             "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                                             "BodyPoint": ti.Vector([4., 2.0, 1.0025]),
                                             "ScaleFactor": 1.0,
                                             "BodyOrientation": "constant"
                                        }]
                        })


lsdem.add_wall(body={
                   "WallType":    "Facet",
                   "MaterialID":   0,
                   "WallShape":   "Polygon",
                   "WallID":      0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([3, 1., 0.5]),
                                    "vertice2": ti.Vector([5., 1, 0.5]),
                                    "vertice3": ti.Vector([5, 3., 0.5]),
                                    "vertice4": ti.Vector([3., 3., 0.5])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.])
                  })


lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")
                            
lsdem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "NormalStiffness":            1e8,
                                "TangentialStiffness":        1e8,
                                "Friction":                   0.,
                                "NormalViscousDamping":       0.,
                                "TangentialViscousDamping":   0.0
                            })         

lsdem.select_save_data(wall=True)

lsdem.run()
