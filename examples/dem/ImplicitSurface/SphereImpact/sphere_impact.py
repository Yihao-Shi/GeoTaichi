import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False, debug=False)

dem = DEM()

dem.set_configuration(domain=ti.Vector([10.,5.,5.]),
                        scheme="PolySuperEllipsoid",
                        gravity=ti.Vector([0., 0., -0.]),
                        track_energy=True,
                        #iterative_model="PCN"
                        )

dem.memory_allocate(memory={
                                 "max_material_number": 1,
                                 "max_rigid_body_number": 2,
                                 "max_rigid_template_number": 1,
                                 "max_facet_number": 12,
                                 "levelset_grid_number": 21875,
                                 "surface_node_number": 1200,
                                 "body_coordination_number":   30,
                                 "wall_coordination_number":   3,
                                 "verlet_distance_multiplier": [0.1, 0.1],
                                 "point_coordination_number":  [4, 2], 
                                 "compaction_ratio":           [0.9, 0.9],
                                 "wall_per_cell":              12,
                             })   

dem.set_solver({
                "Timestep":         1e-5,
                "SimulationTime":   2e-1,
                "SaveInterval":     2e-3,
                "SavePath":         "SphereImpact"
               })  

dem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
                   
dem.add_template(template={
                                "Name":               "Template1",
                                "Object":             polysuperellipsoid(xrad1=0.75, yrad1=0.5, zrad1=0.5, xrad2=0.75, yrad2=0.5, zrad2=0.5, epsilon_e=1.0, epsilon_n=1.0).grids(space=0.05, extent=2),
                                "SurfaceNodeNumber":  1002,
                                "WriteFile":          True}) 


dem.create_body(body={
                            "BodyType": "RigidBody",
                            "Template":[{
                                             "Name": "Template1",
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": [0., -2., 0.],
                                             "InitialAngularVelocity": [0., 0., 0.],
                                             "BodyPoint": [4.95, 2.0, 3.],
                                             "ScaleFactor": 0.3,
                                             "BodyOrientation": [0, 90, -90]
                                        },
                                        {
                                             "Name": "Template1",
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": [0., 0., 0.],
                                             "InitialAngularVelocity": [0., 0., 0.],
                                             "BodyPoint": [5.15, 2.0, 2.78],
                                             "ScaleFactor": 0.35,
                                             "BodyOrientation": [0, 90, 45]
                                        }]
                        })              
'''               
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   [5., 2.5, 0.],
                   "OuterNormal":  [0., 0., 1.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   [5., 2.5, 5.],
                   "OuterNormal":  [0., 0., -1.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   [10., 2.5, 2.5],
                   "OuterNormal":  [-1., 0., 0.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   [0., 2.5, 2.5],
                   "OuterNormal":  [1., 0., 0.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   [5., 0., 2.5],
                   "OuterNormal":  [0., 1., 0.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   [5., 5., 2.5],
                   "OuterNormal":  [0., -1., 0.]
                  })

'''

dem.add_wall(body=[{
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 0.],
                                    "vertice2": [10., 0.0, 0.],
                                    "vertice3": [10., 5., 0.],
                                    "vertice4": [0.0, 5., 0.]
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.])
                  },
                  
                  {
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 5.],
                                    "vertice2": [10., 0.0, 5.],
                                    "vertice3": [10., 5., 5.],
                                    "vertice4": [0.0, 5., 5.]
                                   },
                   "OuterNormal": ti.Vector([0., 0., -1.])
                  },
                  
                  {
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": [0., 0., 0.0],
                                    "vertice2": [0., 5., 0.],
                                    "vertice3": [0., 5., 5.],
                                    "vertice4": [0., 0., 5.]
                                   },
                   "OuterNormal": ti.Vector([1., 0., 0.])
                  },
                  
                  {
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": [10., 0., 0.0],
                                    "vertice2": [10., 5., 0.],
                                    "vertice3": [10., 5., 5.],
                                    "vertice4": [10., 0., 5.]
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.])
                  },
                  
                  {
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": [0., 0., 0.],
                                    "vertice2": [10., 0., 0.],
                                    "vertice3": [10., 0., 5.],
                                    "vertice4": [0., 0., 5.]
                                   },
                   "OuterNormal": ti.Vector([0., 1., 0.])
                  },
                  
                  {
                   "WallID":      5,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": [0., 5., 0.],
                                    "vertice2": [10., 5., 0.],
                                    "vertice3": [10., 5., 5.],
                                    "vertice4": [0., 5., 5.]
                                   },
                   "OuterNormal": ti.Vector([0., -1., 0.])
                  }])

dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            8e5,
                            "TangentialStiffness":        8e5,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.,
                            "TangentialViscousDamping":   0.
                           })   

dem.select_save_data(wall=True)

dem.run()
