import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(debug=True)

dem = DEM()

dem.set_configuration(domain=ti.Vector([7.,7.,10.]),
                      boundary=[None, None, None],
                      gravity=ti.Vector([0.,0.,-9.8]),
                      engine="VelocityVerlet",
                      track_energy=True,
                      search="HierarchicalLinkedCell",
                      search_direction="Up")

dem.set_solver({
                "Timestep":         5e-3,
                "SimulationTime":   8,
                "SaveInterval":     0.08,
                "OutputData":       "Clump"
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_particle_number": 129691,
                            "max_sphere_number": 129691,
                            "max_clump_number": 0,
                            "max_facet_number": 12,
                            "body_coordination_number":15,
                            "verlet_distance_multiplier": 0.4,
                            "body_coordination_number": [32, 32],
                            "wall_coordination_number": 2,
                            "hierarchical_level": 2,
                            "hierarchical_size": [0.015, 0.15],
                            "compaction_ratio": [0.2, 0.05],
                            "wall_per_cell": [6, 6]
                            }, log=True)
                         

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            2650,
                            "ForceLocalDamping":  0.7,
                            "TorqueLocalDamping": 0.7
                            })
                           
dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               }})

                            
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
     
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e5,
                            "TangentialStiffness":        1e5,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.05,
                            "TangentialViscousDamping":   0.05
                           })           
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            5e5,
                            "TangentialStiffness":        5e5,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.05,
                            "TangentialViscousDamping":   0.05
                           })       
                  
dem.add_wall(body=[{
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 0.00],
                                    "vertice2": [7., 0.0, 0.00],
                                    "vertice3": [7., 7., 0.00],
                                    "vertice4": [0.0, 7., 0.00]
                                   },
                   "OuterNormal": [0., 0., 1.]
                  },
                  
                  {
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 10.],
                                    "vertice2": [7., 0.0, 10.],
                                    "vertice3": [7., 7., 10.],
                                    "vertice4": [0.0, 7., 10.]
                                   },
                   "OuterNormal": [0., 0., -1.],
                   "ControlType":  "Force",
                   "TargetStress": 2.45e5,
                   "Alpha":        0.5,
                   "LimitVelocity": 2.
                  },
                  
                  {
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 0.000],
                                    "vertice2": [0.0, 7., 0.000],
                                    "vertice3": [0.0, 7., 10.0],
                                    "vertice4": [0.0, 0.000, 10.0]
                                   },
                   "OuterNormal": [1., 0., 0.]
                  },
                  
                  {
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [7., 0.0, 0.000],
                                    "vertice2": [7., 7., 0.000],
                                    "vertice3": [7., 7., 10.0],
                                    "vertice4": [7., 0.000, 10.0]
                                   },
                   "OuterNormal": [-1., 0., 0.]
                  },
                  
                  {
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 0.000],
                                    "vertice2": [7., 0.0, 0.000],
                                    "vertice3": [7., 0.0, 10.0],
                                    "vertice4": [0.0, 0.0, 10.0]
                                   },
                   "OuterNormal": [0., 1., 0.]
                  },
                  
                  {
                   "WallID":      5,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0.0, 7., 0.000],
                                    "vertice2": [7., 7., 0.000],
                                    "vertice3": [7., 7., 10.0],
                                    "vertice4": [0.0, 7., 10.0]
                                   },
                   "OuterNormal": [0., -1., 0.]
                  }
                  ])
                  
dem.select_save_data(sphere=True, particle_particle_contact=True, particle_wall_contact=True, wall=True)
                  
dem.run()            
