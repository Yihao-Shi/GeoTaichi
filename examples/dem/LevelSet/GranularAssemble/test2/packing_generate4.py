import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False, debug=False)

lsdem = DEM()

yextent = 7.5 * 4
lsdem.set_configuration(domain=ti.Vector([15.,yextent,28.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]),
                        visualize=False)

# verlet_distance_multiplier      point_coordination_number          compaction_ratio                      time                
#       [0.0, 0.0]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]              1736.9612028598785

#       [0.05, 0.0]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              1932.7623410224915
#       [0.05, 0.1]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              491.3294584751129
#       [0.05, 0.2]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              504.59620428085327
#       [0.05, 0.3]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              665.1574399471283
#       [0.05, 0.4]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]              773.508859872818
#       [0.05, 0.5]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]              874.2034590244293

#       [0.1, 0.0]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]              2207.0326607227325        
#       [0.1, 0.1]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]              517.7649660110474   
#       [0.1, 0.2]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]              539.8581161499023         592.1867594718933
#       [0.1, 0.3]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]              553.449286699295          564.4278094768524
#       [0.1, 0.4]                        [4, 2]                 [0.3, 0.1, 0.17, 0.015]              663.4796271324158         669.0696995258331
#       [0.1, 0.5]                        [4, 2]                 [0.3, 0.1, 0.17, 0.015]              756.4779114723206

#       [0.15, 0.0]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              2469.231126308441
#       [0.15, 0.1]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              577.3877367973328
#       [0.15, 0.2]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              535.8891003131866         542.1603066921234
#       [0.15, 0.3]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]              632.4889814853668         
#       [0.15, 0.4]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]              636.5228719711304
#       [0.15, 0.5]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]              721.999353647232


#    Particle Number                Time
#        100000                   296.5343511104584
#        200000                   564.4278094768524
#        300000                   901.9137921333313
#        400000                   1196.83811545372
#        460000                   1387.052277803421

lsdem.memory_allocate(memory={
                                 "max_material_number": 2,
                                 "max_rigid_body_number": 400000,
                                 "levelset_grid_number": 46655,
                                 "surface_node_number": 127,
                                 "max_plane_number": 6,
                                 "body_coordination_number":   25,
                                 "wall_coordination_number":   3,
                                 "verlet_distance_multiplier": [0.05, 0.1],
                                 "point_coordination_number":  [3, 2], 
                                 "compaction_ratio":           [0.3, 0.07, 0.17, 0.015]
                             })  

lsdem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   3.,
                "SaveInterval":     0.15
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.2,
                                "TorqueLocalDamping": 0.2
                            })
                   
lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":             polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand.stl').grids(space=5, extent=3),
                                "WriteFile":          True}) 

lsdem.add_body_from_file(body={
                            "FileType":  "TXT",
                            "Template":[{
                                             "Name": "Template1",
                                             "BodyType": "RigidBody",
                                             "File":'BoundingSphere4.txt',
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.])
                                        }]
                        })
                        
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([7.5, 0.5*yextent, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([7.5, 0.5*yextent, 30.]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([15, 0.5*yextent, 15]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 0.5*yextent, 15]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([7.5, 0., 15]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([7.5, yextent, 15]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })

lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

                            
lsdem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "NormalStiffness":            1e8,
                                "TangentialStiffness":        1e8,
                                "Friction":                   0.5,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            })   
                            
lsdem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "NormalStiffness":            1e9,
                                "TangentialStiffness":        1e9,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            })      

lsdem.select_save_data(particle=False, surface=False)

lsdem.run()
