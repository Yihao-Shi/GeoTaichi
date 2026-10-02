import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import argparse, math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-w', type=float, default=1)
parser.add_argument('-n', type=int, default=0)
parser.add_argument('-p', type=int, default=0)
parser.add_argument('-r', type=float, default=1.0)
parser.add_argument('-f', type=float, default=1.0)
parser.add_argument('-rand', type=int, default=0)
parser.add_argument('-st', type=float, default=0.3)
parser.add_argument('-a', type=float, default=None, required=False)
args = parser.parse_args()

scale = args.w

from geotaichi import *

init(debug=False, device_memory_GB=5.0, random_seed=args.rand)

dempm = DEMPM()

dempm.set_configuration(domain=[50., scale * 50., 20.],
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.2,
                      alphaPIC=0.002, 
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=[0., 0., -9.8],
                      #velocity_projection="Affine",
                      #stabilize="B-Bar Method",
                      #pressure_smooth=True
                      )

dempm.dem.set_configuration(
                      gravity=[0., 0., -9.8],
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      

path = 'Flat'+str(int(args.n))+'_'+str(args.p)+'_r'+str(args.r)
if args.a is not None:
    angle = args.a
    if angle < 0: angle+=180
    path += '_a'+str(angle)
if args.rand > 0:
    path += '_rand'+str(args.rand)
if args.n == 0:
    path = 'Flat'
    
dempm.set_solver({
                      "Timestep":         3e-4,
                      "SimulationTime":   6.,
                      "SaveInterval":     args.st,
                      "SavePath":         path
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 417,
                                "levelset_grid_number": 313875,
                                 "max_rigid_template_number": 1,
                                "surface_node_number": 1502,
                                "max_plane_number": 6,
                                "body_coordination_number":   12,
                                "wall_coordination_number":   6,
                                "verlet_distance_multiplier": [0.15, 0.2],
                                "point_coordination_number":  [3, 4], 
                                "compaction_ratio":           [0.5, 0.5, 0.5, 0.5],
                                "wall_per_cell":              6
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           1800000*args.w,
                                "verlet_distance_multiplier":    0.1,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   126702
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    12,
                                  "wall_coordination_number":    3,
                                  "compaction_ratio": [0.2, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2400,
                                "ForceLocalDamping":  0.15,
                                "TorqueLocalDamping": 0.05
                            })

if args.p==0:
    dempm.dem.add_template(template={
                                "Name":               "template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/Hamburg_sand.stl').grids(space=25, extent=5),
                                "WriteFile":          True}) 
elif args.p == 1:
    dempm.dem.add_template(template={
                                "Name":               "template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand.stl').grids(space=5, extent=5),
                                "WriteFile":          True}) 
elif args.p == 2:
    dempm.dem.add_template(template={
                                "Name":               "template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sphere.stl').grids(space=0.2, extent=5),
                                "WriteFile":          True}) 

def volume_func1():
    return 430. * scale * 50.

def region_func1(pos, rad):
    polygon = ti.Matrix([[0., 0.], [30., 0.], [30., 6.], [20., 16.], [0., 16.]])
    in_polygon = circle_inside_geometry(polygon, pos[0], pos[2], rad)
    in_plane = rad < pos[1] < scale * 50.- rad
    return in_polygon and in_plane
    
def volume_func2():
    return 120. * scale * 50.

def region_func2(pos, rad):
    polygon = ti.Matrix([[30., 0.], [50., 0.], [50., 6.], [30., 6.]])
    in_polygon = circle_inside_geometry(polygon, pos[0], pos[2], rad)
    in_plane = rad < pos[1] < scale * 50.- rad
    return in_polygon and in_plane

body_num = args.n
if body_num > 0:
    dempm.dem.add_region(region=[{
                            "Name": "region1",
                            "Type": "UserDefined",
                            "BoundingBoxPoint": [0.0, 0.0, 0.0],
                            "BoundingBoxSize": [30., scale * 50., 16.],
                            "RegionVolume": volume_func1,
                            "RegionFunction": region_func1
                      },
                      {
                            "Name": "region2",
                            "Type": "UserDefined",
                            "BoundingBoxPoint": [30.0, 0.0, 0.0],
                            "BoundingBoxSize": [20., scale * 50., 16.],
                            "RegionVolume": volume_func2,
                            "RegionFunction": region_func2
                      }])

    ori = 'uniform'
    if args.a is not None:
        ori = [0., args.a, 0.]
        
    dempm.dem.add_body(body={
                     "GenerateType": "Generate",
                     "RegionName": "region1",
                   "BodyType": "RigidBody",
                   "PoissonSampling": False,
                   "TryNumber": 50000,
                   "Template":{
                               "Name": "Template1",
                               "MaxRadius": args.r,
                               "MinRadius": args.r,
                               "BodyNumber": body_num,
                               "BodyOrientation": ori,
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": [0.,0.,0.],
                               "InitialAngularVelocity": [0.,0.,0.]
                                             }})
    dempm.dem.add_body(body={
                     "GenerateType": "Generate",
                     "RegionName": "region2",
                   "BodyType": "RigidBody",
                   "PoissonSampling": False,
                   "TryNumber": 50000,
                   "Template":{
                               "Name": "Template1",
                               "MaxRadius": 1.0,
                               "MinRadius": 1.0,
                               "BodyNumber": 40,
                               "BodyOrientation": ori,
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": [0.,0.,0.],
                               "InitialAngularVelocity": [0.,0.,0.]
                                             }})
    #if dempm.dem.scene.particleNum[0] != body_num: raise RuntimeError(f"body number is {dempm.dem.scene.particleNum[0]}")

dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([25, scale * 25., 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([25, 0., 10]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([25, scale * 50., 10]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., scale * 25., 10]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([50., scale * 25., 10]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })

if body_num > 0:
    dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
    dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e8,
                            "TangentialStiffness":        1e8,
                            "Friction":                   0.6*args.f,
                            "NormalViscousDamping":       0.15,
                            "TangentialViscousDamping":   0.15
                           })      
                           
    dempm.dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            1e9,
                            "TangentialStiffness":        1e9,
                            "Friction":                   0.6*args.f,
                            "NormalViscousDamping":       0.15,
                            "TangentialViscousDamping":   0.15
                           })  
    dempm.dem.select_save_data(surface=True)
else:
    dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                         particle_wall_contact_model=None)
    dempm.dem.select_save_data(surface=False)

                 
dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "Density":              1800.,
                               "YoungModulus":                  1e8,
                               "PossionRatio":                 0.3,
                               "StaticFriction":                np.atan(np.tan(20/180*np.pi)*args.f)*180/np.pi,
                               "DynamicFriction":               np.atan(np.tan(20/180*np.pi)*args.f)*180/np.pi,
                               "Cohesion":                      1e4*args.f,
                               "Dilation":                      9,
                               "dpType":                         "Circumscribed"
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               [1.0, 1.0, 1.0]
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0.0, 0.0, 0.0],
                            "BoundingBoxSize": [50., scale * 50, 6],
                            
                      }])
                      
dempm.mpm.add_region(region=[{
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0.0, 0.0, 6.0],
                            "BoundingBoxSize": [20, scale * 50, 10.],
                            
                      }])
                      
dempm.mpm.add_region(region=[{
                            "Name": "region3",
                            "Type": "TrianglarPrism",
                            "BoundingBoxPoint": [20.0, 0.0, 6.0],
                            "BoundingBoxSize": [10., scale * 50., 10.],
                            
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":   [0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":   [0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   {
                                       "RegionName":         "region3",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":   [0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   
dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [50., 0.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0, scale * 50.0, 0],
                                        "EndPoint":       [50., scale * 50.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., scale * 50.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., 0., 0.],
                                        "StartPoint":     [50., 0.0, 0],
                                        "EndPoint":       [50., scale * 50.0, 20.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., 0., 0.],
                                        "StartPoint":     [0., 0.0, 0.],
                                        "EndPoint":       [50., scale * 50.0, 0.],
                                    }])

dempm.mpm.select_save_data()

dempm.add_body(check_overlap=True)

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            9e6,
                                 "TangentialStiffness":        6e6,
                                 "Friction":                   0.5*args.f,
                                 "NormalViscousDamping":       0.15,
                                 "TangentialViscousDamping":   0.15
                            }, dType="particle-particle")

def get_gravity(points):
    return np.where(points[:, 0]<20.0, 16.-points[:, 2],
                   np.where(points[:, 0]<30.0, 36.0-points[:, 0]-points[:, 2],
                            6.-points[:, 2]))
                            
dempm.run(mpm_gravity_field=get_gravity)

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
