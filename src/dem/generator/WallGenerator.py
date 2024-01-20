import os, warnings
import trimesh

from src.dem.generator.InsertionKernel import *
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.ObjectIO import DictIO
from src.utils.PolygonDiscretization import *
from src.utils.TypeDefination import vec3f
from third_party.pyquaternion.quaternion import Quaternion
from third_party.pyevtk.hl import unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle


class WallGenerator(object):
    def insert_wall(self, wall_dict, sims, scene: myScene):
        wall_type = DictIO.GetEssential(wall_dict, "WallType")
        if wall_type == "Plane":
            if not scene.wall is None:
                print('#', "Start adding plane(s) ......")
                self.add_plane_wall(wall_dict, sims, scene)
            else:
                raise RuntimeError("Plane class has not been activated")
        elif wall_type == "Facet":
            if not scene.wall is None:
                print('#', "Start adding facet(s) ......")
                wallShape = DictIO.GetEssential(wall_dict, "WallShape")
                if wallShape == "Polygon":
                    self.add_polygon_wall(wall_dict, sims, scene)
                elif wallShape == "Cylinder":
                    self.add_cylinder_wall(wall_dict, sims, scene)
            else:
                raise RuntimeError("Facet class has not been activated")
        elif wall_type == "Patch":
            if not scene.wall is None:
                print('#', "Start adding patch(s) ......")
                file_type = DictIO.GetAlternative(wall_dict, "FileType", None)
                if file_type == "geo":
                    pass
                elif file_type == "txt":
                    pass
                else:
                    self.add_patch_wall(wall_dict, sims, scene)
            else:
                raise RuntimeError("Patch class has not been activated")
        else:
            raise ValueError("Wall Shape Type error!")
        
    def print_plane_info(self, matID, point, norm):
        print(" Wall Information ".center(71, '-'))
        print("Generate Type: Create Plane")
        print("Material ID = ", matID)
        print("The center the wall = ", point)
        print("The normal direction of the wall = ", norm, '\n')

    def print_facet_info(self, matID, norm, init_v, facet_count, control_type=None, servo=False):
        print(" Wall Information ".center(71, '-'))
        if servo:
            print("Generate Type: Create Servo Facet(s)")
            print("Servo Type:", control_type)
        else:
            print("Generate Type: Create Facet(s)")
        print("Material ID = ", matID)
        print("The normal direction of the wall = ", norm)
        print("Initial Velocity = ", init_v)
        print("Facet Number = ", facet_count)
        print('\n')
        
    # ========================================================= #
    #                      Create Plane                         #
    # ========================================================= #
    def add_plane_wall(self, wall_dict, sims, scene: myScene):
        wallID = DictIO.GetAlternative(wall_dict, "WallID", int(scene.wallNum[0]))
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        center = DictIO.GetEssential(wall_dict, "WallCenter")
        norm = DictIO.GetEssential(wall_dict, "OuterNormal").normalized()
        scene.check_wall_number(sims, body_number=1)
        wallNum = int(scene.wallNum[0])
        scene.wall[wallNum].add_materialID(matID)
        scene.wall[wallNum].add_wall_geometry(wallID, center, norm)
        scene.wallNum[0] += 1
        self.print_plane_info(matID, center, norm)

    # ========================================================= #
    #                      Create Facet                         #
    # ========================================================= #
    def tranverse_poly(self, poly_arr):
        normalized_poly, scalar, offset = PolyPretreatment(poly_arr)
        poly = Polygon(normalized_poly)
        return poly, scalar, offset

    def discretize_wall_to_facet(self, poly):
        wall_facet=[]
        wall_facet = GetDivTri(poly, wall_facet)
        new_wall_facet = OptAlltris(wall_facet)
        new_wall_facet = OptAlltris(new_wall_facet)
        return new_wall_facet

    def rotate_wall(self, poly_arr, origin, target):
        Q = Quaternion(FromVector=origin, ToVector=target)
        for ver in range(len(poly_arr)):
            poly_arr[ver] = Q.rotate(poly_arr[ver])
        return poly_arr

    def add_polygon_wall(self, wall_dict, sims: Simulation, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        vertices = DictIO.GetEssential(wall_dict, "WallVertice")
        norm = DictIO.GetEssential(wall_dict, "OuterNormal").normalized().to_numpy()
        init_v = DictIO.GetAlternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        
        poly_arr = np.array([list(item) for item in vertices.values()])
        scene.vispts.append(list(poly_arr))

        if np.linalg.norm(np.cross(norm, vec3f([0, 0, 1]))) != 0:
            poly_arr = self.rotate_wall(poly_arr, origin=norm, target=np.array([0, 0, 1]))

        poly, scalar, offset = self.tranverse_poly(poly_arr)
        if IsSimplePoly(poly):
            new_wall_facet = self.discretize_wall_to_facet(poly)
            for tri in new_wall_facet:
                wall_vertices = np.array([*tri.exterior.coords])
                wall_vertices = np.multiply(wall_vertices, scalar) + offset
                
                scene.vistri.append([int(np.where((np.max(np.abs(poly_arr-wall_vertices[0]), 1) < Threshold))[0]), 
                                     int(np.where((np.max(np.abs(poly_arr-wall_vertices[1]), 1) < Threshold))[0]), 
                                     int(np.where((np.max(np.abs(poly_arr-wall_vertices[2]), 1) < Threshold))[0]), 
                                     3 * (int(scene.wallNum[0]) + 1), VtkTriangle.tid])
                
                if np.linalg.norm(np.cross(norm, vec3f([0, 0, 1]))) != 0:
                    wall_vertices = self.rotate_wall(wall_vertices, origin=vec3f([0, 0, 1]), target=norm)

                scene.check_wall_number(sims, body_number=1)
                scene.wall[int(scene.wallNum[0])].add_materialID(matID)
                scene.wall[int(scene.wallNum[0])].add_wall_geometry(wallID, wall_vertices[0], wall_vertices[1], wall_vertices[2], norm, init_v)
                scene.wallNum[0] += 1

            facet_num = len(new_wall_facet)

            if sims.max_servo_wall_num > 0. and DictIO.GetAlternative(wall_dict, "ControlType", None):
                scene.check_servo_number(sims, body_number=1)
                alpha = DictIO.GetAlternative(wall_dict, "Alpha", 0.8)
                target_stress = DictIO.GetEssential(wall_dict, "TargetStress")
                max_velocity = DictIO.GetEssential(wall_dict, "LimitVelocity")
                startIndex = int(scene.wallNum[0]) - facet_num
                endIndex = int(scene.wallNum[0])
                scene.servo[int(scene.servoNum[0])].add_servo_wall(startIndex, endIndex, alpha, target_stress, max_velocity)
                scene.servoNum[0] += 1
                self.print_facet_info(matID, norm, init_v, facet_num, DictIO.GetAlternative(wall_dict, "ControlType", None), servo=True)
            else:
                self.print_facet_info(matID, norm, init_v, facet_num)

            if DictIO.GetAlternative(wall_dict, "Visualize", False):
                self.visualize(sims, scene)
        else:
            raise ValueError("Wall vertices error!")

    def add_cylinder_wall(self, wall_dict, sims, scene: myScene):
        pass


    # ========================================================= #
    #                      Create Patch                         #
    # ========================================================= #
    def add_patch_wall(self, wall_dict, sims, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        file = DictIO.GetEssential(wall_dict, "WallFile")
        scale = DictIO.GetAlternative(wall_dict, "ScaleFactor", 1.)
        offset = DictIO.GetAlternative(wall_dict, "Translation", vec3f([0, 0, 0]))
        direction = DictIO.GetEssential(wall_dict, "Orientation", vec3f([0, 0, 1]))
        init_v = DictIO.GetAlternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))

        mesh = trimesh.load(file)
        mesh.apply_scale(scale)
        
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        center = mesh.vertices.mean(axis=0)

        scene.vispts = list(vertices.copy())
        scene.vistri = list(np.c_(faces.copy(), np.arange(3 * int(scene.wallNum[0]), 3 * (int(scene.wallNum[0]) + faces.shape[0]) + 1, 3), np.repeat(VtkTriangle.tid, faces.shape[0])))

        kernel_add_patch(int(scene.wallNum[0]), wallID, matID, vertices, faces, center, offset, direction, init_v, scene.wall)
        scene.wallNum[0] += faces.shape[0]

        
    # ========================================================= #
    #                         Reload                            #
    # ========================================================= #
    def restart_walls(self, wall_dict, sims: Simulation, scene: myScene):
        print(" Wall Information ".center(71, '-'))
        wall = DictIO.GetAlternative(wall_dict, "WallFile", None)
        servo = DictIO.GetAlternative(wall_dict, "ServoFile", None)
        file_type = DictIO.GetAlternative(wall_dict, "FileType", "NPZ")

        if file_type == "NPZ":
            self.restart_npz_wall(wall, sims, scene)
            self.restart_npz_servo(wall, servo, sims, scene)
        elif file_type == "TXT":
            pass
        
        if DictIO.GetAlternative(wall_dict, "Visualize", False):
            self.visualize(sims, scene)
        
    def restart_npz_wall(self, wall, sims: Simulation, scene: myScene):
        if not wall is None:
            if not os.path.exists(wall):
                raise EOFError("Invaild wall path")
            
            wall_info = np.load(wall, allow_pickle=True) 
            wall_number = int(DictIO.GetEssential(wall_info, "body_num"))
            if sims.is_continue:
                sims.current_time = DictIO.GetAlternative(wall_info, "t_current", 0)
                sims.CurrentTime[None] = DictIO.GetAlternative(wall_info, "t_current", 0)

            scene.check_wall_number(sims, body_number=wall_number)
            if sims.wall_type == 0:
                kernel_rebuild_plane(int(scene.wallNum[0]), wall_number, scene.wall, 
                                     DictIO.GetAlternative(wall_info, "active", np.zeros(wall_number) + 1), 
                                     DictIO.GetAlternative(wall_info, "wallID", np.zeros(wall_number)), 
                                     DictIO.GetEssential(wall_info, "materialID"), 
                                     DictIO.GetEssential(wall_info, "point"), 
                                     DictIO.GetEssential(wall_info, "norm"))
                print("Inserted plane number: ", wall_number)
            elif sims.wall_type == 1 or sims.wall_type == 2:
                point1 = DictIO.GetEssential(wall_info, "point1")
                point2 = DictIO.GetEssential(wall_info, "point2")
                point3 = DictIO.GetEssential(wall_info, "point3")
                unique_point = np.unique(np.vstack((point1, point2, point3)), axis=0)
                scene.vispts += list(unique_point)
                for i in range(point1.shape[0]):
                    scene.vistri.append([int(np.where((unique_point==point1[i]).all(1))[0]), int(np.where((unique_point==point2[i]).all(1))[0]), int(np.where((unique_point==point3[i]).all(1))[0]), 3 * (i + 1), VtkTriangle.tid])
                kernel_rebuild_triangular(int(scene.wallNum[0]), wall_number, scene.wall, 
                                          DictIO.GetAlternative(wall_info, "active", np.zeros(wall_number) + 1), 
                                          DictIO.GetAlternative(wall_info, "wallID", np.zeros(wall_number)), 
                                          DictIO.GetEssential(wall_info, "materialID"), 
                                          DictIO.GetEssential(wall_info, "point1"), 
                                          DictIO.GetEssential(wall_info, "point2"), 
                                          DictIO.GetEssential(wall_info, "point3"), 
                                          DictIO.GetEssential(wall_info, "norm"), 
                                          DictIO.GetAlternative(wall_info, "velocity", np.zeros((wall_number, 3))))
                if sims.wall_type == 1:
                    print("Inserted facet number: ", wall_number)
                elif sims.wall_type == 2:
                    print("Inserted patch number: ", wall_number)
            scene.wallNum[0] += wall_number
        
    def restart_npz_servo(self, wall, servo, sims: Simulation, scene: myScene):    
        if servo:
            if sims.max_servo_wall_num <= 0:
                raise RuntimeError("/max_servo_number/ should be larger than zero")
            
            if sims.wall_type != 1:
                raise RuntimeError("/max_facet_number/ has not been set")
            
            if wall is None:
                raise EOFError("Invalid path to read wall information")
            
            servo_info = np.load(servo, allow_pickle=True)
            servo_number = int(DictIO.GetEssential(servo_info, "body_num"))

            scene.check_servo_number(sims, body_number=servo_number)
            kernel_rebuild_servo(int(scene.servoNum[0]), servo_number, scene.servo, 
                                 DictIO.GetAlternative(servo_info, "active", np.zeros(servo_number) + 1), 
                                 DictIO.GetEssential(servo_info, "startIndex"), 
                                 DictIO.GetEssential(servo_info, "endIndex"), 
                                 DictIO.GetEssential(servo_info, "alpha"), 
                                 DictIO.GetEssential(servo_info, "target_stress"), 
                                 DictIO.GetEssential(servo_info, "max_velocity"))
            scene.servoNum[0] += servo_number
            print("Inserted servo number: ", servo_number)
        print('\n')

    def visualize(self, sims: Simulation, scene: myScene):
        if sims.wall_type == 1 or sims.wall_type == 2:
            vispts = np.ascontiguousarray(np.array(scene.vispts))
            vistri = np.ascontiguousarray(np.array(scene.vistri))

            unstructuredGridToVTK("TriangleWall", np.ascontiguousarray(vispts[:, 0]), np.ascontiguousarray(vispts[:, 1]), np.ascontiguousarray(vispts[:, 2]), 
                                  connectivity=np.ascontiguousarray(vistri[:, 0:3].flatten()), 
                                  offsets=np.ascontiguousarray(vistri[:, 3]), 
                                  cell_types=np.ascontiguousarray(vistri[:, 4]))
        else:
            warnings.warn("Wall type:: /Plane/ is not supported to visualize")
