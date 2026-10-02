import os, warnings

from src.dem.generator.InsertionKernel import *
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.linalg import flip2d
from src.utils.ObjectIO import DictIO
from src.utils.PolygonDiscretization import *
from src.utils.TypeDefination import vec3f
from src.utils.linalg import transformation_matrix_direction, rotation_matrix_direction, triangle_normal
from third_party.pyevtk.hl import unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle
import trimesh as tm


class WallGenerator(object):
    def insert_wall(self, wall_dict, sims: Simulation, scene: myScene):
        wall_type = DictIO.GetEssential(wall_dict, "WallType")
        if wall_type == "Plane":
            if sims.wall_type != 0:
                raise RuntimeError("Keyword:: /max_plane_number/ should be larger than 0")
            if not scene.wall is None:
                print('#', "Start adding plane(s) ......")
                self.add_plane_wall(wall_dict, sims, scene)
            else:
                raise RuntimeError("Plane class has not been activated")
        elif wall_type == "Facet":
            if sims.wall_type != 1:
                raise RuntimeError("Keyword:: /max_facet_number/ should be larger than 0")
            if not scene.wall is None:
                print('#', "Start adding facet(s) ......")
                wallShape = DictIO.GetEssential(wall_dict, "WallShape")
                if wallShape == "Polygon":
                    self.add_polygon_facet(wall_dict, sims, scene)
                elif wallShape == "Cylinder":
                    self.add_cylinder_facet(wall_dict, sims, scene)
                elif wallShape == "Circular":
                    self.add_circular_facet(wall_dict, sims, scene)
                elif wallShape == "File":
                    self.add_file_facet(wall_dict, sims, scene)
            else:
                raise RuntimeError("Facet class has not been activated")
        elif wall_type == "Patch":
            if sims.wall_type != 2:
                raise RuntimeError("Keyword:: /max_patch_number/ should be larger than 0")
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
        elif wall_type == "DigitalElevation":
            if sims.wall_type != 3:
                raise RuntimeError("Keyword:: /max_digital_elevation_number/ should be larger than 0")
            if not scene.wall is None:
                print('#', "Start adding digital elevation model(s) ......")
                self.add_digital_elevation_wall(wall_dict, sims, scene)
            else:
                raise RuntimeError("Other wall class should not been activated")
        else:
            raise ValueError("Wall Shape Type error!")
        
    def print_plane_info(self, matID, point, norm):
        print(" Wall Information ".center(71, '-'))
        print("Generate Type: Create Plane")
        print("Material ID = ", matID)
        print("The center the wall = ", point)
        print("The normal direction of the wall = ", norm, '\n')

    def print_facet_info(self, matID, norm, init_v, facet_count, control_type=None, servo=False, geometry=None):
        print(" Wall Information ".center(71, '-'))
        if servo:
            print("Generate Type: Create Servo Facet(s)")
            print("Servo Type:", control_type)
        else:
            print("Generate Type: Create Facet(s)")
        print("Material ID = ", matID)
        if norm is not None:
            print("The direction of the wall = ", norm)
        if init_v is not None:
            print("Initial Velocity = ", init_v)
        if geometry is not None:
            geometry.print()
        print("Facet Number = ", facet_count)
        print('\n')

    def mesh_from_file(self, wall_dict):
        file = DictIO.GetEssential(wall_dict, "WallFile")
        scale = DictIO.GetAlternative(wall_dict, "ScaleFactor", 1.)
        offset = DictIO.GetAlternative(wall_dict, "Translation", np.array([0, 0, 0]))
        direction = DictIO.GetAlternative(wall_dict, "Orientation", np.array([0, 0, 1]))

        mesh: tm.Trimesh = tm.load(file)
        mass_center = mesh.center_mass
        mesh.apply_translation(-mass_center)
        mesh.apply_scale(scale)
        mesh.apply_transform(transformation_matrix_direction(np.array([0, 0, 1]), direction))
        mesh.apply_translation(mass_center)
        mesh.apply_translation(np.asarray(offset))
        return mesh
        
    # ========================================================= #
    #                      Create Plane                         #
    # ========================================================= #
    def add_plane_wall(self, wall_dict, sims, scene: myScene):
        wallID = DictIO.GetAlternative(wall_dict, "WallID", int(scene.wallNum[0]))
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        center = DictIO.GetEssential(wall_dict, "WallCenter")
        norm = DictIO.GetEssential(wall_dict, "OuterNormal")
        norm = norm / np.linalg.norm(norm) if np.linalg.norm(norm) != 0. else norm
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
        matrix = rotation_matrix_direction(origin, target)
        for ver in range(len(poly_arr)):
            poly_arr[ver] = matrix @ poly_arr[ver]
        return poly_arr

    def add_polygon_facet(self, wall_dict, sims: Simulation, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        vertices = DictIO.GetEssential(wall_dict, "WallVertice")
        norm = np.asarray(DictIO.GetEssential(wall_dict, "OuterNormal"))
        init_v = DictIO.GetAlternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        norm = norm / np.linalg.norm(norm) if np.linalg.norm(norm) != 0. else norm
        
        poly_arr = np.array([list(item) for item in vertices.values()])

        if np.linalg.norm(np.cross(norm, vec3f([0, 0, 1]))) != 0:
            poly_arr = self.rotate_wall(poly_arr, origin=norm, target=np.array([0, 0, 1]))

        poly, scalar, offset = self.tranverse_poly(poly_arr)
        if IsSimplePoly(poly):
            new_wall_facet = self.discretize_wall_to_facet(poly)
            for tri in new_wall_facet:
                wall_vertices = np.array([*tri.exterior.coords])
                wall_vertices = np.multiply(wall_vertices, scalar) + offset
                
                if np.linalg.norm(np.cross(norm, vec3f([0, 0, 1]))) != 0:
                    wall_vertices = self.rotate_wall(wall_vertices, origin=vec3f([0, 0, 1]), target=norm)

                scene.check_wall_number(sims, body_number=1)
                scene.wall[int(scene.wallNum[0])].add_materialID(matID)
                scene.wall[int(scene.wallNum[0])].add_wall_geometry(wallID, wall_vertices[0], wall_vertices[1], wall_vertices[2], norm, init_v)
                scene.wallNum[0] += 1

            facet_num = len(new_wall_facet)

            if sims.max_servo_wall_num > 0. and DictIO.GetAlternative(wall_dict, "ControlType", None):
                self.add_servo_wall(facet_num, wall_dict, sims, scene)
                self.print_facet_info(matID, norm, init_v, facet_num, DictIO.GetAlternative(wall_dict, "ControlType", None), servo=True)
            else:
                self.print_facet_info(matID, norm, init_v, facet_num)

            if DictIO.GetAlternative(wall_dict, "Visualize", False):
                self.visualize(sims, scene)
        else:
            raise ValueError("Wall vertices error!")
        
    def cylinder_norm(self, v0, v1, v2, center, axis, inner=True):
        n_tri = triangle_normal(v0, v1, v2)
        pc = (v0 + v1 + v2) / 3.
        proj = center + np.dot(pc - center, axis) * axis
        r = pc - proj
        if (np.dot(n_tri, r) < 0) is inner:
            n_tri = -n_tri
        return n_tri

    def add_cylinder_facet(self, wall_dict, sims: Simulation, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        center = DictIO.GetEssential(wall_dict, "WallBase")
        radius = DictIO.GetEssential(wall_dict, "WallRadius")
        height = DictIO.GetEssential(wall_dict, "WallHeight")
        resolution = DictIO.GetEssential(wall_dict, "Resolution")
        direction = DictIO.GetEssential(wall_dict, "Direction", "Inner")
        axis = np.asarray(DictIO.GetEssential(wall_dict, "Axis"))
        init_v = DictIO.GetAlternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0. else axis
        direction = True if direction == "Inner" else False

        if abs(axis[0]) < 0.9:
            a = np.array([1., 0., 0.])
        else:
            a = np.array([0., 1., 0.])

        u = np.cross(axis, a)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        top_center = center + height * axis

        facet_num = resolution
        for i in range(resolution):
            theta0 = 2. * np.pi * i / resolution
            theta1 = 2. * np.pi * (i + 1) / resolution

            p0 = center + radius * (np.cos(theta0) * u + np.sin(theta0) * v)
            p1 = center + radius * (np.cos(theta1) * u + np.sin(theta1) * v)

            theta0p = top_center + radius * (np.cos(theta0) * u + np.sin(theta0) * v)
            theta1p = top_center + radius * (np.cos(theta1) * u + np.sin(theta1) * v)

            normal1 = self.cylinder_norm(p0, theta0p, theta1p, center, axis, direction)
            scene.check_wall_number(sims, body_number=1)
            scene.wall[int(scene.wallNum[0])].add_materialID(matID)
            scene.wall[int(scene.wallNum[0])].add_wall_geometry(wallID, p0, theta0p, theta1p, normal1, init_v)
            scene.wallNum[0] += 1

            normal2 = self.cylinder_norm(p0, theta1p, p1, center, axis, direction)
            scene.check_wall_number(sims, body_number=1)
            scene.wall[int(scene.wallNum[0])].add_materialID(matID)
            scene.wall[int(scene.wallNum[0])].add_wall_geometry(wallID, p0, theta1p, p1, normal2, init_v)
            scene.wallNum[0] += 1

        if sims.max_servo_wall_num > 0. and DictIO.GetAlternative(wall_dict, "ControlType", None):
            self.add_servo_wall(facet_num, wall_dict, sims, scene)
            self.print_facet_info(matID, axis, init_v, facet_num, DictIO.GetAlternative(wall_dict, "ControlType", None), servo=True)
        else:
            self.print_facet_info(matID, axis, init_v, facet_num)

        if DictIO.GetAlternative(wall_dict, "Visualize", False):
            self.visualize(sims, scene)

    def add_circular_facet(self, wall_dict, sims: Simulation, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        center = DictIO.GetEssential(wall_dict, "WallCenter")
        radius = DictIO.GetEssential(wall_dict, "WallRadius")
        resolution = DictIO.GetEssential(wall_dict, "Resolution")
        norm = np.asarray(DictIO.GetEssential(wall_dict, "OuterNormal"))
        init_v = DictIO.GetAlternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        norm = norm / np.linalg.norm(norm) if np.linalg.norm(norm) != 0. else norm

        import pygmsh
        with pygmsh.geo.Geometry() as geom:
            geom.add_circle(center, radius, mesh_size=(2. * radius / resolution))
            mesh = geom.generate_mesh()
        points, face = mesh.points, mesh.cells_dict["triangle"]
        facet_num = face.shape[0]
        scene.check_wall_number(sims, body_number=facet_num)
        for f in face:
            p0, p1, p2 = points[f[0]], points[f[1]], points[f[2]]
            scene.wall[int(scene.wallNum[0])].add_materialID(matID)
            scene.wall[int(scene.wallNum[0])].add_wall_geometry(wallID, p0, p1, p2, norm, init_v)
            scene.wallNum[0] += 1

        if sims.max_servo_wall_num > 0. and DictIO.GetAlternative(wall_dict, "ControlType", None):
            self.add_servo_wall(facet_num, wall_dict, sims, scene)
            self.print_facet_info(matID, norm, init_v, facet_num, DictIO.GetAlternative(wall_dict, "ControlType", None), servo=True)
        else:
            self.print_facet_info(matID, norm, init_v, facet_num)

        if DictIO.GetAlternative(wall_dict, "Visualize", False):
            self.visualize(sims, scene)


    def add_file_facet(self, wall_dict, sims: Simulation, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        init_v = DictIO.GetAlternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        direction = DictIO.GetAlternative(wall_dict, "Orientation", np.array([0, 0, 1]))
        iscounterclockwise = DictIO.GetAlternative(wall_dict, "Counterclockwise", None)

        mesh: tm.Trimesh = self.mesh_from_file(wall_dict)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        scene.check_wall_number(sims, body_number=faces.shape[0])
        if iscounterclockwise is None:
            norm = mesh.face_normals
            kernel_add_facet_files(int(scene.wallNum[0]), wallID, matID, vertices, faces, norm, init_v, scene.wall)
        else:
            kernel_add_facet_files_autonorm(iscounterclockwise, int(scene.wallNum[0]), wallID, matID, vertices, faces, init_v, scene.wall)

        facet_num = faces.shape[0]
        scene.wallNum[0] += facet_num

        if sims.max_servo_wall_num > 0. and DictIO.GetAlternative(wall_dict, "ControlType", None):
            self.add_servo_wall(facet_num, wall_dict, sims, scene)
            self.print_facet_info(matID, norm, init_v, facet_num, DictIO.GetAlternative(wall_dict, "ControlType", None), servo=True)
        else:
            self.print_facet_info(matID, direction, init_v, facet_num)

        if DictIO.GetAlternative(wall_dict, "Visualize", False):
            self.visualize_mesh(vertices, faces)

    def add_servo_wall(self, facet_num, wall_dict, sims, scene: myScene):
        scene.check_servo_number(sims, body_number=1)
        alpha = DictIO.GetAlternative(wall_dict, "Alpha", 0.8)
        gain = DictIO.GetAlternative(wall_dict, "Gain", 0.)
        target_stress = DictIO.GetEssential(wall_dict, "TargetStress")
        max_velocity = DictIO.GetEssential(wall_dict, "LimitVelocity")
        startIndex = int(scene.wallNum[0]) - facet_num
        endIndex = int(scene.wallNum[0])
        scene.servo[int(scene.servoNum[0])].add_servo_wall(startIndex, endIndex, alpha, gain, target_stress, max_velocity)
        scene.servoNum[0] += 1


    # ========================================================= #
    #                      Create Patch                         #
    # ========================================================= #
    def add_patch_wall(self, wall_dict, sims, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        direction = DictIO.GetAlternative(wall_dict, "Orientation", np.array([0, 0, 1]))
        shell_offset = DictIO.GetAlternative(wall_dict, "ShellOffset", 0.)
        iscounterclockwise = DictIO.GetAlternative(wall_dict, "Counterclockwise", None)

        mesh: tm.Trimesh = self.mesh_from_file(wall_dict)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces, dtype=np.int32)
        scene.check_wall_number(sims, body_number=faces.shape[0])
        if iscounterclockwise is None:
            norm = mesh.face_normals
            kernel_add_patch(int(scene.wallNum[0]), wallID, matID, shell_offset, vertices, faces, norm, scene.wall)
        else:
            kernel_add_patch_autonorm(iscounterclockwise, int(scene.wallNum[0]), wallID, matID, shell_offset, vertices, faces, scene.wall)
        scene.wallNum[0] += faces.shape[0]
        scene.geometry.append(scene.wallNum[0] - faces.shape[0], scene.wallNum[0], wall_dict, scene.wall)
        self.print_facet_info(matID, direction, None, faces.shape[0], geometry=scene.geometry)

        if DictIO.GetAlternative(wall_dict, "Visualize", False):
            self.visualize_mesh(vertices, faces)


    # ========================================================= #
    #                    Create Level-set                       #
    # ========================================================= #
    def add_digital_elevation_wall(self, wall_dict, sims: Simulation, scene: myScene):
        wallID = DictIO.GetEssential(wall_dict, "WallID")
        matID = DictIO.GetEssential(wall_dict, "MaterialID")
        cell_size = DictIO.GetEssential(wall_dict, "CellSize")
        main_axis = DictIO.GetAlternative(wall_dict, "MainAxis", 'x')
        digital_elevation = DictIO.GetEssential(wall_dict, "DigitalElevation")
        if main_axis == 'y':
            digital_elevation = flip2d(digital_elevation)
        
        grid_number = DictIO.GetAlternative(wall_dict, "GridNumber", digital_elevation.T.shape)
        no_data = DictIO.GetAlternative(wall_dict, "NoData", -9999.)
        cell_number = [int(i - 1) for i in grid_number]
        digital_elevation = np.array(digital_elevation).reshape(-1)

        scene.digital_elevation.set_digital_elevation(matID, cell_size, cell_number, grid_number, no_data, digital_elevation)
        sims.set_digital_elevation_grid_num(grid_number)
        if sims.use_digital_elevation_heightfield():
            print(" Digital Elevation HeightField Information ".center(71, '-'))
            print(("Material ID =  " + str(matID)).ljust(67))
            print(("Grid Number =  " + str(grid_number)).ljust(67))
            print(("Cell Number =  " + str(cell_number)).ljust(67))
            print(("Cell Size =  " + str(cell_size)).ljust(67))
            print(("NoData =  " + str(no_data)).ljust(67))
            print(("Facet wall generation: Disabled").ljust(67))
            print('\n')
        else:
            wall_number = kernel_add_dem_wall(int(scene.wallNum[0]), no_data, cell_size, cell_number, digital_elevation, wallID, matID, scene.wall)
            scene.check_wall_number(sims, body_number=wall_number)
            self.print_facet_info(matID, None, [0., 0., 0.], wall_number)
            scene.wallNum[0] += wall_number

        if DictIO.GetAlternative(wall_dict, "Visualize", False) and not sims.use_digital_elevation_heightfield():
            self.visualize(sims, scene)

    # ========================================================= #
    #                         Reload                            #
    # ========================================================= #
    def restart_walls(self, wall_dict, sims: Simulation, scene: myScene):
        print(" Wall Information ".center(71, '-'))
        wall: str = DictIO.GetAlternative(wall_dict, "WallFile", None)
        servo = DictIO.GetAlternative(wall_dict, "ServoFile", None)
        file_type = DictIO.GetAlternative(wall_dict, "FileType", "NPZ")
        if wall.endswith('npz'):
            file_type = "NPZ"
        elif wall.endswith('txt'):
            file_type = "TXT"

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

            elif sims.wall_type == 1:
                kernel_rebuild_facet(int(scene.wallNum[0]), wall_number, scene.wall, 
                                          DictIO.GetAlternative(wall_info, "active", np.zeros(wall_number) + 1), 
                                          DictIO.GetAlternative(wall_info, "wallID", np.zeros(wall_number)), 
                                          DictIO.GetEssential(wall_info, "materialID"), 
                                          DictIO.GetEssential(wall_info, "point1"), 
                                          DictIO.GetEssential(wall_info, "point2"), 
                                          DictIO.GetEssential(wall_info, "point3"), 
                                          DictIO.GetEssential(wall_info, "norm"), 
                                          DictIO.GetAlternative(wall_info, "velocity", np.zeros((wall_number, 3))))
                print("Inserted facet number: ", wall_number)
            
            elif sims.wall_type == 2:
                kernel_rebuild_patch(int(scene.wallNum[0]), wall_number, scene.wall, 
                                          DictIO.GetAlternative(wall_info, "active", np.zeros(wall_number) + 1), 
                                          DictIO.GetAlternative(wall_info, "wallID", np.zeros(wall_number)), 
                                          DictIO.GetEssential(wall_info, "materialID"), 
                                          DictIO.GetEssential(wall_info, "point1"), 
                                          DictIO.GetEssential(wall_info, "point2"), 
                                          DictIO.GetEssential(wall_info, "point3"), 
                                          DictIO.GetEssential(wall_info, "norm"), 
                                          DictIO.GetAlternative(wall_info, "shell_offset", np.zeros(wall_number)))
                
                geometry_info = np.load(wall, allow_pickle=True) 
                scene.geometry.reload(geometry_info, scene.wall)
                print("Inserted patch number: ", wall_number)
            scene.wallNum[0] += wall_number
        
    def restart_npz_servo(self, wall, servo, sims: Simulation, scene: myScene):    
        if not servo is None:
            if not os.path.exists(servo):
                raise EOFError("Invaild servo path")
            
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
                                 DictIO.GetEssential(servo_info, "gain"), 
                                 DictIO.GetEssential(servo_info, "target_stress"), 
                                 DictIO.GetEssential(servo_info, "max_velocity"))
            scene.servoNum[0] += servo_number
            print("Inserted servo number: ", servo_number)
        print('\n')

    def visualize(self, sims: Simulation, scene: myScene):
        if sims.wall_type == 1 or sims.wall_type == 2 or sims.wall_type == 3:
            ndim = 3 
            point1 = np.ascontiguousarray(scene.wall.vertice1.to_numpy()[0: scene.wallNum[0]])
            point2 = np.ascontiguousarray(scene.wall.vertice2.to_numpy()[0: scene.wallNum[0]])
            point3 = np.ascontiguousarray(scene.wall.vertice3.to_numpy()[0: scene.wallNum[0]])
            points = np.concatenate((point1, point2, point3), axis=1).reshape(-1, ndim)

            point, cell = np.unique(points, axis=0, return_inverse=True)
            faces = cell.reshape((-1, ndim))
            nface = faces.shape[0]
            offset = np.arange(ndim, ndim * nface + 1, ndim)

            unstructuredGridToVTK(f"TriangleWall", np.ascontiguousarray(point[:, 0]), np.ascontiguousarray(point[:, 1]), np.ascontiguousarray(point[:, 2]), 
                              connectivity=np.ascontiguousarray(faces.flatten()), 
                              offsets=np.ascontiguousarray(offset), 
                              cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface)))
        else:
            warnings.warn("Wall type:: /Plane/ is not supported to visualize")

    def visualize_mesh(self, vertices, faces):
        ndim = 3 
        point1 = np.ascontiguousarray(vertices[:, 0])
        point2 = np.ascontiguousarray(vertices[:, 1])
        point3 = np.ascontiguousarray(vertices[:, 2])

        nface = faces.shape[0]
        offset = np.arange(ndim, ndim * nface + 1, ndim)

        unstructuredGridToVTK(f"TriangleWall", np.ascontiguousarray(point1), np.ascontiguousarray(point2), np.ascontiguousarray(point3), 
                            connectivity=np.ascontiguousarray(faces.flatten()), 
                            offsets=np.ascontiguousarray(offset), 
                            cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface)))
