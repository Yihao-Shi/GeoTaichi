from src.utils.errors import SimulationStateError
import os, warnings

from src.dem.generator.InsertionKernel import *
from src.dem.SceneManager import SceneManager
from src.dem.Simulation import Simulation
from src.utils.linalg import flip2d
from src.utils.ObjectIO import DictIO
from src.utils.PolygonDiscretization import *
from src.utils.TypeDefinition import vec3f
from src.utils.linalg import transformation_matrix_direction, rotation_matrix_direction, triangle_normal
from third_party.pyevtk.hl import unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle
import trimesh as tm


class WallGenerator(object):
    def insert_wall(self, wall_dict, sims: Simulation, scene: SceneManager):
        wall_type = DictIO.get_essential(wall_dict, "WallType")
        if wall_type == "Plane":
            if sims.wall_type != 0:
                raise SimulationStateError("Keyword:: /max_plane_number/ should be larger than 0")
            if not scene.wall is None:
                print("#", "Start adding plane(s) ......")
                self.add_plane_wall(wall_dict, sims, scene)
            else:
                raise SimulationStateError("Plane class has not been activated")
        elif wall_type == "Facet":
            if sims.wall_type != 1:
                raise SimulationStateError("Keyword:: /max_facet_number/ should be larger than 0")
            if not scene.wall is None:
                print("#", "Start adding facet(s) ......")
                wall_shape = DictIO.get_essential(wall_dict, "WallShape")
                if wall_shape == "Polygon":
                    self.add_polygon_facet(wall_dict, sims, scene)
                elif wall_shape == "Cylinder":
                    self.add_cylinder_facet(wall_dict, sims, scene)
                elif wall_shape == "Circular":
                    self.add_circular_facet(wall_dict, sims, scene)
                elif wall_shape == "File":
                    self.add_file_facet(wall_dict, sims, scene)
            else:
                raise SimulationStateError("Facet class has not been activated")
        elif wall_type == "Patch":
            if sims.wall_type != 2:
                raise SimulationStateError("Keyword:: /max_patch_number/ should be larger than 0")
            if not scene.wall is None:
                print("#", "Start adding patch(s) ......")
                file_type = DictIO.get_alternative(wall_dict, "FileType", None)
                if file_type == "geo":
                    pass
                elif file_type == "txt":
                    pass
                else:
                    self.add_patch_wall(wall_dict, sims, scene)
            else:
                raise SimulationStateError("Patch class has not been activated")
        elif wall_type == "DigitalElevation":
            if sims.wall_type != 3:
                raise SimulationStateError("Keyword:: /max_digital_elevation_number/ should be larger than 0")
            if not scene.wall is None:
                print("#", "Start adding digital elevation model(s) ......")
                self.add_digital_elevation_wall(wall_dict, sims, scene)
            else:
                raise SimulationStateError("Other wall class should not been activated")
        else:
            raise ValueError("Wall Shape Type error!")

    def print_plane_info(self, matID, point, norm):
        print(" Wall Information ".center(71, "-"))
        print("Generate Type: Create Plane")
        print("Material ID = ", matID)
        print("The center the wall = ", point)
        print("The normal direction of the wall = ", norm, "\n")

    def print_facet_info(self, matID, norm, init_v, facet_count, control_type=None, servo=False, geometry=None):
        print(" Wall Information ".center(71, "-"))
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
        print("\n")

    def mesh_from_file(self, wall_dict):
        file = DictIO.get_essential(wall_dict, "WallFile")
        scale = DictIO.get_alternative(wall_dict, "ScaleFactor", 1.0)
        offset = DictIO.get_alternative(wall_dict, "Translation", np.array([0, 0, 0]))
        direction = DictIO.get_alternative(wall_dict, "Orientation", np.array([0, 0, 1]))

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
    def add_plane_wall(self, wall_dict, sims, scene: SceneManager):
        wall_id = DictIO.get_alternative(wall_dict, "WallID", int(scene.wallNum[0]))
        mat_id = DictIO.get_essential(wall_dict, "MaterialID")
        center = DictIO.get_essential(wall_dict, "WallCenter")
        norm = DictIO.get_essential(wall_dict, "OuterNormal")
        norm = norm / np.linalg.norm(norm) if np.linalg.norm(norm) != 0.0 else norm
        scene.check_wall_number(sims, body_number=1)
        wall_num = int(scene.wallNum[0])
        scene.wall[wall_num].add_material_id(mat_id)
        scene.wall[wall_num].add_wall_geometry(wall_id, center, norm)
        scene.wallNum[0] += 1
        self.print_plane_info(mat_id, center, norm)

    # ========================================================= #
    #                      Create Facet                         #
    # ========================================================= #
    def tranverse_poly(self, poly_arr):
        normalized_poly, scalar, offset = poly_pretreatment(poly_arr)
        poly = Polygon(normalized_poly)
        return poly, scalar, offset

    def discretize_wall_to_facet(self, poly):
        wall_facet = []
        wall_facet = get_div_tri(poly, wall_facet)
        new_wall_facet = opt_alltris(wall_facet)
        new_wall_facet = opt_alltris(new_wall_facet)
        return new_wall_facet

    def rotate_wall(self, poly_arr, origin, target):
        matrix = rotation_matrix_direction(origin, target)
        for ver in range(len(poly_arr)):
            poly_arr[ver] = matrix @ poly_arr[ver]
        return poly_arr

    def add_polygon_facet(self, wall_dict, sims: Simulation, scene: SceneManager):
        wall_id = DictIO.get_essential(wall_dict, "WallID")
        mat_id = DictIO.get_essential(wall_dict, "MaterialID")
        vertices = DictIO.get_essential(wall_dict, "WallVertice")
        norm = np.asarray(DictIO.get_essential(wall_dict, "OuterNormal"))
        init_v = DictIO.get_alternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        norm = norm / np.linalg.norm(norm) if np.linalg.norm(norm) != 0.0 else norm

        poly_arr = np.array([list(item) for item in vertices.values()])

        if np.linalg.norm(np.cross(norm, vec3f([0, 0, 1]))) != 0:
            poly_arr = self.rotate_wall(poly_arr, origin=norm, target=np.array([0, 0, 1]))

        poly, scalar, offset = self.tranverse_poly(poly_arr)
        if is_simple_poly(poly):
            new_wall_facet = self.discretize_wall_to_facet(poly)
            for tri in new_wall_facet:
                wall_vertices = np.array([*tri.exterior.coords])
                wall_vertices = np.multiply(wall_vertices, scalar) + offset

                if np.linalg.norm(np.cross(norm, vec3f([0, 0, 1]))) != 0:
                    wall_vertices = self.rotate_wall(wall_vertices, origin=vec3f([0, 0, 1]), target=norm)

                scene.check_wall_number(sims, body_number=1)
                scene.wall[int(scene.wallNum[0])].add_material_id(mat_id)
                scene.wall[int(scene.wallNum[0])].add_wall_geometry(
                    wall_id, wall_vertices[0], wall_vertices[1], wall_vertices[2], norm, init_v
                )
                scene.wallNum[0] += 1

            facet_num = len(new_wall_facet)

            if sims.max_servo_wall_num > 0.0 and DictIO.get_alternative(wall_dict, "ControlType", None):
                self.add_servo_wall(facet_num, wall_dict, sims, scene)
                self.print_facet_info(
                    mat_id, norm, init_v, facet_num, DictIO.get_alternative(wall_dict, "ControlType", None), servo=True
                )
            else:
                self.print_facet_info(mat_id, norm, init_v, facet_num)

            if DictIO.get_alternative(wall_dict, "Visualize", False):
                self.visualize(sims, scene)
        else:
            raise ValueError("Wall vertices error!")

    def cylinder_norm(self, v0, v1, v2, center, axis, inner=True):
        n_tri = triangle_normal(v0, v1, v2)
        pc = (v0 + v1 + v2) / 3.0
        proj = center + np.dot(pc - center, axis) * axis
        r = pc - proj
        if (np.dot(n_tri, r) < 0) is inner:
            n_tri = -n_tri
        return n_tri

    def add_cylinder_facet(self, wall_dict, sims: Simulation, scene: SceneManager):
        wall_id = DictIO.get_essential(wall_dict, "WallID")
        mat_id = DictIO.get_essential(wall_dict, "MaterialID")
        center = DictIO.get_essential(wall_dict, "WallBase")
        radius = DictIO.get_essential(wall_dict, "WallRadius")
        height = DictIO.get_essential(wall_dict, "WallHeight")
        resolution = DictIO.get_essential(wall_dict, "Resolution")
        direction = DictIO.get_essential(wall_dict, "Direction", "Inner")
        axis = np.asarray(DictIO.get_essential(wall_dict, "Axis"))
        init_v = DictIO.get_alternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0.0 else axis
        direction = True if direction == "Inner" else False

        if abs(axis[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])

        u = np.cross(axis, a)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        top_center = center + height * axis

        facet_num = resolution
        for i in range(resolution):
            theta0 = 2.0 * np.pi * i / resolution
            theta1 = 2.0 * np.pi * (i + 1) / resolution

            p0 = center + radius * (np.cos(theta0) * u + np.sin(theta0) * v)
            p1 = center + radius * (np.cos(theta1) * u + np.sin(theta1) * v)

            theta0p = top_center + radius * (np.cos(theta0) * u + np.sin(theta0) * v)
            theta1p = top_center + radius * (np.cos(theta1) * u + np.sin(theta1) * v)

            normal1 = self.cylinder_norm(p0, theta0p, theta1p, center, axis, direction)
            scene.check_wall_number(sims, body_number=1)
            scene.wall[int(scene.wallNum[0])].add_material_id(mat_id)
            scene.wall[int(scene.wallNum[0])].add_wall_geometry(wall_id, p0, theta0p, theta1p, normal1, init_v)
            scene.wallNum[0] += 1

            normal2 = self.cylinder_norm(p0, theta1p, p1, center, axis, direction)
            scene.check_wall_number(sims, body_number=1)
            scene.wall[int(scene.wallNum[0])].add_material_id(mat_id)
            scene.wall[int(scene.wallNum[0])].add_wall_geometry(wall_id, p0, theta1p, p1, normal2, init_v)
            scene.wallNum[0] += 1

        if sims.max_servo_wall_num > 0.0 and DictIO.get_alternative(wall_dict, "ControlType", None):
            self.add_servo_wall(facet_num, wall_dict, sims, scene)
            self.print_facet_info(
                mat_id, axis, init_v, facet_num, DictIO.get_alternative(wall_dict, "ControlType", None), servo=True
            )
        else:
            self.print_facet_info(mat_id, axis, init_v, facet_num)

        if DictIO.get_alternative(wall_dict, "Visualize", False):
            self.visualize(sims, scene)

    def add_circular_facet(self, wall_dict, sims: Simulation, scene: SceneManager):
        wall_id = DictIO.get_essential(wall_dict, "WallID")
        mat_id = DictIO.get_essential(wall_dict, "MaterialID")
        center = DictIO.get_essential(wall_dict, "WallCenter")
        radius = DictIO.get_essential(wall_dict, "WallRadius")
        resolution = DictIO.get_essential(wall_dict, "Resolution")
        norm = np.asarray(DictIO.get_essential(wall_dict, "OuterNormal"))
        init_v = DictIO.get_alternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        norm = norm / np.linalg.norm(norm) if np.linalg.norm(norm) != 0.0 else norm

        import pygmsh

        with pygmsh.geo.Geometry() as geom:
            geom.add_circle(center, radius, mesh_size=(2.0 * radius / resolution))
            mesh = geom.generate_mesh()
        points, face = mesh.points, mesh.cells_dict["triangle"]
        facet_num = face.shape[0]
        scene.check_wall_number(sims, body_number=facet_num)
        for f in face:
            p0, p1, p2 = points[f[0]], points[f[1]], points[f[2]]
            scene.wall[int(scene.wallNum[0])].add_material_id(mat_id)
            scene.wall[int(scene.wallNum[0])].add_wall_geometry(wall_id, p0, p1, p2, norm, init_v)
            scene.wallNum[0] += 1

        if sims.max_servo_wall_num > 0.0 and DictIO.get_alternative(wall_dict, "ControlType", None):
            self.add_servo_wall(facet_num, wall_dict, sims, scene)
            self.print_facet_info(
                mat_id, norm, init_v, facet_num, DictIO.get_alternative(wall_dict, "ControlType", None), servo=True
            )
        else:
            self.print_facet_info(mat_id, norm, init_v, facet_num)

        if DictIO.get_alternative(wall_dict, "Visualize", False):
            self.visualize(sims, scene)

    def add_file_facet(self, wall_dict, sims: Simulation, scene: SceneManager):
        wall_id = DictIO.get_essential(wall_dict, "WallID")
        mat_id = DictIO.get_essential(wall_dict, "MaterialID")
        init_v = DictIO.get_alternative(wall_dict, "InitialVelocity", vec3f([0, 0, 0]))
        direction = DictIO.get_alternative(wall_dict, "Orientation", np.array([0, 0, 1]))
        iscounterclockwise = DictIO.get_alternative(wall_dict, "Counterclockwise", None)

        mesh: tm.Trimesh = self.mesh_from_file(wall_dict)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        scene.check_wall_number(sims, body_number=faces.shape[0])
        if iscounterclockwise is None:
            norm = mesh.face_normals
            kernel_add_facet_files(int(scene.wallNum[0]), wall_id, mat_id, vertices, faces, norm, init_v, scene.wall)
        else:
            kernel_add_facet_files_autonorm(
                iscounterclockwise, int(scene.wallNum[0]), wall_id, mat_id, vertices, faces, init_v, scene.wall
            )

        facet_num = faces.shape[0]
        scene.wallNum[0] += facet_num

        if sims.max_servo_wall_num > 0.0 and DictIO.get_alternative(wall_dict, "ControlType", None):
            self.add_servo_wall(facet_num, wall_dict, sims, scene)
            self.print_facet_info(
                mat_id, norm, init_v, facet_num, DictIO.get_alternative(wall_dict, "ControlType", None), servo=True
            )
        else:
            self.print_facet_info(mat_id, direction, init_v, facet_num)

        if DictIO.get_alternative(wall_dict, "Visualize", False):
            self.visualize_mesh(vertices, faces)

    def add_servo_wall(self, facet_num, wall_dict, sims, scene: SceneManager):
        scene.check_servo_number(sims, body_number=1)
        alpha = DictIO.get_alternative(wall_dict, "Alpha", 0.8)
        gain = DictIO.get_alternative(wall_dict, "Gain", 0.0)
        target_stress = DictIO.get_essential(wall_dict, "TargetStress")
        max_velocity = DictIO.get_essential(wall_dict, "LimitVelocity")
        start_index = int(scene.wallNum[0]) - facet_num
        end_index = int(scene.wallNum[0])
        scene.servo[int(scene.servoNum[0])].add_servo_wall(
            start_index, end_index, alpha, gain, target_stress, max_velocity
        )
        scene.servoNum[0] += 1

    # ========================================================= #
    #                      Create Patch                         #
    # ========================================================= #
    def add_patch_wall(self, wall_dict, sims, scene: SceneManager):
        wall_id = DictIO.get_essential(wall_dict, "WallID")
        mat_id = DictIO.get_essential(wall_dict, "MaterialID")
        direction = DictIO.get_alternative(wall_dict, "Orientation", np.array([0, 0, 1]))
        shell_offset = DictIO.get_alternative(wall_dict, "ShellOffset", 0.0)
        iscounterclockwise = DictIO.get_alternative(wall_dict, "Counterclockwise", None)

        mesh: tm.Trimesh = self.mesh_from_file(wall_dict)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces, dtype=np.int32)
        scene.check_wall_number(sims, body_number=faces.shape[0])
        if iscounterclockwise is None:
            norm = mesh.face_normals
            kernel_add_patch(int(scene.wallNum[0]), wall_id, mat_id, shell_offset, vertices, faces, norm, scene.wall)
        else:
            kernel_add_patch_autonorm(
                iscounterclockwise, int(scene.wallNum[0]), wall_id, mat_id, shell_offset, vertices, faces, scene.wall
            )
        scene.wallNum[0] += faces.shape[0]
        scene.geometry.append(scene.wallNum[0] - faces.shape[0], scene.wallNum[0], wall_dict, scene.wall)
        self.print_facet_info(mat_id, direction, None, faces.shape[0], geometry=scene.geometry)

        if DictIO.get_alternative(wall_dict, "Visualize", False):
            self.visualize_mesh(vertices, faces)

    # ========================================================= #
    #                    Create Level-set                       #
    # ========================================================= #
    def add_digital_elevation_wall(self, wall_dict, sims: Simulation, scene: SceneManager):
        wall_id = DictIO.get_essential(wall_dict, "WallID")
        mat_id = DictIO.get_essential(wall_dict, "MaterialID")
        cell_size = DictIO.get_essential(wall_dict, "CellSize")
        main_axis = DictIO.get_alternative(wall_dict, "MainAxis", "x")
        digital_elevation = DictIO.get_essential(wall_dict, "DigitalElevation")
        if main_axis == "y":
            digital_elevation = flip2d(digital_elevation)

        grid_number = DictIO.get_alternative(wall_dict, "GridNumber", digital_elevation.T.shape)
        no_data = DictIO.get_alternative(wall_dict, "NoData", -9999.0)
        cell_number = [int(i - 1) for i in grid_number]
        digital_elevation = np.array(digital_elevation).reshape(-1)

        scene.digital_elevation.set_digital_elevation(
            mat_id, cell_size, cell_number, grid_number, no_data, digital_elevation
        )
        sims.set_digital_elevation_grid_num(grid_number)
        if sims.use_digital_elevation_heightfield():
            print(" Digital Elevation HeightField Information ".center(71, "-"))
            print(("Material ID =  " + str(mat_id)).ljust(67))
            print(("Grid Number =  " + str(grid_number)).ljust(67))
            print(("Cell Number =  " + str(cell_number)).ljust(67))
            print(("Cell Size =  " + str(cell_size)).ljust(67))
            print(("NoData =  " + str(no_data)).ljust(67))
            print(("Facet wall generation: Disabled").ljust(67))
            print("\n")
        else:
            wall_number = kernel_add_dem_wall(
                int(scene.wallNum[0]), no_data, cell_size, cell_number, digital_elevation, wall_id, mat_id, scene.wall
            )
            scene.check_wall_number(sims, body_number=wall_number)
            self.print_facet_info(mat_id, None, [0.0, 0.0, 0.0], wall_number)
            scene.wallNum[0] += wall_number

        if DictIO.get_alternative(wall_dict, "Visualize", False) and not sims.use_digital_elevation_heightfield():
            self.visualize(sims, scene)

    # ========================================================= #
    #                         Reload                            #
    # ========================================================= #
    def restart_walls(self, wall_dict, sims: Simulation, scene: SceneManager):
        print(" Wall Information ".center(71, "-"))
        wall: str = DictIO.get_alternative(wall_dict, "WallFile", None)
        servo = DictIO.get_alternative(wall_dict, "ServoFile", None)
        file_type = DictIO.get_alternative(wall_dict, "FileType", "NPZ")
        if wall.endswith("npz"):
            file_type = "NPZ"
        elif wall.endswith("txt"):
            file_type = "TXT"

        if file_type == "NPZ":
            self.restart_npz_wall(wall, sims, scene)
            self.restart_npz_servo(wall, servo, sims, scene)
        elif file_type == "TXT":
            pass

        if DictIO.get_alternative(wall_dict, "Visualize", False):
            self.visualize(sims, scene)

    def restart_npz_wall(self, wall, sims: Simulation, scene: SceneManager):
        if not wall is None:
            if not os.path.exists(wall):
                raise EOFError("Invaild wall path")

            wall_info = np.load(wall, allow_pickle=True)
            wall_number = int(DictIO.get_essential(wall_info, "body_num"))
            if sims.is_continue:
                sims.current_time = DictIO.get_alternative(wall_info, "t_current", 0)
                sims.CurrentTime[None] = DictIO.get_alternative(wall_info, "t_current", 0)

            scene.check_wall_number(sims, body_number=wall_number)
            if sims.wall_type == 0:
                kernel_rebuild_plane(
                    int(scene.wallNum[0]),
                    wall_number,
                    scene.wall,
                    DictIO.get_alternative(wall_info, "active", np.zeros(wall_number) + 1),
                    DictIO.get_alternative(wall_info, "wallID", np.zeros(wall_number)),
                    DictIO.get_essential(wall_info, "materialID"),
                    DictIO.get_essential(wall_info, "point"),
                    DictIO.get_essential(wall_info, "norm"),
                )
                print("Inserted plane number: ", wall_number)

            elif sims.wall_type == 1:
                kernel_rebuild_facet(
                    int(scene.wallNum[0]),
                    wall_number,
                    scene.wall,
                    DictIO.get_alternative(wall_info, "active", np.zeros(wall_number) + 1),
                    DictIO.get_alternative(wall_info, "wallID", np.zeros(wall_number)),
                    DictIO.get_essential(wall_info, "materialID"),
                    DictIO.get_essential(wall_info, "point1"),
                    DictIO.get_essential(wall_info, "point2"),
                    DictIO.get_essential(wall_info, "point3"),
                    DictIO.get_essential(wall_info, "norm"),
                    DictIO.get_alternative(wall_info, "velocity", np.zeros((wall_number, 3))),
                )
                print("Inserted facet number: ", wall_number)

            elif sims.wall_type == 2:
                kernel_rebuild_patch(
                    int(scene.wallNum[0]),
                    wall_number,
                    scene.wall,
                    DictIO.get_alternative(wall_info, "active", np.zeros(wall_number) + 1),
                    DictIO.get_alternative(wall_info, "wallID", np.zeros(wall_number)),
                    DictIO.get_essential(wall_info, "materialID"),
                    DictIO.get_essential(wall_info, "point1"),
                    DictIO.get_essential(wall_info, "point2"),
                    DictIO.get_essential(wall_info, "point3"),
                    DictIO.get_essential(wall_info, "norm"),
                    DictIO.get_alternative(wall_info, "shell_offset", np.zeros(wall_number)),
                )

                geometry_info = np.load(wall, allow_pickle=True)
                scene.geometry.reload(geometry_info, scene.wall)
                print("Inserted patch number: ", wall_number)
            scene.wallNum[0] += wall_number

    def restart_npz_servo(self, wall, servo, sims: Simulation, scene: SceneManager):
        if not servo is None:
            if not os.path.exists(servo):
                raise EOFError("Invaild servo path")

            if sims.max_servo_wall_num <= 0:
                raise SimulationStateError("/max_servo_number/ should be larger than zero")

            if sims.wall_type != 1:
                raise SimulationStateError("/max_facet_number/ has not been set")

            if wall is None:
                raise EOFError("Invalid path to read wall information")

            servo_info = np.load(servo, allow_pickle=True)
            servo_number = int(DictIO.get_essential(servo_info, "body_num"))

            scene.check_servo_number(sims, body_number=servo_number)
            kernel_rebuild_servo(
                int(scene.servoNum[0]),
                servo_number,
                scene.servo,
                DictIO.get_alternative(servo_info, "active", np.zeros(servo_number) + 1),
                DictIO.get_essential(servo_info, "startIndex"),
                DictIO.get_essential(servo_info, "endIndex"),
                DictIO.get_essential(servo_info, "alpha"),
                DictIO.get_essential(servo_info, "gain"),
                DictIO.get_essential(servo_info, "target_stress"),
                DictIO.get_essential(servo_info, "max_velocity"),
            )
            scene.servoNum[0] += servo_number
            print("Inserted servo number: ", servo_number)
        print("\n")

    def visualize(self, sims: Simulation, scene: SceneManager):
        if sims.wall_type == 1 or sims.wall_type == 2 or sims.wall_type == 3:
            ndim = 3
            point1 = np.ascontiguousarray(scene.wall.vertice1.to_numpy()[0 : scene.wallNum[0]])
            point2 = np.ascontiguousarray(scene.wall.vertice2.to_numpy()[0 : scene.wallNum[0]])
            point3 = np.ascontiguousarray(scene.wall.vertice3.to_numpy()[0 : scene.wallNum[0]])
            points = np.concatenate((point1, point2, point3), axis=1).reshape(-1, ndim)

            point, cell = np.unique(points, axis=0, return_inverse=True)
            faces = cell.reshape((-1, ndim))
            nface = faces.shape[0]
            offset = np.arange(ndim, ndim * nface + 1, ndim)

            unstructuredGridToVTK(
                f"TriangleWall",
                np.ascontiguousarray(point[:, 0]),
                np.ascontiguousarray(point[:, 1]),
                np.ascontiguousarray(point[:, 2]),
                connectivity=np.ascontiguousarray(faces.flatten()),
                offsets=np.ascontiguousarray(offset),
                cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface)),
            )
        else:
            warnings.warn("Wall type:: /Plane/ is not supported to visualize")

    def visualize_mesh(self, vertices, faces):
        ndim = 3
        point1 = np.ascontiguousarray(vertices[:, 0])
        point2 = np.ascontiguousarray(vertices[:, 1])
        point3 = np.ascontiguousarray(vertices[:, 2])

        nface = faces.shape[0]
        offset = np.arange(ndim, ndim * nface + 1, ndim)

        unstructuredGridToVTK(
            f"TriangleWall",
            np.ascontiguousarray(point1),
            np.ascontiguousarray(point2),
            np.ascontiguousarray(point3),
            connectivity=np.ascontiguousarray(faces.flatten()),
            offsets=np.ascontiguousarray(offset),
            cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface)),
        )
