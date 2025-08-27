import numpy as np

from src.sdf.BasicShape import BasicShape
from src.dem.generator.Boundings import Boundings
from src.utils.ObjectIO import DictIO


class GeneralShapeTemplate(object):
    objects : BasicShape

    def __init__(self) -> None:
        self.set_up = 1.
        self.name = "Template1"
        self.ray_path = "Spiral"
        self.objects = None
        self.boundings = None
        self.parameter = None
        self.length_size = None
        self.soft_template = False
        self.surface_resolution = 2 ** 22
        self.surface_node_number = 0

    def levelset_template(self, template_dict):
        print('#', "Start calculating properties of level-set template ...".ljust(67))
        self.name = DictIO.GetEssential(template_dict, "Name")
        self.objects = DictIO.GetEssential(template_dict, "Object")
        self.ray_path = DictIO.GetAlternative(template_dict, "RayPath", "Spiral")
        self.surface_resolution = DictIO.GetAlternative(template_dict, "SurfaceResolution", self.surface_resolution)
        self.surface_node_number = DictIO.GetAlternative(template_dict, "SurfaceNodeNumber", self.surface_node_number)
        self.write_file = DictIO.GetAlternative(template_dict, "WriteFile", False)
        self.save_path = DictIO.GetAlternative(template_dict, "SavePath", './')
        self.visualize_mode = DictIO.GetAlternative(template_dict, "VisualizeMode", None)
        self.length_size = DictIO.GetAlternative(template_dict, "LengthSize", None)

        if not self.visualize_mode in ["gui", "matplot", None]:
            raise RuntimeError
        
        if not self.ray_path in ["Rectangle", "Spiral"]:
            raise RuntimeError

        if self.surface_node_number <= 2 and self.surface_node_number != 0:
            raise RuntimeError("You asked for a level set shape with no more than two boundary nodes, for contact detection purposes. \
                               This is too few and will lead to square roots of negative numbers, then unexpected events.")
        
        self.build()
        self.multibody_template_initialize()
        self.print_info()
        self.visualize()
        self.write()
        self.finalize()

    def clear(self):
        pass

    def finalize(self):
        pass

    def build(self):
        if not self.objects is None:
            if self.objects.ray: 
                self.objects.generate(samples=self.surface_node_number, ray_path=self.ray_path)
            else:
                self.objects.generate(samples=self.surface_resolution)
            self.objects._essential_initialize()
        else:
            raise RuntimeError("Keyword:: /Objects/ is None")

        if self.objects.mesh.vertices.shape[0] != self.surface_node_number:
            self.surface_node_number = self.objects.mesh.vertices.shape[0]

    def multibody_template_initialize(self):
        self.boundings = Boundings()
        # self.boundings.create_boundings(self.mesh.vertices, self.mesh.bounding_sphere.center, self.mesh.bounding_sphere.radius)
        self.boundings.set_boundings(self.objects.mesh.bounding_sphere.center, self.objects.mesh.bounding_sphere.primitive.radius, 
                                     self.objects.mesh.center_mass, self.objects.mesh.bounding_box.extents)
        self.calculate_surface_parameter()

    def calculate_surface_parameter(self):
        vertice_neighbor = self.objects.mesh.vertex_faces.copy()
        face_angle = self.objects.mesh.face_angles.copy()
        face_area = self.objects.mesh.area_faces.copy()
        face_normal = self.objects.mesh.face_normals.copy()

        face_area_normal = face_area.reshape((-1, 1)) * face_normal
        area_normal = np.vstack((np.sum(face_area_normal[vertice_neighbor, 0], where=vertice_neighbor>-1, axis=1),
                                 np.sum(face_area_normal[vertice_neighbor, 1], where=vertice_neighbor>-1, axis=1),
                                 np.sum(face_area_normal[vertice_neighbor, 2], where=vertice_neighbor>-1, axis=1))).T
        
        row = np.repeat(np.arange(vertice_neighbor.shape[0]), vertice_neighbor.shape[1]).reshape(vertice_neighbor.shape)
        which_angle0 = np.where(self.objects.mesh.faces[vertice_neighbor, 0]==row, face_angle[vertice_neighbor, 0], 0)
        which_angle1 = np.where(self.objects.mesh.faces[vertice_neighbor, 1]==row, face_angle[vertice_neighbor, 1], 0)
        which_angle2 = np.where(self.objects.mesh.faces[vertice_neighbor, 2]==row, face_angle[vertice_neighbor, 2], 0)
        which_angle = which_angle0 + which_angle1 + which_angle2
        angle_normals = np.vstack((np.sum(which_angle * face_normal[vertice_neighbor, 0], where=vertice_neighbor>-1, axis=1),
                                   np.sum(which_angle * face_normal[vertice_neighbor, 1], where=vertice_neighbor>-1, axis=1),
                                   np.sum(which_angle * face_normal[vertice_neighbor, 2], where=vertice_neighbor>-1, axis=1))).T
        angle_normal = np.where(np.linalg.norm(angle_normals, axis=1) > 0., (angle_normals.T / np.linalg.norm(angle_normals, axis=1)), 0.).T
        self.parameter = np.sum(angle_normal * area_normal, axis=1) / self.objects.mesh.area
        assert (self.parameter > 0.).all()

    def print_info(self):
        print(" Level-set Template Information ".center(71,"-"))
        print("Template name: ",  self.name)
        print("Volume = ",  self.objects.volume)
        print("Equivalent radius = ",  self.objects.eqradius)
        print("Center of mass = ",  self.objects.center)
        print("Inertia tensor = ",  self.objects.inertia)
        print("Center of bounding sphere = ",  self.boundings.x_bound)
        print("Radius of bounding sphere = ",  self.boundings.r_bound)
        print("Bounding box = ",  self.objects.grid.minBox(), ' --> ', self.objects.grid.maxBox())
        print("Number of grid = ",  self.objects.grid.gnum)
        print("Number of surface nodes = ",  self.surface_node_number, '\n')

    def visualize(self):
        if self.visualize_mode == "gui":
            self.objects.mesh.show()
        elif self.visualize_mode == "matplot":
            self.objects.show()

    def write(self):
        if self.write_file:
            self.objects.dump_files(path=self.save_path, pname=self.name+'Particle', gname=self.name+'Grid')
            self.objects.visualize(path=self.save_path, pname=self.name+'Particle', gname=self.name+'Grid', bname=self.name+'Box')

    def read(self):
        pass