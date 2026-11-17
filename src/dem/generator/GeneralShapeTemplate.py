from src.utils.errors import GeometryError
from src.utils.linalg import lumped_vertex_measures
from src.utils.errors import ConfigurationError, SimulationStateError
import numpy as np

from src.sdf.BasicShape import BasicShape
from src.dem.generator.Boundings import Boundings
from src.utils.ObjectIO import DictIO


class GeneralShapeTemplate(object):
    objects: BasicShape

    def __init__(self) -> None:
        self.set_up = 1.0
        self.name = "Template1"
        self.ray_path = "Spiral"
        self.objects = None
        self.boundings = None
        self.parameter = None
        self.surface_area = None
        self.length_size = None
        self.soft_template = False
        self.surface_resolution = 2**22
        self.surface_node_number = 0

    def levelset_template(self, template_dict):
        print("#", "Start calculating properties of level-set template ...".ljust(67))
        self.name = DictIO.get_essential(template_dict, "Name")
        self.objects = DictIO.get_essential(template_dict, "Object")
        self.ray_path = DictIO.get_alternative(template_dict, "RayPath", "Spiral")
        self.surface_resolution = DictIO.get_alternative(template_dict, "SurfaceResolution", self.surface_resolution)
        self.surface_node_number = DictIO.get_alternative(template_dict, "SurfaceNodeNumber", self.surface_node_number)
        self.write_file = DictIO.get_alternative(template_dict, "WriteFile", False)
        self.save_path = DictIO.get_alternative(template_dict, "SavePath", "./")
        self.visualize_mode = DictIO.get_alternative(template_dict, "VisualizeMode", None)
        self.length_size = DictIO.get_alternative(template_dict, "LengthSize", None)

        if not self.visualize_mode in ["gui", "matplot", None]:
            raise ValueError("VisualizeMode must be gui, matplot, or None")

        if not self.ray_path in ["Rectangle", "Spiral"]:
            raise ValueError("RayPath must be Rectangle or Spiral")

        if self.surface_node_number <= 2 and self.surface_node_number != 0:
            raise SimulationStateError(
                "You asked for a level set shape with no more than two boundary nodes, for contact detection purposes. \
                               This is too few and will lead to square roots of negative numbers, then unexpected events."
            )

        self.build()
        self.multibody_template_initialize()
        self.print_info()
        self.visualize()
        self.write()

    def build(self):
        if not self.objects is None:
            if self.objects.ray:
                self.objects.generate(samples=self.surface_node_number, ray_path=self.ray_path)
            else:
                self.objects.generate(samples=self.surface_resolution)
            if not self.objects.mesh.is_watertight:
                raise ConfigurationError("Level-set surface mesh must be closed and watertight")
            self.objects._essential_initialize()
        else:
            raise SimulationStateError("Keyword:: /Objects/ is None")

        if self.objects.mesh.vertices.shape[0] != self.surface_node_number:
            self.surface_node_number = self.objects.mesh.vertices.shape[0]

    def multibody_template_initialize(self):
        self.boundings = Boundings()
        # self.boundings.create_boundings(self.mesh.vertices, self.mesh.bounding_sphere.center, self.mesh.bounding_sphere.radius)
        self.boundings.set_boundings(
            self.objects.mesh.bounding_sphere.center,
            self.objects.mesh.bounding_sphere.primitive.radius,
            self.objects.mesh.center_mass,
            self.objects.mesh.bounding_box.extents,
        )
        # Some mesh backends return an approximate bounding sphere.  Contact
        # broad phases require a conservative radius, so retain its center and
        # enlarge only enough to contain every realized surface vertex.
        surface_vertices = np.asarray(self.objects.mesh.vertices, dtype=np.float64)
        realized_radius = float(
            np.max(
                np.linalg.norm(
                    surface_vertices - self.boundings.x_bound[None, :],
                    axis=1,
                )
            )
        )
        if realized_radius > self.boundings.r_bound:
            self.boundings.r_bound = np.nextafter(realized_radius, np.inf)
        self.calculate_surface_parameter()

    def calculate_surface_parameter(self):
        nodal_area = lumped_vertex_measures(self.objects.mesh.vertices, self.objects.mesh.faces)
        self.surface_area = float(np.sum(nodal_area))
        if not np.isfinite(self.surface_area) or self.surface_area <= 0.0:
            raise GeometryError("Level-set surface quadrature has a non-positive total nodal area")
        if np.any(~np.isfinite(nodal_area)) or np.any(nodal_area <= 0.0):
            raise GeometryError("Level-set surface quadrature requires positive nodal areas")
        self.parameter = nodal_area / self.surface_area

    def print_info(self):
        print(" Level-set Template Information ".center(71, "-"))
        print("Template name: ", self.name)
        print("Volume = ", self.objects.volume)
        print("Surface area = ", self.surface_area)
        print("Equivalent radius = ", self.objects.eqradius)
        print("Center of mass = ", self.objects.center)
        print("Inertia tensor = ", self.objects.inertia)
        print("Center of bounding sphere = ", self.boundings.x_bound)
        print("Radius of bounding sphere = ", self.boundings.r_bound)
        print("Bounding box = ", self.objects.grid.min_box(), " --> ", self.objects.grid.max_box())
        print("Number of grid = ", self.objects.grid.gnum)
        print("Number of surface nodes = ", self.surface_node_number, "\n")

    def visualize(self):
        if self.visualize_mode == "gui":
            self.objects.mesh.show()
        elif self.visualize_mode == "matplot":
            self.objects.show()

    def write(self):
        if self.write_file:
            self.objects.dump_files(path=self.save_path, pname=self.name + "Particle", gname=self.name + "Grid")
            self.objects.visualize(
                path=self.save_path, pname=self.name + "Particle", gname=self.name + "Grid", bname=self.name + "Box"
            )
