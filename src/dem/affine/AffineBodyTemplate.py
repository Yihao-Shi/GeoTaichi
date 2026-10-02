import numpy as np
import trimesh as tm

from src.physics_model.contact_model.ipc.ContactMeasure import (
    lumped_vertex_measures,
)
from src.physics_model.contact_model.ipc.LevelSetAffine import (
    TrilinearLevelSet,
)
from src.utils.ObjectIO import DictIO
from src.utils.linalg import transformation_matrix_coordinate_system


class AffineBodyTemplate(object):
    """Surface-only template for affine body dynamics.

    The level-set DEM template path builds both a surface mesh and a signed
    distance grid. Affine body dynamics only needs the surface mesh, so this
    class mirrors the mesh normalization used by level-set templates without
    allocating or filling the level-set grid.
    """

    def __init__(self) -> None:
        self.name = "AffineTemplate1"
        self.objects = None
        self.vertices = None
        self.faces = None
        self.volume = 0.0
        self.eqradius = 0.0
        self.center = np.zeros(3)
        self.inertia = np.zeros(3)
        self.bounding_radius = 0.0
        self.x_bound = np.zeros(3)
        self.min_box = np.zeros(3)
        self.max_box = np.zeros(3)
        self.surface_node_number = 0
        self.surface_weights = None
        self.contact_representation = "TriangleMesh"
        self.levelset = None

    def surface_template(self, template_dict):
        print("#", "Start building affine-body surface template ...".ljust(67))
        self.name = DictIO.GetEssential(template_dict, "Name")
        self.objects = DictIO.GetEssential(template_dict, "Object")
        representation = DictIO.GetAlternative(
            template_dict,
            "ContactRepresentation",
            DictIO.GetAlternative(template_dict, "contact_representation", "TriangleMesh"),
        )
        key = str(representation).replace("_", "").replace("-", "").replace(" ", "").lower()
        aliases = {
            "trianglemesh": "TriangleMesh",
            "mesh": "TriangleMesh",
            "surface": "TriangleMesh",
            "levelset": "LevelSet",
            "sdf": "LevelSet",
            "signeddistance": "LevelSet",
            "signeddistancefield": "LevelSet",
        }
        if key not in aliases:
            raise RuntimeError("AffineBody ContactRepresentation must be " "'TriangleMesh' or 'LevelSet'")
        self.contact_representation = aliases[key]
        mesh, template_to_grid, template_to_grid_offset = self._extract_mesh(self.objects)
        self.vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        self.surface_node_number = int(self.vertices.shape[0])
        self.surface_weights = np.ascontiguousarray(
            lumped_vertex_measures(self.vertices, self.faces),
            dtype=np.float64,
        )
        if self.surface_weights.size != self.surface_node_number or np.any(self.surface_weights <= 0.0):
            raise RuntimeError("AffineBody surface quadrature weights must be positive")
        if self.contact_representation == "LevelSet":
            grid = getattr(self.objects, "grid", None)
            if grid is None or getattr(grid, "distance_field", None) is None:
                raise RuntimeError("LevelSet affine contact requires Object.grid with a " "signed distance field")
            self.levelset = TrilinearLevelSet.from_object_grid(
                grid,
                template_to_grid=template_to_grid,
                template_to_grid_offset=template_to_grid_offset,
            )
            grid_vertices = self.vertices @ template_to_grid.T + template_to_grid_offset
            tolerance = 1.0e-8 * max(1.0, float(np.linalg.norm(self.levelset.upper)))
            if np.any(grid_vertices < self.levelset.origin - tolerance) or np.any(
                grid_vertices > self.levelset.upper + tolerance
            ):
                raise RuntimeError("LevelSet affine contact surface is outside its SDF grid")
        self.volume = float(abs(mesh.volume))
        self.eqradius = float((3.0 * self.volume / (4.0 * np.pi)) ** (1.0 / 3.0)) if self.volume > 0.0 else 0.0
        self.center = np.asarray(mesh.center_mass, dtype=np.float64)
        self.inertia = np.asarray(np.diag(mesh.moment_inertia), dtype=np.float64)
        self.min_box = np.asarray(mesh.bounds[0], dtype=np.float64)
        self.max_box = np.asarray(mesh.bounds[1], dtype=np.float64)
        self.x_bound = np.zeros(3, dtype=np.float64)
        self.bounding_radius = float(np.linalg.norm(self.vertices, axis=1).max()) if self.vertices.size else 0.0
        self.print_info()

    def _extract_mesh(self, objects):
        if objects is None or getattr(objects, "mesh", None) is None:
            raise RuntimeError("AffineBody template needs an Object with a surface mesh.")

        mesh = objects.mesh.copy()
        mesh.remove_unreferenced_vertices()
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise RuntimeError("AffineBody template mesh is empty.")

        center = np.asarray(mesh.center_mass, dtype=np.float64)
        mesh.apply_translation(-center)
        rotation = np.eye(3, dtype=np.float64)
        if getattr(objects, "_reset", True):
            _, new_axis = tm.inertia.principal_axis(mesh.moment_inertia)
            rotation_matrix = transformation_matrix_coordinate_system(new_axis, np.eye(3))
            mesh.apply_transform(rotation_matrix)
            rotation = np.asarray(rotation_matrix[:3, :3], dtype=np.float64)
        # template = R (grid - center), hence grid = R^T template + center.
        return mesh, rotation.T, center

    def print_info(self):
        print(" Affine Body Template Information ".center(71, "-"))
        print("Template name: ", self.name)
        print("Volume = ", self.volume)
        print("Equivalent radius = ", self.eqradius)
        print("Bounding radius = ", self.bounding_radius)
        print("Contact representation = ", self.contact_representation)
        print("Number of surface nodes = ", self.surface_node_number)
        print("Number of surface faces = ", int(self.faces.shape[0]), "\n")
