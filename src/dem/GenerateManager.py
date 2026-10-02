import numpy as np
import taichi as ti

from src.dem.generator.BodyGenerator import ParticleCreator, ParticleGenerator
from src.dem.generator.LoadFromFile import ParticleReader
from src.dem.generator.ClumpTemplate import ClumpTemplate
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.affine.AffineBodyTemplate import AffineBodyTemplate
from src.dem.generator.WallGenerator import WallGenerator
from src.dem.SceneManager import myScene
from src.utils.ObjectIO import DictIO
from src.utils.FieldIO import runtime_float_numpy_dtype
from src.utils.RegionFunction import RegionFunction


class GenerateManager(object):
    def __init__(self):
        self.myRegion = dict()
        self.myGenerator = []
        self.myTemplate = dict()
        self.wallGenerator = WallGenerator()
        self.bodyCreator = ParticleCreator()

    def add_my_region(self, dims, domain, region_dict):
        name = DictIO.GetEssential(region_dict, "Name")
        if name in self.myRegion:
            region: RegionFunction = self.myRegion[name]
            region.finalize()
            del self.myRegion[name]
            self.add_region(dims, domain, name, region_dict)
        else:
            self.add_region(dims, domain, name, region_dict)

    def add_region(self, dims, domain, name, region_dict):
        DictIO.append(self.myRegion, name, RegionFunction(dims, "DEM"))
        region: RegionFunction = self.myRegion[name]
        region.set_region(region_dict)
        region.check_in_domain(domain)

    def get_region_ptr(self, name):
        if not self.myRegion is None:
            return self.myRegion[name]
        else:
            raise RuntimeError("Region class should be activated first!")

    def check_template_name(self, template_dict):
        name = DictIO.GetEssential(template_dict, "Name")
        if name in self.myTemplate:
            del self.myTemplate[name]
        return name

    def add_my_template(self, scene, template_dict, types):
        if type(template_dict) is dict:
            name = self.check_template_name(template_dict)
            self.add_template(name, template_dict, types, scene)
        elif type(template_dict) is list:
            for dicts in template_dict:
                name = self.check_template_name(dicts)
                self.add_template(name, dicts, types, scene)

    def add_template(self, name, template_dict, types, scene: myScene):
        if types == "DEM":
            if not "Pebble" in template_dict:
                raise RuntimeError("Plase double check if the scheme is set as /DEM/")
            DictIO.append(self.myTemplate, name, ClumpTemplate())
            template_ptr: ClumpTemplate = self.myTemplate[name]
            template_ptr.clump_template(template_dict)
            template_ptr.clear()
        elif types == "LSDEM" or types == "LSMPM" or types == "PolySuperEllipsoid" or types == "PolySuperQuadrics":
            if not "Object" in template_dict:
                raise RuntimeError("Plase double check if the scheme is set as /LSDEM/")
            DictIO.append(self.myTemplate, name, GeneralShapeTemplate())
            ltemplate_ptr: GeneralShapeTemplate = self.myTemplate[name]
            ltemplate_ptr.levelset_template(template_dict)
            if types == "LSDEM" or types == "LSMPM":
                if ltemplate_ptr.soft_template is False:
                    scene.add_rigid_levelset_template(
                        name,
                        ltemplate_ptr.objects.grid.distance_field,
                        ltemplate_ptr.objects.mesh.vertices,
                        ltemplate_ptr.parameter,
                    )
            elif types == "PolySuperEllipsoid" or types == "PolySuperQuadrics":
                scene.add_rigid_implicit_surface_template(
                    name, ltemplate_ptr.objects.mesh.vertices, ltemplate_ptr.objects.physical_parameter
                )
            ltemplate_ptr.clear()
        elif types == "AffineBody":
            if not "Object" in template_dict:
                raise RuntimeError("Plase double check if the scheme is set as /AffineBody/")
            DictIO.append(self.myTemplate, name, AffineBodyTemplate())
            atemplate_ptr: AffineBodyTemplate = self.myTemplate[name]
            atemplate_ptr.surface_template(template_dict)
            scene.add_affine_template(name, atemplate_ptr)
        else:
            valid_list = ["DEM", "LSDEM", "LSMPM", "PolySuperEllipsoid", "PolySuperQuadrics", "AffineBody"]
            raise RuntimeError(f"Only {valid_list} is valid for Keyword:: /types/")

    def insert_template_to_creator(self):
        self.bodyCreator.set_template(self.myTemplate)

    def insert_template_to_generator(self, generator: ParticleGenerator):
        if (
            generator.btype == "Clump"
            or generator.btype == "RigidBody"
            or generator.btype == "AffineBody"
            or generator.btype == "SoftBody"
        ):
            if (
                generator.btype == "Clump"
                or generator.btype == "AffineBody"
                or generator.btype == "SoftBody"
                or (generator.btype == "RigidBody" and generator.write_file is False)
            ):
                if len(self.myTemplate) == 0:
                    raise RuntimeError("The template must be set first")
            generator.set_template(self.myTemplate)

    def insert_region_to_generator(self, generator: ParticleGenerator):
        if generator.type == "Generate" or generator.type == "Distribute" or generator.type == "Lattice":
            if len(self.myRegion) == 0:
                raise RuntimeError("The region must be set first")
            generator.set_region(self.myRegion)

    def create_body(self, body_dict, sims, scene):
        self.insert_template_to_creator()
        self.bodyCreator.create(sims, scene, body_dict)

    def create_body_batch(self, body_dict, sims, scene):
        body_type = DictIO.GetEssential(body_dict, "BodyType")
        if body_type not in ("RigidBody", "SoftBody"):
            raise ValueError("Batch creation currently supports only RigidBody and SoftBody")
        template = DictIO.GetEssential(body_dict, "Template")
        if not isinstance(template, dict):
            raise TypeError("Batch body Template must be a dictionary")

        centers = np.asarray(DictIO.GetEssential(template, "BodyPoints"), dtype=np.float64)
        radii = np.asarray(DictIO.GetEssential(template, "BoundingRadii"), dtype=np.float64)
        orientations = np.asarray(
            DictIO.GetAlternative(
                template,
                "BodyOrientationsRadians",
                np.zeros_like(centers),
            ),
            dtype=np.float64,
        )
        coordinates_are_mass_centers = bool(DictIO.GetAlternative(template, "CoordinatesAreMassCenters", True))
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("BodyPoints must have shape (body_count, 3)")
        body_count = int(centers.shape[0])
        if body_count < 1:
            raise ValueError("Batch body creation requires at least one body")
        if radii.shape != (body_count,):
            raise ValueError("BoundingRadii must have shape (body_count,)")
        if orientations.shape != centers.shape:
            raise ValueError("BodyOrientationsRadians must have shape (body_count, 3)")
        if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(radii)) and np.all(np.isfinite(orientations))):
            raise ValueError("Batch body data must be finite")
        if np.any(radii <= 0.0):
            raise ValueError("BoundingRadii must be positive")

        common_template = dict(template)
        for key in (
            "BodyPoints",
            "BoundingRadii",
            "BodyOrientationsRadians",
            "CoordinatesAreMassCenters",
        ):
            common_template.pop(key, None)

        generator = ParticleGenerator(sims)
        generator.btype = body_type
        generator.template_dict = common_template
        generator.coordinates_are_mass_centers = coordinates_are_mass_centers
        generator.set_template(self.myTemplate)
        generator.allocate_sphere_memory(body_count, levelset=True)
        upload_dtype = runtime_float_numpy_dtype()
        try:
            generator.sphere_coords.from_numpy(np.ascontiguousarray(centers, dtype=upload_dtype))
            generator.sphere_radii.from_numpy(np.ascontiguousarray(radii, dtype=upload_dtype))
            generator.orients.from_numpy(np.ascontiguousarray(orientations, dtype=upload_dtype))
            if body_type == "RigidBody":
                generator.insert_rigid_levelset(scene, common_template, 0, body_count, body_count)
            else:
                backend = generator._soft_particle_backend(scene)
                backend.insert_soft_body_batch(generator, scene, common_template, body_count)
            ti.sync()
        finally:
            # Taichi 1.7 cannot destroy dynamic SNode trees on Metal. Local
            # diagnostics create at most one small tree per phase; the
            # process releases them. CUDA production frees them immediately.
            if ti.lang.impl.current_cfg().arch != ti.metal:
                generator.snode_tree.destroy()

        return {
            "body_type": body_type,
            "body_count": body_count,
            "template_name": DictIO.GetEssential(common_template, "Name"),
            "coordinate_semantics": "center_of_mass",
            "orientation_units": "radians",
        }

    def regenerate(self, scene: myScene):
        if len(self.myGenerator) == 0:
            return 0

        is_insert = 0
        for i in range(len(self.myGenerator)):
            is_insert = self.myGenerator[i].regenerate(scene)
            if not self.myGenerator[i].active:
                self.myGenerator[i].finalize()
                del self.myGenerator[i]
        return is_insert

    def add_body(self, body_dict, sims, scene):
        generator = ParticleGenerator(sims)
        generator.set_system_strcuture(body_dict)
        self.insert_template_to_generator(generator)
        self.insert_region_to_generator(generator)
        generator.begin(scene)
        if generator.active:
            self.myGenerator.append(generator)
        else:
            generator.finalize()

    def read_body_file(self, body_dict, sims, scene: myScene):
        if scene.material is None:
            raise RuntimeError("The attribute must be added first")

        generator = ParticleReader(sims)
        generator.set_system_strcuture(body_dict)
        generator.set_template(self.myTemplate)
        generator.begin(scene)
        if generator.active:
            self.myGenerator.append(generator)
        else:
            generator.finalize()

    def add_wall(self, wall_dict, sims, scene):
        if type(wall_dict) is dict:
            self.wallGenerator.insert_wall(wall_dict, sims, scene)
        elif type(wall_dict) is list:
            for wall in wall_dict:
                self.wallGenerator.insert_wall(wall, sims, scene)

    def read_wall_file(self, wall_dict, sims, scene: myScene):
        if type(wall_dict) is dict:
            self.wallGenerator.restart_walls(wall_dict, sims, scene)
        elif type(wall_dict) is list:
            for wall in wall_dict:
                self.wallGenerator.restart_walls(wall, sims, scene)
