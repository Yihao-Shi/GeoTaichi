from src.dem.generator.BodyGenerator import ParticleCreator, ParticleGenerator
from src.dem.generator.LoadFromFile import ParticleReader
from src.dem.generator.ClumpTemplate import ClumpTemplate
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.generator.WallGenerator import WallGenerator
from src.dem.SceneManager import myScene
from src.utils.ObjectIO import DictIO
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
        elif types == "LSDEM" or types == "PolySuperEllipsoid" or types == "PolySuperQuadrics":
            if not "Object" in template_dict:
                raise RuntimeError("Plase double check if the scheme is set as /LSDEM/")
            DictIO.append(self.myTemplate, name, GeneralShapeTemplate())
            ltemplate_ptr: GeneralShapeTemplate = self.myTemplate[name]
            ltemplate_ptr.levelset_template(template_dict)
            if types == "LSDEM":
                if ltemplate_ptr.soft_template is False:
                    scene.add_rigid_levelset_template(name, ltemplate_ptr.objects.grid.distance_field, ltemplate_ptr.objects.mesh.vertices, ltemplate_ptr.parameter)
            elif types == "PolySuperEllipsoid" or types == "PolySuperQuadrics":
                scene.add_rigid_implicit_surface_template(name, ltemplate_ptr.objects.mesh.vertices, ltemplate_ptr.objects.physical_parameter)
            ltemplate_ptr.clear()
        else:
            valid_list = ["DEM", "LSDEM", "PolySuperEllipsoid", "PolySuperQuadrics"]
            raise RuntimeError(f"Only {valid_list} is valid for Keyword:: /types/")
        
    def insert_template_to_creator(self):
        self.bodyCreator.set_template(self.myTemplate)

    def insert_template_to_generator(self, generator: ParticleGenerator):
        if generator.btype == 'Clump' or generator.btype == 'RigidBody':
            if generator.btype == 'Clump' or (generator.btype == 'RigidBody' and generator.write_file is False):
                if len(self.myTemplate) == 0:
                    raise RuntimeError("The template must be set first")
            generator.set_template(self.myTemplate)

    def insert_region_to_generator(self, generator: ParticleGenerator):
        if generator.type == 'Generate' or generator.type == 'Distribute' or generator.type == 'Lattice':
            if len(self.myRegion) == 0:
                raise RuntimeError("The region must be set first")
            generator.set_region(self.myRegion)

    def create_body(self, body_dict, sims, scene):
        self.insert_template_to_creator()
        self.bodyCreator.create(sims, scene, body_dict)

    def regenerate(self, scene: myScene):
        if len(self.myGenerator) == 0: return 0

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
    