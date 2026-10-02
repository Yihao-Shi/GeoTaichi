from src.mpm.generator.BodyGenerator import BodyGenerator
from src.mpm.generator.LoadFromFile import BodyReader
from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction


class GenerateManager(object):
    def __init__(self):
        self.myRegion = dict()
        self.myClumpTemplate = dict()
        self.myGenerator = []

    def add_my_region(self, dims, domain, region_dict):
        name = DictIO.GetEssential(region_dict, "Name")
        if name in self.myRegion:
            region: RegionFunction = self.myRegion[name]
            region.mpm_finalize()
            del self.myRegion[name]
            self.add_region(dims, domain, name, region_dict, True)
        else:
            self.add_region(dims, domain, name, region_dict, False)

    def add_region(self, dims, domain, name, region_dict, override):
        DictIO.append(self.myRegion, name, RegionFunction(dims, "MPM"))
        region: RegionFunction = self.myRegion[name]
        region.set_region(region_dict, override)
        region.check_in_domain(domain)

    def get_region_ptr(self, name):
        if not self.myRegion is None:
            return self.myRegion[name]
        else:
            raise RuntimeError("Region class should be activated first!")

    def regenerate(self, scene):
        if len(self.myGenerator) == 0: return 0

        is_insert = 0
        for i in range(len(self.myGenerator)):
            is_insert = self.myGenerator[i].regenerate(scene)
            if not self.myGenerator[i].active:
                self.myGenerator[i].finalize()
                del self.myGenerator[i]
        return is_insert

    def add_body(self, body_dict, sims, scene: myScene):
        generator = BodyGenerator(sims)
        generator.set_system_strcuture(body_dict)
        if len(self.myRegion) == 0:
            raise RuntimeError("The region must be set first")
        generator.set_region(self.myRegion)
        generator.begin(scene)
        if generator.active:
            self.myGenerator.append(generator)
        else:
            generator.finalize()

    def read_body_file(self, body_dict, sims, scene: myScene):
        if scene.material is None:
            raise RuntimeError("The attribute must be added first")
        generator = BodyReader(sims)
        generator.set_region(self.myRegion)
        generator.set_system_strcuture(body_dict)
        generator.begin(scene)
        if generator.active:
            self.myGenerator.append(generator)
        else:
            generator.finalize()

    def add_polygons(self, body_dict, sims: Simulation, scene: myScene):
        if scene.contact is None:
            raise RuntimeError("The contact attribute must be added first")
        if sims.contact_detection == "DEMContact":
            generator = BodyReader(sims)
            generator.set_polygons(scene.contact, body_dict)
        else:
            raise RuntimeError(f"The polygon only supports for DEMContact. Current contact type is {sims.contact_detection}")
