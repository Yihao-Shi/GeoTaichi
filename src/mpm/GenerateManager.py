from src.mpm.generator.BodyGenerator import BodyGenerator
from src.mpm.generator.LoadFromFile import BodyReader
from src.mpm.SceneManager import myScene
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction


class GenerateManager(object):
    def __init__(self):
        self.myRegion = dict()
        self.myClumpTemplate = dict()
        self.myGenerator = []

    def add_my_region(self, domain, region_dict):
        name = DictIO.GetEssential(region_dict, "Name")
        if name in self.myRegion:
            region: RegionFunction = self.myRegion[name]
            region.mpm_finalize()
            del self.myRegion[name]
            self.add_region(domain, name, region_dict, True)
        else:
            self.add_region(domain, name, region_dict, False)

    def add_region(self, domain, name, region_dict, override):
        DictIO.append(self.myRegion, name, RegionFunction("MPM"))
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
        """
        读取颗粒文件
        
        """
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