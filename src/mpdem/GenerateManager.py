#from src.mpdem.generator.LSBodyGenerate import ParticleCreator, ParticleGenerator
from src.mpdem.generator.MixtureGenerate import MixtureGenerator


class GenerateManager(object):
    def __init__(self, mpm_generator, dem_generator):
        self.myGenerator = []
        #self.LSbodyCreator = ParticleCreator(dem_generator)
        self.mixGenerator = MixtureGenerator(mpm_generator, dem_generator)

    def create_LSbody(self, body_dict):
        self.LSbodyCreator.create(body_dict)

    def regenerate(self, sims, mscene, dscene):
        self.mixGenerator.mpmGenerator.regenerate(mscene)
        self.mixGenerator.demGenerator.regenerate(dscene)
        
        if len(self.myGenerator) == 0: return 0
        is_insert = 0
        for i in range(len(self.myGenerator)):
            is_insert = self.myGenerator[i].regenerate()
            if not self.myGenerator[i].active:
                self.myGenerator[i].finalize()
                del self.myGenerator[i]
        return is_insert or self.mixGenerator.regenerate(sims, mscene, dscene)
    
    def add_mixture(self, check_overlap, dem_particle, mpm_body, sims, dscene, mscene, dsims, msims):
        self.mixGenerator.set_essentials(check_overlap)
        self.mixGenerator.add_mixture(sims, dscene, mscene, dem_particle, mpm_body, dsims, msims)

    '''def add_LSbody(self, body_dict, dem_generator):
        generator = ParticleGenerator(dem_generator)
        generator.set_system_strcuture(body_dict)
        generator.begin()
        if generator.active:
            self.myGenerator.append(generator)
        else:
            generator.finalize()'''

