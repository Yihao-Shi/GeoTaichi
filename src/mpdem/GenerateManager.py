from src.mpdem.generator.MixtureGenerate import MixtureGenerator


class GenerateManager(object):
    def __init__(self, mpm_generator, dem_generator):
        self.myGenerator = []
        self.mixGenerator = MixtureGenerator(mpm_generator, dem_generator)

    def regenerate(self, sims, mscene, dscene):
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

