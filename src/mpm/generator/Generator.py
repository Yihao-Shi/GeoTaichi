import numpy as np
import types

from src.mpm.generator.InsertionKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec6f


class Generator(object):
    sims: Simulation
    def __init__(self, sims):
        self.sims = sims
        self.active = True
        self.myRegion = None
        self.myTemplate = None

        self.start_time = 0.
        self.end_time = 0.
        self.next_generate_time = 0.
        self.insert_interval = 1e10

        self.FIX = {
                    "Free": 0,
                    "Fix": 1
                   }

    def print_particle_info(self, bodyID, materialID, init_v, fix_v, particle_count, particle_volume=None, nParticlesPerCell=None):
        print(" Body(s) Information ".center(71, '-'))
        if particle_count == 0:
            raise RuntimeError("Zero Particles are inserted into region!")
        print("Body ID = ", bodyID)
        print("Material ID = ", materialID)
        print("Particle Number: ", particle_count)
        if not nParticlesPerCell is None:
            print("The Number of Particle per Cell: ", nParticlesPerCell)
        print("Initial Velocity = ", init_v)
        print("Fixed Velocity = ", fix_v)
        if not particle_volume is None:
            print("Particle Volume = ", particle_volume)
        print('\n')
        
    def set_particle_stress(self, scene: myScene, init_particle_num, particle_num, particle_stress):
        if type(particle_stress) is str:
            stress_file = DictIO.GetAlternative(particle_stress, "File", "ParticleStress.txt")
            stress_cloud = np.loadtxt(stress_file, unpack=True, comments='#').transpose()
            if stress_cloud.shape[0] != particle_num:
                raise ValueError("The length of File:: /ParticleStress/ is error!")
            if stress_cloud.shape[1] != 6:
                raise ValueError("The stress tensor should be transform to viogt format")
            kernel_apply_stress_from_file(init_particle_num, init_particle_num + particle_num, stress_cloud, scene.particle)
        elif type(particle_stress) is dict:
            initialStress = DictIO.GetAlternative(particle_stress, "InternalStress", vec6f([0, 0, 0, 0, 0, 0]))
            if isinstance(initialStress, (int, float)):
                initialStress = vec6f([float(initialStress), float(initialStress), float(initialStress), 0., 0., 0.])
            elif isinstance(initialStress, (tuple, list)):
                initialStress = vec6f(initialStress)
            if self.sims.material_type == "TwoPhaseSingleLayer":
                porePressure = DictIO.GetAlternative(particle_stress, "PorePressure", 0.)
                self.set_internal_stress(scene, particle_num, initialStress, porePressure)
            else:
                self.set_internal_stress(scene, particle_num, initialStress)

    def set_internal_stress(self, scene: myScene, particle_num, initialStress, porePressure=0):
        if initialStress.n != 6:
            raise ValueError(f"The dimension of initial stress: {initialStress.n} is inconsistent with the dimension of stress vigot tensor in 3D: 6")
        kernel_apply_vigot_stress_(int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, initialStress, scene.particle)
        if self.sims.material_type == "TwoPhaseSingleLayer":
            kernel_apply_pore_pressure_(int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, porePressure, scene.particle)

    def set_traction(self, particle_num, tractions, scene: myScene, region: RegionFunction=None):
        scene.boundary.get_essentials(scene.is_rigid, scene.psize, self.myRegion)
        if tractions:
            if type(tractions) is dict:
                scene.boundary.set_particle_traction(self.sims, tractions, particle_num, int(scene.particleNum[0]), scene.particle, scene.psize, region)
            elif type(tractions) is list:
                for traction in tractions:
                    scene.boundary.set_particle_traction(self.sims, traction, particle_num, int(scene.particleNum[0]), scene.particle, scene.psize, region)
