import taichi as ti

from src.dem.neighbor.LinkedCell import LinkedCell 
from src.dem.SceneManager import myScene as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.mpm.SceneManager import myScene as MPMScene
from src.mpm.Simulation import Simulation as MPMSimulation
from src.mpdem.Simulation import Simulation as Simulation
from src.mpdem.fluid_dynamics.SemiResolved import *
from src.mpdem.fluid_dynamics.DragForceModel import DragForce

class CFDMPM:
    sims: Simulation
    msims: MPMSimulation
    dsims: DEMSimulation
    mscene: MPMScene
    dscene: DEMScene
    dneighbor: LinkedCell

    def __init__(self, sims, msims, dsims, mscene, dscene, dneighbor):
        self.sims = sims
        self.msims = msims
        self.dsims = dsims
        self.mscene = mscene
        self.dscene = dscene
        self.dneighbor = dneighbor

    def build_essential_field(self, drag_model):
        self.cell_fraction = ti.field(float, shape=self.mscene.element.cellSum)
        self.cell_drag_force = ti.Vector.field(self.msims.dimension, float, shape=self.mscene.element.cellSum)
        self.fluid_velocity = ti.Vector.field(self.dsims.dimension, float, shape=self.dsims.max_particle_num)
        self.stress_divergence = ti.Vector.field(self.dsims.dimension, float, shape=self.dsims.max_particle_num)
        self.fluid_fraction = ti.field(float, shape=self.dsims.max_particle_num)
        self.fluid_volume = ti.field(float, shape=self.dsims.max_particle_num)
        self.drag_model = DragForce(drag_model)

    def pre_compute(self):
        compute_spherical_particle_void_fraction(int(self.dscene.sphereNum[0]), self.mscene.element.cnum, self.mscene.element.grid_size, self.mscene.element.igrid_size,
                                               self.dscene.particle, self.dscene.sphere, self.cell_fraction, self.fluid_volume)
        finalize_coupling(self.mscene.element.cellSum, self.mscene.element.cnum, self.mscene.element.gnum, self.mscene.element.grid_size,self.mscene.node, self.cell_fraction, self.cell_drag_force)

    def coupling(self):
        compute_ambient_fluid_variable(self.dscene.sphereNum[0], self.sims.dependent_domain, self.mscene.element.cnum, self.mscene.element.gnum, self.mscene.element.grid_size, self.mscene.element.igrid_size,self.dsims.gravity, 
                                      self.mscene.node, self.dscene.particle, self.dscene.sphere, self.cell_fraction, self.fluid_volume, self.fluid_velocity, self.fluid_fraction, self.stress_divergence)
        
        self.cell_fraction.fill(0)
        self.cell_drag_force.fill(0)
        compute_spherical_particle_fluid_force(int(self.dscene.sphereNum[0]), self.sims.infludence_domain, self.mscene.element.cnum, self.mscene.element.grid_size, self.mscene.element.igrid_size,
                                               self.dsims.gravity, self.mscene.material.matProps[1], self.dscene.particle, self.dscene.sphere, self.cell_drag_force, self.fluid_volume, self.fluid_velocity, self.fluid_fraction, self.stress_divergence, self.drag_model)
        compute_spherical_particle_void_fraction(int(self.dscene.sphereNum[0]), self.mscene.element.cnum, self.mscene.element.grid_size, self.mscene.element.igrid_size,
                                               self.dscene.particle, self.dscene.sphere, self.cell_fraction, self.fluid_volume)
        finalize_coupling(self.mscene.element.cellSum, self.mscene.element.cnum, self.mscene.element.gnum, self.mscene.element.grid_size,self.mscene.node, self.cell_fraction, self.cell_drag_force)

