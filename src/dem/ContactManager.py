import warnings

from src.dem.Simulation import Simulation
from src.dem.neighbor.BrustSearch import BrustSearch
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.contact.Linear import LinearModel
from src.dem.contact.HertzMindlin import HertzMindlinModel 
from src.dem.contact.LinearRolling import LinearRollingModel
from src.dem.contact.JiangRolling import JiangRollingResistanceModel
from src.dem.contact.LinearBond import LinearBondModel 


class ContactManager(object):
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(self):
        self.neighbor = None
        self.physpp = None
        self.physpw = None
        self.have_initialise = False

    def initialize(self, sims, scene):
        self.neighbor.neighbor_initialze(scene)
        self.collision_list(sims)
        self.have_initialise = True

    def choose_neighbor(self, sims: Simulation, scene):
        if self.neighbor is None:
            if sims.search == "Brust":
                self.neighbor = BrustSearch(sims, scene)
            elif sims.search == "LinkedCell":
                self.neighbor = LinkedCell(sims, scene)
            else:
                raise RuntimeError("Failed to activate neighbor class!")

    def particle_particle_initialize(self, sims: Simulation):
        if self.physpp is None:
            if sims.max_particle_num > 1:
                if sims.particle_particle_contact_model == "Linear Model":
                    self.physpp = LinearModel(sims.max_material_num)
                elif sims.particle_particle_contact_model == "Hertz Mindlin Model":
                    self.physpp = HertzMindlinModel(sims.max_material_num)
                elif sims.particle_particle_contact_model == "Linear Rolling Model":
                    self.physpp = LinearRollingModel(sims.max_material_num)
                elif sims.particle_particle_contact_model == "Jiang Rolling Model":
                    self.physpp = JiangRollingResistanceModel(sims.max_material_num)
                elif sims.particle_particle_contact_model == "Linear Bond Model":
                    self.physpp = LinearBondModel(sims.max_material_num)
                elif sims.particle_particle_contact_model == "User Defined":
                    pass
                elif not sims.particle_particle_contact_model is None:
                    raise ValueError('Particle to Particle Contact Model error!')
            else:
                self.physpp = ContactModelBase()
            self.physpp.manage_function("particle", sims.particle_work)
                

    def particle_wall_initialize(self, sims: Simulation):
        if self.physpw is None:
            if sims.max_wall_num > 0:
                if sims.particle_wall_contact_model == "Linear Model":
                    self.physpw = LinearModel(sims.max_material_num)
                elif sims.particle_wall_contact_model == "Hertz Mindlin Model":
                    self.physpw = HertzMindlinModel(sims.max_material_num)
                elif sims.particle_particle_contact_model == "Linear Rolling Model":
                    self.physpw = LinearRollingModel(sims.max_material_num)
                elif sims.particle_particle_contact_model == "Jiang Rolling Model":
                    self.physpw = JiangRollingResistanceModel(sims.max_material_num)
                elif sims.particle_wall_contact_model == "Linear Bond Model":
                    self.physpw = LinearBondModel(sims.max_material_num)
                elif sims.particle_wall_contact_model == "User Defined":
                    pass
                else:
                    raise ValueError('Particle to Wall Contact Model error!')
            else:
                self.physpw = ContactModelBase()

            self.physpw.manage_function("wall", sims.wall_work)

    def collision_list(self, sims: Simulation):
        if self.physpp:
            self.physpp.collision_initialize(sims.compaction_ratio, sims.particle_work, sims.max_potential_particle_pairs, sims.max_particle_num, sims.max_particle_num)
        else:
            raise RuntimeError("Particle-Particle contact model have not been activated successfully!")
        
        if self.physpw:    
            self.physpw.collision_initialize(sims.compaction_ratio, sims.wall_work, sims.max_potential_wall_pairs, sims.max_particle_num, sims.max_wall_num)
        else:
            raise RuntimeError("Particle-Wall contact model have not been activated successfully!")
    
    def add_contact_property(self, sims: Simulation, materialID1, materialID2, property, dType):
        if materialID1 > sims.max_material_num - 1 or materialID2 > sims.max_material_num - 1:
            raise RuntimeError("Material ID is out of the scope!")
        else:
            if dType == "particle-particle":
                if self.physpp.null_mode is True:
                    dType = None
                    warnings.warn("Particle-particle contact model is NULL, this procedure is automatically failed")
                    print('\n')
            elif dType == "particle-wall":
                if self.physpw.null_mode is True:
                    dType = None
                    warnings.warn("Particle-wall contact model is NULL, this procedure is automatically failed")
                    print('\n')
            elif dType == "all":
                if self.physpp.null_mode is True and self.physpw.null_mode is True:
                    dType = None
                    warnings.warn("Particle-particle contact model and particle-wall contact model are NULL, this procedure is automatically failed")
                    print('\n')
                elif self.physpp.null_mode is False and self.physpw.null_mode is True:
                    dType = "particle-particle"
                    warnings.warn("Particle-wall contact model is NULL, this procedure automatically transforms to add surface properties into particle-particle contact")
                    print('\n')
                elif self.physpp.null_mode is True and self.physpw.null_mode is False:
                    dType = "particle-wall"
                    warnings.warn("Particle-particle contact model is NULL, this procedure automatically transforms to add surface properties into particle-wall contact")
                    print('\n')
                    
            if dType == "particle-particle":
                componousID = self.physpp.add_surface_properties(sims.max_material_num, materialID1, materialID2, property)
                self.physpp.surfaceProps[componousID].print_surface_info(materialID1, materialID2)

            elif dType == "particle-wall":
                componousID = self.physpw.add_surface_properties(sims.max_material_num, materialID1, materialID2, property)
                self.physpw.surfaceProps[componousID].print_surface_info(materialID1, materialID2)

            elif dType == "all":
                componousID = self.physpp.add_surface_properties(sims.max_material_num, materialID1, materialID2, property)
                componousID = self.physpw.add_surface_properties(sims.max_material_num, materialID1, materialID2, property)
                self.physpp.surfaceProps[componousID].print_surface_info(materialID1, materialID2)

    def update_contact_property(self, sims: Simulation, materialID1, materialID2, property_name, value, overide):
        if materialID1 > sims.max_material_num - 1 or materialID2 > sims.max_material_num - 1:
            raise RuntimeError("Material ID is out of the scope!")
        else:
            if not self.physpp is None:
                self.physpp.update_properties(sims.max_material_num, materialID1, materialID2, property_name, value, overide)
            if not self.physpw is None:
                self.physpw.update_properties(sims.max_material_num, materialID1, materialID2, property_name, value, overide)
                
