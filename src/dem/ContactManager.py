import warnings

from src.dem.Simulation import Simulation
from src.dem.neighbor.BrustSearch import BrustSearch
from src.dem.neighbor.HierarchicalLinkedCell import HierarchicalLinkedCell
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.neighbor.BoundingVolumeHierarchy import BoundingVolumeHierarchy
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.contact.Linear import LinearModel
from src.dem.contact.HertzMindlin import HertzMindlinModel 
from src.dem.contact.LinearRolling import LinearRollingModel
from src.dem.contact.JiangRolling import JiangRollingResistanceModel
from src.dem.contact.LinearBond import LinearBondModel 
from src.dem.contact.EnergyConservation import EnergyConservation

from src.utils.ObjectIO import DictIO

class ContactManager(object):
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(self):
        self.neighbor = None
        self.physpp = None
        self.physpw = None
        self.have_initialise = False

    def initialize(self, sims: Simulation, scene, kwargs):
        self.neighbor.neighbor_initialze(scene, DictIO.GetAlternative(kwargs, "min_bounding_radius", max(sims.domain)), DictIO.GetAlternative(kwargs, "max_bounding_radius", 0.))
        self.collision_list(sims)
        self.have_initialise = True

    def choose_neighbor(self, sims: Simulation, scene):
        if self.neighbor is None:
            if sims.search == "Brust":
                self.neighbor = BrustSearch(sims, scene)
            elif sims.search == "LinkedCell":
                self.neighbor = LinkedCell(sims, scene)
            elif sims.search == "HierarchicalLinkedCell":
                self.neighbor = HierarchicalLinkedCell(sims, scene)
            elif sims.search == "BVH":
                self.neighbor = BoundingVolumeHierarchy(sims, scene)
            else:
                raise RuntimeError("Failed to activate neighbor class!")

    def particle_particle_initialize(self, sims: Simulation):
        if self.physpp is None:
            if sims.scheme == "DEM" or sims.scheme == "PolySuperEllipsoid" or sims.scheme == "PolySuperQuadrics":
                if sims.max_particle_num > 1 and not sims.particle_particle_contact_model is None:
                    if sims.particle_particle_contact_model == "Linear Model":
                        self.physpp = LinearModel(sims)
                    elif sims.particle_particle_contact_model == "Hertz Mindlin Model":
                        self.physpp = HertzMindlinModel(sims)
                    elif sims.particle_particle_contact_model == "Linear Rolling Model":
                        self.physpp = LinearRollingModel(sims)
                    elif sims.particle_particle_contact_model == "Jiang Rolling Model":
                        self.physpp = JiangRollingResistanceModel(sims)
                    elif sims.particle_particle_contact_model == "Linear Bond Model":
                        self.physpp = LinearBondModel(sims)
                    elif sims.particle_particle_contact_model == "User Defined":
                        pass
                    elif not sims.particle_particle_contact_model is None:
                        model_list = ["Linear Model", "Hertz Mindlin Model", "Linear Rolling Model", "Jiang Rolling Model"]
                        raise ValueError('Particle to Particle Contact Model error!')
                else:
                    self.physpp = ContactModelBase(sims)

            elif sims.scheme == "LSDEM":
                if sims.max_particle_num > 1 and not sims.particle_particle_contact_model is None:
                    if sims.particle_particle_contact_model == "Linear Model":
                        self.physpp = LinearModel(sims)
                    elif sims.particle_particle_contact_model == "Hertz Mindlin Model":
                        self.physpp = HertzMindlinModel(sims)
                    elif sims.particle_particle_contact_model == "Energy Conserving Model":
                        self.physpp = EnergyConservation(sims, types='Penalty')
                    elif sims.particle_particle_contact_model == "Barrier Model":
                        self.physpp = EnergyConservation(sims, types='Barrier')
                    else:
                        model_list = ["Linear Model", "Hertz Mindlin Model", "Energy Conserving Model", "Barrier Model"]
                        raise ValueError('Particle to Particle Contact Model error!')
                else:
                    self.physpp = ContactModelBase(sims)
            self.physpp.manage_function("particle", sims.particle_work)
                

    def particle_wall_initialize(self, sims: Simulation):
        if self.physpw is None:
            if sims.scheme == "DEM" or sims.scheme == "PolySuperEllipsoid" or sims.scheme == "PolySuperQuadrics":
                if sims.max_particle_num > 0 and sims.max_wall_num > 0 and not sims.particle_wall_contact_model is None:
                    if sims.particle_wall_contact_model == "Linear Model":
                        self.physpw = LinearModel(sims)
                    elif sims.particle_wall_contact_model == "Hertz Mindlin Model":
                        self.physpw = HertzMindlinModel(sims)
                    elif sims.particle_particle_contact_model == "Linear Rolling Model":
                        self.physpw = LinearRollingModel(sims)
                    elif sims.particle_particle_contact_model == "Jiang Rolling Model":
                        self.physpw = JiangRollingResistanceModel(sims)
                    elif sims.particle_wall_contact_model == "Linear Bond Model":
                        self.physpw = LinearBondModel(sims)
                    elif sims.particle_wall_contact_model == "User Defined":
                        pass
                    else:
                        model_list = ["Linear Model", "Hertz Mindlin Model", "Linear Rolling Model", "Jiang Rolling Model"]
                        raise ValueError(f'Particle to Wall Contact Model error! Input {sims.particle_wall_contact_model} is invalid, only the following is available {model_list}')
                else:
                    self.physpw = ContactModelBase(sims)

            elif sims.scheme == "LSDEM":
                if sims.max_particle_num > 0 and sims.max_wall_num > 0 and not sims.particle_wall_contact_model is None:
                    if sims.particle_wall_contact_model == "Linear Model":
                        self.physpw = LinearModel(sims)
                    elif sims.particle_wall_contact_model == "Hertz Mindlin Model":
                        self.physpw = HertzMindlinModel(sims)
                    elif sims.particle_wall_contact_model == "Energy Conserving Model":
                        self.physpw = EnergyConservation(sims, types='Penalty')
                    elif sims.particle_wall_contact_model == "Barrier Model":
                        self.physpw = EnergyConservation(sims, types='Barrier')
                    else:
                        model_list = ["Linear Model", "Hertz Mindlin Model", "Energy Conserving Model", "Barrier Model"]
                        raise ValueError(f'Particle to Wall Contact Model error! Input {sims.particle_wall_contact_model} is invalid, only the following is available {model_list}')
                else:
                    self.physpw = ContactModelBase(sims)
            self.physpw.manage_function("wall", sims.wall_work)

    def collision_list(self, sims: Simulation):
        if self.physpp:
            self.physpp.collision_initialize('particle', sims.particle_work, sims.particle_contact_list_length, sims.max_particle_num, sims.max_particle_num)
        else:
            raise RuntimeError("Particle-Particle contact model have not been activated successfully!")
        
        if self.physpw:
            self.physpw.collision_initialize('wall', sims.wall_work, sims.wall_contact_list_length, sims.max_particle_num, sims.max_wall_num)
        else:
            raise RuntimeError("Particle-Wall contact model have not been activated successfully!")
    
    def add_contact_property(self, sims: Simulation, materialID1, materialID2, property, dType):
        if materialID1 > sims.max_material_num - 1 or materialID2 > sims.max_material_num - 1:
            raise RuntimeError("Material ID is out of the scope!")
        else:
            if dType == "particle-particle":
                if self.physpp.null_model is True:
                    dType = None
                    warnings.warn("Particle-particle contact model is NULL, this procedure is automatically failed")
                    print('\n')
            elif dType == "particle-wall":
                if self.physpw.null_model is True:
                    dType = None
                    warnings.warn("Particle-wall contact model is NULL, this procedure is automatically failed")
                    print('\n')
            elif dType == "all":
                if self.physpp.null_model is True and self.physpw.null_model is True:
                    dType = None
                    warnings.warn("Particle-particle contact model and particle-wall contact model are NULL, this procedure is automatically failed")
                    print('\n')
                elif self.physpp.null_model is False and self.physpw.null_model is True:
                    dType = "particle-particle"
                    warnings.warn("Particle-wall contact model is NULL, this procedure automatically transforms to add surface properties into particle-particle contact")
                    print('\n')
                elif self.physpp.null_model is True and self.physpw.null_model is False:
                    dType = "particle-wall"
                    warnings.warn("Particle-particle contact model is NULL, this procedure automatically transforms to add surface properties into particle-wall contact")
                    print('\n')

            if dType == "particle-particle":
                componousID = self.physpp.add_surface_properties(materialID1, materialID2, property)
                self.physpp.surfaceProps[componousID].print_surface_info(materialID1, materialID2)

            elif dType == "particle-wall":
                componousID = self.physpw.add_surface_properties(materialID1, materialID2, property)
                self.physpw.surfaceProps[componousID].print_surface_info(materialID1, materialID2)

            elif dType == "all":
                componousID = self.physpp.add_surface_properties(materialID1, materialID2, property)
                componousID = self.physpw.add_surface_properties(materialID1, materialID2, property)
                self.physpp.surfaceProps[componousID].print_surface_info(materialID1, materialID2)

    def update_contact_property(self, sims: Simulation, materialID1, materialID2, property_name, value, overide):
        if materialID1 > sims.max_material_num - 1 or materialID2 > sims.max_material_num - 1:
            raise RuntimeError("Material ID is out of the scope!")
        else:
            if not self.physpp is None:
                self.physpp.update_properties(materialID1, materialID2, property_name, value, overide)
            if not self.physpw is None:
                self.physpw.update_properties(materialID1, materialID2, property_name, value, overide)

