from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.HertzMindlin import HertzMindlinModel
from src.mpdem.contact.Linear import LinearModel
from src.mpdem.contact.LinearBond import LinearBondModel
from src.mpdem.contact.ParticleFluid import ParticleFluid
from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpdem.Simulation import Simulation


class ContactManager:
    def __init__(self):
        self.neighbor = None
        self.physpp = None
        self.physpw = None
        self.have_initialise = False

    def initialize(self, csims, msims, dsims, mscene, dscene):
        self.neighbor.set_potential_contact_list(msims, dsims, mscene, dscene)
        self.collision_list(csims)
        self.have_initialise = True

    def choose_neighbor(self, csims, msims, dsims, mpm_neighbor, dem_neighbor):
        if self.neighbor is None:
            self.neighbor = MultiLinkedCell(csims, msims, dsims, mpm_neighbor, dem_neighbor)

    def particle_particle_initialize(self, sims: Simulation, material_type):
        if self.physpp is None:
            if material_type == "Fluid" and sims.particle_particle_contact_model != "Fluid Particle":
                raise RuntimeError("particle-particle contact model should be set as /Fluid Particle/")
            
            if sims.particle_particle_contact_model == "Linear Model":
                self.physpp = LinearModel(sims.max_material_num)
            elif sims.particle_particle_contact_model == "Hertz Mindlin Model":
                self.physpp = HertzMindlinModel(sims.max_material_num)
            elif sims.particle_particle_contact_model == "Linear Bond Model":
                self.physpp = LinearBondModel(sims.max_material_num)
            elif sims.particle_particle_contact_model == "Fluid Particle":
                self.physpp = ParticleFluid(sims.max_material_num)
            else:
                if sims.particle_particle_contact_model is None:
                    self.physpw = ContactModelBase()
                else:
                    raise ValueError('Particle to Particle Contact Model error!')
                
            no_operation = False
            if sims.particle_particle_contact_model is None or sims.particle_interaction is False:
                no_operation = True
            self.physpp.manage_function("particle", sims.contact_method, no_operation)
        
    def particle_wall_initialize(self, sims: Simulation, material_type):
        if self.physpw is None:
            if material_type == "Fluid" and sims.particle_wall_contact_model != "Fluid Particle":
                raise RuntimeError("particle-wall contact model should be set as /Fluid Particle/")
            
            if sims.particle_wall_contact_model == "Linear Model":
                self.physpw = LinearModel(sims.max_material_num)
            elif sims.particle_wall_contact_model == "Hertz Mindlin Model":
                self.physpw = HertzMindlinModel(sims.max_material_num)
            elif sims.particle_wall_contact_model == "Linear Bond Model":
                self.physpw = LinearBondModel(sims.max_material_num)
            elif sims.particle_wall_contact_model == "Fluid Particle":
                self.physpw = ParticleFluid(sims.max_material_num)
            else:
                if sims.particle_wall_contact_model is None:
                    self.physpw = ContactModelBase()
                else:
                    raise ValueError('Particle to Wall Contact Model error!')
                
            no_operation = False
            if sims.particle_wall_contact_model is None or sims.wall_interaction is False:
                no_operation = True 
            self.physpw.manage_function("wall", None, no_operation)

    def collision_list(self, sims: Simulation):
        if self.physpp:
            no_operation = False
            if sims.particle_particle_contact_model is None or sims.particle_interaction is False:
                no_operation = True
            self.physpp.collision_initialize(sims.compaction_ratio, sims.max_potential_particle_pairs, no_operation)
        else:
            raise RuntimeError("Particle(MPM)-Particle(DEM) contact model have not been activated successfully!")
        
        if self.physpw:   
            no_operation = False
            if sims.particle_wall_contact_model is None or sims.wall_interaction is False:
                no_operation = True 
            self.physpw.collision_initialize(sims.compaction_ratio, sims.max_potential_wall_pairs, no_operation)
        else:
            raise RuntimeError("Particle(MPM)-Wall contact model have not been activated successfully!")

    def add_contact_property(self, sims: Simulation, materialID1, materialID2, property, dType):
        if materialID1 > sims.max_material_num - 1 or materialID2 > sims.max_material_num - 1:
            raise RuntimeError("Material ID is out of the scope!")
        else:
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










    

