import warnings

from src.mpm.Simulation import Simulation as MPMSimulation
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.EnergyConservation import EnergyConservationModel
from src.mpdem.contact.HertzMindlin import HertzMindlinModel
from src.mpdem.contact.Linear import LinearModel
from src.mpdem.contact.ParticleFluid import ParticleFluid
from src.mpdem.contact.NeighborBase import NeighborBase
from src.dem.Simulation import Simulation as DEMSimulation
from src.mpdem.Simulation import Simulation as Simulation


class ContactManager:
    def __init__(self):
        self.neighbor = None
        self.physpp = None
        self.physpw = None
        self.have_initialise = False

    def initialize(self, csims, msims, dsims, mscene, dscene):
        self.neighbor.set_potential_contact_list(mscene, dscene)
        self.collision_list(csims, msims, dsims)
        self.have_initialise = True

    def choose_neighbor(self, csims, msims, dsims: DEMSimulation, mpm_neighbor, dem_neighbor):
        if self.neighbor is None:
            self.neighbor = NeighborBase(csims, msims, dsims, mpm_neighbor, dem_neighbor)

    def particle_particle_initialize(self, sims: Simulation, msims: MPMSimulation, dsims: DEMSimulation):
        deactivate=False
        if self.physpp is None:
            if msims.max_particle_num == 0 or dsims.max_particle_num == 0: deactivate = True
            if sims.particle_interaction is True and msims.material_type == "Fluid" and sims.particle_particle_contact_model != "Fluid Particle":
                raise RuntimeError("particle-particle contact model should be set as /Fluid Particle/")
            
            if dsims.scheme == "DEM":
                if sims.particle_interaction is True and not sims.particle_particle_contact_model is None:
                    if sims.particle_particle_contact_model == "Linear Model":
                        self.physpp = LinearModel(sims.max_material_num)
                    elif sims.particle_particle_contact_model == "Hertz Mindlin Model":
                        self.physpp = HertzMindlinModel(sims.max_material_num)
                    elif sims.particle_particle_contact_model == "Fluid Particle":
                        self.physpp = ParticleFluid(sims.max_material_num)
                    else:
                        raise ValueError('Particle to Particle Contact Model error!')
                else:
                    self.physpp = ContactModelBase()

            elif dsims.scheme == "LSDEM":
                if sims.particle_interaction is True and not sims.particle_particle_contact_model is None:
                    if sims.particle_particle_contact_model == "Linear Model":
                        self.physpp = LinearModel(sims.max_material_num)
                    elif sims.particle_particle_contact_model == "Hertz Mindlin Model":
                        self.physpp = HertzMindlinModel(sims.max_material_num)
                    elif sims.particle_particle_contact_model == "Fluid Particle":
                        self.physpp = ParticleFluid(sims.max_material_num)
                    elif sims.particle_particle_contact_model == "Energy Conserving Model":
                        self.physpp = EnergyConservationModel(sims.max_material_num, types='Penalty')
                    elif sims.particle_particle_contact_model == "Barrier Function Model":
                        self.physpp = EnergyConservationModel(sims.max_material_num, types='Barrier')
                    else:
                        raise ValueError('Particle to Particle Contact Model error!')
                else:
                    self.physpp = ContactModelBase()
                
            if deactivate: self.physpp.null_model = True
            self.physpp.manage_function("particle", sims.enhanced_coupling, dsims.scheme)
        
    def particle_wall_initialize(self, sims: Simulation, msims: MPMSimulation, dsims: DEMSimulation):
        deactivate=False
        if self.physpw is None:
            if msims.max_particle_num == 0 or dsims.max_wall_num == 0: deactivate = True
            if sims.wall_interaction is True and not sims.particle_wall_contact_model is None:
                if msims.material_type == "Fluid" and sims.particle_wall_contact_model != "Fluid Particle":
                    raise RuntimeError("particle-wall contact model should be set as /Fluid Particle/")
                
                if sims.particle_wall_contact_model == "Linear Model":
                    self.physpw = LinearModel(sims.max_material_num)
                elif sims.particle_wall_contact_model == "Hertz Mindlin Model":
                    self.physpw = HertzMindlinModel(sims.max_material_num)
                elif sims.particle_wall_contact_model == "Fluid Particle":
                    self.physpw = ParticleFluid(sims.max_material_num)
                else:
                    raise ValueError('Particle to Wall Contact Model error!')
            else:
                self.physpw = ContactModelBase()
                
            if deactivate: self.physpw.null_model = True
            self.physpw.manage_function("wall", None, None)
            if sims.use_digital_elevation_heightfield(dsims) and not self.physpw.null_model:
                self.physpw.enable_digital_elevation_heightfield()

    def collision_list(self, sims: Simulation, msims: MPMSimulation, dsims: DEMSimulation):
        if self.physpp:
            self.physpp.collision_initialize(sims.enhanced_coupling, False, sims.particle_contact_list_length)
        else:
            raise RuntimeError("Particle(MPM)-Particle(DEM) contact model have not been activated successfully!")
        
        if self.physpw:   
            is_servo = True if dsims.servo_status == "On" else False
            monitor_cf = 'wall' in dsims.monitor_type and msims.max_particle_num > 0 and sims.wall_interaction
            if self.physpw.digital_elevation_heightfield:
                self.physpw.collision_initialize(sims.enhanced_coupling, is_servo or monitor_cf, msims.max_coupling_particle_num)
                self.print_heightfield_contact_memory_estimate(msims, dsims)
            else:
                self.physpw.collision_initialize(sims.enhanced_coupling, is_servo or monitor_cf, sims.wall_contact_list_length)
                self.print_wall_contact_memory_estimate(sims, msims, dsims, is_servo or monitor_cf)
        else:
            raise RuntimeError("Particle(MPM)-Wall contact model have not been activated successfully!")

    def print_heightfield_contact_memory_estimate(self, msims: MPMSimulation, dsims: DEMSimulation):
        if not dsims.use_digital_elevation_heightfield():
            return
        mib = 1024. * 1024.
        height_count = 0
        if not dsims.max_digital_elevation_grid_number is None:
            height_count = int(dsims.max_digital_elevation_grid_number[0] * dsims.max_digital_elevation_grid_number[1])
        height_mib = height_count * 4. / mib
        history_mib = msims.max_coupling_particle_num * 12. / mib
        total_mib = height_mib + history_mib
        print(" DEMPM Digital Elevation HeightField Contact Memory ".center(71, "-"))
        print(("Potential particle-wall list: Disabled").ljust(67))
        print(("Wall contact prefix list: Disabled").ljust(67))
        print(("Facet wall contact table: Disabled").ljust(67))
        print(("Height samples: " + str(height_count)).ljust(67))
        print(("Per-particle contact history: " + str(msims.max_coupling_particle_num)).ljust(67))
        print(("Heightfield estimate: " + f"{height_mib:.3f} MiB").ljust(67))
        print(("History estimate: " + f"{history_mib:.3f} MiB").ljust(67))
        print(("Total heightfield-contact estimate: " + f"{total_mib:.3f} MiB").ljust(67))
        print('\n')

    def print_wall_contact_memory_estimate(self, sims: Simulation, msims: MPMSimulation, dsims: DEMSimulation, stores_force):
        if not sims.wall_interaction or dsims.max_wall_num <= 0:
            return
        mib = 1024. * 1024.
        int_bytes = 4.
        vec3_bytes = 12.
        potential_list_mib = sims.max_potential_wall_pairs * int_bytes / mib
        prefix_mib = 2. * (msims.max_coupling_particle_num + 1) * int_bytes / mib
        if stores_force:
            contact_table_bytes = 2. * int_bytes + 3. * vec3_bytes
        else:
            contact_table_bytes = 2. * int_bytes + vec3_bytes
        history_table_bytes = int_bytes + vec3_bytes
        contact_table_mib = sims.wall_contact_list_length * (contact_table_bytes + history_table_bytes) / mib
        total_mib = potential_list_mib + prefix_mib + contact_table_mib
        print(" DEMPM Wall Contact Memory Estimate ".center(71, "-"))
        print(("Potential particle-wall pairs: " + str(sims.max_potential_wall_pairs)).ljust(67))
        print(("Contact table capacity: " + str(sims.wall_contact_list_length)).ljust(67))
        print(("Potential list estimate: " + f"{potential_list_mib:.3f} MiB").ljust(67))
        print(("Prefix/history index estimate: " + f"{prefix_mib:.3f} MiB").ljust(67))
        print(("Contact table estimate: " + f"{contact_table_mib:.3f} MiB").ljust(67))
        print(("Total wall-contact estimate: " + f"{total_mib:.3f} MiB").ljust(67))
        print('\n')

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










    
