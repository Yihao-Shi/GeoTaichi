import warnings

import numpy as np
import taichi as ti

from src.mpm.BaseKernel import *
from src.mpm.Contact import *
from src.mpm.elements.HexahedronElement8Nodes import HexahedronElement8Nodes
from src.mpm.elements.QuadrilateralElement4Nodes import QuadrilateralElement4Nodes
from src.mpm.MaterialManager import ConstitutiveModel
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.boundaries.BoundaryConstraint import BoundaryConstraints
from src.mpm.Simulation import Simulation
from src.mpm.structs import *
from src.utils.linalg import no_operation
from src.utils.DomainBoundary import DomainBoundary
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec2f, vec3f, vec3u8


class myScene(object):
    material: ConstitutiveModelBase

    def __init__(self) -> None:
        self.domain_boundary = None
        self.contact_parameter = None
        self.contact_type = None
        self.mass_cut_off = Threshold
        self.volume_cut_off = Threshold

        self.element_type = "R8N3D"
        self.boundary = None
        self.particle = None
        self.iparticle = None
        self.material = None
        self.element = None
        self.extra_node = None
        self.node = None
        self.grid = []
        self.is_rigid = None
        self.pid = None
        self.grandparent = None
        self.parent = None
        self.child = None
        self.psize = np.array([], dtype=np.int32)

        self.particleNum = np.zeros(1, dtype=np.int32)
        self.RECTELETYPE = ["Q4N2D", "R8N3D", "Staggered"]
        self.TRIELETYPE = ["T3N2D", "T4N3D"]

    def activate_boundary(self, sims):
        self.boundary = BoundaryConstraints()
        self.boundary.activate_boundary_constraints(sims)

    def is_rectangle_cell(self):
        return self.element_type in self.RECTELETYPE
    
    def is_triangle_cell(self):
        return self.element_type in self.TRIELETYPE
    
    def find_particle_class(self, sims: Simulation):
        ptemp = None
        if sims.coupling or sims.neighbor_detection:
            if sims.solver_type == "Explicit":
                ptemp = ParticleCoupling
            #elif sims.solver_type == "Implicit":
            #    ptemp = ImplicitParticleCoupling
        else:
            if sims.solver_type == "Explicit":
                if sims.dimension == 2:
                    if not sims.is_2DAxisy:
                        if sims.material_type == "TwoPhaseSingleLayer":
                            ptemp = ParticleCloudTwoPhase2D
                        else:
                            ptemp = ParticleCloud2D
                    elif sims.is_2DAxisy:
                        ptemp = ParticleCloud2DAxisy
                elif sims.dimension == 3:
                    ptemp = ParticleCloud
            elif sims.solver_type == "Implicit":
                raise RuntimeError("Invalid")
                #if self.element_type == "Staggered":
                #    if sims.dimension == 2:
                #        ptemp = ParticleCloudIncompressible2D # StaggeredPartilce2D
                #    if sims.dimension == 3:
                #        ptemp = ParticleCloudIncompressible3D # StaggeredPartilce
                #else:
                #    if sims.dimension == 2:
                #        ptemp = ImplicitParticle2D
                #    elif sims.dimension == 3:
                #        ptemp = ImplicitParticle
            
            if sims.contact_detection == "DEMContact":
                ptemp.members.update({"contact_traction": ti.types.vector(sims.dimension, float)})
        if ptemp is None: raise RuntimeError("Wrong particle type!")
        return ptemp

    def activate_particle(self, sims: Simulation):
        if self.particle is None and sims.max_particle_num > 0:
            ptemp = self.find_particle_class(sims)
            if sims.solver_type == "Explicit":
                if sims.stabilize == 'F-Bar Method':
                    ptemp.members.update({"jacobian": float})
            if sims.particle_shifting is True:
                ptemp.members.update({"grad_E2": ti.types.vector(sims.dimension, float)})
            self.particle = ptemp.field()
            ti.root.dense(ti.i, sims.max_particle_num).place(self.particle)
            if sims.solver_type == "Explicit":
                if sims.stabilize == 'F-Bar Method':
                    kernel_initialize_particle_fbar(self.particle)

    def activate_material(self, sims: Simulation, handler: ConstitutiveModel):
        if self.material is None:
            self.material = handler.initialize(sims)
            self.material.model_initialization(handler.material)
        else:
            warnings.warn("Previous material will be override!")

    def check_materials(self, sims):
        if self.material is None:
            self.activate_material(sims, "RigidBody", materials={})

    def activate_element(self, sims: Simulation, handler, element):
        self.element_type = DictIO.GetAlternative(element, "ElementType", "R8N3D")
        if sims.max_particle_num > 0:
            grid_level = self.find_grid_level(sims, DictIO.GetAlternative(element, "Contact", None))
            self.activate_material(sims, handler)

            if sims.dimension == 2:
                if self.element_type == "T3N2D":
                    raise ValueError("The triangle mesh is not supported currently")
                elif self.element_type == "Q4N2D" or self.element_type == "Staggered":
                    if not self.element is None:
                        print("Warning: Previous elements will be override!")
                    self.element = QuadrilateralElement4Nodes(self.element_type)
                else:
                    raise ValueError("Keyword:: /ElementType/ error!")
                self.element.create_nodes(sims, vec2f(DictIO.GetEssential(element, "ElementSize")))
                self.initialize_element(sims, grid_level)
            elif sims.dimension == 3:
                if self.element_type == "T4N3D":
                    raise ValueError("The triangle mesh is not supported currently")
                elif self.element_type == "R8N3D":
                    if not self.element is None:
                        print("Warning: Previous elements will be override!")
                    self.element = HexahedronElement8Nodes(self.element_type)
                else:
                    raise ValueError("Keyword:: /ElementType/ error!")
                self.element.create_nodes(sims, vec3f(DictIO.GetEssential(element, "ElementSize")))
                self.initialize_element(sims, grid_level)
            self.boundary.set_layer_number(grid_level)
            self.activate_grid(sims, DictIO.GetAlternative(element, "Contact", None), grid_level)

            if sims.ptraction_method == "Virtual":
                self.element.set_up_cell_active_flag()

    def initialize_element(self, sims: Simulation, grid_level):
        if sims.gauss_number > 0:
            self.element.element_initialize(sims, local_coordiates=False)
            self.element.activate_gauss_cell(sims, grid_level)
        else:
            self.element.element_initialize(sims)
        self.element.activate_euler_cell()

    def find_grid_level(self, sims: Simulation, contact):
        if contact is None or DictIO.GetAlternative(contact, "ContactDetection", None) is False:
            sims.set_contact_detection(None)
            grid_level = 1
        else:
            sims.set_contact_detection(DictIO.GetAlternative(contact, "ContactDetection", None))
            grid_level = DictIO.GetAlternative(contact, "GridLevel", None)
            if grid_level is None:
                if sims.contact_detection is None:
                    grid_level = 1
                else:
                    if "DEM" in sims.contact_detection:
                        grid_level = 1
                    else:
                        grid_level = 2
        return grid_level
    
    def find_grid_class(self, sims: Simulation):
        gtemp = None
        if sims.contact_detection is not None:
            if sims.solver_type == "Explicit":
                if sims.dimension == 2:
                    gtemp = ContactNodes2D
                elif sims.dimension == 3:
                    gtemp = ContactNodes
                if sims.contact_detection == "GeoContact":
                    gtemp.members.update({"contact_pos": ti.types.vector(sims.dimension, float)})
            else:
                raise RuntimeError("The contact is not supported in implicit MPM currently")
        else:
            if sims.solver_type == "Explicit":
                if sims.dimension == 2:
                    if sims.material_type == "Solid" or sims.material_type == "Fluid":
                        gtemp = Nodes2D
                    elif sims.material_type == "TwoPhaseSingleLayer":
                        gtemp = NodeTwoPhase2D
                    #elif sims.material_type == "TwoPhaseDoubleLayer":
                    #    raise RuntimeError()
                elif sims.dimension == 3:
                    if sims.material_type == "Solid" or sims.material_type == "Fluid":
                        gtemp = Nodes
                    else:
                        raise RuntimeError()
            elif sims.solver_type == "Implicit":
                if sims.material_type == "Solid":
                    if sims.dimension == 2:
                        gtemp = ImplicitNodes2D
                    elif sims.dimension == 3:
                        gtemp = ImplicitNodes
                #if sims.material_type == "Fluid":
                #    if sims.dimension == 2:
                #        gtemp = IncompressibleNodes2D
                #    elif sims.dimension == 3:
                #        gtemp = IncompressibleNodes3D
        if gtemp is None: raise RuntimeError("Wrong background node type!")
        return gtemp

    def activate_grid(self, sims: Simulation, contact, grid_level):
        self.check_grid_inputs(sims, grid_level)
        self.is_rigid = ti.field(int, shape=grid_level)
        sims.set_body_num(grid_level)
        self.element.set_characteristic_length(sims)

        cut_off = 0.
        if not self.node is None:
            print("Warning: Previous node will be override!")
        if sims.contact_detection:
            if sims.contact_detection == "MPMContact":
                self.contact_parameter = MPMContact(contact)
            elif sims.contact_detection == "GeoContact":
                self.contact_parameter = GeoContact(contact)   
            elif sims.contact_detection == "DEMContact":
                self.contact_parameter = DEMContact(contact, self.material)
             
        self.node = self.find_grid_class(sims).field()
        if sims.sparse_grid:
            self.grandparent = ti.root.pointer(ti.ij, (int(np.ceil(self.element.gridSum / sims.block_size[0])), grid_level))
            self.parent = self.grandparent.pointer(ti.i, int(sims.block_size[0] // sims.block_size[1]))
            self.child = self.parent.dense(ti.i, int(sims.block_size[1]))
        else: 
            self.child = ti.root.dense(ti.ij, (self.element.gridSum, grid_level))

        if sims.stabilize == 'F-Bar Method' or sims.pressure_smoothing == True or sims.particle_shifting is True:
            gtemp = ExtraNode
            if sims.stabilize == 'F-Bar Method':
                if sims.material_type == "Solid":
                    gtemp.members.update({"jacobian": float})
                elif sims.material_type == "Fluid":
                    gtemp.members.update({"jacobian": float, "pressure": float})
            if sims.pressure_smoothing == True:
                if 'pressure' not in gtemp.members.keys():
                    gtemp.members.update({"pressure": float})
            self.extra_node = gtemp.field()
            self.child.place(self.node, self.extra_node)
        else:
            self.child.place(self.node)

        #if sims.sparse_grid:
        #    self.pid = ti.field(int)
        #    self.parent.dynamic(ti.i, 1024 * 1024, chunk_size=sims.block_size[1] ** sims.dimension * 8).place(self.pid)
        
        if sims.mapping == "G2P2G":
            self.grandparent = [self.grandparent]
            self.parent = [self.parent]
            self.child = [self.child]
            output_grid = self.find_grid_class(sims, grid_level)
            if sims.sparse_grid:
                grandparent = ti.root.pointer(ti.ij, (int(np.ceil(self.element.gridSum // sims.block_size[0])), grid_level))
                parent = self.grandparent[1].pointer(ti.i, int(sims.block_size[0] // sims.block_size[1]))
                child = self.parent[1].dense(ti.i, int(sims.block_size[1]))
                self.grandparent.append(grandparent)
                self.parent.append(parent)
                self.child.append(child)
            else: 
                self.child = ti.root.dense(ti.ij, (self.element.gridSum, grid_level))
            self.grid = [self.node, self.child[1].place(output_grid)]
        self.element.calculate_basis_function(sims, grid_level)
        self.print_grid_message(sims, grid_level, cut_off)

    def deactivate_sparse_grid(self):
        self.grandparent.deactivate_all()
 
    def check_grid_inputs(self, sims: Simulation, grid_level):
        if grid_level > 2:
            raise ValueError("The mpm only support two body contact detection")
        if grid_level == 2:
            if sims.contact_detection is False:
                warnings.warn("The contact detection has not been activeted yet!")
        if grid_level == 1 and sims.contact_detection is True:
            raise RuntimeError("The contact detection should be turned off!")
    
    def print_grid_message(self, sims: Simulation, grid_level, cut_off=0.):
        print(" Grid Information ".center(71,"-"))
        print("Grid Type: Rectangle")
        if grid_level > 0:
            print("The number of grids = ", grid_level)
        self.element.print_message()
        if sims.contact_detection:
            print("Contact Detection Activated: ", sims.contact_detection)
            print("Cut-off Distance Multiplier =", cut_off)
            print("Friction Coefficient =", self.contact_parameter.friction)
            if sims.contact_detection == "GeoContact":
                print("Penalty Parameter = ", self.contact_parameter.alpha, self.contact_parameter.beta)
            if sims.contact_detection == "DEMContact":
                print("Stiffness Parameter = ", self.contact_parameter.kn, self.contact_parameter.kt)
        print('\n')

    def get_material_ptr(self):
        return self.material
    
    def get_element_ptr(self):
        return self.element
    
    def get_node_ptr(self):
        return self.node
    
    def get_particle_ptr(self):
        return self.particle
        
    def check_particle_num(self, sims: Simulation, particle_number):
        if self.particleNum[0] + particle_number > sims.max_particle_num:
            raise ValueError ("The MPM particles should be set as: ", self.particleNum[0] + particle_number)

    def find_min_z_position(self):
        return find_min_z_position_(int(self.particleNum[0]), self.particle)
    
    def find_min_y_position(self):
        return find_min_y_position_(int(self.particleNum[0]), self.particle)
    
    def find_bounding_sphere_radius(self):
        rad_max = find_particle_max_radius_(int(self.particleNum[0]), self.particle)
        rad_min = find_particle_min_radius_(int(self.particleNum[0]), self.particle)
        return rad_max, rad_min
    
    def find_particle_min_radius(self):
        return find_particle_min_radius_(int(self.particleNum[0]), self.particle)
    
    def find_particle_max_radius(self):
        return find_particle_max_radius_(int(self.particleNum[0]), self.particle)
    
    def find_particle_min_mass(self):
        return find_particle_min_mass_(int(self.particleNum[0]), self.particle)
    
    def reset_verlet_disp(self):
        reset_verlet_disp_(int(self.particleNum[0]), self.particle)

    def get_critical_timestep(self):
        max_vel = find_max_velocity_(int(self.particleNum[0]), self.particle)
        max_vel += self.material.find_max_sound_speed()
        return self.element.calc_critical_timestep(max_vel)
    
    def find_min_density(self):
        mindensity = 1e15
        for nm in range(self.material.matProps.shape[0]):
            if self.material.matProps[nm].density > 0:
                mindensity = ti.min(mindensity, self.material.matProps[nm].density)
        return mindensity
    
    def calc_mass_cutoff(self, sims: Simulation):
        density = 0.
        matcount = 0
        for nm in range(self.material.matProps.shape[0]):
            if self.material.matProps[nm].density > 0:
                density += self.material.matProps[nm].density
                matcount += 1
        grid_size = self.element.grid_size
        if sims.dimension == 3:
            self.mass_cut_off = 1e-8 * density / matcount * grid_size[0] * grid_size[1] * grid_size[2]
            self.volume_cut_off = 1e-8 * grid_size[0] * grid_size[1] * grid_size[2]
        elif sims.dimension == 2:
            self.mass_cut_off = 1e-8 * density / matcount * grid_size[0] * grid_size[1]
            self.volume_cut_off = 1e-5 * grid_size[0] * grid_size[1]
    
    def update_particle_properties_in_region(self, sims: Simulation, override, property_name, value, is_in_region):
        print(" Modify Particle Information ".center(71, '-'))
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        factor = 1 if not override else 0
        if property_name == "bodyID":
            modify_particle_bodyID_in_region(value, int(self.particleNum[0]), self.particle, is_in_region)
        elif property_name == "materialID":
            modify_particle_materialID_in_region(value, int(self.particleNum[0]), self.particle, self.material.matProps, is_in_region)
        elif property_name == "position":
            if sims.dimension == 3:
                modify_particle_position_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif sims.dimension == 2:
                modify_particle_position_in_region_2D(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
        elif property_name == "velocity":
            if sims.dimension == 3:
                modify_particle_velocity_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif sims.dimension == 2:
                modify_particle_velocity_in_region_2D(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
        elif property_name == "stress":
            modify_particle_stress_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
        elif property_name == "fix_velocity":
            FIX = {
                    "Free": 0,
                    "Fix": 1
                   }
            fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
            modify_particle_fix_v_in_region(fix_v, int(self.particleNum[0]), self.particle, is_in_region)
        else:
            valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
            raise KeyError(f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}")
        
    def update_particle_properties(self, sims: Simulation, override, property_name, value, bodyID):
        print(" Modify Body Information ".center(71, '-'))
        print("Target BodyID =", bodyID)
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        factor = 1 if not override else 0
        if property_name == "bodyID":
            modify_particle_bodyID(value, int(self.particleNum[0]), self.particle, bodyID)
        elif property_name == "materialID":
            modify_particle_materialID(value, int(self.particleNum[0]), self.particle, self.material.matProps, bodyID)
        elif property_name == "position":
            if sims.dimension == 3:
                modify_particle_position(factor, value, int(self.particleNum[0]), self.particle, bodyID)
            elif sims.dimension == 2:
                modify_particle_position_2D(factor, value, int(self.particleNum[0]), self.particle, bodyID)
        elif property_name == "velocity":
            if sims.dimension == 3:
                modify_particle_velocity(factor, value, int(self.particleNum[0]), self.particle, bodyID)
            elif sims.dimension == 2:
                modify_particle_velocity_2D(factor, value, int(self.particleNum[0]), self.particle, bodyID)
        elif property_name == "stress":
            modify_particle_stress(factor, value, int(self.particleNum[0]), self.particle, bodyID)
        elif property_name == "fix_velocity":
            FIX = {
                    "Free": 0,
                    "Fix": 1
                   }
            fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
            modify_particle_fix_v(fix_v, int(self.particleNum[0]), self.particle, bodyID)
        else:
            valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
            raise KeyError(f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}")

    def check_overlap_coupling(self):
        initial_particle = self.particleNum[0]
        self.particleNum[0] = update_particle_storage_(int(int(self.particleNum[0])), self.particle, self.material.stateVars)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')
        
    def delete_particles(self, bodyID):
        initial_particle = self.particleNum[0]
        kernel_delete_particles(self.particleNum[0], self.particle, bodyID)
        self.particleNum[0] = update_particle_storage_(int(self.particleNum[0]), self.particle, self.material.stateVars)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')

    def delete_particles_in_region(self, is_in_region):
        initial_particle = self.particleNum[0]
        kernel_delete_particles_in_region(self.particleNum[0], self.particle, is_in_region)
        self.particleNum[0] = update_particle_storage_(int(self.particleNum[0]), self.particle, self.material.stateVars)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')

    def check_in_domain(self, sims: Simulation):
        check_in_domain(sims.domain, int(self.particleNum[0]), self.particle)

    def get_mass_center(self, bodyID=0):
        return kernel_compute_mass_center(int(self.particleNum[0]), self.particle, bodyID)
    
    def choose_coupling_region(self, sims: Simulation, function):
        if sims.coupling:
            kernel_update_coupling_material_points(int(self.particleNum[0]), self.particle, function)

    def update_coupling_points_number(self, sims: Simulation):
        need_coupling = kernel_compute_coupling_material_points_number(int(self.particleNum[0]), self.particle)
        sims.set_coupling_particles(need_coupling)

    def filter_particles(self, sims: Simulation):
        if sims.coupling:
            kernel_tranverse_coupling_particle(int(self.particleNum[0]), self.particle)
        else:
            kernel_tranverse_active_particle(int(self.particleNum[0]), self.particle)

    def check_elastic_material(self):
        return self.material.is_elastic
    
    def push_psize(self, particleNum, psize):
        psize = list(psize)
        dim = len(psize)
        if np.array(psize).ndim == 2:
            dim = len(psize[0])
        self.psize = np.append(self.psize, np.repeat([psize], particleNum, axis=0)).reshape(-1, dim)

    def set_boundary_condition(self, sims: Simulation):
        self.domain_boundary = DomainBoundary(sims.domain)
        self.domain_boundary.set_boundary_condition(sims.boundary)
        if self.domain_boundary.need_run:
            self.apply_boundary_conditions = self.apply_boundary_condition
        else:
            self.apply_boundary_conditions = no_operation

    def apply_boundary_condition(self):
        if self.domain_boundary.apply_boundary_conditions(int(self.particleNum[0]), self.particle):
            update_particle_storage_(self.particleNum, self.particle, self.material.stateVars)