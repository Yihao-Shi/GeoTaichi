import os, warnings

import numpy as np
import taichi as ti

from src.mpm.BaseKernel import *
from src.mpm.elements.HexahedronElement8Nodes import HexahedronElement8Nodes
from src.mpm.elements.QuadrilateralElement4Nodes import QuadrilateralElement4Nodes
from src.mpm.MaterialManager import ConstitutiveModel
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.boundaries.BoundaryCore import *
from src.mpm.boundaries.BoundaryStrcut import *
from src.mpm.BaseStruct import *
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec2i, vec2f, vec3f, vec3i, vec3u8


class myScene(object):
    material: ConstitutiveModelBase
    node: Nodes
    multi_node: Nodes
    particle: ParticleCloud
    extra_particle: ParticleFBar
    velocity_boundary: VelocityConstraint
    friction_boundary: FrictionConstraint
    absorbing_boundary: AbsorbingConstraint
    traction_boundary: TractionConstraint

    def __init__(self) -> None:
        self.mu = 0.
        self.mass_cut_off = Threshold
        self.volume_cut_off = Threshold

        self.particle = None
        self.extra_particle = None
        self.element_type = "R8N3D"
        self.material = None
        self.element = None
        self.extra_node = None
        self.node = None
        self.accmulated_reaction_forces = None
        self.velocity_boundary = None
        self.reflection_boundary = None
        self.friction_boundary = None
        self.absorbing_boundary = None
        self.traction_boundary = None
        self.displacement_boundary = None
        self.is_rigid = None

        self.velocity_list = np.zeros(1, dtype=np.int32)
        self.reflection_list = np.zeros(1, dtype=np.int32)
        self.friction_list = np.zeros(1, dtype=np.int32)
        self.absorbing_list = np.zeros(1, dtype=np.int32)
        self.traction_list = np.zeros(1, dtype=np.int32)
        self.displacement_list = np.zeros(1, dtype=np.int32)
        self.particleNum = np.zeros(1, dtype=np.int32)
        
        self.RECTELETYPE = ["Q4N2D", "R8N3D", "R27N3D"]
        self.TRIELETYPE = ["T4N3D"]

    def activate_basic_class(self, sims):
        self.activate_particle(sims)
        self.activate_boundary_constraints(sims)

    def is_rectangle_cell(self):
        return self.element_type in self.RECTELETYPE
    
    def is_triangle_cell(self):
        return self.element_type in self.TRIELETYPE

    def activate_particle(self, sims: Simulation):
        if self.particle is None and sims.max_particle_num > 0:
            if sims.coupling or sims.neighbor_detection:
                if sims.solver_type == "Explicit":
                    self.particle = ParticleCoupling.field()
                elif sims.solver_type == "Implicit":
                    self.particle = ImplicitParticleCoupling.field()
            else:
                if sims.solver_type == "Explicit":
                    if sims.dimension == 2:
                        self.particle = ParticleCloud2D.field()
                    elif sims.dimension == 3:
                        self.particle = ParticleCloud.field()
                elif sims.solver_type == "Implicit":
                    self.particle = ImplicitParticle.field()

            if sims.solver_type == "Explicit":
                if self.extra_particle is None and sims.stabilize == 'F-Bar Method':
                    self.extra_particle = ParticleFBar.field()
                    ti.root.dense(ti.i, sims.max_particle_num).place(self.particle, self.extra_particle)
                    kernel_initialize_particle_fbar(self.extra_particle)
                else:
                    ti.root.dense(ti.i, sims.max_particle_num).place(self.particle)
            elif sims.solver_type == "Implicit":
                ti.root.dense(ti.i, sims.max_particle_num).place(self.particle)

    def activate_material(self, sims: Simulation, model, materials):
        if self.material is None:
            self.material = ConstitutiveModel.initialize(sims.material_type, sims.stabilize, model, sims.max_material_num, sims.max_particle_num, sims.configuration, sims.solver_type)
            self.material.model_initialization(materials)
        else:
            warnings.warn("Previous material will be override!")

    def check_materials(self, sims):
        if self.material is None:
            self.activate_material(sims, "RigidBody", materials={})

    def activate_element(self, sims: Simulation, element):
        self.element_type = DictIO.GetAlternative(element, "ElementType", "R8N3D")
        if sims.max_particle_num > 0:
            grid_level = self.find_grid_level(sims, DictIO.GetAlternative(element, "Contact", None))
            if sims.dimension == 2:
                if self.element_type == "Q4N2D":
                    if not self.element is None:
                        print("Warning: Previous elements will be override!")
                    self.element = QuadrilateralElement4Nodes()
                    self.element.create_nodes(sims, vec2f(DictIO.GetEssential(element, "ElementSize")))
                    self.initialize_element(sims, grid_level)
                else:
                    raise ValueError("Keyword:: /ElementType/ error!")
            elif sims.dimension == 3:
                if self.element_type == "T4N3D":
                    raise ValueError("The triangle mesh is not supported currently")
                elif self.element_type == "R8N3D":
                    if not self.element is None:
                        print("Warning: Previous elements will be override!")
                    self.element = HexahedronElement8Nodes()
                    self.element.create_nodes(sims, vec3f(DictIO.GetEssential(element, "ElementSize")))
                    self.initialize_element(sims, grid_level)
                elif self.element_type == "R27N3D":
                    pass
                else:
                    raise ValueError("Keyword:: /ElementType/ error!")
            self.activate_grid(sims, DictIO.GetAlternative(element, "Contact", None), grid_level)

    def initialize_element(self, sims: Simulation, grid_level):
        if sims.gauss_number > 0:
            self.element.element_initialize(sims, local_coordiates=False)
            self.element.activate_gauss_cell(sims, grid_level)
        else:
            self.element.element_initialize(sims)

    def find_grid_level(self, sims: Simulation, contact):
        if contact is None:
            sims.set_contact_detection(False)
            grid_level = 1
        else:
            sims.set_contact_detection(DictIO.GetAlternative(contact, "ContactDetection", False))
            grid_level = DictIO.GetAlternative(contact, "GridLevel", None)
            if grid_level is None:
                if not sims.contact_detection:
                    grid_level = 1
                elif sims.contact_detection:
                    grid_level = 2
        return grid_level

    def activate_grid(self, sims: Simulation, contact, grid_level):
        self.check_grid_inputs(sims, grid_level)
        self.is_rigid = ti.field(int, shape=grid_level)
        sims.set_body_num(grid_level)
        self.element.set_characteristic_length(sims)

        if not self.node is None:
            print("Warning: Previous node will be override!")
            
        cut_off = 0.
        if sims.contact_detection:
            cut_off = DictIO.GetAlternative(contact, "CutOff", 0.8)
            self.mu = DictIO.GetAlternative(contact, "Friction", 0.)
            self.element.set_contact_position_offset(cut_off)
            if sims.solver_type == "Explicit":
                if sims.dimension == 2:
                    self.node = ContactNodes2D.field()
                elif sims.dimension == 3:
                    self.node = ContactNodes.field()
            else:
                raise RuntimeError("The contact is not supported in implicit MPM currently")
        else:
            if sims.solver_type == "Explicit":
                if sims.dimension == 2:
                    self.node = Nodes2D.field()
                elif sims.dimension == 3:
                    self.node = Nodes.field()
            elif sims.solver_type == "Implicit":
                self.node = ImplicitNodes.field()

        if sims.stabilize == 'F-Bar Method':
            self.extra_node = ExtraNode.field()
            ti.root.dense(ti.ij, (self.element.gridSum, grid_level)).place(self.node, self.extra_node)
        else:
            if sims.calculate_reaction_force:
                self.accmulated_reaction_forces = ti.Vector.field(3, float)
                ti.root.dense(ti.ij, (self.element.gridSum, grid_level)).place(self.node, self.accmulated_reaction_forces)
            else:
                ti.root.dense(ti.ij, (self.element.gridSum, grid_level)).place(self.node)
        
        self.print_grid_message(sims, grid_level, cut_off)
 
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
            print("Contact Detection Activated")
            print("Cut-off Distance Multiplier =", cut_off)
            print("Friction Coefficient =", self.mu)
        print('\n')

    def activate_boundary_constraints(self, sims: Simulation):
        if self.velocity_boundary is None and sims.nvelocity > 0.:
            if sims.dimension == 2:
                self.velocity_boundary = VelocityConstraint2D.field(shape=sims.nvelocity)
            elif sims.dimension == 3:
                self.velocity_boundary = VelocityConstraint.field(shape=sims.nvelocity)
            kernel_initialize_boundary(self.velocity_boundary)

        if self.reflection_boundary is None and sims.nreflection > 0.:
            if sims.dimension == 2:
                self.reflection_boundary = ReflectionConstraint2D.field(shape=sims.nreflection)
            elif sims.dimension == 3:
                self.reflection_boundary = ReflectionConstraint.field(shape=sims.nreflection)
            kernel_initialize_boundary(self.reflection_boundary)

        if self.friction_boundary is None and sims.nfriction > 0.:
            if sims.dimension == 2:
                self.friction_boundary = FrictionConstraint2D.field(shape=sims.nfriction)
            elif sims.dimension == 3:
                self.friction_boundary = FrictionConstraint.field(shape=sims.nfriction)
            kernel_initialize_boundary(self.friction_boundary)

        if self.absorbing_boundary is None and sims.nabsorbing > 0.:
            if sims.dimension == 2:
                self.absorbing_boundary = AbsorbingConstraint2D.field(shape=sims.nabsorbing)
            elif sims.dimension == 3:
                self.absorbing_boundary = AbsorbingConstraint.field(shape=sims.nabsorbing)
            kernel_initialize_boundary(self.absorbing_boundary)

        if self.traction_boundary is None and sims.ntraction > 0.:
            if sims.dimension == 2:
                self.traction_boundary = TractionConstraint2D.field(shape=sims.ntraction)
            elif sims.dimension == 3:
                self.traction_boundary = TractionConstraint.field(shape=sims.ntraction)
            kernel_initialize_boundary(self.traction_boundary)

        if sims.solver_type == "Implicit":
            if self.displacement_boundary is None and sims.ndisplacement > 0.:
                self.displacement_boundary = DisplacementConstraint.field(shape=sims.ndisplacement)
                kernel_initialize_boundary(self.displacement_boundary)

    def iterate_boundary_constraint(self, sims, boundary_constraint, mode):
        if mode == 0:
            print(" Boundary Information ".center(71,"-"))
            if type(boundary_constraint) is dict:
                self.set_boundary_conditions(sims, boundary_constraint)
            elif type(boundary_constraint) is list:
                for boundary in boundary_constraint:
                    self.set_boundary_conditions(sims, boundary) 
        elif mode == 1:
            print('#', "Boundary Earse".center(67, "="), '-')
            if type(boundary_constraint) is dict:
                self.clear_boundary_constraint(sims, boundary_constraint)
            elif type(boundary_constraint) is list:
                for boundary in boundary_constraint:
                    self.clear_boundary_constraint(sims, boundary) 

    def check_boundary_domain(self, sims: Simulation, start_point, end_point):
        if any(start_point < vec3f(0., 0., 0.)):
            raise RuntimeError(f"KeyWord:: /StartPoint/ {start_point} is out of domain {sims.domain}")
        if any(end_point > sims.domain):
            raise RuntimeError(f"KeyWord:: /EndPoint/ {end_point} is out of domain {sims.domain}")

    def set_boundary_conditions(self, sims: Simulation, boundary):
        """
        Set boundary conditions
        Args:
            sims[Simulation]: Simulation dataclass
            boundary[dict]: Boundary dict
                BoundaryType[str]: Boundary type option:[VelocityConstraint, ReflectionConstraint, FrictionConstraint, AbsorbingConstraint, TractionConstraint, DisplacementConstraint]
                NLevel[str/int][option]:  option:[All, 0, 1, 2, ...]
                StartPoint[vec3f]: Start point of boundary
                EndPoint[vec3f]: End point of boundary
                when Boundary type = VelocityConstraint args include:
                    VelocityX[float/None][option]: Prescribed velocity along X axis
                    VelocityY[float/None][option]: Prescribed velocity along Y axis
                    VelocityZ[float/None][option]: Prescribed velocity along Z axis
                    Velocity[list][option]: Prescribed velocity
                when Boundary type = ReflectionConstraint args include:
                    Norm[vec3f]: Outer normal vector
                when Boundary type = FrictionConstraint args include:
                    Friction[float]: Friction angle
                    Norm[vec3f]: Outer normal vector
                when Boundary type = TractionConstraint args include:
                    ExternalForce[vec3f]: External force
                when Boundary type = DisplacementConstraint args include:
                    DisplacementX[float/None][option]: Prescribed displacement along X axis
                    DisplacementY[float/None][option]: Prescribed displacement along Y axis
                    DisplacementZ[float/None][option]: Prescribed displacement along Z axis
                    Displacement[list][option]: Prescribed displacement
        """
        boundary_type = DictIO.GetEssential(boundary, "BoundaryType")
        level = DictIO.GetAlternative(boundary, "NLevel", "All")
        start_point = DictIO.GetEssential(boundary, "StartPoint")
        end_point = DictIO.GetEssential(boundary, "EndPoint")
        self.check_boundary_domain(sims, start_point, end_point)
        inodes = self.element.get_boundary_nodes(start_point, end_point)

        if boundary_type == "VelocityConstraint":
            if self.velocity_boundary is None:
                raise RuntimeError("Error:: /max_velocity_constraint/ is set as zero!")
            
            default_val = [None, None, None] if sims.dimension == 3 else [None, None, 0]
            
            level, nlevel = self.check_nlevel(level)
            self.check_velocity_constraint_num(sims, inodes.shape[0] * nlevel)
            xvelocity = DictIO.GetAlternative(boundary, "VelocityX", None)
            yvelocity = DictIO.GetAlternative(boundary, "VelocityY", None)
            zvelocity = DictIO.GetAlternative(boundary, "VelocityZ", None)
            velocity = DictIO.GetAlternative(boundary, "Velocity", default_val)

            if sims.dimension == 2 and len(velocity) == 2:
                velocity.append(0.)

            if not velocity[0] is None or not velocity[1] is None or not velocity[2] is None:
                xvelocity = velocity[0]
                yvelocity = velocity[1]
                zvelocity = velocity[2]

            if xvelocity is None and yvelocity is None and zvelocity is None:
                raise KeyError("The prescribed velocity has not been set")
            
            fix_v, velocity = [0, 0, 0], [0., 0., 0.]
            if not xvelocity is None:
                fix_v[0] = 1
                velocity[0] = xvelocity
            if not yvelocity is None:
                fix_v[1] = 1
                velocity[1] = yvelocity
            if not zvelocity is None:
                fix_v[2] = 1
                velocity[2] = zvelocity
            
            if sims.dimension == 2:
                for i in range(level, level + nlevel):
                    set_velocity_constraint2D(self.velocity_list, self.velocity_boundary, inodes, vec2i(fix_v[0], fix_v[1]), vec2f(velocity[0], velocity[1]), i)
                copy_valid_velocity_constraint2D(self.velocity_list, self.velocity_boundary)
            elif sims.dimension == 3:
                for i in range(level, level + nlevel):
                    set_velocity_constraint(self.velocity_list, self.velocity_boundary, inodes, vec3i(fix_v), vec3f(velocity), i)
                copy_valid_velocity_constraint(self.velocity_list, self.velocity_boundary)

            print("Boundary Type: Velocity Constraint")
            print("Start Point: ", start_point)
            print("End Point: ", end_point)
            if not xvelocity is None:
                print("Prescribed Velocity along X axis = ", float(xvelocity))
            if not yvelocity is None:
                print("Prescribed Velocity along Y axis = ", float(yvelocity))
            if not zvelocity is None:
                print("Prescribed Velocity along Z axis = ", float(zvelocity))
            print('\n')

        elif boundary_type == "ReflectionConstraint":
            if self.reflection_boundary is None:
                raise RuntimeError("Error:: /max_reflection_constraint/ is set as zero!")

            level, nlevel = self.check_nlevel(level)
            self.check_reflection_constraint_num(sims, inodes.shape[0] * nlevel)
            norm = DictIO.GetEssential(boundary, "Norm")

            if sims.dimension == 2:
                for i in range(level, level + nlevel):
                    set_reflection_constraint2D(self.reflection_list, self.reflection_boundary, inodes, norm, i)
                copy_valid_reflection_constraint2D(self.reflection_list, self.reflection_boundary)
            elif sims.dimension == 3:
                for i in range(level, level + nlevel):
                    set_reflection_constraint(self.reflection_list, self.reflection_boundary, inodes, norm, i)
                copy_valid_reflection_constraint(self.reflection_list, self.reflection_boundary)
            
            print("Boundary Type: Reflection Constraint")
            print("Start Point: ", start_point)
            print("End Point: ", end_point)
            print("Outer Normal Vector = ", norm, '\n')
        
        elif boundary_type == "FrictionConstraint":
            if self.friction_boundary is None:
                raise RuntimeError("Error:: /max_friction_constraint/ is set as zero!")

            level, nlevel = self.check_nlevel(level)
            self.check_friction_constraint_num(sims, inodes.shape[0] * nlevel)
            mu = DictIO.GetEssential(boundary, "Friction")
            norm = DictIO.GetEssential(boundary, "Norm")

            if sims.dimension == 2:
                for i in range(level, level + nlevel):
                    set_friction_constraint2D(self.friction_list, self.friction_boundary, inodes, mu, norm, i)
                copy_valid_friction_constraint2D(self.friction_list, self.friction_boundary)
            elif sims.dimension == 3:
                for i in range(level, level + nlevel):
                    set_friction_constraint(self.friction_list, self.friction_boundary, inodes, mu, norm, i)
                copy_valid_friction_constraint(self.friction_list, self.friction_boundary)
            
            print("Boundary Type: Friction Constraint")
            print("Start Point: ", start_point)
            print("End Point: ", end_point)
            print("Outer Normal Vector = ", norm)
            print("Friction Angle = ", mu, '\n')
            
        elif boundary_type == "AbsorbingConstraint":
            if self.absorbing_boundary is None:
                raise RuntimeError("Error:: /max_absorbing_constraint/ is set as zero!")

            level, nlevel = self.check_nlevel(level)
            self.check_absorbing_constraint_num(sims, inodes.shape[0] * nlevel)

        elif boundary_type == "TractionConstraint":
            if self.traction_boundary is None:
                raise RuntimeError("Error:: /max_traciton_constraint/ is set as zero!")

            level, nlevel = self.check_nlevel(level)
            self.check_traction_constraint_num(sims, inodes.shape[0] * nlevel)
            fex = DictIO.GetEssential(boundary, "ExternalForce")
            
            if sims.dimension == 2:
                for i in range(level, level + nlevel):
                    if self.is_rigid[i] == 0:
                        set_traction_contraint2D(self.traction_list, self.traction_boundary, inodes, fex, i) 
                    else:
                        raise ValueError(f"Traction boundary will be assigned on rigid body (bodyID = {i})")
                copy_valid_traction_constraint2D(self.traction_list, self.traction_boundary)
            elif sims.dimension == 3:
                for i in range(level, level + nlevel):
                    if self.is_rigid[i] == 0:
                        set_traction_contraint(self.traction_list, self.traction_boundary, inodes, fex, i) 
                    else:
                        raise ValueError(f"Traction boundary will be assigned on rigid body (bodyID = {i})")
                copy_valid_traction_constraint(self.traction_list, self.traction_boundary)

            print("Boundary Type: Traction Constraint")
            print("Start Point: ", start_point)
            print("End Point: ", end_point)
            print("Grid Force = ", fex, '\n')

        elif boundary_type == "DisplacementConstraint":
            if sims.solver_type != "Implicit":
                raise RuntimeError("Only Implicit solver can assign displacement boundary conditions")
        
            if self.displacement_boundary is None:
                raise RuntimeError("Error:: dataclass /displacement_boundary/ is not activated!")
            
            default_val = [None, None, None] if sims.dimension == 3 else [None, None, 0]

            xdisplacement = DictIO.GetAlternative(boundary, "DisplacementX", None)
            ydisplacement = DictIO.GetAlternative(boundary, "DisplacementY", None)
            zdisplacement = DictIO.GetAlternative(boundary, "DisplacementZ", None)
            displacement = DictIO.GetAlternative(boundary, "Displacement", default_val)

            if sims.dimension == 2 and len(displacement) == 2:
                displacement.append(0.)

            if not displacement[0] is None or not displacement[1] is None or not displacement[2] is None:
                xdisplacement = displacement[0]
                ydisplacement = displacement[1]
                zdisplacement = displacement[2]

            if xdisplacement is None and ydisplacement is None and zdisplacement is None:
                raise KeyError("The prescribed displacement has not been set")
            
            dofs, velocity = [0, 0, 0], [0., 0., 0.]
            if not xdisplacement is None:
                dofs[0] = 1
                displacement[0] = xdisplacement
            if not ydisplacement is None:
                dofs[1] = 1
                displacement[1] = ydisplacement
            if not zdisplacement is None:
                dofs[2] = 1
                displacement[2] = zdisplacement

            fix_dofs = dofs[0] + dofs[1] + dofs[2] if sims.dimension == 3 else dofs[0] + dofs[1]

            level, nlevel = self.check_nlevel(level)
            self.check_displacement_constraint_num(sims, inodes.shape[0] * nlevel * fix_dofs)

            for i in range(level, level + nlevel):
                if self.is_rigid[i] == 0:
                    set_displacement_contraint(self.displacement_list, self.displacement_boundary, inodes, dofs, displacement, i, fix_dofs) 
                else:
                    raise ValueError(f"Implicit MPM is not supported for rigid body")
            
            print("Boundary Type: Displacement Constraint")
            print("Start Point: ", start_point)
            print("End Point: ", end_point)
            print("Degree of freedom = ", dofs)
            print("Displacement = ", displacement, '\n')


    def clear_boundary_constraint(self, sims: Simulation, boundary):
        boundary_type = DictIO.GetEssential(boundary, "BoundaryType")
        level = DictIO.GetAlternative(boundary, "NLevel", "All")
        start_point = DictIO.GetAlternative(boundary, "StartPoint", vec3f(0, 0, 0))
        end_point = DictIO.GetEssential(boundary, "EndPoint", sims.domain)
        inodes = self.element.get_boundary_nodes(start_point, end_point)
        print("Start Point: ", start_point)
        print("End Point: ", end_point, '\n')

        if boundary_type == "VelocityConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.velocity_list, self.velocity_boundary, inodes, i)

            if sims.dimension == 2:
                copy_valid_velocity_constraint2D(self.velocity_list, self.velocity_boundary)
            elif sims.dimension == 3:
                copy_valid_velocity_constraint(self.velocity_list, self.velocity_boundary)

        elif boundary_type == "ReflectionConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.reflection_list, self.reflection_boundary, inodes, i)

            if sims.dimension == 2:
                copy_valid_reflection_constraint2D(self.reflection_list, self.reflection_boundary)
            elif sims.dimension == 3:
                copy_valid_reflection_constraint(self.reflection_list, self.reflection_boundary)

        elif boundary_type == "FrictionConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.friction_list, self.friction_boundary, inodes, i)
            
            if sims.dimension == 2:
                copy_valid_friction_constraint2D(self.friction_list, self.friction_boundary)
            elif sims.dimension == 3:
                copy_valid_friction_constraint(self.friction_list, self.friction_boundary)

        elif boundary_type == "AbsorbingConstraint":
            pass

        elif boundary_type == "TractionConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.traction_list, self.traction_boundary, inodes, i) 

            if sims.dimension == 2:
                copy_valid_traction_constraint2D(self.traction_list, self.traction_boundary)
            elif sims.dimension == 3:    
                copy_valid_traction_constraint(self.traction_list, self.traction_boundary)

        elif boundary_type == "DisplacementConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_displacement_constraint(self.displacement_boundary, inodes, i) 

    def check_nlevel(self, level):
        if level == "All":
            level = 0
            nlevel = self.node.shape[1]
        elif type(level)is int and level < self.node.shape[1]:
            nlevel = 1
        else:
            raise ValueError("NLevel should be smaller than the grid level")
        return level, nlevel

    def read_boundary_constraint(self, sims: Simulation, boundary_constraint):
        print(" Read Boundary Information ".center(71,"-"))
        if not os.path.exists(boundary_constraint):
            raise EOFError("Invaild path")

        boundary_constraints = open(boundary_constraint, 'r')
        while True:
            line = str.split(boundary_constraints.readline())
            if not line: break

            elif line[0] == '#': continue

            elif line[0] == "VelocityConstraint":
                if self.velocity_boundary is None:
                    raise RuntimeError("Error:: /max_velocity_constraint/ is set as zero!")

                boundary_size = int(line[1])
                self.check_velocity_constraint_num(sims, boundary_size)
                for _ in range(boundary_size):
                    boundary = str.split(boundary_constraints.readline())
                    self.velocity_boundary[self.velocity_list[0]].set_boundary_condition(int(boundary[0]), int(boundary[1]),
                                                                                            vec3f(float(boundary[2]), float(boundary[3]), float(boundary[4])),
                                                                                            vec3f(float(boundary[5]), float(boundary[6]), float(boundary[7])))
                    self.velocity_list[0] += 1

            elif line[0] == "ReflectionConstraint":
                if self.reflection_boundary is None:
                    raise RuntimeError("Error:: /max_reflection_constraint/ is set as zero!")

                boundary_size = int(line[1])
                self.check_reflection_constraint_num(sims, boundary_size)
                for _ in range(boundary_size):
                    boundary = str.split(boundary_constraints.readline())
                    self.reflection_boundary[self.reflection_list[0]].set_boundary_condition(int(boundary[0]), int(boundary[1]), 
                                                                                                vec3f(float(boundary[3]), float(boundary[4]), float(boundary[5])),
                                                                                                vec3f(float(boundary[6]), float(boundary[7]), float(boundary[8])),
                                                                                                vec3f(float(boundary[9]), float(boundary[10]), float(boundary[11])))
                    self.reflection_list[0] += 1
                    
            elif line[0] == "FrictionConstraint":
                if self.friction_boundary is None:
                    raise RuntimeError("Error:: /max_friction_constraint/ is set as zero!")

                boundary_size = int(line[1])
                self.check_friction_constraint_num(sims, boundary_size)
                for _ in range(boundary_size):
                    boundary = str.split(boundary_constraints.readline())
                    self.friction_boundary[self.friction_list[0]].set_boundary_condition(int(boundary[0]), int(boundary[1]), float(boundary[2]),
                                                                                            vec3f(float(boundary[3]), float(boundary[4]), float(boundary[5])))
                    self.friction_list[0] += 1
                    
            elif line[0] == "AbsorbingConstraint":
                if self.absorbing_boundary is None:
                    raise RuntimeError("Error:: /max_absorbing_constraint/ is set as zero!")

                boundary_size = int(line[1])
                self.check_absorbing_constraint_num(sims, boundary_size)
                    
            elif line[0] == "TractionConstraint":
                if self.traction_boundary is None:
                    raise RuntimeError("Error:: /max_traction_constraint/ is set as zero!")

                boundary_size = int(line[1])
                self.check_velocity_constraint_num(sims, boundary_size)
                for _ in range(boundary_size):
                    boundary = str.split(boundary_constraints.readline())
                    self.traction_boundary[self.traction_list[0]].set_boundary_condition(int(boundary[0]), int(boundary[1]),
                                                                                            vec3f(float(boundary[2]), float(boundary[3]), float(boundary[4])))
                    self.traction_list[0] += 1

            elif line[0] == "DisplacementConstraint":
                pass

    def write_boundary_constraint(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if self.velocity_list[0] > 0:
            pass
        if self.reflection_list[0] > 0:
            pass
        if self.friction_list[0] > 0:
            pass
        if self.absorbing_list[0] > 0:
            pass
        if self.traction_list[0] > 0:
            pass

    def get_material_ptr(self):
        return self.material
    
    def get_element_ptr(self):
        return self.element
    
    def get_node_ptr(self):
        return self.node
    
    def get_particle_ptr(self):
        return self.particle
    
    def check_velocity_constraint_num(self, sims: Simulation, constraint_num):
        if self.velocity_list[0] + constraint_num > sims.nvelocity:
            raise ValueError ("The number of velocity constraints should be set as: ", self.velocity_list[0] + constraint_num)
        
    def check_reflection_constraint_num(self, sims: Simulation, constraint_num):
        if self.reflection_list[0] + constraint_num > sims.nreflection:
            raise ValueError ("The number of reflection constraints should be set as: ", self.reflection_list[0] + constraint_num)
        
    def check_friction_constraint_num(self, sims: Simulation, constraint_num):
        if self.friction_list[0] + constraint_num > sims.nfriction:
            raise ValueError ("The number of friction constraints should be set as: ", self.friction_list[0] + constraint_num)
        
    def check_absorbing_constraint_num(self, sims: Simulation, constraint_num):
        if self.absorbing_list[0] + constraint_num > sims.nabsorbing:
            raise ValueError ("The number of absorbing constraints should be set as: ", self.absorbing_list[0] + constraint_num)
        
    def check_traction_constraint_num(self, sims: Simulation, constraint_num):
        if self.traction_list[0] + constraint_num > sims.ntraction:
            raise ValueError ("The number of traction constraints should be set as: ", self.traction_list[0] + constraint_num)
        
    def check_displacement_constraint_num(self, sims: Simulation, constraint_num):
        if self.displacement_list[0] + constraint_num > sims.ndisplacement:
            raise ValueError ("The number of displacement constraints should be set as: ", self.displacement_list[0] + constraint_num)
        
    def check_particle_num(self, sims: Simulation, particle_number):
        if self.particleNum[0] + particle_number > sims.max_particle_num:
            raise ValueError ("The MPM particles should be set as: ", self.particleNum[0] + particle_number)

    def find_min_z_position(self):
        return find_min_z_position_(self.particleNum[0], self.particle)
    
    def find_bounding_sphere_radius(self):
        rad_max = find_particle_max_radius_(self.particleNum[0], self.particle)
        rad_min = find_particle_min_radius_(self.particleNum[0], self.particle)
        return rad_max, rad_min
    
    def find_particle_min_radius(self):
        return find_particle_min_radius_(self.particleNum[0], self.particle)
    
    def find_particle_max_radius(self):
        return find_particle_max_radius_(self.particleNum[0], self.particle)
    
    def find_particle_min_mass(self):
        return find_particle_min_mass_(self.particleNum[0], self.particle)
    
    def reset_verlet_disp(self):
        reset_verlet_disp_(self.particleNum[0], self.particle)

    def get_critical_timestep(self):
        max_vel = find_max_velocity_(self.particleNum[0], self.particle)
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
        self.mass_cut_off = 1e-5 * density / matcount * grid_size[0] * grid_size[1] * grid_size[2]
        self.volume_cut_off = 1e-5 * grid_size[0] * grid_size[1] * grid_size[2]
    
    def update_particle_properties_in_region(self, override, property_name, value, is_in_region):
        print(" Modify Particle Information ".center(71, '-'))
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        override = 1 if not override else 0
        if property_name == "bodyID":
            modify_particle_bodyID_in_region(value, self.particleNum[0], self.particle, is_in_region)
        elif property_name == "materialID":
            modify_particle_materialID_in_region(value, self.particleNum[0], self.particle, self.material.matProps, is_in_region)
        elif property_name == "position":
            modify_particle_position_in_region(override, value, self.particleNum[0], self.particle, is_in_region)
        elif property_name == "velocity":
            modify_particle_velocity_in_region(override, value, self.particleNum[0], self.particle, is_in_region)
        elif property_name == "traction":
            modify_particle_traction_in_region(override, value, self.particleNum[0], self.particle, is_in_region)
        elif property_name == "stress":
            modify_particle_stress_in_region(override, value, self.particleNum[0], self.particle, is_in_region)
        elif property_name == "fix_velocity":
            FIX = {
                    "Free": 0,
                    "Fix": 1
                   }
            fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
            modify_particle_fix_v_in_region(fix_v, self.particleNum[0], self.particle, is_in_region)
        else:
            valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
            raise KeyError(f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}")
        
    def update_particle_properties(self, override, property_name, value, bodyID):
        print(" Modify Body Information ".center(71, '-'))
        print("Target BodyID =", bodyID)
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, '\n')

        override = 1 if not override else 0
        if property_name == "bodyID":
            modify_particle_bodyID(value, self.particleNum[0], self.particle, bodyID)
        elif property_name == "materialID":
            modify_particle_materialID(value, self.particleNum[0], self.particle, self.material.matProps, bodyID)
        elif property_name == "position":
            modify_particle_position(override, value, self.particleNum[0], self.particle, bodyID)
        elif property_name == "velocity":
            modify_particle_velocity(override, value, self.particleNum[0], self.particle, bodyID)
        elif property_name == "traction":
            modify_particle_traction(override, value, self.particleNum[0], self.particle, bodyID)
        elif property_name == "stress":
            modify_particle_stress(override, value, self.particleNum[0], self.particle, bodyID)
        elif property_name == "fix_velocity":
            FIX = {
                    "Free": 0,
                    "Fix": 1
                   }
            fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
            modify_particle_fix_v(fix_v, self.particleNum[0], self.particle, bodyID)
        else:
            valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
            raise KeyError(f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}")

    def check_overlap_coupling(self):
        initial_particle = self.particleNum[0]
        self.particleNum[0] = update_particle_storage_(self.particleNum[0], self.particle, self.material.stateVars)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')
        
    def delete_particles(self, bodyID):
        initial_particle = self.particleNum[0]
        kernel_delete_particles(self.particleNum[0], self.particle, bodyID)
        self.particleNum[0] = update_particle_storage_(self.particleNum[0], self.particle, self.material.stateVars)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')

    def delete_particles_in_region(self, is_in_region):
        initial_particle = self.particleNum[0]
        kernel_delete_particles_in_region(self.particleNum[0], self.particle, is_in_region)
        self.particleNum[0] = update_particle_storage_(self.particleNum[0], self.particle, self.material.stateVars)
        finial_particle = self.particleNum[0]
        print(f"Total {-finial_particle + initial_particle} particles has been deleted", '\n')

    def check_particle_in_domain(self, sims: Simulation):
        check_in_domain(sims.domain, self.particleNum[0], self.particle)

    def set_boundary(self, sims: Simulation):
        if sims.boundary[0] == 0:
            start_point = vec3f(0, 0, 0)
            end_point = vec3f(0, sims.domain[1], sims.domain[2])
            norm = vec3f(-1, 0, 0)
            self.set_boundary_conditions(sims, boundary={
                                                             "BoundaryType":   "ReflectionConstraint",
                                                             "Norm":           norm,
                                                             "StartPoint":     start_point,
                                                             "EndPoint":       end_point
                                                        })
            
            start_point = vec3f(sims.domain[0], 0, 0)
            end_point = vec3f(sims.domain[0], sims.domain[1], sims.domain[2])
            norm = vec3f(1, 0, 0)
            self.set_boundary_conditions(sims, boundary={
                                                             "BoundaryType":   "ReflectionConstraint",
                                                             "Norm":           norm,
                                                             "StartPoint":     start_point,
                                                             "EndPoint":       end_point
                                                        })
        if sims.boundary[1] == 0:
            start_point = vec3f(0, 0, 0)
            end_point = vec3f(sims.domain[0], 0, sims.domain[2])
            norm = vec3f(0, -1, 0)
            self.set_boundary_conditions(sims, boundary={
                                                             "BoundaryType":   "ReflectionConstraint",
                                                             "Norm":           norm,
                                                             "StartPoint":     start_point,
                                                             "EndPoint":       end_point
                                                        })
            
            start_point = vec3f(0, sims.domain[1], 0)
            end_point = vec3f(sims.domain[0], sims.domain[1], sims.domain[2])
            norm = vec3f(0, 1, 0)
            self.set_boundary_conditions(sims, boundary={
                                                             "BoundaryType":   "ReflectionConstraint",
                                                             "Norm":           norm,
                                                             "StartPoint":     start_point,
                                                             "EndPoint":       end_point
                                                        })
            
        if sims.boundary[2] == 0:
            start_point = vec3f(0, 0, 0)
            end_point = vec3f(sims.domain[0], sims.domain[1], 0)
            norm = vec3f(0, 0, -1)
            self.set_boundary_conditions(sims, boundary={
                                                             "BoundaryType":   "ReflectionConstraint",
                                                             "Norm":           norm,
                                                             "StartPoint":     start_point,
                                                             "EndPoint":       end_point
                                                        })
            
            start_point = vec3f(0, 0, sims.domain[2])
            end_point = vec3f(sims.domain[0], sims.domain[1], sims.domain[2])
            norm = vec3f(0, 0, 1)
            self.set_boundary_conditions(sims, boundary={
                                                             "BoundaryType":   "ReflectionConstraint",
                                                             "Norm":           norm,
                                                             "StartPoint":     start_point,
                                                             "EndPoint":       end_point
                                                        })
            
