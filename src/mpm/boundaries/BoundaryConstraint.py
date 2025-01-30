import os, warnings
import numpy as np

from src.mpm.boundaries.BoundaryCore import *
from src.mpm.boundaries.BoundaryStrcut import *
from src.mpm.elements.ElementBase import ElementBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3i
from src.utils.RegionFunction import RegionFunction


class BoundaryConstraints(object):
    def __init__(self) -> None:
        self.grid_layer = 1
        self.myRegion = None
        self.psize = np.array([], dtype=np.int32)
        self.is_rigid = np.zeros(self.grid_layer)
        self.new_boundaries = False
        self.velocity_boundary = None
        self.reflection_boundary = None
        self.friction_boundary = None
        self.absorbing_boundary = None
        self.traction_boundary = None
        self.displacement_boundary = None
        self.particle_traction = None

        self.velocity_list = np.zeros(1, dtype=np.int32)
        self.reflection_list = np.zeros(1, dtype=np.int32)
        self.friction_list = np.zeros(1, dtype=np.int32)
        self.absorbing_list = np.zeros(1, dtype=np.int32)
        self.traction_list = np.zeros(1, dtype=np.int32)
        self.ptraction_list = np.zeros(1, dtype=np.int32)
        self.displacement_list = np.zeros(1, dtype=np.int32)

        self.velocity_dict = {}
        self.reflection_dict = {}
        self.friction_dict = {}
        self.absorbing_dict = {}
        self.traction_dict = {}
        self.displacement_dict = {}

        self.axis = {0: "X", 1: "Y", 2: "Z"}

    def set_layer_number(self, grid_level):
        self.grid_layer = grid_level

    def get_essentials(self, is_rigid, psize, myRegion):
        self.is_rigid = is_rigid.to_numpy()
        self.myRegion = myRegion
        self.psize = psize

    def activate_boundary_constraints(self, sims: Simulation):
        if self.velocity_boundary is None and sims.nvelocity > 0.:
            self.velocity_boundary = VelocityConstraint.field(shape=sims.nvelocity)
            kernel_initialize_boundary(self.velocity_boundary)

        if self.reflection_boundary is None and sims.nreflection > 0.:
            self.reflection_boundary = ReflectionConstraint.field(shape=sims.nreflection)
            kernel_initialize_boundary(self.reflection_boundary)

        if self.friction_boundary is None and sims.nfriction > 0.:
            self.friction_boundary = FrictionConstraint.field(shape=sims.nfriction)
            kernel_initialize_boundary(self.friction_boundary)

        if self.absorbing_boundary is None and sims.nabsorbing > 0.:
            self.absorbing_boundary = AbsorbingConstraint.field(shape=sims.nabsorbing)
            kernel_initialize_boundary(self.absorbing_boundary)

        if self.traction_boundary is None and sims.ntraction > 0.:
            self.traction_boundary = TractionConstraint.field(shape=sims.ntraction)
            kernel_initialize_boundary(self.traction_boundary)

        if self.particle_traction is None:
            if sims.ptraction_method == "Virtual":
                self.particle_traction = VirtualLoad(sims.dimension)
            else:
                if sims.nptraction > 0.:
                    if sims.dimension == 3:
                        if sims.ptraction_method == "Stable":
                            self.particle_traction = ParticleLoad.field(shape=sims.nptraction)
                        elif sims.ptraction_method == "Nanson":
                            self.particle_traction = ParticleLoadNanson.field(shape=sims.nptraction)
                    elif sims.dimension == 2:
                        if sims.is_2DAxisy:
                            self.particle_traction = ParticleLoad2DAxisy.field(shape=sims.nptraction)
                        else:
                            if sims.ptraction_method == "Stable":
                                if sims.material_type == "TwoPhaseSingleLayer":
                                    self.particle_traction = ParticleLoadTwoPhase2D.field(shape=sims.nptraction)
                                else:
                                    self.particle_traction = ParticleLoad2D.field(shape=sims.nptraction)
                            elif sims.ptraction_method == "Nanson":
                                self.particle_traction = ParticleLoadNanson2D.field(shape=sims.nptraction)
                    kernel_initialize_boundary(self.particle_traction)

        if sims.solver_type == "Implicit":
            if self.displacement_boundary is None and sims.ndisplacement > 0.:
                self.displacement_boundary = DisplacementConstraint.field(shape=sims.ndisplacement)
                kernel_initialize_boundary(self.displacement_boundary)

    def check_nlevel(self, level):
        if level == "All":
            level = 0
            nlevel = self.grid_layer
        elif type(level) is int and level < self.grid_layer:
            nlevel = 1
        else:
            raise ValueError("NLevel should be smaller than the grid level")
        return level, nlevel

    def check_boundary_domain(self, sims: Simulation, start_point, end_point):
        if sims.dimension == 3:
            if any(start_point < vec3f(0., 0., 0.)):
                raise RuntimeError(f"KeyWord:: /StartPoint/ {start_point} is out of domain {sims.domain}")
            if any(end_point > sims.domain):
                raise RuntimeError(f"KeyWord:: /EndPoint/ {end_point} is out of domain {sims.domain}")
        elif sims.dimension == 2:
            if any(start_point < vec2f(0., 0.)):
                raise RuntimeError(f"KeyWord:: /StartPoint/ {start_point} is out of domain {sims.domain}")
            if any(end_point > sims.domain):
                raise RuntimeError(f"KeyWord:: /EndPoint/ {end_point} is out of domain {sims.domain}")
            
    def get_freedoms(self, sims: Simulation, freedoms):
        if sims.dimension == 2 and len(freedoms) == 2:
            freedoms.append(0.)

        xfreedoms = freedoms[0]
        yfreedoms = freedoms[1]
        zfreedoms = freedoms[2]

        if xfreedoms is None and yfreedoms is None and zfreedoms is None:
            raise KeyError("The prescribed displacement has not been set")
        
        return_vals = []
        if not xfreedoms is None:
            return_vals.append((0, xfreedoms))
        if not yfreedoms is None:
            return_vals.append((1, yfreedoms))
        if sims.dimension == 3:
            if not zfreedoms is None:
                return_vals.append((2, zfreedoms))
        return return_vals
    
    def define_signs(self, norm, sign):
        if sign is None:
            sign = "Positive" if abs(norm) / norm > 0 else "Negative"
        elif isinstance(sign, (int, float)): 
            sign = "Positive" if sign > 0 else "Negative"
        return sign
    
    def check_axial_norm(self, dims, norm):
        total_dofs = []
        for i in range(len(norm)):
            if norm[i] != 0:
                total_dofs.append(i)
        if len(total_dofs) > 1:
            default_norm = []
            if total_dofs[0] == 0:
                default_norm = [norm[0]].append([0 for _ in range(dims - 1)]) 
            elif total_dofs[0] == 1:
                default_norm = [0, norm[1]].append([0 for _ in range(dims - 1)]) 
            warnings.warn(f"Keyword:: /Norm/ refers to outer normal direction towards to X or Y or Z axes. Input {norm} are not aligned with these three axes, thus using {default_norm} by default")
        elif len(total_dofs) == 0:
            raise RuntimeError("Invaid Keyword:: At least one component of /Norm/ needs to be nonzero")
        return total_dofs[0]

    def get_norms(self, sims: Simulation, norm, sign):
        if isinstance(norm, (list, tuple, np.ndarray, ti.lang.matrix.Vector)):
            d = self.check_axial_norm(sims.dimension, list(norm))
            dirs = DictIO.GetEssential(self.axis, d)
            sign = self.define_signs(norm[d], sign)
            norm = sign + ' ' + dirs
        elif isinstance(norm, (int, float)):
            if int(abs(norm)) == 0: dirs = "X"
            elif int(abs(norm)) == 1: dirs = "Y"
            elif int(abs(norm)) == 2: dirs = "Z"
            sign = self.define_signs(norm, sign)
            norm = sign + ' ' + dirs
        return norm

    def get_dirs_and_signs(self, norm: str):
        dirs, signs = -1, 0
        if norm.startswith("Negative"): signs = -1
        elif norm.startswith("Positive"): signs = 1
        if norm.endswith("X"): dirs = 0
        elif norm.endswith("Y"): dirs = 1
        elif norm.endswith("Z"): dirs = 2
        return dirs, signs
    
    def build_constraint_dict(self, values, *args):
        keys = np.array(np.meshgrid(*args)).T.reshape(-1, len(args))
        values_repeated = np.tile(values, len(args[0]))
        return dict(zip(map(tuple, keys), values_repeated))
    
    def set_velocity_constraints(self, sims: Simulation, boundary, level, nlevel, start_point, end_point, inodes):
        if self.velocity_boundary is None:
            raise RuntimeError("Error:: /max_velocity_constraint/ is set as zero!")
        
        xvelocity = DictIO.GetAlternative(boundary, "VelocityX", None)
        yvelocity = DictIO.GetAlternative(boundary, "VelocityY", None)
        zvelocity = DictIO.GetAlternative(boundary, "VelocityZ", None)
        default_val = [xvelocity, yvelocity, zvelocity] if sims.dimension == 3 else [xvelocity, yvelocity, 0]
        velocity = DictIO.GetAlternative(boundary, "Velocity", default_val)
        freedoms = self.get_freedoms(sims, velocity)

        dirs, values = zip(*freedoms)
        self.velocity_dict.update(self.build_constraint_dict(np.array(values), inodes, np.array(dirs), np.arange(level, level + nlevel, 1)))

        expected_dirs = {0: "X", 1: "Y", 2: "Z"}
        print("Boundary Type: Velocity Constraint")
        print("Start Point: ", start_point)
        print("End Point: ", end_point)
        print("Total involved nodes: ", inodes.shape[0])
        for freedom in freedoms:
            print(f"Prescribed Velocity along {expected_dirs.get(freedom[0])} axis = ", float(freedom[1]))
        print('\n')

    def set_reflection_constraints(self, sims: Simulation, boundary, level, nlevel, start_point, end_point, inodes):
        if self.reflection_boundary is None:
            raise RuntimeError("Error:: /max_reflection_constraint/ is set as zero!")

        norm = DictIO.GetEssential(boundary, "Norm")
        sign = DictIO.GetAlternative(boundary, "Sign", None)
        norms = self.get_norms(sims, norm, sign)
        dirs, signs = self.get_dirs_and_signs(norms)
        self.reflection_dict.update(self.build_constraint_dict(np.zeros(1), inodes, dirs, signs, np.arange(level, level + nlevel, 1)))
        
        print("Boundary Type: Reflection Constraint")
        print("Start Point: ", start_point)
        print("End Point: ", end_point)
        print("Total involved nodes: ", inodes.shape[0])
        print(f"Outer Normal Direction is {norms} Axis", '\n')

    def set_friction_constraints(self, sims: Simulation, boundary, level, nlevel, start_point, end_point, inodes):
        if self.friction_boundary is None:
            raise RuntimeError("Error:: /max_friction_constraint/ is set as zero!")

        mu = DictIO.GetEssential(boundary, "Friction")
        norm = DictIO.GetEssential(boundary, "Norm")
        sign = DictIO.GetAlternative(boundary, "Sign", None)
        norms = self.get_norms(sims, norm, sign)
        dirs, signs = self.get_dirs_and_signs(norms)
        self.friction_dict.update(self.build_constraint_dict(np.zeros(mu), inodes, dirs, signs, np.arange(level, level + nlevel, 1)))

        print("Boundary Type: Friction Constraint")
        print("Start Point: ", start_point)
        print("End Point: ", end_point)
        print("Total involved nodes: ", inodes.shape[0])
        print(f"Outer Normal Direction is {norms} Axis")
        print("Friction Angle = ", mu, '\n')

    def set_absorbing_constraints(self, sims: Simulation, boundary, level, nlevel, start_point, end_point, inodes):
        if self.absorbing_boundary is None:
            raise RuntimeError("Error:: /max_absorbing_constraint/ is set as zero!")

        self.check_absorbing_constraint_num(sims, inodes.shape[0] * nlevel)

    def set_traction_constraints(self, sims: Simulation, boundary, level, nlevel, start_point, end_point, inodes):
        if self.traction_boundary is None:
            raise RuntimeError("Error:: /max_traciton_constraint/ is set as zero!")
        
        for i in range(level, level + nlevel):    
            if self.is_rigid[i] == 1:
                raise ValueError(f"Traction boundary will be assigned on rigid body (bodyID = {i})")

        xfext = DictIO.GetAlternative(boundary, "ExternalForceX", None)
        yfext = DictIO.GetAlternative(boundary, "ExternalForceY", None)
        zfext = DictIO.GetAlternative(boundary, "ExternalForceZ", None)
        default_val = [xfext, yfext, zfext] if sims.dimension == 3 else [xfext, yfext, 0]
        fext = DictIO.GetAlternative(boundary, "ExternalForce", default_val)
        freedoms = self.get_freedoms(sims, fext)

        dirs, values = zip(*freedoms)
        self.traction_dict.update(self.build_constraint_dict(np.array(values), inodes, np.array(dirs), np.arange(level, level + nlevel, 1)))
        
        expected_dirs = {0: "X", 1: "Y", 2: "Z"}
        print("Boundary Type: Traction Constraint")
        print("Start Point: ", start_point)
        print("End Point: ", end_point)
        print("Total involved nodes: ", inodes.shape[0])
        for freedom in freedoms:
            print(f"Prescribed Grid Force along {expected_dirs.get(freedom[0])} axis = ", float(freedom[1]))
        print('\n')

    def set_displacement_constraints(self, sims: Simulation, boundary, level, nlevel, start_point, end_point, inodes):
        if sims.solver_type != "Implicit":
            raise RuntimeError("Only Implicit solver can assign displacement boundary conditions")
    
        if self.displacement_boundary is None:
            raise RuntimeError("Error:: dataclass /displacement_boundary/ is not activated!")

        xdisplacement = DictIO.GetAlternative(boundary, "DisplacementX", None)
        ydisplacement = DictIO.GetAlternative(boundary, "DisplacementY", None)
        zdisplacement = DictIO.GetAlternative(boundary, "DisplacementZ", None)
        default_val = [xdisplacement, ydisplacement, zdisplacement] if sims.dimension == 3 else [xdisplacement, ydisplacement, 0]
        displacement = DictIO.GetAlternative(boundary, "Displacement", default_val)
        freedoms = self.get_freedoms(sims, displacement)

        dirs, values = zip(*freedoms)
        self.displacement_dict.update(self.build_constraint_dict(np.array(values), inodes, np.array(dirs), np.arange(level, level + nlevel, 1)))
        
        expected_dirs = {0: "X", 1: "Y", 2: "Z"}
        print("Boundary Type: Displacement Constraint")
        print("Start Point: ", start_point)
        print("End Point: ", end_point)
        print("Total involved nodes: ", inodes.shape[0])
        print("Degree of freedom = ", freedoms)
        for freedom in freedoms:
            print(f"Prescribed Displacement along {expected_dirs.get(freedom[0])} axis = ", float(freedom[1]))
        print('\n')

    def split_dict_to_arrays(self, input_dict):
        if not input_dict:
            return [], np.array([])
        
        split_keys = np.array(list(input_dict.keys()))
        values = np.array(list(input_dict.values()))
        return split_keys, values

    def copy_dict_to_field(self, sims):
        nvelocity = len(self.velocity_dict)
        if nvelocity > 0:
            self.check_velocity_constraint_num(sims, nvelocity)
            keys, values = self.split_dict_to_arrays(dict(sorted(self.velocity_dict.items(), key=lambda item: item)))
            nodeID = np.ascontiguousarray(keys[:, 0])
            levels = np.ascontiguousarray(keys[:, 2])
            dirs = np.ascontiguousarray(keys[:, 1])
            set_contraints(self.velocity_boundary, nodeID, levels, dirs, values)
            self.velocity_list[0] = nvelocity

        nreflection = len(self.reflection_dict)
        if nreflection > 0:
            self.check_reflection_constraint_num(sims, nreflection)
            keys, values = self.split_dict_to_arrays(dict(sorted(self.reflection_dict.items(), key=lambda item: item)))
            nodeID = np.ascontiguousarray(keys[:, 0])
            levels = np.ascontiguousarray(keys[:, 3])
            dirs = np.ascontiguousarray(keys[:, 1])
            signs = np.ascontiguousarray(keys[:, 2])
            set_reflection_constraint(self.reflection_boundary, nodeID, levels, dirs, signs)
            self.reflection_list[0] = nreflection

        nfriction = len(self.friction_dict)
        if nfriction > 0:
            self.check_friction_constraint_num(sims, nfriction)
            keys, values = self.split_dict_to_arrays(dict(sorted(self.friction_dict.items(), key=lambda item: item)))
            nodeID = np.ascontiguousarray(keys[:, 0])
            levels = np.ascontiguousarray(keys[:, 4])
            dirs = np.ascontiguousarray(keys[:, 1])
            signs = np.ascontiguousarray(keys[:, 2])
            mu = np.ascontiguousarray(keys[:, 3])
            set_friction_constraint(self.friction_boundary, nodeID, levels, dirs, signs, mu)
            self.friction_list[0] = nfriction
        
        ntraction = len(self.traction_dict)
        if ntraction > 0:
            self.check_traction_constraint_num(sims, ntraction)
            keys, values = self.split_dict_to_arrays(dict(sorted(self.traction_dict.items(), key=lambda item: item)))
            nodeID = np.ascontiguousarray(keys[:, 0])
            levels = np.ascontiguousarray(keys[:, 2])
            dirs = np.ascontiguousarray(keys[:, 1])
            set_contraints(self.traction_boundary, nodeID, levels, dirs, values)
            self.traction_list[0] = ntraction

        ndisplacement = len(self.displacement_dict)
        if ndisplacement > 0:
            self.check_displacement_constraint_num(sims, ndisplacement)
            keys, values = self.split_dict_to_arrays(dict(sorted(self.displacement_dict.items(), key=lambda item: item)))
            nodeID = np.ascontiguousarray(keys[:, 0])
            levels = np.ascontiguousarray(keys[:, 2])
            dirs = np.ascontiguousarray(keys[:, 1])
            set_contraints(self.displacement_boundary, nodeID, levels, dirs, values) 
            self.displacement_list[0] = ndisplacement

    def set_particle_traction(self, sims: Simulation, boundary, particleNum, startNum, particle, psize, region: RegionFunction=None):
        if sims.ptraction_method == "Virtual":
            raise RuntimeError("Please input virtual stress field from function /mainMPM -> MPM().add_virtual_stress_field/")
        traction_force = DictIO.GetEssential(boundary, "Pressure") 
        if np.linalg.norm(np.array(traction_force)) == 0: return
        
        if self.particle_traction is None:
            raise RuntimeError("Error:: /max_particle_traction_constraint/ is set as zero!")
        
        region_function = None
        if not region is None:
            region_function = region.function
        region_name = DictIO.GetAlternative(boundary, "RegionName", None)
        if region_name:
            traction_region: RegionFunction = self.get_region_ptr(region_name)
            region_function = traction_region.function
        region_function = DictIO.GetAlternative(boundary, "RegionFunction", region_function)

        if isinstance(traction_force, float):
            traction_force *= DictIO.GetEssential(boundary, "OuterNormal")
        elif isinstance(traction_force, (list, tuple)):
            if len(traction_force) == 2:
                traction_force = vec2f(traction_force)
            elif len(traction_force) == 3:
                traction_force = vec3f(traction_force)

        fluid_traction = [0., 0.]
        if sims.material_type == "TwoPhaseSingleLayer":
            fluid_traction = DictIO.GetEssential(boundary, "FluidPressure") 
            if isinstance(fluid_traction, float):
                fluid_traction *= DictIO.GetEssential(boundary, "OuterNormal")
            elif isinstance(fluid_traction, (list, tuple)):
                if len(fluid_traction) == 2:
                    fluid_traction = vec2f(fluid_traction)
                elif len(fluid_traction) == 3:
                    fluid_traction = vec3f(fluid_traction)

        ptraction_num = prefind_particle_traction_contraint(self.ptraction_list, self.particle_traction, startNum, particleNum, particle, region_function)
        self.check_particle_traction_constraint_num(sims, ptraction_num)
        if sims.dimension == 3:
            set_particle_traction_contraint(self.ptraction_list, self.particle_traction, startNum, particleNum, particle, region_function, traction_force, psize)
        elif sims.dimension == 2:
            if sims.material_type == "TwoPhaseSingleLayer":
                set_particle_traction_contraint_twophase_2D(self.ptraction_list, self.particle_traction, startNum, particleNum, particle, region_function, traction_force, fluid_traction, psize)
            else:
                set_particle_traction_contraint_2D(self.ptraction_list, self.particle_traction, startNum, particleNum, particle, region_function, traction_force, psize)

        print("Boundary Type: Particle Traction Constraint")
        print("Total involved nodes: ", ptraction_num)
        print("Traction = ", traction_force, '\n')

    def set_virtual_stress_field(self, sims: Simulation, element: ElementBase, boundary, region_function):
        if region_function is None:
            def is_in_region(x): return True
            region_function = is_in_region
        region_function = ti.pyfunc(region_function)

        virtual_stress = None
        if "ConfiningPressure" in boundary:
            pressure = DictIO.GetEssential(boundary, "ConfiningPressure")
            virtual_stress_field = [0., 0., 0., 0., 0., 0.]
            if isinstance(pressure, (int, float)):
                virtual_stress_field = [-pressure, -pressure, -pressure, 0., 0., 0.]
            elif isinstance(pressure, (list, tuple, np.ndarray)):
                virtual_stress_field = pressure.copy()
            def get_virtual_field(x): 
                return virtual_stress_field if region_function(x) else [0., 0., 0., 0., 0., 0.]
            virtual_stress = ti.func(get_virtual_field)
        else:    
            virtual_stress_field = DictIO.GetEssential(boundary, "VirtualStress")
            if not isinstance(virtual_force_field, function):
                raise ValueError("Keyword:: /VirtualStress/ should be a Python function")
            virtual_stress_field = ti.pyfunc(virtual_stress_field)
            def get_virtual_field(x): 
                return virtual_stress_field(x) if region_function(x) else [0., 0., 0., 0., 0., 0.]
            virtual_stress = ti.func(get_virtual_field)

        virtual_force = None
        if "ConfiningPressure" in boundary:
            virtual_force_field = [0. for _ in range(sims.dimension)]
            def get_virtual_force(x): return virtual_force_field
            virtual_force = ti.func(get_virtual_force)
        else:
            virtual_force_field = DictIO.GetAlternative(boundary, "VirtualForce", None)
            zeros = [0. for _ in range(sims.dimension)]
            if virtual_force_field is None or isinstance(virtual_force_field, (list, tuple, np.ndarray)):
                if virtual_force_field is None: virtual_force_field = [0. for _ in range(sims.dimension)]
                if len(list(virtual_force_field)) != sims.dimension: raise ValueError(f"The dimension of /VirtualForce/ should equal to {sims.dimension}")
                def get_virtual_force(x): return virtual_force_field
                virtual_force = ti.func(get_virtual_force)
            elif isinstance(virtual_force_field, function):
                virtual_force_field = ti.pyfunc(virtual_force_field)
                def get_virtual_force(x): return virtual_force_field(x) if region_function(x) else zeros
                virtual_force = ti.func(get_virtual_force)

        self.particle_traction.input(virtual_stress, virtual_force)
        self.particle_traction.set_lists(element.cellSum, element.gridSum, element.grid_level)
        print("Boundary Type: Virtual Stress Field")
        print("Virtual Stress Check: ", self.particle_traction.stress_check(sims.dimension))
        print("Virtual Force Check: ", self.particle_traction.force_check(sims.dimension), '\n')

    def get_region_ptr(self, name):
        if not self.myRegion is None:
            return self.myRegion[name]
        else:
            raise RuntimeError("Region class should be activated first!")

    def iterate_boundary_constraint(self, sims, element, boundary_constraint, mode):
        if mode == 0:
            print(" Boundary Information ".center(71,"-"))
            if type(boundary_constraint) is dict:
                self.set_boundary_conditions(sims, element, boundary_constraint)
            elif type(boundary_constraint) is list:
                for boundary in boundary_constraint:
                    self.set_boundary_conditions(sims, element, boundary) 
        elif mode == 1:
            print('#', "Boundary Earse".center(67, "="), '-')
            if type(boundary_constraint) is dict:
                self.clear_boundary_constraint(sims, boundary_constraint)
            elif type(boundary_constraint) is list:
                for boundary in boundary_constraint:
                    self.clear_boundary_constraint(sims, boundary) 

    def set_boundary_conditions(self, sims: Simulation, element: ElementBase, boundary):
        self.new_boundaries = True
        boundary_type = DictIO.GetEssential(boundary, "BoundaryType")

        level = DictIO.GetAlternative(boundary, "NLevel", "All")
        start_point = DictIO.GetEssential(boundary, "StartPoint")
        end_point = DictIO.GetEssential(boundary, "EndPoint")
        self.check_boundary_domain(sims, start_point, end_point)
        inodes = element.get_boundary_nodes(start_point, end_point)
        level, nlevel = self.check_nlevel(level)

        if sims.mode == "Normal":
            if sims.shape_function == "QuadBSpline" or sims.shape_function == "CubicBSpline":
                for nl in range(level, level + nlevel):
                    add_boundary_flags(nl, element.gridSum, inodes, element.boundary_flag)

        if boundary_type == "VelocityConstraint":
            self.set_velocity_constraints(sims, boundary, level, nlevel, start_point, end_point, inodes)
        elif boundary_type == "ReflectionConstraint":
            self.set_reflection_constraints(sims, boundary, level, nlevel, start_point, end_point, inodes)
        elif boundary_type == "FrictionConstraint":
            self.set_friction_constraints(sims, boundary, level, nlevel, start_point, end_point, inodes)
        elif boundary_type == "AbsorbingConstraint":
            self.set_absorbing_constraints(sims, boundary, level, nlevel, start_point, end_point, inodes)
        elif boundary_type == "TractionConstraint":
            self.set_traction_constraints(sims, boundary, level, nlevel, start_point, end_point, inodes)
        elif boundary_type == "DisplacementConstraint":
            self.set_displacement_constraints(sims, boundary, level, nlevel, start_point, end_point, inodes)
        
    def clear_boundary_constraint(self, sims: Simulation, element: ElementBase, boundary):
        boundary_type = DictIO.GetEssential(boundary, "BoundaryType")
        level = DictIO.GetAlternative(boundary, "NLevel", "All")
        start_point = DictIO.GetAlternative(boundary, "StartPoint", vec3f(0, 0, 0))
        end_point = DictIO.GetEssential(boundary, "EndPoint", sims.domain)
        inodes = element.get_boundary_nodes(start_point, end_point)
        print("Start Point: ", start_point)
        print("End Point: ", end_point, '\n')

        if boundary_type == "VelocityConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.velocity_list, self.velocity_boundary, inodes, i)
            copy_valid_constraint(self.velocity_list, self.velocity_boundary)
        elif boundary_type == "ReflectionConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.reflection_list, self.reflection_boundary, inodes, i)
            copy_valid_constraint(self.reflection_list, self.reflection_boundary)
        elif boundary_type == "FrictionConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.friction_list, self.friction_boundary, inodes, i)
            copy_valid_constraint(self.friction_list, self.friction_boundary)
        elif boundary_type == "AbsorbingConstraint":
            pass
        elif boundary_type == "TractionConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_constraint(self.traction_list, self.traction_boundary, inodes, i) 
            copy_valid_constraint(self.traction_list, self.traction_boundary)
        elif boundary_type == "DisplacementConstraint":
            level, nlevel = self.check_nlevel()
            for i in range(level, level + nlevel):
                clear_displacement_constraint(self.displacement_boundary, inodes, i) 
        elif boundary_type == "ParticleTractionConstraint":
            pass

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
        
    def check_particle_traction_constraint_num(self, sims: Simulation, constraint_num):
        if self.ptraction_list[0] + constraint_num > sims.nptraction:
            raise ValueError ("The number of particle traction constraints should be set as: ", self.ptraction_list[0] + constraint_num)
        
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
            
    def set_boundary_types(self, sims: Simulation, element: ElementBase):
        grid_level = self.grid_layer
        if self.new_boundaries:
            element.set_boundary_type(sims, grid_level)
        self.new_boundaries = False
        