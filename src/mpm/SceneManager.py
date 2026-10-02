import warnings

import numpy as np
import taichi as ti
from taichi.lang.impl import current_cfg

from src.mpm.BaseKernel import *
from src.mpm.Contact import *
from src.mpm.elements.StaggeredQuadrilateralElement import StaggeredQuadrilateralElement
from src.mpm.elements.StaggeredHexahedronElement import StaggeredHexahedronElement
from src.mpm.elements.HexahedronElement8Nodes import HexahedronElement8Nodes
from src.mpm.elements.AdaptiveHexahedronElement import AdaptiveHexahedronElement
from src.mpm.elements.AdaptiveQuadrilateralElement import AdaptiveQuadrilateralElement
from src.mpm.elements.QuadrilateralElement4Nodes import QuadrilateralElement4Nodes
from src.mpm.MaterialManager import MaterialHandle
from src.mpm.sparse_grid import BlockSparseGrid
from src.mpm.boundaries.BoundaryConstraint import BoundaryConstraints
from src.mpm.Simulation import Simulation
from src.mpm.structs import *
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.linalg import no_operation
from src.utils.DomainBoundary import DomainBoundary
from src.utils.ObjectIO import DictIO
from src.utils.GeometryFunction import distance_along_direction_to_surface
from src.utils.TypeDefination import vec2f, vec3f, vec4f, vec3u8, mat2x2, mat3x3


def _dtype_nbytes(dtype):
    if dtype is float:
        dtype = current_cfg().default_fp
    elif dtype is int:
        dtype = current_cfg().default_ip
    dtype_name = str(dtype).lower()
    if "64" in dtype_name:
        return 8
    if "16" in dtype_name:
        return 2
    if "8" in dtype_name:
        return 1
    return 4


def _member_nbytes(member):
    if hasattr(member, "dtype"):
        count = int(getattr(member, "n", 1)) * int(getattr(member, "m", 1))
        return count * _dtype_nbytes(member.dtype)
    return _dtype_nbytes(member)


def _struct_slot_nbytes(struct_type):
    return sum(_member_nbytes(member) for member in struct_type.members.values())


def _format_mib(nbytes):
    return f"{nbytes / (1024 ** 2):.3f} MiB"


class myScene(object):
    def __init__(self) -> None:
        self.domain_boundary = None
        self.contact = None
        self.mass_cut_off = Threshold
        self.volume_cut_off = Threshold
        self.grid_level = 1

        self.element_type = "R8N3D"
        self.boundary = None
        self.particle = None
        self.iparticle = None
        self.material = None
        self.element = None
        self.node = None
        self.sparse_grid = None
        self.node_slot_bytes = 0
        self.grid = []
        self.is_rigid = None
        self.pid = None
        self.grandparent = None
        self.parent = None
        self.child = None
        self.psize = np.array([], dtype=np.int32)

        self.particleNum = np.zeros(1, dtype=np.int32)
        self.couplingNum = np.zeros(1, dtype=np.int32)
        self.free_particleNum = np.zeros(1, dtype=np.int32)
        self.free_particle_list = None
        self.node_type = None
        self.cell_phi = None
        self.cell_volume = None
        self.cell_porosity = None
        self.cell_pressure = None
        self.cell_dpressure = None
        self.cell_rigid = None
        self.cell_free_surface = None
        self.RECTELETYPE = ["Q4N2D", "R8N3D", "Staggered"]
        self.TRIELETYPE = ["T3N2D", "T4N3D"]

    def activate_boundary(self, sims):
        self.boundary = BoundaryConstraints()
        self.boundary.activate_boundary_constraints(sims, self.element.element_type)

    def is_rectangle_cell(self):
        return self.element_type in self.RECTELETYPE

    def is_triangle_cell(self):
        return self.element_type in self.TRIELETYPE

    def find_particle_class(self, sims: Simulation):
        ptemp = None
        if sims.coupling:
            if sims.solver_type == "Explicit":
                ptemp = ParticleCoupling
            elif sims.solver_type == "Implicit":
                if sims.material_type == "Fluid" and sims.discretization == "FDM":
                    if sims.dimension == 2:
                        ptemp = ParticleCloudIncompressible2D
                    elif sims.dimension == 3:
                        ptemp = ParticleCloudIncompressible3D
                else:
                    ptemp = ImplicitParticleCoupling
            elif (
                sims.solver_type == "SemiImplicit" or sims.solver_type == "SemiImplicit_u_p"
            ) and sims.material_type in ("TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"):
                if sims.dimension == 2:
                    if not sims.is_2DAxisy:
                        ptemp = ParticleCloudTwoPhase2D
                    else:
                        ptemp = ParticleCloudTwoPhase2DAxisy
                elif sims.dimension == 3:
                    ptemp = ParticleCloudTwoPhase
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
                    if sims.solver_type == "G2P2G":
                        ptemp = LargeScaleParticle
                    elif sims.material_type == "TwoPhaseSingleLayer":
                        ptemp = ParticleCloudTwoPhase
                    else:
                        ptemp = ParticleCloud
            elif sims.solver_type == "Implicit":
                if self.element_type == "Staggered":
                    if sims.dimension == 2:
                        ptemp = ParticleCloudIncompressible2D  # StaggeredPartilce2D
                    if sims.dimension == 3:
                        ptemp = ParticleCloudIncompressible3D  # StaggeredPartilce
                else:
                    if sims.dimension == 2:
                        ptemp = ImplicitParticle2D
                    elif sims.dimension == 3:
                        ptemp = ImplicitParticle
            elif sims.solver_type == "SemiImplicit" or sims.solver_type == "SemiImplicit_u_p":
                if sims.dimension == 2:
                    if sims.material_type == "TwoPhaseSingleLayer" or sims.material_type == "TwoPhaseDoubleLayer":
                        if not sims.is_2DAxisy:
                            ptemp = ParticleCloudTwoPhase2D
                        elif sims.material_type == "TwoPhaseSingleLayer":
                            ptemp = ParticleCloudTwoPhase2DAxisy
                elif sims.dimension == 3:
                    if sims.material_type == "TwoPhaseSingleLayer" or sims.material_type == "TwoPhaseDoubleLayer":
                        ptemp = ParticleCloudTwoPhase

            if sims.contact_detection == "DEMContact":
                ptemp.members.update({"contact_traction": ti.types.vector(sims.dimension, float)})
        if ptemp is None:
            raise RuntimeError("Wrong particle type!")
        return ptemp

    def activate_particle(self, sims: Simulation):
        self.check_materials(sims)
        if self.particle is None and sims.max_particle_num > 0:
            ptemp = self.find_particle_class(sims)
            """ if sims.solver_type == "Explicit":
                if sims.stabilize == 'F-Bar Method':
                    ptemp.members.update({"jacobian": float}) """
            if sims.configuration == "TLMPM":
                ptemp.members.update({"stress": mat2x2 if sims.dimension == 2 else mat3x3})
            if sims.particle_shifting is True:
                ptemp.members.update({"grad_E2": ti.types.vector(sims.dimension, float)})
            if (
                sims.neighbor_detection is True
                or sims.free_surface_detection is True
                or sims.boundary_direction_detection is True
            ):
                ptemp.members.update(
                    {
                        "free_surface": ti.u8,
                        "mass_density": float,
                        "normal": vec3f,
                        "rad": float,
                        "coupling": ti.u8,
                    }
                )
            self.particle = ptemp.field()
            ti.root.dense(ti.i, sims.max_particle_num).place(self.particle)
            """ if sims.solver_type == "Explicit":
                if sims.stabilize == 'F-Bar Method':
                    kernel_initialize_particle_fbar(self.particle) """
            self.material.activate_state_variables(sims)

    def activate_material(self, sims: Simulation, material_model, parameters):
        if self.material is None:
            self.material = MaterialHandle(sims)
        self.material.setup(sims, self.contact, material_model, parameters)

    def check_materials(self, sims):
        if self.material is None:
            self.activate_material(sims, "RigidBody", parameters=None)

    def activate_contact(self, sims: Simulation, contact_phys):
        if sims.contact_detection:
            if sims.contact_detection == "MPMContact":
                self.contact = MPMContact(contact_phys)
            elif sims.contact_detection == "GeoContact":
                self.contact = GeoContact(contact_phys)
            elif sims.contact_detection == "DEMContact":
                self.contact = DEMContact(contact_phys)
                if not self.material is None:
                    self.material.setup_contact(self.contact)
        self.print_contact_message(sims)

    def print_contact_message(self, sims: Simulation):
        if sims.contact_detection:
            print("Contact Detection Activated: ", sims.contact_detection)
            self.contact.print_contact_message()

    def configure_adaptive_grid_for_solver(self, sims: Simulation, adaptive_grid):
        if not adaptive_grid or not isinstance(adaptive_grid, dict):
            return adaptive_grid

        adaptive_grid = dict(adaptive_grid)
        if sims.solver_type != "Implicit" or sims.material_type != "Solid":
            return adaptive_grid

        mode = str(DictIO.GetAlternative(adaptive_grid, "HangingConstraintMode", "Auto"))
        normalized_mode = mode.lower().replace("_", "").replace("-", "").replace(" ", "")
        bridging_domain = bool(DictIO.GetAlternative(adaptive_grid, "BridgingDomain", False))
        if bridging_domain:
            raise RuntimeError(
                "Implicit solid AdaptiveGrid does not support BridgingDomain=True "
                "because hanging penalty stiffness is not assembled. Disable "
                "BridgingDomain or use explicit MPM."
            )

        if normalized_mode == "auto":
            adaptive_grid["HangingConstraintMode"] = "ShapeFunction"
            if sims.dimension == 3:
                warnings.warn(
                    "Implicit 3D AdaptiveGrid uses HangingConstraintMode=ShapeFunction. "
                    "This expands the dense local stiffness support and can increase "
                    "compile time and memory use.",
                    RuntimeWarning,
                )
        elif normalized_mode in ("shape", "shapefunction"):
            adaptive_grid["HangingConstraintMode"] = "ShapeFunction"
        elif normalized_mode == "penalty":
            raise RuntimeError(
                "Implicit solid AdaptiveGrid requires "
                "HangingConstraintMode=ShapeFunction. Penalty mode is explicit-only "
                "until hanging penalty stiffness is assembled."
            )
        else:
            raise ValueError("AdaptiveGrid/HangingConstraintMode must be Auto, Penalty, " "Shape, or ShapeFunction")

        if adaptive_grid.get("HangingConstraintMode") == "ShapeFunction":
            minimum_capacity_factor = 4 if sims.dimension == 2 else 8
            capacity_factor = int(
                DictIO.GetAlternative(adaptive_grid, "HangingShapeCapacityFactor", minimum_capacity_factor)
            )
            if capacity_factor < minimum_capacity_factor:
                warnings.warn(
                    "AdaptiveGrid/HangingShapeCapacityFactor is too small for "
                    f"{sims.dimension}D ShapeFunction hanging constraints; using "
                    f"{minimum_capacity_factor}.",
                    RuntimeWarning,
                )
                capacity_factor = minimum_capacity_factor
            adaptive_grid["HangingShapeCapacityFactor"] = capacity_factor

        return adaptive_grid

    def activate_element(self, sims: Simulation, element):
        self.element_type = DictIO.GetAlternative(element, "ElementType", "R8N3D")
        if sims.max_particle_num > 0:
            grid_level = self.find_grid_level(sims)

            if sims.dimension == 2:
                if self.element_type == "T3N2D":
                    raise ValueError("The triangle mesh is not supported currently")
                elif self.element_type == "Q4N2D" or self.element_type == "Staggered":
                    if not self.element is None:
                        print("Warning: Previous elements will be override!")
                    adaptive_grid = self.configure_adaptive_grid_for_solver(
                        sims, DictIO.GetAlternative(element, "AdaptiveGrid", None)
                    )
                    if adaptive_grid:
                        if self.element_type != "Q4N2D":
                            raise RuntimeError("AdaptiveGrid currently supports Q4N2D only")
                        self.element = AdaptiveQuadrilateralElement(
                            self.element_type,
                            grid_level,
                            DictIO.GetAlternative(element, "GhostCell", 1),
                            adaptive_grid,
                        )
                    else:
                        self.element = QuadrilateralElement4Nodes(
                            self.element_type,
                            grid_level,
                            DictIO.GetAlternative(element, "GhostCell", 1),
                        )
                else:
                    raise ValueError("Keyword:: /ElementType/ error!")
                grid_size = vec2f(DictIO.GetEssential(element, "ElementSize"))
                if sims.isTHB:
                    if self.element_type != "Q4N2D":
                        raise RuntimeError("THB MPM currently supports Q4N2D only")
                    self.element.create_nodes_THB(sims, grid_size)
                else:
                    self.element.create_nodes(sims, grid_size)
                self.initialize_element(sims, grid_level)
            elif sims.dimension == 3:
                if self.element_type == "T4N3D":
                    raise ValueError("The triangle mesh is not supported currently")
                elif self.element_type == "R8N3D" or self.element_type == "Staggered":
                    if not self.element is None:
                        print("Warning: Previous elements will be override!")
                    adaptive_grid = self.configure_adaptive_grid_for_solver(
                        sims, DictIO.GetAlternative(element, "AdaptiveGrid", None)
                    )
                    if adaptive_grid:
                        if self.element_type != "R8N3D":
                            raise RuntimeError("AdaptiveGrid currently supports R8N3D only")
                        self.element = AdaptiveHexahedronElement(
                            self.element_type,
                            grid_level,
                            DictIO.GetAlternative(element, "GhostCell", 1),
                            adaptive_grid,
                        )
                    else:
                        self.element = HexahedronElement8Nodes(
                            self.element_type, grid_level, DictIO.GetAlternative(element, "GhostCell", 1)
                        )
                else:
                    raise ValueError("Keyword:: /ElementType/ error!")
                self.element.create_nodes(sims, vec3f(DictIO.GetEssential(element, "ElementSize")))
                self.initialize_element(sims, grid_level)

            if self.boundary is None:
                self.activate_boundary(sims)
            self.boundary.set_layer_number(grid_level)
            self.activate_grid(sims, grid_level)

    def initialize_element(self, sims: Simulation, grid_level):
        if sims.gauss_number > 0:
            self.element.element_initialize(sims, local_coordiates=False)
            self.element.activate_gauss_cell(sims)
        else:
            self.element.element_initialize(sims)
        self.element.activate_euler_cell(sims)

    def find_grid_level(self, sims: Simulation):
        grid_level = 1
        if sims.contact_detection:
            if sims.contact_detection == "MPMContact":
                grid_level = max(2, sims.max_body_num)
            elif sims.contact_detection == "GeoContact":
                grid_level = max(2, sims.max_body_num)
            elif sims.contact_detection == "DEMContact":
                grid_level = 1
        self.grid_level = grid_level
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
            elif (
                sims.solver_type == "SemiImplicit" or sims.solver_type == "SemiImplicit_u_p"
            ) and sims.material_type == "TwoPhaseSingleLayer":
                if sims.dimension == 2:
                    gtemp = NodeTwoPhase2D
                elif sims.dimension == 3:
                    gtemp = NodeTwoPhase
            else:
                raise RuntimeError("The contact is not supported in implicit MPM currently")
        else:
            if sims.solver_type == "Explicit":
                if sims.dimension == 2:
                    if sims.material_type == "Solid" or sims.material_type == "Fluid":
                        gtemp = Nodes2D
                    elif sims.material_type == "TwoPhaseSingleLayer":
                        gtemp = NodeTwoPhase2D
                    elif sims.material_type == "TwoPhaseDoubleLayer":
                        raise RuntimeError()
                elif sims.dimension == 3:
                    if sims.material_type == "Solid" or sims.material_type == "Fluid":
                        gtemp = Nodes
                    elif sims.material_type == "TwoPhaseSingleLayer":
                        gtemp = NodeTwoPhase
                    else:
                        raise RuntimeError()
            elif sims.solver_type == "Implicit":
                if sims.material_type == "Solid":
                    if sims.dimension == 2:
                        gtemp = ImplicitNodes2D
                    elif sims.dimension == 3:
                        gtemp = ImplicitNodes
                if sims.material_type == "Fluid":
                    if sims.dimension == 2:
                        gtemp = IncompressibleNodes2D
                    elif sims.dimension == 3:
                        gtemp = IncompressibleNodes3D
            elif sims.solver_type == "SemiImplicit" or sims.solver_type == "SemiImplicit_u_p":
                if sims.material_type == "TwoPhaseSingleLayer" or sims.material_type == "TwoPhaseDoubleLayer":
                    if sims.dimension == 2:
                        gtemp = NodeTwoPhase2D
                    elif sims.dimension == 3:
                        gtemp = NodeTwoPhase
        if gtemp is None:
            raise RuntimeError("Wrong background node type!")
        return gtemp

    def activate_grid(self, sims: Simulation, grid_level):
        self.check_grid_inputs(sims, grid_level)
        self.is_rigid = ti.field(int, shape=grid_level)
        sims.set_body_num(grid_level)
        self.element.set_characteristic_length(sims)

        cut_off = 0.0
        if not self.node is None:
            print("Warning: Previous node will be override!")
        self.sparse_grid = None

        if self.element_type == "Staggered":
            if ("Implicit" in sims.solver_type) and (
                sims.material_type == "Fluid" or sims.material_type == "TwoPhaseDoubleLayer"
            ):
                self.node = StaggeredGrid(sims, self.element.cnum, self.element.ghost_cell)
                self.node.grid_reset(self.mass_cut_off)
                if sims.material_type == "TwoPhaseDoubleLayer":
                    pass
        else:
            gtemp = self.find_grid_class(sims)
            if sims.sparse_grid and not getattr(self.element, "adaptive", False):
                self.sparse_grid = BlockSparseGrid(sims, self.element, grid_level)
                self.child = ti.root.dense(ti.ij, (self.sparse_grid.node_capacity, grid_level))
                self.parent = self.child
            else:
                if sims.AOSOA:
                    self.parent = ti.root.dense(
                        ti.ij, (int(np.ceil(self.element.gridSum / sims.block_size[0])), grid_level)
                    )
                    temp_tree = self.parent
                    for i in range(1, len(sims.block_size)):
                        temp_tree = temp_tree.dense(ti.i, int(sims.block_size[i - 1] // sims.block_size[i]))
                    self.parent = self.grandparent.dense(ti.i, int(sims.block_size[0] // sims.block_size[1]))
                    self.child = self.parent.dense(ti.i, int(sims.block_size[1]))
                else:
                    self.child = ti.root.dense(ti.ij, (self.element.gridSum, grid_level))

            if sims.stabilize == "F-Bar Method":
                gtemp.members.update({"jacobian": float})
            if sims.stabilize == "Displacement F-Bar Method":
                gtemp.members.update({"vol": float})
                gtemp.members.update({"jacobian": float, "pressure": float})
            if sims.pressure_smoothing == True:
                if "pressure" not in gtemp.members.keys():
                    gtemp.members.update({"pressure": float})
            if sims.particle_shifting is True:
                gtemp.members.update({"vol": float})
            self.node_slot_bytes = _struct_slot_nbytes(gtemp)
            self.node = gtemp.field()
            self.child.place(self.node)
            self.node.fill(0)

            if (
                (sims.solver_type == "SemiImplicit" or sims.solver_type == "SemiImplicit_u_p")
                and sims.material_type == "TwoPhaseSingleLayer"
                and sims.use_mgpcg_pressure_solver()
            ):
                indice = ti.ij if sims.dimension == 2 else ti.ijk
                self.node_type = ti.field(dtype=ti.u8)
                self.cell_phi = ti.field(dtype=float)
                self.cell_volume = ti.field(dtype=float)
                self.cell_porosity = ti.field(dtype=float)
                self.cell_pressure = ti.field(dtype=float)
                self.cell_dpressure = ti.field(dtype=float)
                self.cell_rigid = ti.field(dtype=ti.u8)
                self.cell_free_surface = ti.field(dtype=ti.u8)
                ti.root.dense(indice, self.element.cnum).place(
                    self.node_type,
                    self.cell_phi,
                    self.cell_volume,
                    self.cell_porosity,
                    self.cell_pressure,
                    self.cell_dpressure,
                    self.cell_free_surface,
                    self.cell_rigid,
                )
                self.free_particle_list = ti.field(dtype=int)
                ti.root.dense(ti.i, int(sims.max_particle_num)).place(self.free_particle_list)

            if sims.mapping == "G2P2G":
                self.grandparent = [self.grandparent]
                self.parent = [self.parent]
                self.child = [self.child]
                output_grid = self.find_grid_class(sims, grid_level)
                if sims.sparse_grid:
                    raise RuntimeError("BlockScan sparse_grid does not support mapping='G2P2G' yet.")
                else:
                    self.child = ti.root.dense(ti.ij, (self.element.gridSum, grid_level))
                self.grid = [self.node, self.child[1].place(output_grid)]
        self.element.calculate_basis_function(sims, grid_level)
        self.print_grid_message(sims, grid_level, cut_off)

    def check_grid_inputs(self, sims: Simulation, grid_level):
        # if grid_level > 2:
        #     raise ValueError("The mpm only support two body contact detection")
        if grid_level == 2:
            if sims.contact_detection is False:
                warnings.warn("The contact detection has not been activeted yet!")
        if grid_level == 1 and sims.contact_detection is True:
            raise RuntimeError("The contact detection should be turned off!")

    def print_grid_message(self, sims: Simulation, grid_level, cut_off=0.0):
        print(" Grid Information ".center(71, "-"))
        print("Grid Type: Rectangle")
        if grid_level > 0:
            print("The number of grids = ", grid_level)
        self.element.print_message()
        if sims.sparse_grid and self.sparse_grid is not None:
            sparse_info = self.sparse_grid.describe(self.node_slot_bytes)
            print("Sparse Grid Backend = ", sparse_info["backend"])
            print("Sparse Block Size = ", sparse_info["block_size"])
            print("Sparse Block Count = ", sparse_info["block_count"])
            print("Sparse Max Active Blocks = ", sparse_info["max_active_blocks"])
            print("Sparse Node Capacity = ", sparse_info["node_capacity"])
            print("Sparse Allocated/Dense Node Slot Ratio = ", sparse_info["allocated_node_slot_ratio"])
            if self.node_slot_bytes > 0:
                print("Node Slot Bytes = ", sparse_info["node_slot_bytes"])
                print("Dense Node Memory Estimate = ", _format_mib(sparse_info["dense_node_bytes"]))
                print("Sparse Allocated Memory Estimate = ", _format_mib(sparse_info["allocated_sparse_bytes"]))
                print("Sparse Metadata Memory Estimate = ", _format_mib(sparse_info["metadata_bytes"]))
                print("Sparse/Dense Memory Estimate Ratio = ", sparse_info["allocated_sparse_byte_ratio"])
        elif sims.sparse_grid and getattr(self.element, "adaptive", False):
            print("Sparse Grid Backend =  AdaptiveNodeMap")
            print("Sparse Node Capacity = ", self.element.gridSum)
            print("Sparse Grid Note = AdaptiveGrid already stores compact node ids; BlockScan remap is disabled.")
        print("\n")

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
            raise ValueError("The MPM particles should be set as: ", self.particleNum[0] + particle_number)

    def find_min_z_position(self):
        return find_min_z_position_(int(self.particleNum[0]), self.particle)

    def find_min_y_position(self):
        return find_min_y_position_(int(self.particleNum[0]), self.particle)

    def find_bounding_sphere_radius(self):
        rad_max = self.find_particle_max_radius()
        rad_min = self.find_particle_min_radius()
        return rad_max, rad_min

    def _particle_size_radius_bounds(self):
        if self.psize.size == 0:
            return 0.0, 0.0
        psize = np.asarray(self.psize, dtype=np.float64)
        radius = np.abs(psize) if psize.ndim == 1 else np.linalg.norm(psize, axis=1)
        radius = radius[np.isfinite(radius) & (radius > 0.0)]
        if radius.size == 0:
            return 0.0, 0.0
        return float(np.min(radius)), float(np.max(radius))

    def ensure_neighbor_particle_radius(self):
        particle_num = int(self.particleNum[0])
        if particle_num <= 0 or self.psize.size == 0:
            return
        psize = np.asarray(self.psize[:particle_num], dtype=np.float64)
        if psize.ndim == 1:
            radius = np.abs(psize)
        else:
            radius = np.linalg.norm(psize, axis=1)
        radius = np.maximum(radius, 1.0e-12)
        set_particle_radius_from_array_(
            particle_num,
            self.particle,
            np.ascontiguousarray(radius, dtype=np.float64),
        )

    def find_particle_min_radius(self):
        if not hasattr(self.particle, "rad"):
            return self._particle_size_radius_bounds()[0]
        return find_particle_min_radius_(int(self.particleNum[0]), self.particle)

    def find_particle_max_radius(self):
        if not hasattr(self.particle, "rad"):
            return self._particle_size_radius_bounds()[1]
        return find_particle_max_radius_(int(self.particleNum[0]), self.particle)

    def find_particle_min_mass(self):
        return find_particle_min_mass_(int(self.particleNum[0]), self.particle)

    def reset_verlet_disp(self):
        reset_verlet_disp_(int(self.couplingNum[0]), self.particle)

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
        density = 0.0
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
            if sims.isTHB == False:
                self.mass_cut_off = 1e-8 * density / matcount * grid_size[0] * grid_size[1]
                self.volume_cut_off = 1e-5 * grid_size[0] * grid_size[1]
            else:
                l = 2**sims.grid_layer
                self.mass_cut_off = 1e-8 * density / matcount * grid_size[0] / l * grid_size[1] / l
                self.volume_cut_off = 1e-8 * grid_size[0] / l * grid_size[1] / l

    def update_particle_properties_in_region(self, sims: Simulation, override, property_name, value, is_in_region):
        print(" Modify Particle Information ".center(71, "-"))
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, "\n")

        factor = 1 if not override else 0
        if property_name == "bodyID":
            modify_particle_bodyID_in_region(value, int(self.particleNum[0]), self.particle, is_in_region)
        elif property_name == "materialID":
            modify_particle_materialID_in_region(
                value, int(self.particleNum[0]), self.particle, self.material.matProps, is_in_region
            )
        elif property_name == "position":
            if sims.dimension == 3:
                modify_particle_position_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif sims.dimension == 2:
                modify_particle_position_in_region_2D(
                    factor, value, int(self.particleNum[0]), self.particle, is_in_region
                )
        elif property_name == "velocity":
            if sims.dimension == 3:
                modify_particle_velocity_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
            elif sims.dimension == 2:
                modify_particle_velocity_in_region_2D(
                    factor, value, int(self.particleNum[0]), self.particle, is_in_region
                )
        elif property_name == "stress":
            modify_particle_stress_in_region(factor, value, int(self.particleNum[0]), self.particle, is_in_region)
        elif property_name == "fix_velocity":
            FIX = {"Free": 0, "Fix": 1}
            fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
            modify_particle_fix_v_in_region(fix_v, int(self.particleNum[0]), self.particle, is_in_region)
        else:
            valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
            raise KeyError(
                f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}"
            )

    def update_particle_properties(self, sims: Simulation, override, property_name, value, bodyID):
        print(" Modify Body Information ".center(71, "-"))
        print("Target BodyID =", bodyID)
        print("Target Property =", property_name)
        print("Target Value =", value)
        print("Override =", override, "\n")

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
            FIX = {"Free": 0, "Fix": 1}
            fix_v = vec3u8([DictIO.GetEssential(FIX, is_fix) for is_fix in value])
            modify_particle_fix_v(fix_v, int(self.particleNum[0]), self.particle, bodyID)
        else:
            valid_list = ["bodyID", "materialID", "position", "velocity", "traction", "stress", "fix_velocity"]
            raise KeyError(
                f"Invalid property_name: {property_name}! Only the following keywords is valid: {valid_list}"
            )

    def compact_particle_storage(self):
        particle_num = int(self.particleNum[0])
        active = field_to_numpy_prefix(self.particle.active, particle_num).astype(bool)
        if self.psize.shape[0] >= particle_num:
            self.psize = np.ascontiguousarray(self.psize[:particle_num][active])
        remaining = update_particle_storage_(
            particle_num,
            self.particle,
            self.material.stateVars,
            self.material.stateVars is not None,
        )
        self.particleNum[0] = remaining
        if remaining > 0:
            self.material.update_material_mapping(self.particle, remaining)
        else:
            self.material.materialID_numpy = np.empty(0, dtype=np.int32)
        return particle_num - remaining

    def check_overlap_coupling(self):
        deleted = self.compact_particle_storage()
        print(f"Total {deleted} particles has been deleted", "\n")

    def delete_particles(self, bodyID):
        kernel_delete_particles(self.particleNum[0], self.particle, bodyID)
        print(f"Total {self.compact_particle_storage()} particles has been deleted", "\n")

    def delete_particles_in_region(self, is_in_region):
        kernel_delete_particles_in_region(self.particleNum[0], self.particle, is_in_region)
        print(f"Total {self.compact_particle_storage()} particles has been deleted", "\n")

    def delete_overlap_particles(self, position, volume):
        kernel_delete_overlap_particles(self.particleNum[0], self.particle, position, volume)
        print(f"Total {self.compact_particle_storage()} particles has been deleted", "\n")

    def check_in_domain(self, sims: Simulation):
        if sims.dimension == 2:
            check_in_domain_2D(sims.domain, int(self.particleNum[0]), self.particle)
        else:
            check_in_domain(sims.domain, int(self.particleNum[0]), self.particle)

    def check_in_domain_2D(self, sims: Simulation):
        check_in_domain_2D(sims.domain, int(self.particleNum[0]), self.particle)

    def get_mass_center(self, bodyID=0):
        return kernel_compute_mass_center(int(self.particleNum[0]), self.particle, bodyID)

    def choose_coupling_region(self, sims: Simulation, function):
        if sims.coupling:
            kernel_update_coupling_material_points(int(self.particleNum[0]), self.particle, function)

    def update_coupling_points_number(self, sims: Simulation):
        need_coupling = kernel_compute_coupling_material_points_number(int(self.particleNum[0]), self.particle)
        if sims.material_type == "TwoPhaseDoubleLayer" and need_coupling > 0:
            coupling = field_to_numpy_prefix(self.particle.coupling, int(self.particleNum[0]))
            phase = field_to_numpy_prefix(self.particle.phase, int(self.particleNum[0]))
            if np.any(phase[coupling == 1] != 1):
                raise RuntimeError("TwoPhaseDoubleLayer ordinary DEM contact may include solid points only")
            if np.any(coupling[:need_coupling] != 1) or np.any(coupling[need_coupling:] != 0):
                raise RuntimeError(
                    "TwoPhaseDoubleLayer DEM contact points must be a leading particle prefix; "
                    "add the solid-phase body templates before the fluid-phase templates"
                )
        self.couplingNum[0] = need_coupling
        sims.set_coupling_particles(need_coupling)

    def filter_particles(self, sims: Simulation):
        if sims.coupling:
            kernel_tranverse_coupling_particle(int(self.particleNum[0]), self.particle)
        else:
            kernel_tranverse_active_particle(int(self.particleNum[0]), self.particle)

    def check_elastic_material(self):
        is_elastic = True
        for nmat in range(1, self.material.matProps.size()):
            is_elastic &= self.material.matProps[nmat].is_elastic
        return is_elastic

    def push_psize(self, psize):
        psize = list(psize)
        dim = len(psize)
        if np.array(psize).ndim == 2:
            dim = len(psize[0])
        self.psize = np.append(self.psize, psize).reshape(-1, dim)

    def set_boundary_condition(self, sims: Simulation):
        self.domain_boundary = DomainBoundary(sims.domain)
        self.domain_boundary.set_boundary_condition(sims.boundary)
        if self.domain_boundary.need_run:
            self.apply_boundary_conditions = self.apply_boundary_condition
        else:
            self.apply_boundary_conditions = no_operation

    def apply_boundary_condition(self):
        if self.domain_boundary.apply_boundary_conditions(int(self.particleNum[0]), self.particle):
            self.compact_particle_storage()

    def set_initial_gravity_field(self, sims: Simulation, gravity_field):
        if gravity_field is not None and gravity_field is not False:
            import types

            gravity = np.array(sims.gravity)
            if np.linalg.norm(gravity) < 1e-12:
                raise ValueError("Gravity must be activated")

            non_rigid_particles = np.where(self.material.materialID_numpy > 0)[0]
            directions = (gravity / np.linalg.norm(gravity))[0 : sims.dimension]
            point_cloud = np.ascontiguousarray(
                field_to_numpy_prefix(self.particle.x, self.particleNum[0])[non_rigid_particles]
            )
            distances = np.full(point_cloud.shape[0], -1.0)

            get_dist = True
            if isinstance(gravity_field, types.FunctionType):
                distances = gravity_field(point_cloud)
            elif isinstance(gravity_field, (np.ndarray, list, tuple)):
                gravity_field = np.asarray(gravity_field)
                if gravity_field.ndim == 2 and gravity_field.shape[1] == 6:
                    if gravity_field.shape[0] != int(self.particleNum[0]):
                        raise ValueError("Gravity field in Voigt format must have shape (particleNum, 6)")
                    gravity_field = np.ascontiguousarray(gravity_field, dtype=np.float64)
                    get_dist = False
                else:
                    if gravity_field.shape[0] != point_cloud.shape[0]:
                        raise ValueError("Gravity distance field must match the number of non-rigid particles")
                    distances = np.ascontiguousarray(gravity_field.copy(), dtype=np.float64)
            elif isinstance(gravity_field, int) or isinstance(gravity_field, bool):
                if gravity_field is True or gravity_field == 0:
                    projections = np.sum(point_cloud * -directions, axis=1)
                    t_max = np.max(projections)
                    distances = t_max - projections

                    denominator = (directions[0] / self.psize[non_rigid_particles, 0]) ** 2 + (
                        directions[1] / self.psize[non_rigid_particles, 1]
                    ) ** 2
                    if sims.dimension == 3:
                        denominator += (directions[2] / self.psize[non_rigid_particles, 2]) ** 2
                    distances += np.sqrt(1.0 / denominator)
                elif gravity_field == 1:
                    import open3d as o3d
                    import trimesh as tm

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_cloud)

                    dists = pcd.compute_nearest_neighbor_distance()
                    mean_dist = np.mean(dists)

                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4 * mean_dist, max_nn=50)
                    )
                    pcd.orient_normals_consistent_tangent_plane(k=20)

                    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=10, width=0, scale=1.1, linear_fit=False
                    )

                    bbox = pcd.get_axis_aligned_bounding_box()
                    mesh_o3d = mesh_o3d.crop(bbox)

                    mesh_o3d.remove_duplicated_vertices()
                    mesh_o3d.remove_duplicated_triangles()
                    mesh_o3d.remove_degenerate_triangles()
                    mesh_o3d.remove_non_manifold_edges()

                    vertices_tmp = np.asarray(mesh_o3d.vertices)
                    faces_tmp = np.asarray(mesh_o3d.triangles)
                    tmesh = tm.Trimesh(vertices=vertices_tmp, faces=faces_tmp, process=False)

                    tmesh.nondegenerate_faces()
                    tmesh.remove_unreferenced_vertices()
                    tmesh.remove_infinite_values()
                    tmesh.fill_holes()

                    mesh_o3d = o3d.geometry.TriangleMesh()
                    mesh_o3d.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
                    mesh_o3d.triangles = o3d.utility.Vector3iVector(tmesh.faces)
                    mesh_o3d.compute_vertex_normals()

                    # o3d.visualization.draw_geometries([mesh_o3d])

                    vertices = np.asarray(mesh_o3d.vertices)
                    faces = np.asarray(mesh_o3d.triangles)

                    mesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
                    mesh.show()

                    distances = distance_along_direction_to_surface(mesh, point_cloud, -directions, max_distance=np.inf)
            if get_dist:
                for materialID in range(self.material.mapping.shape[0] - 1):
                    start_index = self.material.mapping[materialID]
                    end_index = self.material.mapping[materialID + 1]
                    k0 = self.material.matProps[materialID + 1].get_lateral_coefficient(
                        start_index, end_index, self.material.materialID, self.material.stateVars
                    )
                    kernel_apply_gravity_field_(
                        start_index,
                        end_index,
                        sims.gravity,
                        k0,
                        distances,
                        self.particle,
                        self.material.materialID,
                        self.material.matProps[materialID + 1],
                        self.material.stateVars,
                    )
            else:
                kernel_setting_gravity_field_(int(self.particleNum[0]), self.particle, gravity_field)

    def adaptive_timestep(self, sims: Simulation):
        grid_size = self.element.grid_size
        if getattr(self.element, "adaptive", False):
            grid_size = self.element.fine_grid_size
        return kernel_adaptive_timestep(int(self.particleNum[0]), min(grid_size), self.particle)
