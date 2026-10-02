import numpy as np
import taichi as ti
from itertools import product

from src.mpm.Simulation import Simulation
from src.mpm.elements.AdaptiveNodeMap import (
    AdaptiveNodeMap,
    build_bridging_node_weights,
    build_hanging_constraint_table,
)
from src.mpm.elements.AdaptiveHexahedronKernel import (
    adaptive_bridging_global_update,
    adaptive_global_update,
    apply_hanging_penalty_impulse_list,
    apply_hanging_penalty_impulse,
    apply_hanging_penalty_velocity_list,
    apply_hanging_penalty_velocity,
    assemble_hanging_penalty_impulse_list,
    accumulate_split_cell_volume,
    cache_refined_particle_parents,
    count_new_refined_cells,
    collect_particles_to_split,
    copy_refined_particle_children,
    count_refined_cells,
    dilate_refined_cells,
    initialize_adaptive_particle_size,
    load_hanging_constraint_table,
    mark_refined_cells_migrated_particles,
    mark_refined_cells_matrix_stress,
    mark_refined_cells_deviatoric_strain,
    mark_refined_cells_epdstrain,
    mark_refined_cells_epstrain,
    mark_refined_cells_softening,
    mark_refined_cells_stress,
    mark_refined_cells_strain,
    merge_refined_cells,
    assemble_bridging_penalty_impulse,
    split_refined_particles,
)
from src.mpm.elements.HexahedronElement8Nodes import HexahedronElement8Nodes
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefination import vec3f, vec3i


class AdaptiveHexahedronElement(HexahedronElement8Nodes):
    """Two-level 2:1 adaptive background grid for 3-D MPM."""

    def __init__(self, element_type, grid_level, ghost_cell, adaptive_grid) -> None:
        super().__init__(element_type, grid_level, ghost_cell)
        self.adaptive = True
        self.refinement_ratio = 2
        self.max_level = int(
            DictIO.GetAlternative(
                adaptive_grid,
                "MaxLevel",
                DictIO.GetAlternative(adaptive_grid, "MaxRefinementLevel", 1),
            )
        )
        self.finest_ratio = self.refinement_ratio**self.max_level
        self.refine_interval = int(DictIO.GetAlternative(adaptive_grid, "RefineInterval", 50))
        self.particle_split_interval = int(
            DictIO.GetAlternative(adaptive_grid, "ParticleSplitInterval", self.refine_interval)
        )
        self.refine_threshold = float(DictIO.GetAlternative(adaptive_grid, "RefineThreshold", 2e-3))
        self.refine_criterion = str(DictIO.GetAlternative(adaptive_grid, "RefineCriterion", "Auto")).strip()
        self.max_refined_ratio = float(DictIO.GetAlternative(adaptive_grid, "RefineRatio", 0.2))
        self.refine_particles = bool(DictIO.GetAlternative(adaptive_grid, "RefineParticles", True))
        self.follow_refined_particles = bool(DictIO.GetAlternative(adaptive_grid, "FollowRefinedParticles", True))
        self.follow_refined_particles_interval = int(
            DictIO.GetAlternative(
                adaptive_grid,
                "FollowRefinedParticlesInterval",
                min(self.refine_interval, 20),
            )
        )
        self.particle_split_batch = int(DictIO.GetAlternative(adaptive_grid, "ParticleSplitBatch", 8192))
        self.split_interior_only = bool(DictIO.GetAlternative(adaptive_grid, "SplitInteriorOnly", False))
        self.split_interior_volume_fraction = float(
            DictIO.GetAlternative(adaptive_grid, "SplitInteriorVolumeFraction", 0.95)
        )
        self.buffer_cells = int(DictIO.GetAlternative(adaptive_grid, "BufferCells", 1))
        self.penalty = float(DictIO.GetAlternative(adaptive_grid, "Penalty", 1.0))
        self.penalty_iterations = int(DictIO.GetAlternative(adaptive_grid, "PenaltyIterations", 1))
        self.penalty_beta = float(DictIO.GetAlternative(adaptive_grid, "PenaltyBeta", 10.0))
        self.penalty_young = 0.0
        self.penalty_length = 0.0
        self.penalty_reference_volume = 0.0
        self.bridging_domain = bool(DictIO.GetAlternative(adaptive_grid, "BridgingDomain", False))
        self.bridging_cells = int(DictIO.GetAlternative(adaptive_grid, "BridgingCells", 1))
        self.hanging_constraint_mode = str(DictIO.GetAlternative(adaptive_grid, "HangingConstraintMode", "Penalty"))
        self.strong_hanging_constraint = self.hanging_constraint_mode in (
            "Shape",
            "ShapeFunction",
        )
        self.hanging_shape_capacity_factor = int(DictIO.GetAlternative(adaptive_grid, "HangingShapeCapacityFactor", 3))

        self.coarse_grid_size = vec3f(0.0, 0.0, 0.0)
        self.coarse_igrid_size = vec3f(0.0, 0.0, 0.0)
        self.coarse_cnum = vec3i(0, 0, 0)
        self.coarse_gnum = vec3i(0, 0, 0)
        self.fine_grid_size = vec3f(0.0, 0.0, 0.0)
        self.fine_igrid_size = vec3f(0.0, 0.0, 0.0)

        self.refined_cell = None
        self.refinement_seed = None
        self.refinement_seed_count = None
        self.new_refined_cell_count = None
        self.refined_cell_buffer = None
        self.particle_level = None
        self.particle_refined = None
        self.particle_size = None
        self.unrefined_particle_count = None
        self.split_particle_count = None
        self.split_parent_capacity = 0
        self.split_parent_id = None
        self.split_parent_position = None
        self.split_parent_velocity = None
        self.split_parent_velocity_gradient = None
        self.split_child_size = None
        self.split_child_cal_length = None
        self.split_child_mass = None
        self.split_child_volume = None
        self.split_child_level = None
        self.split_cell_volume = None
        self.particle_split_overflow = None
        self.penalty_impulse = None
        self.node_map = None
        self.logical_grid_sum = 0
        self.max_refined_cells = 0
        self.max_refined_extra_cells = 0
        self.refined_cell_budget_ratio = 0.0
        self.refined_cell_expansion = (self.finest_ratio**3) - 1
        self.logical_to_compact = None
        self.compact_to_logical = None
        self.fine_logical_to_compact = None
        self.last_refine_step = -1
        self.last_reported_refined_cells = 0
        self.adaptive_shape_mode = 0
        self.single_grid_nodes = 0
        self.active_particle_num = 0
        self.bridge_coarse_weight = None
        self.bridge_alpha = None
        self.bridge_coarse_size = None
        self.bridge_body_id = None
        self.hanging_node_count = None
        self.hanging_touched_node_count = None
        self.hanging_node_count_host = 0
        self.hanging_touched_node_count_host = 0
        self.hanging_node_id = None
        self.hanging_lookup = None
        self.hanging_master_id = None
        self.hanging_master_weight = None
        self.hanging_touched_node_id = None
        self.hanging_shape_overflow = None
        self.remaining_unrefined_particles = -1
        self.last_observed_particle_num = 0

        self._validate_configuration_values()

    def _normalized_refine_criterion(self):
        return self.refine_criterion.lower().replace("_", "").replace("-", "").replace(" ", "")

    def _validate_configuration_values(self):
        if self.refine_interval <= 0:
            raise ValueError("AdaptiveGrid/RefineInterval must be larger than zero")
        if self.max_level < 1 or self.max_level > 2:
            raise ValueError("AdaptiveGrid/MaxLevel currently supports 1 or 2")
        if self.follow_refined_particles_interval <= 0:
            raise ValueError("AdaptiveGrid/FollowRefinedParticlesInterval must be larger than zero")
        if self.particle_split_interval <= 0:
            raise ValueError("AdaptiveGrid/ParticleSplitInterval must be larger than zero")
        if self.refine_threshold <= 0.0:
            raise ValueError("AdaptiveGrid/RefineThreshold must be larger than zero")
        if not 0.0 < self.max_refined_ratio <= 1.0:
            raise ValueError("AdaptiveGrid/RefineRatio must be in (0, 1]")
        if self.particle_split_batch < 0:
            raise ValueError("AdaptiveGrid/ParticleSplitBatch cannot be negative")
        if not 0.0 < self.split_interior_volume_fraction <= 1.0:
            raise ValueError("AdaptiveGrid/SplitInteriorVolumeFraction must be in (0, 1]")
        if self.buffer_cells < 0:
            raise ValueError("AdaptiveGrid/BufferCells cannot be negative")
        if not 0.0 < self.penalty <= 1.0:
            raise ValueError("AdaptiveGrid/Penalty must be in (0, 1]")
        if self.penalty_iterations <= 0:
            raise ValueError("AdaptiveGrid/PenaltyIterations must be larger than zero")
        if self.penalty_beta <= 0.0:
            raise ValueError("AdaptiveGrid/PenaltyBeta must be larger than zero")
        if self.bridging_cells <= 0:
            raise ValueError("AdaptiveGrid/BridgingCells must be larger than zero")
        if self.hanging_constraint_mode not in ("Penalty", "Shape", "ShapeFunction"):
            raise ValueError("AdaptiveGrid/HangingConstraintMode must be Penalty or ShapeFunction")
        if self.bridging_domain and self.strong_hanging_constraint:
            raise ValueError(
                "AdaptiveGrid/HangingConstraintMode=ShapeFunction is only valid " "when BridgingDomain is disabled"
            )
        if self.bridging_domain and self.max_level > 1:
            raise ValueError("AdaptiveGrid/BridgingDomain currently supports MaxLevel=1 only")
        if self.hanging_shape_capacity_factor < 1:
            raise ValueError("AdaptiveGrid/HangingShapeCapacityFactor must be >= 1")
        if self._normalized_refine_criterion() not in (
            "auto",
            "plasticstrain",
            "equivalentplasticstrain",
            "plasticdevstrain",
            "plasticdeviatoricstrain",
            "epdstrain",
            "stress",
            "equivalentstress",
            "vonmisesstress",
            "elasticstress",
        ):
            raise ValueError(
                "AdaptiveGrid/RefineCriterion must be Auto, PlasticStrain, " "PlasticDevStrain, or EquivalentStress"
            )

    def validate_solver(self, sims: Simulation):
        unsupported = []
        if sims.dimension != 3:
            unsupported.append("3-D")
        if sims.configuration not in ("ULMPM", "TLMPM"):
            unsupported.append("ULMPM or TLMPM")
        if sims.configuration == "TLMPM" and sims.solver_type != "Explicit":
            unsupported.append("explicit TLMPM")
        supported_solver = sims.solver_type == "Explicit" or (
            sims.solver_type == "Implicit" and sims.material_type == "Solid"
        )
        if not supported_solver:
            unsupported.append("Explicit or Implicit solid solver")
        if sims.solver_type == "Implicit":
            if sims.discretization != "FEM":
                unsupported.append("FEM discretization")
            if self.bridging_domain:
                unsupported.append("BridgingDomain disabled")
            if not self.strong_hanging_constraint:
                unsupported.append("HangingConstraintMode=ShapeFunction")
        if sims.mapping not in ("USL", "USF", "MUSL"):
            unsupported.append("USL, USF, or MUSL mapping")
        if sims.shape_function not in (
            "Linear",
            "GIMP",
            "QuadBSpline",
            "CubicBSpline",
        ):
            unsupported.append("Linear, GIMP, QuadBSpline, or CubicBSpline shape function")
        if sims.contact_detection:
            unsupported.append("contact disabled")
        if sims.stabilize is not None:
            if sims.stabilize == "F-Bar Method":
                if sims.solver_type != "Explicit":
                    unsupported.append("explicit F-Bar Method")
                if self.bridging_domain:
                    unsupported.append("BridgingDomain disabled for F-Bar Method")
            elif sims.stabilize == "Displacement F-Bar Method":
                unsupported.append("standard F-Bar Method for AdaptiveGrid")
            else:
                unsupported.append("stabilization disabled except F-Bar Method")
        if sims.gauss_number > 0:
            unsupported.append("particle integration")
        if sims.velocity_projection_scheme in ("Affine", "Taylor"):
            unsupported.append("PIC/FLIP velocity projection")
        if any(sims.boundary[d] == 2 for d in range(3)):
            unsupported.append("non-periodic boundaries")
        if self.refine_particles and sims.nptraction > 0:
            unsupported.append("particle traction disabled when RefineParticles is enabled")
        if unsupported:
            requirements = ", ".join(unsupported)
            raise RuntimeError(f"AdaptiveGrid currently requires: {requirements}")

    def create_nodes(self, sims: Simulation, grid_size):
        self.validate_solver(sims)
        super().create_nodes(sims, grid_size)

        self.coarse_grid_size = vec3f(self.grid_size)
        self.coarse_igrid_size = vec3f(self.igrid_size)
        self.coarse_cnum = vec3i(self.cnum)
        self.coarse_gnum = vec3i(self.gnum)
        self.fine_grid_size = self.coarse_grid_size / self.finest_ratio
        self.fine_igrid_size = 1.0 / self.fine_grid_size

        fine_cnum = np.asarray(self.coarse_cnum, dtype=np.int32) * self.finest_ratio
        self.gnum = vec3i(fine_cnum + 1)
        fine_grid_size = np.asarray(self.fine_grid_size, dtype=np.float64)
        self.penalty_length = float(np.min(fine_grid_size))
        self.penalty_reference_volume = float(np.prod(fine_grid_size))
        self.logical_grid_sum = int(self.gnum[0] * self.gnum[1] * self.gnum[2])
        self.max_refined_cells = max(
            1,
            min(self.cellSum, int(np.floor(self.max_refined_ratio * self.cellSum))),
        )
        self.max_refined_extra_cells = max(
            1,
            int(np.floor(self.max_refined_ratio * self.cellSum * self.refined_cell_expansion)),
        )
        self.refined_cell_budget_ratio = self.max_refined_cells / self.cellSum
        self.node_map = AdaptiveNodeMap(
            self.coarse_cnum,
            self.ghost_cell,
            self.refined_cell_budget_ratio,
            self.bridging_domain,
            self.max_level,
        )
        self.gridSum = self.node_map.capacity

    def configure_penalty_scale(self, material):
        young_values = []
        mat_props = getattr(material, "matProps", None)
        if mat_props is not None:
            size = mat_props.size() if hasattr(mat_props, "size") else len(mat_props)
            for material_id in range(1, size):
                young = float(getattr(mat_props[material_id], "young", 0.0))
                if young > 0.0:
                    young_values.append(young)
        self.penalty_young = max(young_values) if young_values else 0.0

    def set_characteristic_length(self, sims: Simulation):
        self.calLength = ti.Vector.field(3, float, shape=sims.max_particle_num)
        self.particle_size = ti.Vector.field(3, float, shape=sims.max_particle_num)

    def calculate_characteristic_length(self, sims, particleNum, particle, psize):
        characteristic_mode = {
            "Linear": 0,
            "GIMP": 1,
            "QuadBSpline": 2,
            "CubicBSpline": 3,
        }[sims.shape_function]
        initialize_adaptive_particle_size(
            particleNum,
            characteristic_mode,
            self.coarse_grid_size,
            np.ascontiguousarray(psize),
            self.particle_size,
            self.calLength,
        )

    def set_essential_field(self, is_bbar, max_particle_num, shape_function, mls):
        self.single_grid_nodes = self.grid_nodes
        if self.bridging_domain:
            self.grid_nodes = 2 * self.single_grid_nodes
            self.influenced_dofs = 3 * self.grid_nodes
        elif self.strong_hanging_constraint:
            self.grid_nodes = self.hanging_shape_capacity_factor * self.single_grid_nodes
            self.influenced_dofs = 3 * self.grid_nodes
        if self.grid_nodes > 255:
            raise ValueError(
                "AdaptiveGrid/HangingShapeCapacityFactor is too large: "
                f"{self.grid_nodes} support nodes per particle exceeds the "
                "u8 node_size limit of 255"
            )
        super().set_essential_field(
            is_bbar,
            max_particle_num,
            shape_function,
            mls,
        )

    def element_initialize(self, sims: Simulation, local_coordiates=False):
        self.adaptive_shape_mode = {
            "QuadBSpline": 1,
            "CubicBSpline": 2,
        }.get(sims.shape_function, 0)
        super().element_initialize(sims, local_coordiates)
        self.refined_cell = ti.field(ti.u8, shape=self.cellSum)
        self.refinement_seed = ti.field(ti.u8, shape=self.cellSum)
        self.refinement_seed_count = ti.field(int, shape=())
        self.new_refined_cell_count = ti.field(int, shape=())
        self.refined_cell_buffer = ti.field(ti.u8, shape=self.cellSum)
        self.particle_level = ti.field(ti.u8, shape=sims.max_particle_num)
        self.particle_refined = ti.field(ti.u8, shape=sims.max_particle_num)
        self.unrefined_particle_count = ti.field(int, shape=())
        self.split_particle_count = ti.field(int, shape=())
        self.split_parent_capacity = int(sims.max_particle_num)
        if self.particle_split_batch > 0:
            self.split_parent_capacity = min(
                self.split_parent_capacity,
                self.particle_split_batch,
            )
        self.split_parent_capacity = max(1, self.split_parent_capacity)
        self.split_parent_id = ti.field(int, shape=self.split_parent_capacity)
        self.split_parent_position = ti.Vector.field(
            3,
            float,
            shape=self.split_parent_capacity,
        )
        self.split_parent_velocity = ti.Vector.field(
            3,
            float,
            shape=self.split_parent_capacity,
        )
        self.split_parent_velocity_gradient = ti.Matrix.field(
            3,
            3,
            float,
            shape=self.split_parent_capacity,
        )
        self.split_child_size = ti.Vector.field(
            3,
            float,
            shape=self.split_parent_capacity,
        )
        self.split_child_cal_length = ti.Vector.field(
            3,
            float,
            shape=self.split_parent_capacity,
        )
        self.split_child_mass = ti.field(float, shape=self.split_parent_capacity)
        self.split_child_volume = ti.field(float, shape=self.split_parent_capacity)
        self.split_child_level = ti.field(ti.u8, shape=self.split_parent_capacity)
        self.split_cell_volume = ti.field(float, shape=self.cellSum)
        self.particle_split_overflow = ti.field(int, shape=())
        self.logical_to_compact = ti.field(int, shape=self.logical_grid_sum)
        self.compact_to_logical = ti.field(int, shape=self.gridSum)
        if self.bridging_domain:
            self.fine_logical_to_compact = ti.field(
                int,
                shape=self.logical_grid_sum,
            )
        self.node_map.bind_fields(
            self.logical_to_compact,
            self.compact_to_logical,
            self.fine_logical_to_compact,
        )
        self.penalty_impulse = ti.Vector.field(3, float, shape=(self.gridSum, self.grid_level))
        if self.bridging_domain:
            self.bridge_coarse_weight = ti.field(
                float,
                shape=self.node_map.coarse_node_count,
            )
            self.bridge_alpha = ti.field(float, shape=sims.max_particle_num)
            self.bridge_coarse_size = ti.field(ti.u8, shape=sims.max_particle_num)
            self.bridge_body_id = ti.field(ti.u8, shape=sims.max_particle_num)
        else:
            self.hanging_node_count = ti.field(int, shape=())
            self.hanging_touched_node_count = ti.field(int, shape=())
            self.hanging_node_id = ti.field(int, shape=self.gridSum)
            self.hanging_lookup = ti.field(int, shape=self.gridSum)
            self.hanging_lookup.fill(-1)
            self.hanging_master_id = ti.Vector.field(8, int, shape=self.gridSum)
            self.hanging_master_weight = ti.Vector.field(8, float, shape=self.gridSum)
            self.hanging_touched_node_id = ti.field(int, shape=self.gridSum)
            self.hanging_shape_overflow = ti.field(int, shape=())
        self.calculate = self.calc_shape_fn_adaptive

    def _update_hanging_constraint_table(self, refined_cell):
        if self.bridging_domain:
            return
        slave_ids, master_ids, master_weights, touched_ids = build_hanging_constraint_table(
            refined_cell,
            self.coarse_cnum,
            self.gnum,
            self.node_map.logical_to_compact,
            self.node_map.compact_to_logical,
            self.node_map.next_compact_id,
            self.refinement_ratio,
            self.max_level,
        )
        assert slave_ids.size <= self.gridSum
        assert touched_ids.size <= self.gridSum

        self.hanging_node_count[None] = int(slave_ids.size)
        self.hanging_touched_node_count[None] = int(touched_ids.size)
        self.hanging_node_count_host = int(slave_ids.size)
        self.hanging_touched_node_count_host = int(touched_ids.size)
        self.hanging_lookup.fill(-1)
        if slave_ids.size > 0:
            load_hanging_constraint_table(
                int(slave_ids.size),
                int(touched_ids.size),
                np.ascontiguousarray(slave_ids, dtype=np.int32),
                np.ascontiguousarray(master_ids, dtype=np.int32),
                np.ascontiguousarray(master_weights, dtype=np.float64),
                np.ascontiguousarray(touched_ids, dtype=np.int32),
                self.hanging_node_id,
                self.hanging_lookup,
                self.hanging_master_id,
                self.hanging_master_weight,
                self.hanging_touched_node_id,
            )

    def calc_shape_fn_adaptive(self, particleNum, particle):
        self.active_particle_num = int(particleNum[0])
        if self.bridging_domain:
            adaptive_bridging_global_update(
                self.grid_nodes,
                self.single_grid_nodes,
                self.influenced_node,
                self.adaptive_shape_mode,
                self.refinement_ratio,
                self.coarse_grid_size,
                self.coarse_igrid_size,
                self.coarse_cnum,
                self.fine_grid_size,
                self.fine_igrid_size,
                self.gnum,
                self.active_particle_num,
                particle,
                self.calLength,
                self.refined_cell,
                self.particle_level,
                self.logical_to_compact,
                self.fine_logical_to_compact,
                self.bridge_coarse_weight,
                self.bridge_alpha,
                self.bridge_coarse_size,
                self.bridge_body_id,
                self.node_size,
                self.LnID,
                self.shape_fn,
                self.dshape_fn,
                self.shape_function,
                self.grad_shape_function,
            )
            return
        if self.strong_hanging_constraint:
            self.hanging_shape_overflow[None] = 0
        adaptive_global_update(
            self.grid_nodes,
            self.influenced_node,
            self.adaptive_shape_mode,
            1 if self.strong_hanging_constraint else 0,
            self.refinement_ratio,
            self.max_level,
            self.finest_ratio,
            self.coarse_grid_size,
            self.coarse_igrid_size,
            self.coarse_cnum,
            self.fine_grid_size,
            self.fine_igrid_size,
            self.gnum,
            particleNum[0],
            particle,
            self.calLength,
            self.refined_cell,
            self.particle_level,
            self.logical_to_compact,
            self.hanging_lookup,
            self.hanging_master_id,
            self.hanging_master_weight,
            self.hanging_shape_overflow,
            self.node_size,
            self.LnID,
            self.shape_fn,
            self.dshape_fn,
            self.shape_function,
            self.grad_shape_function,
        )
        if self.strong_hanging_constraint and int(self.hanging_shape_overflow[None]) != 0:
            raise RuntimeError(
                "AdaptiveGrid/HangingShapeCapacityFactor is too small for "
                f"{self.hanging_constraint_mode}. Current support capacity is "
                f"{self.grid_nodes} nodes per particle "
                f"({self.hanging_shape_capacity_factor} x {self.single_grid_nodes})."
            )

    def _select_refinement_marker(self, material, stress_as_matrix=False):
        material_name = material.__class__.__name__
        criterion = self._normalized_refine_criterion()
        is_soft = bool(getattr(material, "is_soft", False))
        finite_elastic_models = (
            "HenckyElasticModel",
            "NeoHookeanModel",
            "MooneyRivlin",
            "Gent",
            "Hydrogel",
        )

        if criterion == "auto":
            if material_name == "LinearElasticModel":
                return mark_refined_cells_matrix_stress if stress_as_matrix else mark_refined_cells_stress
            if material_name in finite_elastic_models:
                return mark_refined_cells_matrix_stress
            if material_name == "ElasticPerfectlyPlasticModel":
                return mark_refined_cells_strain if is_soft else mark_refined_cells_epstrain
            if material_name == "MohrCoulombModel":
                return mark_refined_cells_deviatoric_strain if is_soft else mark_refined_cells_epdstrain
            if material_name in ("DruckerPragerModel", "ModifiedCamClayModel"):
                return mark_refined_cells_softening if is_soft else mark_refined_cells_epdstrain

        if criterion in ("stress", "equivalentstress", "vonmisesstress", "elasticstress"):
            if stress_as_matrix or material_name in finite_elastic_models:
                return mark_refined_cells_matrix_stress
            return mark_refined_cells_stress

        if material_name == "LinearElasticModel":
            raise RuntimeError("AdaptiveGrid/RefineCriterion for LinearElastic must be " "Auto or EquivalentStress")

        if criterion in ("plasticstrain", "equivalentplasticstrain"):
            if material_name == "ElasticPerfectlyPlasticModel":
                return mark_refined_cells_strain if is_soft else mark_refined_cells_epstrain
            if material_name == "MohrCoulombModel":
                return mark_refined_cells_deviatoric_strain if is_soft else mark_refined_cells_epdstrain
            if material_name in ("DruckerPragerModel", "ModifiedCamClayModel"):
                return mark_refined_cells_softening if is_soft else mark_refined_cells_epdstrain

        if criterion in ("plasticdevstrain", "plasticdeviatoricstrain", "epdstrain"):
            if material_name == "ElasticPerfectlyPlasticModel":
                return mark_refined_cells_strain if is_soft else mark_refined_cells_epstrain
            if material_name == "MohrCoulombModel":
                return mark_refined_cells_deviatoric_strain if is_soft else mark_refined_cells_epdstrain
            if material_name in ("DruckerPragerModel", "ModifiedCamClayModel"):
                return mark_refined_cells_softening if is_soft else mark_refined_cells_epdstrain

        raise RuntimeError(
            f"AdaptiveGrid/RefineCriterion={self.refine_criterion} does not " f"support material {material_name}"
        )

    def update_refinement(self, sims: Simulation, scene):
        update_grid = sims.current_step % self.refine_interval == 0 and sims.current_step != self.last_refine_step
        follow_particles = (
            self.refine_particles
            and self.follow_refined_particles
            and self.last_reported_refined_cells > 0
            and sims.current_step % self.follow_refined_particles_interval == 0
        )
        new_refined_cells_added = False
        if update_grid or follow_particles:
            if scene.material.matProps.size() != 2:
                raise RuntimeError("AdaptiveGrid currently supports one deformable material")

            self.refinement_seed.fill(0)
            if update_grid:
                self.last_refine_step = sims.current_step
                material = scene.material.matProps[1]
                marker = self._select_refinement_marker(
                    material,
                    stress_as_matrix=sims.configuration == "TLMPM",
                )
                marker(
                    self.refine_threshold,
                    self.max_level,
                    self.coarse_igrid_size,
                    self.coarse_cnum,
                    int(scene.particleNum[0]),
                    scene.particle,
                    scene.material.stateVars,
                    self.refined_cell,
                    self.refinement_seed,
                )
            if follow_particles:
                mark_refined_cells_migrated_particles(
                    self.coarse_igrid_size,
                    self.coarse_cnum,
                    int(scene.particleNum[0]),
                    scene.particle,
                    self.refined_cell,
                    self.particle_refined,
                    self.refinement_seed,
                )
            count_refined_cells(self.refinement_seed, self.refinement_seed_count)
            if int(self.refinement_seed_count[None]) > 0:
                for _ in range(self.buffer_cells):
                    self.refined_cell_buffer.fill(0)
                    dilate_refined_cells(self.coarse_cnum, self.refinement_seed, self.refined_cell_buffer)
                    merge_refined_cells(self.refinement_seed, self.refined_cell_buffer)
                count_new_refined_cells(
                    self.refined_cell,
                    self.refinement_seed,
                    self.new_refined_cell_count,
                )
                if int(self.new_refined_cell_count[None]) > 0:
                    refined_cell = np.maximum(
                        self.refined_cell.to_numpy(),
                        self.refinement_seed.to_numpy(),
                    ).astype(np.uint8)
                    refined_cell = self._balance_refined_cell_levels(refined_cell)
                    refined_cells = int(np.count_nonzero(refined_cell))
                    refined_ratio = refined_cells / self.cellSum
                    refined_extra_cells = self._refined_extra_cell_count(refined_cell)
                    if refined_extra_cells > self.max_refined_extra_cells:
                        refined_extra_ratio = refined_extra_cells / (self.cellSum * self.refined_cell_expansion)
                        raise RuntimeError(
                            f"AdaptiveGrid refined coarse-cell ratio "
                            f"{refined_ratio:.6f} "
                            f"(extra fine-cell ratio {refined_extra_ratio:.6f}) "
                            f"exceeds AdaptiveGrid/RefineRatio="
                            f"{self.max_refined_ratio:.6f}. Increase RefineRatio "
                            "or tighten the refinement criterion."
                        )
                    self.node_map.ensure_refined_cells(refined_cell, sims.shape_function)
                    self.refined_cell.from_numpy(refined_cell)
                    if self.bridging_domain:
                        self.bridge_coarse_weight.from_numpy(
                            build_bridging_node_weights(
                                refined_cell,
                                self.coarse_cnum,
                                self.bridging_cells,
                            )
                        )
                    else:
                        self._update_hanging_constraint_table(refined_cell)
                    self.last_reported_refined_cells = refined_cells
                    new_refined_cells_added = True

        old_particle_num = int(scene.particleNum[0])
        if old_particle_num > self.last_observed_particle_num or new_refined_cells_added:
            self.remaining_unrefined_particles = -1
        split_update = (
            update_grid
            or new_refined_cells_added
            or old_particle_num > self.last_observed_particle_num
            or sims.current_step % self.particle_split_interval == 0
        )
        split_any = False
        if (
            self.refine_particles
            and self.last_reported_refined_cells > 0
            and self.remaining_unrefined_particles != 0
            and split_update
        ):
            min_split_cell_volume = 0.0
            if self.split_interior_only:
                accumulate_split_cell_volume(
                    self.coarse_igrid_size,
                    self.coarse_cnum,
                    old_particle_num,
                    scene.particle,
                    self.split_cell_volume,
                )
                coarse_cell_volume = float(
                    self.coarse_grid_size[0] * self.coarse_grid_size[1] * self.coarse_grid_size[2]
                )
                min_split_cell_volume = self.split_interior_volume_fraction * coarse_cell_volume
            while True:
                old_particle_num = int(scene.particleNum[0])
                max_split_parents = min(
                    self.split_parent_capacity,
                    max(0, (sims.max_particle_num - old_particle_num) // 7),
                )
                if self.particle_split_batch > 0:
                    max_split_parents = min(max_split_parents, self.particle_split_batch)
                collect_particles_to_split(
                    self.coarse_igrid_size,
                    self.coarse_cnum,
                    old_particle_num,
                    max_split_parents,
                    min_split_cell_volume,
                    scene.particle,
                    self.refined_cell,
                    self.split_cell_volume,
                    self.particle_refined,
                    self.unrefined_particle_count,
                    self.split_particle_count,
                    self.split_parent_id,
                )
                candidate_split_parents = int(self.split_particle_count[None])
                if candidate_split_parents == 0:
                    self.remaining_unrefined_particles = 0
                    break
                required_particle_num = old_particle_num + 7 * candidate_split_parents
                if required_particle_num > sims.max_particle_num:
                    raise RuntimeError(
                        f"AdaptiveGrid particle refinement requires {required_particle_num} "
                        f"particles, but max_particle_number={sims.max_particle_num}. "
                        "Increase max_particle_number or disable RefineParticles."
                    )
                split_parents = min(candidate_split_parents, max_split_parents)
                self.remaining_unrefined_particles = max(
                    0,
                    candidate_split_parents - split_parents,
                )
                if split_parents == 0:
                    raise RuntimeError(
                        "AdaptiveGrid particle refinement requires additional "
                        "particle capacity, but max_particle_number has no room "
                        "for 1-to-8 splitting."
                    )
                self.particle_split_overflow[None] = 0
                cache_refined_particle_parents(
                    split_parents,
                    self.split_parent_id,
                    scene.particle,
                    self.particle_size,
                    self.calLength,
                    self.split_parent_position,
                    self.split_parent_velocity,
                    self.split_parent_velocity_gradient,
                    self.split_child_size,
                    self.split_child_cal_length,
                    self.split_child_mass,
                    self.split_child_volume,
                    self.split_child_level,
                    self.particle_refined,
                )
                copy_refined_particle_children(
                    split_parents,
                    old_particle_num,
                    self.split_parent_id,
                    scene.particle,
                    scene.material.stateVars,
                    sims.max_particle_num,
                    self.particle_split_overflow,
                )
                if int(self.particle_split_overflow[None]) > 0:
                    raise RuntimeError(
                        "AdaptiveGrid particle refinement exceeded "
                        f"max_particle_number={sims.max_particle_num}. "
                        "Increase max_particle_number or disable RefineParticles."
                    )
                split_refined_particles(
                    split_parents,
                    old_particle_num,
                    self.split_parent_id,
                    scene.particle,
                    self.split_parent_position,
                    self.split_parent_velocity,
                    self.split_parent_velocity_gradient,
                    self.split_child_size,
                    self.split_child_cal_length,
                    self.split_child_mass,
                    self.split_child_volume,
                    self.split_child_level,
                    self.particle_size,
                    self.calLength,
                    self.particle_level,
                    self.particle_refined,
                )
                scene.particleNum[0] = old_particle_num + 7 * split_parents
                split_any = True
                if split_parents == candidate_split_parents:
                    self.remaining_unrefined_particles = 0
                    break
        if split_any:
            scene.material.update_material_mapping(
                scene.particle,
                int(scene.particleNum[0]),
            )
        self.last_observed_particle_num = int(scene.particleNum[0])

    def should_update_refinement(self, sims: Simulation, particleNum):
        current_particle_num = int(particleNum[0])
        if sims.current_step % self.refine_interval == 0 and sims.current_step != self.last_refine_step:
            return True
        if current_particle_num > self.last_observed_particle_num:
            return True
        if (
            self.refine_particles
            and self.follow_refined_particles
            and self.last_reported_refined_cells > 0
            and sims.current_step % self.follow_refined_particles_interval == 0
        ):
            return True
        return (
            self.refine_particles
            and self.last_reported_refined_cells > 0
            and self.remaining_unrefined_particles != 0
            and sims.current_step % self.particle_split_interval == 0
        )

    def _refined_extra_cell_count(self, refined_cell):
        levels = np.asarray(refined_cell, dtype=np.int32)
        factors = np.power(self.refinement_ratio, 3 * levels, dtype=np.int64) - 1
        return int(np.sum(factors))

    def _balance_refined_cell_levels(self, refined_cell):
        levels = (
            np.asarray(refined_cell, dtype=np.uint8)
            .reshape(
                tuple(np.asarray(self.coarse_cnum)),
                order="F",
            )
            .copy()
        )
        changed = True
        while changed:
            changed = False
            for cell in np.ndindex(*levels.shape):
                current = int(levels[cell])
                for axis in range(3):
                    for direction in (-1, 1):
                        neighbor = list(cell)
                        neighbor[axis] += direction
                        if not 0 <= neighbor[axis] < levels.shape[axis]:
                            continue
                        neighbor = tuple(neighbor)
                        target = max(current, int(levels[neighbor]) - 1)
                        target = min(target, self.max_level)
                        if target > current:
                            levels[cell] = target
                            current = target
                            changed = True
        return np.ascontiguousarray(levels.flatten(order="F"), dtype=np.uint8)

    def apply_hanging_constraints(self, mass_cutoff, dt, node, particle):
        if self.strong_hanging_constraint:
            return
        for _ in range(self.penalty_iterations):
            if self.bridging_domain:
                self.penalty_impulse.fill(0)
                assemble_bridging_penalty_impulse(
                    mass_cutoff,
                    self.penalty,
                    self.penalty_beta,
                    self.penalty_young,
                    self.penalty_length,
                    dt,
                    self.grid_nodes,
                    self.active_particle_num,
                    particle,
                    self.bridge_alpha,
                    self.bridge_coarse_size,
                    self.bridge_body_id,
                    self.node_size,
                    self.LnID,
                    self.shape_fn,
                    node,
                    self.penalty_impulse,
                )
                apply_hanging_penalty_impulse(mass_cutoff, dt, node, self.penalty_impulse)
            else:
                hanging_count = self.hanging_node_count_host
                if hanging_count <= 0:
                    return
                touched_count = self.hanging_touched_node_count_host
                assemble_hanging_penalty_impulse_list(
                    mass_cutoff,
                    self.penalty,
                    self.penalty_beta,
                    self.penalty_young,
                    self.penalty_length,
                    self.penalty_reference_volume,
                    dt,
                    hanging_count,
                    self.hanging_node_id,
                    self.hanging_master_id,
                    self.hanging_master_weight,
                    node,
                    self.penalty_impulse,
                )
                apply_hanging_penalty_impulse_list(
                    mass_cutoff,
                    dt,
                    touched_count,
                    self.hanging_touched_node_id,
                    node,
                    self.penalty_impulse,
                )

    def apply_hanging_velocity_constraints(self, mass_cutoff, dt, node, particle):
        if self.strong_hanging_constraint:
            return
        for _ in range(self.penalty_iterations):
            if self.bridging_domain:
                self.penalty_impulse.fill(0)
                assemble_bridging_penalty_impulse(
                    mass_cutoff,
                    self.penalty,
                    self.penalty_beta,
                    self.penalty_young,
                    self.penalty_length,
                    dt,
                    self.grid_nodes,
                    self.active_particle_num,
                    particle,
                    self.bridge_alpha,
                    self.bridge_coarse_size,
                    self.bridge_body_id,
                    self.node_size,
                    self.LnID,
                    self.shape_fn,
                    node,
                    self.penalty_impulse,
                )
                apply_hanging_penalty_velocity(mass_cutoff, node, self.penalty_impulse)
            else:
                hanging_count = self.hanging_node_count_host
                if hanging_count <= 0:
                    return
                touched_count = self.hanging_touched_node_count_host
                assemble_hanging_penalty_impulse_list(
                    mass_cutoff,
                    self.penalty,
                    self.penalty_beta,
                    self.penalty_young,
                    self.penalty_length,
                    self.penalty_reference_volume,
                    dt,
                    hanging_count,
                    self.hanging_node_id,
                    self.hanging_master_id,
                    self.hanging_master_weight,
                    node,
                    self.penalty_impulse,
                )
                apply_hanging_penalty_velocity_list(
                    mass_cutoff,
                    touched_count,
                    self.hanging_touched_node_id,
                    node,
                    self.penalty_impulse,
                )

    def get_boundary_nodes(self, start_point, end_point):
        start_point = np.asarray(start_point)
        end_point = np.asarray(end_point)
        region_size = end_point - start_point
        start_point = start_point - 1e-6 * region_size
        end_point = end_point + 1e-6 * region_size
        start_bound = np.ceil(start_point * np.asarray(self.fine_igrid_size))
        end_bound = np.floor(end_point * np.asarray(self.fine_igrid_size)) + 1
        end_bound = np.maximum(end_bound, start_bound + 1)
        start_bound = np.maximum(start_bound, 0).astype(np.int32)
        end_bound = np.minimum(end_bound, np.asarray(self.gnum)).astype(np.int32)

        axes = [np.arange(start_bound[d], end_bound[d], dtype=np.int32) for d in range(3)]
        total_nodes = np.array(list(product(*axes)), dtype=np.int32)
        logical_ids = np.asarray(
            [n[0] + n[1] * self.gnum[0] + n[2] * self.gnum[0] * self.gnum[1] for n in total_nodes],
            dtype=np.int32,
        )
        if self.bridging_domain:
            logical_indices = total_nodes[np.all(total_nodes % self.refinement_ratio == 0, axis=1)]
            coarse_logical_ids = (
                logical_indices[:, 0]
                + logical_indices[:, 1] * self.gnum[0]
                + logical_indices[:, 2] * self.gnum[0] * self.gnum[1]
            )
            coarse_ids = self.node_map.ensure_logical_ids(coarse_logical_ids)
            fine_ids = self.node_map.ensure_fine_logical_ids(logical_ids)
            return np.unique(np.concatenate((coarse_ids, fine_ids)).astype(np.int32))
        return np.asarray(
            self.node_map.ensure_logical_ids(logical_ids),
            dtype=np.int32,
        )

    def set_boundary_type(self, sims: Simulation, grid_level):
        # Adaptive interpolation derives B-spline boundary types from the
        # active coarse/fine grid level instead of compact node IDs.
        return

    def calc_critical_timestep(self, velocity):
        return min(self.fine_grid_size) / velocity if velocity > 0.0 else 10000

    def print_message(self):
        print("Coarse grid size = ", self.coarse_grid_size)
        print("Finest grid size = ", self.fine_grid_size)
        print("Logical finest-grid nodes = ", self.gnum)
        maximum_node_count = self.logical_grid_sum
        if self.bridging_domain:
            maximum_node_count += self.node_map.coarse_node_count
        print(
            f"Dense adaptive node capacity = {self.gridSum}/{maximum_node_count} "
            f"(RefineRatio={self.max_refined_ratio:.2%} finest-cell budget, "
            f"extra finest-cell budget="
            f"{self.max_refined_extra_cells / max(self.cellSum * self.refined_cell_expansion, 1):.2%}, "
            "including boundary reserve)"
        )
        print(
            f"Adaptive refinement = {self.max_level + 1} level 2:1, refine-only, "
            f"particle splitting={self._particle_split_message()}"
        )

    def _particle_split_message(self):
        if not self.refine_particles:
            return "disabled"
        if self.particle_split_batch > 0:
            return f"successive 1-to-8, chunk {self.particle_split_batch}/kernel"
        return "successive 1-to-8"
