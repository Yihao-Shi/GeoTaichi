"""Configuration state shared by explicit and IPC FEM--DEM coupling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import taichi as ti

from src.utils.TimeTicker import Timer
from src.utils.SolverRuntime import StepSchedule


class FEDEMSimulation:
    def __init__(self):
        self.domain = np.zeros(3, dtype=np.float64)
        self.search = "LinkedCell"
        self.dt = ti.field(dtype=float, shape=())
        self.delta = 0.0
        self.current_time = 0.0
        self.current_step = 0
        self.current_print = 0
        self.time = 0.0
        self.cfl = 0.5
        self.save_interval = 1.0e6
        self.path = None
        self.max_particle_num = 0
        self.max_surface_facet_num = 0
        self.max_dem_material_num = 0
        self.max_fem_body_num = 0
        self.contact_coordination_number = 16
        self.point_triangle_coordination_number = 16.0
        self.edge_edge_coordination_number = 64.0
        self.compaction_ratio = 1.0
        self.max_contact_pairs = 0
        self.max_filtered_contact_pairs = 0
        self.max_contact_history_pairs = 0
        self.max_fem_wall_pairs = 0
        self.max_fem_wall_history_pairs = 0
        self.max_point_triangle_pairs = 0
        self.max_edge_edge_pairs = 0
        self.max_facet_cell_pairs = 0
        self.max_levelset_cell_pairs = 0
        self.verlet_distance = 0.0
        self.verlet_distance_multiplier = 0.1
        self.save_contact = True
        self.save_checkpoint = False
        self.contact_work_mode = "Explicit"
        self.track_energy = False
        self.step_schedule = StepSchedule()
        self.timer = Timer()

    def set_domain(self, domain):
        values = np.asarray(domain, dtype=np.float64).reshape(-1)
        if values.size != 3 or np.any(values <= 0.0):
            raise ValueError("FEDEM domain must contain three positive lengths")
        self.domain = values

    def set_timestep(self, timestep):
        value = float(timestep)
        if value <= 0.0:
            raise ValueError("FEDEM timestep must be positive")
        self.dt[None] = value
        self.delta = value

    def set_simulation_time(self, simulation_time):
        self.time = float(simulation_time)
        if self.time < 0.0:
            raise ValueError("FEDEM simulation time cannot be negative")

    def set_save_interval(self, save_interval):
        self.save_interval = float(save_interval)
        if self.save_interval <= 0.0:
            raise ValueError("FEDEM save interval must be positive")

    def set_save_path(self, path):
        self.path = None if path is None else str(Path(path))

    def set_contact_work_mode(self, mode):
        normalized = str(mode).replace("_", "").replace("-", "").replace(" ", "").lower()
        if normalized in ("explicit", "legacy", "none", "off"):
            self.contact_work_mode = "Explicit"
        else:
            raise ValueError("FEDEM contact_work_mode must be 'Explicit'")

    def set_runtime_options(self, options, output_interval):
        self.track_energy = bool(options.get("track_energy", False))
        self.step_schedule = StepSchedule.from_options(options, output_interval=output_interval)

    def configure_memory(self, memory, dem_sims, face_count, body_count):
        self.max_particle_num = int(memory.get("max_particle_number", dem_sims.max_particle_num))
        self.max_surface_facet_num = int(
            memory.get("max_surface_facet_number", memory.get("max_patch_number", face_count))
        )
        self.max_dem_material_num = int(dem_sims.max_material_num)
        self.max_fem_body_num = int(max(body_count, 1))
        self.contact_coordination_number = max(int(memory.get("contact_coordination_number", 16)), 1)
        self.compaction_ratio = float(memory.get("compaction_ratio", 1.0))
        if not 0.0 < self.compaction_ratio <= 1.0:
            raise ValueError("FEDEM compaction_ratio must be in (0, 1]")
        potential = self.max_particle_num * self.contact_coordination_number
        self.max_contact_pairs = int(
            memory.get(
                "max_contact_pairs",
                max(1, int(np.ceil(self.compaction_ratio * potential))),
            )
        )
        self.max_filtered_contact_pairs = int(memory.get("max_filtered_contact_pairs", self.max_contact_pairs))
        # Tangential history is sparse and independent of the raw BVH output.
        # Keep the old full-capacity default for general callers; production
        # jobs may size this from measured nonzero histories.
        self.max_contact_history_pairs = int(memory.get("max_contact_history_pairs", self.max_contact_pairs))
        self.max_fem_wall_pairs = int(memory.get("max_fem_wall_pairs", self.max_contact_pairs))
        self.max_fem_wall_history_pairs = int(memory.get("max_fem_wall_history_pairs", self.max_fem_wall_pairs))
        # Triangle-mesh IPC owns primitive PT/EE streams that scale with the
        # selected surface triangles, not with DEM particle--surface pairs.
        self.point_triangle_coordination_number = float(memory.get("point_triangle_coordination_number", 16.0))
        self.edge_edge_coordination_number = float(memory.get("edge_edge_coordination_number", 64.0))
        if not np.isfinite(self.point_triangle_coordination_number) or self.point_triangle_coordination_number <= 0.0:
            raise ValueError("FEDEM point_triangle_coordination_number must be positive")
        if not np.isfinite(self.edge_edge_coordination_number) or self.edge_edge_coordination_number <= 0.0:
            raise ValueError("FEDEM edge_edge_coordination_number must be positive")
        self.max_point_triangle_pairs = int(
            memory.get(
                "max_point_triangle_pairs",
                max(
                    1,
                    int(np.ceil(face_count * self.point_triangle_coordination_number)),
                ),
            )
        )
        self.max_edge_edge_pairs = int(
            memory.get(
                "max_edge_edge_pairs",
                max(
                    1,
                    int(np.ceil(face_count * self.edge_edge_coordination_number)),
                ),
            )
        )
        self.max_facet_cell_pairs = int(
            memory.get(
                "max_facet_cell_pairs",
                max(self.max_surface_facet_num * 64, self.max_surface_facet_num, 1),
            )
        )
        self.max_levelset_cell_pairs = int(
            memory.get(
                "max_levelset_cell_pairs",
                max(self.max_particle_num * 64, self.max_particle_num, 1),
            )
        )
        self.verlet_distance_multiplier = float(
            memory.get("verlet_distance_multiplier", self.verlet_distance_multiplier)
        )
        explicit_skin = memory.get("verlet_distance", None)
        if explicit_skin is not None:
            self.verlet_distance = float(explicit_skin)

    def validate(self):
        if self.search not in ("LinkedCell", "BVH"):
            raise ValueError("FEDEM search must be 'LinkedCell' or 'BVH'")
        if self.max_particle_num <= 0:
            raise RuntimeError("allocate DEM particle memory before FEDEM")
        if self.max_surface_facet_num <= 0:
            raise RuntimeError("FEDEM coupled surface contains no facets")
        if self.max_dem_material_num <= 0:
            raise RuntimeError("allocate DEM material memory before FEDEM")
        if self.max_contact_pairs <= 0:
            raise ValueError("FEDEM max_contact_pairs must be positive")
        if self.max_filtered_contact_pairs <= 0:
            raise ValueError("FEDEM max_filtered_contact_pairs must be positive")
        if self.max_contact_history_pairs <= 0:
            raise ValueError("FEDEM max_contact_history_pairs must be positive")
        if self.max_fem_wall_pairs <= 0:
            raise ValueError("FEDEM max_fem_wall_pairs must be positive")
        if self.max_fem_wall_history_pairs <= 0:
            raise ValueError("FEDEM max_fem_wall_history_pairs must be positive")
        if self.max_point_triangle_pairs <= 0:
            raise ValueError("FEDEM max_point_triangle_pairs must be positive")
        if self.max_edge_edge_pairs <= 0:
            raise ValueError("FEDEM max_edge_edge_pairs must be positive")
        if self.max_levelset_cell_pairs <= 0:
            raise ValueError("FEDEM max_levelset_cell_pairs must be positive")


__all__ = ["FEDEMSimulation"]
