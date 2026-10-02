"""Shared configuration state for explicit and implicit FEM--MPM coupling."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import taichi as ti

from src.utils.TimeTicker import Timer
from src.utils.SolverRuntime import StepSchedule


class FEMPMSimulation:
    def __init__(self):
        self.dimension = 3
        self.is_axisymmetric = False
        self.axis_offset = 0.0
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
        self.max_mpm_material_num = 0
        self.max_fem_body_num = 0
        self.contact_coordination_number = 16
        self.compaction_ratio = 1.0
        self.max_contact_pairs = 0
        self.max_point_triangle_pairs = 0
        self.max_point_edge_pairs = 0
        self.max_facet_cell_pairs = 0
        self.verlet_distance = 0.0
        self.verlet_distance_multiplier = 0.1
        self.save_contact = True
        self.track_energy = False
        self.step_schedule = StepSchedule()
        self.timer = Timer()

    def set_domain(
        self,
        domain,
        dimension=None,
        axisymmetric=False,
        axis_offset=0.0,
    ):
        values = np.asarray(domain, dtype=np.float64).reshape(-1)
        self.dimension = values.size if dimension is None else int(dimension)
        if self.dimension not in (2, 3) or values.size < self.dimension:
            raise ValueError("FEMPM domain must match dimension=2 or 3")
        if np.any(values[: self.dimension] <= 0.0):
            raise ValueError("FEMPM domain lengths must be positive")
        self.domain = np.zeros(3, dtype=np.float64)
        self.domain[: self.dimension] = values[: self.dimension]
        self.is_axisymmetric = bool(axisymmetric)
        self.axis_offset = float(axis_offset)
        if self.is_axisymmetric and self.dimension != 2:
            raise ValueError("axisymmetric FEMPM requires dimension=2")
        if not math.isfinite(self.axis_offset):
            raise ValueError("FEMPM axis_offset must be finite")

    def set_timestep(self, timestep):
        value = float(timestep)
        if value <= 0.0:
            raise ValueError("FEMPM timestep must be positive")
        self.dt[None] = value
        self.delta = value

    def set_simulation_time(self, simulation_time):
        self.time = float(simulation_time)
        if self.time < 0.0:
            raise ValueError("FEMPM simulation time cannot be negative")

    def set_save_interval(self, save_interval):
        self.save_interval = float(save_interval)
        if self.save_interval <= 0.0:
            raise ValueError("FEMPM save interval must be positive")

    def set_save_path(self, path):
        self.path = None if path is None else str(Path(path))

    def set_runtime_options(self, options, output_interval):
        self.track_energy = bool(options.get("track_energy", False))
        self.step_schedule = StepSchedule.from_options(options, output_interval=output_interval)

    def configure_memory(self, memory, mpm_sims, face_count, body_count):
        self.max_particle_num = int(memory.get("max_particle_number", mpm_sims.max_particle_num))
        self.max_surface_facet_num = int(
            memory.get(
                "max_surface_facet_number",
                memory.get("max_patch_number", face_count),
            )
        )
        if self.max_surface_facet_num < int(face_count):
            raise ValueError("FEMPM max_surface_facet_number is smaller than the selected " "surface facet count")
        # MPM reserves material id 0 and exposes user materials as 1..N.
        self.max_mpm_material_num = int(memory.get("max_mpm_material_number", int(mpm_sims.max_material_num) + 1))
        self.max_fem_body_num = int(max(body_count, 1))
        self.contact_coordination_number = max(int(memory.get("contact_coordination_number", 16)), 1)
        self.compaction_ratio = float(memory.get("compaction_ratio", 1.0))
        if not 0.0 < self.compaction_ratio <= 1.0:
            raise ValueError("FEMPM compaction_ratio must be in (0, 1]")
        potential = self.max_particle_num * self.contact_coordination_number
        self.max_contact_pairs = int(
            memory.get(
                "max_contact_pairs",
                max(1, int(np.ceil(self.compaction_ratio * potential))),
            )
        )
        # Implicit IPC uses a point--triangle stencil in 3D and a point--edge
        # stencil in 2D.  Keep both capacities explicit so broad phase,
        # culling, friction history and sparse matrices can be allocated once
        # at construction instead of rebuilding Taichi fields after a search.
        self.max_point_triangle_pairs = int(memory.get("max_point_triangle_pairs", self.max_contact_pairs))
        self.max_point_edge_pairs = int(memory.get("max_point_edge_pairs", self.max_contact_pairs))
        self.max_facet_cell_pairs = int(
            memory.get(
                "max_facet_cell_pairs",
                max(self.max_surface_facet_num * 64, self.max_surface_facet_num, 1),
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
            raise ValueError("FEMPM search must be 'LinkedCell' or 'BVH'")
        if self.max_particle_num <= 0:
            raise RuntimeError("allocate MPM particle memory before FEMPM")
        if self.max_surface_facet_num <= 0:
            raise RuntimeError("FEMPM coupled surface contains no facets")
        if self.max_mpm_material_num <= 0:
            raise RuntimeError("allocate MPM material memory before FEMPM")
        if self.max_contact_pairs <= 0:
            raise ValueError("FEMPM max_contact_pairs must be positive")
        if self.max_point_triangle_pairs <= 0:
            raise ValueError("FEMPM max_point_triangle_pairs must be positive")
        if self.max_point_edge_pairs <= 0:
            raise ValueError("FEMPM max_point_edge_pairs must be positive")


__all__ = ["FEMPMSimulation"]
