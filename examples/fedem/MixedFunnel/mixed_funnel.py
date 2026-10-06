#!/usr/bin/env python3
"""Explicit soft-FEM/rigid-LSDEM discharge through DEM facet walls."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fedem.MixedFunnel.draw.output_audit import collect_native_output
from examples.fedem.MixedFunnel.mesh_reference import (
    place_reference,
    tetrahedralize_irregular_reference,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "OutputData"
RIGID_IRREGULAR_SURFACE = REPO_ROOT / "assets/mesh/LSDEM/sand.stl"
PROMPT = (
    "Generate an equal mixture of irregular FEM soft grains and level-set "
    "DEM rigid grains in a closed three-dimensional funnel. Allow the "
    "assembly to settle, remove the outlet gate, and record its energy "
    "evolution while it discharges into a separate catcher."
)

SEED = 20260816
PARTICLE_COUNT = 800
SOFT_COUNT = 400
RIGID_COUNT = PARTICLE_COUNT - SOFT_COUNT
SOFT_DENSITY = 1200.0
RIGID_DENSITY = 2500.0
YOUNG = 5.0e5
POISSON = 0.30
GRAVITY = 9.81
FRICTION = 0.30
NORMAL_STIFFNESS = 1.0e7
TANGENTIAL_STIFFNESS = 5.0e6
FEM_CONTACT_NORMAL_STIFFNESS = 2.0e8
FEM_CONTACT_TANGENTIAL_STIFFNESS = 1.0e8
DT = 1.0e-6
SIMULATION_TIME = 1.50
GATE_OPEN_TIME = 0.50
DIAGNOSTIC_INTERVAL = 0.01
OUTPUT_INTERVAL = 0.10
JACOBIAN_CHECK_STRIDE = 1000
NEIGHBOR_CHECK_STRIDE = 10
OUTLET_Z = 0.30
CHUTE_BOTTOM_Z = 0.26
FLOOR_TOP = 0.075
CATCHER_TOP_Z = 0.32
PRODUCTION_CATCHER_WIDTH_X = 0.42
PRODUCTION_CATCHER_WIDTH_Y = 0.30
DEFAULT_CATCHER_WIDTH_X = PRODUCTION_CATCHER_WIDTH_X
DEFAULT_CATCHER_WIDTH_Y = PRODUCTION_CATCHER_WIDTH_Y
IRREGULAR_LEVELSET_SPACING = 5.0
IRREGULAR_FEM_SIZE_FRACTION = 0.12
MINIMUM_TETRA_MEAN_RATIO = 0.05
SOFT_CONTACT_THICKNESS = 1.0e-3
SOFT_CONTACT_VERLET_MULTIPLIER = 8.0
SOFT_PT_CAPACITY = 32768
SOFT_EE_CAPACITY = 131072
SOFT_HISTORY_CAPACITY = 524288
SOFT_PT_CAPACITY_PER_BODY = 32768
SOFT_EE_CAPACITY_PER_BODY = 65536
SOFT_HISTORY_CAPACITY_PER_BODY = 2048
SOFT_FILTERED_PT_CAPACITY_PER_BODY = 24576
SOFT_FILTERED_EE_CAPACITY_PER_BODY = 65536
DEM_BODY_COORDINATION_MINIMUM = 512
DEM_WALL_CAPACITY_FACTOR = 4
FEDEM_CONTACT_COORDINATION = 1024
CAPACITY_HEADROOM_FACTOR = 4.0
SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR = 2.0
CHECKPOINT_PHASE = "step_boundary_before_force_assembly"
CONTACT_NORMAL_DAMPING = 0.20
CONTACT_TANGENTIAL_DAMPING = 0.10
RIGID_LOCAL_DAMPING = 0.05
FEM_DAMPING = 0.02

PROFILE_STAGE_ORDER = (
    "step_reset",
    "contact_search",
    "dem_contact_force",
    "fem_lsdem_wall_contact",
    "fem_fem_contact",
    "rigid_body_integration",
    "fem_constitutive_update",
    "fem_internal_force",
    "fem_nodal_integration",
    "fem_boundary_update",
    "fem_jacobian_check",
)


def _restart_grid_mapping(
    checkpoint_step: int,
    checkpoint_dt: float,
    stored_time: float,
    target_dt: float,
) -> tuple[float, int]:
    """Map a checkpoint boundary to a new time grid without accumulated drift.

    ``current_time`` is advanced by repeated floating-point additions during a
    run, whereas ``current_step`` and ``delta`` identify the intended boundary
    exactly.  A restart therefore uses ``step * delta`` as the authoritative
    physical time after first checking that the stored scalar is consistent
    with that boundary.
    """

    if checkpoint_step < 0 or checkpoint_dt <= 0.0 or target_dt <= 0.0:
        raise ValueError("restart step must be nonnegative and timesteps positive")
    grid_time = checkpoint_step * checkpoint_dt
    stored_tolerance = max(
        1.0e-12,
        min(
            0.25 * checkpoint_dt,
            1.0e-9 * max(abs(grid_time), 1.0),
        ),
    )
    if not math.isclose(
        stored_time,
        grid_time,
        rel_tol=0.0,
        abs_tol=stored_tolerance,
    ):
        raise ValueError("checkpoint time is inconsistent with its saved step and timestep")
    remapped_step = int(round(grid_time / target_dt))
    mapping_tolerance = max(
        1.0e-14,
        64.0 * np.finfo(np.float64).eps * max(abs(grid_time), 1.0),
    )
    if not math.isclose(
        remapped_step * target_dt,
        grid_time,
        rel_tol=0.0,
        abs_tol=mapping_tolerance,
    ):
        raise ValueError("restart time is not representable by the requested timestep")
    return grid_time, remapped_step


_CUMULATIVE_CONTACT_MEMBERS = {
    "elastic_energy",
    "friction_energy",
    "damp_energy",
}
_ENERGY_SENSITIVE_CONTACT_MEMBERS = {
    "kn",
    "ks",
    "emod",
    "kratio",
    "effective_young",
    "effective_shear",
    "thickness",
}


def _struct_parameter_snapshot(field) -> dict[str, np.ndarray]:
    """Copy constitutive parameters while excluding cumulative work fields."""

    return {
        name: np.array(value, copy=True)
        for name, value in field.to_numpy().items()
        if name not in _CUMULATIVE_CONTACT_MEMBERS
    }


def _restore_struct_parameters(
    field,
    requested: dict[str, np.ndarray],
) -> tuple[bool, bool]:
    """Restore requested parameters and report any/stored-energy changes."""

    changed = False
    energy_sensitive_changed = False
    for member, values in requested.items():
        current = getattr(field, member).to_numpy()
        member_changed = not np.array_equal(current, values)
        changed = changed or member_changed
        energy_sensitive_changed = energy_sensitive_changed or (
            member_changed and member in _ENERGY_SENSITIVE_CONTACT_MEMBERS
        )
        getattr(field, member).from_numpy(values)
    return changed, energy_sensitive_changed


class SynchronizedStageProfiler:
    """Sample GPU stage latency without synchronizing the production hot path."""

    def __init__(self, taichi, sample_steps):
        self.taichi = taichi
        self.sample_steps = tuple(sorted(int(step) for step in sample_steps))
        self.records = {name: [] for name in PROFILE_STAGE_ORDER}
        self.raw_records = {name: [] for name in PROFILE_STAGE_ORDER}
        overhead = []
        for _ in range(16):
            self.taichi.sync()
            started = time.perf_counter()
            self.taichi.sync()
            overhead.append(time.perf_counter() - started)
        self.synchronization_overhead_seconds = float(np.median(overhead))

    def measure(self, name, function, *args, **kwargs):
        if name not in self.records:
            self.records[name] = []
        self.taichi.sync()
        started = time.perf_counter()
        result = function(*args, **kwargs)
        self.taichi.sync()
        elapsed = time.perf_counter() - started
        self.raw_records.setdefault(name, []).append(elapsed)
        self.records[name].append(max(elapsed - self.synchronization_overhead_seconds, 0.0))
        return result

    def summary(self):
        stage_rows = []
        total = sum(sum(values) for values in self.records.values())
        for name in PROFILE_STAGE_ORDER:
            values = np.asarray(self.records.get(name, ()), dtype=np.float64)
            if values.size == 0:
                continue
            stage_rows.append(
                {
                    "name": name,
                    "samples": int(values.size),
                    "total_seconds": float(np.sum(values)),
                    "mean_seconds": float(np.mean(values)),
                    "median_seconds": float(np.median(values)),
                    "p95_seconds": float(np.quantile(values, 0.95)),
                    "share": float(np.sum(values) / total) if total > 0.0 else 0.0,
                }
            )
        return {
            "method": (
                "synchronized latency on uniformly sampled steps; ordinary "
                "steps retain the asynchronous GPU hot path"
            ),
            "sample_steps": list(self.sample_steps),
            "sample_count": len(self.sample_steps),
            "empty_sync_median_seconds": self.synchronization_overhead_seconds,
            "overhead_correction": ("one measured empty-sync median is subtracted from every " "sampled stage latency"),
            "raw_profiled_stage_seconds": float(sum(sum(values) for values in self.raw_records.values())),
            "profiled_stage_seconds": float(total),
            "stages": stage_rows,
        }


class SimulationTimingLedger:
    """Partition synchronized wall time without touching the ordinary hot path.

    ``switch`` synchronizes only when the run crosses a timing boundary, such
    as native output, a scalar diagnostic sample, or a deliberately profiled
    step.  Consecutive solver steps therefore retain the normal asynchronous
    GPU execution path.
    """

    def __init__(self, taichi):
        self.taichi = taichi
        self.current = None
        self.last_transition = None
        self.wall_started = None
        self.wall_seconds = None
        self.seconds = {}

    def start(self, category):
        self.taichi.sync()
        now = time.perf_counter()
        self.current = str(category)
        self.last_transition = now
        self.wall_started = now

    def switch(self, category):
        category = str(category)
        if category == self.current:
            return
        self.taichi.sync()
        now = time.perf_counter()
        self.seconds[self.current] = self.seconds.get(self.current, 0.0) + (now - self.last_transition)
        self.current = category
        self.last_transition = now

    def finish(self):
        self.taichi.sync()
        now = time.perf_counter()
        self.seconds[self.current] = self.seconds.get(self.current, 0.0) + (now - self.last_transition)
        self.wall_seconds = now - self.wall_started
        self.current = None
        self.last_transition = None
        return dict(self.seconds)


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _process_gpu_memory_gb() -> float | None:
    """Return this process' allocated NVIDIA memory in decimal GB."""

    output = _command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if output == "unavailable":
        return None
    used_mib = 0.0
    matched = False
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            process_id = int(fields[0])
            memory_mib = float(fields[1])
        except ValueError:
            continue
        if process_id == os.getpid():
            used_mib += memory_mib
            matched = True
    return used_mib * (1024.0**2) / 1.0e9 if matched else None


def _prompt() -> str:
    return PROMPT


def _facet_wall_specs(
    include_gate=False,
    catcher_width_x=DEFAULT_CATCHER_WIDTH_X,
    catcher_width_y=DEFAULT_CATCHER_WIDTH_Y,
):
    """Return a four-sided hopper, outlet chute, separate catcher, and gate."""

    x0, x1 = 0.0, 0.60
    y0, y1 = 0.0, 0.40
    z_top = 0.83
    outlet_x0, outlet_x1 = 0.23, 0.37
    outlet_y0, outlet_y1 = 0.12, 0.28
    catcher_x0 = 0.5 * (x0 + x1 - catcher_width_x)
    catcher_x1 = 0.5 * (x0 + x1 + catcher_width_x)
    catcher_y0 = 0.5 * (y0 + y1 - catcher_width_y)
    catcher_y1 = 0.5 * (y0 + y1 + catcher_width_y)

    side_normal = np.asarray([z_top - OUTLET_Z, 0.0, outlet_x0 - x0])
    side_normal /= np.linalg.norm(side_normal)
    end_normal = np.asarray([0.0, z_top - OUTLET_Z, outlet_y0 - y0])
    end_normal /= np.linalg.norm(end_normal)
    walls = (
        (
            "left_hopper",
            ([x0, y0, z_top], [x0, y1, z_top], [outlet_x0, outlet_y1, OUTLET_Z], [outlet_x0, outlet_y0, OUTLET_Z]),
            side_normal,
        ),
        (
            "right_hopper",
            ([x1, y0, z_top], [outlet_x1, outlet_y0, OUTLET_Z], [outlet_x1, outlet_y1, OUTLET_Z], [x1, y1, z_top]),
            np.asarray([-side_normal[0], 0.0, side_normal[2]]),
        ),
        (
            "front_hopper",
            ([x0, y0, z_top], [outlet_x0, outlet_y0, OUTLET_Z], [outlet_x1, outlet_y0, OUTLET_Z], [x1, y0, z_top]),
            end_normal,
        ),
        (
            "back_hopper",
            ([x0, y1, z_top], [x1, y1, z_top], [outlet_x1, outlet_y1, OUTLET_Z], [outlet_x0, outlet_y1, OUTLET_Z]),
            np.asarray([0.0, -end_normal[1], end_normal[2]]),
        ),
        (
            "left_chute",
            (
                [outlet_x0, outlet_y0, CHUTE_BOTTOM_Z],
                [outlet_x0, outlet_y1, CHUTE_BOTTOM_Z],
                [outlet_x0, outlet_y1, OUTLET_Z],
                [outlet_x0, outlet_y0, OUTLET_Z],
            ),
            np.asarray([1.0, 0.0, 0.0]),
        ),
        (
            "right_chute",
            (
                [outlet_x1, outlet_y0, CHUTE_BOTTOM_Z],
                [outlet_x1, outlet_y0, OUTLET_Z],
                [outlet_x1, outlet_y1, OUTLET_Z],
                [outlet_x1, outlet_y1, CHUTE_BOTTOM_Z],
            ),
            np.asarray([-1.0, 0.0, 0.0]),
        ),
        (
            "front_chute",
            (
                [outlet_x0, outlet_y0, CHUTE_BOTTOM_Z],
                [outlet_x0, outlet_y0, OUTLET_Z],
                [outlet_x1, outlet_y0, OUTLET_Z],
                [outlet_x1, outlet_y0, CHUTE_BOTTOM_Z],
            ),
            np.asarray([0.0, 1.0, 0.0]),
        ),
        (
            "back_chute",
            (
                [outlet_x0, outlet_y1, CHUTE_BOTTOM_Z],
                [outlet_x1, outlet_y1, CHUTE_BOTTOM_Z],
                [outlet_x1, outlet_y1, OUTLET_Z],
                [outlet_x0, outlet_y1, OUTLET_Z],
            ),
            np.asarray([0.0, -1.0, 0.0]),
        ),
        (
            "catcher_floor",
            (
                [catcher_x0, catcher_y0, FLOOR_TOP],
                [catcher_x0, catcher_y1, FLOOR_TOP],
                [catcher_x1, catcher_y1, FLOOR_TOP],
                [catcher_x1, catcher_y0, FLOOR_TOP],
            ),
            np.asarray([0.0, 0.0, 1.0]),
        ),
        (
            "catcher_left",
            (
                [catcher_x0, catcher_y0, FLOOR_TOP],
                [catcher_x0, catcher_y0, CATCHER_TOP_Z],
                [catcher_x0, catcher_y1, CATCHER_TOP_Z],
                [catcher_x0, catcher_y1, FLOOR_TOP],
            ),
            np.asarray([1.0, 0.0, 0.0]),
        ),
        (
            "catcher_right",
            (
                [catcher_x1, catcher_y0, FLOOR_TOP],
                [catcher_x1, catcher_y1, FLOOR_TOP],
                [catcher_x1, catcher_y1, CATCHER_TOP_Z],
                [catcher_x1, catcher_y0, CATCHER_TOP_Z],
            ),
            np.asarray([-1.0, 0.0, 0.0]),
        ),
        (
            "catcher_front",
            (
                [catcher_x0, catcher_y0, FLOOR_TOP],
                [catcher_x1, catcher_y0, FLOOR_TOP],
                [catcher_x1, catcher_y0, CATCHER_TOP_Z],
                [catcher_x0, catcher_y0, CATCHER_TOP_Z],
            ),
            np.asarray([0.0, 1.0, 0.0]),
        ),
        (
            "catcher_back",
            (
                [catcher_x0, catcher_y1, FLOOR_TOP],
                [catcher_x0, catcher_y1, CATCHER_TOP_Z],
                [catcher_x1, catcher_y1, CATCHER_TOP_Z],
                [catcher_x1, catcher_y1, FLOOR_TOP],
            ),
            np.asarray([0.0, -1.0, 0.0]),
        ),
    )
    if not include_gate:
        return walls
    return walls + (
        (
            "outlet_gate",
            (
                [outlet_x0, outlet_y0, OUTLET_Z],
                [outlet_x1, outlet_y0, OUTLET_Z],
                [outlet_x1, outlet_y1, OUTLET_Z],
                [outlet_x0, outlet_y1, OUTLET_Z],
            ),
            np.asarray([0.0, 0.0, 1.0]),
        ),
    )


def _dem_wall_geometry(coupling, wall_facet_ids):
    scene = coupling.dem.scene
    count = int(scene.wallNum[0])
    geometry = {}
    vertex1 = scene.wall.vertice1.to_numpy()[:count]
    vertex2 = scene.wall.vertice2.to_numpy()[:count]
    vertex3 = scene.wall.vertice3.to_numpy()[:count]
    normals = scene.wall.norm.to_numpy()[:count]
    for name, ids in wall_facet_ids.items():
        points = np.concatenate((vertex1[ids], vertex2[ids], vertex3[ids]), axis=0)
        geometry[name] = {
            "facet_ids": ids,
            "bounds": np.stack((points.min(axis=0), points.max(axis=0))).tolist(),
            "normals": normals[ids].tolist(),
        }
    return geometry


def _validate_realized_wall_geometry(coupling, wall_facet_ids, centers, radii, wall_specs=None):
    wall_specs = _facet_wall_specs() if wall_specs is None else wall_specs
    realized = _dem_wall_geometry(coupling, wall_facet_ids)
    expected = {
        name: np.stack((np.min(vertices, axis=0), np.max(vertices, axis=0))) for name, vertices, _ in wall_specs
    }
    expected_normals = {name: normal for name, _, normal in wall_specs}
    for name, record in realized.items():
        if len(record["facet_ids"]) != 2:
            raise RuntimeError(f"DEM polygon wall '{name}' did not create two facets")
        if not np.allclose(record["bounds"], expected[name], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"DEM facet wall '{name}' has incorrect bounds")
        if not np.allclose(record["normals"], expected_normals[name], rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"DEM facet wall '{name}' has an outward normal")
    required_wall_top = float(np.max(centers[:, 2] + radii) + 0.01)
    hopper_names = (
        "left_hopper",
        "right_hopper",
        "front_hopper",
        "back_hopper",
    )
    for name in hopper_names:
        if realized[name]["bounds"][1][2] < required_wall_top:
            raise RuntimeError(f"DEM facet wall '{name}' ends below the required confinement height")
        vertices = next(
            np.asarray(vertices, dtype=np.float64) for wall_name, vertices, _ in wall_specs if wall_name == name
        )
        normal = expected_normals[name]
        initial_gap = (centers - vertices[0]) @ normal - radii
        if float(np.min(initial_gap)) <= SOFT_CONTACT_THICKNESS:
            raise RuntimeError(f"initial particles intersect or enter the contact layer of '{name}'")
    chute_bottom = min(
        realized[name]["bounds"][0][2] for name in ("left_chute", "right_chute", "front_chute", "back_chute")
    )
    catcher_top = max(
        realized[name]["bounds"][1][2]
        for name in (
            "catcher_left",
            "catcher_right",
            "catcher_front",
            "catcher_back",
        )
    )
    if not math.isclose(chute_bottom, CHUTE_BOTTOM_Z, abs_tol=1.0e-12):
        raise RuntimeError("the vertical outlet chute has an incorrect length")
    if not catcher_top > OUTLET_Z:
        raise RuntimeError("the catcher walls must end above the hopper outlet")
    upper_vertices = np.vstack(
        [
            np.asarray(vertices, dtype=np.float64)
            for name, vertices, _ in wall_specs
            if name.endswith("_hopper") or name.endswith("_chute")
        ]
    )
    catcher_vertices = np.vstack(
        [np.asarray(vertices, dtype=np.float64) for name, vertices, _ in wall_specs if name.startswith("catcher_")]
    )
    minimum_separation = float(
        np.min(
            np.linalg.norm(
                upper_vertices[:, None, :] - catcher_vertices[None, :, :],
                axis=2,
            )
        )
    )
    if minimum_separation <= 1.0e-12:
        raise RuntimeError("the hopper and catcher must be separate wall bodies")
    return realized, required_wall_top


def _initial_packing(rng, packing="production"):
    if packing == "validation":
        if PARTICLE_COUNT != 8 or SOFT_COUNT != 4:
            raise ValueError("the gated-funnel validation packing requires four FEM and " "four LSDEM particles")
        # A sparse single-file column exercises closed-gate settling and
        # two-phase release without the accidental four-grain arch that can
        # dominate an eight-particle sample.  Even and odd heights are assigned
        # to the FEM and LSDEM phases so each phase spans the column.
        levels = np.linspace(0.43, 0.745, PARTICLE_COUNT)
        lateral_x = np.asarray((-0.012, 0.010, -0.008, 0.012, -0.010, 0.008, -0.012, 0.010))
        lateral_y = np.asarray((0.006, -0.008, 0.010, -0.006, 0.008, -0.010, 0.006, -0.008))
        ordered = np.column_stack((0.30 + lateral_x, 0.20 + lateral_y, levels))
        centers = np.vstack((ordered[::2], ordered[1::2]))
        radii = rng.uniform(0.0160, 0.0170, PARTICLE_COUNT)
        return centers, radii
    if packing == "scaling":
        if PARTICLE_COUNT > 1200:
            raise ValueError("the common scaling packing supports at most 1200 particles")
        # Every scaling run selects from the same 1200-site lattice and uses
        # the same radius distribution.  Particle count is therefore the only
        # discretization-size variable in the timing study.
        axes = (
            np.linspace(0.167, 0.433, 12),
            np.linspace(0.095, 0.305, 10),
            np.linspace(0.49, 0.80, 10),
        )
        radius_range = (0.0090, 0.0095)
        jitter = 2.0e-4
    elif PARTICLE_COUNT <= 36:
        axes = (
            np.linspace(0.21, 0.39, 4),
            np.linspace(0.12, 0.28, 3),
            np.linspace(0.59, 0.75, 3),
        )
        radius_range = (0.0160, 0.0180)
        jitter = 8.0e-4
    elif PARTICLE_COUNT == 800:
        # Ten-by-eight-by-ten placement fills the upper hopper without an
        # initial overlap.  The lowest layer respects the sloping facet walls.
        axes = (
            np.linspace(0.167, 0.433, 10),
            np.linspace(0.095, 0.305, 8),
            np.linspace(0.49, 0.80, 10),
        )
        # Maintain more than the 1 mm FEM contact thickness after the maximum
        # opposing lattice jitter, so the energy ledger starts contact-free.
        radius_range = (0.0130, 0.0135)
        jitter = 4.0e-4
    else:
        raise ValueError("the validated funnel packings currently support 1--36 or 800 particles")
    candidates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if candidates.shape[0] < PARTICLE_COUNT:
        raise RuntimeError("the requested funnel packing exceeds its lattice capacity")
    centers = candidates[rng.permutation(candidates.shape[0])[:PARTICLE_COUNT]]
    centers += rng.uniform(-jitter, jitter, centers.shape)
    radii = rng.uniform(*radius_range, PARTICLE_COUNT)
    return centers, radii


def _soft_reference_surface(output: Path) -> Path:
    """Create the smooth irregular FEM reference used by the paper case."""

    import trimesh

    path = output / "mesh" / "spherical_harmonic_surface.stl"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    surface = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    directions = np.asarray(surface.vertices, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    x, y, z = directions.T
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    radius = (
        0.94
        + 0.13 * np.sin(theta) ** 2 * np.cos(3.0 * phi)
        + 0.08 * np.sin(2.0 * theta) * np.sin(2.0 * phi)
        + 0.05 * (3.0 * np.cos(theta) ** 2 - 1.0)
    )
    surface.vertices[:] = directions * radius[:, None]
    surface.fix_normals()
    if not surface.is_watertight:
        raise RuntimeError("the generated FEM reference surface is not watertight")
    surface.export(path)
    return path


def _build(
    gt,
    output: Path,
    dt: float,
    young: float,
    dem_contact_model: str,
    dem_engine: str,
    fem_contact_normal_stiffness: float,
    fem_contact_tangential_stiffness: float,
    soft_contact_verlet_multiplier: float,
    contact_work_mode: str,
    packing: str,
    include_gate: bool,
    initial_z_shift: float,
    catcher_width_x: float,
    catcher_width_y: float,
):
    wall_specs = _facet_wall_specs(
        include_gate=include_gate,
        catcher_width_x=catcher_width_x,
        catcher_width_y=catcher_width_y,
    )
    rng = np.random.default_rng(SEED)
    centers, radii = _initial_packing(rng, packing)
    centers[:, 2] += initial_z_shift
    orientations = rng.uniform(0.0, 360.0, (PARTICLE_COUNT, 3))
    soft_shapes = ["smooth_spherical_harmonic"] * SOFT_COUNT
    rigid_shapes = ["angular_sand"] * RIGID_COUNT

    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[0.60, 0.40, 0.85],
        scheme="LSDEM",
        engine=dem_engine,
        search="BVH",
        gravity=[0.0, 0.0, -GRAVITY],
        track_energy=True,
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 2,
            "max_rigid_body_number": RIGID_COUNT,
            "max_rigid_template_number": 1,
            "levelset_grid_number": max(100000, 256 * RIGID_COUNT),
            "surface_node_number": 512,
            "max_sphere_number": 0,
            "max_clump_number": 0,
            "max_plane_number": 0,
            "max_facet_number": 2 * len(wall_specs),
            # A DEM body can see at most every other rigid body.  The 512
            # slot lower bound therefore covers the complete 400-body
            # topology instead of relying on a dense-flow estimate.
            "body_coordination_number": max(DEM_BODY_COORDINATION_MINIMUM, RIGID_COUNT),
            # Every rigid body can query every triangular wall facet; retain
            # four times that topological maximum for implementation slack.
            "wall_coordination_number": max(64, DEM_WALL_CAPACITY_FACTOR * 2 * len(wall_specs)),
            # Surface nodes near a triangulated funnel edge can retain
            # several adjacent facets simultaneously.  The old implicit
            # default of two discarded valid candidates even though the
            # body-level wall list had ample capacity.
            "point_coordination_number": [16, 32],
            "verlet_distance_multiplier": [0.1, 0.1],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    for material_id, density in ((0, RIGID_DENSITY), (1, RIGID_DENSITY)):
        dem.add_attribute(
            materialID=material_id,
            attribute={
                "Density": density,
                "ForceLocalDamping": RIGID_LOCAL_DAMPING,
                "TorqueLocalDamping": RIGID_LOCAL_DAMPING,
            },
        )
    irregular_shape = gt.polyhedron(file=str(RIGID_IRREGULAR_SURFACE)).grids(space=IRREGULAR_LEVELSET_SPACING, extent=3)
    dem.add_template(
        {
            "Name": "funnel_irregular_grain",
            "Object": irregular_shape,
            "WriteFile": False,
        }
    )
    # One device upload/kernel replaces hundreds of Python-side body kernels.
    # The batch API consumes radians; the sampled orientations above are kept
    # in degrees for the FEM placement helper and converted only here.
    dem.create_body_batch(
        {
            "BodyType": "RigidBody",
            "Template": {
                "Name": "funnel_irregular_grain",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoints": centers[SOFT_COUNT:],
                "BoundingRadii": radii[SOFT_COUNT:],
                "BodyOrientationsRadians": np.deg2rad(orientations[SOFT_COUNT:]),
                "InitialVelocity": [0.0, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
            },
        }
    )
    walls_to_create = []
    for wall_id, (name, vertices, normal) in enumerate(wall_specs):
        walls_to_create.append(
            {
                "WallID": wall_id,
                "WallType": "Facet",
                "WallShape": "Polygon",
                "MaterialID": 1,
                "WallVertice": {f"vertice{index + 1}": np.asarray(vertex) for index, vertex in enumerate(vertices)},
                "OuterNormal": np.asarray(normal),
            }
        )
    dem.add_wall(body=walls_to_create)
    dem.set_static_wall(True)
    wall_id_values = dem.scene.wall.wallID.to_numpy()[: int(dem.scene.wallNum[0])]
    wall_facet_ids = {
        name: np.flatnonzero(wall_id_values == wall_id).tolist() for wall_id, (name, _, _) in enumerate(wall_specs)
    }
    model_name = "Energy Conserving Model" if dem_contact_model == "energy-conserving" else "Linear Model"
    dem.choose_contact_model(model_name, model_name)
    dem.select_save_data(
        particle=True,
        surface=True,
        wall=True,
        particle_particle_contact=True,
        particle_wall_contact=True,
    )
    dem_contact_property = {
        "NormalStiffness": NORMAL_STIFFNESS,
        "TangentialStiffness": TANGENTIAL_STIFFNESS,
        "Friction": FRICTION,
        "NormalViscousDamping": CONTACT_NORMAL_DAMPING,
        "TangentialViscousDamping": CONTACT_TANGENTIAL_DAMPING,
    }
    if dem_contact_model == "energy-conserving":
        # theta=2 recovers the same quadratic penalty potential as the linear
        # spring and isolates implementation/ledger differences.
        dem_contact_property["FreeParameter"] = 2.0
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property=dem_contact_property,
        dType="particle-particle",
    )
    dem.add_property(
        materialID1=0,
        materialID2=1,
        property=dem_contact_property,
        dType="particle-wall",
    )

    fem = gt.FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", log=False)
    soft_reference_surface = _soft_reference_surface(output)
    irregular_reference, mesh_quality = tetrahedralize_irregular_reference(
        soft_reference_surface,
        maximum_size_fraction=IRREGULAR_FEM_SIZE_FRACTION,
    )
    if mesh_quality["minimum_tetra_mean_ratio"] < MINIMUM_TETRA_MEAN_RATIO:
        raise RuntimeError(
            "irregular FEM reference mesh failed the tetrahedron quality gate: "
            f"{mesh_quality['minimum_tetra_mean_ratio']:.6g} < "
            f"{MINIMUM_TETRA_MEAN_RATIO:.6g}"
        )
    soft_ranges = []
    node_offset = 0
    soft_meshes = []
    for body_id, (center, radius, orientation) in enumerate(
        zip(centers[:SOFT_COUNT], radii[:SOFT_COUNT], orientations[:SOFT_COUNT])
    ):
        mesh = place_reference(
            irregular_reference,
            center,
            float(radius),
            orientation,
            name=f"funnel_soft_spherical_harmonic_{body_id:03d}",
        )
        soft_meshes.append(mesh)
        soft_ranges.append((node_offset, node_offset + mesh.number_of_nodes))
        node_offset += mesh.number_of_nodes
    from src.fem.generator import FEMMesh

    fem.add_soft_particle(FEMMesh.concatenate(soft_meshes))
    fem.add_material(
        "NeoHookean",
        density=SOFT_DENSITY,
        young_modulus=young,
        poisson_ratio=POISSON,
    )
    fem.add_soft_particle_contact(
        "Linear",
        search="BVH",
        verlet_distance_multiplier=soft_contact_verlet_multiplier,
        ContactThickness=SOFT_CONTACT_THICKNESS,
        max_point_triangle_pairs=max(SOFT_PT_CAPACITY, SOFT_COUNT * SOFT_PT_CAPACITY_PER_BODY),
        max_edge_edge_pairs=max(SOFT_EE_CAPACITY, SOFT_COUNT * SOFT_EE_CAPACITY_PER_BODY),
        max_filtered_point_triangle_pairs=max(
            SOFT_PT_CAPACITY,
            SOFT_COUNT * SOFT_FILTERED_PT_CAPACITY_PER_BODY,
        ),
        max_filtered_edge_edge_pairs=max(
            SOFT_EE_CAPACITY,
            SOFT_COUNT * SOFT_FILTERED_EE_CAPACITY_PER_BODY,
        ),
        contact_history_capacity=max(SOFT_HISTORY_CAPACITY, SOFT_COUNT * SOFT_HISTORY_CAPACITY_PER_BODY),
        NormalStiffness=fem_contact_normal_stiffness,
        TangentialStiffness=fem_contact_tangential_stiffness,
        Friction=FRICTION,
        NormalViscousDamping=CONTACT_NORMAL_DAMPING,
        TangentialViscousDamping=CONTACT_TANGENTIAL_DAMPING,
    )

    coupling = gt.FEDEM(dem=dem, fem=fem, log=False)
    coupling.set_configuration(
        domain=[0.60, 0.40, 0.85],
        gravity=[0.0, 0.0, -GRAVITY],
        search="BVH",
        contact_work_mode=contact_work_mode,
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": SIMULATION_TIME,
            "SaveInterval": OUTPUT_INTERVAL,
            "SavePath": str(output / "native"),
            "damping": FEM_DAMPING,
        },
        log=False,
    )
    coupling.select_save_data(contact=True, checkpoint=True)
    coupling.add_surface(body_ids=list(range(SOFT_COUNT)))
    coupling.memory_allocate(
        {
            # This is the per-rigid-body BVH face buffer and can overflow
            # before the compact global contact list.  Dense discharge uses
            # a fourfold increase over the earlier 256-candidate setting.
            "contact_coordination_number": FEDEM_CONTACT_COORDINATION,
            "max_contact_pairs": max(262144, PARTICLE_COUNT * 8192),
            "max_filtered_contact_pairs": max(1048576, PARTICLE_COUNT * 1024),
            "max_contact_history_pairs": max(65536, PARTICLE_COUNT * 128),
            "max_fem_wall_pairs": max(262144, PARTICLE_COUNT * 1024),
            "max_fem_wall_history_pairs": max(32768, PARTICLE_COUNT * 64),
            "max_levelset_cell_pairs": max(1048576, PARTICLE_COUNT * 32768),
            "verlet_distance_multiplier": 0.12,
        }
    )
    coupling.choose_contact_model("Linear")
    for body_id in range(SOFT_COUNT):
        for material_id in (0, 1):
            coupling.add_property(
                DEMmaterial=material_id,
                FEMbody=body_id,
                property={
                    "NormalStiffness": fem_contact_normal_stiffness,
                    "TangentialStiffness": fem_contact_tangential_stiffness,
                    "Friction": FRICTION,
                    "NormalViscousDamping": CONTACT_NORMAL_DAMPING,
                    "TangentialViscousDamping": CONTACT_TANGENTIAL_DAMPING,
                },
            )
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    # The DEM BVH builds both particle and wall batches once. On later Verlet
    # updates only the moving-particle prefix is rebuilt/refitted; the fixed
    # facet batch is retained.
    dem_neighbor = coupling.dem.contactor.neighbor
    coupling.dem_fixed_wall_broad_phase_reuse = bool(
        coupling.dem.sims.static_wall and dem_neighbor.place_wall_to_cells.__name__ == "no_operation"
    )
    if not coupling.dem_fixed_wall_broad_phase_reuse:
        raise RuntimeError("fixed DEM wall BVH was not retained after initialization")
    coupling.check_critical_timestep()
    (
        coupling.realized_wall_geometry,
        coupling.required_wall_top,
    ) = _validate_realized_wall_geometry(coupling, wall_facet_ids, centers, radii, wall_specs)
    return (
        coupling,
        centers,
        radii,
        soft_shapes,
        rigid_shapes,
        soft_ranges,
        wall_facet_ids,
        mesh_quality,
        irregular_reference.number_of_nodes,
        irregular_reference.number_of_cells,
    )


def _body_centers(coupling, soft_ranges):
    fem_engine = coupling.enginer.fem_engine
    positions = fem_engine.state.position.to_numpy()
    mass = fem_engine.state.mass.to_numpy()
    soft = []
    for start, end in soft_ranges:
        local_mass = mass[start:end]
        soft.append(np.sum(local_mass[:, None] * positions[start:end], axis=0) / np.sum(local_mass))
    rigid = coupling.dem.scene.rigid.mass_center.to_numpy()[:RIGID_COUNT]
    return np.asarray(soft), np.asarray(rigid)


def _cross_candidate_local_counts(neighbor, rigid_count):
    """Return per-query BVH candidate counts and the allocation mode."""
    if hasattr(neighbor, "particle_offsets"):
        offsets = neighbor.particle_offsets.to_numpy()[: rigid_count + 1]
        return np.diff(offsets), "fixed-per-rigid"
    broad_phase = getattr(neighbor, "broad_phase", None)
    if broad_phase is not None and hasattr(broad_phase, "node_prefix"):
        query_count = int(broad_phase.vertex_count)
        offsets = broad_phase.node_prefix.to_numpy()[: query_count + 1]
        return np.diff(offsets), "count-prefix-compact"
    return np.empty(0, dtype=np.int64), "global-only"


def _dem_point_wall_candidate_counts(coupling):
    """Return total and per-surface-node LSDEM--facet candidates."""
    neighbor = coupling.dem.contactor.neighbor
    surface_count = int(coupling.dem.scene.surfaceNum[0])
    prefix = getattr(neighbor, "lsparticle_wall", None)
    candidates = getattr(neighbor, "potential_list_point_wall", None)
    if prefix is None or candidates is None or surface_count <= 0:
        return 0, 0, 0
    offsets = prefix.to_numpy()[: surface_count + 1]
    return (
        int(offsets[-1]),
        int(np.max(np.diff(offsets), initial=0)),
        int(candidates.shape[0]),
    )


def _configured_contact_sum(contact, name, slots):
    values = np.asarray(contact[name]).reshape(-1)[np.asarray(slots)]
    if not np.isfinite(values).all():
        raise FloatingPointError(f"configured DEM contact energy '{name}' is non-finite")
    return float(np.sum(values))


def _dem_contact_energy_output(contact):
    """Return an explicit zero ledger for an intentionally absent channel."""
    if contact is None or bool(getattr(contact, "null_model", False)) or getattr(contact, "surfaceProps", None) is None:
        zeros = np.zeros(2, dtype=np.float64)
        return {
            "elastic_energy": zeros,
            "friction_energy": zeros,
            "viscous_damping_energy": zeros,
        }
    return contact.get_contact_energy_output()


def _completed_dissipation_snapshot(coupling):
    """Read cumulative work before forces for the next step are evaluated."""
    dem_pp = _dem_contact_energy_output(coupling.dem.contactor.physpp)
    dem_pw = _dem_contact_energy_output(coupling.dem.contactor.physpw)
    rigid = coupling.dem.scene.rigid
    soft = coupling.enginer.fem_engine.soft_particle_contact.energy_diagnostics()
    cross = coupling.contactor.energy_diagnostics()
    return {
        "dem_pp_friction": -_configured_contact_sum(dem_pp, "friction_energy", [0]),
        "dem_pw_friction": -_configured_contact_sum(dem_pw, "friction_energy", [1]),
        "dem_pp_viscous": -_configured_contact_sum(dem_pp, "viscous_damping_energy", [0]),
        "dem_pw_viscous": -_configured_contact_sum(dem_pw, "viscous_damping_energy", [1]),
        "rigid_local_damping": float(-np.sum(rigid.damp_energy.to_numpy()[:RIGID_COUNT])),
        "soft_friction": float(soft["friction_dissipation"]),
        "soft_damping": float(soft["damping_dissipation"]),
        "cross_friction": float(cross["friction_dissipation"]),
        "cross_damping": float(cross["damping_dissipation"]),
        "fem_damping": float(coupling.enginer.fem_engine.state.damping_dissipation[None]),
    }


def _energy_snapshot(
    coupling,
    positions,
    velocity,
    mass,
    strain_energy,
    completed_dissipation=None,
):
    """Return a complete stored-energy and cumulative-dissipation ledger."""
    gravity = np.asarray(coupling.dem.sims.gravity, dtype=np.float64)
    rigid = coupling.dem.scene.rigid
    rigid_mass = rigid.m.to_numpy()[:RIGID_COUNT]
    rigid_center = rigid.mass_center.to_numpy()[:RIGID_COUNT]
    rigid_velocity = rigid.v.to_numpy()[:RIGID_COUNT]
    rigid_omega = rigid.w.to_numpy()[:RIGID_COUNT]
    rigid_angular_momentum = rigid.angmoment.to_numpy()[:RIGID_COUNT]

    soft_kinetic = float(0.5 * np.sum(mass[:, None] * velocity * velocity))
    rigid_translational = float(0.5 * np.sum(rigid_mass[:, None] * rigid_velocity * rigid_velocity))
    rigid_rotational = float(0.5 * np.sum(rigid_omega * rigid_angular_momentum))
    soft_potential = float(-np.sum(mass * (positions @ gravity)))
    rigid_potential = float(-np.sum(rigid_mass * (rigid_center @ gravity)))
    rigid_local_damping = float(-np.sum(rigid.damp_energy.to_numpy()[:RIGID_COUNT]))

    dem_pp_contact = _dem_contact_energy_output(coupling.dem.contactor.physpp)
    dem_pw_contact = _dem_contact_energy_output(coupling.dem.contactor.physpw)

    # PairingMapping(0,0)=0 for grain--grain and PairingMapping(0,1)=1 for
    # grain--facet contact.  Keeping the ledgers separate prevents a missing
    # particle--wall term from masquerading as numerical energy loss.
    dem_pp_elastic = _configured_contact_sum(dem_pp_contact, "elastic_energy", [0])
    dem_pw_elastic = _configured_contact_sum(dem_pw_contact, "elastic_energy", [1])
    dem_pp_friction = -_configured_contact_sum(dem_pp_contact, "friction_energy", [0])
    dem_pw_friction = -_configured_contact_sum(dem_pw_contact, "friction_energy", [1])
    dem_pp_viscous = -_configured_contact_sum(dem_pp_contact, "viscous_damping_energy", [0])
    dem_pw_viscous = -_configured_contact_sum(dem_pw_contact, "viscous_damping_energy", [1])
    dem_elastic = dem_pp_elastic + dem_pw_elastic
    dem_friction = dem_pp_friction + dem_pw_friction
    dem_viscous = dem_pp_viscous + dem_pw_viscous
    soft_contact = coupling.enginer.fem_engine.soft_particle_contact.energy_diagnostics()
    cross_contact = coupling.contactor.energy_diagnostics()
    fem_damping = float(coupling.enginer.fem_engine.state.damping_dissipation[None])
    if completed_dissipation is not None:
        dem_pp_friction = completed_dissipation["dem_pp_friction"]
        dem_pw_friction = completed_dissipation["dem_pw_friction"]
        dem_pp_viscous = completed_dissipation["dem_pp_viscous"]
        dem_pw_viscous = completed_dissipation["dem_pw_viscous"]
        rigid_local_damping = completed_dissipation["rigid_local_damping"]
        soft_contact["friction_dissipation"] = completed_dissipation["soft_friction"]
        soft_contact["damping_dissipation"] = completed_dissipation["soft_damping"]
        cross_contact["friction_dissipation"] = completed_dissipation["cross_friction"]
        cross_contact["damping_dissipation"] = completed_dissipation["cross_damping"]
        fem_damping = completed_dissipation["fem_damping"]
        dem_friction = dem_pp_friction + dem_pw_friction
        dem_viscous = dem_pp_viscous + dem_pw_viscous
    kinetic = soft_kinetic + rigid_translational + rigid_rotational
    potential = soft_potential + rigid_potential
    contact = dem_elastic + soft_contact["elastic_energy"] + cross_contact["elastic_energy"]
    friction = dem_friction + soft_contact["friction_dissipation"] + cross_contact["friction_dissipation"]
    damping = (
        dem_viscous
        + rigid_local_damping
        + fem_damping
        + soft_contact["damping_dissipation"]
        + cross_contact["damping_dissipation"]
    )
    wall_removal = float(getattr(coupling, "removed_wall_contact_energy", 0.0))
    mechanical = kinetic + potential + float(strain_energy) + contact
    total = mechanical + friction + damping + wall_removal
    return {
        # Preserve the old names so restart histories and existing
        # postprocessors remain readable. rigid_kinetic_energy is translation.
        "soft_kinetic_energy": soft_kinetic,
        "rigid_kinetic_energy": rigid_translational,
        "rigid_translational_kinetic_energy": rigid_translational,
        "rigid_rotational_kinetic_energy": rigid_rotational,
        "fem_strain_energy": float(strain_energy),
        "soft_gravitational_potential_energy": soft_potential,
        "rigid_gravitational_potential_energy": rigid_potential,
        "dem_contact_elastic_energy": dem_elastic,
        "dem_contact_friction_dissipation": dem_friction,
        "dem_contact_viscous_dissipation": dem_viscous,
        "dem_particle_particle_elastic_energy": dem_pp_elastic,
        "dem_particle_wall_elastic_energy": dem_pw_elastic,
        "dem_particle_particle_friction_dissipation": dem_pp_friction,
        "dem_particle_wall_friction_dissipation": dem_pw_friction,
        "dem_particle_particle_viscous_dissipation": dem_pp_viscous,
        "dem_particle_wall_viscous_dissipation": dem_pw_viscous,
        "rigid_local_damping_dissipation": rigid_local_damping,
        "fem_fem_contact_elastic_energy": soft_contact["elastic_energy"],
        "fem_fem_contact_friction_dissipation": soft_contact["friction_dissipation"],
        "fem_fem_contact_damping_dissipation": soft_contact["damping_dissipation"],
        "fem_lsdem_contact_elastic_energy": cross_contact["elastic_energy"],
        "fem_lsdem_contact_friction_dissipation": cross_contact["friction_dissipation"],
        "fem_lsdem_contact_damping_dissipation": cross_contact["damping_dissipation"],
        "fem_bulk_damping_dissipation": fem_damping,
        # Aggregate columns are the curves reported in the paper.
        "kinetic_energy": kinetic,
        "gravitational_potential_energy": potential,
        "friction_dissipation": friction,
        "damping_dissipation": damping,
        "wall_removal_energy": wall_removal,
        "contact_energy": contact,
        "mechanical_energy": mechanical,
        "total_energy": total,
        # Backward-compatible aliases for earlier postprocessors.
        "tracked_mechanical_energy": mechanical,
        "tracked_dem_dissipation": dem_friction + dem_viscous + rigid_local_damping,
    }


def _write_history(path: Path, history) -> None:
    """Atomically keep the scalar ledger aligned with restart frames."""
    if not history:
        return
    fieldnames = []
    seen = set()
    for row in history:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _write_center_history(path: Path, frames) -> None:
    """Atomically persist sparse particle centers for paper snapshots."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "frames": frames}) + os.linesep,
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _existing_native_checkpoint_steps(root: Path) -> list[int]:
    """Read only checkpoints physically present in this output directory."""

    steps = []
    for path in sorted(root.glob("FEDEMCheckpoint*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        steps.append(int(metadata["scalar_state"]["coupling.current_step"]))
    return sorted(set(steps))


def _clear_fresh_run_output(output: Path) -> None:
    """Remove stale trajectory files before a non-restart production run."""

    native_root = output / "native"
    if native_root.is_dir():
        for path in native_root.rglob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
    for name in (
        "history.csv",
        "center_history.json",
        "config.json",
        "metrics.json",
        "state.json",
        "performance.json",
        "initial_output.json",
    ):
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _select_profile_steps(
    start_step,
    final_step,
    sample_count,
    output_stride,
    jacobian_stride,
    diagnostic_stride,
    warmup_steps,
):
    first_candidate = min(
        final_step - 1,
        start_step + max(1, int(warmup_steps)),
    )
    if sample_count <= 0 or final_step - first_candidate <= 1:
        return ()
    available = final_step - first_candidate
    target = min(int(sample_count), available)
    candidates = np.linspace(
        first_candidate,
        final_step - 1,
        min(available, max(target * 8, target)),
        dtype=np.int64,
    )
    eligible = []
    for raw in candidates:
        step = int(raw)
        if step in eligible:
            continue
        # Output downloads and scheduled Jacobian reductions are diagnostics,
        # not part of the ordinary explicit step represented by the pie chart.
        if step % output_stride == 0:
            continue
        if step % diagnostic_stride == 0:
            continue
        if (step + 1) % jacobian_stride == 0:
            continue
        eligible.append(step)
    if len(eligible) < target:
        raise RuntimeError(
            f"only {len(eligible)} ordinary steps are available for " f"{target} requested stage-profile samples"
        )
    selected_indices = np.linspace(0, len(eligible) - 1, target, dtype=np.int64)
    return tuple(eligible[int(index)] for index in selected_indices)


def main() -> int:
    global PARTICLE_COUNT, SOFT_COUNT, RIGID_COUNT
    global FRICTION, CONTACT_NORMAL_DAMPING, CONTACT_TANGENTIAL_DAMPING
    global RIGID_LOCAL_DAMPING, FEM_DAMPING, SIMULATION_TIME, OUTPUT_INTERVAL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--default-fp", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--young", type=float, default=YOUNG)
    parser.add_argument("--time", type=float, default=SIMULATION_TIME)
    parser.add_argument(
        "--gate-open-time",
        type=float,
        default=GATE_OPEN_TIME,
        help=(
            "Keep a DEM facet plate across the outlet until this physical "
            "time, then deactivate it at a complete explicit-step boundary."
        ),
    )
    parser.add_argument(
        "--closed-gate",
        action="store_true",
        help=(
            "Include the outlet facet plate and keep it fixed for the full "
            "run; intended for stationary-boundary contact-work tests."
        ),
    )
    parser.add_argument(
        "--output-interval",
        type=float,
        default=OUTPUT_INTERVAL,
        help="Physical interval between saved frames and scalar diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-interval",
        type=float,
        default=DIAGNOSTIC_INTERVAL,
        help=(
            "Optional physical interval between scalar energy samples. "
            "Native VTK/NPZ/checkpoint output remains controlled by "
            "--output-interval."
        ),
    )
    parser.add_argument(
        "--flush-diagnostic-history",
        action="store_true",
        help=(
            "Atomically flush scalar history at every diagnostic sample. "
            "This is intended for a short failure-localization run and does "
            "not add VTK or checkpoint output."
        ),
    )
    parser.add_argument("--particles", type=int, default=PARTICLE_COUNT)
    parser.add_argument(
        "--initial-z-shift",
        type=float,
        default=0.0,
        help=(
            "Translate the generated packing vertically before model "
            "construction; intended for short isolated diagnostics."
        ),
    )
    parser.add_argument(
        "--catcher-width-x",
        type=float,
        default=DEFAULT_CATCHER_WIDTH_X,
        help="Horizontal x extent of the separate lower receiving container.",
    )
    parser.add_argument(
        "--catcher-width-y",
        type=float,
        default=DEFAULT_CATCHER_WIDTH_Y,
        help="Horizontal y extent of the separate lower receiving container.",
    )
    parser.add_argument(
        "--soft-count",
        type=int,
        help="Number of FEM particles; defaults to half of --particles.",
    )
    parser.add_argument("--friction", type=float, default=FRICTION)
    parser.add_argument(
        "--fem-contact-normal-stiffness",
        type=float,
        default=FEM_CONTACT_NORMAL_STIFFNESS,
    )
    parser.add_argument(
        "--fem-contact-tangential-stiffness",
        type=float,
        default=FEM_CONTACT_TANGENTIAL_STIFFNESS,
    )
    parser.add_argument(
        "--soft-contact-verlet-distance-multiplier",
        type=float,
        default=SOFT_CONTACT_VERLET_MULTIPLIER,
        help=(
            "Verlet skin multiplier for FEM--FEM soft-particle contact. "
            "The 800-particle discharge uses 8.0 so a rebuild retains "
            "penetrating features during dense impact."
        ),
    )
    parser.add_argument(
        "--contact-normal-damping",
        type=float,
        default=CONTACT_NORMAL_DAMPING,
    )
    parser.add_argument(
        "--contact-tangential-damping",
        type=float,
        default=CONTACT_TANGENTIAL_DAMPING,
    )
    parser.add_argument("--rigid-local-damping", type=float, default=RIGID_LOCAL_DAMPING)
    parser.add_argument("--fem-damping", type=float, default=FEM_DAMPING)
    parser.add_argument(
        "--dem-contact-model",
        choices=("linear", "energy-conserving"),
        default="linear",
    )
    parser.add_argument(
        "--dem-engine",
        choices=("SymplecticEuler", "VelocityVerlet"),
        default="SymplecticEuler",
        help=(
            "Rigid-body integrator. SymplecticEuler applies the same current-"
            "configuration contact force used by the FEM update; the legacy "
            "VelocityVerlet path carries a previous-step acceleration."
        ),
    )
    parser.add_argument(
        "--contact-work-mode",
        choices=("Explicit",),
        default="Explicit",
        help="Explicit FEM--LSDEM/facet contact update.",
    )
    parser.add_argument(
        "--restart",
        type=Path,
        help="Resume from an exact FEDEM checkpoint after rebuilding the same model.",
    )
    parser.add_argument(
        "--debug-stage-sync",
        action="store_true",
        help="Synchronize CUDA after each coupled stage to localize asynchronous failures.",
    )
    parser.add_argument(
        "--debug-taichi",
        action="store_true",
        help="Enable Taichi device bounds and assertion checks for a failing screen.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Build the complete coupled problem, report the realized discretization, and exit.",
    )
    parser.add_argument(
        "--initial-output-only",
        action="store_true",
        help=(
            "Build the requested full-duration model, write its exact initial "
            "native frame and restart checkpoint, and exit without integration."
        ),
    )
    parser.add_argument(
        "--packing",
        choices=("production", "validation", "scaling"),
        default="production",
        help=(
            "Use the production packing, the sparse two-phase gated-funnel "
            "validation column, or the common 1200-site timing packing."
        ),
    )
    parser.add_argument(
        "--profile-stage-samples",
        type=int,
        default=0,
        help=(
            "Synchronize and time this many uniformly distributed steps by "
            "solver stage; zero leaves every step on the asynchronous hot path."
        ),
    )
    parser.add_argument(
        "--timing-warmup-steps",
        type=int,
        default=1,
        help=(
            "Exclude this many initial state-advancing steps from solver "
            "throughput so first-use kernel compilation is not reported as "
            "steady GPU performance."
        ),
    )
    parser.add_argument(
        "--jacobian-check-stride",
        type=int,
        default=JACOBIAN_CHECK_STRIDE,
        help=(
            "Check the dynamic FEM minimum Jacobian every N steps and at "
            "saved/final states; this avoids a host scalar synchronization "
            "on every explicit step."
        ),
    )
    parser.add_argument(
        "--neighbor-check-stride",
        type=int,
        default=NEIGHBOR_CHECK_STRIDE,
        help=(
            "Evaluate device Verlet displacement flags every N steps. Contact "
            "geometry and forces still update every step; only the host flag "
            "read and optional broad-phase rebuild are decimated."
        ),
    )
    args = parser.parse_args()
    if args.young <= 0.0:
        parser.error("--young must be positive")
    if args.jacobian_check_stride <= 0:
        parser.error("--jacobian-check-stride must be positive")
    if args.neighbor_check_stride <= 0:
        parser.error("--neighbor-check-stride must be positive")
    if args.preflight and args.initial_output_only:
        parser.error("--preflight and --initial-output-only are mutually exclusive")
    if args.restart is not None and args.initial_output_only:
        parser.error("--restart cannot be combined with --initial-output-only")
    if args.particles <= 1:
        parser.error("--particles must exceed one")
    if not math.isfinite(args.initial_z_shift):
        parser.error("--initial-z-shift must be finite")
    if not (0.14 < args.catcher_width_x < 0.60):
        parser.error("--catcher-width-x must exceed the outlet and remain inside the domain")
    if not (0.16 < args.catcher_width_y < 0.40):
        parser.error("--catcher-width-y must exceed the outlet and remain inside the domain")
    if args.output_interval <= 0.0:
        parser.error("--output-interval must be positive")
    if args.diagnostic_interval is not None and args.diagnostic_interval <= 0.0:
        parser.error("--diagnostic-interval must be positive")
    if args.gate_open_time is not None and not (0.0 < args.gate_open_time < args.time):
        parser.error("--gate-open-time must lie strictly inside the simulation")
    if args.closed_gate and args.gate_open_time is not None:
        parser.error("--closed-gate and --gate-open-time are mutually exclusive")
    if args.profile_stage_samples < 0:
        parser.error("--profile-stage-samples must be nonnegative")
    if args.timing_warmup_steps < 0:
        parser.error("--timing-warmup-steps must be nonnegative")
    if args.packing == "scaling" and args.particles > 1200:
        parser.error("--packing scaling supports at most 1200 particles")
    soft_count = args.soft_count
    if soft_count is None:
        soft_count = (args.particles + 1) // 2
    if not 1 <= soft_count < args.particles:
        parser.error("--soft-count must be between one and particles-1")
    if args.friction < 0.0:
        parser.error("--friction must be nonnegative")
    if (
        min(
            args.fem_contact_normal_stiffness,
            args.fem_contact_tangential_stiffness,
            args.soft_contact_verlet_distance_multiplier,
        )
        <= 0.0
    ):
        parser.error("FEM contact stiffnesses and Verlet multiplier must be positive")
    if (
        min(
            args.contact_normal_damping,
            args.contact_tangential_damping,
            args.rigid_local_damping,
            args.fem_damping,
        )
        < 0.0
    ):
        parser.error("damping parameters must be nonnegative")
    PARTICLE_COUNT = args.particles
    SOFT_COUNT = soft_count
    RIGID_COUNT = PARTICLE_COUNT - SOFT_COUNT
    FRICTION = args.friction
    CONTACT_NORMAL_DAMPING = args.contact_normal_damping
    CONTACT_TANGENTIAL_DAMPING = args.contact_tangential_damping
    RIGID_LOCAL_DAMPING = args.rigid_local_damping
    FEM_DAMPING = args.fem_damping
    SIMULATION_TIME = args.time
    OUTPUT_INTERVAL = args.output_interval
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["GEOTAICHI_REAL_DTYPE"] = args.default_fp

    import taichi as ti
    import geotaichi as gt

    gt.init(
        arch=args.arch,
        default_fp=args.default_fp,
        log=False,
        debug=args.debug_taichi,
        offline_cache=False,
    )
    started = time.perf_counter()
    (
        coupling,
        initial_centers,
        radii,
        soft_shapes,
        rigid_shapes,
        soft_ranges,
        wall_facet_ids,
        mesh_quality,
        nodes_per_soft_particle,
        elements_per_soft_particle,
    ) = _build(
        gt,
        output,
        args.dt,
        args.young,
        args.dem_contact_model,
        args.dem_engine,
        args.fem_contact_normal_stiffness,
        args.fem_contact_tangential_stiffness,
        args.soft_contact_verlet_distance_multiplier,
        args.contact_work_mode,
        args.packing,
        args.closed_gate or args.gate_open_time is not None,
        args.initial_z_shift,
        args.catcher_width_x,
        args.catcher_width_y,
    )
    soft_property_field = coupling.fem.engine.soft_particle_contact.properties
    coupling.checkpoint_phase = CHECKPOINT_PHASE
    cross_property_field = coupling.contactor.model.surface_properties
    requested_soft_properties = {
        name: np.array(value, copy=True) for name, value in soft_property_field.to_numpy().items()
    }
    requested_cross_properties = {
        name: np.array(value, copy=True) for name, value in cross_property_field.to_numpy().items()
    }
    dem_property_fields = tuple(
        contact.surfaceProps
        for contact in (
            coupling.dem.contactor.physpp,
            coupling.dem.contactor.physpw,
        )
        if contact is not None and getattr(contact, "surfaceProps", None) is not None
    )
    requested_dem_properties = tuple(_struct_parameter_snapshot(field) for field in dem_property_fields)
    dem_material_field = coupling.dem.scene.material
    requested_dem_material_damping = {
        member: np.array(getattr(dem_material_field, member).to_numpy(), copy=True) for member in ("fdamp", "tdamp")
    }
    target_restart_dt = float(coupling.sims.delta)
    restart_repair = None
    if args.restart is not None:
        restart_metadata = coupling.read_restart(args.restart.expanduser().resolve())
        if restart_metadata.get("checkpoint_phase") != CHECKPOINT_PHASE:
            raise ValueError(
                "funnel restart checkpoint was not captured at a complete "
                "step boundary; regenerate it with the current recorder"
            )
        # Checkpoints emitted by this benchmark are step-boundary states:
        # positions have been advanced to the saved time, while the next
        # reset/search/force phase has not started.  Restored DEM and FEM
        # candidate lists and histories are therefore already the exact
        # continuation state. Rebuilding them here would mutate that restored
        # state before the first resumed step.
        checkpoint_dt = float(restart_metadata["scalar_state"]["coupling.delta"])
        checkpoint_step = int(coupling.sims.current_step)
        stored_restart_time = float(coupling.sims.current_time)
        restart_time, remapped_step = _restart_grid_mapping(
            checkpoint_step,
            checkpoint_dt,
            stored_restart_time,
            target_restart_dt,
        )
        # Normalize every coupled clock to the exact saved step boundary.
        # This removes harmless repeated-addition drift without changing the
        # checkpointed positions, velocities, histories, or physical time.
        coupling.sims.current_time = restart_time
        coupling.dem.sims.current_time = restart_time
        coupling.dem.sims.CurrentTime[None] = restart_time
        coupling.fem.engine.time = restart_time
        coupling.sims.current_step = remapped_step
        coupling.dem.sims.current_step = remapped_step
        coupling.fem.engine.step_count = remapped_step
        if not math.isclose(
            checkpoint_dt,
            target_restart_dt,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            # A checkpoint may continue on a smaller explicit grid. State is
            # unchanged; only time-grid counters and device dt scalars move.
            coupling.sims.set_timestep(target_restart_dt)
            coupling.dem.sims.set_timestep(target_restart_dt)
            coupling.dem.sims.init_delta = target_restart_dt
            coupling.fem.engine.dt = target_restart_dt
            coupling.fem.engine.total_step = int(math.ceil(args.time / target_restart_dt))
            restart_repair = {
                "reason": ("the checkpoint continuation was remapped to the requested " "smaller explicit timestep"),
                "checkpoint_timestep": checkpoint_dt,
                "continued_timestep": target_restart_dt,
                "restart_time": restart_time,
                "remapped_start_step": remapped_step,
            }
        coupling.sims.time = float(args.time)
        coupling.dem.sims.time = float(args.time)
        coupling.fem.engine.total_step = int(math.ceil(args.time / target_restart_dt))
        reconfigure_timeline = getattr(
            coupling.fem.engine,
            "reconfigure_device_boundary_timeline",
            None,
        )
        if reconfigure_timeline is not None:
            reconfigure_timeline(
                target_restart_dt,
                coupling.fem.engine.total_step,
            )
        # Recorder checkpoints are captured before save_file increments the
        # frame counter.  Continue at the next index so restart never
        # overwrites the checkpoint it was loaded from.
        saved_print = int(restart_metadata["scalar_state"]["coupling.current_print"])
        coupling.sims.current_print = saved_print + 1
        coupling.dem.sims.current_print = saved_print + 1
        coupling.solver.last_save_time = coupling.sims.current_time
        property_changed, energy_sensitive_changed = _restore_struct_parameters(
            soft_property_field, requested_soft_properties
        )
        changed, sensitive = _restore_struct_parameters(cross_property_field, requested_cross_properties)
        property_changed = property_changed or changed
        energy_sensitive_changed = energy_sensitive_changed or sensitive
        for field, requested in zip(dem_property_fields, requested_dem_properties):
            changed, sensitive = _restore_struct_parameters(field, requested)
            property_changed = property_changed or changed
            energy_sensitive_changed = energy_sensitive_changed or sensitive
        for member, requested in requested_dem_material_damping.items():
            target = getattr(dem_material_field, member)
            current = target.to_numpy()
            property_changed = property_changed or not np.array_equal(current, requested)
            target.from_numpy(requested)
        if energy_sensitive_changed:
            soft_contact = coupling.fem.engine.soft_particle_contact
            soft_active = bool(
                np.any(soft_contact.pt_active.to_numpy()[: soft_contact.pt_count])
                or np.any(soft_contact.ee_active.to_numpy()[: soft_contact.ee_count])
            )
            cross_neighbor = coupling.contactor.neighbor
            cross_active = bool(np.any(cross_neighbor.contacts.active.to_numpy()[: cross_neighbor.contact_count]))
            wall_active = bool(
                coupling.contactor.wall_contacts is not None
                and np.any(coupling.contactor.wall_contacts.active.to_numpy()[: coupling.contactor.wall_contact_count])
            )
            dem_active = any(
                np.any(contact.cplist.normalOverlapActive.to_numpy())
                for contact in (
                    coupling.dem.contactor.physpp,
                    coupling.dem.contactor.physpw,
                )
                if contact is not None and getattr(contact, "cplist", None) is not None
            )
            if soft_active or cross_active or wall_active or dem_active:
                raise ValueError(
                    "contact stiffness cannot be changed at an active-contact "
                    "checkpoint; restart from a contact-free step boundary"
                )
            repair_event = {
                "reason": (
                    "the FEM soft-contact penetration guard rejected the " "lower penalty stiffness during dense impact"
                ),
                "restart_time": float(coupling.sims.current_time),
                "continued_fem_contact_normal_stiffness": (args.fem_contact_normal_stiffness),
                "continued_fem_contact_tangential_stiffness": (args.fem_contact_tangential_stiffness),
            }
            if restart_repair is None:
                restart_repair = repair_event
            else:
                restart_repair = {"events": [restart_repair, repair_event]}
    ti.sync()
    setup_seconds = time.perf_counter() - started
    setup_gpu_memory_gb = _process_gpu_memory_gb()
    engine = coupling.enginer
    effective_dt = float(coupling.sims.delta)
    steps = int(round(args.time / effective_dt))
    start_step = int(coupling.sims.current_step)
    gate_step = None
    gate_wall_id = None
    gate_facet_ids = wall_facet_ids.get("outlet_gate", [])
    if args.gate_open_time is not None:
        gate_step = int(round(args.gate_open_time / effective_dt))
        if not math.isclose(
            gate_step * effective_dt,
            args.gate_open_time,
            rel_tol=0.0,
            abs_tol=max(1.0e-12, 1.0e-8 * effective_dt),
        ):
            raise ValueError("gate opening time must coincide with an explicit-step boundary")
        gate_wall_id = next(index for index, name in enumerate(wall_facet_ids) if name == "outlet_gate")
    gate_opened = bool(gate_facet_ids and not np.any(coupling.dem.scene.wall.active.to_numpy()[gate_facet_ids]))
    if start_step > steps:
        raise ValueError(f"restart step {start_step} exceeds target step {steps}")
    if args.preflight:
        step_started = time.perf_counter()
        engine.reset_message()
        engine.update_verlet_tables()
        engine.system_resolve()
        internal_force = engine.fem_engine._assemble_internal_device(need_stiffness=False)
        engine.integration(
            internal_force=internal_force,
            update_diagnostics=False,
        )
        ti.sync()
        one_step_seconds = time.perf_counter() - step_started
        post_step_gpu_memory_gb = _process_gpu_memory_gb()
        preflight_gpu_memory_gb = (
            max(value for value in (setup_gpu_memory_gb, post_step_gpu_memory_gb) if value is not None)
            if any(value is not None for value in (setup_gpu_memory_gb, post_step_gpu_memory_gb))
            else None
        )
        positions = engine.fem_engine.state.position.to_numpy()
        velocity = engine.fem_engine.state.velocity.to_numpy()
        minimum_jacobian = float(engine.fem_engine._minimum_jacobian_ratio_device(engine.fem_engine.state.position))
        soft_contact_diagnostics = engine.fem_engine.soft_particle_contact.diagnostics()
        cross_neighbor = coupling.contactor.neighbor
        cross_local_counts, cross_allocation_mode = _cross_candidate_local_counts(cross_neighbor, RIGID_COUNT)
        cross_count = int(cross_neighbor.contact_count)
        cross_capacity = int(coupling.sims.max_contact_pairs)
        cross_local_peak = int(np.max(cross_local_counts, initial=0))
        cross_local_capacity = (
            int(coupling.sims.contact_coordination_number) if cross_allocation_mode == "fixed-per-rigid" else None
        )
        wall_count = int(coupling.contactor.wall_candidate_count)
        wall_capacity = int(coupling.contactor.wall_contact_count)
        dem_body_capacity = int(coupling.dem.sims.body_coordination_number)
        dem_wall_capacity = int(coupling.dem.sims.wall_coordination_number)
        (
            dem_point_wall_count,
            dem_point_wall_local_peak,
            dem_point_wall_capacity,
        ) = _dem_point_wall_candidate_counts(coupling)
        dem_point_wall_local_capacity = int(coupling.dem.sims.point_wall_coordination_number)

        def capacity_record(
            used,
            allocated,
            required_headroom=CAPACITY_HEADROOM_FACTOR,
        ):
            used = int(used)
            allocated = int(allocated)
            return {
                "used": used,
                "allocated": allocated,
                "utilization": used / allocated if allocated > 0 else math.inf,
                "headroom_factor": (allocated / max(used, 1) if allocated > 0 else 0.0),
                "passed": bool(allocated > 0 and used <= allocated / required_headroom),
            }

        capacity_audit = {
            "required_headroom_factor": CAPACITY_HEADROOM_FACTOR,
            "soft_filtered_required_headroom_factor": (SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR),
            "dem_body_neighbors": {
                "topological_maximum": max(RIGID_COUNT - 1, 0),
                "allocated_per_body": dem_body_capacity,
                "passed": dem_body_capacity >= max(RIGID_COUNT - 1, 1),
            },
            "dem_wall_neighbors": {
                "topological_maximum": int(coupling.dem.scene.wallNum[0]),
                "allocated_per_body": dem_wall_capacity,
                "headroom_factor": dem_wall_capacity / max(int(coupling.dem.scene.wallNum[0]), 1),
                "passed": dem_wall_capacity >= CAPACITY_HEADROOM_FACTOR * max(int(coupling.dem.scene.wallNum[0]), 1),
            },
            "dem_point_wall_candidates_per_surface_node": capacity_record(
                dem_point_wall_local_peak,
                dem_point_wall_local_capacity,
            ),
            "dem_point_wall_compact_pairs": capacity_record(
                dem_point_wall_count,
                dem_point_wall_capacity,
            ),
            "fedem_candidates_per_query": (
                capacity_record(cross_local_peak, cross_local_capacity)
                if cross_local_capacity is not None
                else {
                    "used": cross_local_peak,
                    "allocated": None,
                    "allocation_mode": cross_allocation_mode,
                    "bounded_by_compact_global_capacity": True,
                    "passed": True,
                }
            ),
            "fedem_compact_pairs": capacity_record(cross_count, cross_capacity),
            "fem_wall_candidates": capacity_record(wall_count, wall_capacity),
            "fem_fem_point_triangle": capacity_record(
                soft_contact_diagnostics["point_triangle_candidates"],
                soft_contact_diagnostics["point_triangle_capacity"],
                SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR,
            ),
            "fem_fem_edge_edge": capacity_record(
                soft_contact_diagnostics["edge_edge_candidates"],
                soft_contact_diagnostics["edge_edge_capacity"],
                SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR,
            ),
            "fem_fem_tangential_history": capacity_record(
                4 * soft_contact_diagnostics["history_entry_count"],
                soft_contact_diagnostics["history_capacity"],
            ),
        }
        capacity_audit["passed"] = all(
            item.get("passed", True) for item in capacity_audit.values() if isinstance(item, dict)
        )

        preflight = {
            "schema_version": 1,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "simulation_time": args.time,
            "steps": steps,
            "soft_particle_count": SOFT_COUNT,
            "rigid_particle_count": RIGID_COUNT,
            "particle_count": PARTICLE_COUNT,
            "initial_z_shift": args.initial_z_shift,
            "packing": args.packing,
            "fem_nodes_per_soft_particle": nodes_per_soft_particle,
            "fem_elements_per_soft_particle": elements_per_soft_particle,
            "fem_reference_mesh_quality": mesh_quality,
            "fem_nodes": int(engine.fem_engine.mesh.number_of_nodes),
            "fem_elements": int(engine.fem_engine.mesh.number_of_cells),
            "cross_search": coupling.sims.search,
            "dem_search": coupling.dem.sims.search,
            "setup_seconds": setup_seconds,
            "gpu_memory_used_gb": preflight_gpu_memory_gb,
            "gpu_memory_after_setup_gb": setup_gpu_memory_gb,
            "gpu_memory_after_first_step_gb": post_step_gpu_memory_gb,
            "gpu_memory_measurement": (
                "maximum nvidia-smi process allocation after setup and after "
                "one complete untimed step; converted from MiB to decimal GB"
            ),
            "one_step_seconds": one_step_seconds,
            "one_step_finite": bool(np.isfinite(positions).all() and np.isfinite(velocity).all()),
            "one_step_minimum_jacobian": minimum_jacobian,
            "one_step_cross_candidate_count": cross_count,
            "one_step_wall_candidate_count": wall_count,
            "one_step_dem_point_wall_candidate_count": dem_point_wall_count,
            "one_step_dem_point_wall_candidates_per_surface_node": (dem_point_wall_local_peak),
            "dem_fixed_wall_broad_phase_reuse": bool(coupling.dem_fixed_wall_broad_phase_reuse),
            "dense_surface_node_wall_pairs": int(coupling.patch.surface_vertex_count * coupling.contactor.wall_count),
            "one_step_soft_contact_diagnostics": soft_contact_diagnostics,
            "capacity_audit": capacity_audit,
            "realized_wall_geometry": coupling.realized_wall_geometry,
            "wall_facet_ids": wall_facet_ids,
            "gate_open_time": args.gate_open_time,
            "closed_gate": args.closed_gate,
            "gate_wall_id": gate_wall_id,
            "required_wall_top": coupling.required_wall_top,
            "contact_work_mode": coupling.sims.contact_work_mode,
        }
        preflight["passed"] = bool(
            preflight["one_step_finite"]
            and minimum_jacobian > 0.10
            and mesh_quality["minimum_tetra_mean_ratio"] >= MINIMUM_TETRA_MEAN_RATIO
            and int(coupling.dem.scene.wallNum[0]) == 2 * len(wall_facet_ids)
            and coupling.dem_fixed_wall_broad_phase_reuse
            and coupling.dem.sims.search == "BVH"
            and coupling.sims.search == "BVH"
            and capacity_audit["passed"]
        )
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + os.linesep, encoding="utf-8")
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["passed"] else 2
    if args.restart is None:
        _clear_fresh_run_output(output)
    progress_stride = max(1, steps // 10)
    output_stride = max(1, int(round(OUTPUT_INTERVAL / effective_dt)))
    # Fine scalar sampling can be enabled for contact-work diagnosis without
    # multiplying the much larger VTK/NPZ/checkpoint output.  Production runs
    # retain the device-resident interval by default.
    diagnostic_interval = OUTPUT_INTERVAL if args.diagnostic_interval is None else args.diagnostic_interval
    sample_stride = max(1, int(round(diagnostic_interval / effective_dt)))
    profile_steps = _select_profile_steps(
        start_step,
        steps,
        args.profile_stage_samples,
        output_stride,
        args.jacobian_check_stride,
        sample_stride,
        args.timing_warmup_steps,
    )
    stage_profiler = SynchronizedStageProfiler(ti, profile_steps) if profile_steps else None
    profile_step_set = set(profile_steps)
    history = []
    center_history = []
    if args.restart is not None and (output / "history.csv").is_file():
        with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = [
                row
                for row in csv.DictReader(stream)
                if float(row["time"]) < start_step * effective_dt - 0.5 * effective_dt
            ]
        center_history_path = output / "center_history.json"
        if center_history_path.is_file():
            center_history = [
                frame
                for frame in json.loads(center_history_path.read_text(encoding="utf-8"))["frames"]
                if float(frame["time"]) < start_step * effective_dt - 0.5 * effective_dt
            ]
    peak_cross_candidates = max(
        (int(float(row["cross_candidate_count"])) for row in history),
        default=0,
    )
    peak_cross_active = max(
        (int(float(row["cross_active_contact_count"])) for row in history),
        default=0,
    )
    peak_cross_local_candidates = 0
    peak_wall_candidates = 0
    peak_dem_point_wall_candidates = max(
        (int(float(row.get("dem_point_wall_candidate_count", 0))) for row in history),
        default=0,
    )
    peak_dem_point_wall_candidates_per_surface_node = max(
        (int(float(row.get("dem_point_wall_candidates_per_surface_node", 0))) for row in history),
        default=0,
    )
    peak_soft_point_triangle_candidates = 0
    peak_soft_edge_edge_candidates = 0
    peak_soft_contact_history_entries = max(
        (int(float(row.get("soft_contact_history_entry_count", 0))) for row in history),
        default=0,
    )
    cross_capacity = int(coupling.sims.max_contact_pairs)
    _, cross_allocation_mode = _cross_candidate_local_counts(coupling.contactor.neighbor, RIGID_COUNT)
    cross_local_capacity = (
        int(coupling.sims.contact_coordination_number) if cross_allocation_mode == "fixed-per-rigid" else None
    )
    wall_capacity = int(coupling.contactor.wall_contact_count)
    dem_point_wall_local_capacity = int(coupling.dem.sims.point_wall_coordination_number)
    _, _, dem_point_wall_capacity = _dem_point_wall_candidate_counts(coupling)
    soft_contact = engine.fem_engine.soft_particle_contact
    soft_point_triangle_capacity = int(soft_contact.pt_capacity)
    soft_edge_edge_capacity = int(soft_contact.ee_capacity)
    soft_contact_history_capacity = int(soft_contact.history_capacity)
    minimum_jacobian = min(
        (float(row["minimum_jacobian"]) for row in history),
        default=float(engine.minimum_jacobian),
    )
    finite = True
    # A fresh run must emit a complete step-zero frame even if a stale or
    # concurrently created checkpoint is present.  A checkpoint alone is not
    # evidence that the FEM/LSDEM/contact artifact families were recorded.
    saved_steps = (
        _existing_native_checkpoint_steps(output / "native" / "checkpoints") if args.restart is not None else []
    )
    initial_soft = np.asarray(initial_centers[:SOFT_COUNT])
    initial_rigid = np.asarray(initial_centers[SOFT_COUNT:])

    warmup_stop_step = min(
        steps,
        start_step + int(args.timing_warmup_steps),
    )
    timing = SimulationTimingLedger(ti)
    timing.start("warmup" if start_step < warmup_stop_step else "solver_compute")
    timed_solver_steps = 0
    warmup_executed_steps = 0
    profiled_executed_steps = 0
    for step in range(start_step, steps + 1):
        time_value = min(step * effective_dt, args.time)
        active_profiler = stage_profiler if step in profile_step_set else None
        if step == steps:
            base_timing_category = "final_state_evaluation"
        elif active_profiler is not None:
            base_timing_category = "stage_profile"
        elif step < warmup_stop_step:
            base_timing_category = "warmup"
        else:
            base_timing_category = "solver_compute"
        timing.switch(base_timing_category)
        if gate_step is not None and step >= gate_step and not gate_opened:
            timing.switch("model_event")
            removed_energy = coupling.deactivate_wall(gate_wall_id, account_contact_energy=True)
            gate_opened = True
            print(
                json.dumps(
                    {
                        "case": "mixed_funnel",
                        "event": "outlet_gate_deactivated",
                        "step": step,
                        "time": time_value,
                        "removed_wall_contact_energy": removed_energy,
                    }
                ),
                flush=True,
            )
            timing.switch(base_timing_category)
        sample_now = step % sample_stride == 0 or step == steps
        # Contact resolution evaluates dissipative work for the upcoming
        # integration.  Capture cumulative terms first so the scalar ledger
        # remains aligned with the positions and velocities at time_value.
        completed_dissipation = None
        if sample_now:
            timing.switch("diagnostics")
            completed_dissipation = _completed_dissipation_snapshot(coupling)
            timing.switch(base_timing_category)
        output_now = step == 0 or step % output_stride == 0 or step == steps
        write_frame = output_now and (not saved_steps or saved_steps[-1] != step)
        if write_frame:
            # A restart must resume at a full-step boundary.  Native geometry
            # and contact output are sampled after force assembly below, but
            # the checkpoint is captured here, before reset/search/contact
            # history is advanced for this step.
            timing.switch("checkpoint_output")
            coupling.save_checkpoint(
                output / "native" / "checkpoints" / f"FEDEMCheckpoint{coupling.sims.current_print:06d}.npz"
            )
            timing.switch(base_timing_category)
        if active_profiler is None:
            engine.reset_message()
        else:
            active_profiler.measure("step_reset", engine.reset_message)
        if args.debug_stage_sync:
            ti.sync()
        check_neighbors = step % args.neighbor_check_stride == 0
        if active_profiler is None:
            engine.update_verlet_tables(check_rebuild=check_neighbors)
        else:
            active_profiler.measure(
                "contact_search",
                engine.update_verlet_tables,
                check_rebuild=check_neighbors,
            )
        if args.debug_stage_sync:
            ti.sync()
        try:
            engine.system_resolve(
                check_rebuild=check_neighbors,
                stage_profiler=active_profiler,
            )
        except RuntimeError as error:
            print(
                json.dumps(
                    {
                        "case": "mixed_funnel",
                        "failure_step": step,
                        "failure_time": time_value,
                        "last_saved_step": (saved_steps[-1] if saved_steps else None),
                        "last_saved_time": (saved_steps[-1] * effective_dt if saved_steps else None),
                        "error": str(error),
                    }
                ),
                flush=True,
            )
            raise
        if args.debug_stage_sync:
            ti.sync()
            if step % 100 == 0:
                soft_diag = engine.fem_engine.soft_particle_contact.diagnostics()
                print(
                    json.dumps(
                        {
                            "debug_step": step,
                            "debug_time": time_value,
                            "cross_candidates": int(coupling.contactor.neighbor.contact_count),
                            "soft_contact": soft_diag,
                        }
                    ),
                    flush=True,
                )
        sampled_internal_force = None
        if sample_now:
            fem_engine = engine.fem_engine
            # This assembly is the force used by the subsequent explicit
            # integration, so it remains solver work.  Only the additional
            # host downloads and scalar reductions below are diagnostics.
            sampled_internal_force = fem_engine._assemble_internal_device(need_stiffness=False)
            timing.switch("diagnostics")
            neighbor = coupling.contactor.neighbor
            count = int(neighbor.contact_count)
            local_counts, _ = _cross_candidate_local_counts(neighbor, RIGID_COUNT)
            peak_cross_local_candidates = max(
                peak_cross_local_candidates,
                int(np.max(local_counts, initial=0)),
            )
            peak_wall_candidates = max(
                peak_wall_candidates,
                int(coupling.contactor.wall_candidate_count),
            )
            (
                dem_point_wall_count,
                dem_point_wall_local_peak,
                _,
            ) = _dem_point_wall_candidate_counts(coupling)
            peak_dem_point_wall_candidates = max(
                peak_dem_point_wall_candidates,
                dem_point_wall_count,
            )
            peak_dem_point_wall_candidates_per_surface_node = max(
                peak_dem_point_wall_candidates_per_surface_node,
                dem_point_wall_local_peak,
            )
            soft_diagnostics = soft_contact.diagnostics()
            peak_soft_point_triangle_candidates = max(
                peak_soft_point_triangle_candidates,
                int(soft_diagnostics["point_triangle_candidates"]),
            )
            peak_soft_edge_edge_candidates = max(
                peak_soft_edge_edge_candidates,
                int(soft_diagnostics["edge_edge_candidates"]),
            )
            peak_soft_contact_history_entries = max(
                peak_soft_contact_history_entries,
                int(soft_diagnostics["history_entry_count"]),
            )
            active = neighbor.contacts.active.to_numpy()[:count].astype(bool)
            peak_cross_candidates = max(peak_cross_candidates, count)
            peak_cross_active = max(peak_cross_active, int(np.sum(active)))
            soft_centers, rigid_centers = _body_centers(coupling, soft_ranges)
            positions = fem_engine.state.position.to_numpy()
            velocity = fem_engine.state.velocity.to_numpy()
            mass = fem_engine.state.mass.to_numpy()
            strain_energy = float(fem_engine._internal_energy_device())
            energy = _energy_snapshot(
                coupling,
                positions,
                velocity,
                mass,
                strain_energy,
                completed_dissipation=completed_dissipation,
            )
            jacobian = float(fem_engine._minimum_jacobian_ratio_device(fem_engine.state.position))
            minimum_jacobian = min(minimum_jacobian, jacobian)
            soft_crossed = int(np.sum(soft_centers[:, 2] < OUTLET_Z))
            rigid_crossed = int(np.sum(rigid_centers[:, 2] < OUTLET_Z))
            history.append(
                {
                    "step": step,
                    "time": time_value,
                    "soft_mean_z": float(np.mean(soft_centers[:, 2])),
                    "rigid_mean_z": float(np.mean(rigid_centers[:, 2])),
                    "soft_outlet_crossings": soft_crossed,
                    "rigid_outlet_crossings": rigid_crossed,
                    "outlet_gate_open": int(gate_opened),
                    **energy,
                    "cross_candidate_count": count,
                    "cross_candidates_per_rigid": int(np.max(local_counts, initial=0)),
                    "cross_active_contact_count": int(np.sum(active)),
                    "fem_wall_candidate_count": int(coupling.contactor.wall_candidate_count),
                    "soft_point_triangle_candidate_count": int(soft_diagnostics["point_triangle_candidates"]),
                    "soft_edge_edge_candidate_count": int(soft_diagnostics["edge_edge_candidates"]),
                    "soft_contact_history_entry_count": int(soft_diagnostics["history_entry_count"]),
                    "dem_point_wall_candidate_count": dem_point_wall_count,
                    "dem_point_wall_candidates_per_surface_node": (dem_point_wall_local_peak),
                    "minimum_jacobian": jacobian,
                }
            )
            center_history.append(
                {
                    "step": step,
                    "time": time_value,
                    "soft_centers": soft_centers.tolist(),
                    "rigid_centers": rigid_centers.tolist(),
                }
            )
            finite = finite and bool(
                np.isfinite(positions).all()
                and np.isfinite(velocity).all()
                and np.isfinite(soft_centers).all()
                and np.isfinite(rigid_centers).all()
            )
            if args.flush_diagnostic_history:
                _write_history(output / "history.csv", history)
            timing.switch(base_timing_category)
        if write_frame:
            # The boundary checkpoint for this frame was already written.
            # Prevent the recorder from replacing it with the later
            # post-contact/pre-integration phase.
            checkpoint_enabled = coupling.sims.save_checkpoint
            coupling.sims.save_checkpoint = False
            timing.switch("native_output")
            try:
                coupling.save_data()
            finally:
                coupling.sims.save_checkpoint = checkpoint_enabled
            saved_steps.append(step)
            _write_history(output / "history.csv", history)
            _write_center_history(output / "center_history.json", center_history)
            timing.switch(base_timing_category)
        if args.initial_output_only and step == 0:
            ti.sync()
            output_evidence = collect_native_output(output / "native", 1)
            initial_output = {
                "schema_version": 1,
                "passed": bool(output_evidence["complete"]),
                "particle_count": PARTICLE_COUNT,
                "soft_particle_count": SOFT_COUNT,
                "rigid_particle_count": RIGID_COUNT,
                "requested_simulation_time": args.time,
                "requested_dt": args.dt,
                "effective_dt": effective_dt,
                "saved_step": 0,
                "saved_time": 0.0,
                "native_output": output_evidence,
            }
            (output / "initial_output.json").write_text(
                json.dumps(initial_output, indent=2) + os.linesep,
                encoding="utf-8",
            )
            print(json.dumps(initial_output, indent=2))
            return 0 if initial_output["passed"] else 2
        if step == steps:
            break
        if step > 0 and step % progress_stride == 0:
            timing.switch("host_reporting")
            print(json.dumps({"case": "mixed_funnel", "progress": step / steps}), flush=True)
            timing.switch(base_timing_category)
        engine.integration(
            internal_force=sampled_internal_force,
            update_diagnostics=False,
            check_jacobian=(
                (step + 1) % args.jacobian_check_stride == 0 or (step + 1) % output_stride == 0 or step + 1 == steps
            ),
            stage_profiler=active_profiler,
        )
        if args.debug_stage_sync:
            ti.sync()
        coupling.sims.current_time += effective_dt
        coupling.dem.sims.current_time += effective_dt
        engine.fem_engine.time = coupling.sims.current_time
        coupling.sims.current_step += 1
        coupling.dem.sims.current_step += 1
        engine.fem_engine.step_count += 1
        if base_timing_category == "solver_compute":
            timed_solver_steps += 1
        elif base_timing_category == "warmup":
            warmup_executed_steps += 1
        elif base_timing_category == "stage_profile":
            profiled_executed_steps += 1
    timing_seconds = timing.finish()
    loop_seconds = float(timing.wall_seconds)
    solver_compute_seconds = float(timing_seconds.get("solver_compute", 0.0))
    executed_steps = steps - start_step
    classified_steps = timed_solver_steps + warmup_executed_steps + profiled_executed_steps
    if classified_steps != executed_steps:
        raise RuntimeError(
            "performance timing classified " f"{classified_steps} of {executed_steps} state-advancing steps"
        )
    timing_accounting_residual = loop_seconds - sum(timing_seconds.values())
    post_loop_analysis_started = time.perf_counter()
    final_gpu_memory_gb = _process_gpu_memory_gb()
    output_evidence = collect_native_output(output / "native", len(saved_steps))
    final_soft, final_rigid = _body_centers(coupling, soft_ranges)
    domain_mask = lambda values: np.all(
        (values >= np.array([0.0, 0.0, 0.0])) & (values <= np.array([0.60, 0.40, 0.85])), axis=1
    )
    inside_count = int(np.sum(domain_mask(final_soft)) + np.sum(domain_mask(final_rigid)))
    final_soft_crossed = int(np.sum(final_soft[:, 2] < OUTLET_Z))
    final_rigid_crossed = int(np.sum(final_rigid[:, 2] < OUTLET_Z))
    total_energy = np.asarray([float(row["total_energy"]) for row in history], dtype=np.float64)
    initial_total_energy = float(total_energy[0])
    energy_scale = max(abs(initial_total_energy), 1.0e-15)
    relative_energy_residual = (total_energy - initial_total_energy) / energy_scale
    gates = {
        "finite_state": finite,
        "positive_jacobian": minimum_jacobian > 0.10,
        "soft_phase_passed_outlet": final_soft_crossed > 0,
        "rigid_phase_passed_outlet": final_rigid_crossed > 0,
        "mass_inside_domain": inside_count >= PARTICLE_COUNT,
        "mesh_quality": mesh_quality["minimum_tetra_mean_ratio"] >= MINIMUM_TETRA_MEAN_RATIO,
        "capacity": bool(
            peak_cross_candidates <= cross_capacity / CAPACITY_HEADROOM_FACTOR
            and (
                cross_local_capacity is None
                or peak_cross_local_candidates <= cross_local_capacity / CAPACITY_HEADROOM_FACTOR
            )
            and peak_wall_candidates <= wall_capacity / CAPACITY_HEADROOM_FACTOR
            and peak_dem_point_wall_candidates <= dem_point_wall_capacity / CAPACITY_HEADROOM_FACTOR
            and peak_dem_point_wall_candidates_per_surface_node
            <= dem_point_wall_local_capacity / CAPACITY_HEADROOM_FACTOR
            and peak_soft_point_triangle_candidates
            <= soft_point_triangle_capacity / SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR
            and peak_soft_edge_edge_candidates <= soft_edge_edge_capacity / SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR
            and 4 * peak_soft_contact_history_entries <= soft_contact_history_capacity
        ),
        "native_output": bool(output_evidence["complete"]),
        "outlet_gate_opened": (gate_opened if args.gate_open_time is not None else True),
        "gate_contact_energy_accounted": (
            float(getattr(coupling, "removed_wall_contact_energy", 0.0)) > 0.0
            if args.gate_open_time is not None
            else True
        ),
    }
    state = {
        "soft_shapes": soft_shapes,
        "rigid_shapes": rigid_shapes,
        "radii": radii.tolist(),
        "initial_soft_centers": initial_soft.tolist(),
        "initial_rigid_centers": initial_rigid.tolist(),
        "final_soft_centers": final_soft.tolist(),
        "final_rigid_centers": final_rigid.tolist(),
    }
    metrics = {
        "schema_version": 1,
        "passed": all(gates.values()),
        "gates": gates,
        "soft_outlet_crossings": final_soft_crossed,
        "rigid_outlet_crossings": final_rigid_crossed,
        "minimum_jacobian": minimum_jacobian,
        "maximum_cross_candidate_count": peak_cross_candidates,
        "maximum_cross_candidates_per_rigid": peak_cross_local_candidates,
        "maximum_cross_active_contact_count": peak_cross_active,
        "maximum_wall_candidate_count": peak_wall_candidates,
        "maximum_dem_point_wall_candidate_count": (peak_dem_point_wall_candidates),
        "maximum_dem_point_wall_candidates_per_surface_node": (peak_dem_point_wall_candidates_per_surface_node),
        "maximum_soft_point_triangle_candidate_count": (peak_soft_point_triangle_candidates),
        "maximum_soft_edge_edge_candidate_count": (peak_soft_edge_edge_candidates),
        "maximum_soft_contact_history_entry_count": (peak_soft_contact_history_entries),
        "capacity_audit": {
            "required_headroom_factor": CAPACITY_HEADROOM_FACTOR,
            "soft_filtered_required_headroom_factor": (SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR),
            "fedem_compact_pairs": {
                "peak": peak_cross_candidates,
                "allocated": cross_capacity,
            },
            "fedem_faces_per_rigid": {
                "peak": peak_cross_local_candidates,
                "allocated": cross_local_capacity,
                "allocation_mode": cross_allocation_mode,
                "bounded_by_compact_global_capacity": (cross_local_capacity is None),
            },
            "fem_wall_candidates": {
                "peak": peak_wall_candidates,
                "allocated": wall_capacity,
            },
            "dem_point_wall_compact_pairs": {
                "peak": peak_dem_point_wall_candidates,
                "allocated": dem_point_wall_capacity,
            },
            "dem_point_wall_candidates_per_surface_node": {
                "peak": peak_dem_point_wall_candidates_per_surface_node,
                "allocated": dem_point_wall_local_capacity,
            },
            "fem_fem_point_triangle": {
                "peak": peak_soft_point_triangle_candidates,
                "allocated": soft_point_triangle_capacity,
            },
            "fem_fem_edge_edge": {
                "peak": peak_soft_edge_edge_candidates,
                "allocated": soft_edge_edge_capacity,
            },
            "fem_fem_tangential_history": {
                "peak_entries": peak_soft_contact_history_entries,
                "required_hash_slots": 4 * peak_soft_contact_history_entries,
                "allocated_hash_slots": soft_contact_history_capacity,
            },
        },
        "inside_domain_count": inside_count,
        "fem_nodes_per_soft_particle": nodes_per_soft_particle,
        "fem_elements_per_soft_particle": elements_per_soft_particle,
        "fem_reference_mesh_quality": mesh_quality,
        "native_output": output_evidence,
        "initial_total_energy": initial_total_energy,
        "final_total_energy": float(total_energy[-1]),
        "maximum_absolute_relative_energy_residual": float(np.max(np.abs(relative_energy_residual))),
        "final_relative_energy_residual": float(relative_energy_residual[-1]),
        "removed_wall_contact_energy": float(getattr(coupling, "removed_wall_contact_energy", 0.0)),
    }
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": _prompt(),
        "units": "SI",
        "seed": SEED,
        "execution": {
            "arch": args.arch,
            "precision": args.default_fp,
            "taichi_version": list(ti.__version__),
            "gpu": (
                _command_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
                if args.arch == "gpu"
                else None
            ),
            "command": [sys.executable, *sys.argv],
            "restart_repair": restart_repair,
        },
        "parameters": {
            "particle_count": PARTICLE_COUNT,
            "initial_z_shift": args.initial_z_shift,
            "soft_particle_count": SOFT_COUNT,
            "rigid_particle_count": RIGID_COUNT,
            "packing": args.packing,
            "particle_geometry": {
                "fem": "smooth spherical-harmonic irregular grain",
                "lsdem": "angular sand grain",
            },
            "fem_reference_surface": str(_soft_reference_surface(output)),
            "lsdem_reference_surface": str(RIGID_IRREGULAR_SURFACE.relative_to(REPO_ROOT)),
            "fem_nodes_per_soft_particle": nodes_per_soft_particle,
            "fem_elements_per_soft_particle": elements_per_soft_particle,
            "fem_reference_mesh_quality": mesh_quality,
            "minimum_tetra_mean_ratio_gate": MINIMUM_TETRA_MEAN_RATIO,
            "soft_density": SOFT_DENSITY,
            "rigid_density": RIGID_DENSITY,
            "young_modulus": args.young,
            "poisson_ratio": POISSON,
            "gravity": GRAVITY,
            "friction": FRICTION,
            "dem_contact_model": args.dem_contact_model,
            "dem_engine": args.dem_engine,
            "contact_work_mode": args.contact_work_mode,
            "contact_normal_damping": CONTACT_NORMAL_DAMPING,
            "contact_tangential_damping": CONTACT_TANGENTIAL_DAMPING,
            "rigid_local_damping": RIGID_LOCAL_DAMPING,
            "fem_bulk_damping": FEM_DAMPING,
            "dem_normal_stiffness": NORMAL_STIFFNESS,
            "dem_tangential_stiffness": TANGENTIAL_STIFFNESS,
            "fem_contact_normal_stiffness": args.fem_contact_normal_stiffness,
            "fem_contact_tangential_stiffness": args.fem_contact_tangential_stiffness,
            "requested_dt": args.dt,
            "effective_dt": effective_dt,
            "simulation_time": args.time,
            "gate_open_time": args.gate_open_time,
            "closed_gate": args.closed_gate,
            "gate_wall_id": gate_wall_id,
            "gate_deactivated": gate_opened,
            "removed_wall_contact_energy": float(getattr(coupling, "removed_wall_contact_energy", 0.0)),
            "outlet_width_x": 0.14,
            "outlet_width_y": 0.16,
            "outlet_top_elevation": OUTLET_Z,
            "outlet_chute_bottom_elevation": CHUTE_BOTTOM_Z,
            "catcher_floor_elevation": FLOOR_TOP,
            "catcher_wall_top_elevation": CATCHER_TOP_Z,
            "catcher_width_x": args.catcher_width_x,
            "catcher_width_y": args.catcher_width_y,
            "search": coupling.sims.search,
            "soft_contact_thickness": SOFT_CONTACT_THICKNESS,
            "soft_contact_verlet_distance_multiplier": (args.soft_contact_verlet_distance_multiplier),
            "soft_point_triangle_capacity": max(
                SOFT_PT_CAPACITY,
                SOFT_COUNT * SOFT_PT_CAPACITY_PER_BODY,
            ),
            "soft_edge_edge_capacity": max(
                SOFT_EE_CAPACITY,
                SOFT_COUNT * SOFT_EE_CAPACITY_PER_BODY,
            ),
            "soft_contact_history_capacity": max(
                SOFT_HISTORY_CAPACITY,
                SOFT_COUNT * SOFT_HISTORY_CAPACITY_PER_BODY,
            ),
            "dem_body_coordination_number": int(coupling.dem.sims.body_coordination_number),
            "dem_wall_coordination_number": int(coupling.dem.sims.wall_coordination_number),
            "dem_point_wall_coordination_number": int(coupling.dem.sims.point_wall_coordination_number),
            "configured_fedem_contact_coordination_number": int(coupling.sims.contact_coordination_number),
            "active_fedem_fixed_per_query_capacity": cross_local_capacity,
            "fedem_candidate_allocation_mode": cross_allocation_mode,
            "fedem_max_contact_pairs": cross_capacity,
            "fem_wall_candidate_capacity": wall_capacity,
            "dem_point_wall_compact_capacity": dem_point_wall_capacity,
            "capacity_headroom_factor": CAPACITY_HEADROOM_FACTOR,
            "soft_filtered_capacity_headroom_factor": (SOFT_FILTERED_CAPACITY_HEADROOM_FACTOR),
            "wall_type": "DEM polygon facets",
            "wall_facet_count": sum(len(ids) for ids in wall_facet_ids.values()),
            "dem_fixed_wall_broad_phase_reuse": bool(coupling.dem_fixed_wall_broad_phase_reuse),
            "dem_search": coupling.dem.sims.search,
            "realized_wall_geometry": coupling.realized_wall_geometry,
            "required_wall_top": coupling.required_wall_top,
            "irregular_levelset_spacing": IRREGULAR_LEVELSET_SPACING,
            "irregular_fem_maximum_size_fraction": IRREGULAR_FEM_SIZE_FRACTION,
            "output_interval": OUTPUT_INTERVAL,
            "diagnostic_interval": diagnostic_interval,
            "checkpoint_output": True,
            "jacobian_check_stride": args.jacobian_check_stride,
            "neighbor_check_stride": args.neighbor_check_stride,
            "profile_stage_samples": len(profile_steps),
            "profile_stage_steps": list(profile_steps),
            "timing_warmup_steps": args.timing_warmup_steps,
            "precision": args.default_fp,
        },
    }
    post_loop_analysis_seconds = time.perf_counter() - post_loop_analysis_started
    performance = {
        "schema_version": 2,
        "host": platform.node(),
        "python": platform.python_version(),
        "taichi": list(ti.__version__),
        "gpu": _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "setup_seconds": setup_seconds,
        "simulation_loop_seconds": loop_seconds,
        "solver_compute_seconds": solver_compute_seconds,
        "post_loop_analysis_seconds": post_loop_analysis_seconds,
        "timing_scope": (
            "compute-only throughput after warm-up; setup/JIT, VTK/NPZ/"
            "checkpoint output, scalar diagnostics, final-state evaluation, "
            "model events, host reporting, and synchronized stage-profile "
            "steps are timed separately"
        ),
        "timing_breakdown_seconds": {
            "solver_compute": solver_compute_seconds,
            "warmup": float(timing_seconds.get("warmup", 0.0)),
            "checkpoint_output": float(timing_seconds.get("checkpoint_output", 0.0)),
            "native_output": float(timing_seconds.get("native_output", 0.0)),
            "diagnostics": float(timing_seconds.get("diagnostics", 0.0)),
            "final_state_evaluation": float(timing_seconds.get("final_state_evaluation", 0.0)),
            "model_event": float(timing_seconds.get("model_event", 0.0)),
            "host_reporting": float(timing_seconds.get("host_reporting", 0.0)),
            "stage_profile": float(timing_seconds.get("stage_profile", 0.0)),
            "accounting_residual": float(timing_accounting_residual),
        },
        "steps": steps,
        "saved_frame_count": len(saved_steps),
        "saved_steps": saved_steps,
        "fem_nodes": int(engine.fem_engine.mesh.number_of_nodes),
        "fem_elements": int(engine.fem_engine.mesh.number_of_cells),
        "particle_count": PARTICLE_COUNT,
        "soft_particle_count": SOFT_COUNT,
        "rigid_particle_count": RIGID_COUNT,
        "packing": args.packing,
        "start_step": start_step,
        "executed_steps": executed_steps,
        "timed_solver_steps": timed_solver_steps,
        "warmup_executed_steps": warmup_executed_steps,
        "profiled_executed_steps": profiled_executed_steps,
        "steps_per_second": (timed_solver_steps / solver_compute_seconds if solver_compute_seconds > 0.0 else 0.0),
        "particle_steps_per_second": (
            PARTICLE_COUNT * timed_solver_steps / solver_compute_seconds if solver_compute_seconds > 0.0 else 0.0
        ),
        "end_to_end_steps_per_second": (executed_steps / loop_seconds if loop_seconds > 0.0 else 0.0),
        "gpu_memory_used_gb": (
            max(value for value in (setup_gpu_memory_gb, final_gpu_memory_gb) if value is not None)
            if any(value is not None for value in (setup_gpu_memory_gb, final_gpu_memory_gb))
            else None
        ),
        "gpu_memory_measurement": (
            "maximum nvidia-smi process allocation measured after setup and "
            "after the simulation loop; converted from MiB to decimal GB and "
            "never polled inside the loop"
        ),
        "stage_profile": (stage_profiler.summary() if stage_profiler is not None else None),
    }
    metadata_output_started = time.perf_counter()
    for filename, payload in (
        ("config.json", config),
        ("metrics.json", metrics),
        ("state.json", state),
    ):
        (output / filename).write_text(
            json.dumps(payload, indent=2) + os.linesep,
            encoding="utf-8",
        )
    _write_history(output / "history.csv", history)
    _write_center_history(output / "center_history.json", center_history)
    performance["post_loop_metadata_output_seconds"] = time.perf_counter() - metadata_output_started
    performance["post_loop_metadata_scope"] = (
        "config, metrics, state, scalar history, and center history; the " "performance.json self-write is not included"
    )
    (output / "performance.json").write_text(
        json.dumps(performance, indent=2) + os.linesep,
        encoding="utf-8",
    )
    print(json.dumps({"passed": metrics["passed"], "metrics": metrics}, indent=2))
    return 0 if metrics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
