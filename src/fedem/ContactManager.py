"""Lifecycle manager for FEM--DEM, FEM--LSDEM, and affine IPC contact."""

from __future__ import annotations

import numpy as np
import taichi as ti

from src.fedem.contact import (
    AffineIPCModel,
    BarrierModel,
    HertzMindlinModel,
    LinearModel,
)
from src.fedem.contact.ContactKernel import (
    _resolve_linear_levelset_contact_with_energy,
    accumulate_linear_facet_wall_elastic_energy,
    build_compact_facet_wall_candidates,
    clear_facet_wall_history,
    commit_facet_wall_search_state,
    commit_moving_facet_wall_search_state,
    count_facet_wall_history,
    measure_facet_wall_rebuild_requirement,
    measure_moving_facet_wall_rebuild_requirement,
    reset_contact_force,
    reset_compact_facet_wall_contact_force,
    reset_levelset_contact_force,
    save_facet_wall_history,
    resolve_barrier_facet_wall_contact,
    resolve_barrier_levelset_contact,
    resolve_hertz_mindlin_contact,
    resolve_hertz_mindlin_levelset_contact,
    resolve_linear_contact,
    resolve_linear_facet_wall_contact,
)
from src.fedem.neighbor import (
    FEDEMBVH,
    FEDEMLinkedCell,
    FEMLevelSetBroadPhase,
)
from src.fedem.structs import FEMFacetWallContact
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.linalg import no_operation


def _power_of_two(value):
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


class ContactManager:
    def __init__(self, simulation):
        self.simulation = simulation
        self.patch = None
        self.neighbor = None
        self.model = None
        self.initialized = False
        self.level_set = False
        self.wall_contacts = None
        self.wall_contact_count = 0
        self.wall_candidate_count = 0
        self.wall_count = 0
        self.wall_candidate_pairs = None
        self.wall_candidate_count_device = None
        self.wall_search_nodes = None
        self.wall_search_centers = None
        self.wall_maximum_node_sweep = None
        self.wall_maximum_facet_sweep = None
        self.wall_rebuild_required = None
        self.wall_elastic_energy_scratch = None
        self.wall_dense_pair_count = 0
        self.wall_history_capacity = 0
        self.wall_history_entry_capacity = 0
        self.wall_history_state = None
        self.wall_history_key = None
        self.wall_history_gap = None
        self.wall_history_overlap = None
        self.wall_history_overflow = None
        self.wall_history_entry_count = None
        self.fixed_facet_walls = True
        self.dem_particle_count = 0
        self.resolve_step = no_operation
        self.resolve_wall_step = no_operation
        self.surface_requires_rebuild_step = no_operation
        self.rebuild_surface_step = no_operation
        self.wall_requires_rebuild_step = no_operation
        self.commit_wall_search_step = no_operation
        self.reset_energy_step = no_operation
        self.reset_surface_contact_step = no_operation
        self.reset_wall_contact_step = no_operation
        self.read_energy_diagnostics = self._zero_energy_diagnostics

    def choose_contact_model(self, model, **kwargs):
        normalized = str(model).replace("_", "").replace("-", "").replace(" ", "").lower()
        if normalized in ("linear", "linearmodel", "linearspring"):
            self.model = LinearModel(self.simulation)
        elif normalized in ("hertz", "hertzmindlin", "hertzmindlinmodel"):
            self.model = HertzMindlinModel(self.simulation)
        elif normalized in ("barrier", "barriermodel"):
            self.model = BarrierModel(self.simulation)
        elif normalized in ("ipc", "incrementalpotentialcontact", "barrieripc", "semi", "semiipc"):
            kwargs.setdefault(
                "ipc_model",
                "SemiIPC" if normalized in ("semi", "semiipc") else "BarrierIPC",
            )
            self.model = AffineIPCModel(self.simulation, **kwargs)
        else:
            raise ValueError("FEDEM contact model must be Linear, HertzMindlin, Barrier, BarrierIPC, or SemiIPC")
        return self.model

    def add_property(self, dem_material, fem_body, parameters):
        if self.model is None:
            raise RuntimeError("choose the FEDEM contact model before adding properties")
        self.model.add_property(dem_material, fem_body, parameters)

    def initialize(self, patch, fem_engine, dem_sims, dem_scene):
        if self.model is None:
            raise RuntimeError("FEDEM contact model has not been selected")
        self.patch = patch
        self.level_set = dem_sims.scheme == "LSDEM"
        self.patch.update(fem_engine.position_field, update_normals=not self.level_set)
        face_radius_min, face_radius_max = self.patch.bounding_radii()
        self.dem_particle_count = int(dem_scene.particleNum[0])
        if self.dem_particle_count > 0:
            particle_radius_min = float(dem_scene.find_particle_min_radius(dem_sims))
            particle_radius_max = float(dem_scene.find_particle_max_radius(dem_sims))
            if particle_radius_min <= 0.0 or particle_radius_max <= 0.0:
                raise RuntimeError("FEDEM contains a non-positive-radius DEM particle")
            if self.simulation.verlet_distance <= 0.0:
                self.simulation.verlet_distance = self.simulation.verlet_distance_multiplier * min(
                    particle_radius_min, face_radius_min
                )
            if self.level_set:
                self.neighbor = FEMLevelSetBroadPhase(
                    self.simulation,
                    self.patch,
                    dem_sims,
                    dem_scene,
                    interaction_distance=getattr(self.model, "maximum_normal_cutoff", 0.0),
                )
                self.neighbor.rebuild(
                    fem_engine.position_field,
                    dem_scene.rigid,
                    dem_scene.box,
                )
            elif self.simulation.search == "LinkedCell":
                self.neighbor = FEDEMLinkedCell(
                    self.simulation,
                    self.patch,
                    particle_radius_max,
                    face_radius_max,
                )
            else:
                self.neighbor = FEDEMBVH(self.simulation, self.patch)
            if not self.level_set:
                self.neighbor.rebuild(self.dem_particle_count, dem_scene.particle)
        elif self.simulation.verlet_distance <= 0.0:
            # A FEM--FEM-only calculation still uses the coupled time/output
            # lifecycle, but has no cross-system candidate list to build.
            self.simulation.verlet_distance = self.simulation.verlet_distance_multiplier * face_radius_min
        if dem_sims.wall_type == 1 and int(dem_scene.wallNum[0]) > 0:
            if not isinstance(self.model, (LinearModel, BarrierModel)):
                raise RuntimeError(
                    "FEM contact with DEM facet walls currently requires " "the Linear or Barrier contact model"
                )
            self.wall_count = int(dem_scene.wallNum[0])
            self.wall_dense_pair_count = int(patch.node_count) * self.wall_count
            self.wall_contact_count = min(
                self.wall_dense_pair_count,
                int(self.simulation.max_fem_wall_pairs),
            )
            self.wall_contacts = FEMFacetWallContact.field(shape=max(self.wall_contact_count, 1))
            self.wall_candidate_pairs = ti.field(
                dtype=ti.i32,
                shape=max(self.wall_contact_count, 1),
            )
            real_type = ti.lang.impl.current_cfg().default_fp
            self.wall_candidate_count_device = ti.field(dtype=ti.i32, shape=())
            self.wall_search_nodes = ti.Vector.field(
                3,
                dtype=real_type,
                shape=max(self.patch.surface_vertex_count, 1),
            )
            self.wall_search_centers = ti.Vector.field(
                3,
                dtype=real_type,
                shape=max(self.wall_count, 1),
            )
            self.wall_maximum_node_sweep = ti.field(dtype=real_type, shape=())
            self.wall_maximum_facet_sweep = ti.field(dtype=real_type, shape=())
            self.wall_rebuild_required = ti.field(dtype=ti.i32, shape=())
            self.wall_elastic_energy_scratch = ti.field(dtype=real_type, shape=())
            self.wall_history_entry_capacity = min(
                self.wall_dense_pair_count,
                int(self.simulation.max_fem_wall_history_pairs),
            )
            self.wall_history_capacity = _power_of_two(4 * self.wall_history_entry_capacity)
            self.wall_history_state = ti.field(dtype=ti.i32, shape=self.wall_history_capacity)
            self.wall_history_key = ti.field(dtype=ti.i64, shape=self.wall_history_capacity)
            self.wall_history_gap = ti.field(dtype=real_type, shape=self.wall_history_capacity)
            self.wall_history_overlap = ti.Vector.field(
                3,
                dtype=real_type,
                shape=self.wall_history_capacity,
            )
            self.wall_history_overflow = ti.field(dtype=ti.i32, shape=())
            self.wall_history_entry_count = ti.field(dtype=ti.i32, shape=())
            clear_facet_wall_history(
                self.wall_history_capacity,
                self.wall_history_state,
                self.wall_history_overflow,
            )
            self.fixed_facet_walls = not np.any(np.abs(dem_scene.wall.v.to_numpy()[: self.wall_count]) > 0.0)
            self._bind_runtime_functions()
            self.rebuild_wall_candidates(dem_scene.wall)
        self._bind_runtime_functions()
        self.initialized = True

    def _bind_runtime_functions(self):
        self.reset_energy_step = self.model.reset_elastic_energy if self.simulation.track_energy else no_operation
        self.read_energy_diagnostics = (
            self.model.energy_diagnostics if self.simulation.track_energy else self._zero_energy_diagnostics
        )
        if self.neighbor is None:
            self.resolve_step = self._resolve_walls_only
            self.surface_requires_rebuild_step = no_operation
            self.rebuild_surface_step = no_operation
            self.reset_surface_contact_step = no_operation
        elif self.level_set:
            self.resolve_step = self._resolve_levelset
            self.surface_requires_rebuild_step = self._levelset_surface_requires_rebuild
            self.rebuild_surface_step = self._rebuild_levelset
            self.reset_surface_contact_step = self._reset_levelset_contacts
            if isinstance(self.model, LinearModel):
                self.resolve_contact_kernel = _resolve_linear_levelset_contact_with_energy
            elif isinstance(self.model, HertzMindlinModel):
                self.resolve_contact_kernel = resolve_hertz_mindlin_levelset_contact
            else:
                self.resolve_contact_kernel = resolve_barrier_levelset_contact
        else:
            self.resolve_step = self._resolve_facets
            self.surface_requires_rebuild_step = self._facet_surface_requires_rebuild
            self.rebuild_surface_step = self._rebuild_facets
            self.reset_surface_contact_step = self._reset_facet_contacts
            if isinstance(self.model, LinearModel):
                self.resolve_contact_kernel = resolve_linear_contact
            elif isinstance(self.model, HertzMindlinModel):
                self.resolve_contact_kernel = resolve_hertz_mindlin_contact
            else:
                raise RuntimeError("explicit Barrier contact currently requires LSDEM")

        if self.wall_contacts is not None:
            self.reset_wall_contact_step = self._reset_wall_contacts
            self.resolve_wall_step = (
                self._resolve_linear_walls if isinstance(self.model, LinearModel) else self._resolve_barrier_walls
            )
            if self.fixed_facet_walls:
                self.wall_requires_rebuild_step = self._fixed_wall_requires_rebuild
                self.commit_wall_search_step = self._commit_fixed_wall_search
            else:
                self.wall_requires_rebuild_step = self._moving_wall_requires_rebuild
                self.commit_wall_search_step = self._commit_moving_wall_search

    def reset(self):
        self.reset_energy_step()
        self.reset_surface_contact_step()
        self.reset_wall_contact_step()

    def _reset_levelset_contacts(self):
        reset_levelset_contact_force(self.neighbor.contact_count, self.neighbor.contacts)

    def _reset_facet_contacts(self):
        reset_contact_force(self.neighbor.contact_count, self.neighbor.contacts)

    def _reset_wall_contacts(self):
        reset_compact_facet_wall_contact_force(self.wall_candidate_count, self.wall_contacts)

    def surface_requires_rebuild(self, dem_scene=None):
        return bool(self.surface_requires_rebuild_step(dem_scene))

    def _levelset_surface_requires_rebuild(self, dem_scene):
        if dem_scene is None:
            raise RuntimeError("FEM--LSDEM rebuild detection requires the DEM scene")
        return self.neighbor.requires_rebuild(dem_scene.rigid)

    def _facet_surface_requires_rebuild(self, _dem_scene):
        return float(self.patch.maximum_displacement[None]) > 0.5 * self.simulation.verlet_distance

    def rebuild(self, dem_scene, fem_engine=None):
        return self.rebuild_surface_step(dem_scene, fem_engine)

    def _rebuild_levelset(self, dem_scene, fem_engine):
        if fem_engine is None:
            raise RuntimeError("FEM--LSDEM rebuild requires the FEM engine")
        return self.neighbor.rebuild(fem_engine.position_field, dem_scene.rigid, dem_scene.box)

    def _rebuild_facets(self, dem_scene, _fem_engine):
        return self.neighbor.rebuild(int(dem_scene.particleNum[0]), dem_scene.particle)

    def wall_requires_rebuild(self, dem_scene):
        return bool(self.wall_requires_rebuild_step(dem_scene))

    def _fixed_wall_requires_rebuild(self, _dem_scene):
        threshold = 0.5 * self.simulation.verlet_distance
        measure_facet_wall_rebuild_requirement(
            self.patch.surface_vertex_count,
            threshold,
            self.patch.nodes,
            self.patch.surface_vertices,
            self.wall_search_nodes,
            self.wall_rebuild_required,
        )
        return int(self.wall_rebuild_required[None]) != 0

    def _moving_wall_requires_rebuild(self, dem_scene):
        threshold = 0.5 * self.simulation.verlet_distance
        measure_moving_facet_wall_rebuild_requirement(
            self.patch.surface_vertex_count,
            self.wall_count,
            threshold,
            self.patch.nodes,
            self.patch.surface_vertices,
            dem_scene.wall,
            self.wall_search_nodes,
            self.wall_search_centers,
            self.wall_maximum_node_sweep,
            self.wall_maximum_facet_sweep,
            self.wall_rebuild_required,
        )
        return int(self.wall_rebuild_required[None]) != 0

    def rebuild_wall_candidates(self, wall):
        if self.wall_contacts is None:
            return 0
        if self.wall_candidate_count > 0:
            count_facet_wall_history(
                self.wall_candidate_count,
                self.wall_contacts,
                self.wall_history_entry_count,
            )
            entries = int(self.wall_history_entry_count[None])
            if entries > self.wall_history_entry_capacity:
                raise RuntimeError(
                    "FEM--facet wall max_fem_wall_history_pairs is too "
                    f"small: need {entries}, allocated "
                    f"{self.wall_history_entry_capacity}"
                )
            clear_facet_wall_history(
                self.wall_history_capacity,
                self.wall_history_state,
                self.wall_history_overflow,
            )
            save_facet_wall_history(
                self.wall_candidate_count,
                self.wall_candidate_pairs,
                self.wall_contacts,
                self.wall_history_capacity,
                self.wall_history_state,
                self.wall_history_key,
                self.wall_history_gap,
                self.wall_history_overlap,
                self.wall_history_overflow,
            )
            if int(self.wall_history_overflow[None]) != 0:
                raise RuntimeError("FEM--facet wall contact-history hash overflow")
        build_compact_facet_wall_candidates(
            self.patch.surface_vertex_count,
            self.wall_count,
            self.simulation.verlet_distance + getattr(self.model, "maximum_normal_cutoff", 0.0),
            wall,
            self.patch.nodes,
            self.patch.surface_vertices,
            self.wall_candidate_count_device,
            self.wall_contact_count,
            self.wall_candidate_pairs,
            self.wall_contacts,
            self.wall_history_capacity,
            self.wall_history_state,
            self.wall_history_key,
            self.wall_history_gap,
            self.wall_history_overlap,
        )
        self.wall_candidate_count = int(self.wall_candidate_count_device[None])
        if self.wall_candidate_count > self.wall_contact_count:
            raise RuntimeError("FEM--facet wall candidate capacity was exceeded")
        self.commit_wall_search_step(wall)
        return self.wall_candidate_count

    def _commit_fixed_wall_search(self, _wall):
        commit_facet_wall_search_state(
            self.patch.surface_vertex_count,
            self.patch.nodes,
            self.patch.surface_vertices,
            self.wall_search_nodes,
        )

    def _commit_moving_wall_search(self, wall):
        commit_moving_facet_wall_search_state(
            self.patch.surface_vertex_count,
            self.wall_count,
            self.patch.nodes,
            self.patch.surface_vertices,
            wall,
            self.wall_search_nodes,
            self.wall_search_centers,
        )

    def resolve(self, dem_scene, fem_engine):
        self.resolve_step(dem_scene, fem_engine)

    def _resolve_walls_only(self, dem_scene, fem_engine):
        self.resolve_wall_step(dem_scene, fem_engine)

    def _resolve_levelset(self, dem_scene, fem_engine):
        arguments = (
            self.neighbor.contact_count,
            self.simulation.delta,
            self.model.surface_properties,
            dem_scene.rigid,
            dem_scene.box,
            dem_scene.rigid_grid,
            self.patch.nodes,
            self.patch.node_body,
            self.patch.node_area,
            fem_engine.velocity_field,
            fem_engine.state.mass,
            fem_engine.state.external_force,
            self.neighbor.contacts,
        )
        self.resolve_contact_kernel(
            *arguments,
            self.model.elastic_energy,
            self.model.friction_dissipation,
            self.model.damping_dissipation,
            self.simulation.track_energy,
        )
        self.resolve_wall_step(dem_scene, fem_engine)

    def _resolve_facets(self, dem_scene, fem_engine):
        arguments = (
            self.neighbor.contact_count,
            self.simulation.delta,
            self.model.surface_properties,
            dem_scene.particle,
            self.patch.nodes,
            self.patch.faces,
            self.patch.face_body,
            self.patch.normals,
            fem_engine.velocity_field,
            fem_engine.state.external_force,
            self.neighbor.contacts,
        )
        self.resolve_contact_kernel(*arguments)
        self.resolve_wall_step(dem_scene, fem_engine)

    def _wall_arguments(self, dem_scene, fem_engine):
        return (
            self.wall_candidate_count,
            self.wall_count,
            self.wall_candidate_pairs,
            self.simulation.delta,
            self.model.surface_properties,
            dem_scene.wall,
            self.patch.nodes,
            self.patch.node_body,
            self.patch.node_area,
            fem_engine.velocity_field,
            fem_engine.state.mass,
            fem_engine.state.external_force,
            self.wall_contacts,
        )

    def _resolve_linear_walls(self, dem_scene, fem_engine):
        arguments = self._wall_arguments(dem_scene, fem_engine)
        resolve_linear_facet_wall_contact(
            *arguments,
            self.model.elastic_energy,
            self.model.friction_dissipation,
            self.model.damping_dissipation,
            self.simulation.track_energy,
        )

    def _resolve_barrier_walls(self, dem_scene, fem_engine):
        arguments = self._wall_arguments(dem_scene, fem_engine)
        resolve_barrier_facet_wall_contact(
            *arguments,
            self.model.elastic_energy,
            self.model.friction_dissipation,
            self.model.damping_dissipation,
            self.simulation.track_energy,
        )

    def wall_contact_diagnostics(self):
        if self.wall_contacts is None:
            return {
                "candidate_count": 0,
                "active_count": 0,
                "maximum_penetration": 0.0,
            }
        import numpy as np

        active = field_to_numpy_prefix(self.wall_contacts.active, self.wall_candidate_count).astype(bool)
        gap = field_to_numpy_prefix(self.wall_contacts.normal_gap, self.wall_candidate_count)
        return {
            "candidate_count": self.wall_candidate_count,
            "active_count": int(np.count_nonzero(active)),
            "maximum_penetration": float(np.max(-gap[active]) if np.any(active) else 0.0),
        }

    def wall_elastic_energy(self, wall_id, wall):
        """Return current FEM penalty energy supported by one wall group."""

        if self.wall_contacts is None:
            return 0.0
        if not isinstance(self.model, LinearModel):
            raise RuntimeError("facet-wall elastic-energy accounting requires Linear contact")
        accumulate_linear_facet_wall_elastic_energy(
            self.wall_candidate_count,
            self.wall_count,
            int(wall_id),
            self.wall_candidate_pairs,
            self.model.surface_properties,
            wall,
            self.patch.nodes,
            self.patch.node_body,
            self.patch.node_area,
            self.wall_contacts,
            self.wall_elastic_energy_scratch,
        )
        return float(self.wall_elastic_energy_scratch[None])

    def critical_timestep(self, dem_scene, dem_sims):
        if self.dem_particle_count <= 0:
            return float("inf")
        minimum_mass = dem_scene.find_particle_min_mass(dem_sims)
        maximum_radius = dem_scene.find_particle_max_radius(dem_sims)
        return self.model.critical_timestep(minimum_mass, maximum_radius)

    def energy_diagnostics(self):
        return self.read_energy_diagnostics()

    @staticmethod
    def _zero_energy_diagnostics():
        return {
            "elastic_energy": 0.0,
            "friction_dissipation": 0.0,
            "damping_dissipation": 0.0,
        }


__all__ = ["ContactManager"]
