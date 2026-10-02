"""Explicit two-way FEM--DEM step orchestration."""

from __future__ import annotations

from src.utils.linalg import no_operation


class Engine:
    def __init__(self, simulation, fem, dem, contactor, patch):
        self.simulation = simulation
        self.fem = fem
        self.dem = dem
        self.contactor = contactor
        self.patch = patch
        self.fem_engine = fem.engine
        self.dem_engine = dem.enginer
        self.minimum_jacobian = 1.0
        self.static_fem = bool(self.fem_engine.is_fully_constrained_static)
        self.initialized = False
        self.stage_call = self._direct_stage_call
        self.update_fem_surface = no_operation
        self.accumulate_dem_point_displacement = no_operation
        self._bind_fem_runtime_functions()
        if self.dem.sims.scheme == "LSDEM":
            self.accumulate_dem_point_displacement = self.dem.contactor.neighbor.accumulate_point_relative_displacement
            self.update_dem_verlet_tables = self._update_lsdem_verlet_tables
        else:
            self.update_dem_verlet_tables = self._update_dem_verlet_tables

    def _bind_fem_runtime_functions(self):
        self.prepare_fem_step = self._prepare_static_fem_step if self.static_fem else self._prepare_dynamic_fem_step
        self.update_fem_surface = no_operation
        self.integrate_fem = no_operation if self.static_fem else self._integrate_dynamic_fem
        self.sample_fem_energy = (
            self.fem_engine.state.reduce_kinetic_energy if self.fem_engine.track_energy else no_operation
        )

    @staticmethod
    def _direct_stage_call(_name, function, *args, **kwargs):
        return function(*args, **kwargs)

    def bind_stage_profiler(self, stage_profiler=None):
        self.stage_call = self._direct_stage_call if stage_profiler is None else stage_profiler.measure

    def _stage_caller(self, stage_profiler):
        return self.stage_call if stage_profiler is None else stage_profiler.measure

    def _prepare_dynamic_fem_step(self):
        self.fem_engine.prepare_explicit_step(
            self.simulation.current_time,
            self.simulation.current_step,
        )

    def _prepare_static_fem_step(self):
        self.fem_engine.update_external_force_step(
            self.simulation.current_time,
            self.simulation.current_step,
        )

    def _update_dynamic_facet_surface(self, check_rebuild):
        self.patch.update(
            self.fem_engine.position_field,
            update_normals=True,
            # Total-Lagrangian contact quadrature uses reference weights.
            update_area=False,
            measure_displacement=check_rebuild,
        )

    def _update_dynamic_levelset_surface(self, _check_rebuild):
        self.patch.update(
            self.fem_engine.position_field,
            update_normals=False,
            update_area=False,
            measure_displacement=False,
        )

    def pre_calculate(self):
        if self.initialized:
            return
        self.dem_engine.pre_calculation(self.dem.sims, self.dem.scene, self.dem.contactor.neighbor)
        self.contactor.initialize(
            self.patch,
            self.fem_engine,
            self.dem.sims,
            self.dem.scene,
        )
        if not self.static_fem:
            self.update_fem_surface = (
                self._update_dynamic_levelset_surface
                if self.contactor.level_set
                else self._update_dynamic_facet_surface
            )
        if self.static_fem:
            # A fully and time-independently constrained FEM surface cannot
            # deform.  Validate its initial state once instead of reducing a
            # constant Jacobian after every contact step.
            self.minimum_jacobian = self.fem_engine._minimum_jacobian_ratio_device(self.fem_engine.state.position)
            if self.minimum_jacobian <= 1.0e-10:
                raise RuntimeError("fixed FEDEM contact surface has an inverted/collapsed FEM element")
        self.initialized = True

    def reset_message(self):
        self.prepare_fem_step()
        self.dem_engine.reset_wall_message(self.dem.scene)
        self.dem_engine.reset_particle_message(self.dem.scene)
        # Match the standalone DEM step contract: elastic contact energy is
        # an instantaneous stored quantity and must be cleared before force
        # assembly. The contact model deliberately leaves friction and
        # viscous work cumulative, so this call does not erase dissipation.
        self.dem_engine.reset_contact_energy()
        self.contactor.reset()

    def update_verlet_tables(self, check_rebuild=True):
        need_dem_rebuild = self.update_dem_verlet_tables(check_rebuild)
        self.update_fem_surface(check_rebuild)
        if check_rebuild:
            if need_dem_rebuild or self.contactor.surface_requires_rebuild(self.dem.scene):
                self.contactor.rebuild(self.dem.scene, self.fem_engine)
            if self.contactor.wall_requires_rebuild(self.dem.scene):
                self.contactor.rebuild_wall_candidates(self.dem.scene.wall)

    def _update_dem_verlet_tables(self, check_rebuild):
        need_rebuild = bool(check_rebuild and self.dem_engine.is_verlet_update(self.dem_engine.limit1) == 1)
        if need_rebuild:
            self.dem_engine.update_verlet_table(
                self.dem.sims,
                self.dem.scene,
                self.dem.contactor.neighbor,
            )
        return need_rebuild

    def _update_lsdem_verlet_tables(self, check_rebuild):
        self.accumulate_dem_point_displacement(self.dem.scene)
        if not check_rebuild:
            return False
        need_rebuild = self.dem_engine.is_verlet_update(self.dem_engine.limit1) == 1
        if need_rebuild:
            # The coarse list owns the fine surface list. Refresh them in
            # dependency order whenever the bounding-sphere list changes.
            self.dem_engine.update_LSDEM_verlet_table1(
                self.dem.sims,
                self.dem.scene,
                self.dem.contactor.neighbor,
            )
            self.dem_engine.update_LSDEM_verlet_table2(
                self.dem.sims,
                self.dem.scene,
                self.dem.contactor.neighbor,
            )
        elif self.dem.contactor.neighbor.accumulated_point_displacement_requires_rebuild(
            self.dem_engine.limit2,
            self.dem.scene,
        ):
            self.dem_engine.update_LSDEM_verlet_table2(
                self.dem.sims,
                self.dem.scene,
                self.dem.contactor.neighbor,
            )
        return need_rebuild

    def system_resolve(self, check_rebuild=True, stage_profiler=None):
        stage_call = self._stage_caller(stage_profiler)
        stage_call(
            "dem_contact_force",
            self.dem_engine.system_resolve,
            self.dem.sims,
            self.dem.scene,
            self.dem.contactor.neighbor,
        )
        stage_call(
            "fem_lsdem_wall_contact",
            self.contactor.resolve,
            self.dem.scene,
            self.fem_engine,
        )
        stage_call(
            "fem_fem_contact",
            self.fem_engine.resolve_soft_particle_contact_step,
            advance_history=True,
            check_rebuild=check_rebuild,
        )

    def integration(
        self,
        internal_force=None,
        update_diagnostics=True,
        check_jacobian=True,
        stage_profiler=None,
    ):
        """Advance one coupled step.

        ``internal_force`` may reuse an assembly already performed at the
        current FEM configuration for sampling.  Setting
        ``update_diagnostics=False`` skips the post-update force assembly
        used only to refresh residuals, reactions, and kinetic-energy output;
        the explicit trajectory is unchanged. ``check_jacobian=False`` also
        skips the device reduction and host scalar read used only as a safety
        gate. Long coupled runs can request this between scheduled safety
        checks while retaining the default per-step behavior for existing
        callers.
        """
        stage_call = self._stage_caller(stage_profiler)
        stage_call(
            "rigid_body_integration",
            self.dem_engine.integration,
            self.dem.sims,
            self.dem.scene,
            self.dem.contactor.neighbor,
        )
        self.integrate_fem(
            internal_force,
            update_diagnostics,
            check_jacobian,
            stage_call,
        )

    def _integrate_dynamic_fem(
        self,
        internal_force,
        update_diagnostics,
        check_jacobian,
        stage_call,
    ):
        stage_call(
            "fem_constitutive_update",
            self.fem_engine.advance_constitutive_state,
        )
        if internal_force is None:
            internal_force = stage_call(
                "fem_internal_force",
                self.fem_engine.assemble_explicit_internal_force,
            )
        stage_call(
            "fem_nodal_integration",
            self.fem_engine.state.explicit_update,
            internal_force,
            self.fem_engine.damping,
            self.simulation.delta,
            self.fem_engine.track_energy,
        )
        next_time = self.simulation.current_time + self.simulation.delta

        stage_call(
            "fem_boundary_update",
            self._update_fem_boundary,
            next_time,
        )
        if check_jacobian:
            self.minimum_jacobian = stage_call(
                "fem_jacobian_check",
                self.fem_engine._minimum_jacobian_ratio_device,
                self.fem_engine.state.position,
            )
            if self.minimum_jacobian <= 1.0e-10:
                raise RuntimeError(
                    "explicit FEDEM produced an inverted/collapsed FEM element "
                    f"at t={next_time:.9g} with minimum Jacobian ratio "
                    f"{self.minimum_jacobian:.9g}; reduce dt or revise the "
                    "impact/material contract"
                )
        if update_diagnostics:
            updated_internal = self.fem_engine.assemble_explicit_internal_force()
            self.fem_engine.state.assemble_equilibrium(updated_internal, self.fem_engine.damping, 0)
            self.sample_fem_energy()

    def _update_fem_boundary(self, next_time):
        self.fem_engine.set_boundary_data_step(next_time, self.simulation.current_step + 1)
        self.fem_engine.apply_boundary_step(self.simulation.delta, 1)

    def step(self, update_diagnostics=True, check_jacobian=True):
        timer = self.simulation.timer
        with timer.section("Reset"):
            self.reset_message()
        with timer.section("Neighbor search"):
            self.update_verlet_tables()
        with timer.section("Contact resolve"):
            self.system_resolve()
        with timer.section("FEDEM integration"):
            self.integration(
                update_diagnostics=update_diagnostics,
                check_jacobian=check_jacobian,
            )


__all__ = ["Engine"]
