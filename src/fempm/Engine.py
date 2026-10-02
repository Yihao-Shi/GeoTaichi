"""Explicit two-way FEM--MPM step orchestration."""

from __future__ import annotations

from src.utils.linalg import no_operation


class Engine:
    def __init__(self, simulation, fem, mpm, contactor, patch):
        self.simulation = simulation
        self.fem = fem
        self.mpm = mpm
        self.contactor = contactor
        self.patch = patch
        self.fem_engine = fem.engine
        self.mpm_engine = mpm.enginer
        self.minimum_jacobian = 1.0
        self.sample_fem_energy = (
            self.fem_engine.state.reduce_kinetic_energy if self.fem_engine.track_energy else no_operation
        )

    def pre_calculate(self):
        self.mpm_engine.pre_calculation(self.mpm.sims, self.mpm.scene, self.mpm.neighbor)
        self.contactor.initialize(self.patch, self.fem_engine, self.mpm.scene)

    def reset_message(self):
        self.fem_engine.state.save_step_state()
        self.fem_engine.update_external_force_step(self.simulation.current_time, self.simulation.current_step)
        self.mpm_engine.reset_grid_message(self.mpm.scene)
        self.mpm_engine.reset_particle_message(self.mpm.scene)
        self.contactor.reset()

    def update_verlet_tables(self):
        mpm_requires_rebuild = self.mpm_engine.is_verlet_update(self.mpm.scene) == 1
        if mpm_requires_rebuild:
            self.mpm_engine.execute_board_serach(self.mpm.sims, self.mpm.scene, self.mpm.neighbor)
        self.patch.update(self.fem_engine.position_field)
        if mpm_requires_rebuild or self.contactor.requires_rebuild(self.mpm.scene):
            self.contactor.rebuild(self.mpm.scene)
        # In Lagrangian neighbor modes MPM moves mass/momentum P2G to this
        # hook. It is a no-op for ordinary MPM, so this call is unconditional.
        self.mpm_engine.system_resolve(self.mpm.sims, self.mpm.scene)

    def system_resolve(self):
        self.contactor.resolve(self.mpm.scene, self.fem_engine)

    def integration(self, update_diagnostics=True, check_jacobian=True):
        self.mpm_engine.compute(self.mpm.sims, self.mpm.scene, self.mpm.neighbor)
        self.fem_engine.advance_constitutive_state()
        internal_force = self.fem_engine.assemble_explicit_internal_force()
        self.fem_engine.state.explicit_update(
            internal_force,
            self.fem_engine.damping,
            self.simulation.delta,
            self.fem_engine.track_energy,
        )
        next_time = self.simulation.current_time + self.simulation.delta
        self.fem_engine.set_boundary_data_step(next_time, self.simulation.current_step + 1)
        self.fem_engine.apply_boundary_step(self.simulation.delta, 1)
        if check_jacobian:
            self.minimum_jacobian = self.fem_engine._minimum_jacobian_ratio_device(self.fem_engine.state.position)
            if self.minimum_jacobian <= 1.0e-10:
                raise RuntimeError("explicit FEMPM produced an inverted/collapsed FEM element; reduce dt")
        if update_diagnostics:
            updated_internal = self.fem_engine.assemble_explicit_internal_force()
            self.fem_engine.state.assemble_equilibrium(updated_internal, self.fem_engine.damping, 0)
            self.sample_fem_energy()

    def step(self, update_diagnostics=True, check_jacobian=True):
        timer = self.simulation.timer
        with timer.section("Reset"):
            self.reset_message()
        with timer.section("Neighbor search"):
            self.update_verlet_tables()
        with timer.section("Contact resolve"):
            self.system_resolve()
        with timer.section("FEMPM integration"):
            self.integration(
                update_diagnostics=update_diagnostics,
                check_jacobian=check_jacobian,
            )


__all__ = ["Engine"]
