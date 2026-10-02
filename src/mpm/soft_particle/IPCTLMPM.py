import taichi as ti

import src.mpm.config as config
from src.mpm.engines.direct.ImplicitTLMPM import ImplicitTLMPM
from src.mpm.soft_particle.IPCMPM import IPCMPM
from src.mpm.generator.Ground import Ground
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import normalize_callbacks


@ti.data_oriented
class IPCTLMPM:
    def __init__(self, bodies, ground: Ground = None, dirichlet=None, neumann=None, **kwargs):
        self.mpm = ImplicitTLMPM(bodies, dirichlet, neumann, **kwargs)
        self.ipc = IPCMPM(self.mpm, ground, **kwargs)

    def initial_simulation(self):
        self.mpm.initial_simulation()

    def _substep_once(self, verbose=True):
        self.mpm.mass_vec.fill(0)
        self.mpm.grid_reset()
        self.mpm.vel_acc_p2g()
        self.mpm.assemble_traction_step()
        self.mpm.compute_nodal_vel_acc()
        if config.DYNAMIC:
            self.mpm.compute_mass_list(self.mpm.integration)

        self.mpm.grid_disp.fill(0)
        iter_num, residual = self.ipc.solve_friction_step(verbose)
        if verbose:
            print(f"Optimize iteration {iter_num}, residual: {residual}")
        self.ipc.differentiate_before_commit()
        self.mpm.update_nodal_acc(self.mpm.integration)
        self.mpm.advent_particles(self.mpm.coeffPIC)
        self.ipc.ground.move(self.mpm.dt)
        return {"iterations": int(iter_num), "residual": float(residual)}

    def substep(self, verbose=True):
        return IPCTLMPM._substep_once(self, verbose)

    def step(self, verbose=True, record_history=True):
        """Run one transactional IPC-MPM step with bounded retry."""
        return self.mpm.run_substep(self.substep, verbose, record_history=record_history)

    def differentiate_elastic_step(self, loss_gradient, verbose=True):
        return self.ipc.differentiate_elastic_step(self.step, loss_gradient, verbose)

    def differentiate_plastic_equilibrium_step(self, loss_gradient, verbose=True):
        return self.ipc.differentiate_plastic_equilibrium_step(self.step, loss_gradient, verbose)

    def diagnostics_snapshot(self):
        snapshot = self.mpm.diagnostics_snapshot()
        snapshot["subsystem"] = "mpm_direct_ipc"
        snapshot["contact"] = {
            "model": self.ipc.barrier.model,
            "constraint_violation": float(self.ipc.semi_constraint_violation[None]) if self.ipc.is_semi else 0.0,
            "newton_iterations": int(self.ipc.last_newton_iterations),
            "newton_residual": float(self.ipc.last_newton_residual),
            "friction_iterations": int(self.ipc.last_friction_iterations),
            "friction_residual": float(self.ipc.last_friction_residual),
            "friction_converged": bool(self.ipc.last_friction_converged),
        }
        return snapshot

    def run(self, verbose=True, postprocessing=()):
        postprocessing = normalize_callbacks(postprocessing)
        self.initial_simulation()
        for f in postprocessing:
            f()
        for output_index in range(self.mpm.total_step):
            for interval_index in range(self.mpm.output_interval):
                next_step = self.mpm.step_count + 1
                output_due = interval_index + 1 == self.mpm.output_interval
                final_step = output_index + 1 == self.mpm.total_step and output_due
                self.step(
                    verbose,
                    record_history=self.mpm.step_schedule.history_due(next_step, output=output_due, final=final_step),
                )
                runtime_checkpoint()
            self.mpm.record()
            for f in postprocessing:
                f()
