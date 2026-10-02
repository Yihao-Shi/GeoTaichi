import taichi as ti

import src.mpm.config as config
from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
from src.mpm.engines.direct.MPMSolver import MPMConvergenceError
from src.mpm.soft_particle.IPCMPM import IPCMPM
from src.mpm.generator.Ground import Ground
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import normalize_callbacks


@ti.data_oriented
class IPCULMPM:
    def __init__(self, bodies, ground: Ground = None, dirichlet=None, neumann=None, **kwargs):
        self.mpm = ImplicitULMPM(bodies, dirichlet, neumann, **kwargs)
        self.ipc = IPCMPM(self.mpm, ground, **kwargs)

    def initial_simulation(self):
        self.mpm.initial_simulation()

    def prepare_step_device(self):
        self.mpm.mass_vec.fill(0)
        self.mpm.grid_reset()
        self.mpm.compute_shapefn()
        self.mpm.mass_vel_acc_p2g()
        self.mpm.assemble_traction_step()
        self.mpm.find_active_node()
        self.mpm.prefix_sum_executor.run(self.mpm.node2dof)
        self.mpm.active_dof = self.mpm.set_active_dof()
        self.mpm.compute_nodal_vel_acc()
        if config.DYNAMIC:
            self.mpm.compute_mass_list(self.mpm.integration)

    def _substep_once(self, verbose=True):
        IPCULMPM.prepare_step_device(self)
        self.mpm.grid_disp.fill(0)
        has_lagged_material = getattr(self.mpm, "has_lagged_material", False) is True
        if has_lagged_material:
            self.mpm.begin_lagged_material_state()
        iter_num = 0
        residual = float("inf")
        material_converged = not has_lagged_material
        outer_limit = self.mpm.material_lagged_max_iterations if has_lagged_material else 1
        for _ in range(outer_limit):
            inner_iterations, residual = self.ipc.solve_friction_step(verbose)
            iter_num += inner_iterations
            if not has_lagged_material:
                material_converged = True
                break
            if self.mpm.refresh_lagged_material_state(self.mpm.grid_disp) <= self.mpm.material_lagged_tolerance:
                material_converged = True
                break
        if not material_converged:
            raise MPMConvergenceError(
                "IPC ULMPM lagged MCC hardening did not converge: "
                f"error={self.mpm.last_material_lagged_error:.6e} after "
                f"{self.mpm.material_lagged_max_iterations} iterations"
            )
        if verbose:
            print(f"Optimize iteration {iter_num}, residual: {residual}")
        self.ipc.differentiate_before_commit()
        self.mpm.update_nodal_acc(self.mpm.integration)
        self.mpm.advent_particles(self.mpm.coeffPIC)
        self.ipc.ground.move(self.mpm.dt)
        return {"iterations": int(iter_num), "residual": float(residual)}

    def substep(self, verbose=True):
        return IPCULMPM._substep_once(self, verbose)

    def step(self, verbose=True, record_history=True):
        """Run one transactional IPC-MPM step with bounded retry."""
        return self.mpm.run_substep(self.substep, verbose, record_history=record_history)

    def differentiate_elastic_step(self, loss_gradient, verbose=True):
        return self.ipc.differentiate_elastic_step(self.step, loss_gradient, verbose)

    def differentiate_plastic_equilibrium_step(self, loss_gradient, verbose=True):
        return self.ipc.differentiate_plastic_equilibrium_step(self.step, loss_gradient, verbose)

    def differentiate_plastic_step(self, loss_gradient, state_vjp, verbose=True):
        return self.ipc.differentiate_plastic_step(self.step, loss_gradient, state_vjp, verbose)

    def differentiate_plastic_particle_step(self, state_vjp, verbose=True):
        return self.ipc.differentiate_plastic_particle_step(self.step, state_vjp, verbose)

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
