import taichi as ti
import time

import src.iga.config as config
from src.iga.engines.IGASolver import IGASolver
from src.iga.engines.EngineUtils import (
    jacobian2parent2parametric1d,
    jacobian2parent2parametric2d,
    linearize,
    vectorize_id,
)
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.linalg import no_operation


@ti.data_oriented
class ExplicitIGA(IGASolver):
    def __init__(self, primitives, dirichlet=None, neumann=None, **kwargs):
        if dirichlet is not None and dirichlet.velocity_dofs.size:
            raise ValueError("append_velocity is supported by implicit IGA only")
        super().__init__(primitives, dirichlet, neumann, **kwargs)
        self.damping = kwargs.get("damping", 0.0)
        self.rhs = ti.field(ti.f64, shape=self.degree_of_freedom)
        self.material_energy = ti.field(ti.f64, shape=())
        self.external_potential_energy = ti.field(ti.f64, shape=())
        self.kinetic_energy = ti.field(ti.f64, shape=())
        self.damping_dissipation = ti.field(ti.f64, shape=())
        self.latest_energy = {}
        self.reset_instantaneous_energy_step = self._reset_instantaneous_energy if self.track_energy else no_operation
        self.sample_energy_step = self._sample_energy if self.track_energy else no_operation
        self.add_energy_record = self._add_energy_record if self.track_energy else no_operation
        self.apply_neumann_step = self.apply_neumann if self.neumann.num > 0 else no_operation
        self.apply_dirichlet_step = self.apply_dirichlet if self.dirichlet.num > 0 else no_operation

    @ti.func
    def compute_local_gradient(self, local_offset, dPsi_dF, dFdx):
        local_gradient = ti.Vector.zero(ti.f64, config.DIM)
        for d in range(config.DIM):
            for i in range(config.DIM * config.DIM):
                local_gradient[d] += dFdx[config.DIM * local_offset + d, i] * dPsi_dF[i]
        return local_gradient

    @ti.kernel
    def assemble_internal_force(
        self,
        total_num_element: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
    ):
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            j2 = jacobian2parent2parametric2d(elrange_u, elrange_v)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]
                j2 *= jacobian2parent2parametric1d(elrange_w)

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                local_offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                ctrlpt_id = prefix_total_num_ctrlpts + global_offset
                rest_control_points = self.patch.rest_control_points[ctrlpt_id]
                control_points = self.patch.control_points[ctrlpt_id]
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[local_offset, j] = rest_control_points[j]
                    current_ctrl_coords[local_offset, j] = control_points[j]

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id,
                    elrange_u,
                    elrange_v,
                    elrange_w,
                    prefix_num_knot,
                    prefix_total_num_ctrlpts,
                    num_knot,
                    self.patch,
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                j1 = self.require_positive_reference_jacobian(jacobian)
                volume = j1 * j2 * self.element.gauss_weights[gauss_id]
                dnatdX = jacobian.inverse()
                deformation_gradient = self.element.compute_deformation_gradient(dNdnat, dnatdX, current_ctrl_coords)
                dFdx = self.element.compute_dF_div_dx(dNdnat, dnatdX)
                dPsi_dF = self.material.dPsi_div_dF(deformation_gradient) * volume
                if ti.static(self.track_energy):
                    ti.atomic_add(
                        self.material_energy[None],
                        self.material.Psi(deformation_gradient) * volume,
                    )

                for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                    global_nodeID = nodeID + eleid
                    local_offset = linearize(nodeID, self.element.knot_range)
                    global_offset = linearize(global_nodeID, num_ctrlpts)
                    ctrlpt_id = prefix_total_num_ctrlpts + global_offset
                    local_gradient = self.compute_local_gradient(local_offset, dPsi_dF, dFdx)
                    for d in ti.static(range(config.DIM)):
                        self.rhs[config.DIM * ctrlpt_id + d] -= local_gradient[d]

    @ti.kernel
    def assemble_body_force(
        self, total_num_ctrlpts: ti.i32, prefix_total_num_ctrlpts: ti.i32, gravity: ti.types.vector(config.DIM, ti.f64)
    ):
        for index in range(total_num_ctrlpts):
            ctrlpt_id = prefix_total_num_ctrlpts + index
            nodal_mass = self.patch.volume[ctrlpt_id] * self.material.density
            for d in ti.static(range(config.DIM)):
                self.rhs[config.DIM * ctrlpt_id + d] += nodal_mass * gravity[d]
            if ti.static(self.track_energy):
                ti.atomic_add(
                    self.external_potential_energy[None],
                    -nodal_mass * gravity.dot(self.patch.control_points[ctrlpt_id]),
                )

    @ti.kernel
    def apply_neumann(self):
        for i in self.neumann.node:
            dof = self.neumann.node[i]
            value = self.neumann.value[i]
            self.rhs[dof] += value
            if ti.static(self.track_energy):
                ctrlpt_id = dof // config.DIM
                direction = dof - ctrlpt_id * config.DIM
                displacement = (
                    self.patch.control_points[ctrlpt_id][direction]
                    - self.patch.initial_control_points[ctrlpt_id][direction]
                )
                ti.atomic_add(
                    self.external_potential_energy[None],
                    -value * displacement,
                )

    @ti.kernel
    def explicit_advance(self, total_num_ctrlpts: ti.i32, prefix_total_num_ctrlpts: ti.i32, damping: ti.f64):
        dt = self.TIdt[None]
        velocity_decay = ti.max(0.0, 1.0 - damping * dt)
        for index in range(total_num_ctrlpts):
            ctrlpt_id = prefix_total_num_ctrlpts + index
            nodal_mass = self.patch.volume[ctrlpt_id] * self.material.density
            if nodal_mass > 1.0e-14:
                acceleration = ti.Vector.zero(ti.f64, config.DIM)
                for d in ti.static(range(config.DIM)):
                    acceleration[d] = self.rhs[config.DIM * ctrlpt_id + d] / nodal_mass
                trial_velocity = self.patch.velocitys[ctrlpt_id] + dt * acceleration
                updated_velocity = velocity_decay * trial_velocity
                if ti.static(self.track_energy):
                    ti.atomic_add(
                        self.damping_dissipation[None],
                        0.5
                        * nodal_mass
                        * ti.max(
                            trial_velocity.norm_sqr() - updated_velocity.norm_sqr(),
                            0.0,
                        ),
                    )
                self.patch.accelerations[ctrlpt_id] = acceleration
                self.patch.velocitys[ctrlpt_id] = updated_velocity
                self.patch.control_points[ctrlpt_id] += dt * self.patch.velocitys[ctrlpt_id]

    @ti.kernel
    def reduce_kinetic_energy(self):
        self.kinetic_energy[None] = 0.0
        for ctrlpt_id in range(self.patch.primitive.num_ctrlpts):
            nodal_mass = self.patch.volume[ctrlpt_id] * self.material.density
            ti.atomic_add(
                self.kinetic_energy[None],
                0.5 * nodal_mass * self.patch.velocitys[ctrlpt_id].norm_sqr(),
            )

    def _reset_instantaneous_energy(self):
        self.material_energy.fill(0.0)
        self.external_potential_energy.fill(0.0)

    def _sample_energy(self):
        # Reassemble at the accepted position so stored and potential energy
        # are time-centered with the sampled post-update kinetic energy.
        self.assemble_force()
        self.reduce_kinetic_energy()
        self.latest_energy = {
            "material_energy": float(self.material_energy[None]),
            "external_potential_energy": float(self.external_potential_energy[None]),
            "kinetic_energy": float(self.kinetic_energy[None]),
            "damping_dissipation": float(self.damping_dissipation[None]),
        }

    def _add_energy_record(self, record):
        record.update(self.latest_energy)

    @ti.kernel
    def apply_dirichlet(self):
        for dof in range(self.degree_of_freedom):
            if self.dirichlet.node[dof] == 1:
                ctrlpt_id = dof // config.DIM
                direction = dof - ctrlpt_id * config.DIM
                self.patch.control_points[ctrlpt_id][direction] = (
                    self.patch.initial_control_points[ctrlpt_id][direction] + self.dirichlet.value[dof]
                )
                self.patch.velocitys[ctrlpt_id][direction] = 0.0
                self.patch.accelerations[ctrlpt_id][direction] = 0.0
                self.rhs[dof] = 0.0

    def assemble_force(self):
        self.rhs.fill(0)
        self.reset_instantaneous_energy_step()
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_num_knot = self.patch.prefix_num_knot[patch_id]
            prefix_num_element = self.patch.prefix_num_element[patch_id]
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            total_num_ctrlpts = self.patch.total_num_ctrlpts[patch_id + 1]
            total_num_element = self.patch.total_num_element[patch_id + 1]
            num_knot = self.patch.num_knot[patch_id + 1]
            num_element = self.patch.num_element[patch_id + 1]
            num_ctrlpts = self.patch.num_ctrlpts[patch_id + 1]
            self.assemble_internal_force(
                total_num_element,
                prefix_total_num_ctrlpts,
                prefix_num_knot,
                prefix_num_element,
                num_knot,
                num_element,
                num_ctrlpts,
            )
            self.assemble_body_force(total_num_ctrlpts, prefix_total_num_ctrlpts, self.gravity)
        self.apply_neumann_step()

    def advance(self):
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            total_num_ctrlpts = self.patch.total_num_ctrlpts[patch_id + 1]
            self.explicit_advance(total_num_ctrlpts, prefix_total_num_ctrlpts, self.damping)
        self.apply_dirichlet_step()

    def initial_simulation(self):
        self.precompute()
        self.apply_dirichlet_step()
        self.visualize_stress()
        self.visualize()

    def record(self, log=True):
        self.visualize_stress()
        self.visualize(log=log)

    def substep(self, verbose=True, record_history=True):
        self.assemble_force()
        self.advance()
        self.time += float(self.dt)
        self.step_count += 1
        if record_history:
            self.sample_energy_step()
            record = {"step": int(self.step_count), "time": float(self.time)}
            self.add_energy_record(record)
            self.step_schedule.append_history(
                self.history,
                record,
            )

    def run(self, verbose=True, postprocessing=()):
        postprocessing = normalize_callbacks(postprocessing)
        with self.timer.section("IGA initialization"):
            self.precompute()
            self.apply_dirichlet_step()
        with self.timer.section("Output"):
            self.record()
        self.timer.profile0()
        with self.timer.section("Postprocess"):
            for f in postprocessing:
                f()
        for output_index in range(self.total_step):
            for interval_index in range(self.output_interval):
                next_step = self.step_count + 1
                output_due = interval_index + 1 == self.output_interval
                final_step = output_index + 1 == self.total_step and output_due
                record_history = self.step_schedule.history_due(next_step, output=output_due, final=final_step)
                compiling = self.compile_seconds is None
                if compiling:
                    print("Compiling first ... ...")
                    compile_start = time.perf_counter()
                with self.timer.section("IGA explicit step"):
                    self.substep(verbose, record_history=record_history)
                if compiling:
                    ti.sync()
                    self.compile_seconds = time.perf_counter() - compile_start
                    print(f"Compiling time = {self.compile_seconds} \n")
                    self.timer.profile1()
                runtime_checkpoint()
            with self.timer.section("Output"):
                self.record()
            self.timer.profile0()
            with self.timer.section("Postprocess"):
                for f in postprocessing:
                    f()

    def diagnostics_snapshot(self):
        return {
            "schema_version": 1,
            "subsystem": "iga_explicit",
            "time": float(self.time),
            "step": int(self.step_count),
            "timestep": float(self.dt),
            "last_step": self.history[-1] if self.history else None,
        }
