import taichi as ti


@ti.data_oriented
class FEMMultiContactAssembler:
    """One implicit contact contribution per configured FEM body pair.

    Each child owns pair-local broad-phase buffers and constants, while all
    children scatter directly into the same nodal force and sparse matrix.
    This avoids host-side candidate partitioning and preserves the exact
    single-parameter IPC implementation for every pair.
    """

    def __init__(self, contact, assemblers):
        self.contact = contact
        self.assemblers = tuple(assemblers)
        self.is_ipc = all(entry.is_ipc for entry in self.assemblers)
        self.total_energy = ti.field(dtype=ti.f64, shape=())
        self.total_energy[None] = 0.0

    @ti.kernel
    def _accumulate_energy(self, source: ti.template()):
        self.total_energy[None] += source[None]

    def set_stitch_exclusions(self, stitches):
        for assembler in self.assemblers:
            assembler.set_stitch_exclusions(stitches)

    def prepare_iteration_device(self, positions, end_positions=None):
        for assembler in self.assemblers:
            assembler.prepare_iteration_device(positions, end_positions=end_positions)

    def begin_step_device(self, positions, dt):
        for assembler in self.assemblers:
            assembler.begin_step_device(positions, dt)

    def refresh_friction_device(self, positions):
        for assembler in self.assemblers:
            assembler.refresh_friction_device(positions)

    @property
    def activate_friction(self):
        return any(assembler.activate_friction for assembler in self.assemblers)

    def assemble_device(
        self,
        positions,
        force,
        stiffness,
        *,
        need_stiffness=False,
    ):
        self.total_energy[None] = 0.0
        for assembler in self.assemblers:
            force, stiffness = assembler.assemble_device(
                positions,
                force,
                stiffness,
                need_stiffness=need_stiffness,
            )
            self._accumulate_energy(assembler.total_energy)
        return force, stiffness

    def maximum_admissible_step_device(self, positions, direction):
        step = 1.0
        for assembler in self.assemblers:
            step = min(
                step,
                assembler.maximum_admissible_step_device(positions, direction),
            )
        return step

    def device_diagnostics(self):
        pairs = [
            {
                "body_pair": assembler.contact.body_pair,
                **assembler.device_diagnostics(),
            }
            for assembler in self.assemblers
        ]
        return {
            "model": self.contact.model,
            "active_contacts": sum(entry["active_contacts"] for entry in pairs),
            "pt_candidates": sum(entry["pt_candidates"] for entry in pairs),
            "ee_candidates": sum(entry["ee_candidates"] for entry in pairs),
            "body_pairs": pairs,
        }
