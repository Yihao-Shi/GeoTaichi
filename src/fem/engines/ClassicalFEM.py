"""Explicit and implicit classical FEM integrators with device assembly."""

from src.fem.engines.ExplicitFEM import ExplicitFEM
from src.fem.engines.ImplicitFEM import ImplicitFEM
from src.fem.engines.SparseMatrix import normalize_assemble_type, normalize_linear_solver
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.fem.engines.HexahedronElastoPlasticAssembler import (
    HexahedronElastoPlasticAssembler,
)
from src.utils.linalg import no_operation


class _ClassicalAssemblyMixin:
    def _create_classical_assembler(self, kwargs):
        self.backend = "taichi"
        if bool(getattr(self.material, "is_fem_elastoplastic", False)):
            if isinstance(self, ImplicitFEM):
                raise ValueError("HEX8 elastoplastic FEM currently uses the explicit " "history-update integrator")
            self.assemble_type = None
            self.linear_solver = None
            self.project_pd = False
            self.classical_assembler = HexahedronElastoPlasticAssembler(self.mesh, self.element, self.material)
            self.classical_assembler.bind_positions(self.state.position)
            # Preprocess the configured rest-to-current deformation once so
            # prestrained meshes and initial output own an accepted Gauss-point
            # state before the first explicit time step.
            self.classical_assembler.advance_state(self.state.position, self.state.velocity, self.dt)
            self._bind_classical_assembly_functions()
            return
        self.assemble_type = normalize_assemble_type(kwargs.get("assemble_type", kwargs.get("assembly", "Hash")))
        self.linear_solver = normalize_linear_solver(kwargs.get("linear_solver", "PCG"))
        self.project_pd = bool(
            kwargs.get(
                "project_pd",
                kwargs.get("project_hessian_pd", self.linear_solver == "PCG"),
            )
        )
        if self.linear_solver == "PCG" and not self.project_pd:
            raise ValueError("classical FEM PCG requires project_pd=True; use BiCGSTAB for an unprojected tangent")
        self.linear_solver_tolerance = float(kwargs.get("linear_solver_tolerance", 1.0e-10))
        self.linear_solver_relative_tolerance = float(kwargs.get("linear_solver_relative_tolerance", 0.0))
        self.linear_solver_max_iters = int(kwargs.get("linear_solver_max_iters", max(100, 10 * self.degree_of_freedom)))
        self.classical_assembler = ClassicalAssembler(
            self.mesh,
            self.element,
            self.material,
            project_pd=self.project_pd,
            assemble_type=self.assemble_type,
            linear_solver=self.linear_solver,
            linear_solver_tolerance=self.linear_solver_tolerance,
            linear_solver_relative_tolerance=self.linear_solver_relative_tolerance,
            linear_solver_max_iters=self.linear_solver_max_iters,
            spatial_dimension=self.dimension,
            allocate_hessian=isinstance(self, ImplicitFEM),
        )
        self.classical_assembler.bind_positions(self.state.position)
        if isinstance(self, ImplicitFEM):
            self._validate_pcg_projection()
        self._bind_classical_assembly_functions()

    def _bind_classical_assembly_functions(self):
        if self.contact_assembler is None:
            self._assemble_internal_force_device = self._assemble_mechanical_force_device
            self._assemble_internal_device_at = self._assemble_mechanical_device_at
            self._internal_energy_device = self._mechanical_energy_device
        else:
            self._assemble_internal_force_device = self._assemble_contact_force_device
            self._assemble_internal_device_at = self._assemble_contact_device_at
            self._internal_energy_device = self._contact_energy_device
        self._advance_constitutive_state = (
            self._advance_classical_constitutive_state
            if isinstance(self.classical_assembler, HexahedronElastoPlasticAssembler)
            else no_operation
        )

    def assemble_internal(self, positions=None, need_stiffness=False, need_stress=False):
        positions = self.positions if positions is None else positions
        mechanical = self.classical_assembler.assemble(positions, need_stiffness, need_stress)
        combine = getattr(self, "_combine_contact_assembly", None)
        if combine is not None:
            return combine(mechanical, positions, need_stiffness)
        return mechanical

    def _assemble_output_diagnostics(self):
        return self.classical_assembler.assemble(self.positions, need_stress=True)

    def _assemble_internal_device(self, need_stiffness=False):
        return self._assemble_internal_device_at(self.state.position, need_stiffness=need_stiffness)

    def _assemble_mechanical_force_device(self):
        force = self.classical_assembler.assemble_force_device(self.state.position)
        self._current_stiffness = None
        return force

    def _assemble_contact_force_device(self):
        force = self.classical_assembler.assemble_force_device(self.state.position)
        force, _ = self.contact_assembler.assemble_device(
            self.state.position,
            force,
            None,
            need_stiffness=False,
        )
        self._current_stiffness = None
        return force

    def _assemble_mechanical_device_at(self, positions, need_stiffness=False):
        force, stiffness = self.classical_assembler.assemble_device(positions, need_stiffness=need_stiffness)
        self._current_stiffness = stiffness
        return force

    def _assemble_contact_device_at(self, positions, need_stiffness=False):
        force, stiffness = self.classical_assembler.assemble_device(positions, need_stiffness=need_stiffness)
        force, stiffness = self.contact_assembler.assemble_device(
            positions,
            force,
            stiffness,
            need_stiffness=need_stiffness,
        )
        self._current_stiffness = stiffness
        return force

    def _mechanical_energy_device(self):
        return float(self.classical_assembler.total_energy[None])

    def _contact_energy_device(self):
        return self._mechanical_energy_device() + float(self.contact_assembler.total_energy[None])

    def _minimum_jacobian_ratio_device(self, positions):
        return self.classical_assembler.minimum_jacobian_ratio_device(positions)

    def _maximum_admissible_step_device(self):
        contact_step = super()._maximum_admissible_step_device()
        material_step = self.classical_assembler.maximum_material_step_device(self.state.position, self.state.direction)
        return min(contact_step, material_step)

    def _advance_classical_constitutive_state(self):
        self.classical_assembler.advance_state(self.state.position, self.state.velocity, self.dt)


class ClassicalExplicitFEM(_ClassicalAssemblyMixin, ExplicitFEM):
    def __init__(self, mesh, material, dirichlet=None, neumann=None, **kwargs):
        super().__init__(mesh, material, dirichlet, neumann, **kwargs)
        self._create_classical_assembler(kwargs)
        self._initialize_soft_particle_contact(kwargs)
        self.bind_runtime_functions()


class ClassicalImplicitFEM(_ClassicalAssemblyMixin, ImplicitFEM):
    def __init__(self, mesh, material, dirichlet=None, neumann=None, **kwargs):
        super().__init__(mesh, material, dirichlet, neumann, **kwargs)
        self._create_classical_assembler(kwargs)


__all__ = ["ClassicalExplicitFEM", "ClassicalImplicitFEM"]
