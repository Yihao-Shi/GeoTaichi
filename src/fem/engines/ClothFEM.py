"""Explicit and implicit integrators for triangular cloth FEM."""

import numpy as np

from src.fem.cloth.ClothAssembler import ClothAssembler
from src.fem.engines.ExplicitFEM import ExplicitFEM
from src.fem.engines.ImplicitFEM import ImplicitFEM
from src.fem.engines.SparseMatrix import normalize_assemble_type, normalize_linear_solver


class _ClothAssemblyMixin:
    def _create_cloth_assembler(self, kwargs):
        self.backend = "taichi"
        self.assemble_type = normalize_assemble_type(kwargs.get("assemble_type", kwargs.get("assembly", "Hash")))
        self.linear_solver = normalize_linear_solver(kwargs.get("linear_solver", "PCG"))
        self.project_pd = bool(
            kwargs.get(
                "project_pd",
                kwargs.get("project_hessian_pd", self.linear_solver == "PCG"),
            )
        )
        self.project_bending_pd = bool(kwargs.get("project_bending_pd", self.project_pd))
        if self.linear_solver == "PCG" and not (self.project_pd and self.project_bending_pd):
            raise ValueError(
                "ClothFEM PCG requires project_pd=True and project_bending_pd=True; "
                "use BiCGSTAB for an unprojected tangent"
            )
        self.linear_solver_tolerance = float(kwargs.get("linear_solver_tolerance", 1.0e-10))
        self.linear_solver_relative_tolerance = float(kwargs.get("linear_solver_relative_tolerance", 0.0))
        self.linear_solver_max_iters = int(kwargs.get("linear_solver_max_iters", max(100, 10 * self.degree_of_freedom)))
        self.cloth_assembler = ClothAssembler(
            self.mesh,
            self.material,
            bending_model=kwargs.get("bending_model", getattr(self.material, "bending_model", "Quadratic")),
            cloth_energies=kwargs.get("cloth_energies", ()),
            project_pd=self.project_pd,
            project_bending_pd=self.project_bending_pd,
            assemble_type=self.assemble_type,
            linear_solver=self.linear_solver,
            linear_solver_tolerance=self.linear_solver_tolerance,
            linear_solver_relative_tolerance=self.linear_solver_relative_tolerance,
            linear_solver_max_iters=self.linear_solver_max_iters,
        )
        self.cloth_assembler.bind_positions(self.state.position)
        if self.contact_assembler is not None:
            self.contact_assembler.set_stitch_exclusions(self.cloth_assembler.energy_data.stitch_nodes)
        if isinstance(self, ImplicitFEM):
            self._validate_pcg_projection()
        self._bind_cloth_assembly_functions()

    def _bind_cloth_assembly_functions(self):
        if self.contact_assembler is None:
            self._assemble_internal_force_device = self._assemble_cloth_force_device
            self._assemble_internal_device_at = self._assemble_cloth_device_at
            self._internal_energy_device = self._cloth_energy_device
        else:
            self._assemble_internal_force_device = self._assemble_cloth_contact_force_device
            self._assemble_internal_device_at = self._assemble_cloth_contact_device_at
            self._internal_energy_device = self._cloth_contact_energy_device

    def assemble_internal(self, positions=None, need_stiffness=False, need_stress=False):
        positions = self.positions if positions is None else positions
        mechanical = self.cloth_assembler.assemble_system(positions, need_stiffness, need_stress)
        combine = getattr(self, "_combine_contact_assembly", None)
        if combine is not None:
            return combine(mechanical, positions, need_stiffness)
        return mechanical

    def _assemble_output_diagnostics(self):
        return self.cloth_assembler.assemble_system(self.positions, need_stress=True)

    def _assemble_internal_device(self, need_stiffness=False):
        return self._assemble_internal_device_at(self.state.position, need_stiffness=need_stiffness)

    def _assemble_cloth_force_device(self):
        force, _ = self.cloth_assembler.assemble_device(self.state.position, need_stiffness=False)
        self._current_stiffness = None
        return force

    def _assemble_cloth_contact_force_device(self):
        force = self._assemble_cloth_force_device()
        force, _ = self.contact_assembler.assemble_device(
            self.state.position,
            force,
            None,
            need_stiffness=False,
        )
        return force

    def _assemble_cloth_device_at(self, positions, need_stiffness=False):
        force, stiffness = self.cloth_assembler.assemble_device(positions, need_stiffness=need_stiffness)
        self._current_stiffness = stiffness
        return force

    def _assemble_cloth_contact_device_at(self, positions, need_stiffness=False):
        force, stiffness = self.cloth_assembler.assemble_device(positions, need_stiffness=need_stiffness)
        force, stiffness = self.contact_assembler.assemble_device(
            positions,
            force,
            stiffness,
            need_stiffness=need_stiffness,
        )
        self._current_stiffness = stiffness
        return force

    def _cloth_energy_device(self):
        return float(self.cloth_assembler.total_energy[None])

    def _cloth_contact_energy_device(self):
        return self._cloth_energy_device() + float(self.contact_assembler.total_energy[None])

    def _minimum_jacobian_ratio_device(self, positions):
        return self.cloth_assembler.minimum_jacobian_ratio_device(positions)


class ClothExplicitFEM(_ClothAssemblyMixin, ExplicitFEM):
    def __init__(self, mesh, material, dirichlet=None, neumann=None, **kwargs):
        if mesh.cell_type != "triangle":
            raise ValueError("ClothFEM requires a TRI3 mesh")
        super().__init__(mesh, material, dirichlet, neumann, **kwargs)
        self._create_cloth_assembler(kwargs)
        self.bind_runtime_functions()

    def stable_time_step(self, cfl=0.5):
        membrane_step = super().stable_time_step(cfl)
        bending_modulus = float(self.material.quadratic_bending_modulus)
        if bending_modulus <= 0.0:
            return membrane_step
        minimum_edge = np.inf
        for triangle in self.mesh.cells:
            points = self.mesh.rest_shape[triangle]
            minimum_edge = min(
                minimum_edge,
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[1]),
                np.linalg.norm(points[0] - points[2]),
            )
        areal_density = self.material.density * self.material.thickness
        bending_step = float(cfl) * minimum_edge**2 * np.sqrt(areal_density / bending_modulus)
        return min(membrane_step, bending_step)


class ClothImplicitFEM(_ClothAssemblyMixin, ImplicitFEM):
    def __init__(self, mesh, material, dirichlet=None, neumann=None, **kwargs):
        if mesh.cell_type != "triangle":
            raise ValueError("ClothFEM requires a TRI3 mesh")
        super().__init__(mesh, material, dirichlet, neumann, **kwargs)
        self._create_cloth_assembler(kwargs)


__all__ = ["ClothExplicitFEM", "ClothImplicitFEM"]
