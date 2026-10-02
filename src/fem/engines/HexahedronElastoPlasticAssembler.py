"""Updated-Lagrangian HEX8 assembly for shared incremental solid models."""

import numpy as np
import taichi as ti

from src.utils.MatrixFunction import matrix_form


@ti.dataclass
class _IntegrationPointState:
    stress: ti.types.vector(6, float)


@ti.data_oriented
class HexahedronElastoPlasticAssembler:
    """History-dependent Gauss-point update and internal-force assembly.

    The constitutive update reuses GeoTaichi's incremental solid models.  The
    models return Cauchy stress in Voigt order, while HEX8 force assembly uses
    ``P = J sigma F^{-T}`` over the reference quadrature volume.  State and
    stress remain device resident for the complete explicit solve.
    """

    def __init__(self, mesh, element, material):
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError("HEX8 elastoplastic FEM requires an initialized Taichi runtime")
        if mesh.cell_type != "hexahedron":
            raise ValueError("incremental FEM elastoplasticity currently requires HEX8")
        self.mesh = mesh
        self.element = element
        self.material = material
        self.node_count = mesh.number_of_nodes
        self.cell_count = mesh.number_of_cells
        self.nodes_per_cell = 8
        self.quadrature_count = element.quadrature_count
        self.integration_point_count = self.cell_count * self.quadrature_count
        self.real_type = ti.lang.impl.current_cfg().default_fp
        self.numpy_type = np.float64 if self.real_type == ti.f64 else np.float32

        self.connectivity = ti.Vector.field(8, dtype=ti.i32, shape=self.cell_count)
        self.reference_gradient = ti.field(
            dtype=self.real_type,
            shape=(self.cell_count, self.quadrature_count, 8, 3),
        )
        self.reference_weight = ti.field(
            dtype=self.real_type,
            shape=(self.cell_count, self.quadrature_count),
        )
        self.deformation_gradient = ti.Matrix.field(
            3,
            3,
            dtype=self.real_type,
            shape=(self.cell_count, self.quadrature_count),
        )
        self.cauchy_stress = ti.Vector.field(6, dtype=self.real_type, shape=self.integration_point_count)
        self.integration_points = _IntegrationPointState.field(shape=self.integration_point_count)
        self.state_variables = ti.Struct.field(material.get_state_vars(), shape=self.integration_point_count)
        self.previous_deformation_gradient = ti.Matrix.field(
            3,
            3,
            dtype=self.real_type,
            shape=(self.cell_count, self.quadrature_count),
        )
        self.internal_force = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.cell_stress = ti.Matrix.field(3, 3, dtype=self.real_type, shape=self.cell_count)
        self.total_energy = ti.field(dtype=self.real_type, shape=())
        self.minimum_jacobian = ti.field(dtype=self.real_type, shape=())
        self.time_step = ti.field(dtype=self.real_type, shape=())

        self.connectivity.from_numpy(np.ascontiguousarray(element.connectivity, dtype=np.int32))
        self.reference_gradient.from_numpy(np.ascontiguousarray(element.shape_gradients, dtype=self.numpy_type))
        self.reference_weight.from_numpy(np.ascontiguousarray(element.reference_weights, dtype=self.numpy_type))
        self.state_variables.fill(0.0)
        self._initialize_history(
            tuple(
                float(value)
                for value in getattr(
                    material,
                    "fem_initial_stress",
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
            )
        )
        self.internal_force.fill(0.0)
        self.cell_stress.fill(0.0)
        self.total_energy[None] = 0.0
        self.time_step[None] = 0.0

    def bind_positions(self, positions):
        self.bound_positions = positions

    @ti.kernel
    def _initialize_history(self, initial_stress: ti.types.vector(6, float)):
        identity = ti.Matrix.identity(self.real_type, 3)
        for point_id in range(self.integration_point_count):
            self.cauchy_stress[point_id] = initial_stress
            self.integration_points[point_id].stress = initial_stress
            self.material._initialize_vars(point_id, self.integration_points, self.state_variables)
        for cell_id, quadrature_id in ti.ndrange(self.cell_count, self.quadrature_count):
            self.previous_deformation_gradient[cell_id, quadrature_id] = identity
            self.deformation_gradient[cell_id, quadrature_id] = identity

    @ti.func
    def _deformation_gradient(self, cell_id, quadrature_id, positions):
        value = ti.Matrix.zero(self.real_type, 3, 3)
        nodes = self.connectivity[cell_id]
        for local_id in range(8):
            for spatial, material in ti.static(ti.ndrange(3, 3)):
                value[spatial, material] += (
                    positions[nodes[local_id]][spatial]
                    * self.reference_gradient[cell_id, quadrature_id, local_id, material]
                )
        return value

    @ti.kernel
    def _advance_constitutive_state(self, positions: ti.template(), velocity: ti.template()):
        for cell_id, quadrature_id in ti.ndrange(self.cell_count, self.quadrature_count):
            deformation_gradient = self._deformation_gradient(cell_id, quadrature_id, positions)
            jacobian = deformation_gradient.determinant()
            assert jacobian > 1.0e-12, "HEX8 elastoplastic constitutive update requires det(F) > 0"
            previous = self.previous_deformation_gradient[cell_id, quadrature_id]
            incremental_gradient = deformation_gradient @ previous.inverse()
            velocity_gradient = (incremental_gradient - ti.Matrix.identity(self.real_type, 3)) / self.time_step[None]
            point_id = cell_id * self.quadrature_count + quadrature_id
            self.cauchy_stress[point_id] = self.material.ComputeStress(
                point_id,
                self.cauchy_stress[point_id],
                velocity_gradient,
                self.state_variables,
                self.time_step,
            )
            self.deformation_gradient[cell_id, quadrature_id] = deformation_gradient
            self.previous_deformation_gradient[cell_id, quadrature_id] = deformation_gradient

    @ti.kernel
    def _assemble_force(self, positions: ti.template()):
        for cell_id in range(self.cell_count):
            nodes = self.connectivity[cell_id]
            averaged_stress = ti.Matrix.zero(self.real_type, 3, 3)
            total_weight = 0.0
            for quadrature_id in range(self.quadrature_count):
                deformation_gradient = self._deformation_gradient(cell_id, quadrature_id, positions)
                jacobian = deformation_gradient.determinant()
                assert jacobian > 1.0e-12, "HEX8 elastoplastic force assembly requires det(F) > 0"
                point_id = cell_id * self.quadrature_count + quadrature_id
                cauchy = matrix_form(self.cauchy_stress[point_id])
                first_piola = jacobian * cauchy @ deformation_gradient.inverse().transpose()
                weight = self.reference_weight[cell_id, quadrature_id]
                total_weight += weight
                averaged_stress += weight * cauchy
                self.deformation_gradient[cell_id, quadrature_id] = deformation_gradient
                for local_id in range(8):
                    for spatial in ti.static(range(3)):
                        value = 0.0
                        for material in ti.static(range(3)):
                            value += (
                                first_piola[spatial, material]
                                * self.reference_gradient[
                                    cell_id,
                                    quadrature_id,
                                    local_id,
                                    material,
                                ]
                            )
                        ti.atomic_add(
                            self.internal_force[nodes[local_id]][spatial],
                            weight * value,
                        )
            self.cell_stress[cell_id] = averaged_stress / total_weight

    @ti.kernel
    def _reduce_minimum_jacobian(self, positions: ti.template()):
        self.minimum_jacobian[None] = 1.0e30
        for cell_id, quadrature_id in ti.ndrange(self.cell_count, self.quadrature_count):
            value = self._deformation_gradient(cell_id, quadrature_id, positions).determinant()
            ti.atomic_min(self.minimum_jacobian[None], value)

    def advance_state(self, positions, velocity, dt):
        self.time_step[None] = float(dt)
        self._advance_constitutive_state(positions, velocity)

    def assemble_device(self, positions=None, need_stiffness=False):
        if need_stiffness:
            raise ValueError("incremental HEX8 elastoplasticity is an explicit FEM path")
        return self.assemble_force_device(positions), None

    def assemble_force_device(self, positions=None):
        positions = getattr(self, "bound_positions") if positions is None else positions
        self.internal_force.fill(0.0)
        self.total_energy[None] = 0.0
        self._assemble_force(positions)
        return self.internal_force

    def minimum_jacobian_ratio_device(self, positions):
        self._reduce_minimum_jacobian(positions)
        return float(self.minimum_jacobian[None])

    def assemble(self, positions, need_stiffness=False, need_stress=False):
        if hasattr(positions, "to_numpy"):
            source = positions
        else:
            values = np.ascontiguousarray(positions, dtype=self.numpy_type)
            source = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
            source.from_numpy(values)
        force, stiffness = self.assemble_device(source, need_stiffness=need_stiffness)
        stress = self.cell_stress.to_numpy() if need_stress else None
        return (
            float(self.total_energy[None]),
            force.to_numpy(),
            stiffness,
            stress,
        )

    def deformation_gradients(self, positions):
        self.assemble(positions)
        return self.deformation_gradient.to_numpy()


__all__ = ["HexahedronElastoPlasticAssembler"]
