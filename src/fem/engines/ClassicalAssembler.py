"""Device assembly for classical total-Lagrangian FEM elements."""

import numpy as np
import taichi as ti

from src.contact_detection.continuous_contact_detection import (
    deformation_gradient_ccd,
)
from src.fem.engines.SparseMatrix import FEMSparseMatrix
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd


def _require_initialized_taichi():
    if ti.lang.impl.get_runtime().prog is None:
        raise RuntimeError(
            "classical FEM device assembly requires an initialized runtime; "
            "call geotaichi.init(...) or taichi.init(...) first"
        )


@ti.data_oriented
class ClassicalAssembler:
    """Device energy, internal-force and element-tangent assembly.

    TRI3 uses the local-frame formula ``F = Jx @ inv(JX)``. TET4 and HEX8 use
    the standard 3x3 total-
    Lagrangian map.  Element constitutive work and Hessian scattering run in
    Taichi; the selected COO/HashTriplet matrix then uses Scipy, PCG or
    BiCGSTAB for the global solve.
    """

    def __init__(
        self,
        mesh,
        element,
        material,
        project_pd=True,
        tangent_epsilon=None,
        assemble_type="Hash",
        linear_solver="PCG",
        linear_solver_tolerance=1.0e-10,
        linear_solver_max_iters=500,
        spatial_dimension=3,
        allocate_hessian=True,
        linear_solver_relative_tolerance=0.0,
    ):
        _require_initialized_taichi()
        self.mesh = mesh
        self.element = element
        self.material = material
        self.node_count = mesh.number_of_nodes
        self.cell_count = mesh.number_of_cells
        self.nodes_per_cell = element.nodes_per_cell
        self.quadrature_count = element.quadrature_count
        self.material_dimension = element.material_dimension
        self.constitutive_dimension = element.constitutive_dimension
        self.local_dofs = 3 * self.nodes_per_cell
        self._stiffness_unique_block_pair_count = None
        self.is_membrane = bool(element.is_classical_membrane)
        self.is_axisymmetric = bool(element.is_axisymmetric)
        self.is_tetrahedron = element.name == "TET4"
        self.spatial_dimension = int(spatial_dimension)
        self.is_planar = self.spatial_dimension == 2 and self.is_membrane
        self.project_pd = bool(project_pd)
        self.allocate_hessian = bool(allocate_hessian)
        self.assemble_type = assemble_type
        self.linear_solver = linear_solver
        self.linear_solver_tolerance = float(linear_solver_tolerance)
        self.linear_solver_relative_tolerance = float(linear_solver_relative_tolerance)
        self.linear_solver_max_iters = int(linear_solver_max_iters)
        # Accepted for compatibility with older inputs. The current tangent is
        # analytic and therefore has no differentiation step size.
        del tangent_epsilon

        material_name = type(material).__name__
        if material_name not in (
            "StVenantKirchhoffModel",
            "NeoHookeanModel",
        ):
            raise TypeError(
                "classical FEM device assembly currently supports StVenantKirchhoffModel and NeoHookeanModel"
            )
        if self.is_membrane and material_name == "NeoHookeanModel":
            raise ValueError("classical TRI3 currently uses the StVK plane-stress constitutive law")

        self.real_type = ti.lang.impl.current_cfg().default_fp
        self.numpy_type = np.float64 if self.real_type == ti.f64 else np.float32
        self.positions = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.stiffness_positions = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.bound_positions = self.positions
        self.internal_force = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.connectivity = ti.Vector.field(self.nodes_per_cell, dtype=ti.i32, shape=self.cell_count)
        self.shape_gradients = ti.field(
            dtype=self.real_type,
            shape=(self.cell_count, self.quadrature_count, self.nodes_per_cell, self.material_dimension),
        )
        self.reference_weights = ti.field(dtype=self.real_type, shape=(self.cell_count, self.quadrature_count))
        self.shape_values = ti.field(
            dtype=self.real_type,
            shape=(self.quadrature_count, self.nodes_per_cell),
        )
        self.reference_radii = ti.field(
            dtype=self.real_type,
            shape=(self.cell_count, self.quadrature_count),
        )
        self.deformation_gradient = ti.Matrix.field(
            self.constitutive_dimension,
            self.constitutive_dimension,
            dtype=self.real_type,
            shape=(self.cell_count, self.quadrature_count),
        )
        self.total_energy = ti.field(dtype=self.real_type, shape=())
        self.minimum_jacobian = ti.field(dtype=self.real_type, shape=())
        self.maximum_material_step = ti.field(dtype=self.real_type, shape=())
        # Stress output is reconstructed from deformation gradients and
        # mapped directly to nodes. The former cell-sized field was never
        # written or read by the classical assembler.
        self.cell_stress = ti.Matrix.field(3, 3, dtype=self.real_type, shape=1)
        self._stiffness_matrix = None

        self.connectivity.from_numpy(np.ascontiguousarray(element.connectivity, dtype=np.int32))
        self.shape_gradients.from_numpy(element.shape_gradients.astype(self.numpy_type))
        self.reference_weights.from_numpy(element.reference_weights.astype(self.numpy_type))
        self.shape_values.from_numpy(element.shape_values.astype(self.numpy_type))
        if self.is_axisymmetric:
            self.reference_radii.from_numpy(element.reference_radii.astype(self.numpy_type))
        else:
            self.reference_radii.fill(1.0)

    def bind_positions(self, positions):
        """Bind the assembler directly to the integrator's persistent field."""
        self.bound_positions = positions

    @ti.kernel
    def _copy_positions_from_field(self, source: ti.template(), target: ti.template()):
        for node in range(self.node_count):
            target[node] = source[node]

    @ti.func
    def _current_frame(self, coordinates):
        edge_01 = ti.Vector([coordinates[1, i] - coordinates[0, i] for i in ti.static(range(3))])
        edge_02 = ti.Vector([coordinates[2, i] - coordinates[0, i] for i in ti.static(range(3))])
        basis_1 = edge_01.normalized(1.0e-30)
        basis_3 = edge_01.cross(edge_02).normalized(1.0e-30)
        basis_2 = basis_3.cross(basis_1)
        return ti.Matrix.rows([basis_1, basis_2, basis_3])

    @ti.func
    def _material_response(self, deformation_gradient):
        return (
            self.material.strain_energy_density(deformation_gradient),
            self.material.first_piola_stress(deformation_gradient),
        )

    @ti.func
    def _axisymmetric_deformation_gradient(self, element_id, quadrature_id, coordinates):
        deformation_gradient = ti.Matrix.zero(self.real_type, 3, 3)
        for local_id in ti.static(range(self.nodes_per_cell)):
            for spatial, material_component in ti.static(ti.ndrange(2, 2)):
                deformation_gradient[spatial, material_component] += (
                    coordinates[local_id, spatial]
                    * self.shape_gradients[
                        element_id,
                        quadrature_id,
                        local_id,
                        material_component,
                    ]
                )
            deformation_gradient[2, 2] += (
                self.shape_values[quadrature_id, local_id]
                * (coordinates[local_id, 0] - ti.static(self.element.axis_offset))
                / self.reference_radii[element_id, quadrature_id]
            )
        return deformation_gradient

    @ti.func
    def _axisymmetric_local_derivative(self, element_id, quadrature_id, local_id, component):
        derivative = ti.Matrix.zero(self.real_type, 3, 3)
        for material_component in ti.static(range(2)):
            derivative[component, material_component] = self.shape_gradients[
                element_id, quadrature_id, local_id, material_component
            ]
        if component == 0:
            derivative[2, 2] = (
                self.shape_values[quadrature_id, local_id] / self.reference_radii[element_id, quadrature_id]
            )
        return derivative

    @ti.func
    def _tetrahedron_deformation_gradient(self, element_id, coordinates):
        """Evaluate the constant TET4 map ``F = Ds @ inverse(Dm)``."""
        deformation_gradient = ti.Matrix.zero(self.real_type, 3, 3)
        # For Dm=[X1-X0, X2-X0, X3-X0], reference gradients 1--3
        # are the rows of inverse(Dm), precomputed once during mesh setup.
        for spatial, material_component in ti.static(ti.ndrange(3, 3)):
            for edge in ti.static(range(3)):
                deformation_gradient[spatial, material_component] += (
                    coordinates[edge + 1, spatial] - coordinates[0, spatial]
                ) * self.shape_gradients[element_id, 0, edge + 1, material_component]
        return deformation_gradient

    @ti.func
    def _volume_deformation_gradient(self, element_id, quadrature_id, coordinates):
        deformation_gradient = ti.Matrix.zero(self.real_type, 3, 3)
        if ti.static(self.is_tetrahedron):
            deformation_gradient = self._tetrahedron_deformation_gradient(element_id, coordinates)
        else:
            for spatial, material_component in ti.static(ti.ndrange(3, 3)):
                for local_id in ti.static(range(self.nodes_per_cell)):
                    deformation_gradient[spatial, material_component] += (
                        coordinates[local_id, spatial]
                        * self.shape_gradients[
                            element_id,
                            quadrature_id,
                            local_id,
                            material_component,
                        ]
                    )
        return deformation_gradient

    @ti.func
    def _element_energy_gradient(
        self,
        element_id,
        local_coordinates,
        record_energy: ti.template(),
    ):
        coordinates = ti.Matrix.zero(self.real_type, self.nodes_per_cell, 3)
        for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
            coordinates[local_id, component] = local_coordinates[3 * local_id + component]
        energy = 0.0
        gradient = ti.Vector.zero(self.real_type, self.local_dofs)
        for quadrature_id in ti.static(range(self.quadrature_count)):
            weight = self.reference_weights[element_id, quadrature_id]
            if ti.static(self.is_axisymmetric):
                deformation_gradient = self._axisymmetric_deformation_gradient(element_id, quadrature_id, coordinates)
                first_piola = ti.Matrix.zero(self.real_type, 3, 3)
                if ti.static(record_energy):
                    self.deformation_gradient[element_id, quadrature_id] = deformation_gradient
                    density, response = self._material_response(deformation_gradient)
                    first_piola = response
                    energy += density * weight
                else:
                    first_piola = self.material.first_piola_stress(deformation_gradient)
                for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 2)):
                    derivative = self._axisymmetric_local_derivative(element_id, quadrature_id, local_id, component)
                    value = 0.0
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        value += first_piola[row, column] * derivative[row, column]
                    gradient[3 * local_id + component] += weight * value
            elif ti.static(self.is_membrane):
                deformation_gradient = ti.Matrix.zero(self.real_type, 3, 2)
                for spatial, material_component in ti.static(ti.ndrange(3, 2)):
                    for local_id in ti.static(range(self.nodes_per_cell)):
                        deformation_gradient[spatial, material_component] += (
                            coordinates[local_id, spatial]
                            * self.shape_gradients[
                                element_id,
                                quadrature_id,
                                local_id,
                                material_component,
                            ]
                        )
                if ti.static(record_energy):
                    density = self.material.surface_strain_energy_density(deformation_gradient)
                    energy += density * weight
                first_piola = self.material.surface_first_piola_stress(deformation_gradient)
                for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                    value = 0.0
                    for material_component in ti.static(range(2)):
                        value += (
                            first_piola[component, material_component]
                            * self.shape_gradients[element_id, quadrature_id, local_id, material_component]
                        )
                    gradient[3 * local_id + component] += weight * value
            else:
                deformation_gradient = self._volume_deformation_gradient(element_id, quadrature_id, coordinates)
                first_piola = ti.Matrix.zero(self.real_type, 3, 3)
                if ti.static(record_energy):
                    self.deformation_gradient[element_id, quadrature_id] = deformation_gradient
                    density, response = self._material_response(deformation_gradient)
                    first_piola = response
                    energy += density * weight
                else:
                    first_piola = self.material.first_piola_stress(deformation_gradient)
                for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                    value = 0.0
                    for material_component in ti.static(range(3)):
                        value += (
                            first_piola[component, material_component]
                            * self.shape_gradients[element_id, quadrature_id, local_id, material_component]
                        )
                    gradient[3 * local_id + component] += weight * value
        return energy, gradient

    @ti.func
    def _gather_local_coordinates(self, element_id, positions: ti.template()):
        coordinates = ti.Vector.zero(self.real_type, self.local_dofs)
        cell = self.connectivity[element_id]
        for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
            coordinates[3 * local_id + component] = positions[cell[local_id]][component]
        return coordinates

    @ti.kernel
    def _assemble_energy_force(self, positions: ti.template()):
        for element_id in range(self.cell_count):
            coordinates = self._gather_local_coordinates(element_id, positions)
            energy, gradient = self._element_energy_gradient(element_id, coordinates, True)
            ti.atomic_add(self.total_energy[None], energy)
            cell = self.connectivity[element_id]
            for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                ti.atomic_add(
                    self.internal_force[cell[local_id]][component],
                    gradient[3 * local_id + component],
                )

    @ti.kernel
    def _assemble_force(self, positions: ti.template()):
        """Assemble explicit force without output-only energy/F writes."""
        for element_id in range(self.cell_count):
            coordinates = self._gather_local_coordinates(element_id, positions)
            _, gradient = self._element_energy_gradient(element_id, coordinates, False)
            cell = self.connectivity[element_id]
            for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                ti.atomic_add(
                    self.internal_force[cell[local_id]][component],
                    gradient[3 * local_id + component],
                )

    @ti.kernel
    def _store_deformation_gradients(self, positions: ti.template()):
        for element_id in range(self.cell_count):
            coordinates = ti.Matrix.zero(self.real_type, self.nodes_per_cell, 3)
            cell = self.connectivity[element_id]
            for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                coordinates[local_id, component] = positions[cell[local_id]][component]
            for quadrature_id in ti.static(range(self.quadrature_count)):
                if ti.static(self.is_axisymmetric):
                    self.deformation_gradient[element_id, quadrature_id] = self._axisymmetric_deformation_gradient(
                        element_id, quadrature_id, coordinates
                    )
                elif ti.static(self.is_membrane):
                    frame = self._current_frame(coordinates)
                    value = ti.Matrix.zero(self.real_type, 2, 2)
                    for i, j in ti.static(ti.ndrange(2, 2)):
                        for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                            value[i, j] += (
                                frame[i, component]
                                * coordinates[local_id, component]
                                * self.shape_gradients[element_id, quadrature_id, local_id, j]
                            )
                    self.deformation_gradient[element_id, quadrature_id] = value
                else:
                    value = self._volume_deformation_gradient(element_id, quadrature_id, coordinates)
                    self.deformation_gradient[element_id, quadrature_id] = value

    @ti.kernel
    def _assemble_element_stiffness_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        scalar_offset: ti.i32,
        block_offset: ti.i32,
        assemble_hash: ti.template(),
    ):
        for element_id in range(self.cell_count):
            coordinates = ti.Matrix.zero(self.real_type, self.nodes_per_cell, 3)
            cell = self.connectivity[element_id]
            for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                coordinates[local_id, component] = positions[cell[local_id]][component]
            for local_i, local_j in ti.static(ti.ndrange(self.nodes_per_cell, self.nodes_per_cell)):
                if ti.static(assemble_hash):
                    if local_i != local_j:
                        pair = (
                            element_id * self.nodes_per_cell * (self.nodes_per_cell - 1)
                            + local_i * (self.nodes_per_cell - 1)
                            + local_j
                            - ti.cast(local_j > local_i, ti.i32)
                        )
                        target.initialize_raw_block_slot(block_offset + pair, cell[local_i], cell[local_j])
                else:
                    block = (
                        element_id * self.nodes_per_cell * self.nodes_per_cell + local_i * self.nodes_per_cell + local_j
                    )
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        entry = scalar_offset + 9 * block + 3 * row + column
                        target.rows[entry] = 3 * cell[local_i] + row
                        target.cols[entry] = 3 * cell[local_j] + column
                        target.data[entry] = 0.0
            for quadrature_id in ti.static(range(self.quadrature_count)):
                weight = self.reference_weights[element_id, quadrature_id]
                if ti.static(self.is_axisymmetric):
                    deformation_gradient = self._axisymmetric_deformation_gradient(
                        element_id, quadrature_id, coordinates
                    )
                    material_tangent = self.material.first_piola_tangent(deformation_gradient)
                    material_tangent = 0.5 * (material_tangent + material_tangent.transpose())
                    if ti.static(self.project_pd):
                        material_tangent = psd_project_nd(material_tangent)
                    for local_i, local_j in ti.static(ti.ndrange(self.nodes_per_cell, self.nodes_per_cell)):
                        block = ti.Matrix.zero(self.real_type, 3, 3)
                        for component_i, component_j in ti.static(ti.ndrange(2, 2)):
                            derivative_i = self._axisymmetric_local_derivative(
                                element_id, quadrature_id, local_i, component_i
                            )
                            derivative_j = self._axisymmetric_local_derivative(
                                element_id, quadrature_id, local_j, component_j
                            )
                            for column_i, row_i, column_j, row_j in ti.ndrange(3, 3, 3, 3):
                                block[component_i, component_j] += weight * (
                                    derivative_i[row_i, column_i]
                                    * material_tangent[
                                        row_i + 3 * column_i,
                                        row_j + 3 * column_j,
                                    ]
                                    * derivative_j[row_j, column_j]
                                )
                        if ti.static(assemble_hash):
                            if local_i == local_j:
                                target.add_block_entry(cell[local_i], cell[local_j], block)
                            else:
                                pair = (
                                    element_id * self.nodes_per_cell * (self.nodes_per_cell - 1)
                                    + local_i * (self.nodes_per_cell - 1)
                                    + local_j
                                    - ti.cast(local_j > local_i, ti.i32)
                                )
                                target.atomic_add_raw_block_slot(block_offset + pair, block)
                        else:
                            base = scalar_offset + 9 * (
                                element_id * self.nodes_per_cell * self.nodes_per_cell
                                + local_i * self.nodes_per_cell
                                + local_j
                            )
                            for row, column in ti.static(ti.ndrange(3, 3)):
                                target.data[base + 3 * row + column] += block[row, column]
                elif ti.static(self.is_membrane):
                    deformation_gradient = ti.Matrix.zero(self.real_type, 3, 2)
                    for spatial, material_component in ti.static(ti.ndrange(3, 2)):
                        for local_id in ti.static(range(self.nodes_per_cell)):
                            deformation_gradient[spatial, material_component] += (
                                coordinates[local_id, spatial]
                                * self.shape_gradients[
                                    element_id,
                                    quadrature_id,
                                    local_id,
                                    material_component,
                                ]
                            )
                    material_tangent = self.material.surface_first_piola_tangent(deformation_gradient)
                    material_tangent = 0.5 * (material_tangent + material_tangent.transpose())
                    if ti.static(self.project_pd):
                        material_tangent = psd_project_nd(material_tangent)
                    for local_i, local_j in ti.static(ti.ndrange(self.nodes_per_cell, self.nodes_per_cell)):
                        block = ti.Matrix.zero(self.real_type, 3, 3)
                        for spatial_i, spatial_j in ti.static(ti.ndrange(3, 3)):
                            for material_i, material_j in ti.static(ti.ndrange(2, 2)):
                                block[spatial_i, spatial_j] += weight * (
                                    self.shape_gradients[element_id, quadrature_id, local_i, material_i]
                                    * material_tangent[
                                        3 * material_i + spatial_i,
                                        3 * material_j + spatial_j,
                                    ]
                                    * self.shape_gradients[element_id, quadrature_id, local_j, material_j]
                                )
                        if ti.static(assemble_hash):
                            if local_i == local_j:
                                target.add_block_entry(cell[local_i], cell[local_j], block)
                            else:
                                pair = (
                                    element_id * self.nodes_per_cell * (self.nodes_per_cell - 1)
                                    + local_i * (self.nodes_per_cell - 1)
                                    + local_j
                                    - ti.cast(local_j > local_i, ti.i32)
                                )
                                target.atomic_add_raw_block_slot(block_offset + pair, block)
                        else:
                            base = scalar_offset + 9 * (
                                element_id * self.nodes_per_cell * self.nodes_per_cell
                                + local_i * self.nodes_per_cell
                                + local_j
                            )
                            for row, column in ti.static(ti.ndrange(3, 3)):
                                target.data[base + 3 * row + column] += block[row, column]
                else:
                    deformation_gradient = self._volume_deformation_gradient(element_id, quadrature_id, coordinates)
                    material_tangent = self.material.first_piola_tangent(deformation_gradient)
                    material_tangent = 0.5 * (material_tangent + material_tangent.transpose())
                    if ti.static(self.project_pd):
                        material_tangent = psd_project_nd(material_tangent)
                    for local_i, local_j in ti.static(ti.ndrange(self.nodes_per_cell, self.nodes_per_cell)):
                        block = ti.Matrix.zero(self.real_type, 3, 3)
                        for spatial_i, spatial_j in ti.static(ti.ndrange(3, 3)):
                            for material_i, material_j in ti.static(ti.ndrange(3, 3)):
                                block[spatial_i, spatial_j] += weight * (
                                    self.shape_gradients[element_id, quadrature_id, local_i, material_i]
                                    * material_tangent[
                                        3 * material_i + spatial_i,
                                        3 * material_j + spatial_j,
                                    ]
                                    * self.shape_gradients[element_id, quadrature_id, local_j, material_j]
                                )
                        if ti.static(assemble_hash):
                            if local_i == local_j:
                                target.add_block_entry(cell[local_i], cell[local_j], block)
                            else:
                                pair = (
                                    element_id * self.nodes_per_cell * (self.nodes_per_cell - 1)
                                    + local_i * (self.nodes_per_cell - 1)
                                    + local_j
                                    - ti.cast(local_j > local_i, ti.i32)
                                )
                                target.atomic_add_raw_block_slot(block_offset + pair, block)
                        else:
                            base = scalar_offset + 9 * (
                                element_id * self.nodes_per_cell * self.nodes_per_cell
                                + local_i * self.nodes_per_cell
                                + local_j
                            )
                            for row, column in ti.static(ti.ndrange(3, 3)):
                                target.data[base + 3 * row + column] += block[row, column]

    @ti.kernel
    def _compute_minimum_jacobian(self, positions: ti.template()):
        self.minimum_jacobian[None] = 1.0e30
        for element_id in range(self.cell_count):
            cell = self.connectivity[element_id]
            for quadrature_id in ti.static(range(self.quadrature_count)):
                if ti.static(self.is_axisymmetric):
                    coordinates = ti.Matrix.zero(self.real_type, self.nodes_per_cell, 3)
                    for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                        coordinates[local_id, component] = positions[cell[local_id]][component]
                    value = self._axisymmetric_deformation_gradient(element_id, quadrature_id, coordinates)
                    ti.atomic_min(self.minimum_jacobian[None], value.determinant())
                elif ti.static(self.is_membrane):
                    value = ti.Matrix.zero(self.real_type, 3, 2)
                    for i, j in ti.static(ti.ndrange(3, 2)):
                        for local_id in ti.static(range(self.nodes_per_cell)):
                            value[i, j] += (
                                positions[cell[local_id]][i]
                                * self.shape_gradients[
                                    element_id,
                                    quadrature_id,
                                    local_id,
                                    j,
                                ]
                            )
                    jacobian = ti.sqrt(ti.max((value.transpose() @ value).determinant(), 0.0))
                    ti.atomic_min(self.minimum_jacobian[None], jacobian)
                else:
                    coordinates = ti.Matrix.zero(self.real_type, self.nodes_per_cell, 3)
                    for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                        coordinates[local_id, component] = positions[cell[local_id]][component]
                    value = self._volume_deformation_gradient(element_id, quadrature_id, coordinates)
                    ti.atomic_min(self.minimum_jacobian[None], value.determinant())

    def minimum_jacobian_ratio_device(self, positions=None):
        self._compute_minimum_jacobian(self.positions if positions is None else positions)
        return float(self.minimum_jacobian[None])

    @ti.kernel
    def _compute_maximum_material_step(
        self,
        positions: ti.template(),
        direction: ti.template(),
        safety: ti.f64,
    ):
        self.maximum_material_step[None] = 1.0
        for element_id in range(self.cell_count):
            cell = self.connectivity[element_id]
            current_coordinates = ti.Matrix.zero(self.real_type, self.nodes_per_cell, 3)
            direction_coordinates = ti.Matrix.zero(self.real_type, self.nodes_per_cell, 3)
            for local_id, component in ti.static(ti.ndrange(self.nodes_per_cell, 3)):
                current_coordinates[local_id, component] = positions[cell[local_id]][component]
                direction_coordinates[local_id, component] = direction[cell[local_id]][component]
            for quadrature_id in ti.static(range(self.quadrature_count)):
                toc = 1.0
                if ti.static(self.is_axisymmetric):
                    current = self._axisymmetric_deformation_gradient(element_id, quadrature_id, current_coordinates)
                    increment = ti.Matrix.zero(self.real_type, 3, 3)
                    for local_id in ti.static(range(self.nodes_per_cell)):
                        for spatial, material_component in ti.static(ti.ndrange(2, 2)):
                            increment[spatial, material_component] += (
                                direction_coordinates[local_id, spatial]
                                * self.shape_gradients[
                                    element_id,
                                    quadrature_id,
                                    local_id,
                                    material_component,
                                ]
                            )
                        increment[2, 2] += (
                            self.shape_values[quadrature_id, local_id]
                            * direction_coordinates[local_id, 0]
                            / self.reference_radii[element_id, quadrature_id]
                        )
                    toc = deformation_gradient_ccd(current, increment, safety)
                elif ti.static(self.is_planar):
                    current = ti.Matrix.zero(self.real_type, 2, 2)
                    increment = ti.Matrix.zero(self.real_type, 2, 2)
                    for spatial, material_component in ti.static(ti.ndrange(2, 2)):
                        for local_id in ti.static(range(self.nodes_per_cell)):
                            gradient = self.shape_gradients[
                                element_id,
                                quadrature_id,
                                local_id,
                                material_component,
                            ]
                            current[spatial, material_component] += current_coordinates[local_id, spatial] * gradient
                            increment[spatial, material_component] += (
                                direction_coordinates[local_id, spatial] * gradient
                            )
                    toc = deformation_gradient_ccd(current, increment, safety)
                elif ti.static(not self.is_membrane):
                    current = self._volume_deformation_gradient(element_id, quadrature_id, current_coordinates)
                    increment = self._volume_deformation_gradient(element_id, quadrature_id, direction_coordinates)
                    toc = deformation_gradient_ccd(current, increment, safety)
                ti.atomic_min(self.maximum_material_step[None], toc)

    def maximum_material_step_device(self, positions, direction, safety=0.9):
        self._compute_maximum_material_step(positions, direction, float(safety))
        return max(0.0, min(1.0, float(self.maximum_material_step[None])))

    @property
    def stiffness_entry_count(self):
        self._require_hessian_storage()
        return self.cell_count * self.local_dofs * self.local_dofs

    @property
    def stiffness_block_pair_count(self):
        self._require_hessian_storage()
        return self.cell_count * self.nodes_per_cell * (self.nodes_per_cell - 1)

    @property
    def stiffness_unique_block_pair_count(self):
        """Exact directed off-diagonal block topology of the FEM mesh.

        Raw assembly needs one slot per element/local ordered pair.  Reduced
        HashTriplet storage only needs one slot per unique mesh edge and
        orientation.  Connectivity is immutable, so compute this once at
        construction-time topology resolution; it never depends on a Newton
        iterate or contact search.
        """
        self._require_hessian_storage()
        if self._stiffness_unique_block_pair_count is None:
            cells = np.asarray(self.mesh.cells, dtype=np.int64)
            pair_columns = self.nodes_per_cell * (self.nodes_per_cell - 1) // 2
            keys = np.empty((self.cell_count, pair_columns), dtype=np.int64)
            column = 0
            for first in range(self.nodes_per_cell):
                for second in range(first + 1, self.nodes_per_cell):
                    node_i = np.minimum(cells[:, first], cells[:, second])
                    node_j = np.maximum(cells[:, first], cells[:, second])
                    keys[:, column] = node_i * self.node_count + node_j
                    column += 1
            undirected = int(np.unique(keys.reshape(-1)).size)
            self._stiffness_unique_block_pair_count = 2 * undirected
        return self._stiffness_unique_block_pair_count

    def scatter_stiffness_to_coo(self, matrix, offset=0):
        self._require_hessian_storage()
        self._assemble_element_stiffness_direct(self.stiffness_positions, matrix, int(offset), 0, False)

    def scatter_stiffness_to_hash(self, matrix):
        self._require_hessian_storage()
        block_offset = matrix.reserve_raw_block_slots(self.stiffness_block_pair_count)
        self._assemble_element_stiffness_direct(self.stiffness_positions, matrix, 0, block_offset, True)

    def _sparse_stiffness(self, positions):
        self._require_hessian_storage()
        self._copy_positions_from_field(positions, self.stiffness_positions)
        if self._stiffness_matrix is None:
            self._stiffness_matrix = FEMSparseMatrix(
                3 * self.node_count,
                assemble_type=self.assemble_type,
                linear_solver=self.linear_solver,
                linear_solver_tolerance=self.linear_solver_tolerance,
                linear_solver_relative_tolerance=self.linear_solver_relative_tolerance,
                linear_solver_max_iters=self.linear_solver_max_iters,
                base_assembler=self,
            )
        else:
            self._stiffness_matrix.reset_device_assembly()
        return self._stiffness_matrix

    def _cell_stress(self, positions):
        deformation_gradients = self.deformation_gradient.to_numpy()
        stress = np.zeros((self.cell_count, 3, 3), dtype=np.float64)
        weights = np.asarray(self.element.reference_weights)
        for element_id in range(self.cell_count):
            total_weight = float(np.sum(weights[element_id]))
            if self.is_membrane:
                frame = self.element._triangle_frame(positions[self.element.connectivity[element_id]])
            for quadrature_id in range(self.quadrature_count):
                value = self.material.cauchy_stress(
                    np.asarray(deformation_gradients[element_id, quadrature_id], dtype=np.float64)
                )
                if self.is_membrane:
                    value = frame[:2].T @ value @ frame[:2]
                stress[element_id] += weights[element_id, quadrature_id] * value
            stress[element_id] /= total_weight
        return stress

    def assemble_device(self, positions=None, need_stiffness=False, stiffness=None):
        if need_stiffness:
            self._require_hessian_storage()
        source = self.bound_positions if positions is None else positions
        self.internal_force.fill(0.0)
        self.total_energy[None] = 0.0
        self._assemble_energy_force(source)
        if self.is_membrane:
            # Membrane constitutive work uses a 3x2 surface map, whereas the
            # stored output is the co-rotated 2x2 map.
            self._store_deformation_gradients(source)
        if need_stiffness:
            if stiffness is None:
                stiffness = self._sparse_stiffness(source)
            else:
                if stiffness.base_assembler is not self:
                    raise ValueError("reused FEM stiffness belongs to another assembler")
                stiffness.reset_device_assembly()
                self._copy_positions_from_field(source, self.stiffness_positions)
        else:
            stiffness = None
        return self.internal_force, stiffness

    def assemble_force_device(self, positions=None):
        """Explicit hot path; diagnostics are refreshed only when sampled."""
        source = self.bound_positions if positions is None else positions
        self.internal_force.fill(0.0)
        self._assemble_force(source)
        return self.internal_force

    def _require_hessian_storage(self):
        if not self.allocate_hessian:
            raise RuntimeError("element Hessian storage is disabled for explicit FEM")

    def assemble(self, positions, need_stiffness=False, need_stress=False):
        """Explicit host snapshot adapter for tests and output only."""
        if hasattr(positions, "to_numpy"):
            source = positions
        else:
            values = np.ascontiguousarray(positions, dtype=self.numpy_type)
            if values.shape != (self.node_count, 3):
                raise ValueError("FEM positions must have shape (number_of_nodes, 3)")
            self.positions.from_numpy(values)
            source = self.positions
        force, stiffness = self.assemble_device(source, need_stiffness=need_stiffness)
        stress = self._cell_stress(source.to_numpy()) if need_stress else None
        result = (
            float(self.total_energy[None]),
            force.to_numpy(),
            stiffness,
            stress,
        )
        return result

    def deformation_gradients(self, positions):
        self.assemble(positions)
        return self.deformation_gradient.to_numpy()


__all__ = ["ClassicalAssembler"]
