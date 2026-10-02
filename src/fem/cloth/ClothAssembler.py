"""Taichi kernels for triangular cloth finite-element assembly."""

import numpy as np
import taichi as ti

from src.fem.cloth.ClothEnergy import (
    normalize_bending_model,
    prepare_cloth_energies,
)
from src.fem.cloth.ClothElement import create_cloth_element
from src.fem.engines.SparseMatrix import FEMSparseMatrix
from src.physics_model.contact_model.ipc.ContactAssembly import (
    psd_project_cloth_tangent_6x6,
)


def _require_initialized_taichi():
    if ti.lang.impl.get_runtime().prog is None:
        raise RuntimeError(
            "Taichi cloth FEM requires an initialized runtime; call geotaichi.init(...) or taichi.init(...) first"
        )


@ti.func
def _hinge_projection_direction(point, edge0, edge1):
    edge = edge1 - edge0
    denominator = ti.max(edge.dot(edge), 1.0e-30)
    direction = edge0 + (point - edge0).dot(edge) / denominator * edge - point
    return direction / ti.max(direction.norm(), 1.0e-15)


@ti.func
def signed_dihedral_angle(x0, x1, x2, x3):
    normal0 = (x1 - x0).cross(x2 - x0)
    normal1 = (x2 - x3).cross(x1 - x3)
    denominator = ti.max(normal0.norm() * normal1.norm(), 1.0e-30)
    cosine = ti.max(-1.0, ti.min(1.0, normal0.dot(normal1) / denominator))
    angle = ti.acos(cosine)
    if normal1.cross(normal0).dot(x1 - x2) < 0.0:
        angle = -angle
    return angle


@ti.func
def signed_dihedral_gradient(x0, x1, x2, x3):
    edge0 = x2 - x1
    edge1 = x0 - x1
    edge2 = x3 - x1
    edge3 = x0 - x2
    edge4 = x3 - x2
    normal0 = edge0.cross(edge1)
    normal1 = edge2.cross(edge0)
    normal0_squared = ti.max(normal0.dot(normal0), 1.0e-30)
    normal1_squared = ti.max(normal1.dot(normal1), 1.0e-30)
    hinge_length = ti.max(edge0.norm(), 1.0e-15)
    gradient = ti.Vector.zero(float, 12)
    value0 = -hinge_length / normal0_squared * normal0
    value1 = (
        -edge0.dot(edge3) / (hinge_length * normal0_squared) * normal0
        - edge0.dot(edge4) / (hinge_length * normal1_squared) * normal1
    )
    value2 = (
        edge0.dot(edge1) / (hinge_length * normal0_squared) * normal0
        + edge0.dot(edge2) / (hinge_length * normal1_squared) * normal1
    )
    value3 = -hinge_length / normal1_squared * normal1
    for component in ti.static(range(3)):
        gradient[component] = value0[component]
        gradient[3 + component] = value1[component]
        gradient[6 + component] = value2[component]
        gradient[9 + component] = value3[component]
    return gradient


@ti.func
def _store_direct_block(
    target: ti.template(),
    assemble_hash: ti.template(),
    scalar_entry: ti.i32,
    raw_entry: ti.i32,
    node_i: ti.i32,
    node_j: ti.i32,
    block,
):
    if ti.static(assemble_hash):
        if raw_entry < 0:
            target.add_block_entry(node_i, node_j, block)
        else:
            target.initialize_raw_block_slot(raw_entry, node_i, node_j)
            target.atomic_add_raw_block_slot(raw_entry, block)
    else:
        for row, column in ti.static(ti.ndrange(3, 3)):
            entry = scalar_entry + 3 * row + column
            target.rows[entry] = 3 * node_i + row
            target.cols[entry] = 3 * node_j + column
            target.data[entry] = block[row, column]


@ti.func
def assemble_signed_dihedral_hessian_blocks(
    target: ti.template(),
    assemble_hash: ti.template(),
    scalar_base: ti.i32,
    raw_base: ti.i32,
    nodes,
    x0,
    x1,
    x2,
    x3,
    first,
    second,
    angle_gradient,
    project_pd: ti.template(),
    store_only: ti.template(),
    tangent_target: ti.template(),
    tangent_id: ti.i32,
):
    edge0 = x2 - x1
    edge1 = x0 - x1
    edge2 = x3 - x1
    edge3 = x0 - x2
    edge4 = x3 - x2
    length0 = ti.max(edge0.norm(), 1.0e-15)
    length1 = ti.max(edge1.norm(), 1.0e-15)
    length2 = ti.max(edge2.norm(), 1.0e-15)
    length3 = ti.max(edge3.norm(), 1.0e-15)
    length4 = ti.max(edge4.norm(), 1.0e-15)
    normal0 = edge0.cross(edge1)
    normal1 = edge2.cross(edge0)
    normal0_length = ti.max(normal0.norm(), 1.0e-15)
    normal1_length = ti.max(normal1.norm(), 1.0e-15)

    m1 = _hinge_projection_direction(x2, x1, x0)
    m2 = _hinge_projection_direction(x2, x1, x3)
    m3 = _hinge_projection_direction(x1, x2, x0)
    m4 = _hinge_projection_direction(x1, x2, x3)
    m01 = _hinge_projection_direction(x0, x1, x2)
    m02 = _hinge_projection_direction(x3, x1, x2)
    cosine1 = edge0.dot(edge1) / (length0 * length1)
    cosine2 = edge0.dot(edge2) / (length0 * length2)
    cosine3 = -edge0.dot(edge3) / (length0 * length3)
    cosine4 = -edge0.dot(edge4) / (length0 * length4)
    height1 = normal0_length / length1
    height2 = normal1_length / length2
    height3 = normal0_length / length3
    height4 = normal1_length / length4
    height01 = normal0_length / length0
    height02 = normal1_length / length0

    n1_01 = normal0.outer_product(m01) / (height01 * height01 * normal0_length)
    n2_02 = normal1.outer_product(m02) / (height02 * height02 * normal1_length)
    n1_3 = normal0.outer_product(m3) / (height01 * height3 * normal0_length)
    n1_1 = normal0.outer_product(m1) / (height01 * height1 * normal0_length)
    n2_4 = normal1.outer_product(m4) / (height02 * height4 * normal1_length)
    n2_2 = normal1.outer_product(m2) / (height02 * height2 * normal1_length)
    m3_01_1 = cosine3 / (height3 * height01 * normal0_length) * m01.outer_product(normal0)
    m1_01_1 = cosine1 / (height1 * height01 * normal0_length) * m01.outer_product(normal0)
    m1_1_1 = cosine1 / (height1 * height1 * normal0_length) * m1.outer_product(normal0)
    m3_3_1 = cosine3 / (height3 * height3 * normal0_length) * m3.outer_product(normal0)
    m3_1_1 = cosine3 / (height3 * height1 * normal0_length) * m1.outer_product(normal0)
    m1_3_1 = cosine1 / (height1 * height3 * normal0_length) * m3.outer_product(normal0)
    m4_02_2 = cosine4 / (height4 * height02 * normal1_length) * m02.outer_product(normal1)
    m2_02_2 = cosine2 / (height2 * height02 * normal1_length) * m02.outer_product(normal1)
    m4_4_2 = cosine4 / (height4 * height4 * normal1_length) * m4.outer_product(normal1)
    m2_4_2 = cosine2 / (height2 * height4 * normal1_length) * m4.outer_product(normal1)
    m4_2_2 = cosine4 / (height4 * height2 * normal1_length) * m2.outer_product(normal1)
    m2_2_2 = cosine2 / (height2 * height2 * normal1_length) * m2.outer_product(normal1)
    b1 = normal0.outer_product(m01) / (length0 * length0 * normal0_length)
    b2 = normal1.outer_product(m02) / (length0 * length0 * normal1_length)

    block00 = -(n1_01 + n1_01.transpose())
    block10 = m3_01_1 - n1_3
    block20 = m1_01_1 - n1_1
    block11 = m3_3_1 + m3_3_1.transpose() - b1 + m4_4_2 + m4_4_2.transpose() - b2
    block12 = m3_1_1 + m1_3_1.transpose() + b1 + m4_2_2 + m2_4_2.transpose() + b2
    block13 = m4_02_2 - n2_4
    block22 = m1_1_1 + m1_1_1.transpose() - b1 + m2_2_2 + m2_2_2.transpose() - b2
    block23 = m2_02_2 - n2_2
    block33 = -(n2_02 + n2_02.transpose())
    angle_hessian = ti.Matrix.zero(float, 12, 12)
    for row, column in ti.static(ti.ndrange(3, 3)):
        angle_hessian[row, column] = block00[row, column]
        angle_hessian[3 + row, column] = block10[row, column]
        angle_hessian[row, 3 + column] = block10[column, row]
        angle_hessian[6 + row, column] = block20[row, column]
        angle_hessian[row, 6 + column] = block20[column, row]
        angle_hessian[3 + row, 3 + column] = block11[row, column]
        angle_hessian[3 + row, 6 + column] = block12[row, column]
        angle_hessian[6 + row, 3 + column] = block12[column, row]
        angle_hessian[3 + row, 9 + column] = block13[row, column]
        angle_hessian[9 + row, 3 + column] = block13[column, row]
        angle_hessian[6 + row, 6 + column] = block22[row, column]
        angle_hessian[6 + row, 9 + column] = block23[row, column]
        angle_hessian[9 + row, 6 + column] = block23[column, row]
        angle_hessian[9 + row, 9 + column] = block33[row, column]
    angle_hessian = 0.5 * (angle_hessian + angle_hessian.transpose())
    local_hessian = first * angle_hessian + second * angle_gradient.outer_product(angle_gradient)
    local_hessian = 0.5 * (local_hessian + local_hessian.transpose())
    if ti.static(project_pd):
        # Gauss--Newton bending block: ``second * g g.T`` is PSD by
        # construction and avoids a 12x12 eigensolve/majorizer in the sparse
        # scatter kernel. The exact angle-Hessian term remains available when
        # project_pd=False (BiCGSTAB/debug path).
        local_hessian = second * angle_gradient.outer_product(angle_gradient)
    if ti.static(store_only):
        tangent_target[tangent_id] = local_hessian
    else:
        for local_i, local_j in ti.static(ti.ndrange(4, 4)):
            block = ti.Matrix.zero(float, 3, 3)
            for row, column in ti.static(ti.ndrange(3, 3)):
                block[row, column] = local_hessian[3 * local_i + row, 3 * local_j + column]
            raw_entry = -1
            if local_i != local_j:
                raw_entry = raw_base + local_i * 3 + local_j - ti.cast(local_j > local_i, ti.i32)
            _store_direct_block(
                target,
                assemble_hash,
                scalar_base + 9 * (4 * local_i + local_j),
                raw_entry,
                nodes[local_i],
                nodes[local_j],
                block,
            )


@ti.data_oriented
class ClothAssembler:
    """Assemble cloth energy, internal force and element tangent on the device."""

    def __init__(
        self,
        mesh,
        material,
        bending_model=None,
        cloth_energies=None,
        project_pd=True,
        assemble_type="Hash",
        linear_solver="PCG",
        linear_solver_tolerance=1.0e-10,
        linear_solver_max_iters=500,
        project_bending_pd=None,
        linear_solver_relative_tolerance=0.0,
    ):
        _require_initialized_taichi()
        self.mesh = mesh
        self.material = material
        self.element = create_cloth_element(mesh, material)
        self.bending_model = normalize_bending_model(material.bending_model if bending_model is None else bending_model)
        self.bending_model_id = {
            "None": 0,
            "Quadratic": 1,
            "Dihedral": 2,
        }[self.bending_model]
        self.bending_modulus = float(material.quadratic_bending_modulus)
        self.energy_data = prepare_cloth_energies(mesh, cloth_energies)
        self.node_count = mesh.number_of_nodes
        self.element_count = mesh.number_of_cells
        self.bending_element_count = (
            0 if self.bending_model_id == 0 else int(self.element.bending_connectivity.shape[0])
        )
        self.stitch_count = int(self.energy_data.stitch_nodes.shape[0])
        self.spring_count = int(self.energy_data.spring_nodes.size)
        self.sdf_count = int(self.energy_data.sdf_nodes.size)
        self.project_pd = bool(project_pd)
        self.project_bending_pd = self.project_pd if project_bending_pd is None else bool(project_bending_pd)
        self.assemble_type = assemble_type
        self.linear_solver = linear_solver
        self.linear_solver_tolerance = float(linear_solver_tolerance)
        self.linear_solver_relative_tolerance = float(linear_solver_relative_tolerance)
        self.linear_solver_max_iters = int(linear_solver_max_iters)
        self.real_type = ti.lang.impl.current_cfg().default_fp
        self.numpy_type = np.float64 if self.real_type == ti.f64 else np.float32

        self.positions = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.stiffness_positions = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.internal_force = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.connectivity = ti.Vector.field(3, dtype=ti.i32, shape=self.element_count)
        self.inverse_reference_jacobian = ti.Matrix.field(2, 2, dtype=self.real_type, shape=self.element_count)
        self.integration_weight = ti.field(dtype=self.real_type, shape=self.element_count)
        self.deformation_gradient = ti.Matrix.field(3, 2, dtype=self.real_type, shape=self.element_count)
        # Keep constitutive projection in its own device pass.  Inlining a
        # 6x6 PSD reduction into the sparse scatter kernel makes CUDA compile
        # the material tangent once per static block stencil.
        self.membrane_tangent = ti.Matrix.field(6, 6, dtype=self.real_type, shape=self.element_count)
        self.element_energy = ti.field(dtype=self.real_type, shape=self.element_count)
        allocated_bending_elements = max(self.bending_element_count, 1)
        self.bending_connectivity = ti.Vector.field(4, dtype=ti.i32, shape=allocated_bending_elements)
        self.bending_tangent = ti.Matrix.field(12, 12, dtype=self.real_type, shape=allocated_bending_elements)
        self.bending_stiffness = ti.Matrix.field(4, 4, dtype=self.real_type, shape=allocated_bending_elements)
        self.bending_edge_length = ti.field(dtype=self.real_type, shape=allocated_bending_elements)
        self.bending_height = ti.field(dtype=self.real_type, shape=allocated_bending_elements)
        self.bending_rest_angle = ti.field(dtype=self.real_type, shape=allocated_bending_elements)

        allocated_stitches = max(self.stitch_count, 1)
        self.stitch_nodes = ti.Vector.field(3, dtype=ti.i32, shape=allocated_stitches)
        self.stitch_ratio = ti.field(dtype=self.real_type, shape=allocated_stitches)
        self.stitch_stiffness = ti.field(dtype=self.real_type, shape=allocated_stitches)
        allocated_springs = max(self.spring_count, 1)
        self.spring_nodes = ti.field(dtype=ti.i32, shape=allocated_springs)
        self.spring_target = ti.Vector.field(3, dtype=self.real_type, shape=allocated_springs)
        self.spring_stiffness = ti.field(dtype=self.real_type, shape=allocated_springs)
        allocated_sdf = max(self.sdf_count, 1)
        self.sdf_nodes = ti.field(dtype=ti.i32, shape=allocated_sdf)
        self.sdf_target = ti.Vector.field(3, dtype=self.real_type, shape=allocated_sdf)
        self.sdf_normal = ti.Vector.field(3, dtype=self.real_type, shape=allocated_sdf)
        self.sdf_stiffness = ti.field(dtype=self.real_type, shape=allocated_sdf)
        self.sdf_dhat = ti.field(dtype=self.real_type, shape=allocated_sdf)
        self.node_area = ti.field(dtype=self.real_type, shape=self.node_count)
        self.total_energy = ti.field(dtype=self.real_type, shape=())
        self.minimum_jacobian_value = ti.field(dtype=self.real_type, shape=())
        self._stiffness_matrix = None

        self.connectivity.from_numpy(self.element.connectivity)
        self.inverse_reference_jacobian.from_numpy(self.element.inverse_reference_jacobian.astype(self.numpy_type))
        self.integration_weight.from_numpy(self.element.integration_weight.astype(self.numpy_type))
        if self.bending_element_count:
            connectivity_buffer = np.zeros((allocated_bending_elements, 4), dtype=np.int32)
            stiffness_buffer = np.zeros((allocated_bending_elements, 4, 4), dtype=self.numpy_type)
            connectivity_buffer[: self.bending_element_count] = self.element.bending_connectivity
            stiffness_buffer[: self.bending_element_count] = self.element.bending_stiffness
            self.bending_connectivity.from_numpy(connectivity_buffer)
            self.bending_stiffness.from_numpy(stiffness_buffer)
            self.bending_edge_length.from_numpy(
                np.pad(
                    self.element.bending_edge_length.astype(self.numpy_type),
                    (0, allocated_bending_elements - self.bending_element_count),
                )
            )
            self.bending_height.from_numpy(
                np.pad(
                    self.element.bending_height.astype(self.numpy_type),
                    (0, allocated_bending_elements - self.bending_element_count),
                )
            )
            self.bending_rest_angle.from_numpy(
                np.pad(
                    self.element.bending_rest_angle.astype(self.numpy_type),
                    (0, allocated_bending_elements - self.bending_element_count),
                )
            )
        self.node_area.from_numpy(self.energy_data.node_area.astype(self.numpy_type))
        self._initialize_optional_energies(allocated_stitches, allocated_springs, allocated_sdf)

    def _initialize_optional_energies(self, allocated_stitches, allocated_springs, allocated_sdf):
        if self.stitch_count:
            nodes = np.zeros((allocated_stitches, 3), dtype=np.int32)
            ratio = np.zeros(allocated_stitches, dtype=self.numpy_type)
            stiffness = np.zeros(allocated_stitches, dtype=self.numpy_type)
            nodes[: self.stitch_count] = self.energy_data.stitch_nodes
            ratio[: self.stitch_count] = self.energy_data.stitch_ratio
            stiffness[: self.stitch_count] = self.energy_data.stitch_stiffness
            self.stitch_nodes.from_numpy(nodes)
            self.stitch_ratio.from_numpy(ratio)
            self.stitch_stiffness.from_numpy(stiffness)
        if self.spring_count:
            nodes = np.zeros(allocated_springs, dtype=np.int32)
            target = np.zeros((allocated_springs, 3), dtype=self.numpy_type)
            stiffness = np.zeros(allocated_springs, dtype=self.numpy_type)
            nodes[: self.spring_count] = self.energy_data.spring_nodes
            target[: self.spring_count] = self.energy_data.spring_target
            stiffness[: self.spring_count] = self.energy_data.spring_stiffness
            self.spring_nodes.from_numpy(nodes)
            self.spring_target.from_numpy(target)
            self.spring_stiffness.from_numpy(stiffness)
        if self.sdf_count:
            nodes = np.zeros(allocated_sdf, dtype=np.int32)
            target = np.zeros((allocated_sdf, 3), dtype=self.numpy_type)
            normal = np.zeros((allocated_sdf, 3), dtype=self.numpy_type)
            stiffness = np.zeros(allocated_sdf, dtype=self.numpy_type)
            dhat = np.ones(allocated_sdf, dtype=self.numpy_type)
            nodes[: self.sdf_count] = self.energy_data.sdf_nodes
            target[: self.sdf_count] = self.energy_data.sdf_target
            normal[: self.sdf_count] = self.energy_data.sdf_normal
            stiffness[: self.sdf_count] = self.energy_data.sdf_stiffness
            dhat[: self.sdf_count] = self.energy_data.sdf_dhat
            self.sdf_nodes.from_numpy(nodes)
            self.sdf_target.from_numpy(target)
            self.sdf_normal.from_numpy(normal)
            self.sdf_stiffness.from_numpy(stiffness)
            self.sdf_dhat.from_numpy(dhat)

    def bind_positions(self, positions):
        self.bound_positions = positions

    @ti.kernel
    def _copy_positions_from_field(self, source: ti.template(), target: ti.template()):
        for node in range(self.node_count):
            target[node] = source[node]

    @ti.func
    def _calculate_current_jacobian(self, x0, x1, x2):
        first = x1 - x0
        second = x2 - x0
        return ti.Matrix(
            [
                [first[0], second[0]],
                [first[1], second[1]],
                [first[2], second[2]],
            ]
        )

    @ti.func
    def _compute_membrane_response(self, element_id, x0, x1, x2):
        current_jacobian = self._calculate_current_jacobian(x0, x1, x2)
        inverse_reference_jacobian = self.inverse_reference_jacobian[element_id]
        deformation_gradient = current_jacobian @ inverse_reference_jacobian
        energy_density = self.material.strain_energy_density(deformation_gradient)
        first_piola = self.material.first_piola_stress(deformation_gradient)
        jacobian_force = first_piola @ inverse_reference_jacobian.transpose()
        weight = self.integration_weight[element_id]
        local_force = ti.Vector.zero(self.real_type, 9)
        for component in ti.static(range(3)):
            local_force[component] = -weight * (jacobian_force[component, 0] + jacobian_force[component, 1])
            local_force[3 + component] = weight * jacobian_force[component, 0]
            local_force[6 + component] = weight * jacobian_force[component, 1]
        return weight * energy_density, local_force, deformation_gradient

    @ti.kernel
    def _assemble_membrane_force(self):
        for element_id in range(self.element_count):
            nodes = self.connectivity[element_id]
            value, local_force, deformation_gradient = self._compute_membrane_response(
                element_id,
                self.positions[nodes[0]],
                self.positions[nodes[1]],
                self.positions[nodes[2]],
            )
            self.element_energy[element_id] = value
            self.deformation_gradient[element_id] = deformation_gradient
            ti.atomic_add(self.total_energy[None], value)
            for local_id, component in ti.static(ti.ndrange(3, 3)):
                ti.atomic_add(
                    self.internal_force[nodes[local_id]][component],
                    local_force[3 * local_id + component],
                )

    @ti.kernel
    def _assemble_bending_force(self):
        for bending_id in range(self.bending_element_count):
            nodes = self.bending_connectivity[bending_id]
            value = 0.0
            local_gradient = ti.Vector.zero(self.real_type, 12)
            if ti.static(self.bending_model_id == 1):
                local_stiffness = self.bending_stiffness[bending_id]
                for first, second in ti.static(ti.ndrange(4, 4)):
                    value += (
                        0.5
                        * local_stiffness[first, second]
                        * self.positions[nodes[first]].dot(self.positions[nodes[second]])
                    )
                    for component in ti.static(range(3)):
                        local_gradient[3 * first + component] += (
                            local_stiffness[first, second] * self.positions[nodes[second]][component]
                        )
            elif ti.static(self.bending_model_id == 2):
                x0 = self.positions[nodes[0]]
                x1 = self.positions[nodes[1]]
                x2 = self.positions[nodes[2]]
                x3 = self.positions[nodes[3]]
                angle = signed_dihedral_angle(x0, x1, x2, x3)
                angle_gradient = signed_dihedral_gradient(x0, x1, x2, x3)
                difference = angle - self.bending_rest_angle[bending_id]
                ratio = self.bending_edge_length[bending_id] / self.bending_height[bending_id]
                value = self.bending_modulus * ratio * difference * difference
                local_gradient = 2.0 * self.bending_modulus * ratio * difference * angle_gradient
            ti.atomic_add(self.total_energy[None], value)
            for local, component in ti.static(ti.ndrange(4, 3)):
                ti.atomic_add(
                    self.internal_force[nodes[local]][component],
                    local_gradient[3 * local + component],
                )

    @ti.kernel
    def _assemble_optional_energy_force(self):
        for stitch_id in range(self.stitch_count):
            nodes = self.stitch_nodes[stitch_id]
            ratio = self.stitch_ratio[stitch_id]
            weights = ti.Vector([1.0, ratio - 1.0, -ratio])
            difference = ti.Vector.zero(self.real_type, 3)
            for local in ti.static(range(3)):
                difference += weights[local] * self.positions[nodes[local]]
            coefficient = self.stitch_stiffness[stitch_id] * self.node_area[nodes[0]]
            ti.atomic_add(
                self.total_energy[None],
                0.5 * coefficient * difference.dot(difference),
            )
            for local, component in ti.static(ti.ndrange(3, 3)):
                ti.atomic_add(
                    self.internal_force[nodes[local]][component],
                    coefficient * weights[local] * difference[component],
                )

        for spring_id in range(self.spring_count):
            node = self.spring_nodes[spring_id]
            difference = self.positions[node] - self.spring_target[spring_id]
            stiffness = self.spring_stiffness[spring_id]
            ti.atomic_add(
                self.total_energy[None],
                0.5 * stiffness * difference.dot(difference),
            )
            for component in ti.static(range(3)):
                ti.atomic_add(
                    self.internal_force[node][component],
                    stiffness * difference[component],
                )

        for sdf_id in range(self.sdf_count):
            node = self.sdf_nodes[sdf_id]
            normal = self.sdf_normal[sdf_id]
            distance = (self.positions[node] - self.sdf_target[sdf_id]).dot(normal)
            dhat = self.sdf_dhat[sdf_id]
            if distance <= dhat:
                normalized_gap = distance / dhat - 1.0
                coefficient = self.sdf_stiffness[sdf_id] * self.node_area[node]
                ti.atomic_add(
                    self.total_energy[None],
                    -coefficient * dhat / 6.0 * normalized_gap**3,
                )
                gradient_scale = -0.5 * coefficient * normalized_gap**2
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        self.internal_force[node][component],
                        gradient_scale * normal[component],
                    )

    @ti.kernel
    def _assemble_membrane_tangent(self, positions: ti.template()):
        for element_id in range(self.element_count):
            nodes = self.connectivity[element_id]
            deformation_gradient = (
                self._calculate_current_jacobian(positions[nodes[0]], positions[nodes[1]], positions[nodes[2]])
                @ self.inverse_reference_jacobian[element_id]
            )
            tangent = self.material.first_piola_tangent(deformation_gradient)
            tangent = 0.5 * (tangent + tangent.transpose())
            self.membrane_tangent[element_id] = tangent

    @ti.kernel
    def _project_membrane_tangents(self):
        for element_id in range(self.element_count):
            self.membrane_tangent[element_id] = psd_project_cloth_tangent_6x6(self.membrane_tangent[element_id])

    @ti.kernel
    def _assemble_membrane_stiffness_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        scalar_offset: ti.i32,
        raw_offset: ti.i32,
        assemble_hash: ti.template(),
    ):
        for element_id in range(self.element_count):
            nodes = self.connectivity[element_id]
            x0 = positions[nodes[0]]
            x1 = positions[nodes[1]]
            x2 = positions[nodes[2]]
            material_tangent = self.membrane_tangent[element_id]
            inverse_reference = self.inverse_reference_jacobian[element_id]
            gradients = ti.Matrix.zero(self.real_type, 3, 2)
            for material_component in ti.static(range(2)):
                gradients[0, material_component] = -(
                    inverse_reference[0, material_component] + inverse_reference[1, material_component]
                )
                gradients[1, material_component] = inverse_reference[0, material_component]
                gradients[2, material_component] = inverse_reference[1, material_component]
            weight = self.integration_weight[element_id]
            for local_i, local_j in ti.static(ti.ndrange(3, 3)):
                block = ti.Matrix.zero(self.real_type, 3, 3)
                for spatial_i, spatial_j in ti.static(ti.ndrange(3, 3)):
                    for material_i, material_j in ti.static(ti.ndrange(2, 2)):
                        block[spatial_i, spatial_j] += weight * (
                            gradients[local_i, material_i]
                            * material_tangent[
                                3 * material_i + spatial_i,
                                3 * material_j + spatial_j,
                            ]
                            * gradients[local_j, material_j]
                        )
                raw_entry = -1
                if local_i != local_j:
                    raw_entry = raw_offset + element_id * 6 + local_i * 2 + local_j - ti.cast(local_j > local_i, ti.i32)
                _store_direct_block(
                    target,
                    assemble_hash,
                    scalar_offset + 9 * (9 * element_id + 3 * local_i + local_j),
                    raw_entry,
                    nodes[local_i],
                    nodes[local_j],
                    block,
                )

    @ti.kernel
    def _assemble_bending_tangent(self, positions: ti.template()):
        for bending_id in range(self.bending_element_count):
            nodes = self.bending_connectivity[bending_id]
            x0 = positions[nodes[0]]
            x1 = positions[nodes[1]]
            x2 = positions[nodes[2]]
            x3 = positions[nodes[3]]
            angle = signed_dihedral_angle(x0, x1, x2, x3)
            angle_gradient = signed_dihedral_gradient(x0, x1, x2, x3)
            difference = angle - self.bending_rest_angle[bending_id]
            ratio = self.bending_edge_length[bending_id] / self.bending_height[bending_id]
            second = 2.0 * self.bending_modulus * ratio
            if ti.static(self.project_bending_pd):
                self.bending_tangent[bending_id] = second * angle_gradient.outer_product(angle_gradient)
            else:
                assemble_signed_dihedral_hessian_blocks(
                    self.bending_tangent,
                    False,
                    0,
                    0,
                    nodes,
                    x0,
                    x1,
                    x2,
                    x3,
                    2.0 * self.bending_modulus * ratio * difference,
                    second,
                    angle_gradient,
                    False,
                    True,
                    self.bending_tangent,
                    bending_id,
                )

    @ti.kernel
    def _assemble_bending_stiffness_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        scalar_offset: ti.i32,
        raw_offset: ti.i32,
        assemble_hash: ti.template(),
    ):
        for bending_id in range(self.bending_element_count):
            nodes = self.bending_connectivity[bending_id]
            if ti.static(self.bending_model_id == 1):
                stiffness = self.bending_stiffness[bending_id]
                for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                    block = stiffness[local_i, local_j] * ti.Matrix.identity(self.real_type, 3)
                    raw_entry = -1
                    if local_i != local_j:
                        raw_entry = (
                            raw_offset + bending_id * 12 + local_i * 3 + local_j - ti.cast(local_j > local_i, ti.i32)
                        )
                    _store_direct_block(
                        target,
                        assemble_hash,
                        scalar_offset + 9 * (16 * bending_id + 4 * local_i + local_j),
                        raw_entry,
                        nodes[local_i],
                        nodes[local_j],
                        block,
                    )
            elif ti.static(self.bending_model_id == 2):
                local_hessian = self.bending_tangent[bending_id]
                for local_i, local_j in ti.static(ti.ndrange(4, 4)):
                    block = ti.Matrix.zero(float, 3, 3)
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        block[row, column] = local_hessian[3 * local_i + row, 3 * local_j + column]
                    raw_entry = -1
                    if local_i != local_j:
                        raw_entry = (
                            raw_offset + bending_id * 12 + local_i * 3 + local_j - ti.cast(local_j > local_i, ti.i32)
                        )
                    _store_direct_block(
                        target,
                        assemble_hash,
                        scalar_offset + bending_id * 144 + 9 * (4 * local_i + local_j),
                        raw_entry,
                        nodes[local_i],
                        nodes[local_j],
                        block,
                    )

    @ti.kernel
    def _compute_minimum_jacobian(self, positions: ti.template()):
        self.minimum_jacobian_value[None] = 1.0e30
        for element_id in range(self.element_count):
            nodes = self.connectivity[element_id]
            jacobian = self._calculate_current_jacobian(positions[nodes[0]], positions[nodes[1]], positions[nodes[2]])
            deformation_gradient = jacobian @ self.inverse_reference_jacobian[element_id]
            ratio = ti.sqrt(
                ti.max(
                    (deformation_gradient.transpose() @ deformation_gradient).determinant(),
                    0.0,
                )
            )
            ti.atomic_min(self.minimum_jacobian_value[None], ratio)

    def minimum_jacobian_ratio_device(self, positions=None):
        self._compute_minimum_jacobian(self.positions if positions is None else positions)
        return float(self.minimum_jacobian_value[None])

    def maximum_material_step_device(self, positions, direction, safety=0.9):
        """Compatibility limiter for monolithic FEM--MPM line searches.

        Embedded TRI3 cloth has no volumetric determinant barrier.  The
        coupled engine checks the trial metric through
        ``minimum_jacobian_ratio_device`` on every Armijo trial, so a unit
        material step is the correct conservative upper bound here.
        """
        del positions, direction, safety
        return 1.0

    @property
    def stiffness_entry_count(self):
        return (
            self.element_count * 81
            + self.bending_element_count * 144
            + self.stitch_count * 81
            + (self.spring_count + self.sdf_count) * 9
        )

    @property
    def stiffness_block_pair_count(self):
        return self.element_count * 6 + self.bending_element_count * 12 + self.stitch_count * 6

    @ti.kernel
    def _assemble_optional_stiffness_direct(
        self,
        positions: ti.template(),
        target: ti.template(),
        scalar_offset: ti.i32,
        raw_offset: ti.i32,
        assemble_hash: ti.template(),
    ):
        for stitch_id in range(self.stitch_count):
            nodes = self.stitch_nodes[stitch_id]
            ratio = self.stitch_ratio[stitch_id]
            weights = ti.Vector([1.0, ratio - 1.0, -ratio])
            for local_i, local_j in ti.static(ti.ndrange(3, 3)):
                coefficient = (
                    self.stitch_stiffness[stitch_id] * self.node_area[nodes[0]] * weights[local_i] * weights[local_j]
                )
                raw_entry = -1
                if local_i != local_j:
                    raw_entry = raw_offset + stitch_id * 6 + local_i * 2 + local_j - ti.cast(local_j > local_i, ti.i32)
                _store_direct_block(
                    target,
                    assemble_hash,
                    scalar_offset + 9 * (9 * stitch_id + 3 * local_i + local_j),
                    raw_entry,
                    nodes[local_i],
                    nodes[local_j],
                    coefficient * ti.Matrix.identity(self.real_type, 3),
                )

        spring_offset = scalar_offset + self.stitch_count * 81
        for spring_id in range(self.spring_count):
            node = self.spring_nodes[spring_id]
            _store_direct_block(
                target,
                assemble_hash,
                spring_offset + spring_id * 9,
                -1,
                node,
                node,
                self.spring_stiffness[spring_id] * ti.Matrix.identity(self.real_type, 3),
            )

        sdf_offset = spring_offset + self.spring_count * 9
        for sdf_id in range(self.sdf_count):
            node = self.sdf_nodes[sdf_id]
            normal = self.sdf_normal[sdf_id]
            distance = (positions[node] - self.sdf_target[sdf_id]).dot(normal)
            coefficient = 0.0
            if distance <= self.sdf_dhat[sdf_id]:
                coefficient = (
                    self.sdf_stiffness[sdf_id]
                    * self.node_area[node]
                    / self.sdf_dhat[sdf_id]
                    * (1.0 - distance / self.sdf_dhat[sdf_id])
                )
            _store_direct_block(
                target,
                assemble_hash,
                sdf_offset + sdf_id * 9,
                -1,
                node,
                node,
                coefficient * normal.outer_product(normal),
            )

    def scatter_stiffness_to_coo(self, matrix, offset=0):
        offset = int(offset)
        bending_offset = offset + self.element_count * 81
        optional_offset = bending_offset + self.bending_element_count * 144
        self._assemble_membrane_tangent(self.stiffness_positions)
        if self.project_pd:
            self._project_membrane_tangents()
        if self.bending_model_id == 2:
            self._assemble_bending_tangent(self.stiffness_positions)
        self._assemble_membrane_stiffness_direct(self.stiffness_positions, matrix, offset, 0, False)
        if self.bending_element_count:
            self._assemble_bending_stiffness_direct(self.stiffness_positions, matrix, bending_offset, 0, False)
        self._assemble_optional_stiffness_direct(self.stiffness_positions, matrix, optional_offset, 0, False)

    def scatter_stiffness_to_hash(self, matrix):
        raw_offset = matrix.reserve_raw_block_slots(self.stiffness_block_pair_count)
        bending_offset = raw_offset + self.element_count * 6
        optional_offset = bending_offset + self.bending_element_count * 12
        self._assemble_membrane_tangent(self.stiffness_positions)
        if self.project_pd:
            self._project_membrane_tangents()
        if self.bending_model_id == 2:
            self._assemble_bending_tangent(self.stiffness_positions)
        self._assemble_membrane_stiffness_direct(self.stiffness_positions, matrix, 0, raw_offset, True)
        if self.bending_element_count:
            self._assemble_bending_stiffness_direct(self.stiffness_positions, matrix, 0, bending_offset, True)
        self._assemble_optional_stiffness_direct(self.stiffness_positions, matrix, 0, optional_offset, True)

    def _assemble_stiffness_matrix(self):
        self._copy_positions_from_field(self.positions, self.stiffness_positions)
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

    def assemble_device(self, positions=None, need_stiffness=False):
        source = getattr(self, "bound_positions", self.positions) if positions is None else positions
        if source is not self.positions:
            self._copy_positions_from_field(source, self.positions)
        self.internal_force.fill(0.0)
        self.total_energy[None] = 0.0
        self._assemble_membrane_force()
        if self.bending_element_count:
            self._assemble_bending_force()
        if self.stitch_count or self.spring_count or self.sdf_count:
            self._assemble_optional_energy_force()
        stiffness = self._assemble_stiffness_matrix() if need_stiffness else None
        return self.internal_force, stiffness

    def assemble_bending_force_device(self, positions=None):
        """Return only the bending force, reusing the device assembly fields."""
        source = getattr(self, "bound_positions", self.positions) if positions is None else positions
        if source is not self.positions:
            self._copy_positions_from_field(source, self.positions)
        self.internal_force.fill(0.0)
        self.total_energy[None] = 0.0
        if self.bending_element_count:
            self._assemble_bending_force()
        return self.internal_force

    def assemble_system(self, positions, need_stiffness=False, need_stress=False):
        """Explicit host snapshot adapter for tests and output only."""
        if hasattr(positions, "to_numpy"):
            source = positions
        else:
            values = np.ascontiguousarray(positions, dtype=self.numpy_type)
            if values.shape != (self.node_count, 3):
                raise ValueError("cloth positions must have shape (number_of_nodes, 3)")
            self.positions.from_numpy(values)
            source = self.positions
        force, stiffness = self.assemble_device(source, need_stiffness=need_stiffness)
        stress = None
        if need_stress:
            gradients = self.deformation_gradient.to_numpy()
            stress = np.asarray([self.material.cauchy_stress(value) for value in gradients])
        result = (
            float(self.total_energy[None]),
            force.to_numpy(),
            stiffness,
            stress,
        )
        return result

    def compute_deformation_gradient(self, positions):
        self.assemble_system(positions)
        return self.deformation_gradient.to_numpy()


__all__ = ["ClothAssembler"]
