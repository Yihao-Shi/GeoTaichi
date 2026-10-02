import taichi as ti

from src.utils.constants import PENALTY, BLOCK_SZ, Threshold
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefination import vec2f, vec3f, vec4f, vec9f, vec2i, vec3i
from src.utils.VectorFunction import summation
from src.utils.ScalarFunction import linearize, vectorize_id
from src.mpm.sparse_grid.BlockSparseGrid import compact_node_grid_coord, compact_node_id
from src.levelset.FluidLevelSetKernel import free_surface_theta as pressure_free_surface_theta


# ========================================================= #
#     Matrix free Preconditioning conjuction gradient       #
# ========================================================= #
@ti.kernel
def unknow_reset(old_total_dofs: int, unknown_vector: ti.template()):
    for i in range(old_total_dofs):
        unknown_vector[i] = 0.0


@ti.kernel
def matrix_reset_(total_dofs: int, diag_A: ti.template(), mass_matrix: ti.template(), unknown_vector: ti.template()):
    for i in range(total_dofs):
        diag_A[i] = mass_matrix[i]
        unknown_vector[i] = 0.0


@ti.kernel
def local_stiffness_reset(old_total_nodes: int, influenced_dofs: int, local_stiffness: ti.template()):
    for i in range(old_total_nodes * influenced_dofs * influenced_dofs):
        local_stiffness[i] = 0.0


@ti.kernel
def kernel_symmetrize_local_stiffness(particleNum: int, influenced_dofs: int, local_stiffness: ti.template()):
    for np, i, j in ti.ndrange(particleNum, influenced_dofs, influenced_dofs):
        if i < j:
            value = 0.5 * (local_stiffness[np, i, j] + local_stiffness[np, j, i])
            local_stiffness[np, i, j] = value
            local_stiffness[np, j, i] = value


@ti.kernel
def MvP_reset(total_dofs: int, m_dot_v: ti.template()):
    for i in range(total_dofs):
        m_dot_v[i] = 0.0


@ti.func
def assemble_elastic_diagonal_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec2f(0.0, 0.0)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3])
        + 0.5 * jdshape[1] * (2.0 * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    )
    local_stiffness[1] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1])
        - 0.5 * jdshape[0] * (-2.0 * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    )
    return local_stiffness


@ti.func
def assemble_elastic_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec4f(0.0, 0.0, 0.0, 0.0)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3])
        + 0.5 * jdshape[1] * (2.0 * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    )
    local_stiffness[1] = (
        idshape[0] * jdshape[1] * material_stiffness[0, 1]
        + idshape[1] * jdshape[0] * material_stiffness[3, 3]
        + jdshape[1] * (idshape[0] * stress[0] + idshape[1] * stress[3])
        - 0.5 * jdshape[0] * (2.0 * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    )

    local_stiffness[2] = (
        idshape[1] * jdshape[0] * material_stiffness[1, 0]
        + idshape[0] * jdshape[1] * material_stiffness[3, 3]
        + jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1])
        + 0.5 * jdshape[1] * (-2.0 * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    )
    local_stiffness[3] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1])
        - 0.5 * jdshape[0] * (-2.0 * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    )
    return local_stiffness


@ti.func
def assemble_diagonal_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec2f(0.0, 0.0)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[0] * material_stiffness[3, 0]
        + idshape[0] * jdshape[1] * material_stiffness[0, 3]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3])
        + 0.5 * jdshape[1] * (2.0 * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    )
    local_stiffness[1] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[1] * material_stiffness[3, 1]
        + idshape[1] * jdshape[0] * material_stiffness[1, 3]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1])
        - 0.5 * jdshape[0] * (-2.0 * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    )
    return local_stiffness


@ti.func
def assemble_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec4f(0.0, 0.0, 0.0, 0.0)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[0] * material_stiffness[3, 0]
        + idshape[0] * jdshape[1] * material_stiffness[0, 3]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3])
        + 0.5 * jdshape[1] * (2.0 * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    )
    local_stiffness[1] = (
        idshape[0] * jdshape[1] * material_stiffness[0, 1]
        + idshape[1] * jdshape[1] * material_stiffness[3, 1]
        + idshape[0] * jdshape[0] * material_stiffness[0, 3]
        + idshape[1] * jdshape[0] * material_stiffness[3, 3]
        + jdshape[1] * (idshape[0] * stress[0] + idshape[1] * stress[3])
        - 0.5 * jdshape[0] * (2.0 * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    )

    local_stiffness[2] = (
        idshape[1] * jdshape[0] * material_stiffness[1, 0]
        + idshape[0] * jdshape[0] * material_stiffness[3, 0]
        + idshape[1] * jdshape[1] * material_stiffness[1, 3]
        + idshape[0] * jdshape[1] * material_stiffness[3, 3]
        + jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1])
        + 0.5 * jdshape[1] * (-2.0 * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    )
    local_stiffness[3] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[1] * material_stiffness[3, 1]
        + idshape[1] * jdshape[0] * material_stiffness[1, 3]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1])
        - 0.5 * jdshape[0] * (-2.0 * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    )
    return local_stiffness


@ti.kernel
def kernel_assemble_local_stiffness_2D(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    stiffness_matrix: ti.template(),
    local_stiffness: ti.template(),
    assemble_matrix: ti.template(),
):
    for np in range(particleNum):
        material_stiffness = stiffness_matrix[np]
        offset = np * total_nodes
        volume = particle[np].vol
        stress = particle[np].stress
        for lni in range(offset, offset + int(node_size[np])):
            idshape = dshapefn[lni]
            local_lni = lni - offset
            for lnj in range(offset, offset + int(node_size[np])):
                jdshape = dshapefn[lnj]
                local_lnj = lnj - offset
                particle_stiffness = assemble_matrix(idshape, jdshape, material_stiffness, stress) * volume
                local_stiffness[np, 2 * local_lni, 2 * local_lnj] = particle_stiffness[0]
                local_stiffness[np, 2 * local_lni, 2 * local_lnj + 1] = particle_stiffness[1]
                local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj] = particle_stiffness[2]
                local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj + 1] = particle_stiffness[3]


@ti.func
def assemble_stress_stiffness_matrix(idshape, jdshape, stress):
    local_stiffness = vec9f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sxx, syy, szz = stress[0], stress[1], stress[2]
    sxy, syz, sxz = stress[3], stress[4], stress[5]

    rowx = idshape[0] * sxx + idshape[1] * sxy + idshape[2] * sxz
    rowy = idshape[0] * sxy + idshape[1] * syy + idshape[2] * syz
    rowz = idshape[0] * sxz + idshape[1] * syz + idshape[2] * szz

    # Column x: L = e_x \otimes grad(N_j)
    r0 = -jdshape[1] * sxy - jdshape[2] * sxz
    r1 = jdshape[1] * sxy
    r2 = jdshape[2] * sxz
    r3 = -0.5 * jdshape[2] * syz - 0.5 * jdshape[1] * (syy - sxx)
    r4 = 0.5 * jdshape[1] * sxz + 0.5 * jdshape[2] * sxy
    r5 = -0.5 * jdshape[1] * syz + 0.5 * jdshape[2] * (sxx - szz)
    local_stiffness[0] = jdshape[0] * rowx - (idshape[0] * r0 + idshape[1] * r3 + idshape[2] * r5)
    local_stiffness[3] = jdshape[0] * rowy - (idshape[0] * r3 + idshape[1] * r1 + idshape[2] * r4)
    local_stiffness[6] = jdshape[0] * rowz - (idshape[0] * r5 + idshape[1] * r4 + idshape[2] * r2)

    # Column y: L = e_y \otimes grad(N_j)
    r0 = jdshape[0] * sxy
    r1 = -jdshape[0] * sxy - jdshape[2] * syz
    r2 = jdshape[2] * syz
    r3 = -0.5 * jdshape[2] * sxz + 0.5 * jdshape[0] * (syy - sxx)
    r4 = -0.5 * jdshape[0] * sxz - 0.5 * jdshape[2] * (szz - syy)
    r5 = 0.5 * jdshape[2] * sxy + 0.5 * jdshape[0] * syz
    local_stiffness[1] = jdshape[1] * rowx - (idshape[0] * r0 + idshape[1] * r3 + idshape[2] * r5)
    local_stiffness[4] = jdshape[1] * rowy - (idshape[0] * r3 + idshape[1] * r1 + idshape[2] * r4)
    local_stiffness[7] = jdshape[1] * rowz - (idshape[0] * r5 + idshape[1] * r4 + idshape[2] * r2)

    # Column z: L = e_z \otimes grad(N_j)
    r0 = jdshape[0] * sxz
    r1 = jdshape[1] * syz
    r2 = -jdshape[1] * syz - jdshape[0] * sxz
    r3 = 0.5 * jdshape[0] * syz + 0.5 * jdshape[1] * sxz
    r4 = -0.5 * jdshape[0] * sxy + 0.5 * jdshape[1] * (szz - syy)
    r5 = -0.5 * jdshape[1] * sxy - 0.5 * jdshape[0] * (sxx - szz)
    local_stiffness[2] = jdshape[2] * rowx - (idshape[0] * r0 + idshape[1] * r3 + idshape[2] * r5)
    local_stiffness[5] = jdshape[2] * rowy - (idshape[0] * r3 + idshape[1] * r1 + idshape[2] * r4)
    local_stiffness[8] = jdshape[2] * rowz - (idshape[0] * r5 + idshape[1] * r4 + idshape[2] * r2)
    return local_stiffness


@ti.func
def _coo_neighbor_nodes(influenced_node: int):
    neighbor_width = 2 * influenced_node - 1
    neighbor_nodes = neighbor_width * neighbor_width
    if ti.static(GlobalVariable.DIMENSION == 3):
        neighbor_nodes *= neighbor_width
    return neighbor_nodes


@ti.func
def _coo_center_slot(influenced_node: int):
    neighbor_width = 2 * influenced_node - 1
    radius = influenced_node - 1
    center = radius + radius * neighbor_width
    if ti.static(GlobalVariable.DIMENSION == 3):
        center += radius * neighbor_width * neighbor_width
    return center


@ti.func
def _coo_slot_index(block: int, slot: int, row_comp: int, col_comp: int, neighbor_nodes: int):
    return ((block * neighbor_nodes + slot) * GlobalVariable.DIMENSION + row_comp) * GlobalVariable.DIMENSION + col_comp


@ti.func
def _dense_node_grid_coord(node_id: int, gnum: ti.template()):
    coord = ti.Vector.zero(int, GlobalVariable.DIMENSION)
    if ti.static(GlobalVariable.DIMENSION == 2):
        coord[0] = node_id % gnum[0]
        coord[1] = node_id // gnum[0]
    else:
        layer = gnum[0] * gnum[1]
        coord[2] = node_id // layer
        local = node_id - coord[2] * layer
        coord[1] = local // gnum[0]
        coord[0] = local - coord[1] * gnum[0]
    return coord


@ti.func
def _coo_neighbor_slot(coord_i, coord_j, influenced_node: int):
    neighbor_width = 2 * influenced_node - 1
    radius = influenced_node - 1
    slot = -1
    dx = coord_j[0] - coord_i[0] + radius
    dy = coord_j[1] - coord_i[1] + radius
    if ti.static(GlobalVariable.DIMENSION == 2):
        if dx >= 0 and dx < neighbor_width and dy >= 0 and dy < neighbor_width:
            slot = dx + dy * neighbor_width
    else:
        dz = coord_j[2] - coord_i[2] + radius
        if dx >= 0 and dx < neighbor_width and dy >= 0 and dy < neighbor_width and dz >= 0 and dz < neighbor_width:
            slot = dx + dy * neighbor_width + dz * neighbor_width * neighbor_width
    return slot


@ti.kernel
def kernel_assemble_coo_mass(
    active_dofs: int,
    influenced_node: int,
    mass_matrix: ti.template(),
    rows: ti.template(),
    cols: ti.template(),
    values: ti.template(),
):
    neighbor_nodes = _coo_neighbor_nodes(influenced_node)
    center = _coo_center_slot(influenced_node)
    for dof in range(active_dofs):
        block = dof // GlobalVariable.DIMENSION
        comp = dof - block * GlobalVariable.DIMENSION
        idx = _coo_slot_index(block, center, comp, comp, neighbor_nodes)
        rows[idx] = dof
        cols[idx] = dof
        values[idx] += mass_matrix[dof]


@ti.kernel
def kernel_assemble_coo_stiffness(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    influenced_node: int,
    gnum: ti.template(),
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
    rows: ti.template(),
    cols: ti.template(),
    values: ti.template(),
):
    neighbor_nodes = _coo_neighbor_nodes(influenced_node)
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        material_stiffness = stiffness_matrix[np]
        stress = particle[np].stress
        volume = particle[np].vol
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            coord_i = _dense_node_grid_coord(nodeIDi, gnum)
            dofsi = flag[nodeIDi + bodyID * gridSum]
            block_i = dofsi // GlobalVariable.DIMENSION
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                coord_j = _dense_node_grid_coord(nodeIDj, gnum)
                slot = _coo_neighbor_slot(coord_i, coord_j, influenced_node)
                if slot >= 0:
                    dofsj = flag[nodeIDj + bodyID * gridSum]
                    block = assemble_matrix(dshapefn[lni], dshapefn[lnj], material_stiffness, stress) * volume
                    for di, dj in ti.static(ti.ndrange(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)):
                        idx = _coo_slot_index(block_i, slot, di, dj, neighbor_nodes)
                        rows[idx] = dofsi + di
                        cols[idx] = dofsj + dj
                        ti.atomic_add(values[idx], block[GlobalVariable.DIMENSION * di + dj])


@ti.kernel
def kernel_assemble_coo_stiffness_sparse(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    influenced_node: int,
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
    rows: ti.template(),
    cols: ti.template(),
    values: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    active_block_ids: ti.template(),
):
    neighbor_nodes = _coo_neighbor_nodes(influenced_node)
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        material_stiffness = stiffness_matrix[np]
        stress = particle[np].stress
        volume = particle[np].vol
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            coord_i = compact_node_grid_coord(nodeIDi, block_count, block_size, block_volume, active_block_ids)
            dofsi = flag[nodeIDi + bodyID * gridSum]
            block_i = dofsi // GlobalVariable.DIMENSION
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                coord_j = compact_node_grid_coord(nodeIDj, block_count, block_size, block_volume, active_block_ids)
                slot = _coo_neighbor_slot(coord_i, coord_j, influenced_node)
                if slot >= 0:
                    dofsj = flag[nodeIDj + bodyID * gridSum]
                    block = assemble_matrix(dshapefn[lni], dshapefn[lnj], material_stiffness, stress) * volume
                    for di, dj in ti.static(ti.ndrange(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)):
                        idx = _coo_slot_index(block_i, slot, di, dj, neighbor_nodes)
                        rows[idx] = dofsi + di
                        cols[idx] = dofsj + dj
                        ti.atomic_add(values[idx], block[GlobalVariable.DIMENSION * di + dj])


@ti.func
def assemble_elastic_diagonal_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec3f(0.0, 0.0, 0.0)
    stress_stiffness = assemble_stress_stiffness_matrix(idshape, jdshape, stress)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + idshape[2] * jdshape[2] * material_stiffness[5, 5]
        + stress_stiffness[0]
    )
    local_stiffness[1] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + idshape[2] * jdshape[2] * material_stiffness[4, 4]
        + stress_stiffness[4]
    )
    local_stiffness[2] = (
        idshape[2] * jdshape[2] * material_stiffness[2, 2]
        + idshape[1] * jdshape[1] * material_stiffness[4, 4]
        + idshape[0] * jdshape[0] * material_stiffness[5, 5]
        + stress_stiffness[8]
    )
    return local_stiffness


@ti.func
def assemble_diagonal_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec3f(0.0, 0.0, 0.0)
    stress_stiffness = assemble_stress_stiffness_matrix(idshape, jdshape, stress)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[0] * material_stiffness[3, 0]
        + idshape[2] * jdshape[0] * material_stiffness[5, 0]
        + idshape[0] * jdshape[1] * material_stiffness[0, 3]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + idshape[2] * jdshape[1] * material_stiffness[5, 3]
        + idshape[0] * jdshape[2] * material_stiffness[0, 5]
        + idshape[1] * jdshape[2] * material_stiffness[3, 5]
        + idshape[2] * jdshape[2] * material_stiffness[5, 5]
        + stress_stiffness[0]
    )

    local_stiffness[1] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[1] * material_stiffness[3, 1]
        + idshape[2] * jdshape[1] * material_stiffness[4, 1]
        + idshape[1] * jdshape[0] * material_stiffness[1, 3]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + idshape[2] * jdshape[0] * material_stiffness[4, 3]
        + idshape[1] * jdshape[2] * material_stiffness[1, 4]
        + idshape[0] * jdshape[2] * material_stiffness[3, 4]
        + idshape[2] * jdshape[2] * material_stiffness[4, 4]
        + stress_stiffness[4]
    )

    local_stiffness[2] = (
        idshape[2] * jdshape[2] * material_stiffness[2, 2]
        + idshape[1] * jdshape[2] * material_stiffness[4, 2]
        + idshape[0] * jdshape[2] * material_stiffness[5, 2]
        + idshape[2] * jdshape[1] * material_stiffness[2, 4]
        + idshape[1] * jdshape[1] * material_stiffness[4, 4]
        + idshape[0] * jdshape[1] * material_stiffness[5, 4]
        + idshape[2] * jdshape[0] * material_stiffness[2, 5]
        + idshape[1] * jdshape[0] * material_stiffness[4, 5]
        + idshape[0] * jdshape[0] * material_stiffness[5, 5]
        + stress_stiffness[8]
    )
    return local_stiffness


@ti.func
def assemble_elastic_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec9f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stress_stiffness = assemble_stress_stiffness_matrix(idshape, jdshape, stress)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + idshape[2] * jdshape[2] * material_stiffness[5, 5]
    )
    local_stiffness[1] = (
        idshape[0] * jdshape[1] * material_stiffness[0, 1] + idshape[1] * jdshape[0] * material_stiffness[3, 3]
    )
    local_stiffness[2] = (
        idshape[0] * jdshape[2] * material_stiffness[0, 2] + idshape[2] * jdshape[0] * material_stiffness[5, 5]
    )

    local_stiffness[3] = (
        idshape[1] * jdshape[0] * material_stiffness[1, 0] + idshape[0] * jdshape[1] * material_stiffness[3, 3]
    )
    local_stiffness[4] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + idshape[2] * jdshape[2] * material_stiffness[4, 4]
    )
    local_stiffness[5] = (
        idshape[1] * jdshape[2] * material_stiffness[1, 2] + idshape[2] * jdshape[1] * material_stiffness[4, 4]
    )

    local_stiffness[6] = (
        idshape[2] * jdshape[0] * material_stiffness[2, 0] + idshape[0] * jdshape[2] * material_stiffness[5, 5]
    )
    local_stiffness[7] = (
        idshape[2] * jdshape[1] * material_stiffness[2, 1] + idshape[1] * jdshape[2] * material_stiffness[4, 4]
    )
    local_stiffness[8] = (
        idshape[2] * jdshape[2] * material_stiffness[2, 2]
        + idshape[1] * jdshape[1] * material_stiffness[4, 4]
        + idshape[0] * jdshape[0] * material_stiffness[5, 5]
    )
    local_stiffness += stress_stiffness
    return local_stiffness


@ti.func
def assemble_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec9f(0, 0, 0, 0, 0, 0, 0, 0, 0)
    local_stiffness[0] = (
        idshape[0] * jdshape[0] * material_stiffness[0, 0]
        + idshape[1] * jdshape[0] * material_stiffness[3, 0]
        + idshape[2] * jdshape[0] * material_stiffness[5, 0]
        + idshape[0] * jdshape[1] * material_stiffness[0, 3]
        + idshape[1] * jdshape[1] * material_stiffness[3, 3]
        + idshape[2] * jdshape[1] * material_stiffness[5, 3]
        + idshape[0] * jdshape[2] * material_stiffness[0, 5]
        + idshape[1] * jdshape[2] * material_stiffness[3, 5]
        + idshape[2] * jdshape[2] * material_stiffness[5, 5]
    )
    local_stiffness[1] = (
        idshape[0] * jdshape[1] * material_stiffness[0, 1]
        + idshape[1] * jdshape[1] * material_stiffness[3, 1]
        + idshape[2] * jdshape[1] * material_stiffness[5, 1]
        + idshape[0] * jdshape[0] * material_stiffness[0, 3]
        + idshape[1] * jdshape[0] * material_stiffness[3, 3]
        + idshape[2] * jdshape[0] * material_stiffness[5, 3]
        + idshape[0] * jdshape[2] * material_stiffness[0, 4]
        + idshape[1] * jdshape[2] * material_stiffness[3, 4]
        + idshape[2] * jdshape[2] * material_stiffness[5, 4]
    )
    local_stiffness[2] = (
        idshape[0] * jdshape[2] * material_stiffness[0, 2]
        + idshape[1] * jdshape[2] * material_stiffness[3, 2]
        + idshape[2] * jdshape[2] * material_stiffness[5, 2]
        + idshape[0] * jdshape[1] * material_stiffness[0, 4]
        + idshape[1] * jdshape[1] * material_stiffness[3, 4]
        + idshape[2] * jdshape[1] * material_stiffness[5, 4]
        + idshape[0] * jdshape[0] * material_stiffness[0, 5]
        + idshape[1] * jdshape[0] * material_stiffness[3, 5]
        + idshape[2] * jdshape[0] * material_stiffness[5, 5]
    )

    local_stiffness[3] = (
        idshape[1] * jdshape[0] * material_stiffness[1, 0]
        + idshape[0] * jdshape[0] * material_stiffness[3, 0]
        + idshape[2] * jdshape[0] * material_stiffness[4, 0]
        + idshape[1] * jdshape[1] * material_stiffness[1, 3]
        + idshape[0] * jdshape[1] * material_stiffness[3, 3]
        + idshape[2] * jdshape[1] * material_stiffness[4, 3]
        + idshape[1] * jdshape[2] * material_stiffness[1, 5]
        + idshape[0] * jdshape[2] * material_stiffness[3, 5]
        + idshape[2] * jdshape[2] * material_stiffness[4, 5]
    )
    local_stiffness[4] = (
        idshape[1] * jdshape[1] * material_stiffness[1, 1]
        + idshape[0] * jdshape[1] * material_stiffness[3, 1]
        + idshape[2] * jdshape[1] * material_stiffness[4, 1]
        + idshape[1] * jdshape[0] * material_stiffness[1, 3]
        + idshape[0] * jdshape[0] * material_stiffness[3, 3]
        + idshape[2] * jdshape[0] * material_stiffness[4, 3]
        + idshape[1] * jdshape[2] * material_stiffness[1, 4]
        + idshape[0] * jdshape[2] * material_stiffness[3, 4]
        + idshape[2] * jdshape[2] * material_stiffness[4, 4]
    )
    local_stiffness[5] = (
        idshape[1] * jdshape[2] * material_stiffness[1, 2]
        + idshape[0] * jdshape[2] * material_stiffness[3, 2]
        + idshape[2] * jdshape[2] * material_stiffness[4, 2]
        + idshape[1] * jdshape[1] * material_stiffness[1, 4]
        + idshape[0] * jdshape[1] * material_stiffness[3, 4]
        + idshape[2] * jdshape[1] * material_stiffness[4, 4]
        + idshape[1] * jdshape[0] * material_stiffness[1, 5]
        + idshape[0] * jdshape[0] * material_stiffness[3, 5]
        + idshape[2] * jdshape[0] * material_stiffness[4, 5]
    )

    local_stiffness[6] = (
        idshape[2] * jdshape[0] * material_stiffness[2, 0]
        + idshape[1] * jdshape[0] * material_stiffness[4, 0]
        + idshape[0] * jdshape[0] * material_stiffness[5, 0]
        + idshape[2] * jdshape[1] * material_stiffness[2, 3]
        + idshape[1] * jdshape[1] * material_stiffness[4, 3]
        + idshape[0] * jdshape[1] * material_stiffness[5, 3]
        + idshape[2] * jdshape[2] * material_stiffness[2, 5]
        + idshape[1] * jdshape[2] * material_stiffness[4, 5]
        + idshape[0] * jdshape[2] * material_stiffness[5, 5]
    )
    local_stiffness[7] = (
        idshape[2] * jdshape[1] * material_stiffness[2, 1]
        + idshape[1] * jdshape[1] * material_stiffness[4, 1]
        + idshape[0] * jdshape[1] * material_stiffness[5, 1]
        + idshape[2] * jdshape[0] * material_stiffness[2, 3]
        + idshape[1] * jdshape[0] * material_stiffness[4, 3]
        + idshape[0] * jdshape[0] * material_stiffness[5, 3]
        + idshape[2] * jdshape[2] * material_stiffness[2, 4]
        + idshape[1] * jdshape[2] * material_stiffness[4, 4]
        + idshape[0] * jdshape[2] * material_stiffness[5, 4]
    )
    local_stiffness[8] = (
        idshape[2] * jdshape[2] * material_stiffness[2, 2]
        + idshape[1] * jdshape[2] * material_stiffness[4, 2]
        + idshape[0] * jdshape[2] * material_stiffness[5, 2]
        + idshape[2] * jdshape[1] * material_stiffness[2, 4]
        + idshape[1] * jdshape[1] * material_stiffness[4, 4]
        + idshape[0] * jdshape[1] * material_stiffness[5, 4]
        + idshape[2] * jdshape[0] * material_stiffness[2, 5]
        + idshape[1] * jdshape[0] * material_stiffness[4, 5]
        + idshape[0] * jdshape[0] * material_stiffness[5, 5]
    )
    local_stiffness += assemble_stress_stiffness_matrix(idshape, jdshape, stress)
    return local_stiffness


@ti.kernel
def kernel_assemble_local_stiffness(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    stiffness_matrix: ti.template(),
    local_stiffness: ti.template(),
    assemble_matrix: ti.template(),
):
    for np in range(particleNum):
        material_stiffness = stiffness_matrix[np]
        offset = np * total_nodes
        volume = particle[np].vol
        stress = particle[np].stress
        for lni in range(offset, offset + int(node_size[np])):
            idshape = dshapefn[lni]
            local_lni = lni - offset
            for lnj in range(offset, offset + int(node_size[np])):
                jdshape = dshapefn[lnj]
                local_lnj = lnj - offset
                particle_stiffness = assemble_matrix(idshape, jdshape, material_stiffness, stress) * volume
                local_stiffness[np, 3 * local_lni, 3 * local_lnj] = particle_stiffness[0]
                local_stiffness[np, 3 * local_lni, 3 * local_lnj + 1] = particle_stiffness[1]
                local_stiffness[np, 3 * local_lni, 3 * local_lnj + 2] = particle_stiffness[2]
                local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj] = particle_stiffness[3]
                local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 1] = particle_stiffness[4]
                local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 2] = particle_stiffness[5]
                local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj] = particle_stiffness[6]
                local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 1] = particle_stiffness[7]
                local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 2] = particle_stiffness[8]


@ti.kernel
def kernel_compute_mass_matrix(
    gridSum: int,
    beta: float,
    cutoff: float,
    dt: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    mass_matrix: ti.template(),
):
    constant = 1.0 / dt[None] / dt[None] / beta
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            nodal_mass = node[ng, nb].m
            if nodal_mass > cutoff:
                dof0 = flag[ng + nb * gridSum]
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    mass_matrix[dof0 + d] = nodal_mass * constant


@ti.kernel
def kernel_compute_penalty_matrix(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    mass_matrix: ti.template(),
):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            mass_matrix[global_dof + dofID] += PENALTY


@ti.kernel
def kernel_compute_penalty_matrix_sparse(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    mass_matrix: ti.template(),
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            mass_matrix[global_dof + dofID] += PENALTY


@ti.kernel
def kernel_moment_balance_cg_2D(
    gridSum: int,
    total_nodes: int,
    total_dofs: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    mass_matrix: ti.template(),
    local_stiffness: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    for ndof in range(total_dofs):
        m_dot_v[ndof] = mass_matrix[ndof] * unknown_vector[ndof]

    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            local_lni = lni - offset
            dofsi = flag[nodeIDi + bodyID * gridSum]
            Axb, Ayb = 0.0, 0.0
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                local_lnj = lnj - offset
                dofsj = flag[nodeIDj + bodyID * gridSum]
                Axb += (
                    local_stiffness[np, 2 * local_lni, 2 * local_lnj] * unknown_vector[dofsj]
                    + local_stiffness[np, 2 * local_lni, 2 * local_lnj + 1] * unknown_vector[dofsj + 1]
                )
                Ayb += (
                    local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj] * unknown_vector[dofsj]
                    + local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj + 1] * unknown_vector[dofsj + 1]
                )
            m_dot_v[dofsi] += Axb
            m_dot_v[dofsi + 1] += Ayb


@ti.kernel
def kernel_moment_balance_cg(
    gridSum: int,
    total_nodes: int,
    total_dofs: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    mass_matrix: ti.template(),
    local_stiffness: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    for ndof in range(total_dofs):
        m_dot_v[ndof] = mass_matrix[ndof] * unknown_vector[ndof]

    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            local_lni = lni - offset
            dofsi = flag[nodeIDi + bodyID * gridSum]
            Axb, Ayb, Azb = 0.0, 0.0, 0.0
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                local_lnj = lnj - offset
                dofsj = flag[nodeIDj + bodyID * gridSum]
                Axb += (
                    local_stiffness[np, 3 * local_lni, 3 * local_lnj] * unknown_vector[dofsj]
                    + local_stiffness[np, 3 * local_lni, 3 * local_lnj + 1] * unknown_vector[dofsj + 1]
                    + local_stiffness[np, 3 * local_lni, 3 * local_lnj + 2] * unknown_vector[dofsj + 2]
                )
                Ayb += (
                    local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj] * unknown_vector[dofsj]
                    + local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 1] * unknown_vector[dofsj + 1]
                    + local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 2] * unknown_vector[dofsj + 2]
                )
                Azb += (
                    local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj] * unknown_vector[dofsj]
                    + local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 1] * unknown_vector[dofsj + 1]
                    + local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 2] * unknown_vector[dofsj + 2]
                )
            m_dot_v[dofsi] += Axb
            m_dot_v[dofsi + 1] += Ayb
            m_dot_v[dofsi + 2] += Azb


@ti.kernel
def kernel_moment_balance_direct_2D(
    gridSum: int,
    total_nodes: int,
    total_dofs: int,
    particleNum: int,
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
    mass_matrix: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    for ndof in range(total_dofs):
        m_dot_v[ndof] = mass_matrix[ndof] * unknown_vector[ndof]
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        material_stiffness = stiffness_matrix[np]
        stress = particle[np].stress
        volume = particle[np].vol
        for lni in range(offset, offset + int(node_size[np])):
            dofsi = flag[LnID[lni] + bodyID * gridSum]
            result = ti.Vector.zero(float, 2)
            for lnj in range(offset, offset + int(node_size[np])):
                dofsj = flag[LnID[lnj] + bodyID * gridSum]
                block = assemble_matrix(dshapefn[lni], dshapefn[lnj], material_stiffness, stress) * volume
                for row, column in ti.static(ti.ndrange(2, 2)):
                    result[row] += block[2 * row + column] * unknown_vector[dofsj + column]
            for component in ti.static(range(2)):
                ti.atomic_add(m_dot_v[dofsi + component], result[component])


@ti.kernel
def kernel_moment_balance_direct(
    gridSum: int,
    total_nodes: int,
    total_dofs: int,
    particleNum: int,
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
    mass_matrix: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    for ndof in range(total_dofs):
        m_dot_v[ndof] = mass_matrix[ndof] * unknown_vector[ndof]
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        material_stiffness = stiffness_matrix[np]
        stress = particle[np].stress
        volume = particle[np].vol
        for lni in range(offset, offset + int(node_size[np])):
            dofsi = flag[LnID[lni] + bodyID * gridSum]
            result = ti.Vector.zero(float, 3)
            for lnj in range(offset, offset + int(node_size[np])):
                dofsj = flag[LnID[lnj] + bodyID * gridSum]
                block = assemble_matrix(dshapefn[lni], dshapefn[lnj], material_stiffness, stress) * volume
                for row, column in ti.static(ti.ndrange(3, 3)):
                    result[row] += block[3 * row + column] * unknown_vector[dofsj + column]
            for component in ti.static(range(3)):
                ti.atomic_add(m_dot_v[dofsi + component], result[component])


@ti.kernel
def kernel_preconditioning_matrix_2D(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    diag_A: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    local_stiffness: ti.template(),
):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            local_ln = ln - offset
            dofs = flag[nodeID + bodyID * gridSum]
            diag_A[dofs] += local_stiffness[np, 2 * local_ln, 2 * local_ln]
            diag_A[dofs + 1] += local_stiffness[np, 2 * local_ln + 1, 2 * local_ln + 1]


@ti.kernel
def kernel_preconditioning_matrix(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    diag_A: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    local_stiffness: ti.template(),
):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            local_ln = ln - offset
            dofs = flag[nodeID + bodyID * gridSum]
            diag_A[dofs] += local_stiffness[np, 3 * local_ln, 3 * local_ln]
            diag_A[dofs + 1] += local_stiffness[np, 3 * local_ln + 1, 3 * local_ln + 1]
            diag_A[dofs + 2] += local_stiffness[np, 3 * local_ln + 2, 3 * local_ln + 2]


@ti.kernel
def kernel_preconditioning_matrix_direct_2D(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    diag_A: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        material_stiffness = stiffness_matrix[np]
        stress = particle[np].stress
        volume = particle[np].vol
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            dofs = flag[nodeID + bodyID * gridSum]
            block = assemble_matrix(dshapefn[ln], dshapefn[ln], material_stiffness, stress) * volume
            ti.atomic_add(diag_A[dofs], block[0])
            ti.atomic_add(diag_A[dofs + 1], block[3])


@ti.kernel
def kernel_preconditioning_matrix_direct(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    diag_A: ti.template(),
    LnID: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        material_stiffness = stiffness_matrix[np]
        stress = particle[np].stress
        volume = particle[np].vol
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            dofs = flag[nodeID + bodyID * gridSum]
            block = assemble_matrix(dshapefn[ln], dshapefn[ln], material_stiffness, stress) * volume
            ti.atomic_add(diag_A[dofs], block[0])
            ti.atomic_add(diag_A[dofs + 1], block[4])
            ti.atomic_add(diag_A[dofs + 2], block[8])


@ti.kernel
def kernel_copy_flat_rhs_to_hash_triplet(
    active_dofs: int, right_hand_vector: ti.template(), triplet_rhs: ti.template()
):
    for ndof in range(active_dofs):
        block = ndof // GlobalVariable.DIMENSION
        comp = ndof - block * GlobalVariable.DIMENSION
        triplet_rhs[block][comp] = right_hand_vector[ndof]


@ti.kernel
def kernel_copy_hash_triplet_solution_to_flat(
    active_dofs: int, triplet_x: ti.template(), unknown_vector: ti.template()
):
    for ndof in range(active_dofs):
        block = ndof // GlobalVariable.DIMENSION
        comp = ndof - block * GlobalVariable.DIMENSION
        unknown_vector[ndof] = triplet_x[block][comp]


@ti.kernel
def kernel_copy_scalar_field(active_dofs: int, source: ti.template(), target: ti.template()):
    for ndof in range(active_dofs):
        target[ndof] = source[ndof]


@ti.kernel
def kernel_assemble_hash_triplet_mass(active_dofs: int, mass_matrix: ti.template(), hash_triplet: ti.template()):
    for ndof in range(active_dofs):
        block = ndof // GlobalVariable.DIMENSION
        comp = ndof - block * GlobalVariable.DIMENSION
        hash_triplet.diag[block][comp * GlobalVariable.DIMENSION + comp] += mass_matrix[ndof]


@ti.kernel
def kernel_assemble_hash_triplet_stiffness_2D(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
    hash_triplet: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            material_stiffness = stiffness_matrix[np]
            stress = particle[np].stress
            volume = particle[np].vol
            for lni in range(offset, offset + int(node_size[np])):
                nodeIDi = LnID[lni]
                dofsi = flag[nodeIDi + bodyID * gridSum]
                block_i = dofsi // 2
                for lnj in range(offset, offset + int(node_size[np])):
                    nodeIDj = LnID[lnj]
                    dofsj = flag[nodeIDj + bodyID * gridSum]
                    block_j = dofsj // 2
                    block = assemble_matrix(dshapefn[lni], dshapefn[lnj], material_stiffness, stress) * volume
                    if block_i == block_j:
                        for di, dj in ti.static(ti.ndrange(2, 2)):
                            value = block[2 * di + dj]
                            if ti.static(hash_triplet.matrix_symmetric):
                                value = 0.5 * (value + block[2 * dj + di])
                            ti.atomic_add(hash_triplet.diag[block_i][di * 2 + dj], value)
                    else:
                        if ti.static(hash_triplet.matrix_symmetric):
                            if block_i < block_j:
                                reverse = (
                                    assemble_matrix(dshapefn[lnj], dshapefn[lni], material_stiffness, stress) * volume
                                )
                                idx = ti.atomic_add(hash_triplet.raw_non_diag_count[0], 1)
                                if idx < hash_triplet.non_diag.blockI.shape[0]:
                                    hash_triplet.non_diag.blockI[idx] = block_i
                                    hash_triplet.non_diag.blockJ[idx] = block_j
                                    for di, dj in ti.static(ti.ndrange(2, 2)):
                                        value = 0.5 * (block[2 * di + dj] + reverse[2 * dj + di])
                                        hash_triplet.non_diag.blockH[idx][di * 2 + dj] = value
                                else:
                                    hash_triplet.overflow[0] = 1
                        else:
                            idx = ti.atomic_add(hash_triplet.raw_non_diag_count[0], 1)
                            if idx < hash_triplet.non_diag.blockI.shape[0]:
                                hash_triplet.non_diag.blockI[idx] = block_i
                                hash_triplet.non_diag.blockJ[idx] = block_j
                                for di, dj in ti.static(ti.ndrange(2, 2)):
                                    hash_triplet.non_diag.blockH[idx][di * 2 + dj] = block[2 * di + dj]
                            else:
                                hash_triplet.overflow[0] = 1


@ti.kernel
def kernel_assemble_hash_triplet_stiffness(
    gridSum: int,
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    dshapefn: ti.template(),
    flag: ti.template(),
    stiffness_matrix: ti.template(),
    assemble_matrix: ti.template(),
    hash_triplet: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            material_stiffness = stiffness_matrix[np]
            stress = particle[np].stress
            volume = particle[np].vol
            for lni in range(offset, offset + int(node_size[np])):
                nodeIDi = LnID[lni]
                dofsi = flag[nodeIDi + bodyID * gridSum]
                block_i = dofsi // 3
                for lnj in range(offset, offset + int(node_size[np])):
                    nodeIDj = LnID[lnj]
                    dofsj = flag[nodeIDj + bodyID * gridSum]
                    block_j = dofsj // 3
                    block = assemble_matrix(dshapefn[lni], dshapefn[lnj], material_stiffness, stress) * volume
                    if block_i == block_j:
                        for di, dj in ti.static(ti.ndrange(3, 3)):
                            value = block[3 * di + dj]
                            if ti.static(hash_triplet.matrix_symmetric):
                                value = 0.5 * (value + block[3 * dj + di])
                            ti.atomic_add(hash_triplet.diag[block_i][di * 3 + dj], value)
                    else:
                        if ti.static(hash_triplet.matrix_symmetric):
                            if block_i < block_j:
                                reverse = (
                                    assemble_matrix(dshapefn[lnj], dshapefn[lni], material_stiffness, stress) * volume
                                )
                                idx = ti.atomic_add(hash_triplet.raw_non_diag_count[0], 1)
                                if idx < hash_triplet.non_diag.blockI.shape[0]:
                                    hash_triplet.non_diag.blockI[idx] = block_i
                                    hash_triplet.non_diag.blockJ[idx] = block_j
                                    for di, dj in ti.static(ti.ndrange(3, 3)):
                                        value = 0.5 * (block[3 * di + dj] + reverse[3 * dj + di])
                                        hash_triplet.non_diag.blockH[idx][di * 3 + dj] = value
                                else:
                                    hash_triplet.overflow[0] = 1
                        else:
                            idx = ti.atomic_add(hash_triplet.raw_non_diag_count[0], 1)
                            if idx < hash_triplet.non_diag.blockI.shape[0]:
                                hash_triplet.non_diag.blockI[idx] = block_i
                                hash_triplet.non_diag.blockJ[idx] = block_j
                                for di, dj in ti.static(ti.ndrange(3, 3)):
                                    hash_triplet.non_diag.blockH[idx][di * 3 + dj] = block[3 * di + dj]
                            else:
                                hash_triplet.overflow[0] = 1


@ti.kernel
def kernel_assemble_displacement_load(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    right_hand_vector: ti.template(),
    diag_A: ti.template(),
):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            right_hand_vector[global_dof + dofID] = diag_A[global_dof + dofID] * displacement_constraint[i].value


@ti.kernel
def kernel_assemble_displacement_load_sparse(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    right_hand_vector: ti.template(),
    diag_A: ti.template(),
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            right_hand_vector[global_dof + dofID] = diag_A[global_dof + dofID] * displacement_constraint[i].value


@ti.kernel
def kernel_apply_dirichlet_coo(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    original_nnz: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    rows: ti.template(),
    cols: ti.template(),
    values: ti.template(),
    right_hand_vector: ti.template(),
    diag_A: ti.template(),
):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            prescribed_value = displacement_constraint[i].value
            for k in range(original_nnz):
                if cols[k] == global_dof:
                    ti.atomic_add(right_hand_vector[rows[k]], -values[k] * prescribed_value)

    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        identity = original_nnz + i
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            for k in range(original_nnz):
                if rows[k] == global_dof or cols[k] == global_dof:
                    values[k] = 0.0
            rows[identity] = global_dof
            cols[identity] = global_dof
            values[identity] = 1.0
            right_hand_vector[global_dof] = displacement_constraint[i].value
            diag_A[global_dof] = 1.0
        else:
            rows[identity] = 0
            cols[identity] = 0
            values[identity] = 0.0


@ti.kernel
def kernel_apply_dirichlet_coo_sparse(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    original_nnz: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    rows: ti.template(),
    cols: ti.template(),
    values: ti.template(),
    right_hand_vector: ti.template(),
    diag_A: ti.template(),
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            prescribed_value = displacement_constraint[i].value
            for k in range(original_nnz):
                if cols[k] == global_dof:
                    ti.atomic_add(right_hand_vector[rows[k]], -values[k] * prescribed_value)

    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        identity = original_nnz + i
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            for k in range(original_nnz):
                if rows[k] == global_dof or cols[k] == global_dof:
                    values[k] = 0.0
            rows[identity] = global_dof
            cols[identity] = global_dof
            values[identity] = 1.0
            right_hand_vector[global_dof] = displacement_constraint[i].value
            diag_A[global_dof] = 1.0
        else:
            rows[identity] = 0
            cols[identity] = 0
            values[identity] = 0.0


@ti.kernel
def kernel_apply_dirichlet_hash_triplet(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    active_nodes: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    hash_triplet: ti.template(),
    diag_A: ti.template(),
):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            block = global_dof // GlobalVariable.DIMENSION
            comp = global_dof - block * GlobalVariable.DIMENSION
            prescribed_value = displacement_constraint[i].value
            if block < active_nodes:
                for row_comp in ti.static(range(GlobalVariable.DIMENSION)):
                    h_index = row_comp * GlobalVariable.DIMENSION + comp
                    ti.atomic_add(
                        hash_triplet.rhs[block][row_comp], -hash_triplet.diag[block][h_index] * prescribed_value
                    )

                nnz = hash_triplet.non_diag.element_pair_num[0]
                for k in range(nnz):
                    bi = hash_triplet.non_diag.tripletI[k]
                    bj = hash_triplet.non_diag.tripletJ[k]
                    if 0 <= bi < active_nodes and 0 <= bj < active_nodes:
                        if bj == block:
                            for row_comp in ti.static(range(GlobalVariable.DIMENSION)):
                                h_index = row_comp * GlobalVariable.DIMENSION + comp
                                ti.atomic_add(
                                    hash_triplet.rhs[bi][row_comp],
                                    -hash_triplet.non_diag.tripletH[k][h_index] * prescribed_value,
                                )
                        if ti.static(hash_triplet.matrix_symmetric):
                            if bi == block:
                                for col_comp in ti.static(range(GlobalVariable.DIMENSION)):
                                    h_index = comp * GlobalVariable.DIMENSION + col_comp
                                    ti.atomic_add(
                                        hash_triplet.rhs[bj][col_comp],
                                        -hash_triplet.non_diag.tripletH[k][h_index] * prescribed_value,
                                    )

    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            block = global_dof // GlobalVariable.DIMENSION
            comp = global_dof - block * GlobalVariable.DIMENSION
            if block < active_nodes:
                for row_comp, col_comp in ti.static(ti.ndrange(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)):
                    if row_comp == comp or col_comp == comp:
                        hash_triplet.diag[block][row_comp * GlobalVariable.DIMENSION + col_comp] = 0.0

                nnz = hash_triplet.non_diag.element_pair_num[0]
                for k in range(nnz):
                    bi = hash_triplet.non_diag.tripletI[k]
                    bj = hash_triplet.non_diag.tripletJ[k]
                    if 0 <= bi < active_nodes and 0 <= bj < active_nodes:
                        for row_comp, col_comp in ti.static(
                            ti.ndrange(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
                        ):
                            if (bi == block and row_comp == comp) or (bj == block and col_comp == comp):
                                hash_triplet.non_diag.tripletH[k][row_comp * GlobalVariable.DIMENSION + col_comp] = 0.0

                hash_triplet.diag[block][comp * GlobalVariable.DIMENSION + comp] = 1.0
                hash_triplet.rhs[block][comp] = displacement_constraint[i].value
                diag_A[global_dof] = 1.0


@ti.kernel
def kernel_apply_dirichlet_hash_triplet_sparse(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    active_nodes: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    hash_triplet: ti.template(),
    diag_A: ti.template(),
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            block = global_dof // GlobalVariable.DIMENSION
            comp = global_dof - block * GlobalVariable.DIMENSION
            prescribed_value = displacement_constraint[i].value
            if block < active_nodes:
                for row_comp in ti.static(range(GlobalVariable.DIMENSION)):
                    h_index = row_comp * GlobalVariable.DIMENSION + comp
                    ti.atomic_add(
                        hash_triplet.rhs[block][row_comp], -hash_triplet.diag[block][h_index] * prescribed_value
                    )

                nnz = hash_triplet.non_diag.element_pair_num[0]
                for k in range(nnz):
                    bi = hash_triplet.non_diag.tripletI[k]
                    bj = hash_triplet.non_diag.tripletJ[k]
                    if 0 <= bi < active_nodes and 0 <= bj < active_nodes:
                        if bj == block:
                            for row_comp in ti.static(range(GlobalVariable.DIMENSION)):
                                h_index = row_comp * GlobalVariable.DIMENSION + comp
                                ti.atomic_add(
                                    hash_triplet.rhs[bi][row_comp],
                                    -hash_triplet.non_diag.tripletH[k][h_index] * prescribed_value,
                                )
                        if ti.static(hash_triplet.matrix_symmetric):
                            if bi == block:
                                for col_comp in ti.static(range(GlobalVariable.DIMENSION)):
                                    h_index = comp * GlobalVariable.DIMENSION + col_comp
                                    ti.atomic_add(
                                        hash_triplet.rhs[bj][col_comp],
                                        -hash_triplet.non_diag.tripletH[k][h_index] * prescribed_value,
                                    )

    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            block = global_dof // GlobalVariable.DIMENSION
            comp = global_dof - block * GlobalVariable.DIMENSION
            if block < active_nodes:
                for row_comp, col_comp in ti.static(ti.ndrange(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)):
                    if row_comp == comp or col_comp == comp:
                        hash_triplet.diag[block][row_comp * GlobalVariable.DIMENSION + col_comp] = 0.0

                nnz = hash_triplet.non_diag.element_pair_num[0]
                for k in range(nnz):
                    bi = hash_triplet.non_diag.tripletI[k]
                    bj = hash_triplet.non_diag.tripletJ[k]
                    if 0 <= bi < active_nodes and 0 <= bj < active_nodes:
                        for row_comp, col_comp in ti.static(
                            ti.ndrange(GlobalVariable.DIMENSION, GlobalVariable.DIMENSION)
                        ):
                            if (bi == block and row_comp == comp) or (bj == block and col_comp == comp):
                                hash_triplet.non_diag.tripletH[k][row_comp * GlobalVariable.DIMENSION + col_comp] = 0.0

                hash_triplet.diag[block][comp * GlobalVariable.DIMENSION + comp] = 1.0
                hash_triplet.rhs[block][comp] = displacement_constraint[i].value
                diag_A[global_dof] = 1.0


@ti.kernel
def kernel_assemble_residual_force_quasi_static_2D(
    gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), right_hand_vector: ti.template()
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                external_force = node[ng, nb].ext_force
                internal_force = node[ng, nb].int_force
                right_hand_vector[dof0] = external_force[0] + internal_force[0]
                right_hand_vector[dof0 + 1] = external_force[1] + internal_force[1]


@ti.kernel
def kernel_assemble_residual_force_quasi_static(
    gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), right_hand_vector: ti.template()
):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                external_force = node[ng, nb].ext_force
                internal_force = node[ng, nb].int_force
                right_hand_vector[dof0] = external_force[0] + internal_force[0]
                right_hand_vector[dof0 + 1] = external_force[1] + internal_force[1]
                right_hand_vector[dof0 + 2] = external_force[2] + internal_force[2]


@ti.kernel
def kernel_assemble_residual_force_dynamic_2D(
    gridSum: int,
    cutoff: float,
    beta: float,
    node: ti.template(),
    flag: ti.template(),
    right_hand_vector: ti.template(),
    dt: ti.template(),
):
    constant1 = 1.0 / dt[None] / dt[None] / beta
    constant2 = constant1 * dt[None]
    constant3 = 0.5 / beta - 1.0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                external_force = node[ng, nb].ext_force
                internal_force = node[ng, nb].int_force
                mass = node[ng, nb].m
                displacement = node[ng, nb].displacement
                acceleration = node[ng, nb].inertia
                velocity = node[ng, nb].momentum

                right_hand_vector[dof0] = (
                    external_force[0]
                    + internal_force[0]
                    - mass * (constant1 * displacement[0] - constant2 * velocity[0] - constant3 * acceleration[0])
                )
                right_hand_vector[dof0 + 1] = (
                    external_force[1]
                    + internal_force[1]
                    - mass * (constant1 * displacement[1] - constant2 * velocity[1] - constant3 * acceleration[1])
                )


@ti.kernel
def kernel_assemble_residual_force_dynamic(
    gridSum: int,
    cutoff: float,
    beta: float,
    node: ti.template(),
    flag: ti.template(),
    right_hand_vector: ti.template(),
    dt: ti.template(),
):
    constant1 = 1.0 / dt[None] / dt[None] / beta
    constant2 = constant1 * dt[None]
    constant3 = 0.5 / beta - 1.0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                external_force = node[ng, nb].ext_force
                internal_force = node[ng, nb].int_force
                mass = node[ng, nb].m
                displacement = node[ng, nb].displacement
                acceleration = node[ng, nb].inertia
                velocity = node[ng, nb].momentum

                right_hand_vector[dof0] = (
                    external_force[0]
                    + internal_force[0]
                    - mass * (constant1 * displacement[0] - constant2 * velocity[0] - constant3 * acceleration[0])
                )
                right_hand_vector[dof0 + 1] = (
                    external_force[1]
                    + internal_force[1]
                    - mass * (constant1 * displacement[1] - constant2 * velocity[1] - constant3 * acceleration[1])
                )
                right_hand_vector[dof0 + 2] = (
                    external_force[2]
                    + internal_force[2]
                    - mass * (constant1 * displacement[2] - constant2 * velocity[2] - constant3 * acceleration[2])
                )


@ti.kernel
def kernel_calculate_reaction_forces(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    accmulated_reaction_forces: ti.template(),
):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            accmulated_reaction_forces[global_dof + dofID] = (
                displacement_constraint[i].value - node[nodeID, bodyID].displacement[dofID]
            ) * PENALTY


@ti.kernel
def kernel_calculate_reaction_forces_sparse(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    accmulated_reaction_forces: ti.template(),
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            accmulated_reaction_forces[global_dof + dofID] = (
                displacement_constraint[i].value - node[nodeID, bodyID].displacement[dofID]
            ) * PENALTY


@ti.kernel
def kernel_extract_assembled_reaction_forces(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    lhs_vector: ti.template(),
    rhs_vector: ti.template(),
    accmulated_reaction_forces: ti.template(),
):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            accmulated_reaction_forces[global_dof] = lhs_vector[global_dof] - rhs_vector[global_dof]


@ti.kernel
def kernel_extract_assembled_reaction_forces_sparse(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    lhs_vector: ti.template(),
    rhs_vector: ti.template(),
    accmulated_reaction_forces: ti.template(),
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            accmulated_reaction_forces[global_dof] = lhs_vector[global_dof] - rhs_vector[global_dof]


@ti.kernel
def kernel_extract_matrix_free_reaction_forces(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    lhs_vector: ti.template(),
    rhs_vector: ti.template(),
    unknown_vector: ti.template(),
    accmulated_reaction_forces: ti.template(),
):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            accmulated_reaction_forces[global_dof] = (
                lhs_vector[global_dof] - PENALTY * unknown_vector[global_dof] - rhs_vector[global_dof]
            )


@ti.kernel
def kernel_extract_matrix_free_reaction_forces_sparse(
    gridSum: int,
    cut_off: float,
    displacementNum: int,
    displacement_constraint: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    lhs_vector: ti.template(),
    rhs_vector: ti.template(),
    unknown_vector: ti.template(),
    accmulated_reaction_forces: ti.template(),
    gnum: ti.template(),
    block_count: ti.template(),
    block_size: ti.template(),
    block_volume: ti.template(),
    block_map: ti.template(),
):
    for i in range(displacementNum):
        physical_node = displacement_constraint[i].node
        nodeID = compact_node_id(physical_node, gnum, block_count, block_size, block_volume, block_map)
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if nodeID >= 0 and node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum] + dofID
            accmulated_reaction_forces[global_dof] = (
                lhs_vector[global_dof] - PENALTY * unknown_vector[global_dof] - rhs_vector[global_dof]
            )


# ========================================================= #
#                  Convergence criterion                    #
# ========================================================= #
@ti.kernel
def compute_disp_error_2D(
    gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), disp: ti.template()
) -> float:
    delta_u = 0.0
    u = 0.0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                dux, duy = disp[dof0], disp[dof0 + 1]
                displacement = node[ng, nb].displacement
                delta_u += dux * dux + duy * duy
                u += summation(displacement)
    # The zero-load/equilibrium state has both ``delta_u == 0`` and ``u == 0``
    # and is already converged.  Returning one in that case made an exact
    # equilibrium consume the full Newton budget and then get accepted only
    # because the legacy driver did not report non-convergence.
    result = 0.0
    if delta_u > Threshold:
        result = ti.sqrt(delta_u / u) if ti.abs(u) > Threshold else 1.0
    return result


@ti.kernel
def compute_disp_error(
    gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), disp: ti.template()
) -> float:
    delta_u = 0.0
    u = 0.0
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                dux, duy, duz = disp[dof0], disp[dof0 + 1], disp[dof0 + 2]
                displacement = node[ng, nb].displacement
                delta_u += dux * dux + duy * duy + duz * duz
                u += summation(displacement)
    result = 0.0
    if delta_u > Threshold:
        result = ti.sqrt(delta_u / u) if ti.abs(u) > Threshold else 1.0
    return result


@ti.kernel
def compute_residual_error(active_id: int, right_hand_vector: ti.template()) -> float:
    rfs = 0.0
    ti.loop_config(block_dim=64)
    size = ti.ceil(active_id / 3, int)
    for i in range(size):
        thread_id = i % BLOCK_SZ
        pad_vector1 = ti.simt.block.SharedArray((64,), ti.f64)
        pad_vector2 = ti.simt.block.SharedArray((64,), ti.f64)
        pad_vector3 = ti.simt.block.SharedArray((64,), ti.f64)

        pad_vector1[thread_id] = right_hand_vector[3 * i]
        pad_vector2[thread_id] = right_hand_vector[3 * i + 1]
        pad_vector3[thread_id] = right_hand_vector[3 * i + 2]
        ti.simt.block.sync()

        temp = 0.0
        if thread_id == BLOCK_SZ - 1 or i == size - 1:
            for k in range(thread_id + 1):
                temp += (
                    pad_vector1[k] * pad_vector1[k] + pad_vector2[k] * pad_vector2[k] + pad_vector3[k] * pad_vector3[k]
                )
        rfs += temp
    return ti.sqrt(rfs)


# ========================================================= #
#                   FDM Poisson equation                    #
# ========================================================= #
@ti.func
def get_node_id(index, gnum):
    return (linearize(index, gnum), 0)


@ti.func
def select_cut_cell_face_fraction(
    direction: ti.template(),
    face,
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
):
    fraction = 1.0
    if ti.static(direction == 0):
        fraction = face_fraction0[face]
    elif ti.static(direction == 1):
        fraction = face_fraction1[face]
    elif ti.static(GlobalVariable.DIMENSION == 3 and direction == 2):
        fraction = face_fraction2[face]
    return fraction


@ti.func
def select_cut_cell_solid_velocity(
    direction: ti.template(),
    face,
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
):
    velocity = 0.0
    if ti.static(direction == 0):
        velocity = solid_velocity0[face]
    elif ti.static(direction == 1):
        velocity = solid_velocity1[face]
    elif ti.static(GlobalVariable.DIMENSION == 3 and direction == 2):
        velocity = solid_velocity2[face]
    return velocity


@ti.func
def coupled_projection_cell_fraction_3d(cell, mode: ti.template(), solid_fraction: ti.template()):
    fraction = 1.0
    if ti.static(mode == 1):
        fraction = ti.max(0.05, 1.0 - ti.min(1.0, ti.max(0.0, solid_fraction[cell])))
    return fraction


@ti.func
def coupled_projection_cell_density_3d(
    cell, rho_f, mode: ti.template(), solid_fraction: ti.template(), solid_density: ti.template()
):
    density = rho_f
    if ti.static(mode == 2):
        phi_s = ti.min(1.0, ti.max(0.0, solid_fraction[cell]))
        rho_s = solid_density[cell]
        if rho_s <= Threshold:
            rho_s = rho_f
        density = (1.0 - phi_s) * rho_f + phi_s * rho_s
    return density


@ti.func
def coupled_projection_face_terms_3d(
    cell,
    neighbor,
    rho_f,
    mode: ti.template(),
    cell_type: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
):
    flux_fraction = coupled_projection_cell_fraction_3d(cell, mode, solid_fraction)
    density = coupled_projection_cell_density_3d(cell, rho_f, mode, solid_fraction, solid_density)
    if int(cell_type[neighbor]) == 1:
        flux_fraction = 0.5 * (flux_fraction + coupled_projection_cell_fraction_3d(neighbor, mode, solid_fraction))
        density = 0.5 * (
            density + coupled_projection_cell_density_3d(neighbor, rho_f, mode, solid_fraction, solid_density)
        )
    pressure_coefficient = flux_fraction / ti.max(density, Threshold)
    return flux_fraction, pressure_coefficient


@ti.func
def coupled_projection_open_fraction_3d(
    direction: ti.template(),
    face,
    use_cut_cell: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
):
    fraction = 1.0
    if ti.static(use_cut_cell):
        fraction = select_cut_cell_face_fraction(direction, face, face_fraction0, face_fraction1, face_fraction2)
    return fraction


@ti.kernel
def kernel_assemble_poisson_equation_coupled_3d(
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    igrid_size: ti.types.vector(3, float),
    dt: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    solid_fraction: ti.template(),
    previous_solid_fraction: ti.template(),
    solid_density: ti.template(),
    matProps: ti.template(),
    mode: ti.template(),
    use_cut_cell: ti.template(),
    use_free_surface_theta: ti.template(),
    right_hand_vector: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    inv_dt = 1.0 / ti.max(dt[None], Threshold)
    for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
        if int(cell_type[I]) == 1:
            cell_id = linearize(I, active_cnum)
            dof_index = flag[cell_id]
            rhs = 0.0
            for d in ti.static(range(3)):
                unit = ti.Vector.unit(3, d)
                left_neighbor = I - unit
                right_neighbor = I + unit
                left_fraction, _ = coupled_projection_face_terms_3d(
                    I, left_neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                )
                right_fraction, _ = coupled_projection_face_terms_3d(
                    I, right_neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                )
                left_open = coupled_projection_open_fraction_3d(
                    d, I, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                )
                right_open = coupled_projection_open_fraction_3d(
                    d, I + unit, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                )
                left_velocity = node.velocity[d][I]
                right_velocity = node.velocity[d][I + unit]
                if ti.static(use_cut_cell):
                    left_velocity = left_open * left_velocity + (1.0 - left_open) * select_cut_cell_solid_velocity(
                        d, I, solid_velocity0, solid_velocity1, solid_velocity2
                    )
                    right_velocity = right_open * right_velocity + (1.0 - right_open) * select_cut_cell_solid_velocity(
                        d, I + unit, solid_velocity0, solid_velocity1, solid_velocity2
                    )
                else:
                    if int(cell_type[left_neighbor]) == 2:
                        left_velocity = 0.0
                    if int(cell_type[right_neighbor]) == 2:
                        right_velocity = 0.0
                rhs += (left_fraction * left_velocity - right_fraction * right_velocity) * igrid_size[d] * inv_dt

            if ti.static(mode == 1):
                current_fraction = coupled_projection_cell_fraction_3d(I, mode, solid_fraction)
                previous_fraction = coupled_projection_cell_fraction_3d(I, mode, previous_solid_fraction)
                rhs += (previous_fraction - current_fraction) * inv_dt * inv_dt

            for d in ti.static(range(3)):
                for side in ti.static((-1, 1)):
                    offset = side * ti.Vector.unit(3, d)
                    neighbor = I + offset
                    face = I
                    if ti.static(side > 0):
                        face = I + offset
                    face_open = coupled_projection_open_fraction_3d(
                        d, face, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                    )
                    _, coefficient = coupled_projection_face_terms_3d(
                        I, neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                    )
                    if int(cell_type[neighbor]) == 0 and face_open > 0.0:
                        theta = 1.0
                        if ti.static(use_free_surface_theta):
                            theta = pressure_free_surface_theta(I, neighbor, fluid_sdf)
                        rhs += (
                            face_open
                            * coefficient
                            * igrid_size[d]
                            * igrid_size[d]
                            / theta
                            * (matProps.atmospheric_pressure + surface_tension[neighbor])
                        )
            right_hand_vector[dof_index] = rhs


@ti.kernel
def kernel_preconditioning_poisson_equation_coupled_3d(
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    igrid_size: ti.types.vector(3, float),
    flag: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
    matProps: ti.template(),
    mode: ti.template(),
    use_cut_cell: ti.template(),
    use_free_surface_theta: ti.template(),
    diag_A: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
        if int(cell_type[I]) == 1:
            dof_index = flag[linearize(I, active_cnum)]
            if dof_index >= 0:
                center_coefficient = 0.0
                for d in ti.static(range(3)):
                    for side in ti.static((-1, 1)):
                        offset = side * ti.Vector.unit(3, d)
                        neighbor = I + offset
                        face = I
                        if ti.static(side > 0):
                            face = I + offset
                        face_open = coupled_projection_open_fraction_3d(
                            d, face, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                        )
                        _, coefficient = coupled_projection_face_terms_3d(
                            I, neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                        )
                        if int(cell_type[neighbor]) == 0:
                            theta = 1.0
                            if ti.static(use_free_surface_theta):
                                theta = pressure_free_surface_theta(I, neighbor, fluid_sdf)
                            center_coefficient += face_open * coefficient * igrid_size[d] * igrid_size[d] / theta
                        elif int(cell_type[neighbor]) == 1:
                            center_coefficient += face_open * coefficient * igrid_size[d] * igrid_size[d]
                if center_coefficient <= 0.0:
                    center_coefficient = 1.0
                diag_A[dof_index] = center_coefficient


@ti.kernel
def kernel_poisson_equation_cg_coupled_3d(
    ghost_cell: int,
    cnum: ti.types.vector(3, int),
    igrid_size: ti.types.vector(3, float),
    flag: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_fraction: ti.template(),
    solid_density: ti.template(),
    matProps: ti.template(),
    mode: ti.template(),
    use_cut_cell: ti.template(),
    use_free_surface_theta: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    active_cnum = cnum - 2 * ghost_cell
    for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
        if int(cell_type[I]) == 1:
            dof_index = flag[linearize(I, active_cnum)]
            if dof_index >= 0:
                center_coefficient = 0.0
                neighbor_value = 0.0
                for d in ti.static(range(3)):
                    for side in ti.static((-1, 1)):
                        offset = side * ti.Vector.unit(3, d)
                        neighbor = I + offset
                        face = I
                        if ti.static(side > 0):
                            face = I + offset
                        face_open = coupled_projection_open_fraction_3d(
                            d, face, use_cut_cell, face_fraction0, face_fraction1, face_fraction2
                        )
                        _, coefficient = coupled_projection_face_terms_3d(
                            I, neighbor, matProps.density, mode, cell_type, solid_fraction, solid_density
                        )
                        coefficient *= face_open * igrid_size[d] * igrid_size[d]
                        if int(cell_type[neighbor]) == 0:
                            theta = 1.0
                            if ti.static(use_free_surface_theta):
                                theta = pressure_free_surface_theta(I, neighbor, fluid_sdf)
                            center_coefficient += coefficient / theta
                        elif int(cell_type[neighbor]) == 1:
                            neighbor_dof = flag[linearize(neighbor, active_cnum)]
                            center_coefficient += coefficient
                            if neighbor_dof >= 0:
                                neighbor_value -= coefficient * unknown_vector[neighbor_dof]
                if center_coefficient <= 0.0:
                    center_coefficient = 1.0
                m_dot_v[dof_index] = neighbor_value + center_coefficient * unknown_vector[dof_index]


# =========================== Matrix free conjuction gradient method ============================== #
@ti.kernel
def kernel_assemble_poisson_equation_dynamic(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    matProps: ti.template(),
    use_free_surface_theta: ti.template(),
    right_hand_vector: ti.template(),
):
    parameter1 = matProps.density / dt[None] * igrid_size
    parameter2 = 1.0 * igrid_size * igrid_size
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                rhs = 0.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                    rhs += (node.velocity[d][I] - node.velocity[d][I + offset]) * parameter1[d]
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                        neighbor = I + offset
                        if cell_type[neighbor] == 2:
                            if s < 0:
                                rhs -= parameter1[d] * node.velocity[d][I]
                            else:
                                rhs += parameter1[d] * node.velocity[d][I + offset]
                        elif cell_type[neighbor] == 0:
                            theta = 1.0
                            if ti.static(use_free_surface_theta):
                                theta = pressure_free_surface_theta(I, neighbor, fluid_sdf)
                            rhs += parameter2[d] / theta * (matProps.atmospheric_pressure + surface_tension[neighbor])
                right_hand_vector[dof_index] = rhs
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                rhs = 0.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                    rhs += (node.velocity[d][I] - node.velocity[d][I + offset]) * parameter1[d]
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                        neighbor = I + offset
                        if cell_type[neighbor] == 2:
                            if s < 0:
                                rhs -= parameter1[d] * node.velocity[d][I]
                            else:
                                rhs += parameter1[d] * node.velocity[d][I + offset]
                        elif cell_type[neighbor] == 0:
                            theta = 1.0
                            if ti.static(use_free_surface_theta):
                                theta = pressure_free_surface_theta(I, neighbor, fluid_sdf)
                            rhs += parameter2[d] / theta * (matProps.atmospheric_pressure + surface_tension[neighbor])
                right_hand_vector[dof_index] = rhs


@ti.kernel
def kernel_preconditioning_poisson_equation_matrix(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    flag: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
    diag_A: ti.template(),
):
    parameter = 1.0 * igrid_size * igrid_size
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            neighbor_id = I + ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            if cell_type[neighbor_id] == 0:
                                theta = 1.0
                                if ti.static(use_free_surface_theta):
                                    theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                center_coeff += parameter[d] / theta
                            elif cell_type[neighbor_id] == 1:
                                center_coeff += parameter[d]
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    diag_A[dof_index] = center_coeff
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            neighbor_id = I + ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            if cell_type[neighbor_id] == 0:
                                theta = 1.0
                                if ti.static(use_free_surface_theta):
                                    theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                center_coeff += parameter[d] / theta
                            elif cell_type[neighbor_id] == 1:
                                center_coeff += parameter[d]
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    diag_A[dof_index] = center_coeff


@ti.kernel
def kernel_poisson_equation_cg(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    flag: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    use_free_surface_theta: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    parameter = 1.0 * igrid_size * igrid_size
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    neighbor_value = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            neighbor_id = I + ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            if cell_type[neighbor_id] == 0:
                                theta = 1.0
                                if ti.static(use_free_surface_theta):
                                    theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                center_coeff += parameter[d] / theta
                            elif cell_type[neighbor_id] == 1:
                                linear_neighbor_id = linearize(neighbor_id, active_cnum)
                                neighbor_dof_index = flag[linear_neighbor_id]
                                center_coeff += parameter[d]
                                if neighbor_dof_index >= 0:
                                    neighbor_value -= parameter[d] * unknown_vector[neighbor_dof_index]
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    m_dot_v[dof_index] = neighbor_value + center_coeff * unknown_vector[dof_index]
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    neighbor_value = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            neighbor_id = I + ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            if cell_type[neighbor_id] == 0:
                                theta = 1.0
                                if ti.static(use_free_surface_theta):
                                    theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                center_coeff += parameter[d] / theta
                            elif cell_type[neighbor_id] == 1:
                                linear_neighbor_id = linearize(neighbor_id, active_cnum)
                                neighbor_dof_index = flag[linear_neighbor_id]
                                center_coeff += parameter[d]
                                if neighbor_dof_index >= 0:
                                    neighbor_value -= parameter[d] * unknown_vector[neighbor_dof_index]
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    m_dot_v[dof_index] = neighbor_value + center_coeff * unknown_vector[dof_index]


@ti.kernel
def kernel_assemble_poisson_equation_dynamic_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    gnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    dt: ti.template(),
    node: ti.template(),
    flag: ti.template(),
    surface_tension: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    solid_velocity0: ti.template(),
    solid_velocity1: ti.template(),
    solid_velocity2: ti.template(),
    matProps: ti.template(),
    use_free_surface_theta: ti.template(),
    right_hand_vector: ti.template(),
):
    parameter1 = matProps.density / dt[None] * igrid_size
    parameter2 = 1.0 * igrid_size * igrid_size
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                rhs = 0.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                    left_face = I
                    right_face = I + offset
                    left_fraction = select_cut_cell_face_fraction(
                        d, left_face, face_fraction0, face_fraction1, face_fraction2
                    )
                    right_fraction = select_cut_cell_face_fraction(
                        d, right_face, face_fraction0, face_fraction1, face_fraction2
                    )
                    left_velocity = left_fraction * node.velocity[d][left_face] + (
                        1.0 - left_fraction
                    ) * select_cut_cell_solid_velocity(d, left_face, solid_velocity0, solid_velocity1, solid_velocity2)
                    right_velocity = right_fraction * node.velocity[d][right_face] + (
                        1.0 - right_fraction
                    ) * select_cut_cell_solid_velocity(d, right_face, solid_velocity0, solid_velocity1, solid_velocity2)
                    rhs += (left_velocity - right_velocity) * parameter1[d]
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                        neighbor = I + offset
                        face = I
                        if ti.static(s > 0):
                            face = I + offset
                        face_fraction = select_cut_cell_face_fraction(
                            d, face, face_fraction0, face_fraction1, face_fraction2
                        )
                        if cell_type[neighbor] == 0 and solid_sdf[neighbor] >= 0.0 and face_fraction > 0.0:
                            theta = 1.0
                            if ti.static(use_free_surface_theta):
                                theta = pressure_free_surface_theta(I, neighbor, fluid_sdf)
                            rhs += (
                                face_fraction
                                * parameter2[d]
                                / theta
                                * (matProps.atmospheric_pressure + surface_tension[neighbor])
                            )
                right_hand_vector[dof_index] = rhs
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                rhs = 0.0
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    offset = ti.Vector.unit(GlobalVariable.DIMENSION, d)
                    left_face = I
                    right_face = I + offset
                    left_fraction = select_cut_cell_face_fraction(
                        d, left_face, face_fraction0, face_fraction1, face_fraction2
                    )
                    right_fraction = select_cut_cell_face_fraction(
                        d, right_face, face_fraction0, face_fraction1, face_fraction2
                    )
                    left_velocity = left_fraction * node.velocity[d][left_face] + (
                        1.0 - left_fraction
                    ) * select_cut_cell_solid_velocity(d, left_face, solid_velocity0, solid_velocity1, solid_velocity2)
                    right_velocity = right_fraction * node.velocity[d][right_face] + (
                        1.0 - right_fraction
                    ) * select_cut_cell_solid_velocity(d, right_face, solid_velocity0, solid_velocity1, solid_velocity2)
                    rhs += (left_velocity - right_velocity) * parameter1[d]
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    for s in ti.static((-1, 1)):
                        offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * s
                        neighbor = I + offset
                        face = I
                        if ti.static(s > 0):
                            face = I + offset
                        face_fraction = select_cut_cell_face_fraction(
                            d, face, face_fraction0, face_fraction1, face_fraction2
                        )
                        if cell_type[neighbor] == 0 and solid_sdf[neighbor] >= 0.0 and face_fraction > 0.0:
                            theta = 1.0
                            if ti.static(use_free_surface_theta):
                                theta = pressure_free_surface_theta(I, neighbor, fluid_sdf)
                            rhs += (
                                face_fraction
                                * parameter2[d]
                                / theta
                                * (matProps.atmospheric_pressure + surface_tension[neighbor])
                            )
                right_hand_vector[dof_index] = rhs


@ti.kernel
def kernel_preconditioning_poisson_equation_matrix_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    flag: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    use_free_surface_theta: ti.template(),
    diag_A: ti.template(),
):
    parameter = 1.0 * igrid_size * igrid_size
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            neighbor_id = I + offset
                            face = I
                            if ti.static(dirs > 0):
                                face = I + offset
                            face_fraction = select_cut_cell_face_fraction(
                                d, face, face_fraction0, face_fraction1, face_fraction2
                            )
                            if face_fraction > 0.0:
                                if cell_type[neighbor_id] == 0 and solid_sdf[neighbor_id] >= 0.0:
                                    theta = 1.0
                                    if ti.static(use_free_surface_theta):
                                        theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                    center_coeff += face_fraction * parameter[d] / theta
                                elif cell_type[neighbor_id] == 1:
                                    center_coeff += face_fraction * parameter[d]
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    diag_A[dof_index] = center_coeff
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            neighbor_id = I + offset
                            face = I
                            if ti.static(dirs > 0):
                                face = I + offset
                            face_fraction = select_cut_cell_face_fraction(
                                d, face, face_fraction0, face_fraction1, face_fraction2
                            )
                            if face_fraction > 0.0:
                                if cell_type[neighbor_id] == 0 and solid_sdf[neighbor_id] >= 0.0:
                                    theta = 1.0
                                    if ti.static(use_free_surface_theta):
                                        theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                    center_coeff += face_fraction * parameter[d] / theta
                                elif cell_type[neighbor_id] == 1:
                                    center_coeff += face_fraction * parameter[d]
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    diag_A[dof_index] = center_coeff


@ti.kernel
def kernel_poisson_equation_cg_cut_cell(
    ghost_cell: int,
    cnum: ti.types.vector(GlobalVariable.DIMENSION, int),
    grid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    igrid_size: ti.types.vector(GlobalVariable.DIMENSION, float),
    flag: ti.template(),
    cell_type: ti.template(),
    fluid_sdf: ti.template(),
    solid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
    face_fraction2: ti.template(),
    use_free_surface_theta: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    parameter = 1.0 * igrid_size * igrid_size
    active_cnum = cnum - 2 * ghost_cell
    if ti.static(GlobalVariable.DIMENSION == 2):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    neighbor_value = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            neighbor_id = I + offset
                            face = I
                            if ti.static(dirs > 0):
                                face = I + offset
                            face_fraction = select_cut_cell_face_fraction(
                                d, face, face_fraction0, face_fraction1, face_fraction2
                            )
                            if face_fraction > 0.0:
                                if cell_type[neighbor_id] == 0 and solid_sdf[neighbor_id] >= 0.0:
                                    theta = 1.0
                                    if ti.static(use_free_surface_theta):
                                        theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                    center_coeff += face_fraction * parameter[d] / theta
                                elif cell_type[neighbor_id] == 1:
                                    linear_neighbor_id = linearize(neighbor_id, active_cnum)
                                    neighbor_dof_index = flag[linear_neighbor_id]
                                    center_coeff += face_fraction * parameter[d]
                                    if neighbor_dof_index >= 0:
                                        neighbor_value -= (
                                            face_fraction * parameter[d] * unknown_vector[neighbor_dof_index]
                                        )
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    m_dot_v[dof_index] = neighbor_value + center_coeff * unknown_vector[dof_index]
    elif ti.static(GlobalVariable.DIMENSION == 3):
        for I in ti.grouped(ti.ndrange((0, active_cnum[0]), (0, active_cnum[1]), (0, active_cnum[2]))):
            if cell_type[I] == 1:
                cell_id = linearize(I, active_cnum)
                dof_index = flag[cell_id]
                if dof_index >= 0:
                    center_coeff = 0.0
                    neighbor_value = 0.0
                    for d in ti.static(range(GlobalVariable.DIMENSION)):
                        for dirs in ti.static((-1, 1)):
                            offset = ti.Vector.unit(GlobalVariable.DIMENSION, d) * dirs
                            neighbor_id = I + offset
                            face = I
                            if ti.static(dirs > 0):
                                face = I + offset
                            face_fraction = select_cut_cell_face_fraction(
                                d, face, face_fraction0, face_fraction1, face_fraction2
                            )
                            if face_fraction > 0.0:
                                if cell_type[neighbor_id] == 0 and solid_sdf[neighbor_id] >= 0.0:
                                    theta = 1.0
                                    if ti.static(use_free_surface_theta):
                                        theta = pressure_free_surface_theta(I, neighbor_id, fluid_sdf)
                                    center_coeff += face_fraction * parameter[d] / theta
                                elif cell_type[neighbor_id] == 1:
                                    linear_neighbor_id = linearize(neighbor_id, active_cnum)
                                    neighbor_dof_index = flag[linear_neighbor_id]
                                    center_coeff += face_fraction * parameter[d]
                                    if neighbor_dof_index >= 0:
                                        neighbor_value -= (
                                            face_fraction * parameter[d] * unknown_vector[neighbor_dof_index]
                                        )
                    if center_coeff <= 0.0:
                        center_coeff = 1.0
                    m_dot_v[dof_index] = neighbor_value + center_coeff * unknown_vector[dof_index]


@ti.kernel
def kernel_assemble_residual_poisson_2D(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    dshapefn: ti.template(),
    shapefn: ti.template(),
    right_hand_vector: ti.template(),
):
    right_hand_vector.fill(0)
    ti.sync()
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            for lni in range(offset, offset + int(node_size[np])):
                nodeIDi = LnID[lni]
                ishape = shapefn[lni]
                idshape = dshapefn[lni]
                dofsi = node[nodeIDi, bodyID].dof
                if dofsi >= 0:
                    B1, B2 = 0.0, 0.0
                    for lnj in range(offset, offset + int(node_size[np])):
                        nodeIDj = LnID[lnj]
                        jshape = shapefn[lnj]
                        jdshape = dshapefn[lnj]
                        # The material boundary moves with the skeleton.
                        # Keep its divergence in strong form: dropping the
                        # surface term invents flux even for rigid translation.
                        B1 -= (
                            ishape * jdshape[0] * volume * node[nodeIDj, bodyID].momentums[0]
                            + ishape * jdshape[1] * volume * node[nodeIDj, bodyID].momentums[1]
                        )
                        # Differentiate the interpolated relative flux n*(vf-vs),
                        # retaining porosity gradients without a missing surface term.
                        B2 -= ishape * jdshape[0] * volume * node[nodeIDj, bodyID].porosity * (
                            node[nodeIDj, bodyID].momentumf[0] - node[nodeIDj, bodyID].momentums[0]
                        ) + ishape * jdshape[1] * volume * node[nodeIDj, bodyID].porosity * (
                            node[nodeIDj, bodyID].momentumf[1] - node[nodeIDj, bodyID].momentums[1]
                        )
                    right_hand_vector[dofsi] += B1 + B2


@ti.kernel
def kernel_assemble_residual_poisson_2DAxisy(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    dshapefn: ti.template(),
    shapefn: ti.template(),
    right_hand_vector: ti.template(),
):
    right_hand_vector.fill(0)
    ti.sync()
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        position = particle[np].x
        materialID = int(particle[np].materialID)
        if materialID > 0 and int(particle[np].active) == 1:
            for lni in range(offset, offset + int(node_size[np])):
                nodeIDi = LnID[lni]
                ishape = shapefn[lni]
                idshape = dshapefn[lni]
                dofsi = node[nodeIDi, bodyID].dof
                if dofsi >= 0:
                    B1, B2 = 0.0, 0.0
                    for lnj in range(offset, offset + int(node_size[np])):
                        nodeIDj = LnID[lnj]
                        jshape = shapefn[lnj]
                        jdshape = dshapefn[lnj]
                        B1 -= (
                            ishape * jdshape[0] * volume * node[nodeIDj, bodyID].momentums[0]
                            + ishape * jdshape[1] * volume * node[nodeIDj, bodyID].momentums[1]
                            + ishape * jshape / position[0] * volume * node[nodeIDj, bodyID].momentums[0]
                        )
                        B2 -= (
                            ishape
                            * jdshape[0]
                            * volume
                            * porosity
                            * (node[nodeIDj, bodyID].momentumf[0] - node[nodeIDj, bodyID].momentums[0])
                            + ishape
                            * jdshape[1]
                            * volume
                            * porosity
                            * (node[nodeIDj, bodyID].momentumf[1] - node[nodeIDj, bodyID].momentums[1])
                            + ishape
                            * jshape
                            / position[0]
                            * volume
                            * porosity
                            * (node[nodeIDj, bodyID].momentumf[0] - node[nodeIDj, bodyID].momentums[0])
                        )
                    right_hand_vector[dofsi] += B1 + B2


@ti.func
def single_point_fic_time_scale(grid_size, young, poisson, solid_density):
    shear = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    return 0.1 * grid_size[0] / ti.sqrt((bulk + 4.0 * shear / 3.0) / solid_density)


@ti.kernel
def kernel_assemble_residual_poisson_FIC_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    particle: ti.template(),
    material_mapping: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    matProp: ti.template(),
    dshapefn: ti.template(),
    shapefn: ti.template(),
    right_hand_vector: ti.template(),
    dt: ti.template(),
    grid_size: ti.types.vector(2, float),
    beta: float,
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            density_s, density_f = matProp.solid_density, matProp.fluid_density
            tau_ = single_point_fic_time_scale(grid_size, matProp.young, matProp.poisson, density_s)
            H = volume * tau_ * (porosity / density_f + (1.0 - porosity) / density_s)
            for lni in range(offset, offset + int(node_size[np])):
                nodeIDi = LnID[lni]
                local_lni = lni - offset
                ishape = shapefn[lni]
                idshape = dshapefn[lni]
                dofsi = node[nodeIDi, bodyID].dof
                if dofsi >= 0:
                    B1, B2, B3, B4 = 0.0, 0.0, 0.0, 0.0
                    for lnj in range(offset, offset + int(node_size[np])):
                        nodeIDj = LnID[lnj]
                        jshape = shapefn[lnj]
                        jdshape = dshapefn[lnj]
                        B1 -= (
                            ishape * jdshape[0] * volume * node[nodeIDj, bodyID].momentums[0]
                            + ishape * jdshape[1] * volume * node[nodeIDj, bodyID].momentums[1]
                        )
                        B2 -= ishape * jdshape[0] * volume * node[nodeIDj, bodyID].porosity * (
                            node[nodeIDj, bodyID].momentumf[0] - node[nodeIDj, bodyID].momentums[0]
                        ) + ishape * jdshape[1] * volume * node[nodeIDj, bodyID].porosity * (
                            node[nodeIDj, bodyID].momentumf[1] - node[nodeIDj, bodyID].momentums[1]
                        )
                        B3 -= (
                            idshape[0] * jdshape[0] * H * beta * node[nodeIDj, bodyID].pressure
                            + idshape[1] * jdshape[1] * H * beta * node[nodeIDj, bodyID].pressure
                        )
                        B4 -= (
                            idshape[0] * jshape * volume * tau_ * node[nodeIDj, bodyID].extra_stabilize[0]
                            + idshape[1] * jshape * volume * tau_ * node[nodeIDj, bodyID].extra_stabilize[1]
                        )
                    right_hand_vector[dofsi] += B1 + B2 + B3 + B4


@ti.kernel
def kernel_assemble_residual_poisson_2D_u_p(
    total_nodes: int,
    start_index: int,
    end_index: int,
    particle: ti.template(),
    material_mapping: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    matProp: ti.template(),
    gravity: ti.types.vector(3, float),
    dshapefn: ti.template(),
    shapefn: ti.template(),
    right_hand_vector: ti.template(),
    dt: ti.template(),
    beta: float,
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            density_s, density_f = matProp.solid_density, matProp.fluid_density
            kk = matProp.permeability
            H = volume * kk / matProp.fluid_unit_weight
            storage = volume * porosity / matProp.fluid_bulk / dt[None]
            for lni in range(offset, offset + int(node_size[np])):
                nodeIDi = LnID[lni]
                local_lni = lni - offset
                ishape = shapefn[lni]
                idshape = dshapefn[lni]
                dofsi = node[nodeIDi, bodyID].dof
                if dofsi >= 0:
                    B1, B2 = 0.0, storage * ishape * particle[np].pressure
                    for lnj in range(offset, offset + int(node_size[np])):
                        nodeIDj = LnID[lnj]
                        jshape = shapefn[lnj]
                        jdshape = dshapefn[lnj]
                        B1 -= (
                            ishape * jdshape[0] * volume * node[nodeIDj, bodyID].momentum[0]
                            + ishape * jdshape[1] * volume * node[nodeIDj, bodyID].momentum[1]
                        )
                        B2 -= idshape[0] * H * (
                            beta * jdshape[0] * node[nodeIDj, bodyID].pressure - jshape * density_f * gravity[0]
                        ) + idshape[1] * H * (
                            beta * jdshape[1] * node[nodeIDj, bodyID].pressure - jshape * density_f * gravity[1]
                        )
                        # p_new = beta*N*p_old_grid + N*dp. The physical old
                        # pressure is carried by the particle, not N*p_old_grid.
                        B2 -= storage * ishape * jshape * beta * node[nodeIDj, bodyID].pressure
                    right_hand_vector[dofsi] += B1 + B2


@ti.kernel
def kernel_compute_penalty_matrix_poisson(
    cut_off: float,
    pressureNum: int,
    pressure_constraint: ti.template(),
    node: ti.template(),
    mass_matrix: ti.template(),
):
    for i in range(pressureNum):
        nodeID = pressure_constraint[i]
        bodyID = 0
        if node[nodeID, bodyID].m > cut_off:
            dofID = node[nodeID, bodyID].dof
            mass_matrix[dofID] += PENALTY


@ti.kernel
def kernel_preconditioning_matrix_poisson2D(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    diag_A: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    local_stiffness: ti.template(),
):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            local_ln = ln - offset
            dofs = node[nodeID, bodyID].dof
            if dofs >= 0:
                diag_A[dofs] += local_stiffness[np, local_ln, local_ln]


@ti.kernel
def kernel_assemble_local_stiffness_poisson_FIC_2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    particle: ti.template(),
    material_mapping: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    matProp: ti.template(),
    local_stiffness: ti.template(),
    dt: ti.template(),
    grid_size: ti.types.vector(2, float),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            density_s, density_f = matProp.solid_density, matProp.fluid_density
            H = volume * (porosity / density_f + (1.0 - porosity) / density_s) * dt[None]

            tau_ = single_point_fic_time_scale(grid_size, matProp.young, matProp.poisson, density_s)
            H_ = volume * (porosity / density_f + (1.0 - porosity) / density_s) * tau_

            for lni in range(offset, offset + int(node_size[np])):
                ishape = shapefn[lni]
                idshape = dshapefn[lni]
                local_lni = lni - offset
                for lnj in range(offset, offset + int(node_size[np])):
                    jshape = shapefn[lnj]
                    jdshape = dshapefn[lnj]
                    local_lnj = lnj - offset
                    local_stiffness[np, local_lni, local_lnj] = (idshape[0] * jdshape[0] + idshape[1] * jdshape[1]) * (
                        H + H_
                    )


@ti.kernel
def kernel_assemble_local_stiffness_poisson2D(
    total_nodes: int,
    start_index: int,
    end_index: int,
    particle: ti.template(),
    material_mapping: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    matProp: ti.template(),
    local_stiffness: ti.template(),
    dt: ti.template(),
    grid_size: ti.types.vector(2, float),
    stateVars: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            density_s, density_f = matProp.solid_density, matProp.fluid_density
            L = volume * (porosity / density_f + (1.0 - porosity) / density_s) * dt[None]

            for lni in range(offset, offset + int(node_size[np])):
                idshape = dshapefn[lni]
                local_lni = lni - offset
                for lnj in range(offset, offset + int(node_size[np])):
                    jdshape = dshapefn[lnj]
                    local_lnj = lnj - offset
                    local_stiffness[np, local_lni, local_lnj] = (idshape[0] * jdshape[0] + idshape[1] * jdshape[1]) * L


@ti.kernel
def kernel_assemble_local_stiffness_poisson2D_u_p(
    total_nodes: int,
    start_index: int,
    end_index: int,
    particle: ti.template(),
    material_mapping: ti.template(),
    shapefn: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    matProp: ti.template(),
    local_stiffness: ti.template(),
    dt: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            density_s, density_f = matProp.solid_density, matProp.fluid_density
            kk = matProp.permeability
            bulk = matProp.fluid_bulk
            rho = porosity * density_f + (1.0 - porosity) * density_s
            S = volume * porosity / bulk / dt[None]
            H = volume * kk / matProp.fluid_unit_weight
            H_ = volume / rho * dt[None]
            for lni in range(offset, offset + int(node_size[np])):
                ishape = shapefn[lni]
                idshape = dshapefn[lni]
                local_lni = lni - offset
                for lnj in range(offset, offset + int(node_size[np])):
                    jshape = shapefn[lnj]
                    jdshape = dshapefn[lnj]
                    local_lnj = lnj - offset
                    local_stiffness[np, local_lni, local_lnj] = (idshape[0] * jdshape[0] + idshape[1] * jdshape[1]) * (
                        H + H_
                    ) + ishape * jshape * S


@ti.kernel
def kernel_assemble_stiffness_poisson2D(
    ifnode: int,
    total_nodes: int,
    start_index: int,
    end_index: int,
    particle: ti.template(),
    material_mapping: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    matProp: ti.template(),
    sparse_matrix: ti.template(),
    dt: ti.template(),
    grid_size: ti.types.vector(2, float),
    stateVars: ti.template(),
    cutoff: float,
    beta: float,
    boundary_pressure: float,
    right_hand_vector: ti.template(),
    fic: ti.template(),
):
    for i in range(start_index, end_index):
        np = material_mapping[i]
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        volume = particle[np].vol
        porosity = particle[np].porosity
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            density_s, density_f = matProp.solid_density, matProp.fluid_density
            L = volume * (porosity / density_f + (1.0 - porosity) / density_s) * dt[None]
            if ti.static(fic):
                L += (
                    volume
                    * (porosity / density_f + (1.0 - porosity) / density_s)
                    * single_point_fic_time_scale(grid_size, matProp.young, matProp.poisson, density_s)
                )

            for lni in range(offset, offset + int(node_size[np])):
                idshape = dshapefn[lni]
                local_lni = lni - offset
                nodeIDi = LnID[lni]
                dofsi = node[nodeIDi, bodyID].dof
                if dofsi >= 0:
                    for lnj in range(offset, offset + int(node_size[np])):
                        jdshape = dshapefn[lnj]
                        local_lnj = lnj - offset
                        nodeIDj = LnID[lnj]
                        dofsj = node[nodeIDj, bodyID].dof
                        if dofsj >= 0:
                            index = ifnode * ifnode * np + local_lni * ifnode + local_lnj
                            sparse_matrix.rows[index] = dofsi
                            sparse_matrix.cols[index] = dofsj
                            sparse_matrix.data[index] = (idshape[0] * jdshape[0] + idshape[1] * jdshape[1]) * L
                        elif node[nodeIDj, bodyID].m > cutoff:
                            prescribed_increment = boundary_pressure - beta * node[nodeIDj, bodyID].pressure
                            right_hand_vector[dofsi] -= (
                                (idshape[0] * jdshape[0] + idshape[1] * jdshape[1]) * L * prescribed_increment
                            )


@ti.kernel
def kernel_eliminate_pressure_increment_dirichlet(
    total_nodes: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    local_stiffness: ti.template(),
    cutoff: float,
    beta: float,
    boundary_pressure: float,
    right_hand_vector: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].materialID) > 0 and int(particle[np].active) == 1:
            bodyID = int(particle[np].bodyID)
            offset = np * total_nodes
            for lni in range(offset, offset + int(node_size[np])):
                nodeIDi = LnID[lni]
                dofsi = node[nodeIDi, bodyID].dof
                if dofsi >= 0:
                    local_lni = lni - offset
                    for lnj in range(offset, offset + int(node_size[np])):
                        nodeIDj = LnID[lnj]
                        if node[nodeIDj, bodyID].dof < 0 and node[nodeIDj, bodyID].m > cutoff:
                            prescribed_increment = boundary_pressure - beta * node[nodeIDj, bodyID].pressure
                            right_hand_vector[dofsi] -= (
                                local_stiffness[np, local_lni, lnj - offset] * prescribed_increment
                            )


# ---- SemiImplicit TwoPhaseSingleLayer compatibility kernels ----
@ti.kernel
def kernel_mass_balance_cg_poisson2D(
    total_nodes: int,
    total_dofs: int,
    particleNum: int,
    particle: ti.template(),
    node_size: ti.template(),
    LnID: ti.template(),
    node: ti.template(),
    mass_matrix: ti.template(),
    local_stiffness: ti.template(),
    unknown_vector: ti.template(),
    m_dot_v: ti.template(),
):
    # print(total_dofs)
    for ndof in range(total_dofs):
        m_dot_v[ndof] = 0.0
        # if ndof >= 10100:
        #     print(ndof, mass_matrix[ndof], unknown_vector[ndof])

    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            local_lni = lni - offset
            dofsi = node[nodeIDi, bodyID].dof
            if dofsi >= 0:
                Ab = 0.0
                for lnj in range(offset, offset + int(node_size[np])):
                    nodeIDj = LnID[lnj]
                    local_lnj = lnj - offset
                    dofsj = node[nodeIDj, bodyID].dof
                    if dofsj >= 0:
                        Ab += local_stiffness[np, local_lni, local_lnj] * unknown_vector[dofsj]
                m_dot_v[dofsi] += Ab
