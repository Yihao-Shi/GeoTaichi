import taichi as ti

from src.utils.constants import PENALTY, BLOCK_SZ, Threshold
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefination import vec2f, vec3f, vec4f, vec9f, vec2i, vec3i
from src.utils.VectorFunction import summation
from src.utils.ScalarFunction import linearize, vectorize_id


# ========================================================= #
#     Matrix free Preconditioning conjuction gradient       #
# ========================================================= #
@ti.kernel
def unknow_reset(old_total_dofs: int, unknown_vector: ti.template()):
    for i in range(old_total_dofs):
        unknown_vector[i] = 0.

@ti.kernel
def matrix_reset_(total_dofs: int, diag_A: ti.template(), mass_matrix: ti.template(), unknown_vector: ti.template()):
    for i in range(total_dofs):
        diag_A[i] = mass_matrix[i]
        unknown_vector[i] = 0.

@ti.kernel
def local_stiffness_reset(old_total_nodes: int, influenced_dofs: int, local_stiffness: ti.template()):
    for i in range(old_total_nodes * influenced_dofs * influenced_dofs):
        local_stiffness[i] = 0.

@ti.kernel
def MvP_reset(total_dofs: int, m_dot_v: ti.template()):
    for i in range(total_dofs):
        m_dot_v[i] = 0.

@ti.func
def assemble_elastic_diagonal_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec2f(0., 0.)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + \
                         jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3]) + \
                         0.5 * jdshape[1] * (2. * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    local_stiffness[1] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1]) - \
                         0.5 * jdshape[0] * (-2. * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    return local_stiffness

@ti.func
def assemble_elastic_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec4f(0., 0., 0., 0.)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + \
                         jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3]) + \
                         0.5 * jdshape[1] * (2. * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    local_stiffness[1] = idshape[0] * jdshape[1] * material_stiffness[0, 1] + idshape[1] * jdshape[0] * material_stiffness[3, 3] + \
                         jdshape[1] * (idshape[0] * stress[0] + idshape[1] * stress[3]) - \
                         0.5 * jdshape[0] * (2. * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    
    local_stiffness[2] = idshape[1] * jdshape[0] * material_stiffness[1, 0] + idshape[0] * jdshape[1] * material_stiffness[3, 3] + \
                         jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1]) + \
                         0.5 * jdshape[1] * (-2. * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    local_stiffness[3] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1]) - \
                         0.5 * jdshape[0] * (-2. * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    return local_stiffness

@ti.func
def assemble_diagonal_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec2f(0., 0.)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[0] * material_stiffness[3, 0] + \
                         idshape[0] * jdshape[1] * material_stiffness[0, 3] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + \
                         jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3]) + \
                         0.5 * jdshape[1] * (2. * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    local_stiffness[1] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[1] * material_stiffness[3, 1] + \
                         idshape[1] * jdshape[0] * material_stiffness[1, 3] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1]) - \
                         0.5 * jdshape[0] * (-2. * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    return local_stiffness

@ti.func
def assemble_stiffness_matrix_2D(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec4f(0., 0., 0., 0.)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[0] * material_stiffness[3, 0] + \
                         idshape[0] * jdshape[1] * material_stiffness[0, 3] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + \
                         jdshape[0] * (idshape[0] * stress[0] + idshape[1] * stress[3]) + \
                         0.5 * jdshape[1] * (2. * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    local_stiffness[1] = idshape[0] * jdshape[1] * material_stiffness[0, 1] + idshape[1] * jdshape[1] * material_stiffness[3, 1] + \
                         idshape[0] * jdshape[0] * material_stiffness[0, 3] + idshape[1] * jdshape[0] * material_stiffness[3, 3] + \
                         jdshape[1] * (idshape[0] * stress[0] + idshape[1] * stress[3]) - \
                         0.5 * jdshape[0] * (2. * idshape[0] * stress[3] + idshape[1] * (stress[1] - stress[0]))
    
    local_stiffness[2] = idshape[1] * jdshape[0] * material_stiffness[1, 0] + idshape[0] * jdshape[0] * material_stiffness[3, 0] + \
                         idshape[1] * jdshape[1] * material_stiffness[1, 3] + idshape[0] * jdshape[1] * material_stiffness[3, 3] + \
                         jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1]) + \
                         0.5 * jdshape[1] * (-2. * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    local_stiffness[3] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[1] * material_stiffness[3, 1] + \
                         idshape[1] * jdshape[0] * material_stiffness[1, 3] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1]) - \
                         0.5 * jdshape[0] * (-2. * idshape[1] * stress[3] + idshape[0] * (stress[1] - stress[0]))
    return local_stiffness

@ti.kernel
def kernel_assemble_local_stiffness_2D(total_nodes: int, particleNum: int, particle: ti.template(), dshapefn: ti.template(), node_size: ti.template(), 
                                    stiffness_matrix: ti.template(), local_stiffness: ti.template(), assemble_matrix: ti.template()):
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
                local_stiffness[np, 2 * local_lni    , 2 * local_lnj    ] = particle_stiffness[0]
                local_stiffness[np, 2 * local_lni    , 2 * local_lnj + 1] = particle_stiffness[1]
                local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj    ] = particle_stiffness[2]
                local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj + 1] = particle_stiffness[3]

@ti.kernel
def kernel_counting_row_offsets(influenced_node: int, gridSum: int, cutoff: float, gnum: ti.types.vector(3, int), node: ti.template(), flag: ti.template(), offset: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                ig, jg, kg = vectorize_id(ng, gnum)
                dof0 = flag[ng + nb * gridSum]
                counting = 0
                for i in range(0, 2 * influenced_node + 1):
                    if ig - influenced_node + i and node[linearize(vec3i(ig - influenced_node + i, jg, kg), gnum)].m > cutoff:
                        counting += 3
                    if jg - influenced_node + i and node[linearize(vec3i(ig, jg - influenced_node + i, kg), gnum)].m > cutoff:
                        counting += 3
                    if kg - influenced_node + i and node[linearize(vec3i(ig, jg, kg - influenced_node + i), gnum)].m > cutoff:
                        counting += 3
                offset[dof0] = counting
                offset[dof0 + 1] = counting
                offset[dof0 + 2] = counting

@ti.kernel
def kernel_assemble_global_stiffness(gridSum: int, total_nodes: int, particleNum: int, node: ti.template(), particle: ti.template(), LnID: ti.template(), dshapefn: ti.template(), node_size: ti.template(), flag: ti.template(), 
                                    stiffness_matrix: ti.template(), offset: ti.template(), indices: ti.template(), values: ti.template(), assemble_matrix: ti.template()):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        volume = particle[np].vol
        material_stiffness = stiffness_matrix[np]
        offset = np * total_nodes
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            idshape = dshapefn[lni]
            dofsi = flag[nodeIDi + bodyID * gridSum]
            xrow_offset = offset[dofsi]
            yrow_offset = offset[dofsi]
            zrow_offset = offset[dofsi]
            local_lni = lni - offset
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                jdshape = dshapefn[lnj]
                dofsj = flag[nodeIDj + bodyID * gridSum]
                local_lnj = lnj - offset
                particle_stiffness = assemble_matrix(idshape, jdshape, material_stiffness) * volume

                for dim in ti.static(range(GlobalVariable.DIMENSION * GlobalVariable.DIMENSION)):

                    values[row_offset] += particle_stiffness[0]
                    row_offset += 1

@ti.func
def assemble_elastic_diagonal_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec3f(0., 0., 0.)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + idshape[2] * jdshape[2] * material_stiffness[5, 5] + \
                         jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[5])
    local_stiffness[1] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + idshape[2] * jdshape[2] * material_stiffness[4, 4] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[4])
    local_stiffness[2] = idshape[2] * jdshape[2] * material_stiffness[2, 2] + idshape[1] * jdshape[1] * material_stiffness[4, 4] + idshape[0] * jdshape[0] * material_stiffness[5, 5] + \
                         jdshape[2] * (idshape[0] * stress[5] + idshape[1] * stress[4] + idshape[2] * stress[2])
    return local_stiffness

@ti.func
def assemble_diagonal_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec3f(0., 0., 0.)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[0] * material_stiffness[3, 0] + idshape[2] * jdshape[0] * material_stiffness[5, 0] + \
                         idshape[0] * jdshape[1] * material_stiffness[0, 3] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + idshape[2] * jdshape[1] * material_stiffness[5, 3] + \
                         idshape[0] * jdshape[2] * material_stiffness[0, 5] + idshape[1] * jdshape[2] * material_stiffness[3, 5] + idshape[2] * jdshape[2] * material_stiffness[5, 5] + \
                         jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[5])
    
    local_stiffness[1] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[1] * material_stiffness[3, 1] + idshape[2] * jdshape[1] * material_stiffness[4, 1] + \
                         idshape[1] * jdshape[0] * material_stiffness[1, 3] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + idshape[2] * jdshape[0] * material_stiffness[4, 3] + \
                         idshape[1] * jdshape[2] * material_stiffness[1, 4] + idshape[0] * jdshape[2] * material_stiffness[3, 4] + idshape[2] * jdshape[2] * material_stiffness[4, 4] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[4])
    
    local_stiffness[2] = idshape[2] * jdshape[2] * material_stiffness[2, 2] + idshape[1] * jdshape[2] * material_stiffness[4, 2] + idshape[0] * jdshape[2] * material_stiffness[5, 2] + \
                         idshape[2] * jdshape[1] * material_stiffness[2, 4] + idshape[1] * jdshape[1] * material_stiffness[4, 4] + idshape[0] * jdshape[1] * material_stiffness[5, 4] + \
                         idshape[2] * jdshape[0] * material_stiffness[2, 5] + idshape[1] * jdshape[0] * material_stiffness[4, 5] + idshape[0] * jdshape[0] * material_stiffness[5, 5] + \
                         jdshape[2] * (idshape[0] * stress[5] + idshape[1] * stress[4] + idshape[2] * stress[2])
    return local_stiffness

@ti.func
def assemble_elastic_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec9f(0., 0., 0., 0., 0., 0., 0., 0., 0.)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + idshape[2] * jdshape[2] * material_stiffness[5, 5] + \
                         jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[5])
    local_stiffness[1] = idshape[0] * jdshape[1] * material_stiffness[0, 1] + idshape[1] * jdshape[0] * material_stiffness[3, 3] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[5])
    local_stiffness[2] = idshape[0] * jdshape[2] * material_stiffness[0, 2] + idshape[2] * jdshape[0] * material_stiffness[5, 5] + \
                         jdshape[2] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[5])
    
    local_stiffness[3] = idshape[1] * jdshape[0] * material_stiffness[1, 0] + idshape[0] * jdshape[1] * material_stiffness[3, 3] + \
                         jdshape[0] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[4])
    local_stiffness[4] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + idshape[2] * jdshape[2] * material_stiffness[4, 4] + \
                         jdshape[1] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[4])
    local_stiffness[5] = idshape[1] * jdshape[2] * material_stiffness[1, 2] + idshape[2] * jdshape[1] * material_stiffness[4, 4] + \
                         jdshape[2] * (idshape[0] * stress[3] + idshape[1] * stress[1] + idshape[2] * stress[4])
    
    local_stiffness[6] = idshape[2] * jdshape[0] * material_stiffness[2, 0] + idshape[0] * jdshape[2] * material_stiffness[5, 5] + \
                         jdshape[0] * (idshape[0] * stress[5] + idshape[1] * stress[4] + idshape[2] * stress[2])
    local_stiffness[7] = idshape[2] * jdshape[1] * material_stiffness[2, 1] + idshape[1] * jdshape[2] * material_stiffness[4, 4] + \
                         jdshape[1] * (idshape[0] * stress[5] + idshape[1] * stress[4] + idshape[2] * stress[2])
    local_stiffness[8] = idshape[2] * jdshape[2] * material_stiffness[2, 2] + idshape[1] * jdshape[1] * material_stiffness[4, 4] + idshape[0] * jdshape[0] * material_stiffness[5, 5] + \
                         jdshape[2] * (idshape[0] * stress[5] + idshape[1] * stress[4] + idshape[2] * stress[2])
    return local_stiffness

@ti.func
def assemble_stiffness_matrix(idshape, jdshape, material_stiffness, stress):
    local_stiffness = vec9f(0, 0, 0, 0, 0, 0, 0, 0, 0)
    local_stiffness[0] = idshape[0] * jdshape[0] * material_stiffness[0, 0] + idshape[1] * jdshape[0] * material_stiffness[3, 0] + idshape[2] * jdshape[0] * material_stiffness[5, 0] + \
                         idshape[0] * jdshape[1] * material_stiffness[0, 3] + idshape[1] * jdshape[1] * material_stiffness[3, 3] + idshape[2] * jdshape[1] * material_stiffness[5, 3] + \
                         idshape[0] * jdshape[2] * material_stiffness[0, 5] + idshape[1] * jdshape[2] * material_stiffness[3, 5] + idshape[2] * jdshape[2] * material_stiffness[5, 5]
    local_stiffness[1] = idshape[0] * jdshape[1] * material_stiffness[0, 1] + idshape[1] * jdshape[1] * material_stiffness[3, 1] + idshape[2] * jdshape[1] * material_stiffness[5, 1] + \
                         idshape[0] * jdshape[0] * material_stiffness[0, 3] + idshape[1] * jdshape[0] * material_stiffness[3, 3] + idshape[2] * jdshape[0] * material_stiffness[5, 3] + \
                         idshape[0] * jdshape[2] * material_stiffness[0, 4] + idshape[1] * jdshape[2] * material_stiffness[3, 4] + idshape[2] * jdshape[2] * material_stiffness[5, 4]
    local_stiffness[2] = idshape[0] * jdshape[2] * material_stiffness[0, 2] + idshape[1] * jdshape[2] * material_stiffness[3, 2] + idshape[2] * jdshape[2] * material_stiffness[5, 2] + \
                         idshape[0] * jdshape[1] * material_stiffness[0, 4] + idshape[1] * jdshape[1] * material_stiffness[3, 4] + idshape[2] * jdshape[1] * material_stiffness[5, 4]+ \
                         idshape[0] * jdshape[0] * material_stiffness[0, 5] + idshape[1] * jdshape[0] * material_stiffness[3, 5] + idshape[2] * jdshape[0] * material_stiffness[5, 5]
    
    local_stiffness[3] = idshape[1] * jdshape[0] * material_stiffness[1, 0] + idshape[0] * jdshape[0] * material_stiffness[3, 0] + idshape[2] * jdshape[0] * material_stiffness[4, 0] + \
                         idshape[1] * jdshape[1] * material_stiffness[1, 3] + idshape[0] * jdshape[1] * material_stiffness[3, 3] + idshape[2] * jdshape[1] * material_stiffness[4, 3] + \
                         idshape[1] * jdshape[2] * material_stiffness[1, 5] + idshape[0] * jdshape[2] * material_stiffness[3, 5] + idshape[2] * jdshape[2] * material_stiffness[4, 5]
    local_stiffness[4] = idshape[1] * jdshape[1] * material_stiffness[1, 1] + idshape[0] * jdshape[1] * material_stiffness[3, 1] + idshape[2] * jdshape[1] * material_stiffness[4, 1] + \
                         idshape[1] * jdshape[0] * material_stiffness[1, 3] + idshape[0] * jdshape[0] * material_stiffness[3, 3] + idshape[2] * jdshape[0] * material_stiffness[4, 3] + \
                         idshape[1] * jdshape[2] * material_stiffness[1, 4] + idshape[0] * jdshape[2] * material_stiffness[3, 4] + idshape[2] * jdshape[2] * material_stiffness[4, 4]
    local_stiffness[5] = idshape[1] * jdshape[2] * material_stiffness[1, 2] + idshape[0] * jdshape[2] * material_stiffness[3, 2] + idshape[2] * jdshape[2] * material_stiffness[4, 2] + \
                         idshape[1] * jdshape[1] * material_stiffness[1, 4] + idshape[0] * jdshape[1] * material_stiffness[3, 4] + idshape[2] * jdshape[1] * material_stiffness[4, 4] + \
                         idshape[1] * jdshape[0] * material_stiffness[1, 5] + idshape[0] * jdshape[0] * material_stiffness[3, 5] + idshape[2] * jdshape[0] * material_stiffness[4, 5]

    local_stiffness[6] = idshape[2] * jdshape[0] * material_stiffness[2, 0] + idshape[1] * jdshape[0] * material_stiffness[4, 0] + idshape[0] * jdshape[0] * material_stiffness[5, 0] + \
                         idshape[2] * jdshape[1] * material_stiffness[2, 3] + idshape[1] * jdshape[1] * material_stiffness[4, 3] + idshape[0] * jdshape[1] * material_stiffness[5, 3] + \
                         idshape[2] * jdshape[2] * material_stiffness[2, 5] + idshape[1] * jdshape[2] * material_stiffness[4, 5] + idshape[0] * jdshape[2] * material_stiffness[5, 5]
    local_stiffness[7] = idshape[2] * jdshape[1] * material_stiffness[2, 1] + idshape[1] * jdshape[1] * material_stiffness[4, 1] + idshape[0] * jdshape[1] * material_stiffness[5, 1] + \
                         idshape[2] * jdshape[0] * material_stiffness[2, 3] + idshape[1] * jdshape[0] * material_stiffness[4, 3] + idshape[0] * jdshape[0] * material_stiffness[5, 3] + \
                         idshape[2] * jdshape[2] * material_stiffness[2, 4] + idshape[1] * jdshape[2] * material_stiffness[4, 4] + idshape[0] * jdshape[2] * material_stiffness[5, 4]
    local_stiffness[8] = idshape[2] * jdshape[2] * material_stiffness[2, 2] + idshape[1] * jdshape[2] * material_stiffness[4, 2] + idshape[0] * jdshape[2] * material_stiffness[5, 2] + \
                         idshape[2] * jdshape[1] * material_stiffness[2, 4] + idshape[1] * jdshape[1] * material_stiffness[4, 4] + idshape[0] * jdshape[1] * material_stiffness[5, 4] + \
                         idshape[2] * jdshape[0] * material_stiffness[2, 5] + idshape[1] * jdshape[0] * material_stiffness[4, 5] + idshape[0] * jdshape[0] * material_stiffness[5, 5]
    return local_stiffness

@ti.kernel
def kernel_assemble_local_stiffness(total_nodes: int, particleNum: int, particle: ti.template(), dshapefn: ti.template(), node_size: ti.template(), 
                                    stiffness_matrix: ti.template(), local_stiffness: ti.template(), assemble_matrix: ti.template()):
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
                local_stiffness[np, 3 * local_lni    , 3 * local_lnj    ] = particle_stiffness[0]
                local_stiffness[np, 3 * local_lni    , 3 * local_lnj + 1] = particle_stiffness[1]
                local_stiffness[np, 3 * local_lni    , 3 * local_lnj + 2] = particle_stiffness[2]
                local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj    ] = particle_stiffness[3]
                local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 1] = particle_stiffness[4]
                local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 2] = particle_stiffness[5]
                local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj    ] = particle_stiffness[6]
                local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 1] = particle_stiffness[7]
                local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 2] = particle_stiffness[8]

@ti.kernel
def kernel_compute_mass_matrix(gridSum: int, beta: float, cutoff: float, dt: ti.template(), node: ti.template(), flag: ti.template(), mass_matrix: ti.template()):
    constant = 1. / dt[None] / dt[None] / beta
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            nodal_mass = node[ng, nb].m
            if nodal_mass > cutoff:
                dof0 = flag[ng + nb * gridSum]
                for d in ti.static(range(GlobalVariable.DIMENSION)):
                    mass_matrix[dof0 + d] = nodal_mass * constant

@ti.kernel
def kernel_compute_penalty_matrix(gridSum: int, cut_off: float, displacementNum: int, displacement_constraint: ti.template(), node: ti.template(), flag: ti.template(), mass_matrix: ti.template()):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node 
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            mass_matrix[global_dof + dofID] += PENALTY

@ti.kernel
def kernel_moment_balance_cg_2D(gridSum: int, total_nodes: int, total_dofs: int, particleNum: int, particle: ti.template(), node_size: ti.template(), LnID: ti.template(), flag: ti.template(), 
                                mass_matrix: ti.template(), local_stiffness: ti.template(), unknown_vector: ti.template(), m_dot_v: ti.template()):
    for ndof in range(total_dofs):
        m_dot_v[ndof] = mass_matrix[ndof] * unknown_vector[ndof]

    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            local_lni = lni - offset
            dofsi = flag[nodeIDi + bodyID * gridSum]
            Axb, Ayb = 0., 0.
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                local_lnj = lnj - offset
                dofsj = flag[nodeIDj + bodyID * gridSum]
                Axb += local_stiffness[np, 2 * local_lni,     2 * local_lnj    ] * unknown_vector[dofsj    ] + \
                       local_stiffness[np, 2 * local_lni,     2 * local_lnj + 1] * unknown_vector[dofsj + 1]
                Ayb += local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj    ] * unknown_vector[dofsj    ] + \
                       local_stiffness[np, 2 * local_lni + 1, 2 * local_lnj + 1] * unknown_vector[dofsj + 1]
            m_dot_v[dofsi    ] += Axb
            m_dot_v[dofsi + 1] += Ayb

@ti.kernel
def kernel_moment_balance_cg(gridSum: int, total_nodes: int, total_dofs: int, particleNum: int, particle: ti.template(), node_size: ti.template(), LnID: ti.template(), flag: ti.template(), 
                             mass_matrix: ti.template(), local_stiffness: ti.template(), unknown_vector: ti.template(), m_dot_v: ti.template()):
    for ndof in range(total_dofs):
        m_dot_v[ndof] = mass_matrix[ndof] * unknown_vector[ndof]
    
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for lni in range(offset, offset + int(node_size[np])):
            nodeIDi = LnID[lni]
            local_lni = lni - offset
            dofsi = flag[nodeIDi + bodyID * gridSum]
            Axb, Ayb, Azb = 0., 0., 0.
            for lnj in range(offset, offset + int(node_size[np])):
                nodeIDj = LnID[lnj]
                local_lnj = lnj - offset
                dofsj = flag[nodeIDj + bodyID * gridSum]
                Axb += local_stiffness[np, 3 * local_lni,     3 * local_lnj    ] * unknown_vector[dofsj    ] + \
                       local_stiffness[np, 3 * local_lni,     3 * local_lnj + 1] * unknown_vector[dofsj + 1] + \
                       local_stiffness[np, 3 * local_lni,     3 * local_lnj + 2] * unknown_vector[dofsj + 2]
                Ayb += local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj    ] * unknown_vector[dofsj    ] + \
                       local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 1] * unknown_vector[dofsj + 1] + \
                       local_stiffness[np, 3 * local_lni + 1, 3 * local_lnj + 2] * unknown_vector[dofsj + 2]
                Azb += local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj    ] * unknown_vector[dofsj    ] + \
                       local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 1] * unknown_vector[dofsj + 1] + \
                       local_stiffness[np, 3 * local_lni + 2, 3 * local_lnj + 2] * unknown_vector[dofsj + 2]
            m_dot_v[dofsi    ] += Axb
            m_dot_v[dofsi + 1] += Ayb
            m_dot_v[dofsi + 2] += Azb

@ti.kernel
def kernel_moment_balance_cg_sparse_matrix(offset: ti.template(), indice: ti.template(), data: ti.template(), unknown_vector: ti.template(), m_dot_v: ti.template()):
    m_dot_v.fill(0)
    for nrows in range(1, offset.shape[0]):
        sums = 0.
        for noffs in range(offset[nrows - 1], offset[nrows]):
            ncols = indice[noffs]
            sums += data[noffs] * unknown_vector[ncols]
        m_dot_v[nrows] = sums

@ti.kernel
def kernel_assemble_sparse_matrix(total_nodes: int, particleNum: int, particle: ti.template(), node_size: ti.template(), diag_A: ti.template(), LnID: ti.template(), dshapefn: ti.template(),
                                  node: ti.template(), stiffness_matrix: ti.template(), assemble_matrix: ti.template()):
    for np in range(particleNum):
        material_stiffness = stiffness_matrix[np]
        offset = np * total_nodes
        volume = particle[np].vol
        for lni in range(offset, offset + int(node_size[np])):
            idshape = dshapefn[lni]
            local_lni = lni - offset
            for lnj in range(offset, offset + int(node_size[np])):
                jdshape = dshapefn[lnj]
                local_lnj = lnj - offset
                particle_stiffness = assemble_matrix(idshape, jdshape, material_stiffness) * volume
                

@ti.kernel
def kernel_preconditioning_matrix_2D(gridSum: int, total_nodes: int, particleNum: int, particle: ti.template(), node_size: ti.template(), diag_A: ti.template(), 
                                     LnID: ti.template(), flag: ti.template(), local_stiffness: ti.template()):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            local_ln = ln - offset
            dofs = flag[nodeID + bodyID * gridSum]
            diag_A[dofs]     += local_stiffness[np, 2 * local_ln    , 2 * local_ln    ] 
            diag_A[dofs + 1] += local_stiffness[np, 2 * local_ln + 1, 2 * local_ln + 1] 

@ti.kernel
def kernel_preconditioning_matrix(gridSum: int, total_nodes: int, particleNum: int, particle: ti.template(), node_size: ti.template(), diag_A: ti.template(), LnID: ti.template(), flag: ti.template(), local_stiffness: ti.template()):
    for np in range(particleNum):
        bodyID = int(particle[np].bodyID)
        offset = np * total_nodes
        for ln in range(offset, offset + int(node_size[np])):
            nodeID = LnID[ln]
            local_ln = ln - offset
            dofs = flag[nodeID + bodyID * gridSum]
            diag_A[dofs]     += local_stiffness[np, 3 * local_ln    , 3 * local_ln    ] 
            diag_A[dofs + 1] += local_stiffness[np, 3 * local_ln + 1, 3 * local_ln + 1]
            diag_A[dofs + 2] += local_stiffness[np, 3 * local_ln + 2, 3 * local_ln + 2]

@ti.kernel
def kernel_assemble_displacement_load(gridSum: int, cut_off: float, displacementNum: int, displacement_constraint: ti.template(), node: ti.template(), flag: ti.template(), right_hand_vector: ti.template(), diag_A: ti.template()):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node 
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            right_hand_vector[global_dof + dofID] = diag_A[global_dof + dofID] * displacement_constraint[i].value 

@ti.kernel
def kernel_assemble_residual_force_quasi_static_2D(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), right_hand_vector: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                external_force = node[ng, nb].ext_force
                internal_force = node[ng, nb].int_force
                right_hand_vector[dof0]     = external_force[0] + internal_force[0] 
                right_hand_vector[dof0 + 1] = external_force[1] + internal_force[1] 

@ti.kernel
def kernel_assemble_residual_force_quasi_static(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), right_hand_vector: ti.template()):
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                external_force = node[ng, nb].ext_force
                internal_force = node[ng, nb].int_force
                right_hand_vector[dof0]     = external_force[0] + internal_force[0] 
                right_hand_vector[dof0 + 1] = external_force[1] + internal_force[1] 
                right_hand_vector[dof0 + 2] = external_force[2] + internal_force[2] 

@ti.kernel
def kernel_assemble_residual_force_dynamic_2D(gridSum: int, cutoff: float, beta: float, node: ti.template(), flag: ti.template(), right_hand_vector: ti.template(), dt: ti.template()):
    constant1 = 1. / dt[None] / dt[None] / beta
    constant2 = constant1 * dt[None]
    constant3 = 0.5 / beta - 1.
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

                right_hand_vector[dof0]     = external_force[0] + internal_force[0] - mass * (constant1 * displacement[0] - constant2 * velocity[0] - constant3 * acceleration[0])
                right_hand_vector[dof0 + 1] = external_force[1] + internal_force[1] - mass * (constant1 * displacement[1] - constant2 * velocity[1] - constant3 * acceleration[1])

@ti.kernel
def kernel_assemble_residual_force_dynamic(gridSum: int, cutoff: float, beta: float, node: ti.template(), flag: ti.template(), right_hand_vector: ti.template(), dt: ti.template()):
    constant1 = 1. / dt[None] / dt[None] / beta
    constant2 = constant1 * dt[None]
    constant3 = 0.5 / beta - 1.
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

                right_hand_vector[dof0]     = external_force[0] + internal_force[0] - mass * (constant1 * displacement[0] - constant2 * velocity[0] - constant3 * acceleration[0])
                right_hand_vector[dof0 + 1] = external_force[1] + internal_force[1] - mass * (constant1 * displacement[1] - constant2 * velocity[1] - constant3 * acceleration[1])
                right_hand_vector[dof0 + 2] = external_force[2] + internal_force[2] - mass * (constant1 * displacement[2] - constant2 * velocity[2] - constant3 * acceleration[2])

@ti.kernel
def kernel_calculate_reaction_forces(gridSum: int, cut_off: float, displacementNum: int, displacement_constraint: ti.template(), node: ti.template(), flag: ti.template(), accmulated_reaction_forces: ti.template()):
    for i in range(displacementNum):
        nodeID = displacement_constraint[i].node 
        dofID = int(displacement_constraint[i].dof)
        bodyID = int(displacement_constraint[i].level)
        if node[nodeID, bodyID].m > cut_off:
            global_dof = flag[nodeID + bodyID * gridSum]
            accmulated_reaction_forces[global_dof + dofID] = (displacement_constraint[i].value - node[nodeID, bodyID].displacement[dofID]) * PENALTY

# ========================================================= #
#                  Convergence criterion                    #
# ========================================================= #    
@ti.kernel
def compute_disp_error_2D(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), disp: ti.template()) -> float:
    delta_u = 0.
    u = 0.
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                dux, duy = disp[dof0], disp[dof0 + 1]
                displacement = node[ng, nb].displacement
                delta_u += dux * dux + duy * duy
                u += summation(displacement)
    return ti.sqrt(delta_u / u)  if ti.abs(u) > Threshold else 1.

@ti.kernel
def compute_disp_error(gridSum: int, cutoff: float, node: ti.template(), flag: ti.template(), disp: ti.template()) -> float:
    delta_u = 0.
    u = 0.
    for ng in range(node.shape[0]):
        for nb in range(node.shape[1]):
            if node[ng, nb].m > cutoff:
                dof0 = flag[ng + nb * gridSum]
                dux, duy, duz = disp[dof0], disp[dof0 + 1], disp[dof0 + 2]
                displacement = node[ng, nb].displacement
                delta_u += dux * dux + duy * duy + duz * duz
                u += summation(displacement)
    return ti.sqrt(delta_u / u) if ti.abs(u) > Threshold else 1.

@ti.kernel
def compute_residual_error(active_id: int, right_hand_vector: ti.template()) -> float:
    rfs = 0.
    ti.loop_config(block_dim=64)
    size = ti.ceil(active_id / 3, int)
    for i in range(size):
        thread_id = i % BLOCK_SZ
        pad_vector1 = ti.simt.block.SharedArray((64, ), ti.f64)
        pad_vector2 = ti.simt.block.SharedArray((64, ), ti.f64)
        pad_vector3 = ti.simt.block.SharedArray((64, ), ti.f64)

        pad_vector1[thread_id] = right_hand_vector[3 * i]
        pad_vector2[thread_id] = right_hand_vector[3 * i + 1]
        pad_vector3[thread_id] = right_hand_vector[3 * i + 2]
        ti.simt.block.sync()

        temp = 0.    
        if thread_id == BLOCK_SZ - 1 or i == size - 1:
            for k in range(thread_id + 1):
                temp += pad_vector1[k] * pad_vector1[k] + pad_vector2[k] * pad_vector2[k] + pad_vector3[k] * pad_vector3[k]
        rfs += temp
    return ti.sqrt(rfs)

