import taichi as ti


@ti.func
def jacobian2parent2parametric1d(eleknot):
    return 0.5 * (eleknot[1] - eleknot[0])


@ti.func
def jacobian2parent2parametric2d(eleknot_u, eleknot_v):
    return jacobian2parent2parametric1d(eleknot_u) * jacobian2parent2parametric1d(eleknot_v)


@ti.func
def linearize(index, vector):
    assert index.n == vector.n
    if ti.static(vector.n == 2):
        return int(index[0] + index[1] * vector[0])
    elif ti.static(vector.n == 3):
        return int(index[0] + index[1] * vector[0] + index[2] * vector[0] * vector[1])


@ti.func
def vectorize_id(index, countVec):
    if ti.static(countVec.n == 2):
        ig = index % countVec[0]
        jg = index // countVec[0]
        return ig, jg
    elif ti.static(countVec.n == 3):
        ig = (index % (countVec[0] * countVec[1])) % countVec[0]
        jg = (index % (countVec[0] * countVec[1])) // countVec[0]
        kg = index // (countVec[0] * countVec[1])
        return ig, jg, kg


@ti.func
def matrix_cols(matrix, cols):
    return ti.Vector([matrix[i, cols] for i in ti.static(range(matrix.m))])


@ti.kernel
def copy_group_field(a: ti.template(), b: ti.template()):
    for I in ti.grouped(a):
        a[I] = b[I]
