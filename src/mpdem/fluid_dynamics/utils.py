import taichi as ti


from src.utils.TypeDefination import vec3i, vec3u8
from src.utils.ScalarFunction import vectorize_id


@ti.kernel
def kernel_set_boundary_type(gridSum: int, gnum: ti.types.vector(3, int), boundary_type: ti.template()):
    for i in range(boundary_type.shape[0]):
        Ind = vec3i(vectorize_id(i, gnum))
        xtype, ytype, ztype = 0, 0, 0
        if Ind[0] - 1 < 0:
            xtype = 1
        elif Ind[0] + 1 >= gnum[0]:
            xtype = 4
        if Ind[1] - 1 < 0:
            ytype = 1
        elif Ind[1] + 1 >= gnum[1]:
            ytype = 4
        if Ind[2] - 1 < 0:
            ztype = 1
        elif Ind[2] + 1 >= gnum[2]:
            ztype = 4
        boundary_type[int(i % gridSum), int(i // gridSum)] = vec3u8(xtype, ytype, ztype)