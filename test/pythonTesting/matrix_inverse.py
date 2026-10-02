from src.utils.MatrixFunction import get_jacobian_inverse5

import taichi as ti
ti.init(debug=True)
import numpy as np

a4 = ti.Matrix([
                   [1,2,3,4],
                   [3,3,4,5],
                   [7,4,5,6],
                   [1,2,2,5]
])

a5 = ti.Matrix([
                   [1,6,3,4,5],
                   [4,3,3,5,6],
                   [3,4,4,6,7],
                   [1,2,2,2,3],
                   [5,3,5,3,4]
])

b = np.array([
                   [1,6,3,4,5],
                   [4,3,3,5,6],
                   [3,4,4,6,7],
                   [1,2,2,2,3],
                   [5,3,5,3,4]
])

print(np.linalg.inv(b))

@ti.kernel
def k4(mat: ti.types.matrix(4, 4, float)):
    print(mat.inverse())

#k4(a4)

@ti.kernel
def k5(mat: ti.types.matrix(5, 5, float)):
    print(get_jacobian_inverse5(mat)@mat)

k5(a5)