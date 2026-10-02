import sys
from pathlib import Path

import numpy as np
repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *


mode = 2

if mode == 1:
    dfdx = make_matrix('dfdx', 3, 4)
    d2psid2f = make_matrix('d2psid2f', 4, 4)
    d2psid2x = dfdx @ d2psid2f @ dfdx.T

    code = ''
    pair, reduced = cse(d2psid2x)
    for p in pair:
        code += pycode(p[0]) + ' = ' + pycode(p[1]) + ';\n'
    for i, r in enumerate(reduced):
        code += '\tout[' + str(i) + '] = ' + pycode(r) + ';'

    print(code)

    """ dFdx = np.random.rand(3, 9)
    d2Psi_d2F = np.random.rand(9, 9)
    d2Psi_d2F = 0.5 * (d2Psi_d2F + d2Psi_d2F.T)
    x0 = d2Psi_d2F[0, 0] * dFdx[0, 0] + d2Psi_d2F[1, 0] * dFdx[0, 1] + d2Psi_d2F[2, 0] * dFdx[0, 2] + d2Psi_d2F[3, 0] * dFdx[0, 3] + d2Psi_d2F[4, 0] * dFdx[0, 4] + d2Psi_d2F[5, 0] * dFdx[0, 5] + d2Psi_d2F[6, 0] * dFdx[0, 6] + d2Psi_d2F[7, 0] * dFdx[0, 7] + d2Psi_d2F[8, 0] * dFdx[0, 8]
    x1 = d2Psi_d2F[0, 1] * dFdx[0, 0] + d2Psi_d2F[1, 1] * dFdx[0, 1] + d2Psi_d2F[2, 1] * dFdx[0, 2] + d2Psi_d2F[3, 1] * dFdx[0, 3] + d2Psi_d2F[4, 1] * dFdx[0, 4] + d2Psi_d2F[5, 1] * dFdx[0, 5] + d2Psi_d2F[6, 1] * dFdx[0, 6] + d2Psi_d2F[7, 1] * dFdx[0, 7] + d2Psi_d2F[8, 1] * dFdx[0, 8]
    x2 = d2Psi_d2F[0, 2] * dFdx[0, 0] + d2Psi_d2F[1, 2] * dFdx[0, 1] + d2Psi_d2F[2, 2] * dFdx[0, 2] + d2Psi_d2F[3, 2] * dFdx[0, 3] + d2Psi_d2F[4, 2] * dFdx[0, 4] + d2Psi_d2F[5, 2] * dFdx[0, 5] + d2Psi_d2F[6, 2] * dFdx[0, 6] + d2Psi_d2F[7, 2] * dFdx[0, 7] + d2Psi_d2F[8, 2] * dFdx[0, 8]
    x3 = d2Psi_d2F[0, 3] * dFdx[0, 0] + d2Psi_d2F[1, 3] * dFdx[0, 1] + d2Psi_d2F[2, 3] * dFdx[0, 2] + d2Psi_d2F[3, 3] * dFdx[0, 3] + d2Psi_d2F[4, 3] * dFdx[0, 4] + d2Psi_d2F[5, 3] * dFdx[0, 5] + d2Psi_d2F[6, 3] * dFdx[0, 6] + d2Psi_d2F[7, 3] * dFdx[0, 7] + d2Psi_d2F[8, 3] * dFdx[0, 8]
    x4 = d2Psi_d2F[0, 4] * dFdx[0, 0] + d2Psi_d2F[1, 4] * dFdx[0, 1] + d2Psi_d2F[2, 4] * dFdx[0, 2] + d2Psi_d2F[3, 4] * dFdx[0, 3] + d2Psi_d2F[4, 4] * dFdx[0, 4] + d2Psi_d2F[5, 4] * dFdx[0, 5] + d2Psi_d2F[6, 4] * dFdx[0, 6] + d2Psi_d2F[7, 4] * dFdx[0, 7] + d2Psi_d2F[8, 4] * dFdx[0, 8]
    x5 = d2Psi_d2F[0, 5] * dFdx[0, 0] + d2Psi_d2F[1, 5] * dFdx[0, 1] + d2Psi_d2F[2, 5] * dFdx[0, 2] + d2Psi_d2F[3, 5] * dFdx[0, 3] + d2Psi_d2F[4, 5] * dFdx[0, 4] + d2Psi_d2F[5, 5] * dFdx[0, 5] + d2Psi_d2F[6, 5] * dFdx[0, 6] + d2Psi_d2F[7, 5] * dFdx[0, 7] + d2Psi_d2F[8, 5] * dFdx[0, 8]
    x6 = d2Psi_d2F[0, 6] * dFdx[0, 0] + d2Psi_d2F[1, 6] * dFdx[0, 1] + d2Psi_d2F[2, 6] * dFdx[0, 2] + d2Psi_d2F[3, 6] * dFdx[0, 3] + d2Psi_d2F[4, 6] * dFdx[0, 4] + d2Psi_d2F[5, 6] * dFdx[0, 5] + d2Psi_d2F[6, 6] * dFdx[0, 6] + d2Psi_d2F[7, 6] * dFdx[0, 7] + d2Psi_d2F[8, 6] * dFdx[0, 8]
    x7 = d2Psi_d2F[0, 7] * dFdx[0, 0] + d2Psi_d2F[1, 7] * dFdx[0, 1] + d2Psi_d2F[2, 7] * dFdx[0, 2] + d2Psi_d2F[3, 7] * dFdx[0, 3] + d2Psi_d2F[4, 7] * dFdx[0, 4] + d2Psi_d2F[5, 7] * dFdx[0, 5] + d2Psi_d2F[6, 7] * dFdx[0, 6] + d2Psi_d2F[7, 7] * dFdx[0, 7] + d2Psi_d2F[8, 7] * dFdx[0, 8]
    x8 = d2Psi_d2F[0, 8] * dFdx[0, 0] + d2Psi_d2F[1, 8] * dFdx[0, 1] + d2Psi_d2F[2, 8] * dFdx[0, 2] + d2Psi_d2F[3, 8] * dFdx[0, 3] + d2Psi_d2F[4, 8] * dFdx[0, 4] + d2Psi_d2F[5, 8] * dFdx[0, 5] + d2Psi_d2F[6, 8] * dFdx[0, 6] + d2Psi_d2F[7, 8] * dFdx[0, 7] + d2Psi_d2F[8, 8] * dFdx[0, 8]
    x9 = d2Psi_d2F[0, 0] * dFdx[1, 0] + d2Psi_d2F[1, 0] * dFdx[1, 1] + d2Psi_d2F[2, 0] * dFdx[1, 2] + d2Psi_d2F[3, 0] * dFdx[1, 3] + d2Psi_d2F[4, 0] * dFdx[1, 4] + d2Psi_d2F[5, 0] * dFdx[1, 5] + d2Psi_d2F[6, 0] * dFdx[1, 6] + d2Psi_d2F[7, 0] * dFdx[1, 7] + d2Psi_d2F[8, 0] * dFdx[1, 8]
    x10 = d2Psi_d2F[0, 1] * dFdx[1, 0] + d2Psi_d2F[1, 1] * dFdx[1, 1] + d2Psi_d2F[2, 1] * dFdx[1, 2] + d2Psi_d2F[3, 1] * dFdx[1, 3] + d2Psi_d2F[4, 1] * dFdx[1, 4] + d2Psi_d2F[5, 1] * dFdx[1, 5] + d2Psi_d2F[6, 1] * dFdx[1, 6] + d2Psi_d2F[7, 1] * dFdx[1, 7] + d2Psi_d2F[8, 1] * dFdx[1, 8]
    x11 = d2Psi_d2F[0, 2] * dFdx[1, 0] + d2Psi_d2F[1, 2] * dFdx[1, 1] + d2Psi_d2F[2, 2] * dFdx[1, 2] + d2Psi_d2F[3, 2] * dFdx[1, 3] + d2Psi_d2F[4, 2] * dFdx[1, 4] + d2Psi_d2F[5, 2] * dFdx[1, 5] + d2Psi_d2F[6, 2] * dFdx[1, 6] + d2Psi_d2F[7, 2] * dFdx[1, 7] + d2Psi_d2F[8, 2] * dFdx[1, 8]
    x12 = d2Psi_d2F[0, 3] * dFdx[1, 0] + d2Psi_d2F[1, 3] * dFdx[1, 1] + d2Psi_d2F[2, 3] * dFdx[1, 2] + d2Psi_d2F[3, 3] * dFdx[1, 3] + d2Psi_d2F[4, 3] * dFdx[1, 4] + d2Psi_d2F[5, 3] * dFdx[1, 5] + d2Psi_d2F[6, 3] * dFdx[1, 6] + d2Psi_d2F[7, 3] * dFdx[1, 7] + d2Psi_d2F[8, 3] * dFdx[1, 8]
    x13 = d2Psi_d2F[0, 4] * dFdx[1, 0] + d2Psi_d2F[1, 4] * dFdx[1, 1] + d2Psi_d2F[2, 4] * dFdx[1, 2] + d2Psi_d2F[3, 4] * dFdx[1, 3] + d2Psi_d2F[4, 4] * dFdx[1, 4] + d2Psi_d2F[5, 4] * dFdx[1, 5] + d2Psi_d2F[6, 4] * dFdx[1, 6] + d2Psi_d2F[7, 4] * dFdx[1, 7] + d2Psi_d2F[8, 4] * dFdx[1, 8]
    x14 = d2Psi_d2F[0, 5] * dFdx[1, 0] + d2Psi_d2F[1, 5] * dFdx[1, 1] + d2Psi_d2F[2, 5] * dFdx[1, 2] + d2Psi_d2F[3, 5] * dFdx[1, 3] + d2Psi_d2F[4, 5] * dFdx[1, 4] + d2Psi_d2F[5, 5] * dFdx[1, 5] + d2Psi_d2F[6, 5] * dFdx[1, 6] + d2Psi_d2F[7, 5] * dFdx[1, 7] + d2Psi_d2F[8, 5] * dFdx[1, 8]
    x15 = d2Psi_d2F[0, 6] * dFdx[1, 0] + d2Psi_d2F[1, 6] * dFdx[1, 1] + d2Psi_d2F[2, 6] * dFdx[1, 2] + d2Psi_d2F[3, 6] * dFdx[1, 3] + d2Psi_d2F[4, 6] * dFdx[1, 4] + d2Psi_d2F[5, 6] * dFdx[1, 5] + d2Psi_d2F[6, 6] * dFdx[1, 6] + d2Psi_d2F[7, 6] * dFdx[1, 7] + d2Psi_d2F[8, 6] * dFdx[1, 8]
    x16 = d2Psi_d2F[0, 7] * dFdx[1, 0] + d2Psi_d2F[1, 7] * dFdx[1, 1] + d2Psi_d2F[2, 7] * dFdx[1, 2] + d2Psi_d2F[3, 7] * dFdx[1, 3] + d2Psi_d2F[4, 7] * dFdx[1, 4] + d2Psi_d2F[5, 7] * dFdx[1, 5] + d2Psi_d2F[6, 7] * dFdx[1, 6] + d2Psi_d2F[7, 7] * dFdx[1, 7] + d2Psi_d2F[8, 7] * dFdx[1, 8]
    x17 = d2Psi_d2F[0, 8] * dFdx[1, 0] + d2Psi_d2F[1, 8] * dFdx[1, 1] + d2Psi_d2F[2, 8] * dFdx[1, 2] + d2Psi_d2F[3, 8] * dFdx[1, 3] + d2Psi_d2F[4, 8] * dFdx[1, 4] + d2Psi_d2F[5, 8] * dFdx[1, 5] + d2Psi_d2F[6, 8] * dFdx[1, 6] + d2Psi_d2F[7, 8] * dFdx[1, 7] + d2Psi_d2F[8, 8] * dFdx[1, 8]
    x18 = d2Psi_d2F[0, 0] * dFdx[2, 0] + d2Psi_d2F[1, 0] * dFdx[2, 1] + d2Psi_d2F[2, 0] * dFdx[2, 2] + d2Psi_d2F[3, 0] * dFdx[2, 3] + d2Psi_d2F[4, 0] * dFdx[2, 4] + d2Psi_d2F[5, 0] * dFdx[2, 5] + d2Psi_d2F[6, 0] * dFdx[2, 6] + d2Psi_d2F[7, 0] * dFdx[2, 7] + d2Psi_d2F[8, 0] * dFdx[2, 8]
    x19 = d2Psi_d2F[0, 1] * dFdx[2, 0] + d2Psi_d2F[1, 1] * dFdx[2, 1] + d2Psi_d2F[2, 1] * dFdx[2, 2] + d2Psi_d2F[3, 1] * dFdx[2, 3] + d2Psi_d2F[4, 1] * dFdx[2, 4] + d2Psi_d2F[5, 1] * dFdx[2, 5] + d2Psi_d2F[6, 1] * dFdx[2, 6] + d2Psi_d2F[7, 1] * dFdx[2, 7] + d2Psi_d2F[8, 1] * dFdx[2, 8]
    x20 = d2Psi_d2F[0, 2] * dFdx[2, 0] + d2Psi_d2F[1, 2] * dFdx[2, 1] + d2Psi_d2F[2, 2] * dFdx[2, 2] + d2Psi_d2F[3, 2] * dFdx[2, 3] + d2Psi_d2F[4, 2] * dFdx[2, 4] + d2Psi_d2F[5, 2] * dFdx[2, 5] + d2Psi_d2F[6, 2] * dFdx[2, 6] + d2Psi_d2F[7, 2] * dFdx[2, 7] + d2Psi_d2F[8, 2] * dFdx[2, 8]
    x21 = d2Psi_d2F[0, 3] * dFdx[2, 0] + d2Psi_d2F[1, 3] * dFdx[2, 1] + d2Psi_d2F[2, 3] * dFdx[2, 2] + d2Psi_d2F[3, 3] * dFdx[2, 3] + d2Psi_d2F[4, 3] * dFdx[2, 4] + d2Psi_d2F[5, 3] * dFdx[2, 5] + d2Psi_d2F[6, 3] * dFdx[2, 6] + d2Psi_d2F[7, 3] * dFdx[2, 7] + d2Psi_d2F[8, 3] * dFdx[2, 8]
    x22 = d2Psi_d2F[0, 4] * dFdx[2, 0] + d2Psi_d2F[1, 4] * dFdx[2, 1] + d2Psi_d2F[2, 4] * dFdx[2, 2] + d2Psi_d2F[3, 4] * dFdx[2, 3] + d2Psi_d2F[4, 4] * dFdx[2, 4] + d2Psi_d2F[5, 4] * dFdx[2, 5] + d2Psi_d2F[6, 4] * dFdx[2, 6] + d2Psi_d2F[7, 4] * dFdx[2, 7] + d2Psi_d2F[8, 4] * dFdx[2, 8]
    x23 = d2Psi_d2F[0, 5] * dFdx[2, 0] + d2Psi_d2F[1, 5] * dFdx[2, 1] + d2Psi_d2F[2, 5] * dFdx[2, 2] + d2Psi_d2F[3, 5] * dFdx[2, 3] + d2Psi_d2F[4, 5] * dFdx[2, 4] + d2Psi_d2F[5, 5] * dFdx[2, 5] + d2Psi_d2F[6, 5] * dFdx[2, 6] + d2Psi_d2F[7, 5] * dFdx[2, 7] + d2Psi_d2F[8, 5] * dFdx[2, 8]
    x24 = d2Psi_d2F[0, 6] * dFdx[2, 0] + d2Psi_d2F[1, 6] * dFdx[2, 1] + d2Psi_d2F[2, 6] * dFdx[2, 2] + d2Psi_d2F[3, 6] * dFdx[2, 3] + d2Psi_d2F[4, 6] * dFdx[2, 4] + d2Psi_d2F[5, 6] * dFdx[2, 5] + d2Psi_d2F[6, 6] * dFdx[2, 6] + d2Psi_d2F[7, 6] * dFdx[2, 7] + d2Psi_d2F[8, 6] * dFdx[2, 8]
    x25 = d2Psi_d2F[0, 7] * dFdx[2, 0] + d2Psi_d2F[1, 7] * dFdx[2, 1] + d2Psi_d2F[2, 7] * dFdx[2, 2] + d2Psi_d2F[3, 7] * dFdx[2, 3] + d2Psi_d2F[4, 7] * dFdx[2, 4] + d2Psi_d2F[5, 7] * dFdx[2, 5] + d2Psi_d2F[6, 7] * dFdx[2, 6] + d2Psi_d2F[7, 7] * dFdx[2, 7] + d2Psi_d2F[8, 7] * dFdx[2, 8]
    x26 = d2Psi_d2F[0, 8] * dFdx[2, 0] + d2Psi_d2F[1, 8] * dFdx[2, 1] + d2Psi_d2F[2, 8] * dFdx[2, 2] + d2Psi_d2F[3, 8] * dFdx[2, 3] + d2Psi_d2F[4, 8] * dFdx[2, 4] + d2Psi_d2F[5, 8] * dFdx[2, 5] + d2Psi_d2F[6, 8] * dFdx[2, 6] + d2Psi_d2F[7, 8] * dFdx[2, 7] + d2Psi_d2F[8, 8] * dFdx[2, 8]

    local_hessian = np.zeros((3, 3))
    for d1 in range(3):
        for d2 in range(3):
            for i in range(3 * 3):
                for j in range(3 * 3):
                    local_hessian[d1, d2] += dFdx[d1, i] * d2Psi_d2F[i, j] * dFdx[d2, j] 

    local_hessian1 = np.array([[dFdx[0, 0]*x0 + dFdx[0, 1]*x1 + dFdx[0, 2]*x2 + dFdx[0, 3]*x3 + dFdx[0, 4]*x4 + dFdx[0, 5]*x5 + dFdx[0, 6]*x6 + dFdx[0, 7]*x7 + dFdx[0, 8]*x8, 
                                dFdx[1, 0]*x0 + dFdx[1, 1]*x1 + dFdx[1, 2]*x2 + dFdx[1, 3]*x3 + dFdx[1, 4]*x4 + dFdx[1, 5]*x5 + dFdx[1, 6]*x6 + dFdx[1, 7]*x7 + dFdx[1, 8]*x8, 
                                dFdx[2, 0]*x0 + dFdx[2, 1]*x1 + dFdx[2, 2]*x2 + dFdx[2, 3]*x3 + dFdx[2, 4]*x4 + dFdx[2, 5]*x5 + dFdx[2, 6]*x6 + dFdx[2, 7]*x7 + dFdx[2, 8]*x8], 
                                [dFdx[0, 0]*x9 + dFdx[0, 1]*x10 + dFdx[0, 2]*x11 + dFdx[0, 3]*x12 + dFdx[0, 4]*x13 + dFdx[0, 5]*x14 + dFdx[0, 6]*x15 + dFdx[0, 7]*x16 + dFdx[0, 8]*x17, 
                                dFdx[1, 0]*x9 + dFdx[1, 1]*x10 + dFdx[1, 2]*x11 + dFdx[1, 3]*x12 + dFdx[1, 4]*x13 + dFdx[1, 5]*x14 + dFdx[1, 6]*x15 + dFdx[1, 7]*x16 + dFdx[1, 8]*x17, 
                                dFdx[2, 0]*x9 + dFdx[2, 1]*x10 + dFdx[2, 2]*x11 + dFdx[2, 3]*x12 + dFdx[2, 4]*x13 + dFdx[2, 5]*x14 + dFdx[2, 6]*x15 + dFdx[2, 7]*x16 + dFdx[2, 8]*x17], 
                                [dFdx[0, 0]*x18 + dFdx[0, 1]*x19 + dFdx[0, 2]*x20 + dFdx[0, 3]*x21 + dFdx[0, 4]*x22 + dFdx[0, 5]*x23 + dFdx[0, 6]*x24 + dFdx[0, 7]*x25 + dFdx[0, 8]*x26, 
                                dFdx[1, 0]*x18 + dFdx[1, 1]*x19 + dFdx[1, 2]*x20 + dFdx[1, 3]*x21 + dFdx[1, 4]*x22 + dFdx[1, 5]*x23 + dFdx[1, 6]*x24 + dFdx[1, 7]*x25 + dFdx[1, 8]*x26, 
                                dFdx[2, 0]*x18 + dFdx[2, 1]*x19 + dFdx[2, 2]*x20 + dFdx[2, 3]*x21 + dFdx[2, 4]*x22 + dFdx[2, 5]*x23 + dFdx[2, 6]*x24 + dFdx[2, 7]*x25 + dFdx[2, 8]*x26]])
    print(local_hessian, local_hessian1, dFdx@d2Psi_d2F@dFdx.T) """
else:
    dshape1 = make_vector('dshape1', 3)
    dshape2 = make_vector('dshape2', 3)
    d2psid2f = make_matrix('d2psid2f', 9, 9)
    dfdx1 = Matrix([[dshape1[0], 0, 0, dshape1[1], 0, 0, dshape1[2], 0, 0],
                    [0, dshape1[0], 0, 0, dshape1[1], 0, 0, dshape1[2], 0],
                    [0, 0, dshape1[0], 0, 0, dshape1[1], 0, 0, dshape1[2]]])
    dfdx2 = Matrix([[dshape2[0], 0, 0, dshape2[1], 0, 0, dshape2[2], 0, 0],
                    [0, dshape2[0], 0, 0, dshape2[1], 0, 0, dshape2[2], 0],
                    [0, 0, dshape2[0], 0, 0, dshape2[1], 0, 0, dshape2[2]]])
    d2psid2x = dfdx1 @ d2psid2f @ dfdx2.T

    code = ''
    pair, reduced = cse(d2psid2x)
    for p in pair:
        code += pycode(p[0]) + ' = ' + pycode(p[1]) + ';\n'
    for i, r in enumerate(reduced):
        code += '\tout[' + str(i) + '] = ' + pycode(r) + ';'

    print(code)
