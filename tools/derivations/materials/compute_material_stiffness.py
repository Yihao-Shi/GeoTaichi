import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *

ishape = make_vector('ishape', 3)
jshape = make_vector('jshape', 3)
k = make_matrix('k', 3, 3)
k_33 = make_scalar('k_33')
k_44 = make_scalar('k_44')
k_55 = make_scalar('k_55')


iB = Matrix([[ishape[0], 0, 0], [0, ishape[1], 0], [0, 0, ishape[2]],
             [ishape[1], ishape[0], 0], [0, ishape[2], ishape[1]], [ishape[2], 0, ishape[0]]]).transpose()
jB = Matrix([[jshape[0], 0, 0], [0, jshape[1], 0], [0, 0, jshape[2]],
             [jshape[1], jshape[0], 0], [0, jshape[2], jshape[1]], [jshape[2], 0, jshape[0]]])
K = Matrix([[k[0,0], k[0,1], k[0,2], 0, 0, 0], [k[1,0], k[1,1], k[1,2], 0, 0, 0], [k[2,0], k[2,1], k[2,2], 0, 0, 0],
            [0, 0, 0, k_33, 0, 0], [0, 0, 0, 0, k_44, 0], [0, 0, 0, 0, 0, k_55]])

total = iB@K@jB
print(0, 0, total[0, 0])
print(0, 1, total[0, 1])
print(0, 2, total[0, 2])
print(1, 0, total[1, 0])
print(1, 1, total[1, 1])
print(1, 2, total[1, 2])
print(2, 0, total[2, 0])
print(2, 1, total[2, 1])
print(2, 2, total[2, 2])
