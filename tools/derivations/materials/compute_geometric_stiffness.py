import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *

ishape = make_vector('ishape', 3)
jshape = make_vector('jshape', 3)
sigma = make_matrix('sigma', 3, 3)

iB = Matrix([[ishape[0], 0, 0], [0, ishape[1], 0], [0, 0, ishape[2]]]).transpose()
jB = Matrix([[jshape[0], 0, 0], [0, jshape[1], 0], [0, 0, jshape[2]]])


total = outer_prod(ishape.T@sigma, jshape)
print(0, 0, total[0, 0])
print(0, 1, total[0, 1])
print(0, 2, total[0, 2])
print(1, 0, total[1, 0])
print(1, 1, total[1, 1])
print(1, 2, total[1, 2])
print(2, 0, total[2, 0])
print(2, 1, total[2, 1])
print(2, 2, total[2, 2])
