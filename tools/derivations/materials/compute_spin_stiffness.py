import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *

ishape = make_vector('ishape', 2)
jshape = make_vector('jshape', 2)
sigma = make_vector('sigma', 3)

iB = Matrix([[ishape[0], 0], [0, ishape[1]], [ishape[1], ishape[0]]]).transpose()
jB = Matrix([[0, 0], [0, 0], [0.5 * jshape[1], -0.5 * jshape[0]]])
K = Matrix([[2. * sigma[0], 0., 2. * sigma[2]], [0., 2. * sigma[1], -2. * sigma[2]], [sigma[2], sigma[2], sigma[1] - sigma[0]]])

total = iB@K@jB
print(0, 0, total[0, 0])
print(0, 1, total[0, 1])
print(1, 0, total[1, 0])
print(1, 1, total[1, 1])


""" ishape = make_vector('ishape', 3)
jshape = make_vector('jshape', 3)
sigma = make_vector('sigma', 6)

iB = Matrix([[ishape[0], 0, 0], [0, ishape[1], 0], [0, 0, ishape[2]],
             [ishape[1], ishape[0], 0], [0, ishape[2], ishape[1]], [ishape[2], 0, ishape[0]]]).transpose()
jB = Matrix([[0, 0.5 * jshape], [0, 0], [0.5 * jshape[1], -0.5 * jshape[0]]])
K = Matrix([[2. * sigma[0], 0., 0., sigma[3], 0., sigma[5]], [0., 2. * sigma[1], 0., sigma[3], sigma[4], 0.], [0., 0., 2. * sigma[2], 0., sigma[4], sigma[5]], 
            [sigma[2], sigma[2], 0., 0.5 * (sigma[1] + sigma[0]), 0., 0.], [0., sigma[4], sigma[4], 0., 0.5 * (sigma[1] + sigma[2]), 0.], [sigma[5], 0., sigma[5], 0., 0., 0.5 * (sigma[0] + sigma[2])]])

total = iB@K@jB
print(0, 0, total[0, 0])
print(0, 1, total[0, 1])
print(1, 0, total[1, 0])
print(1, 1, total[1, 1])
 """
