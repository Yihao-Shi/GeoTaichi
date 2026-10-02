import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *

stress = make_matrix('stress', 3, 3)
gradv = make_matrix('gradv', 3, 3)  

""" oemga = 0.5 * (gradv - gradv.transpose())

jaumann = -oemga @ stress + stress @ oemga

print(jaumann[0, 0])
print(jaumann[0, 1])
print(jaumann[0, 2])
print(jaumann[1, 0])
print(jaumann[1, 1])
print(jaumann[1, 2])
print(jaumann[2, 0])
print(jaumann[2, 1])
print(jaumann[2, 2]) """

sigma = make_vector('sigma', 6)
dw = make_vector('dw', 3)
domega = Matrix([gradv[1, 0] - gradv[0, 1], gradv[2, 1] - gradv[1, 2], gradv[0, 2] - gradv[2, 0]])
W = Matrix([[0., -dw[0], dw[2]], [dw[0], 0., -dw[1]], [-dw[2], dw[1], 0.]])
Omega = Matrix([[0., -domega[0], domega[2]], [domega[0], 0., -domega[1]], [-domega[2], domega[1], 0.]])
stress = Matrix([[sigma[0], sigma[3], sigma[5]], [sigma[3], sigma[1], sigma[4]], [sigma[5], sigma[4], sigma[2]]])

jaumann = -W @ stress + stress @ W
Jaumann = -Omega @ stress + stress @ Omega

print(jaumann[0, 0])
print(jaumann[0, 1])
print(jaumann[0, 2])
print(jaumann[1, 0])
print(jaumann[1, 1])
print(jaumann[1, 2])
print(jaumann[2, 0])
print(jaumann[2, 1])
print(jaumann[2, 2])
           
