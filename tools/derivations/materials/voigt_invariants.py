import sys
from pathlib import Path

import numpy as np
repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *

vstress = make_vector('vstrain', 6)
stress = Matrix([[vstress[0], vstress[3], vstress[5]], [vstress[3], vstress[1], vstress[4]], [vstress[5], vstress[4], vstress[2]]])
dev_stress = stress - 1./3. * Matrix([[trace(stress), 0., 0.], [0., trace(stress), 0.], [0., 0., trace(stress)]])

I2 = 0.5 * (trace(stress) * trace(stress) - trace(stress * stress))
I3 = det(stress)
J2 = 0.5 * double_dot(dev_stress, dev_stress)
J3 = det(dev_stress)

print("I2 = ", simplify(I2))
print("I3 = ", simplify(I3))
print("J2 = ", simplify(J2))
print("J3 = ", simplify(J3))

vstrain = make_vector('vstrain', 6)
strain = Matrix([[vstrain[0], 0.5 * vstrain[3], 0.5 * vstrain[5]], [0.5 * vstrain[3], vstrain[1], 0.5 * vstrain[4]], [0.5 * vstrain[5], 0.5 * vstrain[4], vstrain[2]]])
dev_strain = strain - 1./3. * Matrix([[trace(strain), 0., 0.], [0., trace(strain), 0.], [0., 0., trace(strain)]])

I2 = 0.5 * (trace(strain) * trace(strain) - trace(strain * strain))
I3 = det(strain)
J2 = 0.5 * double_dot(dev_strain, dev_strain)
J3 = det(dev_strain)

print("I2 = ", simplify(I2))
print("I3 = ", simplify(I3))
print("J2 = ", simplify(J2))
print("J3 = ", simplify(J3))

print(sqrt(4/3*J2))
