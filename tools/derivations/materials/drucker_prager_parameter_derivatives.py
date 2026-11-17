import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))


from tools.derivations.symfunc import *


fai = make_scalar('fai')
c = make_scalar('c')

def yield1():
    f1 = 6 * sin(fai) / (sqrt(3) * (3. - sin(fai)))
    f2 = 6 * c * cos(fai) / (sqrt(3) * (3. - sin(fai)))

    print(simplify(f1.diff(fai)))
    print(simplify(f2.diff(fai)))
    print(simplify(f2.diff(c)))

def yield2():
    f1 = 6 * sin(fai) / (sqrt(3) * (3. + sin(fai)))
    f2 = 6 * c * cos(fai) / (sqrt(3) * (3. + sin(fai)))

    print(simplify(f1.diff(fai)))
    print(simplify(f2.diff(fai)))
    print(simplify(f2.diff(c)))

def yield3():
    f1 = 3 * sin(fai) / sqrt(9. + 3. * sin(fai) * sin(fai))
    f2 = 3 * c * cos(fai) / sqrt(9. + 3. * sin(fai) * sin(fai))

    print(simplify(f1.diff(fai)))
    print(simplify(f2.diff(fai)))
    print(simplify(f2.diff(c)))

yield3()
