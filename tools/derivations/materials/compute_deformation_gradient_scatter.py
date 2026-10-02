import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *

dNdX = make_matrix('dNdX', 3, 3)
dNdnat = make_matrix('dNdnat', 27, 3)
x = make_matrix('x', 27, 3)

f = ((dNdnat @ dNdX).transpose() @ x).reshape(9, 1)

print(f[1])
