import bootstrap
import numpy as np
from igakit.cad import *

def test_grid():
    N = (1, 2, 3, 4)
    P = (1, 2, 3, 4, 5)
    L = ([0,1], [1,2], [2,3])
    W = (True, False)
    for n1, n2, n3 in zip(N, N, N): 
        for p1, p2, p3 in zip(P, P, P):
            C1 = range(-p1+0, p1)
            C2 = range(-p2+0, p2)
            C3 = range(-p3+0, p3)
            for c1, c2, c3 in zip (C1, C2, C2):
                for w in W:
                    m = grid(
                        (n1, n2, n3),
                        (p1, p2, p3),
                        (c1, c2, c3),
                        limits=L,
                        wrap=w,
                        )
                    u = [(a+b)/2.0 for a, b in L]
                    x = m(*u)
                    assert np.allclose(x, u)
                    u = [a for a, b in L]
                    x = m(*u)
                    assert np.allclose(x, u)
                    u = [b for a, b in L]
                    x = m(*u)
                    assert np.allclose(x, u)

if __name__ == '__main__':
    test_grid()
