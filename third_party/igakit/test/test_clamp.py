import numpy as np
from igakit.cad import circle, Pi

def make_crv(p,u):
    c = circle(radius=1, angle=Pi/2)
    c.rotate(Pi/4)
    c.elevate(0,p-2)
    c.refine(0,u)
    return c

def check_crv(c):
    u0, u1 = c.breaks(0)[[0,-1]]
    u = np.linspace(u0,u1,100)
    x, y, z = c(u).T
    r = np.hypot(x,y)
    return np.allclose(r, 1)

def test_clamp():
    for p in range(2,6):
        for u in ([],[0.5],[1/3.0,2/3.0],[0.1,0.9]):
            c = make_crv(p,u)
            check_crv(c)
            for continuity in range(c.degree[0]):
                for side in (0, 1, None):
                    cc = c.copy()
                    cc.unclamp(0, continuity=continuity, side=side)
                    check_crv(cc)
                    cc.clamp(0, side=side)
                    check_crv(cc)
                    cc.clamp(0)
                    check_crv(cc)
                    assert np.allclose(cc.knots[0], c.knots[0])
                    assert np.allclose(cc.array,    c.array)

if __name__ == '__main__':
    test_clamp()
