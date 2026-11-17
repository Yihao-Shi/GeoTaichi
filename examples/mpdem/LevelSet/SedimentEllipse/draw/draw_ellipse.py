import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

f = polysuperellipsoid(xrad1=0.005, yrad1=0.0025, zrad1=0.0025, epsilon_e=1.0, epsilon_n=1.0).grids(space=0.00025)
f.save(os.path.join(ROOT, "assets", "mesh", "MPDEM", "ellipsoid.stl"), samples=2502, sparse=False)
