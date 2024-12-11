from third_party.geomdl import NURBS
from third_party.geomdl import operations
from third_party.geomdl.visualization import VisMPL

from math import sqrt


# Control points
ctrlpts = [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

# Generate surface
surf = NURBS.Curve()
surf.degree = 2
surf.ctrlpts = ctrlpts
surf.weights = [1.0, 0.5*sqrt(2), 1.0]
surf.knotvector = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
surf.sample_size = 30

# Set visualization component
#surf.vis = VisMPL.VisCurve3D()

# Refine knot vectors
operations.refine_knotvector(surf, [1])

# Visualize
#surf.render()