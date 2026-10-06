import numpy as np


# Alternating outer/inner radii make a simple, filled five-point star,
# not the self-intersecting pentagram obtained by joining only the five tips.
ANGLES = np.pi / 2.0 + np.arange(10) * np.pi / 5.0
RADII = np.tile([0.075, 0.035], 5)
VERTICES = np.array([0.405, 0.110]) + RADII[:, None] * np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
