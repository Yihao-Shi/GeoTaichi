from quaternion import Quaternion
import numpy as np
Q = Quaternion(ToVector=[1., 0., 0.])

position = np.array([1, 2, 3])
pos = np.array([[1, 2, 3], [2, 3, 4]])
print(Q.rotate(pos[0:1,:]))