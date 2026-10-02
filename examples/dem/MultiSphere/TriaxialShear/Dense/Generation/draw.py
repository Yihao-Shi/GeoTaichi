#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

for i in range(2, 3):
    data = np.load('particles/DEMParticle{0:06d}.npz'.format(i), allow_pickle=True)
    pos=data["position"]
    cf=data["contact_force"]
    for j in range(pos.shape[0]):
        print(j, pos[j], cf[j])
    print('\n')
