#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

pressure=[]
vol0 = 0.
targs = 200000

q=[]
p=[]

time=[]

start_num=0
end_num=7
for printNum in range(start_num, end_num):
    data = np.load('walls/DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)
    
    down_force = -(data["contact_force"][2][2]+data["contact_force"][3][2])
    pressure.append(down_force/1.296e-7)
    time.append(data["t_current"])


plt.scatter(time, pressure)
# plt.ylim([100000, 300000])
plt.show()
