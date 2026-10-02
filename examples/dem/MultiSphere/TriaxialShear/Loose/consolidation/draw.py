#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

down_pressure=[]
up_pressure=[]
left_pressure=[]
right_pressure=[]
front_pressure=[]
back_pressure=[]

down_area=[]
up_area=[]
left_area=[]
right_area=[]
front_area=[]
back_area=[]
vol0 = 0.
targs = 200000

q=[]
p=[]

time=[]

start_num=0
end_num=15
for printNum in range(start_num, end_num):
    data = np.load('walls/DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)
    
    down_force = -(data["contact_force"][0][2]+data["contact_force"][1][2])
    up_force = (data["contact_force"][2][2]+data["contact_force"][3][2])
    left_force = -(data["contact_force"][4][0]+data["contact_force"][5][0])
    right_force = (data["contact_force"][6][0]+data["contact_force"][7][0])
    front_force = -(data["contact_force"][8][1]+data["contact_force"][9][1])
    back_force = (data["contact_force"][10][1]+data["contact_force"][11][1])
    
    down_position = (data["point1"][0][2]+data["point2"][0][2]+data["point3"][0][2]+data["point1"][1][2]+data["point2"][1][2]+data["point3"][1][2])/6.
    up_position = (data["point1"][2][2]+data["point2"][2][2]+data["point3"][2][2]+data["point1"][3][2]+data["point2"][3][2]+data["point3"][3][2])/6.
    left_position = (data["point1"][4][0]+data["point2"][4][0]+data["point3"][4][0]+data["point1"][5][0]+data["point2"][5][0]+data["point3"][5][0])/6.
    right_position = (data["point1"][6][0]+data["point2"][6][0]+data["point3"][6][0]+data["point1"][7][0]+data["point2"][7][0]+data["point3"][7][0])/6.
    front_position = (data["point1"][8][1]+data["point2"][8][1]+data["point3"][8][1]+data["point1"][9][1]+data["point2"][9][1]+data["point3"][9][1])/6.
    back_position = (data["point1"][10][1]+data["point2"][10][1]+data["point3"][10][1]+data["point1"][11][1]+data["point2"][11][1]+data["point3"][11][1])/6.
    
    if printNum==end_num-1:
        vol0=(up_position-down_position)*(right_position-left_position)*(back_position-front_position)
        print(up_position,down_position,right_position,left_position,back_position,front_position)
    
    down_area.append((right_position-left_position)*(back_position-front_position))
    up_area.append((right_position-left_position)*(back_position-front_position))
    left_area.append((up_position-down_position)*(back_position-front_position))
    right_area.append((up_position-down_position)*(back_position-front_position))
    front_area.append((up_position-down_position)*(right_position-left_position))
    back_area.append((up_position-down_position)*(right_position-left_position))
    
    down_pressure.append(down_force/(right_position-left_position)/(back_position-front_position))
    up_pressure.append(up_force/(right_position-left_position)/(back_position-front_position))
    left_pressure.append(left_force/(up_position-down_position)/(back_position-front_position))
    right_pressure.append(right_force/(up_position-down_position)/(back_position-front_position))
    front_pressure.append(front_force/(up_position-down_position)/(right_position-left_position))
    back_pressure.append(back_force/(up_position-down_position)/(right_position-left_position))
    time.append(data["t_current"])


plt.scatter(time, down_pressure)
plt.scatter(time, up_pressure)
plt.scatter(time, left_pressure)
plt.scatter(time, right_pressure)
plt.scatter(time, front_pressure)
plt.scatter(time, back_pressure)
# plt.ylim([100000, 300000])
plt.show()

data = np.load('particles/DEMParticle{0:06d}.npz'.format(end_num-1), allow_pickle=True)
particle_vol = 4./3.*math.pi*(np.power(data["radius"],3).sum())
print(particle_vol, vol0)
print(f"Initial Void Ratio: {(vol0-particle_vol)/particle_vol}")
