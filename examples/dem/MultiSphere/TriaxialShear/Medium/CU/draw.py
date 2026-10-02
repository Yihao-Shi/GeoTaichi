#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

start_num=0
end_num=47
def macro(path):
    q=[]
    p=[]
    qp=[]
    evol=[]
    epsilon=[]
    vol0 = 0.
    leng0 = 0.

    for printNum in range(start_num, end_num):
        if printNum==start_num:
            data = np.load(path+'walls/DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)
            down_position = (data["point1"][0][2]+data["point2"][0][2]+data["point3"][0][2]+data["point1"][1][2]+data["point2"][1][2]+data["point3"][1][2])/6.
            up_position = (data["point1"][2][2]+data["point2"][2][2]+data["point3"][2][2]+data["point1"][3][2]+data["point2"][3][2]+data["point3"][3][2])/6.
            left_position = (data["point1"][4][0]+data["point2"][4][0]+data["point3"][4][0]+data["point1"][5][0]+data["point2"][5][0]+data["point3"][5][0])/6.
            right_position = (data["point1"][6][0]+data["point2"][6][0]+data["point3"][6][0]+data["point1"][7][0]+data["point2"][7][0]+data["point3"][7][0])/6.
            front_position = (data["point1"][8][1]+data["point2"][8][1]+data["point3"][8][1]+data["point1"][9][1]+data["point2"][9][1]+data["point3"][9][1])/6.
            back_position = (data["point1"][10][1]+data["point2"][10][1]+data["point3"][10][1]+data["point1"][11][1]+data["point2"][11][1]+data["point3"][11][1])/6.
            vol0=(up_position-down_position)*(right_position-left_position)*(back_position-front_position)
            leng0 = up_position-down_position
            leng = up_position-down_position
            wexx=200000 
            weyy=200000  
            wezz=200000 
        else:
            data = np.load(path+'contacts/DEMContactPW{0:06d}.npz'.format(printNum), allow_pickle=True)
            end2=data["end2"][np.linalg.norm(data["normal_force"], axis=1) > 0.]
            fn=data["normal_force"][np.linalg.norm(data["normal_force"], axis=1) > 0.]
            ft=data["tangential_force"][np.linalg.norm(data["normal_force"], axis=1) > 0.]
        
            down_force = (np.sum(fn[end2==0],axis=0) + np.sum(fn[end2==1],axis=0)+np.sum(ft[end2==0],axis=0) + np.sum(ft[end2==1],axis=0))[2]
            up_force = -(np.sum(fn[end2==2],axis=0) + np.sum(fn[end2==3],axis=0)+np.sum(ft[end2==2],axis=0) + np.sum(ft[end2==3],axis=0))[2]
            left_force = (np.sum(fn[end2==4],axis=0) + np.sum(fn[end2==5],axis=0)+np.sum(ft[end2==4],axis=0) + np.sum(ft[end2==5],axis=0))[0]
            right_force = -(np.sum(fn[end2==6],axis=0) + np.sum(fn[end2==7],axis=0)+np.sum(ft[end2==6],axis=0) + np.sum(ft[end2==7],axis=0))[0]
            front_force = (np.sum(fn[end2==8],axis=0) + np.sum(fn[end2==9],axis=0)+np.sum(ft[end2==8],axis=0) + np.sum(ft[end2==9],axis=0))[1]
            back_force = -(np.sum(fn[end2==10],axis=0) + np.sum(fn[end2==11],axis=0)+np.sum(ft[end2==10],axis=0) + np.sum(ft[end2==11],axis=0))[1]
        
            data = np.load(path+'walls/DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)
            down_position = (data["point1"][0][2]+data["point2"][0][2]+data["point3"][0][2]+data["point1"][1][2]+data["point2"][1][2]+data["point3"][1][2])/6.
            up_position = (data["point1"][2][2]+data["point2"][2][2]+data["point3"][2][2]+data["point1"][3][2]+data["point2"][3][2]+data["point3"][3][2])/6.
            left_position = (data["point1"][4][0]+data["point2"][4][0]+data["point3"][4][0]+data["point1"][5][0]+data["point2"][5][0]+data["point3"][5][0])/6.
            right_position = (data["point1"][6][0]+data["point2"][6][0]+data["point3"][6][0]+data["point1"][7][0]+data["point2"][7][0]+data["point3"][7][0])/6.
            front_position = (data["point1"][8][1]+data["point2"][8][1]+data["point3"][8][1]+data["point1"][9][1]+data["point2"][9][1]+data["point3"][9][1])/6.
            back_position = (data["point1"][10][1]+data["point2"][10][1]+data["point3"][10][1]+data["point1"][11][1]+data["point2"][11][1]+data["point3"][11][1])/6.
    
    
            down_pressure=down_force/(right_position-left_position)/(back_position-front_position)
            up_pressure=up_force/(right_position-left_position)/(back_position-front_position)
            left_pressure=left_force/(up_position-down_position)/(back_position-front_position)
            right_pressure=right_force/(up_position-down_position)/(back_position-front_position)
            front_pressure=front_force/(up_position-down_position)/(right_position-left_position)
            back_pressure=back_force/(up_position-down_position)/(right_position-left_position)
    
            wexx=0.5*(left_pressure+right_pressure)
            weyy=0.5*(front_pressure+back_pressure)
            wezz=0.5*(down_pressure+up_pressure)
    
        
            leng = up_position-down_position
    
        p.append(1./3.*(wexx+weyy+wezz))
        q.append(wezz - 0.5 * (weyy + wexx))
        qp.append((wezz - 0.5 * (weyy + wexx))/(1./3.*(wexx+weyy+wezz)))
        epsilon.append(math.log(leng0/leng))
        print(wexx, weyy, wezz)
    return p, q, qp, epsilon
    
pm, qm, qpm, epsilonm = macro('')
plt.plot(epsilonm, qpm)
plt.xlim([0, 0.5])
plt.show()

plt.plot(pm, qm)
plt.show()

