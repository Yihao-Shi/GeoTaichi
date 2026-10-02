#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Plot consolidation wall pressures and velocities.")
parser.add_argument("--case-dir", default=str(CASE_DIR), help="Simulation directory containing walls/.")
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--end-frame", type=int, default=10, help="Inclusive end frame.")
arguments = parser.parse_args()
case_dir = Path(arguments.case_dir).expanduser().resolve()
print(case_dir)

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

down_vel=[]
up_vel=[]
left_vel=[]
right_vel=[]
front_vel=[]
back_vel=[]

void_ratio=[]
packing_fraction=[]

vol0 = 0.

q=[]
p=[]

time=[]

start_num = arguments.start_frame
end_num = arguments.end_frame

end_num += 1

for printNum in range(start_num, end_num):
    data = np.load(case_dir / 'walls' / 'DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)

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

    down_velocity = (data["velocity"][0][2]+data["velocity"][1][2])/2.
    up_velocity = (data["velocity"][2][2]+data["velocity"][3][2])/2.
    left_velocity = (data["velocity"][4][0]+data["velocity"][5][0])/2.
    right_velocity = (data["velocity"][6][0]+data["velocity"][7][0])/2.
    front_velocity = (data["velocity"][8][1]+data["velocity"][9][1])/2.
    back_velocity = (data["velocity"][10][1]+data["velocity"][11][1])/2.

    if printNum==end_num-1:
        vol0=(up_position-down_position)*(right_position-left_position)*(back_position-front_position)
    
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

    # print(down_force/(right_position-left_position)/(back_position-front_position), 
    #       up_force/(right_position-left_position)/(back_position-front_position), 
    #       left_force/(up_position-down_position)/(back_position-front_position), 
    #       right_force/(up_position-down_position)/(back_position-front_position), 
    #       front_force/(up_position-down_position)/(right_position-left_position), 
    #       back_force/(up_position-down_position)/(right_position-left_position))

    down_vel.append(down_velocity)
    up_vel.append(up_velocity)
    left_vel.append(left_velocity)
    right_vel.append(right_velocity)
    front_vel.append(front_velocity)
    back_vel.append(back_velocity)

    time.append(data["t_current"])

print('initial_z = ',up_position-down_position)

plt.plot(time, down_pressure, label='down')
plt.plot(time, up_pressure, label='up')
plt.plot(time, left_pressure, label='left')
plt.plot(time, right_pressure, label='right')
plt.plot(time, front_pressure, label='front')
plt.plot(time, back_pressure, label='back')
plt.ylabel('Pressure (Pa)')
plt.xlabel("Time (s)")
plt.legend()
# plt.ylim([100000, 300000])
plt.show()
plt.close()


plt.plot(time, down_vel, label='down')
plt.plot(time, up_vel, label='up')
plt.plot(time, left_vel, label='left')
plt.plot(time, right_vel, label='right')
plt.plot(time, front_vel, label='front')
plt.plot(time, back_vel, label='back')
plt.ylabel('Vel (m/s)')
plt.xlabel("Time (s)")
plt.legend()
plt.show()

