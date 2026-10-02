#!/usr/bin/env python
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Plot powder-compaction stress and packing fraction.")
parser.add_argument("--case-dir", default=str(CASE_DIR), help="Simulation directory containing particles, walls, and contacts.")
parser.add_argument("--output-dir", default=str(CASE_DIR))
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--end-frame", type=int, default=4, help="Exclusive end frame.")
arguments = parser.parse_args()
case_dir = Path(arguments.case_dir).expanduser().resolve()
output_dir = Path(arguments.output_dir).expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)

params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 3,
             'lines.markersize': 10,
             'xtick.labelsize': 26,
             'ytick.labelsize': 26,
             'xtick.major.pad': 12,
             'ytick.major.pad': 12,
             "axes.labelpad":   8,
             'legend.fontsize': 26,
             'figure.figsize': [12, 9],
             'font.family': 'serif',
             'text.usetex': False,
             'font.serif': 'Arial',
             'savefig.dpi': 300
         }
rcParams.update(params)

         
color = [(0/255, 0/255, 0/255), 
         (255/255, 0/255, 0/255), 
         (94/255, 114/255, 255/255), 
         (0/255, 128/255, 0/255)]


pressure=[]
love_stress = []
time=[]
fraction=[]

particle = np.load(case_dir / 'particles' / 'DEMParticle{0:06d}.npz'.format(arguments.start_frame), allow_pickle=True)
radius = particle["radius"]
vol = np.sum(4./3.*math.pi*radius**3)


start_num = arguments.start_frame
end_num = arguments.end_frame
for printNum in range(start_num, end_num):
    
    data = np.load(case_dir / 'walls' / 'DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)
    zcoord = data["point1"][2][2] - data["point1"][0][2]
    
    wall_planes = {
    0: (0, 0.0, np.array([0.0, 0.0, 1.0])),
    1: (0, 0.0, np.array([0.0, 0.0, 1.0])),
    2: (0, zcoord, np.array([ 0.0, 0.0, -1.0])),
    3: (0, zcoord, np.array([ 0.0, 0.0, -1.0]))
    }
    wall_ids = np.array(list(wall_planes.keys()), dtype=int)
    wall_normals = np.array([wall_planes[k][2] for k in wall_ids])  # (Nw, 3)
    wall_d       = np.array([wall_planes[k][1] for k in wall_ids])  # (Nw,)

    
    particle = np.load(case_dir / 'particles' / 'DEMParticle{0:06d}.npz'.format(arguments.start_frame), allow_pickle=True)
    pos = particle["position"]
    
    total_v = zcoord*3.481956e-8
    fraction.append((vol)/(total_v))
    down_force = abs(data["contact_force"][2][2]+data["contact_force"][3][2])
    pressure.append(down_force/3.481956e-8)
    time.append(data["t_current"])
    
    data = np.load(case_dir / 'contacts' / 'DEMContactPW{0:06d}.npz'.format(printNum), allow_pickle=True)
    contact_num = data["contact_num"][-1]
    particle_id = data["end1"][:contact_num]
    wall_id = data["end2"][:contact_num]
    mask = (wall_id == 2) | (wall_id == 3)
    wall_normal_force = data["normal_force"][:contact_num]
    wall_tangential_force = data["tangential_force"][:contact_num]
    wall_forces = wall_normal_force + wall_tangential_force

plt.plot(time, pressure, linestyle='-', marker='o', color=color[0], label="Wall stress")
plt.plot(time, np.repeat(1000000, len(time)), color=color[2], label="Target stress")
plt.xlabel('Time, $T$ (s)')
plt.ylabel('$\\sigma$ / $\\sigma_{target}$')
plt.legend(frameon=False)
plt.savefig(output_dir / "stress_strain.png")
plt.close()

plt.plot(time, fraction, linestyle='-', color=color[0], label="case 1")
plt.xlabel('Time, $T$ (s)')
plt.ylabel('$\\rho$')
plt.legend(frameon=False)
plt.savefig(output_dir / "fraction.png")
plt.close()
