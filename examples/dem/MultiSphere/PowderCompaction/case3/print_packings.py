#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, default=0)
parser.add_argument('-f', type=int, default=0)
args = parser.parse_args()

if args.p==0:
    path = "DEM_Generation"
    name = "pack_initial"
    current_num = 30
else:
    path = "DEM_Compression"
    name = "pack_final"
    current_num = 10

pp_contact = np.load(os.path.join(path, f'contacts/DEMContactPP{current_num:06d}.npz'))
particle_data = np.load(os.path.join(path, f'particles/DEMParticle{current_num:06d}.npz'))
pos_data = particle_data["position"]
radius_data = particle_data["radius"]

print('#', "Writing sphere(s) into files ......")
if args.f==0:
    np.savetxt(name+'.txt', np.column_stack((pos_data, radius_data)), header="     PositionX            PositionY                PositionZ            Radius", delimiter=" ")
else:
    df = pd.DataFrame({
        "id": np.arange(0, radius_data.shape[0], 1),
        "particle_type": np.where(radius_data < 3.2e-6, "A", "B"),
        "x": pos_data[:,0],
        "y": pos_data[:,1],
        "z": pos_data[:,2],
        "r": radius_data
    })
    df.to_csv(name+'.csv', index=False)
