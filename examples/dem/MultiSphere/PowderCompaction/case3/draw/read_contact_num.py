#!/usr/bin/env python
import numpy as np
import os

path = "DEM_Generation"


start_num = 20
end_num = 33
box_point = [3.35e-6, 3.35e-6, 3.35e-6]
box_size = [2 * 93.3e-6, 2 * 93.3e-6, 2 * 93.3e-6]

for printNum in range(start_num, end_num + 1):
    k1 = 0  # 统计超出box的颗粒数量
    k2 = 0  # 统计box内部的颗粒数量

    pp_contact = np.load(os.path.join(path, f"contacts/DEMContactPP{printNum:06d}.npz"))
    # pw_contact = np.load(os.path.join(path, f'contacts/DEMContactPW{printNum:06d}.npz'))
    particle_data = np.load(os.path.join(path, f"particles/DEMParticle{printNum:06d}.npz"))
    pos_data = particle_data["position"]
    radius_data = particle_data["radius"]
    pp_force_filter = np.linalg.norm(pp_contact["normal_force"], axis=1) > 0

    for x, y, z in particle_data["position"]:
        if (
            x > box_point[0]
            or y > box_point[1]
            or z > box_point[2]
            or x < box_point[0] + box_size[0]
            or y < box_point[1] + box_size[1]
            or z < box_point[2] + box_size[2]
        ):
            k2 += 1

    for x, y, z in particle_data["position"]:
        if (
            x < box_point[0]
            or y < box_point[1]
            or z < box_point[2]
            or x > box_point[0] + box_size[0]
            or y > box_point[1] + box_size[1]
            or z > box_point[2] + box_size[2]
        ):
            k1 += 1
    # pw_force_filter = np.linalg.norm(pw_contact["normal_force"], axis=1) > 0
    t = pp_contact["t_current"]
    end1 = pp_contact["end1"][pp_force_filter]
    end2 = pp_contact["end2"][pp_force_filter]
    overlap = radius_data[end1] + radius_data[end2] - np.linalg.norm(pos_data[end1] - pos_data[end2], axis=1)
    max_gap = np.max(overlap[overlap > 0])
    sum_gap = np.sum(overlap[overlap > 0])

    print("时间：", t)
    print("有效接触数目：", len(end1))
    print("超出数目：", k1)
    print("域内数目：", k2)
    print("最大重叠量：", max_gap)
    print("总重叠量：", sum_gap)
    print("\n")

    if max_gap < 1e-10:
        break
