import os

import numpy as np

from src.mpm.Simulation import Simulation
from src.mpm.SoftParticleOutput import (
    pk1_to_cauchy,
    stress_to_tensor3,
    von_mises_stress,
    vtk_vector,
)
from src.utils.ObjectIO import DictIO
import src.utils.GlobalVariable as GlobalVariable
from third_party.pyevtk.hl import pointsToVTK, gridToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle


def CheckFirst(sims: Simulation, read_path, write_path, end_file):
    if not os.path.exists(read_path):
        raise EOFError(f"Invaild path: {read_path}")
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    if end_file == -1:
        if sims.current_print == 0:
            raise ValueError("Invalid end_file")
        end_file = sims.current_print
    return end_file


def _slice_vtk_data(data, mask):
    sliced = {}
    for name, value in data.items():
        if isinstance(value, tuple):
            sliced[name] = tuple(np.ascontiguousarray(component[mask]) for component in value)
        else:
            sliced[name] = np.ascontiguousarray(value[mask])
    return sliced


def _vtk_velocity(velocity):
    velocity = np.asarray(velocity)
    z = velocity[:, 2] if velocity.shape[1] == 3 else np.zeros(velocity.shape[0])
    return tuple(np.ascontiguousarray(component, dtype=np.float64) for component in (velocity[:, 0], velocity[:, 1], z))


import taichi as ti


class SmoothVar:
    def __init__(self, smooth_setting, particle_num):
        self.smooth_rad = smooth_setting["smooth_rad"]
        self.lower_bound = np.asarray(smooth_setting["lower_bound"], dtype=np.float64)
        self.upper_bound = np.asarray(smooth_setting["upper_bound"], dtype=np.float64)
        nsize = self.upper_bound - self.lower_bound
        extend = 0.1 * nsize
        self.lower_bound -= extend
        self.upper_bound += extend
        self.particle_num = particle_num
        self.ndim = GlobalVariable.DIMENSION
        self.ti_src_0 = ti.field(float, shape=self.particle_num)
        self.ti_src_2 = ti.Vector.field(2, float, shape=self.particle_num)
        self.ti_src_3 = ti.Vector.field(3, float, shape=self.particle_num)
        self.ti_src_6 = ti.Vector.field(6, float, shape=self.particle_num)
        self.ti_pos = ti.Vector.field(self.ndim, float, shape=self.particle_num)
        self.ti_m = ti.field(float, shape=self.particle_num)

        self.cell_num = np.astype(np.ceil((self.upper_bound - self.lower_bound) / self.smooth_rad), np.int32)
        self.ti_grid_m = ti.field(float, shape=self.cell_num)
        self.ti_grid_src_0 = ti.field(float, shape=self.cell_num)
        self.ti_grid_src_2 = ti.Vector.field(2, float, shape=self.cell_num)
        self.ti_grid_src_3 = ti.Vector.field(3, float, shape=self.cell_num)
        self.ti_grid_src_6 = ti.Vector.field(6, float, shape=self.cell_num)

    def reset(self):
        self.ti_src_0.fill(0)
        self.ti_src_2.fill(0)
        self.ti_src_3.fill(0)
        self.ti_src_6.fill(0)
        self.ti_pos.fill(0)
        self.ti_m.fill(0)
        self.ti_grid_src_0.fill(0)
        self.ti_grid_src_2.fill(0)
        self.ti_grid_src_3.fill(0)
        self.ti_grid_src_6.fill(0)
        self.ti_grid_m.fill(0)

    def run(self, position, weight, src):
        assert position.shape[0] == weight.shape[0] == src.shape[0]
        self.reset()
        ti_src = None
        ti_grid_src = None
        if src.ndim == 1:
            self.ti_src_0.from_numpy(src)
            ti_src = self.ti_src_0
            ti_grid_src = self.ti_grid_src_0
        elif src.ndim == 2:
            if src.shape[1] == 2:
                self.ti_src_2.from_numpy(src)
                ti_src = self.ti_src_2
                ti_grid_src = self.ti_grid_src_2
            elif src.shape[1] == 3:
                self.ti_src_3.from_numpy(src)
                ti_src = self.ti_src_3
                ti_grid_src = self.ti_grid_src_3
            elif src.shape[1] == 6:
                self.ti_src_6.from_numpy(src)
                ti_src = self.ti_src_6
                ti_grid_src = self.ti_grid_src_6
        self.ti_pos.from_numpy(position)
        self.ti_m.from_numpy(weight)

        @ti.kernel
        def smooth(
            inv_dx: float,
            x0: ti.types.vector(self.ndim, float),
            p_m: ti.template(),
            p_pos: ti.template(),
            p_src: ti.template(),
            g_m: ti.template(),
            g_src: ti.template(),
        ):
            for p in p_pos:  # Particle state update and scatter to grid (P2G)
                base = ((p_pos[p] - x0) * inv_dx - 0.5).cast(int)
                fx = (p_pos[p] - x0) * inv_dx - base.cast(float)
                p_mass = p_m[p]
                p_s = p_src[p]
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                if ti.static(self.ndim == 2):
                    for i, j in ti.static(ti.ndrange(3, 3)):
                        offset = ti.Vector([i, j])
                        weight = w[i][0] * w[j][1]
                        g_m[base + offset] += weight * p_mass
                        g_src[base + offset] += weight * p_mass * p_s
                else:
                    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                        offset = ti.Vector([i, j, k])
                        weight = w[i][0] * w[j][1] * w[k][2]
                        g_m[base + offset] += weight * p_mass
                        g_src[base + offset] += weight * p_mass * p_s

            for i in ti.grouped(g_m):
                if g_m[i] > 0:
                    g_src[i] /= g_m[i]

            for p in p_pos:
                base = ((p_pos[p] - x0) * inv_dx - 0.5).cast(int)
                fx = (p_pos[p] - x0) * inv_dx - base.cast(float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_stress = p_src[p] * 0.0
                if ti.static(self.ndim == 2):
                    for i, j in ti.static(ti.ndrange(3, 3)):
                        weight = w[i][0] * w[j][1]
                        new_stress += weight * g_src[base + ti.Vector([i, j])]
                else:
                    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                        weight = w[i][0] * w[j][1] * w[k][2]
                        new_stress += weight * g_src[base + ti.Vector([i, j, k])]
                p_src[p] = new_stress

        smooth(1.0 / self.smooth_rad, self.lower_bound, self.ti_m, self.ti_pos, ti_src, self.ti_grid_m, ti_grid_src)
        return ti_src.to_numpy()


def _make_particle_reference(position, particle_id):
    return np.ascontiguousarray(position), {int(pid): idx for idx, pid in enumerate(particle_id)}


def _reference_indices(position, particle_id, reference_position, reference_id_map):
    indices = np.empty(particle_id.shape[0], dtype=np.int64)
    new_positions = []
    for idx, pid in enumerate(particle_id):
        pid = int(pid)
        reference_idx = reference_id_map.get(pid)
        if reference_idx is None:
            reference_idx = reference_position.shape[0] + len(new_positions)
            reference_id_map[pid] = reference_idx
            new_positions.append(position[idx])
        indices[idx] = reference_idx
    if new_positions:
        reference_position = np.concatenate(
            (reference_position, np.asarray(new_positions, dtype=reference_position.dtype)),
            axis=0,
        )
    return reference_position, indices


def _rotation_from_quaternion(q):
    qx, qy, qz, qw = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _stress_to_tensor3(stress):
    return stress_to_tensor3(stress)


def _von_mises_stress(stress):
    return von_mises_stress(stress)


def write_soft_particle_point_vtk(printNum, read_path, write_path, kwargs):
    point_file = read_path + "/particles/LSMPMSoftPoint{0:06d}.npz".format(printNum)
    if not os.access(point_file, os.F_OK):
        return False

    point_info = np.load(point_file, allow_pickle=True)
    point_num = int(DictIO.GetEssential(point_info, "point_num"))
    if point_num == 0:
        return False

    position = np.ascontiguousarray(DictIO.GetEssential(point_info, "position"))[:point_num]
    active = np.ascontiguousarray(DictIO.GetAlternative(point_info, "active", np.ones(point_num, dtype=np.int32)))[
        :point_num
    ]
    output_active_only = DictIO.GetAlternative(kwargs, "write_active_soft_point_only", False)
    select = active > 0 if output_active_only else np.ones(point_num, dtype=bool)
    if not np.any(select):
        return False

    data = {}
    if DictIO.GetAlternative(kwargs, "write_bodyID", True):
        bodyID = np.ascontiguousarray(DictIO.GetEssential(point_info, "bodyID"))[:point_num]
        data["bodyID"] = np.ascontiguousarray(bodyID[select])
    if DictIO.GetAlternative(kwargs, "write_groupID", True):
        groupID = np.ascontiguousarray(DictIO.GetEssential(point_info, "groupID"))[:point_num]
        data["groupID"] = np.ascontiguousarray(groupID[select])
    if DictIO.GetAlternative(kwargs, "write_materialID", True):
        materialID = np.ascontiguousarray(DictIO.GetEssential(point_info, "materialID"))[:point_num]
        data["materialID"] = np.ascontiguousarray(materialID[select])
    if DictIO.GetAlternative(kwargs, "write_active", False):
        data["active"] = np.ascontiguousarray(active[select].astype(np.int32))
    if DictIO.GetAlternative(kwargs, "write_velocity", True):
        velocity = np.ascontiguousarray(DictIO.GetEssential(point_info, "velocity"))[:point_num]
        data["velocity"] = (
            np.ascontiguousarray(velocity[select, 0]),
            np.ascontiguousarray(velocity[select, 1]),
            np.ascontiguousarray(velocity[select, 2]),
        )
    if DictIO.GetAlternative(kwargs, "write_contact_force", True):
        contact_force = np.ascontiguousarray(DictIO.GetEssential(point_info, "contact_force"))[:point_num]
        data["contact_force"] = vtk_vector(contact_force[select])
    if DictIO.GetAlternative(kwargs, "write_external_force", True) and "external_force" in point_info:
        external_force = np.ascontiguousarray(DictIO.GetEssential(point_info, "external_force"))[:point_num]
        data["external_force"] = vtk_vector(external_force[select])
    if DictIO.GetAlternative(kwargs, "write_displacement", True) and "reference" in point_info:
        reference = np.ascontiguousarray(DictIO.GetEssential(point_info, "reference"))[:point_num]
        displacement = np.ascontiguousarray(position - reference)
        data["displacement"] = vtk_vector(displacement[select])
    if "stress" in point_info and "F" in point_info:
        first_piola = np.ascontiguousarray(DictIO.GetEssential(point_info, "stress"))[:point_num]
        deformation = np.ascontiguousarray(DictIO.GetEssential(point_info, "F"))[:point_num]
        cauchy_stress, _ = pk1_to_cauchy(first_piola, deformation)
        if DictIO.GetAlternative(kwargs, "write_von_mises_stress", True):
            data["von_mises_stress"] = von_mises_stress(cauchy_stress)[select]
        if DictIO.GetAlternative(kwargs, "write_stress_component", True):
            data["cauchy_stress_xx"] = np.ascontiguousarray(cauchy_stress[select, 0, 0])
            data["cauchy_stress_yy"] = np.ascontiguousarray(cauchy_stress[select, 1, 1])
            data["cauchy_stress_zz"] = np.ascontiguousarray(cauchy_stress[select, 2, 2])

    selected_position = np.ascontiguousarray(position[select])
    pointsToVTK(
        write_path + f"/GraphicLSMPMPoint{printNum:06d}",
        np.ascontiguousarray(selected_position[:, 0]),
        np.ascontiguousarray(selected_position[:, 1]),
        np.ascontiguousarray(selected_position[:, 2]),
        data=data,
    )
    return True


def write_soft_particle_surface_from_levelset(printNum, particle_info, read_path, write_path, kwargs):
    grid_file = read_path + "/particles/LSDEMGrid{0:06d}.npz".format(printNum)
    bounding_file = read_path + "/particles/BoundingBox{0:06d}.npz".format(printNum)
    if not os.access(grid_file, os.F_OK) or not os.access(bounding_file, os.F_OK):
        return False

    body_num = int(DictIO.GetEssential(particle_info, "body_num"))
    is_soft = np.ascontiguousarray(
        DictIO.GetAlternative(particle_info, "is_soft", np.zeros(body_num, dtype=np.int32))
    ).astype(bool)
    if not np.any(is_soft):
        return False

    try:
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("Soft-particle surface reconstruction requires scikit-image.") from exc

    grid_info = np.load(grid_file, allow_pickle=True)
    bounding_info = np.load(bounding_file, allow_pickle=True)
    distance_field = np.ascontiguousarray(DictIO.GetEssential(grid_info, "distance_field"))
    start_grid = np.ascontiguousarray(DictIO.GetEssential(bounding_info, "startGrid")).astype(np.int64)
    grid_num = np.ascontiguousarray(DictIO.GetEssential(bounding_info, "grid_num")).astype(np.int64)
    grid_space = np.ascontiguousarray(DictIO.GetEssential(bounding_info, "grid_space"))
    min_box = np.ascontiguousarray(DictIO.GetEssential(bounding_info, "min_box"))
    position = np.ascontiguousarray(DictIO.GetEssential(particle_info, "mass_center"))
    quanternion = np.ascontiguousarray(DictIO.GetEssential(particle_info, "quanternion"))
    groupID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "groupID"))
    materialID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "materialID"))

    all_vertices = []
    all_faces = []
    all_body = []
    all_group = []
    all_material = []
    vertex_offset = 0
    for bodyID in np.where(is_soft)[0]:
        gnum = grid_num[bodyID]
        if np.any(gnum < 2):
            continue
        node_count = int(np.prod(gnum))
        start = int(start_grid[bodyID])
        sdf = np.asarray(distance_field[start : start + node_count], dtype=np.float64)
        if sdf.shape[0] != node_count or np.min(sdf) > 0.0 or np.max(sdf) < 0.0:
            continue
        volume = sdf.reshape(tuple(gnum.tolist()), order="F")
        spacing = (float(grid_space[bodyID]),) * 3
        try:
            vertices, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=spacing)
        except ValueError:
            continue
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            continue
        local_vertices = vertices + min_box[bodyID]
        rotation = _rotation_from_quaternion(quanternion[bodyID])
        world_vertices = position[bodyID] + local_vertices @ rotation.T
        all_vertices.append(world_vertices)
        all_faces.append(faces.astype(np.int32) + vertex_offset)
        all_body.append(np.full(vertices.shape[0], bodyID, dtype=np.int32))
        all_group.append(np.full(vertices.shape[0], groupID[bodyID], dtype=np.int32))
        all_material.append(np.full(vertices.shape[0], materialID[bodyID], dtype=np.int32))
        vertex_offset += vertices.shape[0]

    if len(all_vertices) == 0:
        return False

    vertices = np.ascontiguousarray(np.vstack(all_vertices))
    faces = np.ascontiguousarray(np.vstack(all_faces).astype(np.int32))
    nface = int(faces.shape[0])
    pointData = {}
    if DictIO.GetAlternative(kwargs, "write_bodyID", True):
        pointData["bodyID"] = np.ascontiguousarray(np.concatenate(all_body))
    if DictIO.GetAlternative(kwargs, "write_groupID", True):
        pointData["groupID"] = np.ascontiguousarray(np.concatenate(all_group))
    if DictIO.GetAlternative(kwargs, "write_materialID", True):
        pointData["materialID"] = np.ascontiguousarray(np.concatenate(all_material))
    unstructuredGridToVTK(
        write_path + f"/GraphicLSMPMSoftSurface{printNum:06d}",
        np.ascontiguousarray(vertices[:, 0]),
        np.ascontiguousarray(vertices[:, 1]),
        np.ascontiguousarray(vertices[:, 2]),
        connectivity=np.ascontiguousarray(faces.flatten()),
        offsets=np.ascontiguousarray(np.arange(3, 3 * nface + 1, 3, dtype=np.int32)),
        cell_types=np.repeat(VtkTriangle.tid, nface),
        pointData=pointData,
    )
    return True


def write_vtk_file(sims: Simulation, start_file, end_file, read_path, write_path, kwargs):
    end_file = CheckFirst(sims, read_path, write_path, end_file)
    smooth_setting = DictIO.GetAlternative(kwargs, "smooth_setting", False)

    total_disp = DictIO.GetAlternative(kwargs, "total_displacement", False)
    smoother = None
    reference_position = None
    reference_id_map = None
    if total_disp or smooth_setting:
        particle_file0 = (read_path + "/particles/MPMParticle{0:06d}.npz").format(0)
        if not os.access(particle_file0, os.F_OK):
            raise ValueError(f"File {read_path}/particles/MPMParticle{0:06d}.npz does not exist!".format(0))
        particle_info = np.load(particle_file0, allow_pickle=True)
        position0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "position"))
        particle_id0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "particleID"))
        if total_disp:
            reference_position, reference_id_map = _make_particle_reference(position0, particle_id0)

        smoother = SmoothVar(smooth_setting, position0.shape[0])

    for printNum in range(start_file, end_file):
        data = {}
        particle_file = (read_path + "/particles/MPMParticle{0:06d}.npz").format(printNum)
        if not os.access(particle_file, os.F_OK):
            if write_soft_particle_point_vtk(printNum, read_path, write_path, kwargs):
                continue
            raise ValueError(f"File {read_path}/particles/MPMParticle{0:06d}.npz does not exist!")

        print((" MPM Postprocessing: Output VTK File" + str(printNum) + " ").center(71, "-"))
        particle_info = np.load(particle_file, allow_pickle=True)
        if printNum == start_file and not total_disp:
            position0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "position"))
            particle_id0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "particleID"))
            reference_position, reference_id_map = _make_particle_reference(position0, particle_id0)

        particle_id = np.ascontiguousarray(DictIO.GetEssential(particle_info, "particleID"))
        position = np.ascontiguousarray(DictIO.GetEssential(particle_info, "position"))
        posx = np.ascontiguousarray(position[:, 0], dtype=np.float64)
        posy = np.ascontiguousarray(position[:, 1], dtype=np.float64)
        posz = np.zeros(position.shape[0], dtype=np.float64)
        if GlobalVariable.DIMENSION == 3:
            posz = np.ascontiguousarray(position[:, 2], dtype=np.float64)

        weight = DictIO.GetEssential(particle_info, "mass")

        if DictIO.GetAlternative(kwargs, "write_bodyID", True):
            bodyID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "bodyID"))
            data.update({"bodyID": bodyID})
        if DictIO.GetAlternative(kwargs, "write_materialID", True):
            materialID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "materialID"))
            data.update({"materialID": materialID})
        if DictIO.GetAlternative(kwargs, "write_volume", True):
            volume = np.ascontiguousarray(DictIO.GetEssential(particle_info, "volume"))
            data.update({"volume": volume})
        if DictIO.GetAlternative(kwargs, "write_displacement", True):
            if reference_id_map is None:
                reference_position, reference_id_map = _make_particle_reference(position, particle_id)
            reference_position, indices = _reference_indices(
                position, particle_id, reference_position, reference_id_map
            )
            disp = position - reference_position[indices]
            dispx = np.ascontiguousarray(disp[:, 0], dtype=np.float64)
            dispy = np.ascontiguousarray(disp[:, 1], dtype=np.float64)
            dispz = np.zeros(disp.shape[0], dtype=np.float64)
            if GlobalVariable.DIMENSION == 3:
                dispz = np.ascontiguousarray(disp[:, 2], dtype=np.float64)
            if smooth_setting:
                dispx = smoother.run(position, weight, dispx)
                dispy = smoother.run(position, weight, dispy)
                dispz = smoother.run(position, weight, dispz)
            displacement = (dispx, dispy, dispz)
            data.update({"displacement": displacement})
        if DictIO.GetAlternative(kwargs, "write_velocity", True):
            vel = DictIO.GetEssential(particle_info, "velocity")
            velx = np.ascontiguousarray(vel[:, 0], dtype=np.float64)
            vely = np.ascontiguousarray(vel[:, 1], dtype=np.float64)
            velz = np.zeros(vel.shape[0], dtype=np.float64)
            if GlobalVariable.DIMENSION == 3:
                velz = np.ascontiguousarray(vel[:, 2], dtype=np.float64)
            if smooth_setting:
                velx = smoother.run(position, weight, velx)
                vely = smoother.run(position, weight, vely)
                velz = smoother.run(position, weight, velz)
            velocity = (velx, vely, velz)
            data.update({"velocity": velocity})

        if "strain" in particle_info:
            strain = DictIO.GetEssential(particle_info, "strain")
            if smooth_setting:
                strain = smoother.run(position, weight, strain)
            if DictIO.GetAlternative(kwargs, "write_strain_component", False):
                strainxx = np.ascontiguousarray(strain[:, 0])
                strainyy = np.ascontiguousarray(strain[:, 1])
                strainzz = np.ascontiguousarray(strain[:, 2])
                strainxy = np.ascontiguousarray(strain[:, 3])
                strainyz = np.ascontiguousarray(strain[:, 4])
                strainxz = np.ascontiguousarray(strain[:, 5])
                principle_strain = (strainxx, strainyy, strainzz)
                shear_strain = (strainxy, strainyz, strainxz)
                data.update({"principle_strain": principle_strain, "shear_strain": shear_strain})

        if "stress" in particle_info:
            stress = DictIO.GetEssential(particle_info, "stress")
            if smooth_setting:
                stress = smoother.run(position, weight, stress)
            if DictIO.GetAlternative(kwargs, "write_mean_stress", True):
                mean_stress = np.ascontiguousarray((stress[:, 0] + stress[:, 1] + stress[:, 2]) / 3.0)
                data.update({"mean_stress": mean_stress})
            if DictIO.GetAlternative(kwargs, "write_stress_component", True):
                stressxx = np.ascontiguousarray(stress[:, 0])
                stressyy = np.ascontiguousarray(stress[:, 1])
                stresszz = np.ascontiguousarray(stress[:, 2])
                stressxy = np.ascontiguousarray(stress[:, 3])
                stressyz = np.ascontiguousarray(stress[:, 4])
                stressxz = np.ascontiguousarray(stress[:, 5])
                principle_stress = (stressxx, stressyy, stresszz)
                shear_stress = (stressxy, stressyz, stressxz)
                data.update({"principle_stress": principle_stress, "shear_stress": shear_stress})
            if DictIO.GetAlternative(kwargs, "write_radial_stress", False):
                center = DictIO.GetAlternative(kwargs, "center", False)
                axis = DictIO.GetAlternative(kwargs, "axis", False)

        if "pressure" in particle_info and DictIO.GetAlternative(kwargs, "write_pressure", True):
            pressure = np.ascontiguousarray(DictIO.GetEssential(particle_info, "pressure"))
            if smooth_setting:
                pressure = smoother.run(position, weight, pressure)
            data.update({"pressure": pressure})

        if DictIO.GetAlternative(kwargs, "write_state_variables", True) and "state_vars" in particle_info:
            state_vars = np.ascontiguousarray(DictIO.GetEssential(particle_info, "state_vars")).item()
            for state_var_name in state_vars:
                if smooth_setting:
                    data.update({state_var_name: smoother.run(position, weight, state_vars[state_var_name])})
                else:
                    data.update({state_var_name: state_vars[state_var_name]})
        if "normal" in particle_info and DictIO.GetAlternative(kwargs, "write_outer_norm", True):
            normal = DictIO.GetEssential(particle_info, "normal")
            xnorm = np.ascontiguousarray(normal[:, 0], dtype=np.float64)
            ynorm = np.ascontiguousarray(normal[:, 1], dtype=np.float64)
            znorm = np.zeros(normal.shape[0], dtype=np.float64)
            if GlobalVariable.DIMENSION == 3:
                znorm = np.ascontiguousarray(normal[:, 2], dtype=np.float64)
            data.update({"normal": (xnorm, ynorm, znorm)})
        if "external_force" in particle_info and DictIO.GetAlternative(kwargs, "write_external_force", True):
            external_force = DictIO.GetEssential(particle_info, "external_force")
            xforce = np.ascontiguousarray(external_force[:, 0], dtype=np.float64)
            yforce = np.ascontiguousarray(external_force[:, 1], dtype=np.float64)
            zforce = np.zeros(external_force.shape[0], dtype=np.float64)
            if GlobalVariable.DIMENSION == 3:
                zforce = np.ascontiguousarray(external_force[:, 2], dtype=np.float64)
            data.update({"external_force": (xforce, yforce, zforce)})
        if "free_surface" in particle_info and DictIO.GetAlternative(kwargs, "write_free_surface", True):
            free_surface = DictIO.GetEssential(particle_info, "free_surface")
            data.update({"free_surface": free_surface})
        if sims.material_type == "TwoPhaseDoubleLayer" and "phase" in particle_info:
            phase = DictIO.GetEssential(particle_info, "phase")
            for phase_id, name, velocity_name in (
                (1, "GraphicMPMSolidParticle", "solid_velocity"),
                (2, "GraphicMPMFluidParticle", "fluid_velocity"),
            ):
                mask = phase == phase_id
                phase_data = _slice_vtk_data(data, mask)
                if DictIO.GetAlternative(kwargs, "write_velocity", True) and velocity_name in particle_info:
                    phase_data["velocity"] = _vtk_velocity(particle_info[velocity_name][mask])
                pointsToVTK(
                    write_path + f"/{name}{printNum:06d}",
                    posx[mask],
                    posy[mask],
                    posz[mask],
                    data=phase_data,
                )
        else:
            pointsToVTK(write_path + f"/GraphicMPMParticle{printNum:06d}", posx, posy, posz, data=data)
        write_soft_particle_point_vtk(printNum, read_path, write_path, kwargs)
        if "body_num" in particle_info:
            write_soft_particle_surface_from_levelset(printNum, particle_info, read_path, write_path, kwargs)

        if DictIO.GetAlternative(kwargs, "write_background_grid", False):
            grid_data = {}
            grid_info = np.load((read_path + "/grids/MPMGrid{0:06d}.npz").format(printNum), allow_pickle=True)

            coords = DictIO.GetEssential(grid_info, "coords")
            posx = np.unique(np.ascontiguousarray(coords[:, 0]))
            posy = np.unique(np.ascontiguousarray(coords[:, 1]))
            posz = np.zeros(1)
            if GlobalVariable.DIMENSION == 3:
                posz = np.unique(np.ascontiguousarray(coords[:, 2]))

            if "contact_force" in grid_info:
                contact_force = DictIO.GetEssential(grid_info, "contact_force")
                cforcex = np.ascontiguousarray(contact_force[:, 0][:, 0])
                cforcey = np.ascontiguousarray(contact_force[:, 0][:, 1])
                cforcez = np.zeros(cforcex.shape[0])
                if GlobalVariable.DIMENSION == 3:
                    cforcez = np.ascontiguousarray(contact_force[:, 1][:, 2])
                grid_data.update({"contact_force": (cforcex, cforcey, cforcez)})

            if "normal" in grid_info:
                norm = DictIO.GetEssential(grid_info, "normal")
                xnorm = np.ascontiguousarray(norm[:, 1][:, 0])
                ynorm = np.ascontiguousarray(norm[:, 1][:, 1])
                znorm = np.zeros(xnorm.shape[0])
                if GlobalVariable.DIMENSION == 3:
                    znorm = np.ascontiguousarray(norm[:, 1][:, 2])
                grid_data.update({"normal": (xnorm, ynorm, znorm)})

            gridToVTK(write_path + f"/GraphicMPMGrid{printNum:06d}", posx, posy, posz, pointData=grid_data)
