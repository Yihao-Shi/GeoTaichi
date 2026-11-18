import os
import numpy as np

from src.dem.BaseKernel import kernel_postvisualize_surface
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from third_party.pyevtk.hl import pointsToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle


def check_first(sims: Simulation, read_path, write_path, end_file):
    if not os.path.exists(read_path):
        print(read_path)
        raise EOFError("Invaild path")
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    if end_file == -1:
        if sims.current_print == 0:
            raise ValueError("Invalid end_file")
        end_file = sims.current_print
    return end_file


def write_dem_vtk_file(sims: Simulation, start_file, end_file, read_path, write_path, kwargs):
    end_file = check_first(sims, read_path, write_path, end_file)

    total_disp = DictIO.get_alternative(kwargs, "total_displacement", False)
    if total_disp:
        particle_file0 = read_path + "/particles/DEMParticle{0:06d}.npz".format(0)
        sphere_file0 = read_path + "/particles/DEMSphere{0:06d}.npz".format(0)
        clump_file0 = read_path + "/particles/DEMClump{0:06d}.npz".format(0)

        particle_info0 = np.load(particle_file0, allow_pickle=True)
        position0 = np.ascontiguousarray(DictIO.get_essential(particle_info0, "position"))
        if os.access(sphere_file0, os.F_OK):
            sphere_info0 = np.load(sphere_file0, allow_pickle=True)
            sphere_id0 = np.ascontiguousarray(DictIO.get_essential(sphere_info0, "grainIndex"))
            particle_index0 = np.ascontiguousarray(DictIO.get_essential(sphere_info0, "sphereIndex"))
        if os.access(clump_file0, os.F_OK):
            clump_info0 = np.load(clump_file0, allow_pickle=True)
            clump_id0 = np.ascontiguousarray(DictIO.get_essential(clump_info0, "grainIndex"))
            mass_center0 = np.ascontiguousarray(DictIO.get_essential(clump_info0, "centerOfMass"))

    for print_num in range(start_file, end_file):
        data = {}
        particle_file = read_path + "/particles/DEMParticle{0:06d}.npz".format(print_num)
        sphere_file = read_path + "/particles/DEMSphere{0:06d}.npz".format(print_num)
        clump_file = read_path + "/particles/DEMClump{0:06d}.npz".format(print_num)
        if not os.access(particle_file, os.F_OK):
            continue

        print((" DEM Postprocessing: Output VTK File" + str(print_num) + " ").center(71, "-"))
        particle_info = np.load(particle_file, allow_pickle=True)

        if print_num == start_file and not total_disp:
            position0 = np.ascontiguousarray(DictIO.get_essential(particle_info, "position"))
            if os.access(sphere_file, os.F_OK):
                sphere_info = np.load(sphere_file, allow_pickle=True)
                sphere_id0 = np.ascontiguousarray(DictIO.get_essential(sphere_info, "grainIndex"))
                particle_index0 = np.ascontiguousarray(DictIO.get_essential(sphere_info, "sphereIndex"))
            if os.access(clump_file, os.F_OK):
                clump_info = np.load(clump_file, allow_pickle=True)
                clump_id0 = np.ascontiguousarray(DictIO.get_essential(clump_info, "grainIndex"))
                mass_center0 = np.ascontiguousarray(DictIO.get_essential(clump_info, "centerOfMass"))

        if os.access(sphere_file, os.F_OK):
            sphere_info = np.load(sphere_file, allow_pickle=True)
            sphere_id = np.ascontiguousarray(DictIO.get_essential(sphere_info, "grainIndex"))
            particle_index = np.ascontiguousarray(DictIO.get_essential(sphere_info, "sphereIndex"))
        if os.access(clump_file, os.F_OK):
            clump_info = np.load(clump_file, allow_pickle=True)
            clump_id = np.ascontiguousarray(DictIO.get_essential(clump_info, "grainIndex"))
            start_index = np.ascontiguousarray(DictIO.get_essential(clump_info, "startIndex"))
            end_index = np.ascontiguousarray(DictIO.get_essential(clump_info, "endIndex"))
            mass_center = np.ascontiguousarray(DictIO.get_essential(clump_info, "centerOfMass"))

        position = np.ascontiguousarray(DictIO.get_essential(particle_info, "position"))
        posx = np.ascontiguousarray(position[:, 0])
        posy = np.ascontiguousarray(position[:, 1])
        posz = np.ascontiguousarray(position[:, 2])

        if DictIO.get_alternative(kwargs, "write_bodyID", True):
            body_id = np.ascontiguousarray(DictIO.get_essential(particle_info, "Index"))
            data.update({"bodyID": body_id})
        if DictIO.get_alternative(kwargs, "write_groupID", True):
            group_id = np.ascontiguousarray(DictIO.get_essential(particle_info, "groupID"))
            data.update({"groupID": group_id})
        if DictIO.get_alternative(kwargs, "write_radii", True):
            radii = np.ascontiguousarray(DictIO.get_essential(particle_info, "radius"))
            data.update({"radius": radii})
        if DictIO.get_alternative(kwargs, "write_displacement", True):
            disp = np.zeros((position.shape[0], 3))
            if os.access(sphere_file, os.F_OK):
                id_map = {pid: idx for idx, pid in enumerate(sphere_id0)}
                indices = np.array([id_map[pid] for pid in sphere_id])
                disp[particle_index] = position[particle_index] - position0[particle_index0[indices]]
            if os.access(clump_file, os.F_OK):
                id_map = {pid: idx for idx, pid in enumerate(clump_id0)}
                indices = np.array([id_map[pid] for pid in clump_id])
                counts = end_index - start_index + 1
                total_length = counts.sum()
                cum = np.concatenate(([0], np.cumsum(counts)))
                x = np.arange(total_length)
                segments = np.searchsorted(cum, x, side="right") - 1
                offsets = x - cum[segments]
                index = start_index[segments] + offsets
                # index = np.concatenate([index, np.array([end_index[-1]])])
                disp[index] = np.repeat(mass_center - mass_center0[indices], counts, axis=0)
            dispx = np.ascontiguousarray(disp[:, 0])
            dispy = np.ascontiguousarray(disp[:, 1])
            dispz = np.ascontiguousarray(disp[:, 2])
            displacement = (dispx, dispy, dispz)
            data.update({"displacement": displacement})
        if DictIO.get_alternative(kwargs, "write_velocity", True):
            vel = np.ascontiguousarray(DictIO.get_essential(particle_info, "velocity"))
            velx = np.ascontiguousarray(vel[:, 0])
            vely = np.ascontiguousarray(vel[:, 1])
            velz = np.ascontiguousarray(vel[:, 2])
            velocity = (velx, vely, velz)
            data.update({"velocity": velocity})
        if DictIO.get_alternative(kwargs, "write_angular_velocity", True):
            w = np.ascontiguousarray(DictIO.get_essential(particle_info, "omega"))
            wx = np.ascontiguousarray(w[:, 0])
            wy = np.ascontiguousarray(w[:, 1])
            wz = np.ascontiguousarray(w[:, 2])
            omega = (wx, wy, wz)
            data.update({"omega": omega})

        if len(data) > 0:
            pointsToVTK(write_path + f"/GraphicDEMParticle{print_num:06d}", posx, posy, posz, data=data)

        plot_walls(position, print_num, read_path, write_path, kwargs)
        plot_force_chains(position, print_num, read_path, write_path, kwargs)


def write_lsdem_vtk_file(sims: Simulation, start_file, end_file, read_path, write_path, kwargs):
    end_file = check_first(sims, read_path, write_path, end_file)

    import taichi as ti

    max_surface_num = 0
    for print_num in range(start_file, end_file):
        surface_file = read_path + "/particles/LSDEMSurface{0:06d}.npz".format(print_num)
        if not os.access(surface_file, os.F_OK):
            continue
        surface_info = np.load(surface_file, allow_pickle=True)
        max_surface_num = max(int(DictIO.get_essential(surface_info, "total_surface_num")), max_surface_num)
    vertices = ti.Vector.field(3, float, shape=max_surface_num) if max_surface_num > 0 else None

    position0 = None
    for print_num in range(start_file, end_file):
        particle_file = read_path + "/particles/LSDEMRigid{0:06d}.npz".format(print_num)
        surface_file = read_path + "/particles/LSDEMSurface{0:06d}.npz".format(print_num)
        if not os.access(particle_file, os.F_OK):
            continue

        print((" LSDEM Postprocessing: Output VTK File" + str(print_num) + " ").center(71, "-"))
        particle_info = np.load(particle_file, allow_pickle=True)

        if position0 is None:
            position0 = np.ascontiguousarray(DictIO.get_essential(particle_info, "mass_center"))

        position = np.ascontiguousarray(DictIO.get_essential(particle_info, "mass_center"))
        if os.access(surface_file, os.F_OK) and vertices is not None:
            data = {}
            surface_info = np.load(surface_file, allow_pickle=True)
            surface_num = int(DictIO.get_essential(surface_info, "total_surface_num"))
            master = np.ascontiguousarray(DictIO.get_essential(surface_info, "master")).astype(np.int32)
            connectivity = np.ascontiguousarray(DictIO.get_essential(surface_info, "connectivity"))
            surface_node = np.ascontiguousarray(DictIO.get_essential(surface_info, "vertices"))
            quanternion = np.ascontiguousarray(DictIO.get_essential(particle_info, "quanternion"))
            start_node = np.ascontiguousarray(
                DictIO.get_alternative(surface_info, "startNode", DictIO.get_essential(particle_info, "startNode"))
            )
            local_node = np.ascontiguousarray(
                DictIO.get_alternative(surface_info, "localNode", DictIO.get_essential(particle_info, "localNode"))
            )
            scale = np.ascontiguousarray(DictIO.get_essential(particle_info, "scale"))

            world_vertices = DictIO.get_alternative(surface_info, "world_vertices", None)
            if world_vertices is None:
                kernel_postvisualize_surface(
                    surface_num, surface_node, position, quanternion, start_node, local_node, master, scale, vertices
                )
                surface = vertices.to_numpy()[0:surface_num]
            else:
                surface = np.ascontiguousarray(world_vertices)[0:surface_num]

            is_soft = DictIO.get_alternative(particle_info, "is_soft", None)
            if is_soft is not None and master.shape[0] > 0:
                is_soft = np.ascontiguousarray(is_soft).astype(bool)
                if int(np.max(master)) < is_soft.shape[0]:
                    keep_node = np.logical_not(is_soft[master])
                    if not np.all(keep_node):
                        old_to_new = np.full(master.shape[0], -1, dtype=np.int32)
                        old_to_new[keep_node] = np.arange(np.count_nonzero(keep_node), dtype=np.int32)
                        if connectivity.size > 0:
                            keep_face = np.all(keep_node[connectivity], axis=1)
                            connectivity = old_to_new[connectivity[keep_face]]
                        surface = surface[keep_node]
                        master = master[keep_node]
                        surface_num = int(master.shape[0])

            if surface_num == 0 or connectivity.shape[0] == 0:
                plot_walls(position, print_num, read_path, write_path, kwargs)
                plot_force_chains_lsdem(position, print_num, read_path, write_path, kwargs)
                plot_boundings(print_num, read_path, write_path, kwargs)
                continue

            posx = np.ascontiguousarray(surface[:, 0])
            posy = np.ascontiguousarray(surface[:, 1])
            posz = np.ascontiguousarray(surface[:, 2])
            ndim, nface = 3, connectivity.shape[0]

            if DictIO.get_alternative(kwargs, "write_bodyID", True):
                data.update({"bodyID": master})
            if DictIO.get_alternative(kwargs, "write_groupID", True):
                group_id = np.ascontiguousarray(DictIO.get_essential(particle_info, "groupID"))
                data.update({"groupID": np.ascontiguousarray(group_id[master])})
            if DictIO.get_alternative(kwargs, "write_radii", True):
                radii = np.ascontiguousarray(DictIO.get_essential(particle_info, "equivalentRadius"))
                data.update({"radius": np.ascontiguousarray(radii[master])})
            if DictIO.get_alternative(kwargs, "write_displacement", True):
                disp = position - position0
                displacement = (
                    np.ascontiguousarray(disp[master, 0]),
                    np.ascontiguousarray(disp[master, 1]),
                    np.ascontiguousarray(disp[master, 2]),
                )
                data.update({"displacement": displacement})
            if DictIO.get_alternative(kwargs, "write_velocity", True):
                vel = np.ascontiguousarray(DictIO.get_essential(particle_info, "velocity"))
                velocity = (
                    np.ascontiguousarray(vel[master, 0]),
                    np.ascontiguousarray(vel[master, 1]),
                    np.ascontiguousarray(vel[master, 2]),
                )
                data.update({"velocity": velocity})
            if DictIO.get_alternative(kwargs, "write_angular_velocity", True):
                w = np.ascontiguousarray(DictIO.get_essential(particle_info, "omega"))
                omega = (
                    np.ascontiguousarray(w[master, 0]),
                    np.ascontiguousarray(w[master, 1]),
                    np.ascontiguousarray(w[master, 2]),
                )
                data.update({"omega": omega})

            if len(data) > 0 and nface > 0:
                unstructuredGridToVTK(
                    write_path + f"/GraphicLSDEMSurface{print_num:06d}",
                    posx,
                    posy,
                    posz,
                    connectivity=np.ascontiguousarray(connectivity.flatten()),
                    offsets=np.ascontiguousarray(np.arange(ndim, ndim * nface + 1, ndim, dtype=np.int32)),
                    cell_types=np.repeat(VtkTriangle.tid, nface),
                    pointData=data,
                )

        plot_walls(position, print_num, read_path, write_path, kwargs)
        plot_force_chains_lsdem(position, print_num, read_path, write_path, kwargs)
        plot_boundings(print_num, read_path, write_path, kwargs)


def plot_boundings(printNum, read_path, write_path, kwargs):
    if DictIO.get_alternative(kwargs, "write_bounding_sphere", True):
        pass
    elif DictIO.get_alternative(kwargs, "write_bounding_box", True):
        pass


def plot_walls(sims: Simulation, printNum, read_path, write_path, kwargs):
    if DictIO.get_alternative(kwargs, "write_wall", False):
        wall_info = np.load(read_path + "/walls/DEMWall{0:06d}.npz".format(printNum))
        if sims.wall_type == "Plane":
            pass
        elif sims.wall_type == "Facet" or sims.wall_type == "Patch":
            ndim = 3
            point1 = np.ascontiguousarray(wall_info["point1"].to_numpy())
            point2 = np.ascontiguousarray(wall_info["point2"].to_numpy())
            point3 = np.ascontiguousarray(wall_info["point3"].to_numpy())
            points = np.concatenate((point1, point2, point3), axis=1).reshape(-1, ndim)

            point, cell = np.unique(points, axis=0, return_inverse=True)
            faces = cell.reshape((-1, ndim))
            nface = faces.shape[0]
            offset = np.arange(ndim, ndim * nface + 1, ndim)

            unstructuredGridToVTK(
                write_path + f"/TriangleWall{sims.current_print:06d}",
                np.ascontiguousarray(point[:, 0]),
                np.ascontiguousarray(point[:, 1]),
                np.ascontiguousarray(point[:, 2]),
                connectivity=np.ascontiguousarray(faces.flatten()),
                offsets=np.ascontiguousarray(offset),
                cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface)),
            )


def pointon_plan(wall_info, position, wall_id):
    if DictIO.get_alternative(wall_info, "point", None) is None:
        point = (
            1.0
            / 3.0
            * (
                DictIO.get_essential(wall_info, "point1")[wall_id]
                + DictIO.get_essential(wall_info, "point2")[wall_id]
                + DictIO.get_essential(wall_info, "point3")[wall_id]
            )
        )
    else:
        point = DictIO.get_essential(wall_info, "point")[wall_id]
    norm = DictIO.get_essential(wall_info, "norm")[wall_id]
    return position - np.dot(position - point, norm) * norm


def plot_force_chains(position, printNum, read_path, write_path, kwargs):
    write_force_chain = DictIO.get_alternative(kwargs, "write_force_chain", False) or DictIO.get_alternative(
        kwargs, "write_strong_force_chain", False
    )
    if write_force_chain:
        write_strong_force_chain = DictIO.get_alternative(kwargs, "write_strong_force_chain", False)
        wall_info = np.load(read_path + "/walls/DEMWall{0:06d}.npz".format(printNum))
        ppcontact_info = np.load(read_path + "/contacts/DEMContactPP{0:06d}.npz".format(printNum))
        pwcontact_info = np.load(read_path + "/contacts/DEMContactPW{0:06d}.npz".format(printNum))

        out_contact_file = open(write_path + f"/GraphicForceChain{printNum:06d}.vtp", "w")
        selectpp = np.linalg.norm(DictIO.get_essential(ppcontact_info, "normal_force"), axis=1) > 0.0
        selectpw = np.linalg.norm(DictIO.get_essential(pwcontact_info, "normal_force"), axis=1) > 0.0
        ppend1 = DictIO.get_essential(ppcontact_info, "end1")[selectpp]
        ppend2 = DictIO.get_essential(ppcontact_info, "end2")[selectpp]
        ppfn = DictIO.get_essential(ppcontact_info, "normal_force")[selectpp]
        pwend1 = DictIO.get_essential(pwcontact_info, "end1")[selectpw]
        pwend2 = DictIO.get_essential(pwcontact_info, "end2")[selectpw]
        pwfn = DictIO.get_essential(pwcontact_info, "normal_force")[selectpw]

        ave_ppfn = np.mean(np.linalg.norm(ppfn, axis=1))
        ave_pwfn = np.mean(np.linalg.norm(pwfn, axis=1))
        strong_n_intrs = 0
        for cp in range(ppend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(ppfn[cp]) < ave_ppfn:
                continue
            strong_n_intrs += 1
        for cw in range(pwend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(pwfn[cw]) < ave_pwfn:
                continue
            strong_n_intrs += 1
        n_intrs = strong_n_intrs if write_strong_force_chain else ppend1.shape[0] + pwend1.shape[0]

        # head
        out_contact_file.write(
            "<?xml version='1.0'?>\n<VTKFile type='PolyData' version='0.1' byte_order='LittleEndian'>\n<PolyData>\n"
        )
        out_contact_file.write(
            "<Piece NumberOfPoints='%s' NumberOfVerts='0' NumberOfLines='%s' NumberOfStrips='0' NumberOfPolys='0'>\n"
            % (str(2 * n_intrs), str(n_intrs))
        )

        # write coords of intrs bodies (also taking into account possible periodicity
        out_contact_file.write("<Points>\n<DataArray type='Float32' NumberOfComponents='3' format='ascii'>\n")
        for cp in range(ppend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(ppfn[cp]) < ave_ppfn:
                continue
            pos = position[ppend1[cp]]
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))
            pos = position[ppend2[cp]]
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))

        for cw in range(pwend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(pwfn[cw]) < ave_pwfn:
                continue
            pos = pointon_plan(wall_info, position[pwend1[cw]], pwend2[cw])
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))
            pos = position[pwend1[cw]]
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))

        out_contact_file.write(
            "</DataArray>\n</Points>\n<Lines>\n<DataArray type='Int32' Name='connectivity' format='ascii'>\n"
        )

        ss = ""
        for con in range(2 * n_intrs):
            ss += " " + str(con)
        out_contact_file.write(ss + "\n")
        out_contact_file.write("</DataArray>\n<DataArray type='Int32' Name='offsets' format='ascii'>\n")
        ss = ""
        for con in range(n_intrs):
            ss += " " + str(con * 2 + 2)
        out_contact_file.write(ss)
        out_contact_file.write("\n</DataArray>\n</Lines>\n")

        name = "Fn"
        out_contact_file.write(
            "<PointData Scalars='%s'>\n<DataArray type='Float32' Name='%s' format='ascii'>\n" % (name, name)
        )
        for cp in range(ppend1.shape[0]):
            fn = 0.5 * np.linalg.norm(ppfn[cp])
            if write_strong_force_chain and fn < 0.5 * ave_ppfn:
                continue
            out_contact_file.write("%g %g\n" % (fn, fn))
        for cp in range(pwend1.shape[0]):
            fn = 0.5 * np.linalg.norm(pwfn[cp])
            if write_strong_force_chain and fn < 0.5 * ave_pwfn:
                continue
            out_contact_file.write("%g %g\n" % (fn, fn))
        out_contact_file.write("</DataArray>\n</PointData>")
        out_contact_file.write("\n</Piece>\n</PolyData>\n</VTKFile>")
        out_contact_file.close()


def plot_force_chains_lsdem(position, printNum, read_path, write_path, kwargs):
    write_force_chain = DictIO.get_alternative(kwargs, "write_force_chain", False) or DictIO.get_alternative(
        kwargs, "write_strong_force_chain", False
    )
    if write_force_chain:
        write_strong_force_chain = DictIO.get_alternative(kwargs, "write_strong_force_chain", False)
        wall_info = np.load(read_path + "/walls/DEMWall{0:06d}.npz".format(printNum))
        ppcontact_info = np.load(read_path + "/contacts/DEMContactPP{0:06d}.npz".format(printNum))
        pwcontact_info = np.load(read_path + "/contacts/DEMContactPW{0:06d}.npz".format(printNum))
        surface_data = np.load(read_path + "/particles/LSDEMSurface{0:06d}.npz".format(printNum))

        out_contact_file = open(write_path + f"/GraphicForceChain{printNum:06d}.vtp", "w")
        selectpp = np.linalg.norm(DictIO.get_essential(ppcontact_info, "normal_force"), axis=1) > 0.0
        selectpw = np.linalg.norm(DictIO.get_essential(pwcontact_info, "normal_force"), axis=1) > 0.0

        ppglobal_node = DictIO.get_essential(ppcontact_info, "end1")[selectpp]
        ppend1 = surface_data["master"][ppglobal_node]
        ppend2 = DictIO.get_essential(ppcontact_info, "end2")[selectpp]
        ppfn = DictIO.get_essential(ppcontact_info, "normal_force")[selectpp]

        pwglobal_node = DictIO.get_essential(pwcontact_info, "end1")[selectpw]
        pwend1 = surface_data["master"][pwglobal_node]
        pwend2 = DictIO.get_essential(pwcontact_info, "end2")[selectpw]
        pwfn = DictIO.get_essential(pwcontact_info, "normal_force")[selectpw]

        ave_ppfn = np.mean(np.linalg.norm(ppfn, axis=1))
        ave_pwfn = np.mean(np.linalg.norm(pwfn, axis=1))
        strong_n_intrs = 0
        for cp in range(ppend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(ppfn[cp]) < ave_ppfn:
                continue
            strong_n_intrs += 1
        for cw in range(pwend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(pwfn[cw]) < ave_pwfn:
                continue
            strong_n_intrs += 1
        n_intrs = strong_n_intrs if write_strong_force_chain else ppend1.shape[0] + pwend1.shape[0]

        # head
        out_contact_file.write(
            "<?xml version='1.0'?>\n<VTKFile type='PolyData' version='0.1' byte_order='LittleEndian'>\n<PolyData>\n"
        )
        out_contact_file.write(
            "<Piece NumberOfPoints='%s' NumberOfVerts='0' NumberOfLines='%s' NumberOfStrips='0' NumberOfPolys='0'>\n"
            % (str(2 * n_intrs), str(n_intrs))
        )

        # write coords of intrs bodies (also taking into account possible periodicity
        out_contact_file.write("<Points>\n<DataArray type='Float32' NumberOfComponents='3' format='ascii'>\n")
        for cp in range(ppend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(ppfn[cp]) < ave_ppfn:
                continue
            pos = position[ppend1[cp]]
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))
            pos = position[ppend2[cp]]
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))

        for cw in range(pwend1.shape[0]):
            if write_strong_force_chain and np.linalg.norm(pwfn[cw]) < ave_pwfn:
                continue
            pos = pointon_plan(wall_info, position[pwend1[cw]], pwend2[cw])
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))
            pos = position[pwend2[cw]]
            out_contact_file.write("%g %g %g\n" % (pos[0], pos[1], pos[2]))

        out_contact_file.write(
            "</DataArray>\n</Points>\n<Lines>\n<DataArray type='Int32' Name='connectivity' format='ascii'>\n"
        )

        ss = ""
        for con in range(2 * n_intrs):
            ss += " " + str(con)
        out_contact_file.write(ss + "\n")
        out_contact_file.write("</DataArray>\n<DataArray type='Int32' Name='offsets' format='ascii'>\n")
        ss = ""
        for con in range(n_intrs):
            ss += " " + str(con * 2 + 2)
        out_contact_file.write(ss)
        out_contact_file.write("\n</DataArray>\n</Lines>\n")

        name = "Force normal"
        out_contact_file.write(
            "<PointData Scalars='%s'>\n<DataArray type='Float32' Name='%s' format='ascii'>\n" % (name, name)
        )
        for cp in range(ppend1.shape[0]):
            fn = 0.5 * np.linalg.norm(ppfn[cp])
            if write_strong_force_chain and fn < 0.5 * ave_ppfn:
                continue
            out_contact_file.write("%g %g\n" % (fn, fn))
        for cp in range(pwend1.shape[0]):
            fn = 0.5 * np.linalg.norm(pwfn[cp])
            if write_strong_force_chain and fn < 0.5 * ave_pwfn:
                continue
            out_contact_file.write("%g %g\n" % (fn, fn))
        out_contact_file.write("</DataArray>\n</PointData>")
        out_contact_file.write("\n</Piece>\n</PolyData>\n</VTKFile>")
        out_contact_file.close()
