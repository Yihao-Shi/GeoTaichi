import os
import numpy as np

from src.dem.BaseKernel import kernel_postvisualize_surface_
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from third_party.pyevtk.hl import pointsToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle


def CheckFirst(sims: Simulation, read_path, write_path, end_file):
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
    end_file = CheckFirst(sims, read_path, write_path, end_file)

    for printNum in range(start_file, end_file):
        data = {}
        particle_file = read_path + "/particles/DEMParticle{0:06d}.npz".format(printNum)
        sphere_file = read_path + "/particles/DEMSphere{0:06d}.npz".format(printNum)
        clump_file = read_path + "/particles/DEMClump{0:06d}.npz".format(printNum)
        if not os.access(particle_file, os.F_OK): continue

        print((" DEM Postprocessing: Output VTK File" + str(printNum) + ' ').center(71, '-'))
        particle_info = np.load(particle_file, allow_pickle=True)
        
        if printNum == start_file:
            position0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "position"))
            if os.access(sphere_file, os.F_OK): 
                sphere_info = np.load(sphere_file, allow_pickle=True)
                sphere_id0 = np.ascontiguousarray(DictIO.GetEssential(sphere_info, "grainIndex"))
                particle_index0 = -1 - np.ascontiguousarray(DictIO.GetEssential(sphere_info, "sphereIndex"))
            if os.access(clump_file, os.F_OK): 
                clump_info = np.load(sphere_file, allow_pickle=True)
                clump_id0 = np.ascontiguousarray(DictIO.GetEssential(clump_info, "grainIndex"))
                mass_center0 = np.ascontiguousarray(DictIO.GetEssential(clump_info, "mass_center"))

        if os.access(sphere_file, os.F_OK): 
            sphere_info = np.load(sphere_file, allow_pickle=True)
            sphere_id = np.ascontiguousarray(DictIO.GetEssential(sphere_info, "grainIndex"))
            particle_index = -1 - np.ascontiguousarray(DictIO.GetEssential(sphere_info, "sphereIndex"))
        if os.access(clump_file, os.F_OK): 
            clump_info = np.load(sphere_file, allow_pickle=True)
            clump_id = np.ascontiguousarray(DictIO.GetEssential(clump_info, "grainIndex"))
            start_index = np.ascontiguousarray(DictIO.GetEssential(clump_info, "startIndex"))
            end_index = np.ascontiguousarray(DictIO.GetEssential(clump_info, "endIndex"))
            mass_center = np.ascontiguousarray(DictIO.GetEssential(clump_info, "mass_center"))
            
        position = np.ascontiguousarray(DictIO.GetEssential(particle_info, "position"))
        posx = np.ascontiguousarray(position[:, 0])
        posy = np.ascontiguousarray(position[:, 1])
        posz = np.ascontiguousarray(position[:, 2])

        if DictIO.GetAlternative(kwargs, "write_bodyID", True):
            bodyID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "Index"))
            data.update({"bodyID": bodyID})
        if DictIO.GetAlternative(kwargs, "write_groupID", True):
            groupID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "groupID"))
            data.update({"groupID": groupID})
        if DictIO.GetAlternative(kwargs, "write_radii", True):
            radii = np.ascontiguousarray(DictIO.GetEssential(particle_info, "radius"))
            data.update({"radius": radii})
        if DictIO.GetAlternative(kwargs, "write_displacement", True):
            disp = np.zeros((position.shape[0], 3))
            if os.access(sphere_file, os.F_OK): 
                id_map = {pid: idx for idx, pid in enumerate(sphere_id0)}
                indices = np.array([id_map[pid] for pid in sphere_id])
                disp[particle_index] = position[particle_index] - position0[particle_index0[indices]]
            if os.access(clump_file, os.F_OK): 
                id_map = {pid: idx for idx, pid in enumerate(clump_id0)}
                indices = np.array([id_map[pid] for pid in clump_id])
                counts = end_index - start_index
                total_length = counts.sum()
                cum = np.concatenate(([0], np.cumsum(counts)))
                x = np.arange(total_length)
                segments = np.searchsorted(cum, x, side='right') - 1
                offsets = x - cum[segments]
                index = start_index[segments] + offsets
                index = np.concatenate([index, np.array([end_index[-1]])])
                disp[index] = np.repeat(mass_center - mass_center0[indices], counts)
            dispx = np.ascontiguousarray(disp[:, 0])
            dispy = np.ascontiguousarray(disp[:, 1])
            dispz = np.ascontiguousarray(disp[:, 2])
            displacement = (dispx, dispy, dispz)
            data.update({"displacement": displacement})
        if DictIO.GetAlternative(kwargs, "write_velocity", True):
            vel = np.ascontiguousarray(DictIO.GetEssential(particle_info, "velocity"))
            velx = np.ascontiguousarray(vel[:, 0])
            vely = np.ascontiguousarray(vel[:, 1])
            velz = np.ascontiguousarray(vel[:, 2])
            velocity = (velx, vely, velz)
            data.update({"velocity": velocity})
        if DictIO.GetAlternative(kwargs, "write_angular_velocity", True):
            w = np.ascontiguousarray(DictIO.GetEssential(particle_info, "omega"))
            wx = np.ascontiguousarray(w[:, 0])
            wy = np.ascontiguousarray(w[:, 1])
            wz = np.ascontiguousarray(w[:, 2])
            omega = (wx, wy, wz)
            data.update({"omega": omega})

        if len(data) > 0:
            pointsToVTK(write_path+f'/GraphicDEMParticle{printNum:06d}', posx, posy, posz, data=data)

        PlotWalls(position, printNum, read_path, write_path, kwargs) 
        PlotForceChains(position, printNum, read_path, write_path, kwargs)


def write_lsdem_vtk_file(sims: Simulation, start_file, end_file, read_path, write_path, kwargs):
    end_file = CheckFirst(sims, read_path, write_path, end_file)

    import taichi as ti
    surface_num = 0
    for printNum in range(start_file, end_file):
        surface_file = read_path + "/particles/LSDEMSurface{0:06d}.npz".format(printNum)
        if not os.access(surface_file, os.F_OK): continue
        surface_info = np.load(surface_file, allow_pickle=True)
        surface_num = max(int(DictIO.GetEssential(surface_info, "total_surface_num")), surface_num)
    vertices = ti.Vector.field(3, float, shape=surface_num)

    for printNum in range(start_file, end_file):
        data = {}
        particle_file = read_path + "/particles/LSDEMRigid{0:06d}.npz".format(printNum)
        surface_file = read_path + "/particles/LSDEMSurface{0:06d}.npz".format(printNum)
        if not os.access(particle_file, os.F_OK) or not os.access(surface_file, os.F_OK): continue

        print((" LSDEM Postprocessing: Output VTK File" + str(printNum) + ' ').center(71, '-'))
        particle_info = np.load(particle_file, allow_pickle=True)
        surface_info = np.load(surface_file, allow_pickle=True)

        if printNum == start_file:
            position0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "mass_center"))

        surface_num = np.ascontiguousarray(DictIO.GetEssential(surface_info, "total_surface_num"))
        master = np.ascontiguousarray(DictIO.GetEssential(surface_info, "master"))
        connectivity = np.ascontiguousarray(DictIO.GetEssential(surface_info, "connectivity"))
        surface_node = np.ascontiguousarray(DictIO.GetEssential(surface_info, "vertices"))
        position = np.ascontiguousarray(DictIO.GetEssential(particle_info, "mass_center"))
        quanternion = np.ascontiguousarray(DictIO.GetEssential(particle_info, "quanternion"))
        startNode = np.ascontiguousarray(DictIO.GetEssential(particle_info, "startNode"))
        localNode = np.ascontiguousarray(DictIO.GetEssential(particle_info, "localNode"))
        scale = np.ascontiguousarray(DictIO.GetEssential(particle_info, "scale"))
        
        kernel_postvisualize_surface_(int(surface_num), surface_node, position, quanternion, startNode, localNode, master, scale, vertices)
        
        startID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "startNode"))
        endID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "endNode"))
        posx = np.ascontiguousarray(vertices.to_numpy()[:, 0])
        posy = np.ascontiguousarray(vertices.to_numpy()[:, 1])
        posz = np.ascontiguousarray(vertices.to_numpy()[:, 2])
        node_number = endID - startID 
        ndim, nface = 3, connectivity.shape[0]

        if DictIO.GetAlternative(kwargs, "write_bodyID", True):
            bodyID = np.ascontiguousarray(DictIO.GetEssential(surface_info, "master"))
            data.update({"bodyID": bodyID})
        if DictIO.GetAlternative(kwargs, "write_groupID", True):
            groupID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "groupID"))
            data.update({"groupID": np.ascontiguousarray(np.repeat(groupID, node_number))})
        if DictIO.GetAlternative(kwargs, "write_radii", True):
            radii = np.ascontiguousarray(DictIO.GetEssential(particle_info, "equivalentRadius"))
            data.update({"radius": np.ascontiguousarray(np.repeat(radii, node_number))})
        if DictIO.GetAlternative(kwargs, "write_displacement", True):
            disp =  position - position0
            dispx = np.ascontiguousarray(np.repeat(disp[:, 0], node_number))
            dispy = np.ascontiguousarray(np.repeat(disp[:, 1], node_number))
            dispz = np.ascontiguousarray(np.repeat(disp[:, 2], node_number))
            displacement = (dispx, dispy, dispz)
            data.update({"displacement": displacement})
        if DictIO.GetAlternative(kwargs, "write_velocity", True):
            vel = np.ascontiguousarray(DictIO.GetEssential(particle_info, "velocity"))
            velx = np.ascontiguousarray(np.repeat(vel[:, 0], node_number))
            vely = np.ascontiguousarray(np.repeat(vel[:, 1], node_number))
            velz = np.ascontiguousarray(np.repeat(vel[:, 2], node_number))
            velocity = (velx, vely, velz)
            data.update({"velocity": velocity})
        if DictIO.GetAlternative(kwargs, "write_angular_velocity", True):
            w = np.ascontiguousarray(DictIO.GetEssential(particle_info, "omega"))
            wx = np.ascontiguousarray(np.repeat(w[:, 0], node_number))
            wy = np.ascontiguousarray(np.repeat(w[:, 1], node_number))
            wz = np.ascontiguousarray(np.repeat(w[:, 2], node_number))
            omega = (wx, wy, wz)
            data.update({"omega": omega})

        if len(data) > 0:
            unstructuredGridToVTK(write_path+f'/GraphicLSDEMSurface{printNum:06d}', posx, posy, posz, connectivity=np.ascontiguousarray(connectivity.flatten()), 
                                    offsets=np.ascontiguousarray(np.arange(ndim, ndim * nface + 1, ndim, dtype=np.int32)), 
                                    cell_types=np.repeat(VtkTriangle.tid, nface), pointData=data)

        PlotWalls(position, printNum, read_path, write_path, kwargs) 
        PlotForceChains(position, printNum, read_path, write_path, kwargs)
        PlotBoundings(printNum, read_path, write_path, kwargs)


def PlotBoundings(printNum, read_path, write_path, kwargs):
    if DictIO.GetAlternative(kwargs, "write_bounding_sphere", True):
        pass
    elif DictIO.GetAlternative(kwargs, "write_bounding_box", True):
        pass


def PlotWalls(sims: Simulation, printNum, read_path, write_path, kwargs):
    if DictIO.GetAlternative(kwargs, "write_wall", False):
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

            unstructuredGridToVTK(write_path+f"/TriangleWall{sims.current_print:06d}", np.ascontiguousarray(point[:, 0]), np.ascontiguousarray(point[:, 1]), np.ascontiguousarray(point[:, 2]), 
                                connectivity=np.ascontiguousarray(faces.flatten()), 
                                offsets=np.ascontiguousarray(offset), 
                                cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface)))


def PointonPlan(wall_info, position, wall_id):
    if DictIO.GetAlternative(wall_info, "point", None) is None:
        point = 1./3. * (DictIO.GetEssential(wall_info, "point1")[wall_id] + DictIO.GetEssential(wall_info, "point2")[wall_id] + DictIO.GetEssential(wall_info, "point3")[wall_id])
    else:
        point = DictIO.GetEssential(wall_info, "point")[wall_id]
    norm = DictIO.GetEssential(wall_info, "norm")[wall_id]
    return position - np.dot(position - point, norm) * norm


def PlotForceChains(position, printNum, read_path, write_path, kwargs):
    if DictIO.GetAlternative(kwargs, "write_force_chain", False):
        wall_info = np.load(read_path + "/walls/DEMWall{0:06d}.npz".format(printNum))
        ppcontact_info = np.load(read_path + "/contacts/DEMContactPP{0:06d}.npz".format(printNum))
        pwcontact_info = np.load(read_path + "/contacts/DEMContactPW{0:06d}.npz".format(printNum))

        outContactFile = open(write_path+f'/GraphicForceChain{printNum:06d}.vtp', 'w')
        selectpp = np.linalg.norm(DictIO.GetEssential(ppcontact_info, "normal_force") ,axis=1) > 0.
        selectpw = np.linalg.norm(DictIO.GetEssential(pwcontact_info, "normal_force") ,axis=1) > 0.
        ppend1 = DictIO.GetEssential(ppcontact_info, "end1")[selectpp]
        ppend2 = DictIO.GetEssential(ppcontact_info, "end2")[selectpp]
        ppfn = DictIO.GetEssential(ppcontact_info, "normal_force")[selectpp]
        pwend1 = DictIO.GetEssential(pwcontact_info, "end1")[selectpw]
        pwend2 = DictIO.GetEssential(pwcontact_info, "end2")[selectpw]
        pwfn = DictIO.GetEssential(pwcontact_info, "normal_force")[selectpw]
        nIntrs = ppend1.shape[0] + pwend1.shape[0]

        # head
        outContactFile.write("<?xml version='1.0'?>\n<VTKFile type='PolyData' version='0.1' byte_order='LittleEndian'>\n<PolyData>\n")
        outContactFile.write("<Piece NumberOfPoints='%s' NumberOfVerts='0' NumberOfLines='%s' NumberOfStrips='0' NumberOfPolys='0'>\n"%(str(2 * nIntrs), str(nIntrs)))

        # write coords of intrs bodies (also taking into account possible periodicity
        outContactFile.write("<Points>\n<DataArray type='Float32' NumberOfComponents='3' format='ascii'>\n")
        for cp in range(ppend1.shape[0]):
            pos = position[ppend1[cp]]
            outContactFile.write("%g %g %g\n"%(pos[0], pos[1], pos[2]))
            pos = position[ppend2[cp]]
            outContactFile.write("%g %g %g\n"%(pos[0], pos[1], pos[2]))

        for cw in range(pwend1.shape[0]):
            pos = PointonPlan(wall_info, position[pwend1[cw]], pwend2[cw])    
            outContactFile.write("%g %g %g\n"%(pos[0], pos[1], pos[2]))
            pos = position[pwend1[cw]]
            outContactFile.write("%g %g %g\n"%(pos[0], pos[1], pos[2]))

        outContactFile.write("</DataArray>\n</Points>\n<Lines>\n<DataArray type='Int32' Name='connectivity' format='ascii'>\n")

        ss=''
        for con in range(2 * nIntrs):
            ss+=' '+str(con)
        outContactFile.write(ss+'\n')
        outContactFile.write("</DataArray>\n<DataArray type='Int32' Name='offsets' format='ascii'>\n")
        ss=''
        for con in range(nIntrs):
            ss+=' '+str(con * 2 + 2)
        outContactFile.write(ss)
        outContactFile.write("\n</DataArray>\n</Lines>\n")

        name = 'Fn'
        outContactFile.write("<PointData Scalars='%s'>\n<DataArray type='Float32' Name='%s' format='ascii'>\n"%(name,name))
        for cp in range(ppend1.shape[0]):
            fn = 0.5 * np.linalg.norm(ppfn[cp])
            outContactFile.write("%g %g\n"%(fn, fn))
        for cp in range(pwend1.shape[0]):
            fn = 0.5 * np.linalg.norm(pwfn[cp])
            outContactFile.write("%g %g\n"%(fn, fn))
        outContactFile.write("</DataArray>\n</PointData>")
        outContactFile.write("\n</Piece>\n</PolyData>\n</VTKFile>")
        outContactFile.close()
