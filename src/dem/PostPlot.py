import os
import numpy as np

from src.dem.Simulation import Simulation
from third_party.pyevtk.hl import pointsToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle
from src.utils.ObjectIO import DictIO


def write_vtk_file(sims: Simulation, start_file, end_file, read_path, write_path, kwargs):
    if not os.path.exists(read_path):
        print(read_path)
        raise EOFError("Invaild path")
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    if end_file == -1:
        if sims.current_print == 0:
            raise ValueError("Invalid end_file")
        end_file = sims.current_print

    for printNum in range(start_file, end_file):
        data = {}
        print((" Postprocessing: Output VTK File" + str(printNum) + ' ').center(71, '-'))
        particle_info = np.load(read_path + "/particles/DEMParticle{0:06d}.npz".format(printNum))
        if printNum == start_file:
            position0 = DictIO.GetEssential(particle_info, "position")

        position = DictIO.GetEssential(particle_info, "position")
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
            disp =  position - position0
            dispx = np.ascontiguousarray(disp[:, 0])
            dispy = np.ascontiguousarray(disp[:, 1])
            dispz = np.ascontiguousarray(disp[:, 2])
            displacement = (dispx, dispy, dispz)
            data.update({"displacement": displacement})
        if DictIO.GetAlternative(kwargs, "write_velocity", True):
            vel = DictIO.GetEssential(particle_info, "velocity")
            velx = np.ascontiguousarray(vel[:, 0])
            vely = np.ascontiguousarray(vel[:, 1])
            velz = np.ascontiguousarray(vel[:, 2])
            velocity = (velx, vely, velz)
            data.update({"velocity": velocity})
        if DictIO.GetAlternative(kwargs, "write_angular_velocity", True):
            w = DictIO.GetEssential(particle_info, "omega")
            wx = np.ascontiguousarray(w[:, 0])
            wy = np.ascontiguousarray(w[:, 1])
            wz = np.ascontiguousarray(w[:, 2])
            omega = (wx, wy, wz)
            data.update({"omega": omega})
        if DictIO.GetAlternative(kwargs, "write_force_chain", False):
            wall_info = np.load(read_path + "/walls/DEMWall{0:06d}.npz".format(printNum))
            ppcontact_info = np.load(read_path + "/contacts/DEMContactPP{0:06d}.npz".format(printNum))
            pwcontact_info = np.load(read_path + "/contacts/DEMContactPW{0:06d}.npz".format(printNum))
            ForceChains(position, wall_info, ppcontact_info, pwcontact_info, printNum, write_path)

        if "wall" in sims.monitor_type and DictIO.GetAlternative(kwargs, "write_wall", False):
            wall_info = np.load(read_path + "/walls/DEMWall{0:06d}.npz".format(printNum))
            PlotWalls(sims, wall_info, printNum, write_path)
            
        if len(data) > 0:
            pointsToVTK(write_path+f'/GraphicDEMParticle{printNum:06d}', posx, posy, posz, data=data)


def PlotWalls(sims: Simulation, wall_info, printNum, write_path):
    if sims.wall_type == "Plane":
        pass
    elif sims.wall_type == "Facet" or sims.wall_type == "Patch":
        vistri = []
        point1 = np.ascontiguousarray(wall_info["point1"].to_numpy())
        point2 = np.ascontiguousarray(wall_info["point2"].to_numpy())
        point3 = np.ascontiguousarray(wall_info["point3"].to_numpy())
        unique_point = np.unique(np.vstack((point1, point2, point3)), axis=0)
        for i in range(point1.shape[0]):
            vistri.append([int(np.where((unique_point==point1[i]).all(1))[0]), int(np.where((unique_point==point2[i]).all(1))[0]), int(np.where((unique_point==point3[i]).all(1))[0]), 3 * (i + 1), VtkTriangle.tid])
        vispts = np.ascontiguousarray(np.array(unique_point))
        vistri = np.ascontiguousarray(np.array(vistri))

        unstructuredGridToVTK(write_path + f"TriangleWall{printNum:06d}", np.ascontiguousarray(vispts[:, 0]), np.ascontiguousarray(vispts[:, 1]), np.ascontiguousarray(vispts[:, 2]), 
                                connectivity=np.ascontiguousarray(vistri[:, 0:3].flatten()), 
                                offsets=np.ascontiguousarray(vistri[:, 3]), 
                                cell_types=np.ascontiguousarray(vistri[:, 4]))


def PointonPlan(wall_info, position, wall_id):
    if DictIO.GetAlternative(wall_info, "point", None) is None:
        point = 1./3. * (DictIO.GetEssential(wall_info, "point1")[wall_id] + DictIO.GetEssential(wall_info, "point2")[wall_id] + DictIO.GetEssential(wall_info, "point3")[wall_id])
    else:
        point = DictIO.GetEssential(wall_info, "point")[wall_id]
    norm = DictIO.GetEssential(wall_info, "norm")[wall_id]
    return position - np.dot(position - point, norm) * norm


def ForceChains(position, wall_info, ppcontact_info, pwcontact_info, printNum, write_path):
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
