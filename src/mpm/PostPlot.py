import os

import numpy as np

from src.mpm.Simulation import Simulation
from third_party.pyevtk.hl import pointsToVTK, gridToVTK
from src.utils.ObjectIO import DictIO


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

def write_vtk_file(sims: Simulation, start_file, end_file, read_path, write_path, kwargs):
    end_file = CheckFirst(sims, read_path, write_path, end_file)

    for printNum in range(start_file, end_file):
        data = {}
        particle_file = (read_path + "/particles/MPMParticle{0:06d}.npz").format(printNum)
        if not os.access(particle_file, os.F_OK): 
            raise ValueError(f"File {read_path}/particles/MPMParticle f{0:06d}.npz does not exist!")

        print((" MPM Postprocessing: Output VTK File" + str(printNum) + ' ').center(71, '-'))
        particle_info = np.load(particle_file, allow_pickle=True)
        if printNum == start_file:
            position0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "position"))
            particle_id0 = np.ascontiguousarray(DictIO.GetEssential(particle_info, "particleID"))
        
        particle_id = np.ascontiguousarray(DictIO.GetEssential(particle_info, "particleID"))
        position = DictIO.GetEssential(particle_info, "position")
        posx = np.ascontiguousarray(position[:, 0])
        posy = np.ascontiguousarray(position[:, 1])
        posz = np.zeros(position.shape[0])
        if sims.dimension == 3:
            posz = np.ascontiguousarray(position[:, 2])

        if DictIO.GetAlternative(kwargs, "write_bodyID", True):
            bodyID = np.ascontiguousarray(DictIO.GetEssential(particle_info, "bodyID"))
            data.update({"bodyID": bodyID})
        if DictIO.GetAlternative(kwargs, "write_volume", True):
            volume = np.ascontiguousarray(DictIO.GetEssential(particle_info, "volume"))
            data.update({"volume": volume})
        if DictIO.GetAlternative(kwargs, "write_displacement", True):
            id_map = {pid: idx for idx, pid in enumerate(particle_id0)}
            indices = np.array([id_map[pid] for pid in particle_id])
            disp = position - position0[indices]
            dispx = np.ascontiguousarray(disp[:, 0])
            dispy = np.ascontiguousarray(disp[:, 1])
            dispz = np.zeros(disp.shape[0])
            if sims.dimension == 3:
                dispz = np.ascontiguousarray(disp[:, 2])
            displacement = (dispx, dispy, dispz)
            data.update({"displacement": displacement})
        if DictIO.GetAlternative(kwargs, "write_velocity", True):
            vel = DictIO.GetEssential(particle_info, "velocity")
            velx = np.ascontiguousarray(vel[:, 0])
            vely = np.ascontiguousarray(vel[:, 1])
            velz = np.zeros(vel.shape[0])
            if sims.dimension == 3:
                velz = np.ascontiguousarray(vel[:, 2])
            velocity = (velx, vely, velz)
            data.update({"velocity": velocity})
        if DictIO.GetAlternative(kwargs, "write_mean_stress", True):
            stress = DictIO.GetEssential(particle_info, "stress")
            mean_stress = np.ascontiguousarray((stress[:, 0] + stress[:, 1] + stress[:, 2]) / 3.)
            data.update({"mean_stress": mean_stress})
        if DictIO.GetAlternative(kwargs, "write_strain_component", False):
            strain = DictIO.GetEssential(particle_info, "strain")
            strainxx = np.ascontiguousarray(strain[:, 0])
            strainyy = np.ascontiguousarray(strain[:, 1])
            strainzz = np.ascontiguousarray(strain[:, 2])
            strainxy = np.ascontiguousarray(strain[:, 3])
            strainyz = np.ascontiguousarray(strain[:, 4])
            strainxz = np.ascontiguousarray(strain[:, 5])
            principle_strain = (strainxx, strainyy, strainzz)
            shear_strain = (strainxy, strainyz, strainxz)
            data.update({"principle_strain": principle_strain, "shear_strain": shear_strain})
        if DictIO.GetAlternative(kwargs, "write_stress_component", True):
            stress = DictIO.GetEssential(particle_info, "stress")
            stressxx = np.ascontiguousarray(stress[:, 0])
            stressyy = np.ascontiguousarray(stress[:, 1])
            stresszz = np.ascontiguousarray(stress[:, 2])
            stressxy = np.ascontiguousarray(stress[:, 3])
            stressyz = np.ascontiguousarray(stress[:, 4])
            stressxz = np.ascontiguousarray(stress[:, 5])
            principle_stress = (stressxx, stressyy, stresszz)
            shear_stress = (stressxy, stressyz, stressxz)
            data.update({"principle_stress": principle_stress, "shear_stress": shear_stress})
        if DictIO.GetAlternative(kwargs, "write_state_variables", True):
            state_vars = np.ascontiguousarray(DictIO.GetEssential(particle_info, "state_vars"))
            data.update(state_vars.item())
        if "normal" in particle_info and DictIO.GetAlternative(kwargs, "write_outer_norm", True):
            normal = DictIO.GetEssential(particle_info, "normal")
            xnorm = np.ascontiguousarray(normal[:, 0])
            ynorm = np.ascontiguousarray(normal[:, 1])
            znorm = np.zeros(normal.shape[0])
            if sims.dimension == 3:
                znorm = np.ascontiguousarray(normal[:, 2])
            data.update({"normal": (xnorm, ynorm, znorm)})
        if "external_force" in particle_info and DictIO.GetAlternative(kwargs, "write_external_force", True):
            external_force = DictIO.GetEssential(particle_info, "external_force")
            xforce = np.ascontiguousarray(external_force[:, 0])
            yforce = np.ascontiguousarray(external_force[:, 1])
            zforce = np.zeros(external_force.shape[0])
            if sims.dimension == 3:
                zforce = np.ascontiguousarray(external_force[:, 2])
            data.update({"external_force": (xforce, yforce, zforce)})
        if "free_surface" in particle_info and DictIO.GetAlternative(kwargs, "write_free_surface", True):
            free_surface = DictIO.GetEssential(particle_info, "free_surface")
            data.update({"free_surface": free_surface})
        pointsToVTK(write_path+f'/GraphicMPMParticle{printNum:06d}', posx, posy, posz, data=data)

        if DictIO.GetAlternative(kwargs, "write_background_grid", False):
            grid_data = {}
            grid_info = np.load((read_path + "/grids/MPMGrid{0:06d}.npz").format(printNum), allow_pickle=True)

            coords = DictIO.GetEssential(grid_info, "coords")
            posx = np.unique(np.ascontiguousarray(coords[:, 0]))
            posy = np.unique(np.ascontiguousarray(coords[:, 1]))
            posz = np.zeros(1)
            if sims.dimension == 3:
                posz = np.unique(np.ascontiguousarray(coords[:, 2]))

            if "contact_force" in grid_info:
                contact_force = DictIO.GetEssential(grid_info, "contact_force")
                cforcex = np.ascontiguousarray(contact_force[:, 0][:, 0])
                cforcey = np.ascontiguousarray(contact_force[:, 0][:, 1])
                cforcez = np.zeros(cforcex.shape[0])
                if sims.dimension == 3:
                    cforcez = np.ascontiguousarray(contact_force[:, 1][:, 2])
                grid_data.update({"contact_force": (cforcex, cforcey, cforcez)})

            if "normal" in grid_info:
                norm = DictIO.GetEssential(grid_info, "normal")
                xnorm = np.ascontiguousarray(norm[:, 1][:, 0])
                ynorm = np.ascontiguousarray(norm[:, 1][:, 1])
                znorm = np.zeros(xnorm.shape[0])
                if sims.dimension == 3:
                    znorm = np.ascontiguousarray(norm[:, 1][:, 2])
                grid_data.update({"normal": (xnorm, ynorm, znorm)})

            gridToVTK(write_path+f'/GraphicMPMGrid{printNum:06d}', posx, posy, posz, pointData=grid_data)
            