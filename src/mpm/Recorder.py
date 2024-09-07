import os

import numpy as np

from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from third_party.pyevtk.hl import pointsToVTK, gridToVTK


class WriteFile:
    def __init__(self, sims):
        self.vtk_path = None
        self.particle_path  = None
        self.grid_path = None

        self.save_particle = self.no_operation
        self.save_grid = self.no_operation

        self.mkdir(sims)
        self.manage_function(sims)

    def no_operation(self, sims, scene):
        pass

    def manage_function(self, sims: Simulation):
        if 'particle' in sims.monitor_type:
            if sims.neighbor_detection or sims.coupling:
                self.save_particle = self.MonitorParticleCoupling
            else:
                self.save_particle = self.MonitorParticle

        if 'grid' in sims.monitor_type:
            if sims.contact_detection:
                self.save_grid = self.MonitorContactGrid
            else:
                self.save_grid = self.MonitorGrid

    def output(self, sims, scene):
        self.save_particle(sims, scene)
        self.save_grid(sims, scene)

    def mkdir(self, sims: Simulation):
        if not os.path.exists(sims.path):
            os.makedirs(sims.path)

        self.vtk_path = None
        self.particle_path  = None
        self.grid_path = None

        self.particle_path = sims.path + '/particles'
        self.vtk_path = sims.path + '/vtks'
        self.grid_path = sims.path + '/grids'
        if not os.path.exists(self.particle_path):
            os.makedirs(self.particle_path)
        if not os.path.exists(self.vtk_path):
            os.makedirs(self.vtk_path)
        if not os.path.exists(self.grid_path):
            os.makedirs(self.grid_path)

    def VisualizeParticle(self, sims: Simulation, position, velocity, volume, state_vars):
        posx = np.ascontiguousarray(position[:, 0])
        posy = np.ascontiguousarray(position[:, 1])
        posz = np.ascontiguousarray(position[:, 2])
        velx = np.ascontiguousarray(velocity[:, 0])
        vely = np.ascontiguousarray(velocity[:, 1])
        velz = np.ascontiguousarray(velocity[:, 2])
        data = {"velocity": (velx, vely, velz), "volume": volume}
        data.update(state_vars)
        pointsToVTK(self.vtk_path+f'/GraphicMPMParticle{sims.current_print:06d}', posx, posy, posz, data=data)

    def VisualizeGrid(self, sims: Simulation, coords, data):
        coordx = np.ascontiguousarray(coords[:, 0])
        coordy = np.ascontiguousarray(coords[:, 1])
        coordz = np.ascontiguousarray(coords[:, 2])
        pointsToVTK(self.vtk_path+f'/GraphicMPMGrid{sims.current_print:06d}', coordx, coordy, coordz, data=data)
        
    def MonitorParticleCoupling(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        
        position = scene.particle.x.to_numpy()[0:scene.particleNum[0]]
        bodyID = scene.particle.bodyID.to_numpy()[0:scene.particleNum[0]]
        materialID = scene.particle.materialID.to_numpy()[0:scene.particleNum[0]]
        active = scene.particle.active.to_numpy()[0:scene.particleNum[0]]
        free_surface = scene.particle.free_surface.to_numpy()[0:scene.particleNum[0]]
        mass = scene.particle.m.to_numpy()[0:scene.particleNum[0]]
        normal = scene.particle.normal.to_numpy()[0:scene.particleNum[0]]
        velocity = scene.particle.v.to_numpy()[0:scene.particleNum[0]]
        volume = scene.particle.vol.to_numpy()[0:scene.particleNum[0]]
        radius = scene.particle.rad.to_numpy()[0:scene.particleNum[0]]
        traction = scene.particle.traction.to_numpy()[0:scene.particleNum[0]]
        strain = scene.particle.strain.to_numpy()[0:scene.particleNum[0]]
        stress = scene.particle.stress.to_numpy()[0:scene.particleNum[0]]
        psize = scene.particle.psize.to_numpy()[0:scene.particleNum[0]] 
        velocity_gradient = scene.particle.velocity_gradient.to_numpy()[0:scene.particleNum[0]] 
        fix_v = scene.particle.fix_v.to_numpy()[0:scene.particleNum[0]] 
        state_vars = scene.material.get_state_vars_dict(0, scene.particleNum[0])
        
        xnorm = np.ascontiguousarray(normal[:, 0])
        ynorm = np.ascontiguousarray(normal[:, 1])
        znorm = np.ascontiguousarray(normal[:, 2])

        state_vars.update({"free_surface": free_surface, "normal": (xnorm, ynorm, znorm)})
        self.VisualizeParticle(sims, position, velocity, volume, state_vars)
        np.savez(self.particle_path+f'/MPMParticle{sims.current_print:06d}', t_current=sims.current_time, body_num = particle_num, 
                                                                             bodyID=bodyID, materialID=materialID, active=active, free_surface=free_surface, normal=normal, volume=volume, radius=radius, mass=mass,
                                                                             position=position, velocity=velocity, traction=traction, psize=psize, strain=strain, stress=stress, velocity_gradient=velocity_gradient, fix_v=fix_v, state_vars=state_vars)
        
    def MonitorParticle(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        
        position = scene.particle.x.to_numpy()[0:scene.particleNum[0]]
        bodyID = scene.particle.bodyID.to_numpy()[0:scene.particleNum[0]]
        materialID = scene.particle.materialID.to_numpy()[0:scene.particleNum[0]]
        active = scene.particle.active.to_numpy()[0:scene.particleNum[0]]
        velocity = scene.particle.v.to_numpy()[0:scene.particleNum[0]]
        mass = scene.particle.m.to_numpy()[0:scene.particleNum[0]]
        volume = scene.particle.vol.to_numpy()[0:scene.particleNum[0]]
        traction = scene.particle.traction.to_numpy()[0:scene.particleNum[0]]
        strain = scene.particle.strain.to_numpy()[0:scene.particleNum[0]]
        stress = scene.particle.stress.to_numpy()[0:scene.particleNum[0]]
        psize = scene.particle.psize.to_numpy()[0:scene.particleNum[0]] 
        velocity_gradient = scene.particle.velocity_gradient.to_numpy()[0:scene.particleNum[0]] 
        fix_v = scene.particle.fix_v.to_numpy()[0:scene.particleNum[0]] 
        state_vars = scene.material.get_state_vars_dict(0, scene.particleNum[0])
        
        self.VisualizeParticle(sims, position, velocity, volume, state_vars)
        np.savez(self.particle_path+f'/MPMParticle{sims.current_print:06d}', t_current=sims.current_time, body_num = particle_num, 
                                                                             bodyID=bodyID, materialID=materialID, active=active, mass=mass, volume=volume, position=position, velocity=velocity, 
                                                                             traction=traction, psize=psize, strain=strain, stress=stress, velocity_gradient=velocity_gradient, fix_v=fix_v, state_vars=state_vars)

    def MonitorContactGrid(self, sims: Simulation, scene: myScene):
        coords = scene.element.get_nodal_coords()
        contact_force = scene.node.contact_force.to_numpy()
        norm = scene.node.grad_domain.to_numpy()
        # self.VisualizeGrid(sims, coords, {"contact_force": (fcx, fcy, fcz)})
        np.savez(self.grid_path+f'/MPMGrid{sims.current_print:06d}', t_current=sims.current_time, dims=scene.element.gnum, coords=coords, contact_force=contact_force, normal=norm)

    def MonitorGrid(self, sims: Simulation, scene: myScene):
        coords = scene.element.get_nodal_coords()
        # self.VisualizeGrid(sims, coords, {})
        np.savez(self.grid_path+f'/MPMGrid{sims.current_print:06d}', t_current=sims.current_time, dims=scene.element.gnum, coords=coords)
    
 
