import os

import numpy as np

from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.utils.linalg import no_operation
from third_party.pyevtk.vtk import VtkPolygon
from third_party.pyevtk.hl import pointsToVTK, gridToVTK, unstructuredGridToVTK


class WriteFile:
    def __init__(self, sims):
        self.vtk_path = None
        self.particle_path  = None
        self.grid_path = None

        self.save_particle = no_operation
        self.save_grid = no_operation

        self.mkdir(sims)
        self.manage_function(sims)


    def manage_function(self, sims: Simulation):
        if 'particle' in sims.monitor_type:
            self.visualizeParticle = no_operation
            if sims.visualize:
                if sims.dimension == 2:
                    self.visualizeParticle = self.VisualizeParticle2D
                elif sims.dimension == 3:
                    self.visualizeParticle = self.VisualizeParticle
            if sims.neighbor_detection or sims.coupling:
                if ("Implicit" in sims.solver_type) and (sims.material_type == "Fluid" or sims.material_type == "TwoPhaseDoubleLayer"):
                    self.save_particle = self.MonitorIncompressibleParticleCoupling
                else:
                    self.save_particle = self.MonitorParticleCoupling
            else:
                if sims.material_type == "TwoPhaseSingleLayer":
                    self.save_particle = self.MonitorParticleTwoPhase
                else:
                    if ("Implicit" in sims.solver_type) and (sims.material_type == "Fluid" or sims.material_type == "TwoPhaseDoubleLayer"):
                        self.save_particle = self.MonitorIncompressibleParticle
                    else:
                        self.save_particle = self.MonitorParticle

        if 'grid' in sims.monitor_type:
            self.visualizeGrid = no_operation
            if sims.visualize:
                if sims.dimension == 2:
                    self.visualizeGrid = self.VisualizeGrid2D
                elif sims.dimension == 3:
                    self.visualizeGrid = self.VisualizeGrid
            if sims.contact_detection:
                self.save_grid = self.MonitorContactGrid
            else:
                self.save_grid = self.MonitorGrid

        self.visualizedObject = no_operation
        if 'object' in sims.monitor_type:
            self.visualizedObject = self.VisualizeObject2D

    def output(self, sims, scene):
        self.save_particle(sims, scene)
        self.save_grid(sims, scene)
        self.visualizedObject(sims, scene)

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

    def VisualizeObject2D(self, sims: Simulation, scene: myScene):
        polygon = scene.contact_parameter.polygon_vertices.to_numpy()
        points_flattened = polygon.flatten()
        posx = np.ascontiguousarray(polygon[:, 0])
        posy = np.ascontiguousarray(polygon[:, 1])
        posz = np.zeros(posx.shape[0])
        connectivity = np.arange(polygon.shape[0])
        offsets = np.array([polygon.shape[0]])
        cell_types = np.array([7])
        unstructuredGridToVTK(self.vtk_path+f'/GraphicObject{sims.current_print:06d}', posx, posy, posz, connectivity, offsets, cell_types)

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

    def VisualizeParticle2D(self, sims: Simulation, position, velocity, volume, state_vars):
        posx = np.ascontiguousarray(position[:, 0])
        posy = np.ascontiguousarray(position[:, 1])
        posz = np.zeros(position.shape[0])
        velx = np.ascontiguousarray(velocity[:, 0])
        vely = np.ascontiguousarray(velocity[:, 1])
        velz = np.zeros(velocity.shape[0])
        data = {"velocity": (velx, vely, velz), "volume": volume}
        data.update(state_vars)
        pointsToVTK(self.vtk_path+f'/GraphicMPMParticle{sims.current_print:06d}', posx, posy, posz, data=data)

    def VisualizeGrid(self, sims: Simulation, coords, data):
        coordx = np.unique(np.ascontiguousarray(coords[:, 0]))
        coordy = np.unique(np.ascontiguousarray(coords[:, 1]))
        coordz = np.unique(np.ascontiguousarray(coords[:, 2]))
        gridToVTK(self.vtk_path+f'/GraphicMPMGrid{sims.current_print:06d}', coordx, coordy, coordz, pointData=data)

    def VisualizeGrid2D(self, sims: Simulation, coords, data):
        coordx = np.unique(np.ascontiguousarray(coords[:, 0]))
        coordy = np.unique(np.ascontiguousarray(coords[:, 1]))
        coordz = np.zeros(1)
        gridToVTK(self.vtk_path+f'/GraphicMPMGrid{sims.current_print:06d}', coordx, coordy, coordz, pointData=data)
        
    def MonitorParticleCoupling(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)
        
        #free_surface = scene.particle.free_surface.to_numpy()[0:scene.particleNum[0]]
        #normal = scene.particle.normal.to_numpy()[0:scene.particleNum[0]]
        radius = scene.particle.rad.to_numpy()[0:scene.particleNum[0]]
        stress = scene.particle.stress.to_numpy()[0:scene.particleNum[0]]
        external_force = scene.particle.external_force.to_numpy()[0:scene.particleNum[0]] 
        velocity_gradient = scene.particle.velocity_gradient.to_numpy()[0:scene.particleNum[0]] 
        state_vars: dict = scene.material.get_state_vars_dict(0, scene.particleNum[0])
        output.update({'stress': stress, 'radius': radius, 'external_force': external_force, 'velocity_gradient': velocity_gradient, 'state_vars': state_vars})
        
        #xnorm = np.ascontiguousarray(normal[:, 0])
        #ynorm = np.ascontiguousarray(normal[:, 1])
        #znorm = np.ascontiguousarray(normal[:, 2])

        #state_vars.update({"normal": (xnorm, ynorm, znorm)})
        self.visualizeParticle(sims, output['position'], output['velocity'], output['volume'], state_vars)
        np.savez(self.particle_path+f'/MPMParticle{sims.current_print:06d}', **output)

    def MonitorParticleTwoPhase(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        position = scene.particle.x.to_numpy()[0:scene.particleNum[0]]
        bodyID = scene.particle.bodyID.to_numpy()[0:scene.particleNum[0]]
        materialID = scene.particle.materialID.to_numpy()[0:scene.particleNum[0]]
        active = scene.particle.active.to_numpy()[0:scene.particleNum[0]]
        velocity = scene.particle.v.to_numpy()[0:scene.particleNum[0]]
        solid_velocity = scene.particle.vs.to_numpy()[0:scene.particleNum[0]]
        fluid_velocity = scene.particle.vf.to_numpy()[0:scene.particleNum[0]]
        mass = scene.particle.m.to_numpy()[0:scene.particleNum[0]]
        solid_mass = scene.particle.ms.to_numpy()[0:scene.particleNum[0]]
        fluid_mass = scene.particle.mf.to_numpy()[0:scene.particleNum[0]]
        volume = scene.particle.vol.to_numpy()[0:scene.particleNum[0]]
        stress = scene.particle.stress.to_numpy()[0:scene.particleNum[0]]
        pressure = scene.particle.pressure.to_numpy()[0:scene.particleNum[0]]
        permeability = scene.particle.permeability.to_numpy()[0:scene.particleNum[0]]
        porosity = scene.particle.porosity.to_numpy()[0:scene.particleNum[0]]
        solid_velocity_gradient = scene.particle.solid_velocity_gradient.to_numpy()[0:scene.particleNum[0]] 
        fluid_velocity_gradient = scene.particle.fluid_velocity_gradient.to_numpy()[0:scene.particleNum[0]] 
        fix_v = scene.particle.fix_v.to_numpy()[0:scene.particleNum[0]] 
        state_vars: dict = scene.material.get_state_vars_dict(0, scene.particleNum[0])
        state_vars.update({"pressure": pressure})
        self.visualizeParticle(sims, position, velocity, volume, state_vars)
        np.savez(self.particle_path+f'/MPMParticle{sims.current_print:06d}', t_current=sims.current_time, body_num = particle_num, 
                                                                             bodyID=bodyID, materialID=materialID, active=active, mass=mass, volume=volume, position=position, velocity=velocity, 
                                                                             stress=stress, solid_velocity_gradient=solid_velocity_gradient, fluid_velocity_gradient=fluid_velocity_gradient, fix_v=fix_v, state_vars=state_vars, 
                                                                             solid_velocity=solid_velocity, fluid_velocity=fluid_velocity, solid_mass=solid_mass, fluid_mass=fluid_mass, pressure=pressure, permeability=permeability, porosity=porosity)

    def MonitorIncompressibleParticleCoupling(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)
        
        pressure = scene.particle.pressure.to_numpy()[0:scene.particleNum[0]]
        if sims.dimension == 2:
            xvelocity_gradient = scene.particle.xvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            yvelocity_gradient = scene.particle.yvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            output.update({'xvelocity_gradient': xvelocity_gradient, 'yvelocity_gradient': yvelocity_gradient})
        elif sims.dimension == 3:
            xvelocity_gradient = scene.particle.xvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            yvelocity_gradient = scene.particle.yvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            zvelocity_gradient = scene.particle.zvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            output.update({'xvelocity_gradient': xvelocity_gradient, 'yvelocity_gradient': yvelocity_gradient, 'zvelocity_gradient': zvelocity_gradient})
        state_vars = scene.material.get_state_vars_dict(0, scene.particleNum[0])
        state_vars.update({'pressure': pressure})
        
        self.visualizeParticle(sims, output['position'], output['velocity'], output['volume'], state_vars)
        np.savez(self.particle_path+f'/MPMParticle{sims.current_print:06d}', **output)
        
    def MonitorParticle(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)

        stress = scene.particle.stress.to_numpy()[0:scene.particleNum[0]]
        velocity_gradient = scene.particle.velocity_gradient.to_numpy()[0:scene.particleNum[0]] 
        state_vars = scene.material.get_state_vars_dict(0, scene.particleNum[0])
        output.update({'stress': stress, 'velocity_gradient': velocity_gradient, 'state_vars': state_vars})
        
        self.visualizeParticle(sims, output['position'], output['velocity'], output['volume'], state_vars)
        np.savez(self.particle_path+f'/MPMParticle{sims.current_print:06d}', **output)
        
    def MonitorIncompressibleParticle(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)
        
        pressure = scene.particle.pressure.to_numpy()[0:scene.particleNum[0]]
        '''if sims.dimension == 2:
            xvelocity_gradient = scene.particle.xvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            yvelocity_gradient = scene.particle.yvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            output.update({'xvelocity_gradient': xvelocity_gradient, 'yvelocity_gradient': yvelocity_gradient})
        elif sims.dimension == 3:
            xvelocity_gradient = scene.particle.xvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            yvelocity_gradient = scene.particle.yvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            zvelocity_gradient = scene.particle.zvelocity_gradient.to_numpy()[0:scene.particleNum[0]] 
            output.update({'xvelocity_gradient': xvelocity_gradient, 'yvelocity_gradient': yvelocity_gradient, 'zvelocity_gradient': zvelocity_gradient})'''
        self.visualizeParticle(sims, output['position'], output['velocity'], output['volume'], {'pressure': pressure})
        np.savez(self.particle_path+f'/MPMParticle{sims.current_print:06d}', **output)

    def MonitorParticleBase(self, sims: Simulation, scene: myScene, particle_num):
        position = scene.particle.x.to_numpy()[0:scene.particleNum[0]]
        bodyID = scene.particle.bodyID.to_numpy()[0:scene.particleNum[0]]
        materialID = scene.particle.materialID.to_numpy()[0:scene.particleNum[0]]
        active = scene.particle.active.to_numpy()[0:scene.particleNum[0]]
        velocity = scene.particle.v.to_numpy()[0:scene.particleNum[0]]
        mass = scene.particle.m.to_numpy()[0:scene.particleNum[0]]
        volume = scene.particle.vol.to_numpy()[0:scene.particleNum[0]]
        fix_v = scene.particle.fix_v.to_numpy()[0:scene.particleNum[0]] 
        psize = scene.psize
        return {'t_current': sims.current_time, 'body_num': particle_num, 'active': active, 'bodyID': bodyID, 'materialID': materialID, 'mass': mass, 'volume': volume,
                'position': position, 'velocity': velocity, 'fix_v': fix_v, 'psize': psize}

    def MonitorContactGrid(self, sims: Simulation, scene: myScene):
        coords = scene.element.get_nodal_coords()
        contact_force = scene.node.contact_force.to_numpy()
        norm = scene.node.grad_domain.to_numpy()
        self.visualizeGrid(sims, coords, {})
        np.savez(self.grid_path+f'/MPMGrid{sims.current_print:06d}', t_current=sims.current_time, dims=scene.element.gnum, coords=coords, contact_force=contact_force, normal=norm)

    def MonitorGrid(self, sims: Simulation, scene: myScene):
        coords = scene.element.get_nodal_coords()
        #typex = np.ascontiguousarray(scene.element.boundary_type.to_numpy()[:,0][:,0])
        #typey = np.ascontiguousarray(scene.element.boundary_type.to_numpy()[:,0][:,1])
        self.visualizeGrid(sims, coords, {})
        np.savez(self.grid_path+f'/MPMGrid{sims.current_print:06d}', t_current=sims.current_time, dims=scene.element.gnum, coords=coords)
