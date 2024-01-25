import os

import numpy as np

from src.dem.SceneManager import myScene
from src.dem.contact.ContactModelBase import ContactModelBase 
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.Simulation import Simulation
from third_party.pyevtk.hl import pointsToVTK, gridToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle


class WriteFile:
    physpp: ContactModelBase
    physpw: ContactModelBase
    pcontact: NeighborBase

    def __init__(self, sims, physpp, physpw, pcontact): 
        self.physpp = physpp
        self.physpw = physpw
        self.pcontact = pcontact
        self.vtk_path = None
        self.particle_path  = None
        self.wall_path = None
        self.contact_path = None

        self.save_particle = self.no_operation
        self.save_sphere = self.no_operation
        self.save_clump = self.no_operation
        self.save_wall = self.no_operation
        self.save_servo = self.no_operation
        self.save_ppcontact = self.no_operation
        self.save_pwcontact = self.no_operation
        
        self.mkdir(sims)
        self.manage_function(sims)
    
    def mkdir(self, sims: Simulation):
        if not os.path.exists(sims.path):
            os.makedirs(sims.path)

        self.vtk_path = sims.path + '/vtks'
        self.particle_path = sims.path + '/particles'
        self.wall_path = sims.path + '/walls'
        self.contact_path = sims.path + '/contacts'
        
        if not os.path.exists(self.vtk_path):
            os.makedirs(self.vtk_path)
        if not os.path.exists(self.particle_path):
            os.makedirs(self.particle_path)
        if not os.path.exists(self.wall_path):
            os.makedirs(self.wall_path)
        if not os.path.exists(self.contact_path):
            os.makedirs(self.contact_path)

    def no_operation(self, sims, scene):
        pass

    def manage_function(self, sims: Simulation):
        if 'particle' in sims.monitor_type:
            self.save_particle = self.MonitorParticle
        if sims.max_sphere_num > 0 and 'sphere' in sims.monitor_type:
            self.save_sphere = self.MonitorSphere
        if sims.max_clump_num > 0 and 'clump' in sims.monitor_type:
            self.save_clump = self.MonitorClump

        if 'wall' in sims.monitor_type:
            if sims.wall_type == 0:
                self.save_wall = self.MonitorPlane
            elif sims.wall_type == 1:
                self.save_wall = self.MonitorFacet
                if sims.max_servo_wall_num > 0:
                    self.save_servo = self.MonitorServo
            elif sims.wall_type == 2:
                self.save_wall = self.MonitorPatch
        
        if sims.max_particle_num > 1. and 'ppcontact' in sims.monitor_type:
            self.save_ppcontact = self.MonitorPPContact
        if sims.max_wall_num > 0. and 'pwcontact' in sims.monitor_type:
            self.save_pwcontact = self.MonitorPWContact

    def output(self, sims: Simulation, scene: myScene):
        self.save_particle(sims, scene)
        self.save_sphere(sims, scene)
        self.save_clump(sims, scene)
        self.save_wall(sims, scene)
        self.save_servo(sims, scene)
        self.save_ppcontact(sims, scene)
        self.save_pwcontact(sims, scene)

    def VisualizeDEM(self, sims: Simulation, position, bodyID, groupID, rad):
        posx = np.ascontiguousarray(position[:, 0])
        posy = np.ascontiguousarray(position[:, 1])
        posz = np.ascontiguousarray(position[:, 2])
        pointsToVTK(self.vtk_path+f'/GraphicDEMParticle{sims.current_print:06d}', posx, posy, posz, data={'body': bodyID, 'group': groupID, "radius": rad})


    def VisualizePlane(self, sims: Simulation, scene: myScene):    
        point = np.ascontiguousarray(scene.wall.point.to_numpy()[0:scene.wallNum[0]])
        norm = np.ascontiguousarray(scene.wall.norm.to_numpy()[0:scene.wallNum[0]])
        gridToVTK(self.vtk_path+f'/GraphicDEMWall{sims.current_print:06d}')


    def VisualizeTriangular(self, sims: Simulation, vertice1, vertice2, vertice3):    
        vistri = []
        point1 = np.ascontiguousarray(vertice1)
        point2 = np.ascontiguousarray(vertice2)
        point3 = np.ascontiguousarray(vertice3)
        unique_point = np.unique(np.vstack((point1, point2, point3)), axis=0)
        for i in range(point1.shape[0]):
            vistri.append([int(np.where((unique_point==point1[i]).all(1))[0]), int(np.where((unique_point==point2[i]).all(1))[0]), int(np.where((unique_point==point3[i]).all(1))[0]), 3 * (i + 1), VtkTriangle.tid])
        vispts = np.ascontiguousarray(np.array(unique_point))
        vistri = np.ascontiguousarray(np.array(vistri))

        unstructuredGridToVTK(self.vtk_path+f"/TriangleWall{sims.current_print:06d}", np.ascontiguousarray(vispts[:, 0]), np.ascontiguousarray(vispts[:, 1]), np.ascontiguousarray(vispts[:, 2]), 
                                connectivity=np.ascontiguousarray(vistri[:, 0:3].flatten()), 
                                offsets=np.ascontiguousarray(vistri[:, 3]), 
                                cell_types=np.ascontiguousarray(vistri[:, 4]))


    def MonitorParticle(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]

        active = np.ascontiguousarray(scene.particle.active.to_numpy()[0: particle_num])
        Index = np.ascontiguousarray(scene.particle.multisphereIndex.to_numpy()[0: particle_num])
        groupID = np.ascontiguousarray(scene.particle.groupID.to_numpy()[0: particle_num])
        materialID = np.ascontiguousarray(scene.particle.materialID.to_numpy()[0: particle_num])
        mass = np.ascontiguousarray(scene.particle.m.to_numpy()[0: particle_num])
        radius = np.ascontiguousarray(scene.particle.rad.to_numpy()[0: particle_num])
        position = np.ascontiguousarray(scene.particle.x.to_numpy()[0: particle_num])
        velocity = np.ascontiguousarray(scene.particle.v.to_numpy()[0: particle_num])
        omega = np.ascontiguousarray(scene.particle.w.to_numpy()[0: particle_num])
        contact_force = np.ascontiguousarray(scene.particle.contact_force.to_numpy()[0: particle_num])
        contact_torque = np.ascontiguousarray(scene.particle.contact_torque.to_numpy()[0: particle_num])
        self.VisualizeDEM(sims, position, Index, groupID, radius)
        np.savez(self.particle_path+f'/DEMParticle{sims.current_print:06d}', t_current=sims.current_time, body_num = particle_num, active=active,
                                                                                Index=Index, groupID=groupID, materialID=materialID, mass=mass, radius=radius,
                                                                                position=position, velocity=velocity, omega=omega, contact_force=contact_force, contact_torque=contact_torque)
    
    def MonitorSphere(self, sims: Simulation, scene: myScene):    
        sphere_num = scene.sphereNum[0]
        Index = np.ascontiguousarray(scene.sphere.sphereIndex.to_numpy()[0: sphere_num])
        inverseInertia = np.ascontiguousarray(scene.sphere.inv_I.to_numpy()[0: sphere_num])
        quanternion = np.ascontiguousarray(scene.sphere.q.to_numpy()[0: sphere_num])
        a = np.ascontiguousarray(scene.sphere.a.to_numpy()[0: sphere_num])
        angmoment = np.ascontiguousarray(scene.sphere.angmoment.to_numpy()[0: sphere_num])
        fix_v = np.ascontiguousarray(scene.sphere.fix_v.to_numpy()[0: sphere_num])
        fix_w = np.ascontiguousarray(scene.sphere.fix_w.to_numpy()[0: sphere_num])
        np.savez(self.particle_path+f'/DEMSphere{sims.current_print:06d}', t_current=sims.current_time, body_num = sphere_num, 
                                                                                Index=Index, inverseInertia=inverseInertia, quanternion=quanternion, 
                                                                                acceleration=a, angular_moment=angmoment, fix_v=fix_v, fix_w=fix_w)
        
    def MonitorClump(self, sims: Simulation, scene: myScene):        
        clump_num = scene.clumpNum[0]
        startIndex = np.ascontiguousarray(scene.clump.startIndex.to_numpy()[0: clump_num])
        endIndex = np.ascontiguousarray(scene.clump.endIndex.to_numpy()[0: clump_num])
        mass = np.ascontiguousarray(scene.clump.m.to_numpy()[0: clump_num])
        equivalentRadius = np.ascontiguousarray(scene.clump.equi_r.to_numpy()[0: clump_num])
        centerOfMass = np.ascontiguousarray(scene.clump.mass_center.to_numpy()[0: clump_num])
        velocity = np.ascontiguousarray(scene.clump.v.to_numpy()[0: clump_num])
        omega = np.ascontiguousarray(scene.clump.w.to_numpy()[0: clump_num])
        a = np.ascontiguousarray(scene.clump.a.to_numpy()[0: clump_num])
        angmoment = np.ascontiguousarray(scene.clump.angmoment.to_numpy()[0: clump_num])
        quanternion = np.ascontiguousarray(scene.clump.q.to_numpy()[0: clump_num])
        inverse_inertia = np.ascontiguousarray(scene.clump.inv_I.to_numpy()[0: clump_num])
        np.savez(self.particle_path+f'/DEMClump{sims.current_print:06d}', t_current=sims.current_time, body_num = clump_num, 
                                                                                startIndex=startIndex, endIndex=endIndex, mass=mass, equivalentRadius=equivalentRadius, centerOfMass=centerOfMass,
                                                                                acceleration=a, angular_moment=angmoment, velocity=velocity, omega=omega, quanternion=quanternion, inverse_inertia=inverse_inertia)

    def MonitorPlane(self, sims: Simulation, scene: myScene):  
        active = np.ascontiguousarray(scene.wall.active.to_numpy()[0: scene.wallNum[0]])
        wallID = np.ascontiguousarray(scene.wall.wallID.to_numpy()[0:scene.wallNum[0]])
        materialID = np.ascontiguousarray(scene.wall.materialID.to_numpy()[0:scene.wallNum[0]])
        point = np.ascontiguousarray(scene.wall.point.to_numpy()[0:scene.wallNum[0]])
        norm = np.ascontiguousarray(scene.wall.norm.to_numpy()[0:scene.wallNum[0]])
        np.savez(self.wall_path+f'/DEMWall{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.wallNum[0], active=active,
                                                                     wallID=wallID, materialID=materialID, point=point, norm=norm)

    def MonitorServo(self, sims: Simulation, scene: myScene):  
        active = np.ascontiguousarray(scene.wall.active.to_numpy()[0: scene.wallNum[0]])
        startIndex = np.ascontiguousarray(scene.servo.startIndex.to_numpy()[0:scene.servoNum[0]])
        endIndex = np.ascontiguousarray(scene.servo.endIndex.to_numpy()[0:scene.servoNum[0]])
        alpha = np.ascontiguousarray(scene.servo.alpha.to_numpy()[0:scene.servoNum[0]])
        target_stress = np.ascontiguousarray(scene.servo.target_stress.to_numpy()[0:scene.servoNum[0]])
        max_velocity = np.ascontiguousarray(scene.servo.max_velocity.to_numpy()[0:scene.servoNum[0]])
        np.savez(self.wall_path+f'/DEMServo{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.servoNum[0], active=active,
                                                                      startIndex=startIndex, endIndex=endIndex, alpha=alpha, target_stress=target_stress, max_velocity=max_velocity)    

    def MonitorFacet(self, sims: Simulation, scene: myScene):    
        active = np.ascontiguousarray(scene.wall.active.to_numpy()[0: scene.wallNum[0]])
        wallID = np.ascontiguousarray(scene.wall.wallID.to_numpy()[0:scene.wallNum[0]])
        materialID = np.ascontiguousarray(scene.wall.materialID.to_numpy()[0:scene.wallNum[0]])
        point1 = np.ascontiguousarray(scene.wall.vertice1.to_numpy()[0:scene.wallNum[0]])
        point2 = np.ascontiguousarray(scene.wall.vertice2.to_numpy()[0:scene.wallNum[0]])
        point3 = np.ascontiguousarray(scene.wall.vertice3.to_numpy()[0:scene.wallNum[0]])
        norm = np.ascontiguousarray(scene.wall.norm.to_numpy()[0:scene.wallNum[0]])
        velocity = np.ascontiguousarray(scene.wall.v.to_numpy()[0:scene.wallNum[0]])
        contact_force = np.ascontiguousarray(scene.wall.contact_force.to_numpy()[0:scene.wallNum[0]])
        self.VisualizeTriangular(sims, point1, point2, point3)
        np.savez(self.wall_path+f'/DEMWall{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.wallNum[0], active=active,
                                                                     wallID=wallID, materialID=materialID, point1=point1, point2=point2, point3=point3, 
                                                                     norm=norm, velocity=velocity, contact_force=contact_force)

    def MonitorPatch(self, sims: Simulation, scene: myScene):  
        active = np.ascontiguousarray(scene.wall.active.to_numpy()[0: scene.wallNum[0]])  
        wallID = np.ascontiguousarray(scene.wall.wallID.to_numpy()[0:scene.wallNum[0]])
        materialID = np.ascontiguousarray(scene.wall.materialID.to_numpy()[0:scene.wallNum[0]])
        point1 = np.ascontiguousarray(scene.wall.vertice1.to_numpy()[0:scene.wallNum[0]])
        point2 = np.ascontiguousarray(scene.wall.vertice2.to_numpy()[0:scene.wallNum[0]])
        point3 = np.ascontiguousarray(scene.wall.vertice3.to_numpy()[0:scene.wallNum[0]])
        norm = np.ascontiguousarray(scene.wall.norm.to_numpy()[0:scene.wallNum[0]])
        self.VisualizeTriangular(sims, point1, point2, point3)
        np.savez(self.wall_path+f'/DEMWall{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.wallNum[0], active=active,
                                                                     wallID=wallID, materialID=materialID, point1=point1, point2=point2, point3=point3, norm=norm)
    
    def MonitorPPContact(self, sims: Simulation, scene: myScene): 
        self.physpp.get_ppcontact_output(self.contact_path+'/DEMContactPP', sims.current_time, sims.current_print, scene, self.pcontact)

    def MonitorPWContact(self, sims: Simulation, scene: myScene): 
        self.physpw.get_pwcontact_output(self.contact_path+'/DEMContactPW', sims.current_time, sims.current_print, scene, self.pcontact)
    
    