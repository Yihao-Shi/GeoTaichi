import os

import numpy as np

from src.dem.SceneManager import myScene
from src.dem.contact.ContactModelBase import ContactModelBase 
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.Simulation import Simulation
from src.utils.linalg import no_operation
from third_party.pyevtk.hl import pointsToVTK, gridToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle, VtkQuad


class WriteFile:
    physpp: ContactModelBase
    physpw: ContactModelBase
    pcontact: NeighborBase

    def __init__(self, sims, physpp, physpw, pcontact, engine=None):
        self.physpp = physpp
        self.physpw = physpw
        self.pcontact = pcontact
        self.engine = engine
        self.vtk_path = None
        self.particle_path  = None
        self.wall_path = None
        self.contact_path = None
        self.output = None
        self._affine_snapshot = None
        self._affine_snapshot_id = -1

        self.save_particle = no_operation
        self.save_sphere = no_operation
        self.save_clump = no_operation
        self.save_wall = no_operation
        self.save_servo = no_operation
        self.save_ppcontact = no_operation
        self.save_pwcontact = no_operation
        self.save_body = no_operation
        self.save_grid = no_operation
        self.save_bounding = no_operation
        self.save_surface = no_operation
        self.save_soft_point = no_operation
        
        self.mkdir(sims)
        self.manage_function(sims)

    def set_engine(self, engine):
        self.engine = engine
    
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

    def manage_function(self, sims: Simulation):
        self.visualizeParticle = no_operation
        monitor_type = sims.monitor_type
        if sims.scheme == "DEM":
            if sims.visualize:
                self.visualizeParticle = self.VisualizeDEM
            self.output = self.outputDEM
            if sims.max_particle_num > 0 and 'particle' in monitor_type:
                self.save_particle = self.MonitorParticle
            if sims.max_sphere_num > 0 and 'sphere' in monitor_type:
                self.save_sphere = self.MonitorSphere
            if sims.max_clump_num > 0 and 'clump' in monitor_type:
                self.save_clump = self.MonitorClump
        elif sims.scheme == "LSDEM" or sims.scheme == "LSMPM":
            if sims.visualize:
                self.visualizeParticle = self.VisualizeSurface
            self.output = self.outputLSDEM
            if sims.max_particle_num > 0 and 'particle' in monitor_type:
                self.save_body = self.MonitorLSBody
            if sims.max_level_grid_num > 0 and ('grid' in monitor_type or (sims.scheme == "LSMPM" and 'surface' in monitor_type)):
                self.save_grid = self.MonitorLSGrid
            if sims.max_particle_num > 0 and 'bounding' in monitor_type:
                self.save_bounding = self.MonitorLSBounding
            if sims.max_particle_num > 0 and 'surface' in monitor_type:
                self.save_surface = self.MonitorLSSurface
        elif sims.scheme == "PolySuperEllipsoid" or sims.scheme == "PolySuperQuadrics":
            if sims.visualize:
                self.visualizeParticle = self.VisualizeSurface
            self.output = self.outputImplicitSurface
            if sims.max_rigid_body_num > 0 and 'particle' in monitor_type:
                self.save_body = self.MonitorISBody
            if sims.max_rigid_body_num > 0 and 'bounding' in monitor_type:
                self.save_bounding = self.MonitorBoundingSphere
            if sims.max_particle_num > 0 and 'surface' in monitor_type:
                self.save_surface = self.MonitorImplicitSurface
        elif sims.scheme == "AffineBody":
            monitor_type = list(sims.monitor_type)
            if not getattr(sims, "save_data_selected", False):
                monitor_type = ["particle", "surface"]
                if sims.max_wall_num > 0:
                    monitor_type.append("wall")
            self.output = self.outputAffineBody
            if sims.max_rigid_body_num > 0 and 'particle' in monitor_type:
                self.save_body = self.MonitorAffineBody
            if sims.max_rigid_body_num > 0 and 'bounding' in monitor_type:
                self.save_bounding = self.MonitorAffineBounding
            if sims.max_surface_node_num > 0 and 'surface' in monitor_type:
                self.save_surface = self.MonitorAffineSurface
            if sims.max_particle_num > 1 and 'ppcontact' in monitor_type:
                self.save_ppcontact = self.MonitorAffinePPContact
            if sims.max_particle_num > 0 and sims.max_wall_num > 0 and 'pwcontact' in monitor_type:
                self.save_pwcontact = self.MonitorAffinePWContact

        self.visualizeWall = no_operation
        if 'wall' in monitor_type:
            if sims.wall_type == 0:
                self.save_wall = self.MonitorPlane
            else:
                self.visualizeWall = no_operation
                if sims.visualize:
                    self.visualizeWall = self.VisualizeTriangular
                if sims.wall_type == 1:
                    self.save_wall = self.MonitorFacet
                    if sims.max_servo_wall_num > 0:
                        self.save_servo = self.MonitorServo
                elif sims.wall_type == 2:
                    self.save_wall = self.MonitorPatch
        
        if sims.scheme != "AffineBody":
            if sims.max_particle_num > 1. and 'ppcontact' in monitor_type:
                self.save_ppcontact = self.MonitorPPContact
            if sims.max_particle_num > 0. and sims.max_wall_num > 0. and 'pwcontact' in monitor_type:
                self.save_pwcontact = self.MonitorPWContact

    def outputWall(self, sims: Simulation, scene: myScene):
        self.save_wall(sims, scene)
        self.save_servo(sims, scene)

    def ouptutContact(self, sims: Simulation, scene: myScene):
        self.save_ppcontact(sims, scene)
        self.save_pwcontact(sims, scene)

    def outputImplicitSurface(self, sims: Simulation, scene: myScene):
        self.save_body(sims, scene)
        self.save_bounding(sims, scene)
        self.save_surface(sims, scene)
        self.outputWall(sims, scene)
        self.ouptutContact(sims, scene)

    def outputAffineBody(self, sims: Simulation, scene: myScene):
        self._affine_snapshot = None
        self._affine_snapshot_id = -1
        self.save_body(sims, scene)
        self.save_bounding(sims, scene)
        self.save_surface(sims, scene)
        self.outputWall(sims, scene)
        self.ouptutContact(sims, scene)

    def outputLSDEM(self, sims: Simulation, scene: myScene):
        self.save_body(sims, scene)
        self.save_soft_point(sims, scene)
        self.save_grid(sims, scene)
        self.save_bounding(sims, scene)
        self.save_surface(sims, scene)
        self.outputWall(sims, scene)
        self.ouptutContact(sims, scene)

    def outputDEM(self, sims: Simulation, scene: myScene):
        self.save_particle(sims, scene)
        self.save_sphere(sims, scene)
        self.save_clump(sims, scene)
        self.outputWall(sims, scene)
        self.ouptutContact(sims, scene)

    def VisualizeDEM(self, sims: Simulation, position, bodyID, groupID, rad):
        posx = np.ascontiguousarray(position[:, 0])
        posy = np.ascontiguousarray(position[:, 1])
        posz = np.ascontiguousarray(position[:, 2])
        pointsToVTK(self.vtk_path+f'/GraphicDEMParticle{sims.current_print:06d}', posx, posy, posz, data={'bodyID': bodyID, 'group': groupID, "radius": rad})

    def VisualizePlane(self, sims: Simulation, scene: myScene):    
        point = np.ascontiguousarray(scene.wall.point.to_numpy()[0:scene.wallNum[0]])
        norm = np.ascontiguousarray(scene.wall.norm.to_numpy()[0:scene.wallNum[0]])
        gridToVTK(self.vtk_path+f'/GraphicDEMWall{sims.current_print:06d}')

    def VisualizeTriangular(self, sims: Simulation, vertice1, vertice2, vertice3):   
        ndim = 3 
        point1 = np.ascontiguousarray(vertice1)
        point2 = np.ascontiguousarray(vertice2)
        point3 = np.ascontiguousarray(vertice3)
        points = np.concatenate((point1, point2, point3), axis=1).reshape(-1, ndim)

        point, cell = np.unique(points, axis=0, return_inverse=True)
        faces = cell.reshape((-1, ndim))
        nface = faces.shape[0]
        offset = np.arange(ndim, ndim * nface + 1, ndim)

        unstructuredGridToVTK(self.vtk_path+f"/TriangleWall{sims.current_print:06d}", np.ascontiguousarray(point[:, 0]), np.ascontiguousarray(point[:, 1]), np.ascontiguousarray(point[:, 2]), 
                              connectivity=np.ascontiguousarray(faces.flatten()), 
                              offsets=np.ascontiguousarray(offset), 
                              cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface)))
        
    def VisualizeSurface(self, sims: Simulation, scene: myScene):
        connectivity, keep_node, master, _, _, _ = self._surface_output_view(sims, scene)
        if connectivity.shape[0] == 0 or master.shape[0] == 0:
            return
        ndim = connectivity.shape[1]
        nface = int(connectivity.shape[0])
        npoints = int(master.shape[0])
        pointData, surface = scene.visualize_surface(sims)
        surface = surface[0: int(scene.surfaceNum[0])]
        if keep_node is not None:
            surface = surface[keep_node]
            pointData = {key: np.ascontiguousarray(value[keep_node]) for key, value in pointData.items()}
        posx = np.ascontiguousarray(surface[0: npoints, 0])
        posy = np.ascontiguousarray(surface[0: npoints, 1])
        posz = np.ascontiguousarray(surface[0: npoints, 2])
        unstructuredGridToVTK(self.vtk_path+f"/GraphicLSDEMSurface{sims.current_print:06d}", posx, posy, posz, 
                              connectivity=np.ascontiguousarray(connectivity.flatten()), 
                              offsets=np.ascontiguousarray(np.arange(ndim, ndim * nface + 1, ndim, dtype=np.int32)), 
                              cell_types=np.repeat(VtkTriangle.tid, nface), pointData=pointData)

    def _surface_output_view(self, sims: Simulation, scene: myScene):
        total_surface_num = int(scene.surfaceNum[0])
        master = np.ascontiguousarray(scene.surface.to_numpy()[0: total_surface_num])
        connectivity = np.ascontiguousarray(scene.connectivity)
        keep_node = None
        start_node = None
        node_number = None
        if sims.scheme == "LSMPM" and total_surface_num > 0:
            body_num = int(scene.particleNum[0])
            is_soft = np.ascontiguousarray(scene.rigid.is_soft.to_numpy()[0: body_num]).astype(bool)
            keep_node = np.logical_not(is_soft[master])
            old_to_new = np.full(total_surface_num, -1, dtype=np.int32)
            old_to_new[keep_node] = np.arange(np.count_nonzero(keep_node), dtype=np.int32)
            if connectivity.size > 0:
                keep_face = np.all(keep_node[connectivity], axis=1)
                connectivity = old_to_new[connectivity[keep_face]]
            master = master[keep_node]
            node_number = np.bincount(master, minlength=body_num).astype(np.int32)
            start_node = np.zeros(body_num, dtype=np.int32)
            if body_num > 1:
                start_node[1:] = np.cumsum(node_number[:-1])
        return np.ascontiguousarray(connectivity.astype(np.int32)), keep_node, np.ascontiguousarray(master), start_node, node_number, total_surface_num

    def VisualizeBoundingBox(self, sims: Simulation, scene: myScene):  
        body_num = scene.particleNum[0]
        xmin, xmax = scene.box.xmin.to_numpy(), scene.box.xmax.to_numpy()
        x0, y0, z0 = np.ascontiguousarray(xmin[0:body_num, 0]), np.ascontiguousarray(xmin[0:body_num, 1]), np.ascontiguousarray(xmin[0:body_num, 2])
        x1, y1, z1 = np.ascontiguousarray(xmax[0:body_num, 0]), np.ascontiguousarray(xmax[0:body_num, 1]), np.ascontiguousarray(xmax[0:body_num, 2])
        p1, p2, p3, p4 = np.array([x0, y0, z0]), np.array([x1, y0, z0]), np.array([x0, y1, z0]), np.array([x1, y1, z0])
        p5, p6, p7, p8 = np.array([x0, y0, z1]), np.array([x1, y0, z1]), np.array([x0, y1, z1]), np.array([x1, y1, z1])
        center = np.ascontiguousarray(scene.rigid.mass_center.to_numpy()[0:body_num, :])
        q = np.ascontiguousarray(scene.rigid.q.to_numpy()[0:body_num, :])

        from src.utils.linalg import matrix_from_quanternion
        rotation_matrix = matrix_from_quanternion(q)
        p1 = rotation_matrix @ p1 + center
        p2 = rotation_matrix @ p2 + center
        p3 = rotation_matrix @ p3 + center
        p4 = rotation_matrix @ p4 + center
        p5 = rotation_matrix @ p5 + center
        p6 = rotation_matrix @ p6 + center
        p7 = rotation_matrix @ p7 + center
        p8 = rotation_matrix @ p8 + center

        px = np.zeros((body_num, 8))
        py = np.zeros((body_num, 8))
        pz = np.zeros((body_num, 8))
        px[:, 0], py[:, 0], pz[:, 0] = p1
        px[:, 1], py[:, 1], pz[:, 1] = p2
        px[:, 2], py[:, 2], pz[:, 2] = p3
        px[:, 3], py[:, 3], pz[:, 3] = p4
        px[:, 4], py[:, 4], pz[:, 4] = p5
        px[:, 5], py[:, 5], pz[:, 5] = p6
        px[:, 6], py[:, 6], pz[:, 6] = p7
        px[:, 7], py[:, 7], pz[:, 7] = p8
        # Define connectivity or vertices that belongs to each element
        conn = np.zeros(24)
        conn[0], conn[1], conn[2], conn[3] = 0, 1, 3, 2  # rectangle
        conn[4], conn[5], conn[6], conn[7] = 0, 1, 5, 4  
        conn[8], conn[9], conn[10], conn[11] = 1, 3, 7, 5
        conn[12], conn[13], conn[14], conn[15] = 2, 3, 7, 6
        conn[16], conn[17], conn[18], conn[19] = 2, 0, 4, 6
        conn[20], conn[21], conn[22], conn[23] = 4, 5, 7, 6
        # Define offset of last vertex of each element
        offset = np.zeros(6)
        offset[0], offset[1], offset[2] = 4, 8, 12
        offset[3], offset[4], offset[5] = 16, 20, 24
        # Define cell types
        ctype = np.zeros(6)
        ctype[0], ctype[1], ctype[2] = VtkQuad.tid, VtkQuad.tid, VtkQuad.tid
        ctype[3], ctype[4], ctype[5] = VtkQuad.tid, VtkQuad.tid, VtkQuad.tid
        unstructuredGridToVTK(self.vtk_path+f"/BoundingBox{sims.current_print:06d}", px, py, pz, 
                                connectivity=conn, offsets=offset, cell_types=ctype)

    def _get_affine_snapshot(self, sims: Simulation):
        if self.engine is None:
            raise RuntimeError("AffineBody recorder requires an AffineBodyEngine.")
        if self._affine_snapshot_id != sims.current_print:
            self._affine_snapshot = self.engine.output_snapshot()
            self._affine_snapshot_id = sims.current_print
        return self._affine_snapshot

    def VisualizeAffineSurface(self, sims: Simulation, snapshot):
        vertices = snapshot["vertices"]
        faces = snapshot["faces"]
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            return
        nface = int(faces.shape[0])
        unstructuredGridToVTK(
            self.vtk_path+f"/GraphicAffineBody{sims.current_print:06d}",
            np.ascontiguousarray(vertices[:, 0]),
            np.ascontiguousarray(vertices[:, 1]),
            np.ascontiguousarray(vertices[:, 2]),
            connectivity=np.ascontiguousarray(faces.reshape(-1).astype(np.int32)),
            offsets=np.ascontiguousarray(np.arange(3, 3 * nface + 1, 3, dtype=np.int32)),
            cell_types=np.ascontiguousarray(np.repeat(VtkTriangle.tid, nface).astype(np.uint8)),
            pointData={
                "bodyID": np.ascontiguousarray(snapshot["bodyID"].astype(np.int32)),
                "groupID": np.ascontiguousarray(snapshot["groupID"].astype(np.int32)),
            },
            cellData={
                "bodyID": np.ascontiguousarray(snapshot["faceBodyID"].astype(np.int32)),
                "groupID": np.ascontiguousarray(snapshot["faceGroupID"].astype(np.int32)),
                "materialID": np.ascontiguousarray(snapshot["faceMaterialID"].astype(np.int32)),
            },
        )

    def MonitorAffineBody(self, sims: Simulation, scene: myScene):
        snapshot = self._get_affine_snapshot(sims)
        output = {
            "t_current": sims.current_time,
            "body_num": snapshot["body_num"],
            "vertices": snapshot["vertices"],
            "faces": snapshot["faces"],
            "bodyID": snapshot["bodyID"],
            "groupID": snapshot["groupID"],
            "faceBodyID": snapshot["faceBodyID"],
            "faceGroupID": snapshot["faceGroupID"],
            "faceMaterialID": snapshot["faceMaterialID"],
            "bodyGroupID": snapshot["bodyGroupID"],
            "materialID": snapshot["materialID"],
            "volume": snapshot["volume"],
            "mass_matrix": snapshot["mass_matrix"],
            "young_modulus": snapshot["young_modulus"],
            "friction": snapshot["friction"],
            "forceLocalDamping": snapshot["forceLocalDamping"],
            "torqueLocalDamping": snapshot["torqueLocalDamping"],
            "center": snapshot["center"],
            "y": snapshot["y"],
            "v_y": snapshot["v_y"],
        }
        np.savez(self.particle_path+f'/AffineBody{sims.current_print:06d}', **output)

    def MonitorAffineBounding(self, sims: Simulation, scene: myScene):
        snapshot = self._get_affine_snapshot(sims)
        np.savez(
            self.particle_path+f'/AffineBounding{sims.current_print:06d}',
            t_current=sims.current_time,
            body_num=snapshot["body_num"],
            center=snapshot["center"],
            radius=snapshot["radius"],
            min_box=snapshot["bbox_min"],
            max_box=snapshot["bbox_max"],
            materialID=snapshot["materialID"],
            groupID=snapshot["bodyGroupID"],
        )

    def MonitorAffineSurface(self, sims: Simulation, scene: myScene):
        snapshot = self._get_affine_snapshot(sims)
        self.VisualizeAffineSurface(sims, snapshot)
        np.savez(
            self.particle_path+f'/AffineSurface{sims.current_print:06d}',
            t_current=sims.current_time,
            total_surface_num=snapshot["vertices"].shape[0],
            total_face_num=snapshot["faces"].shape[0],
            vertices=snapshot["vertices"],
            faces=snapshot["faces"],
            bodyID=snapshot["bodyID"],
            groupID=snapshot["groupID"],
            faceBodyID=snapshot["faceBodyID"],
            faceGroupID=snapshot["faceGroupID"],
            faceMaterialID=snapshot["faceMaterialID"],
        )

    def MonitorAffinePPContact(self, sims: Simulation, scene: myScene):
        contact = self.engine.contact_snapshot()
        np.savez(
            self.contact_path+f'/AffineContactPP{sims.current_print:06d}',
            t_current=sims.current_time,
            search=contact["search"],
            vertex_face_contact_num=contact["vf_count"],
            edge_edge_contact_num=contact["ee_count"],
            vertexID=contact["vertex"],
            faceID=contact["face"],
            edgeID0=contact["edge0"],
            edgeID1=contact["edge1"],
        )

    @staticmethod
    def _closest_point_triangle(point, tri):
        a, b, c = tri
        ab = b - a
        ac = c - a
        ap = point - a
        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return a
        bp = point - b
        d3 = np.dot(ab, bp)
        d4 = np.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return b
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / max(d1 - d3, 1.0e-30)
            return a + v * ab
        cp = point - c
        d5 = np.dot(ab, cp)
        d6 = np.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return c
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / max(d2 - d6, 1.0e-30)
            return a + w * ac
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-30)
            return b + w * (c - b)
        denom = max(va + vb + vc, 1.0e-30)
        v = vb / denom
        w = vc / denom
        return a + ab * v + ac * w

    def MonitorAffinePWContact(self, sims: Simulation, scene: myScene):
        snapshot = self._get_affine_snapshot(sims)
        if scene.wall is None or scene.wallNum[0] == 0:
            return
        vertices = snapshot["vertices"]
        wall_raw = scene.wall.to_numpy()
        dhat = float(self.engine.operator.dhat)
        candidate_vertex = []
        candidate_wall = []
        gaps = []
        for wall_id in range(int(scene.wallNum[0])):
            if wall_raw["active"][wall_id] == 0:
                continue
            if sims.wall_type == 0:
                point = np.asarray(wall_raw["point"][wall_id], dtype=np.float64)
                normal = np.asarray(wall_raw["norm"][wall_id], dtype=np.float64)
                norm = np.linalg.norm(normal)
                if norm > 0.0:
                    normal = normal / norm
                distance = (vertices - point) @ normal
                active = np.where(distance < dhat)[0]
                candidate_vertex.extend(active.tolist())
                candidate_wall.extend([wall_id] * active.shape[0])
                gaps.extend(distance[active].tolist())
            elif sims.wall_type in (1, 2):
                tri = np.vstack((
                    wall_raw["vertice1"][wall_id],
                    wall_raw["vertice2"][wall_id],
                    wall_raw["vertice3"][wall_id],
                )).astype(np.float64)
                for vertex_id, point in enumerate(vertices):
                    closest = self._closest_point_triangle(point, tri)
                    gap = float(np.linalg.norm(point - closest))
                    if gap < dhat:
                        candidate_vertex.append(vertex_id)
                        candidate_wall.append(wall_id)
                        gaps.append(gap)
        np.savez(
            self.contact_path+f'/AffineContactPW{sims.current_print:06d}',
            t_current=sims.current_time,
            contact_num=len(candidate_vertex),
            vertexID=np.asarray(candidate_vertex, dtype=np.int32),
            wallID=np.asarray(candidate_wall, dtype=np.int32),
            gap=np.asarray(gaps, dtype=np.float64),
        )

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
        self.visualizeParticle(sims, position, Index, groupID, radius)
        output = {'t_current': sims.current_time, 'body_num': particle_num, 'active': active,
                  'Index': Index, 'groupID': groupID, 'materialID': materialID, 'mass': mass, 'radius': radius,
                  'position': position, 'velocity': velocity, 'omega': omega, 'contact_force': contact_force, 'contact_torque': contact_torque}
        if sims.energy_tracking:
            local_damping_energy = np.ascontiguousarray(scene.particle.damp_energy.to_numpy()[0: particle_num])
            output.update({"local_damping_energy": local_damping_energy})
        np.savez(self.particle_path+f'/DEMParticle{sims.current_print:06d}', **output)
    
    def MonitorSphere(self, sims: Simulation, scene: myScene):    
        sphere_num = scene.sphereNum[0]
        grainIndex = np.ascontiguousarray(scene.sphere.grainIndex.to_numpy()[0: sphere_num])
        sphereIndex = np.ascontiguousarray(scene.sphere.sphereIndex.to_numpy()[0: sphere_num])
        inverseInertia = np.ascontiguousarray(scene.sphere.inv_I.to_numpy()[0: sphere_num])
        quanternion = np.ascontiguousarray(scene.sphere.q.to_numpy()[0: sphere_num])
        a = np.ascontiguousarray(scene.sphere.a.to_numpy()[0: sphere_num])
        angmoment = np.ascontiguousarray(scene.sphere.angmoment.to_numpy()[0: sphere_num])
        fix_v = np.ascontiguousarray(scene.sphere.fix_v.to_numpy()[0: sphere_num])
        fix_w = np.ascontiguousarray(scene.sphere.fix_w.to_numpy()[0: sphere_num])
        np.savez(self.particle_path+f'/DEMSphere{sims.current_print:06d}', t_current=sims.current_time, body_num = sphere_num, 
                                                                                grainIndex=grainIndex, sphereIndex=sphereIndex, inverseInertia=inverseInertia, quanternion=quanternion, 
                                                                                acceleration=a, angular_moment=angmoment, fix_v=fix_v, fix_w=fix_w)
        
    def MonitorClump(self, sims: Simulation, scene: myScene):        
        clump_num = scene.clumpNum[0]
        
        startIndex = np.ascontiguousarray(scene.clump.startIndex.to_numpy()[0: clump_num])
        endIndex = np.ascontiguousarray(scene.clump.endIndex.to_numpy()[0: clump_num])
        grainIndex = np.ascontiguousarray(scene.clump.grainIndex.to_numpy()[0: clump_num])
        mass = np.ascontiguousarray(scene.clump.m.to_numpy()[0: clump_num])
        equivalentRadius = np.ascontiguousarray(scene.clump.equi_r.to_numpy()[0: clump_num])
        centerOfMass = np.ascontiguousarray(scene.clump.mass_center.to_numpy()[0: clump_num])
        velocity = np.ascontiguousarray(scene.clump.v.to_numpy()[0: clump_num])
        omega = np.ascontiguousarray(scene.clump.w.to_numpy()[0: clump_num])
        acceleration = np.ascontiguousarray(scene.clump.a.to_numpy()[0: clump_num])
        angular_moment = np.ascontiguousarray(scene.clump.angmoment.to_numpy()[0: clump_num])
        quanternion = np.ascontiguousarray(scene.clump.q.to_numpy()[0: clump_num])
        inverse_inertia = np.ascontiguousarray(scene.clump.inv_I.to_numpy()[0: clump_num])
        np.savez(self.particle_path+f'/DEMClump{sims.current_print:06d}', t_current=sims.current_time, body_num = clump_num, 
                                                                                grainIndex=grainIndex, startIndex=startIndex, endIndex=endIndex, mass=mass, equivalentRadius=equivalentRadius, centerOfMass=centerOfMass,
                                                                                acceleration=acceleration, angular_moment=angular_moment, velocity=velocity, omega=omega, quanternion=quanternion, inverse_inertia=inverse_inertia)

    def MonitorLSBody(self, sims: Simulation, scene: myScene):
        body_num = scene.particleNum[0]
        groupID = np.ascontiguousarray(scene.rigid.groupID.to_numpy()[0: body_num])
        materialID = np.ascontiguousarray(scene.rigid.materialID.to_numpy()[0: body_num])
        startNode = np.ascontiguousarray(scene.rigid.startNode.to_numpy()[0: body_num])
        endNode = np.ascontiguousarray(scene.rigid.endNode.to_numpy()[0: body_num])
        localNode = np.ascontiguousarray(scene.rigid.localNode.to_numpy()[0: body_num])
        mass = np.ascontiguousarray(scene.rigid.m.to_numpy()[0: body_num])
        equivalentRadius = np.ascontiguousarray(scene.rigid.equi_r.to_numpy()[0: body_num])
        mass_center = np.ascontiguousarray(scene.rigid.mass_center.to_numpy()[0: body_num])
        acceleration = np.ascontiguousarray(scene.rigid.a.to_numpy()[0: body_num])
        velocity = np.ascontiguousarray(scene.rigid.v.to_numpy()[0: body_num])
        omega = np.ascontiguousarray(scene.rigid.w.to_numpy()[0: body_num])
        angular_moment = np.ascontiguousarray(scene.rigid.angmoment.to_numpy()[0: body_num])
        quanternion = np.ascontiguousarray(scene.rigid.q.to_numpy()[0: body_num])
        inverse_inertia = np.ascontiguousarray(scene.rigid.inv_I.to_numpy()[0: body_num])
        contact_force = np.ascontiguousarray(scene.rigid.contact_force.to_numpy()[0: body_num])
        contact_torque = np.ascontiguousarray(scene.rigid.contact_torque.to_numpy()[0: body_num])
        is_fix = np.ascontiguousarray(scene.rigid.is_fix.to_numpy()[0: body_num])
        is_soft = np.ascontiguousarray(scene.rigid.is_soft.to_numpy()[0: body_num])
        scale = np.ascontiguousarray(scene.box.scale.to_numpy()[0: body_num])
        output = {'t_current': sims.current_time, 'body_num': body_num, 
                  'groupID': groupID, 'materialID': materialID, 'startNode': startNode, 'endNode': endNode, 'localNode': localNode, 'scale': scale,
                  'mass': mass, 'equivalentRadius': equivalentRadius, 'mass_center': mass_center, 'acceleration': acceleration, 'angular_moment': angular_moment, 
                  'velocity': velocity, 'omega': omega, 'quanternion': quanternion, 'inverse_inertia': inverse_inertia, 'contact_force': contact_force, 'contact_torque': contact_torque, 'is_fix': is_fix, 'is_soft': is_soft}
        if sims.energy_tracking:
            local_damping_energy = np.ascontiguousarray(scene.rigid.damp_energy.to_numpy()[0: body_num])
            translational_kinetic_energy = 0.5 * mass * np.einsum(
                "ij,ij->i", velocity, velocity
            )
            rotational_kinetic_energy = 0.5 * np.einsum(
                "ij,ij->i", omega, angular_moment
            )
            gravity = np.asarray(sims.gravity, dtype=np.float64)
            gravitational_potential_energy = -mass * (mass_center @ gravity)
            output.update({
                "local_damping_energy": local_damping_energy,
                "translational_kinetic_energy": translational_kinetic_energy,
                "rotational_kinetic_energy": rotational_kinetic_energy,
                "gravitational_potential_energy": gravitational_potential_energy,
            })
        np.savez(self.particle_path+f'/LSDEMRigid{sims.current_print:06d}', **output)

    def MonitorISBody(self, sims: Simulation, scene: myScene):
        body_num = scene.particleNum[0]
        groupID = np.ascontiguousarray(scene.rigid.groupID.to_numpy()[0: body_num])
        materialID = np.ascontiguousarray(scene.rigid.materialID.to_numpy()[0: body_num])
        templateID = np.ascontiguousarray(scene.rigid.templateID.to_numpy()[0: body_num])
        mass = np.ascontiguousarray(scene.rigid.m.to_numpy()[0: body_num])
        equivalentRadius = np.ascontiguousarray(scene.rigid.equi_r.to_numpy()[0: body_num])
        mass_center = np.ascontiguousarray(scene.rigid.mass_center.to_numpy()[0: body_num])
        acceleration = np.ascontiguousarray(scene.rigid.a.to_numpy()[0: body_num])
        velocity = np.ascontiguousarray(scene.rigid.v.to_numpy()[0: body_num])
        omega = np.ascontiguousarray(scene.rigid.w.to_numpy()[0: body_num])
        angular_moment = np.ascontiguousarray(scene.rigid.angmoment.to_numpy()[0: body_num])
        quanternion = np.ascontiguousarray(scene.rigid.q.to_numpy()[0: body_num])
        inverse_inertia = np.ascontiguousarray(scene.rigid.inv_I.to_numpy()[0: body_num])
        contact_force = np.ascontiguousarray(scene.rigid.contact_force.to_numpy()[0: body_num])
        contact_torque = np.ascontiguousarray(scene.rigid.contact_torque.to_numpy()[0: body_num])
        scale = np.ascontiguousarray(scene.rigid.scale.to_numpy()[0: body_num])
        is_fix = np.ascontiguousarray(scene.rigid.is_fix.to_numpy()[0: body_num])
        output = {'t_current': sims.current_time, 'body_num': body_num, 
                  'groupID': groupID, 'materialID': materialID, 'templateID': templateID, 'scale': scale,
                  'mass': mass, 'equivalentRadius': equivalentRadius, 'mass_center': mass_center, 'acceleration': acceleration, 'angular_moment': angular_moment, 
                  'velocity': velocity, 'omega': omega, 'quanternion': quanternion, 'inverse_inertia': inverse_inertia, 'contact_force': contact_force, 'contact_torque': contact_torque, 'is_fix': is_fix}
        if sims.energy_tracking:
            local_damping_energy = np.ascontiguousarray(scene.rigid.damp_energy.to_numpy()[0: body_num])
            output.update({"local_damping_energy": local_damping_energy})
        np.savez(self.particle_path+f'/ImplicitRigid{sims.current_print:06d}', **output)

    def MonitorLSGrid(self, sims: Simulation, scene: myScene):
        total_grid_num = scene.gridNum[0]
        distance_field = np.ascontiguousarray(scene.rigid_grid.distance_field.to_numpy()[0: total_grid_num])
        np.savez(self.particle_path+f'/LSDEMGrid{sims.current_print:06d}', t_current=sims.current_time, total_grid_num=total_grid_num, grid_num=np.asarray(scene.gridID), distance_field=distance_field)

    def MonitorBoundingSphere(self, sims: Simulation, scene: myScene):
        body_num = scene.particleNum[0]
        active = np.ascontiguousarray(scene.particle.active.to_numpy()[0: body_num])
        radius = np.ascontiguousarray(scene.particle.rad.to_numpy()[0: body_num])
        center = np.ascontiguousarray(scene.particle.x.to_numpy()[0: body_num])
        np.savez(self.particle_path+f'/BoundingSphere{sims.current_print:06d}', t_current=sims.current_time, body_num=body_num, active=active, radius=radius, center=center)

    def MonitorBoundingBox(self, sims: Simulation, scene: myScene):
        body_num = scene.particleNum[0]
        min_box = np.ascontiguousarray(scene.box.xmin.to_numpy()[0: body_num])
        max_box = np.ascontiguousarray(scene.box.xmax.to_numpy()[0: body_num])
        shape_min = np.ascontiguousarray(scene.box.shape_min.to_numpy()[0: body_num])
        shape_max = np.ascontiguousarray(scene.box.shape_max.to_numpy()[0: body_num])
        startGrid = np.ascontiguousarray(scene.box.startGrid.to_numpy()[0: body_num])
        grid_num = np.ascontiguousarray(scene.box.gnum.to_numpy()[0: body_num])
        grid_space = np.ascontiguousarray(scene.box.grid_space.to_numpy()[0: body_num])
        scale = np.ascontiguousarray(scene.box.scale.to_numpy()[0: body_num])
        extent = np.ascontiguousarray(scene.box.extent.to_numpy()[0: body_num])
        np.savez(self.particle_path+f'/BoundingBox{sims.current_print:06d}', t_current=sims.current_time, body_num=body_num, min_box=min_box, max_box=max_box, startGrid=startGrid, 
                                                                                  grid_num=grid_num, grid_space=grid_space, scale=scale, extent=extent, shape_min=shape_min, shape_max=shape_max)

    def MonitorImplicitSurface(self, sims: Simulation, scene: myScene):
        total_surface_num = scene.surfaceNum[0]
        connectivity = np.ascontiguousarray(scene.connectivity)
        self.visualizeParticle(sims, scene)
        np.savez(self.particle_path+f'/ImplicitSurface{sims.current_print:06d}', t_current=sims.current_time, total_surface_num=total_surface_num, surface_num=np.asarray(scene.verticeID), face_index=np.asarray(scene.face_index), vertice_index=np.asarray(scene.vertice_index),connectivity=connectivity)

    def MonitorLSBounding(self, sims: Simulation, scene: myScene):
        self.MonitorBoundingSphere(sims, scene)
        self.MonitorBoundingBox(sims, scene)

    def MonitorLSSurface(self, sims: Simulation, scene: myScene):
        connectivity, keep_node, master, start_node, node_number, _ = self._surface_output_view(sims, scene)
        total_surface_num = int(master.shape[0])
        if total_surface_num == 0 or connectivity.shape[0] == 0:
            return
        node_num = int(scene.verticeID[-1])
        if sims.scheme == "LSMPM":
            node_num = int(scene.verticeID[-1] + scene.surfaceNum[0])
        vertices = np.ascontiguousarray(scene.vertice.x.to_numpy()[0: node_num])
        velocities = np.ascontiguousarray(scene.vertice.v.to_numpy()[0: node_num])
        contact_force = np.ascontiguousarray(scene.vertice.contact_force.to_numpy()[0: node_num])
        parameters = np.ascontiguousarray(scene.vertice.parameter.to_numpy()[0: node_num])
        self.visualizeParticle(sims, scene)
        world_vertices = None
        if sims.scheme == "LSMPM":
            _, surface = scene.visualize_surface(sims)
            surface = np.ascontiguousarray(surface[0: int(scene.surfaceNum[0])])
            world_vertices = np.ascontiguousarray(surface[keep_node]) if keep_node is not None else surface[0: total_surface_num]
        output = dict(t_current=sims.current_time, total_surface_num=total_surface_num, surface_num=np.asarray(scene.verticeID), face_index=np.asarray(scene.face_index), 
                      vertice_index=np.asarray(scene.vertice_index), master=master, vertices=vertices, velocity=velocities, contact_force=contact_force, parameters=parameters, connectivity=connectivity)
        if sims.scheme == "LSMPM":
            local_node = np.ascontiguousarray(scene.rigid.localNode.to_numpy()[0: scene.particleNum[0]])
            end_node = start_node + node_number
            output.update(startNode=start_node, endNode=end_node, localNode=local_node, node_number=node_number, world_vertices=world_vertices)
        np.savez(self.particle_path+f'/LSDEMSurface{sims.current_print:06d}', **output)

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
        gain = np.ascontiguousarray(scene.servo.gain.to_numpy()[0:scene.servoNum[0]])
        np.savez(self.wall_path+f'/DEMServo{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.servoNum[0], active=active,
                                                                      startIndex=startIndex, endIndex=endIndex, alpha=alpha, target_stress=target_stress, max_velocity=max_velocity, gain=gain)    

    def MonitorFacet(self, sims: Simulation, scene: myScene):    
        active = np.ascontiguousarray(scene.wall.active.to_numpy()[0: scene.wallNum[0]])
        wallID = np.ascontiguousarray(scene.wall.wallID.to_numpy()[0: scene.wallNum[0]])
        materialID = np.ascontiguousarray(scene.wall.materialID.to_numpy()[0: scene.wallNum[0]])
        point1 = np.ascontiguousarray(scene.wall.vertice1.to_numpy()[0: scene.wallNum[0]])
        point2 = np.ascontiguousarray(scene.wall.vertice2.to_numpy()[0: scene.wallNum[0]])
        point3 = np.ascontiguousarray(scene.wall.vertice3.to_numpy()[0: scene.wallNum[0]])
        norm = np.ascontiguousarray(scene.wall.norm.to_numpy()[0: scene.wallNum[0]])
        velocity = np.ascontiguousarray(scene.wall.v.to_numpy()[0: scene.wallNum[0]])
        contact_force = np.ascontiguousarray(scene.wall.contact_force.to_numpy()[0: scene.wallNum[0]])
        self.visualizeWall(sims, point1, point2, point3)
        np.savez(self.wall_path+f'/DEMWall{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.wallNum[0], active=active,
                                                                     wallID=wallID, materialID=materialID, point1=point1, point2=point2, point3=point3, 
                                                                     norm=norm, velocity=velocity, contact_force=contact_force)

    def MonitorPatch(self, sims: Simulation, scene: myScene):
        active = np.ascontiguousarray(scene.wall.active.to_numpy()[0: scene.wallNum[0]])  
        wallID = np.ascontiguousarray(scene.wall.wallID.to_numpy()[0: scene.wallNum[0]])
        materialID = np.ascontiguousarray(scene.wall.materialID.to_numpy()[0: scene.wallNum[0]])
        shell_offset = np.ascontiguousarray(scene.wall.offset.to_numpy()[0: scene.wallNum[0]])
        point1 = np.ascontiguousarray(scene.wall.vertice1.to_numpy()[0: scene.wallNum[0]])
        point2 = np.ascontiguousarray(scene.wall.vertice2.to_numpy()[0: scene.wallNum[0]])
        point3 = np.ascontiguousarray(scene.wall.vertice3.to_numpy()[0: scene.wallNum[0]])
        norm = np.ascontiguousarray(scene.wall.norm.to_numpy()[0: scene.wallNum[0]])
        self.visualizeWall(sims, point1, point2, point3)
        if sims.enable_shell:
            contact_force = np.ascontiguousarray(scene.wall.contact_force.to_numpy()[0: scene.wallNum[0]])
            contact_torque = np.ascontiguousarray(scene.wall.contact_torque.to_numpy()[0: scene.wallNum[0]])
            np.savez(self.wall_path+f'/DEMWall{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.wallNum[0], active=active, shell_offset=shell_offset,
                                                                     wallID=wallID, materialID=materialID, point1=point1, point2=point2, point3=point3, norm=norm,
                                                                     contact_force=contact_force, contact_torque=contact_torque)
        else:
            np.savez(self.wall_path+f'/DEMWall{sims.current_print:06d}', t_current=sims.current_time, body_num=scene.wallNum[0], active=active, shell_offset=shell_offset,
                                                                     wallID=wallID, materialID=materialID, point1=point1, point2=point2, point3=point3, norm=norm)
    
    def MonitorPPContact(self, sims: Simulation, scene: myScene): 
        self.physpp.get_ppcontact_output(self.contact_path+'/DEMContactPP', sims.current_time, sims.current_print, scene, self.pcontact)

    def MonitorPWContact(self, sims: Simulation, scene: myScene): 
        self.physpw.get_pwcontact_output(self.contact_path+'/DEMContactPW', sims.current_time, sims.current_print, scene, self.pcontact)
    
