import numpy as np
import open3d as o3d
import meshio
import itertools, multiprocessing
import time, math, warnings, types

from functools import partial
from multiprocessing.pool import ThreadPool
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

import src.sdf.progress as progress
import src.sdf.mesh as mesh 
from src.sdf.LevelSetGrid import LocalGrid
from src.sdf.BuildSurfaceNode import RayTracing
from src.sdf.FastMarchingMethod import FastMarchingMethod
from src.utils.linalg import (heaviside_function, transformation_matrix_coordinate_system, generate_grid)
from src.utils.Root import newton
import trimesh as tm
from trimesh.interfaces import gmsh
from third_party.pyevtk.hl import unstructuredGridToVTK, gridToVTK
from third_party.pyevtk.vtk import VtkTriangle, VtkQuad


WORKERS = multiprocessing.cpu_count()
SAMPLES = 2 ** 22
BATCH_SIZE = 32
LThreshold = 1e-9
ILThreshold = 1e9
DBL_EPSILON = 2.2204460492503131e-16
Threshold = 1e-14


class BasicShape(object):
    def __init__(self, ray) -> None:
        self.mesh = None
        self.grid = None
        self.prox = None
        self.tetramesh = None
        self.gauss_point = None
        self._resolution = 50
        self._smear = 1.5
        self._is_bound = False
        self._is_volume = False
        self._is_center = False
        self._is_inertia = False
        self._reset = True
        self.physical_parameter = {}
        
        self.is_simple_shape = 0
        self._ray = ray
        self._volume = 0.
        self._eqradius = 0.
        self._center = np.zeros(3)
        self._inertia = np.zeros(3)
        self._lower_bound = np.zeros(3) - ILThreshold
        self._higher_bound = np.zeros(3) + ILThreshold
        self._sphericity = 1.
        self.calculate_distance = self._distance
        
    def __call__(self, point):
        point = self.transfer(point)
        return self.calculate_distance(point).reshape((-1, 1))
    
    def __or__(self, other):
        return self.union(other)
        
    def __and__(self, other):
        return self.intersection(other)
    
    def __add__(self, other):
        return self.union(other)
        
    def __sub__(self, other):
        return self.difference(other)
    
    def union(self, other, smooth=0.):
        raise NotImplementedError
    
    def difference(self, other, smooth=0.):
        raise NotImplementedError
    
    def intersection(self, other, smooth=0.):
       raise NotImplementedError
    
    def reset(self, reset):
        self._reset = reset
        return self
    
    def resolutions(self, resolution):
        self._resolution = resolution
        return self
    
    def rays(self, ray):
        self._ray = ray
        return self
    
    def smears(self, smear):
        self._smear = smear
        return self
    
    def clear(self):
        del self.mesh, self.grid
    
    def grids(self, space, extent=0):
        if self._is_bound is False:
            self._estimate_bounding_box()
        region_size = self.upper_bound - self.lower_bound

        if space >= np.min(region_size):
            raise RuntimeError(f"/space/ is too large. The minimum bounding box extent is {region_size}")
        
        self.grid = LocalGrid()
        self.grid.read_grid(space=space, extent=int(extent))
        return self

    @property
    def sphericity(self):
        return self._sphericity

    @sphericity.setter
    def sphericity(self, sphericity):
        self._sphericity = sphericity

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        self._center = np.array(center)

    @property
    def ray(self):
        return self._ray

    @ray.setter
    def ray(self, ray):
        self._ray = ray

    @property
    def smear(self):
        if self._is_bound is False:
            self._estimate_bounding_box()
        (x0, y0, z0) = self._lower_bound
        (x1, y1, z1) = self._higher_bound
        dx = (x1 - x0) / self._resolution
        dy = (y1 - y0) / self._resolution
        dz = (z1 - z0) / self._resolution

        if self._smear == 0.:
            smear_coeff = 0.
            warnings.warn("You passed smearCoeff = 0, was that intended ? (there will be no smearing)")
        else:
            smear_coeff = 0.5 * math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / self._smear
        return smear_coeff

    @smear.setter
    def smear(self, smear):
        self._smear = smear

    @property
    def lower_bound(self):
        if self._is_bound is False:
            self._estimate_bounding_box()
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        self._lower_bound = np.array(lower_bound)
        self._is_bound = True

    @property
    def upper_bound(self):
        if self._is_bound is False:
            self._estimate_bounding_box()
        return self._higher_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        self._higher_bound = np.array(upper_bound)
        self._is_bound = True

    @property
    def volume(self):
        if self._is_volume is False:
            self._estimate_volume()
        return self._volume

    @volume.setter
    def volume(self, volume):
        self._volume = volume
        self._is_volume = True

    @property
    def eqradius(self):
        if self._is_volume is False:
            self._estimate_volume()
        return (3./4. * self._volume / math.pi) ** (1./3.)

    @eqradius.setter
    def eqradius(self, eqradius):
        self._eqradius = eqradius

    @property
    def inertia(self):
        if self._is_inertia is False:
            self._estimate_inertia()
        return self._inertia

    @inertia.setter
    def inertia(self, inertia):
        self._inertia = np.array(inertia)
        self._is_inertia = True

    def transfer(self, points):
        points = np.array(points)
        if points.ndim == 1 and points.shape[0] == 3: 
            points = points.reshape((1, 3))
        return points
    
    def _distance(self, p):
        return -self.prox.signed_distance(p)
    
    def mesh_distance(self, p):
        if self._reset:
            return -self.prox.signed_distance(p)
        else:
            return self(p)
    
    def side(self, p):
        raise NotImplementedError
    
    def _normal(self, p):
        raise NotImplementedError
    
    def _estimate_bounding_box(self):
        raise NotImplementedError
    
    def _essential_initialize(self):
        if self._is_volume is False:
            self._get_volume()
        if self._is_center is False:
            self._get_mass_cener()
        if self._is_inertia is False:
            self._get_inertia()
        self._eqradius = (3./4. * self._volume / math.pi) ** (1./3.)
    
    def _estimate_volume(self):
        if self._is_bound is False:
            self._estimate_bounding_box()
        (x0, y0, z0) = self._lower_bound
        (x1, y1, z1) = self._higher_bound
        dx = (x1 - x0) / self._resolution
        dy = (y1 - y0) / self._resolution
        dz = (z1 - z0) / self._resolution

        _, _, _, P = generate_grid(x0 - dx, y0 - dy, z0 - dz, x1 + dx, y1 + dy, z1 + dz, self._resolution + 2)
        self._volume = float(np.sum(heaviside_function(self.smear, -self(P)) * dx * dy * dz, axis=0))
        self._is_volume = True

    def _estimate_mass_center(self):
        if self._is_bound is False:
            self._estimate_bounding_box()
        if self._is_volume is False:
            self._estimate_volume()
        (x0, y0, z0) = self._lower_bound
        (x1, y1, z1) = self._higher_bound
        dx = (x1 - x0) / self._resolution
        dy = (y1 - y0) / self._resolution
        dz = (z1 - z0) / self._resolution

        _, _, _, P = generate_grid(x0 - dx, y0 - dy, z0 - dz, x1 + dx, y1 + dy, z1 + dz, self._resolution + 2)
        self._center = np.sum(heaviside_function(self.smear, -self(P)) * dx * dy * dz * P, axis=0) / self._volume
        self._is_center = True

    def _estimate_inertia(self):
        if self._is_bound is False:
            self._estimate_bounding_box()
        if self._is_center is False:
            self._estimate_mass_center()
        (x0, y0, z0) = self._lower_bound
        (x1, y1, z1) = self._higher_bound
        dx = (x1 - x0) / self._resolution
        dy = (y1 - y0) / self._resolution
        dz = (z1 - z0) / self._resolution

        _, _, _, P = generate_grid(x0 - dx, y0 - dy, z0 - dz, x1 + dx, y1 + dy, z1 + dz, self._resolution + 2)
        h = heaviside_function(self.smear, -self(P)) * dx * dy * dz
        inertia_xx = np.sum(h[:, 0] * ((P[:, 1] - self._center[1]) * (P[:, 1] - self._center[1]) + (P[:, 2] - self._center[2]) * (P[:, 2] - self._center[2])), axis=0)
        inertia_yy = np.sum(h[:, 0] * ((P[:, 0] - self._center[0]) * (P[:, 0] - self._center[0]) + (P[:, 2] - self._center[2]) * (P[:, 2] - self._center[2])), axis=0)
        inertia_zz = np.sum(h[:, 0] * ((P[:, 0] - self._center[0]) * (P[:, 0] - self._center[0]) + (P[:, 1] - self._center[1]) * (P[:, 1] - self._center[1])), axis=0)
        inertia_xy = -np.sum(h[:, 0] * (P[:, 1] - self._center[1]) * (P[:, 2] - self._center[2]), axis=0)
        inertia_yz = -np.sum(h[:, 0] * (P[:, 0] - self._center[0]) * (P[:, 2] - self._center[2]), axis=0)
        inertia_xz = -np.sum(h[:, 0] * (P[:, 0] - self._center[0]) * (P[:, 1] - self._center[1]), axis=0)

        inertia = np.array([inertia_xx, inertia_yy, inertia_zz, inertia_xy, inertia_yz, inertia_xz])
        diagI = np.array([inertia_xx, inertia_yy, inertia_zz, 0., 0., 0.])
        ratio = np.linalg.norm(inertia - diagI) / np.linalg.norm(diagI)
        if ratio > 5e-3:
            raise RuntimeError(f"Incorrect LevelSet input: local frame is non-inertial.",
                               f"Indeed, Ixx = {inertia[0]}; Iyy = {inertia[1]}; Izz = {inertia[1]}; Ixy = {inertia[3]}; Ixz = {inertia[4]}; Iyz = {inertia[5]}",
		                       f" for inertia matrix I, making for a {ratio} non-diagonality ratio.")
        else:
            self._inertia = np.array([inertia_xx, inertia_yy, inertia_zz])
        self._is_inertia = True

    def _get_bounding_box(self):
        box_center = self.mesh.bounding_box.transform[0:3, 3]
        self._lower_bound = box_center -0.5 * self.mesh.bounding_box.extents
        self._higher_bound = box_center + 0.5 * self.mesh.bounding_box.extents
        self._is_bound = True

    def _get_volume(self):
        self._volume = self.mesh.volume
        self._is_volume = True

    def _get_mass_cener(self):
        self._center = self.mesh.center_mass
        self._is_center = True

    def _get_inertia(self):
        moment_inertia = self.mesh.moment_inertia
        inertia_xx = moment_inertia[0, 0]
        inertia_yy = moment_inertia[1, 1]
        inertia_zz = moment_inertia[2, 2]
        inertia_xy = moment_inertia[0, 1] + moment_inertia[1, 0]
        inertia_xz = moment_inertia[0, 2] + moment_inertia[2, 0]
        inertia_yz = moment_inertia[1, 2] + moment_inertia[2, 1]
        inertia = np.array([inertia_xx, inertia_yy, inertia_zz, inertia_xy, inertia_xz, inertia_yz])
        diagI = np.array([inertia_xx, inertia_yy, inertia_zz, 0., 0., 0.])
        ratio = np.linalg.norm(inertia - diagI) / np.linalg.norm(diagI)
        if ratio > 1e-2:
            raise RuntimeError(f"Incorrect LevelSet input: local frame is non-inertial.",
                               f"Indeed, Ixx = {inertia[0]}; Iyy = {inertia[1]}; Izz = {inertia[1]}; Ixy = {inertia[3]}; Ixz = {inertia[4]}; Iyz = {inertia[5]}",
		                       f" for inertia matrix I, making for a {ratio} non-diagonality ratio.")
        else:
            self._inertia = np.array([inertia_xx, inertia_yy, inertia_zz])
        self._is_inertia = True

    def mesh_operation(self):
        self.mesh.apply_translation(-self.mesh.center_mass)
        if self._reset:
            _, new_axis = tm.inertia.principal_axis(self.mesh.moment_inertia)
            default_axis = np.eye(3)
            rotation_matrix = transformation_matrix_coordinate_system(new_axis, default_axis)
            self.mesh.apply_transform(rotation_matrix)
        self.generate_sdf()

    def split(self, iteration=1):
        if self.mesh is None:
            raise RuntimeError("Primitive should be generate first")
        pass

    def generate_sdf(self, estimate=False):
        if estimate:
            self._estimate_bounding_box()
        else:
            self._get_bounding_box()
        region_size = self.upper_bound - self.lower_bound
        self.grid.clear()
        self.grid.set_grid(self.lower_bound, region_size)
        self.grid.build_node_coords()
        if estimate:
            self.grid.generate_sdf(self(self.grid.node_coord).reshape(-1))
        else:
            self.grid.generate_sdf(self.mesh_distance(self.grid.node_coord).reshape(-1))

    def construct_surface_mesh_from_point_cloud(self, points, estimate_normals=False):
        pcd = o3d.geometry.PointCloud()
        if not estimate_normals:
            normals = self._normal(points)
        else:
            normals = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # ball pivoting method
        distances = pcd.compute_nearest_neighbor_distance()
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        meshs = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(np.linspace(0.2*min_dist, 8*max_dist, 8)))
        meshs.remove_degenerate_triangles()
        meshs.remove_duplicated_triangles()
        meshs.remove_duplicated_vertices()
        meshs.remove_non_manifold_edges()
        return meshs

    def ray_tracing(self, samples, ray_path):
        rayTracing = RayTracing()
        rayTracing.determine_node_path(ray_path)
        rayTracing.add_essentials(samples, self.grid)
        rayTracing.run(self.center, self.is_simple_shape, self)
        points = np.array(rayTracing.surface_node.copy())
        meshs = self.construct_surface_mesh_from_point_cloud(points)
        self.mesh = tm.Trimesh(vertices=np.asarray(meshs.vertices), faces=np.asarray(meshs.triangles))
        self.prox = tm.proximity.ProximityQuery(self.mesh)
        self.mesh_operation()

    def marching_cube(self, step, samples, workers, batch_size, verbose, sparse):
        start = time.time()

        region_size = self.upper_bound - self.lower_bound
        (x0, y0, z0) = self.lower_bound - 0.1 * region_size
        (x1, y1, z1) = self.upper_bound + 0.1 * region_size

        if step is None and samples is not None:
            volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
            step = (volume / samples) ** (1 / 3)

        try:
            dx, dy, dz = step
        except TypeError:
            dx = dy = dz = step

        if verbose:
            print('Lower Bounding = %g, %g, %g' % (x0, y0, z0))
            print('Upper Bounding = %g, %g, %g' % (x1, y1, z1))
            print('Grid Space = %g, %g, %g' % (dx, dy, dz))

        X = np.arange(x0, x1, dx)
        Y = np.arange(y0, y1, dy)
        Z = np.arange(z0, z1, dz)

        s = batch_size
        Xs = [X[i: i + s + 1] for i in range(0, len(X), s)]
        Ys = [Y[i: i + s + 1] for i in range(0, len(Y), s)]
        Zs = [Z[i: i + s + 1] for i in range(0, len(Z), s)]

        batches = list(itertools.product(Xs, Ys, Zs))
        num_batches = len(batches)
        num_samples = sum(len(xs) * len(ys) * len(zs) for xs, ys, zs in batches)

        if verbose:
            print('%d samples in %d batches with %d workers' %(num_samples, num_batches, workers))

        points = []
        skipped = empty = nonempty = 0
        bar = progress.Bar(num_batches, enabled=verbose)
        pool = ThreadPool(workers)
        f = partial(mesh._worker, self, sparse=sparse)
        for result in pool.imap(f, batches):
            bar.increment(1)
            if result is None:
                skipped += 1
            elif len(result) == 0:
                empty += 1
            else:
                nonempty += 1
                points.extend(result)
        bar.done()
        pool.close()
        pool.join()

        if verbose:
            print('%d skipped, %d empty, %d nonempty' % (skipped, empty, nonempty))
            triangles = len(points) // 3
            seconds = time.time() - start
            print('%d triangles in %g seconds' % (triangles, seconds))

        points, cells = np.unique(points, axis=0, return_inverse=True)
        faces = cells.reshape((-1, 3))
        self.mesh = tm.Trimesh(vertices=points, faces=faces)
        self.prox = tm.proximity.ProximityQuery(self.mesh)
        self.mesh_operation()
    
    def generate(self, step=None, samples=SAMPLES, workers=WORKERS, batch_size=BATCH_SIZE, verbose=True, sparse=True, ray_path='Spiral'):
        if self.grid is None:
            raise RuntimeError("Local grid for SDF primitive has not been defined, Use objects.grids(space) first!")
        if self.mesh is None:
            if self.ray is False:
                self.marching_cube(step, samples, workers, batch_size, verbose, sparse)
            else:
                self.ray_tracing(samples, ray_path)
        
    def save(self, path, step=None, samples=SAMPLES, workers=WORKERS, batch_size=BATCH_SIZE, verbose=True, sparse=True, state=False, ray_path='Spiral'):
        self.generate(step, samples, workers, batch_size, verbose, sparse, ray_path)
        success = self.mesh.export(path)
        if state:
            return self.mesh.vertices, self.mesh.faces, success
        else:
            return self.mesh.vertices, self.mesh.faces
    
    def show(self, points=None, faces=None, step=None, samples=SAMPLES, workers=WORKERS, batch_size=BATCH_SIZE, verbose=True, sparse=True, ray_path='Spiral'):
        if points is None or faces is None:
            self.generate(step, samples, workers, batch_size, verbose, sparse, ray_path)
            points = self.mesh.vertices
            faces = self.mesh.faces
        fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw=dict(projection='3d'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        face = points[faces]
        for f in face:
            face = Axes3D.art3d.Poly3DCollection([f])
            face.set_color(mpl.colors.rgb2hex(np.random.rand(3)))
            face.set_edgecolor('k')
            face.set_alpha(0.5)
            ax.add_collection3d(face)
        ax.plot(points[:,0], points[:,1], points[:,2], 'o')
        
    def show_slice(self, show_abs=False, w=1024, h=1024, x=None, y=None, z=None):
        a, extent, axes = mesh.sample_slice(self, w, h, x, y, z)
        if show_abs:
            a = np.abs(a)
        im = plt.imshow(a, extent=extent, origin='lower')
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
        plt.colorbar(im)
        plt.show()

    def dump_files(self, path='./', pname='LSParticle', gname='LSGrid', particle=True, grid=True):
        if particle:
            np.savez(path+pname, vertices=self.mesh.vertices, faces=self.mesh.faces)

        if grid:
            np.savez(path+gname, coordinate=self.grid.node_coord, distance_field=self.grid.distance_field, 
                     start_point=self.grid.start_point, region_size=self.grid.region_size, grid_number=self.grid.gnum, 
                     grid_space=self.grid.grid_space, extent=self.grid.extent)

    def visualize(self, path='./', pname='LSParticle', gname='LSGrid', bname='LSBoundingBox', particle=True, grid=True, bounding_box=True):
        if particle:
            ndim = 3
            nface = self.mesh.faces.shape[0]
            posx = np.ascontiguousarray(self.mesh.vertices[:, 0])
            posy = np.ascontiguousarray(self.mesh.vertices[:, 1])
            posz = np.ascontiguousarray(self.mesh.vertices[:, 2])
            unstructuredGridToVTK(path+pname, posx, posy, posz, 
                                 connectivity=np.ascontiguousarray(self.mesh.faces.flatten()), 
                                 offsets=np.ascontiguousarray(np.arange(ndim, ndim * nface + 1, ndim)), 
                                 cell_types=np.repeat(VtkTriangle.tid, nface))

        if grid:
            sdf = np.ascontiguousarray(self.grid.distance_field)
            posx = np.ascontiguousarray(np.unique(self.grid.node_coord[:, 0]))
            posy = np.ascontiguousarray(np.unique(self.grid.node_coord[:, 1]))
            posz = np.ascontiguousarray(np.unique(self.grid.node_coord[:, 2]))
            gridToVTK(path+gname, posx, posy, posz, pointData=({'distance': sdf}))

        if bounding_box:
            x0, y0, z0 = self.lower_bound
            x1, y1, z1 = self.upper_bound
            px = np.zeros(8)
            py = np.zeros(8)
            pz = np.zeros(8)
            px[0], py[0], pz[0] = x0, y0, z0
            px[1], py[1], pz[1] = x1, y0, z0
            px[2], py[2], pz[2] = x0, y1, z0
            px[3], py[3], pz[3] = x1, y1, z0
            px[4], py[4], pz[4] = x0, y0, z1
            px[5], py[5], pz[5] = x1, y0, z1
            px[6], py[6], pz[6] = x0, y1, z1
            px[7], py[7], pz[7] = x1, y1, z1
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
            unstructuredGridToVTK(path+bname, px, py, pz, connectivity=conn, offsets=offset, cell_types=ctype)

    def tetrahedization(self, max_length=None, algorithm=1, refine_number=0):
        if self.tetramesh is None:
            gmsh.to_volume(self.mesh, "tetraObject.msh", max_length, algorithm, refine_number)
            self.tetramesh = meshio.read("tetraObject.msh")


class pointcloud(BasicShape):
    def __init__(self, points) -> None:
        super().__init__(ray=False)
        meshs = self.construct_surface_mesh_from_point_cloud(points, estimate_normals=True)
        self.mesh = tm.Trimesh(vertices=np.asarray(meshs.vertices), faces=np.asarray(meshs.triangles))
        self.prox = tm.proximity.ProximityQuery(self.mesh)

    def generate(self, *args, **kwargs):
        self.generate_sdf()
        self.mesh_operation()


class MeshBased(BasicShape):
    def __init__(self):
        super().__init__(ray=False)

    def _estimate_bounding_box(self):
        self._get_bounding_box()
    
    def side(self, p):
        p = self.transfer(p)
        return self.mesh.contains(p)
    
    def generate(self, *args, **kwargs):
        self.mesh_operation()

    def _normal(self, p):
        pass


class arbitrarily(MeshBased):
    def __init__(self, start_point, region_size, distance_field, node_coordinate, vertices, faces) -> None:
        super().__init__()
        self.is_simple_shape = 1
        self.mesh = tm.Trimesh(vertices=vertices, faces=faces)
        self.prox = tm.proximity.ProximityQuery(self.mesh)
        self.grid = LocalGrid()
        self.grid.set_distance_field(np.array(start_point), np.array(region_size), 
                                     np.array(node_coordinate), np.array(distance_field), self.mesh.center_mass)


class polyhedron(MeshBased):
    def __init__(self, file: str) -> None:
        super().__init__()
        self.is_simple_shape = 1
        self.mesh: tm.Trimesh = tm.load(file, force='mesh')
        self.prox = tm.proximity.ProximityQuery(self.mesh)


class Surface(BasicShape):
    def __init__(self) -> None:
        super().__init__(ray=True)
        self.is_simple_shape = 2

    def fx(self, p):
        raise NotImplementedError
    
    def dfx(self, p):
        raise NotImplementedError
    
    def radial(self, theta, phi):
        raise NotImplementedError
    
    def _normal(self, p):
        return self.dfx(p)
    
    def side(self, p):
        p = self.transfer(p)
        field = self.fx(p)
        return np.select([field > Threshold, field < -Threshold], [math.inf, -math.inf], default=0.)
    
    def _approximate_distance(self, p):
        # reference: Gabriel Taubin (1994). Distance approximations for rasterizing implicit curves. ACM Transactions on Graphics, 13(1), 3-42.
        # First order approximation
        fxyz = self.fx(p)
        grad_f = self.dfx(p)
        dnorm = np.select([np.linalg.norm(grad_f) > (self.upper_bound - self.lower_bound) * DBL_EPSILON], [np.linalg.norm(grad_f)], default=1.)
        returnVal = np.select([np.abs(fxyz) > DBL_EPSILON], [fxyz / dnorm], default=0.)
        return returnVal
    
    def fast_marching(self):
        nodal_coords = self.grid.node_coord.copy()
        step = self.grid.grid_space
        gnum = self.grid.gnum.copy()
        fmm = FastMarchingMethod(step, gnum, nodal_coords)
        fmm.phiIni(self)
        fmm.phi()
        self.grid.generate_sdf(fmm.phiField.copy().reshape(-1))
        fmm.clear()
    

class surfacefunction(Surface):
    def __init__(self, f=None, df=None, radial=None, bounding_box=None) -> None:
        super().__init__()
        self.fx = types.MethodType(f, self)
        self.dfx = types.MethodType(df, self)
        self.radial = types.MethodType(radial, self)
        self._estimate_bounding_box = types.MethodType(bounding_box, self)


class SuperSurface(Surface):
    def __init__(self, xrad1, yrad1, zrad1, xrad2=None, yrad2=None, zrad2=None) -> None:
        super().__init__()
        self.xrad1 = xrad1
        self.yrad1 = yrad1
        self.zrad1 = zrad1
        self.xrad2 = xrad2 or xrad1
        self.yrad2 = yrad2 or yrad1
        self.zrad2 = zrad2 or zrad1

    def _estimate_bounding_box(self):
        self._lower_bound = -np.array([self.xrad1, self.yrad1, self.zrad1])
        self._higher_bound = np.array([self.xrad2, self.yrad2, self.zrad2])
        self._is_bound = True


class polysuperellipsoid(SuperSurface):
    def __init__(self, xrad1, yrad1, zrad1, epsilon_e, epsilon_n, xrad2=None, yrad2=None, zrad2=None) -> None:
        super().__init__(xrad1, yrad1, zrad1, xrad2, yrad2, zrad2)
        self.epsilon_e = epsilon_e
        self.epsilon_n = epsilon_n
        self.physical_parameter.update({"xrad1": self.xrad1, "yrad1": self.yrad1, "zrad1": self.zrad1, 
                                        "epsilon_e": self.epsilon_e, "epsilon_n": self.epsilon_n, 
                                        "xrad2": self.xrad2, "yrad2": self.yrad2, "zrad2": self.zrad2})

        if epsilon_e == epsilon_n == 0:
            raise ValueError("Please define non zero epsilons for superellipsoid shape.")
        self._reset = False
        
    def fx(self, p):
        p = self.transfer(p)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        funcs = np.array(pow(pow(abs(x / np.where(x < 0., self.xrad1, self.xrad2)), 2. / self.epsilon_e) \
                    + pow(abs(y / np.where(y < 0., self.yrad1, self.yrad2)), 2. / self.epsilon_e), self.epsilon_e / self.epsilon_n) \
                    + pow(abs(z / np.where(z < 0., self.zrad1, self.zrad2)), 2. / self.epsilon_n) - 1.)
        return funcs.reshape(-1)
    
    def dfx(self, p):
        p = self.transfer(p)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        xrad = np.where(x < 0., -1. / self.xrad1, 1. / self.xrad2)
        yrad = np.where(y < 0., -1. / self.yrad1, 1. / self.yrad2)
        zrad = np.where(z < 0., -1. / self.zrad1, 1. / self.zrad2)
        mult = 2. / self.epsilon_n * pow(pow(abs(x * xrad), 2. / self.epsilon_e) + \
                                         pow(abs(y * yrad), 2. / self.epsilon_e), self.epsilon_e / self.epsilon_n - 1.)
        grad_f = np.array([mult * pow(abs(x * xrad), 2. / self.epsilon_e - 1.) * xrad,
                           mult * pow(abs(y * yrad), 2. / self.epsilon_e - 1.) * yrad,
                           2. / self.epsilon_n * pow(abs(z * zrad), 2. / self.epsilon_n - 1) * zrad]).T
        return (grad_f.T / np.linalg.norm(grad_f, axis=1)).T.reshape((-1, 3))
    
    def radial(self, theta, phi):
        para1 = np.sin(theta) * np.cos(phi)
        para2 = np.sin(theta) * np.sin(phi)
        para3 = np.cos(theta)

        coeff1 = np.where(para1 < 0., -1. / self.xrad1, 1. / self.xrad2) 
        coeff2 = np.where(para2 < 0., -1. / self.yrad1, 1. / self.yrad2) 
        coeff3 = np.where(para3 < 0., -1. / self.zrad1, 1. / self.zrad2) 
        r_dist = 1. / ((np.abs(para1 * coeff1) ** (2. / self.epsilon_e) + np.abs(para2 * coeff2) ** (2. / self.epsilon_e)) ** (self.epsilon_n / self.epsilon_e) + \
                       np.abs(para3 * coeff3) ** (2. / self.epsilon_n))
        return r_dist ** (0.5 * self.epsilon_n)


class polysuperquadrics(SuperSurface):
    def __init__(self, xrad1, yrad1, zrad1, epsilon_x, epsilon_y, epsilon_z, xrad2=None, yrad2=None, zrad2=None) -> None:
        super().__init__(xrad1, yrad1, zrad1, xrad2, yrad2, zrad2)
        self.epsilon_x = epsilon_x
        self.epsilon_y = epsilon_y
        self.epsilon_z = epsilon_z
        self.physical_parameter.update({"xrad1": self.xrad1, "yrad1": self.yrad1, "zrad1": self.zrad1, 
                                        "epsilon_x": self.epsilon_x, "epsilon_y": self.epsilon_y, "epsilon_z": self.epsilon_z, 
                                        "xrad2": self.xrad2, "yrad2": self.yrad2, "zrad2": self.zrad2})


        if epsilon_x == epsilon_y == epsilon_z == 0:
            raise ValueError("Please define non zero epsilons for polysuperquadrics shape.")
        self._reset = False
        
    def _estimate_bounding_box(self):
        self._lower_bound = -np.array([self.xrad1, self.yrad1, self.zrad1])
        self._higher_bound = np.array([self.xrad2, self.yrad2, self.zrad2])
        self._is_bound = True
        
    def fx(self, p):
        p = self.transfer(p)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        funcs = np.array(pow(abs(x / np.where(x < 0., self.xrad1, self.xrad2)), 2. / self.epsilon_x) \
                + pow(abs(y / np.where(y < 0., self.yrad1, self.yrad2)), 2. / self.epsilon_y) \
                + pow(abs(z / np.where(z < 0., self.zrad1, self.zrad2)), 2. / self.epsilon_z) - 1.)
        return funcs.reshape(-1)
    
    def dfx(self, p): 
        p = self.transfer(p)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        xrad = np.where(x < 0., -1. / self.xrad1, 1. / self.xrad2)
        yrad = np.where(y < 0., -1. / self.yrad1, 1. / self.yrad2)
        zrad = np.where(z < 0., -1. / self.zrad1, 1. / self.zrad2)
        grad_f = np.array([2 / self.epsilon_x * pow(abs(x * xrad), 2. / self.epsilon_x - 1.) * xrad,
                           2 / self.epsilon_y * pow(abs(y * yrad), 2. / self.epsilon_y - 1.) * yrad,
                           2 / self.epsilon_z * pow(abs(z * zrad), 2. / self.epsilon_z - 1.) * zrad]).T
        return (grad_f.T / np.linalg.norm(grad_f, axis=1)).T.reshape((-1, 3))
    
    def radial(self, theta, phi):
        para1 = np.sin(theta) * np.cos(phi)
        para2 = np.sin(theta) * np.sin(phi)
        para3 = np.cos(theta)

        coeff1 = pow(np.abs(para1 * np.where(para1 < 0., -1. / self.xrad1, 1. / self.xrad2)), 2. / self.epsilon_x)
        coeff2 = pow(np.abs(para2 * np.where(para2 < 0., -1. / self.yrad1, 1. / self.yrad2)), 2. / self.epsilon_y)
        coeff3 = pow(np.abs(para3 * np.where(para3 < 0., -1. / self.zrad1, 1. / self.zrad2)), 2. / self.epsilon_z)
        
        def f(r, a, b, c, ex, ey, ez):
            return a * pow(r, 2. / ex) + b * pow(r, 2. / ey) + c * pow(r, 2. / ez) - 1.
        
        def df(r, a, b, c, ex, ey, ez):
            return a * (2. / ex) * pow(r, 2. / ex - 1) + b * (2. / ey) * pow(r, 2. / ey - 1) + c * (2. / ez) * pow(r, 2. / ez - 1)
        
        start_value = min(self.xrad1, self.xrad2, self.yrad1, self.yrad2, self.zrad1, self.zrad2)
        root = newton(func=f, x0=start_value, args=(coeff1, coeff2, coeff3, self.epsilon_x, self.epsilon_y, self.epsilon_z), fprime=df)
        return root


class SDF(BasicShape):
    def __init__(self, ray=False):
        super().__init__(ray)
        self._smooth = 0.
        self.is_simple_shape = 3
    
    @property
    def smooth(self):
        return self._smooth
    
    @smooth.setter
    def smooth(self, smooth):
        self._smooth = smooth

    def smooths(self, smooth):
        self._smooth = smooth
        return self
    
    def _distance(self, p):
        raise NotImplementedError
    
    def _normal(self, p):
        point = self.transfer(p)
        space = np.min(self.upper_bound - self.lower_bound)
        gapnx = self._distance(point + np.array([LThreshold * space, 0, 0])).reshape(-1)
        gapny = self._distance(point + np.array([0, LThreshold * space, 0])).reshape(-1)
        gapnz = self._distance(point + np.array([0, 0, LThreshold * space])).reshape(-1)
        gapnx_ = self._distance(point - np.array([LThreshold * space, 0, 0])).reshape(-1)
        gapny_ = self._distance(point - np.array([0, LThreshold * space, 0])).reshape(-1)
        gapnz_ = self._distance(point - np.array([0, 0, LThreshold * space])).reshape(-1)
        normals = 0.5 * ILThreshold * np.array([gapnx - gapnx_, gapny - gapny_, gapnz - gapnz_]).T / space
        return np.where(np.linalg.norm(normals, axis=1) > 0., (normals.T / np.linalg.norm(normals, axis=1)), 0.).T

    def _estimate_bounding_box(self):
        # TODO: raise exception if bound estimation fails
        s = 31
        (x0, y0, z0) = self._lower_bound
        (x1, y1, z1) = self._higher_bound
        prev = None
        for _ in range(64):
            X, Y, Z, P = generate_grid(x0, y0, z0, x1, y1, z1, s)
            d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
            threshold = np.linalg.norm(d) / 2
            if threshold == prev:
                break
            prev = threshold
            volume = self(P).reshape((len(X), len(Y), len(Z)))
            where = np.argwhere(np.abs(volume) <= threshold)
            x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
            x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2
        self._lower_bound = np.array([x0, y0, z0])
        self._higher_bound = np.array([x1, y1, z1])
        self._is_bound = True

    def radial(self, theta, phi):
        pass