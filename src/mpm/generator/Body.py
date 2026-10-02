import numpy as np
from scipy.stats import qmc


def sample_solid_hemisphere_halton(
    center,
    radius,
    n_particles,
    *,
    side="lower",
    center_is_mass_center=True,
    seed=None,
):
    """Return volume-uniform Halton points in a solid upper or lower hemisphere."""
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("hemisphere center must contain three coordinates")
    if radius <= 0.0 or int(n_particles) < 1:
        raise ValueError("hemisphere radius and particle count must be positive")
    side_sign = {"lower": -1.0, "upper": 1.0}.get(str(side).strip().lower())
    if side_sign is None:
        raise ValueError("hemisphere side must be 'lower' or 'upper'")

    sampler = qmc.Halton(d=3, scramble=True, seed=seed)
    uvw = sampler.random(int(n_particles))
    cosine = side_sign * uvw[:, 0]
    sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
    theta = 2.0 * np.pi * uvw[:, 1]
    radial = float(radius) * np.cbrt(uvw[:, 2])
    points = np.column_stack(
        (
            radial * sine * np.cos(theta),
            radial * sine * np.sin(theta),
            radial * cosine,
        )
    )
    if center_is_mass_center:
        points[:, 2] -= side_sign * 3.0 * float(radius) / 8.0
    return np.ascontiguousarray(points + center[None, :], dtype=np.float64)


class Body:
    def __init__(self):
        self.bodies = {}
        self.body_counter = 0
        self.particle_counter = 0
        self.boundary_extractor = None

    def _get_name(self, name):
        if name is None:
            name = f"body_{self.body_counter}"
            self.body_counter += 1
        return name

    def _pack_meta(self, points, init_v, volume, name, grid_size=None, xmin=None, xmax=None):
        dim = points.shape[1]  # 自动判断是2D还是3D
        if init_v is None:
            init_v = np.zeros(dim, dtype=np.float64)
        # 默认非法值
        if grid_size is None:
            grid_size = -1.0
        if xmin is None:
            xmin = np.full(dim, np.nan)
        if xmax is None:
            xmax = np.full(dim, np.nan)

        # 维度检查
        if len(xmin) != dim or len(xmax) != dim or len(init_v) != dim:
            raise ValueError(
                f"{name}: dimension mismatch! points are {dim}D, " f"but xmin={xmin}, xmax={xmax}, init_v={init_v}"
            )

        self.bodies[name] = {
            "poffset": self.particle_counter,
            "points": points,
            "init_v": np.array(init_v),
            "volume": volume,
            "grid_size": grid_size,
            "xmin": np.array(xmin),
            "xmax": np.array(xmax),
        }
        self.particle_counter += points.shape[0]

    def add_particles(
        self,
        points,
        volume=1.0,
        init_v=None,
        name=None,
        grid_size=None,
        xmin=None,
        xmax=None,
        boundary_ids=None,
        surface_measure=None,
    ):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("points should be a 2D array with shape (particle_count, dimension).")
        if points.shape[0] == 0:
            raise ValueError("points should contain at least one particle.")

        if init_v is None:
            init_v = np.zeros(points.shape[1], dtype=np.float64)

        name = self._get_name(name)
        self._pack_meta(points, init_v, volume, name, grid_size, xmin, xmax)
        if boundary_ids is not None:
            self.bodies[name].update({"boundary_ids": np.asarray(boundary_ids, dtype=np.int32)})
        if surface_measure is not None:
            # Validation and the per-particle/per-boundary selection are kept
            # in MPMSolver.build_surface_node(), where the final boundary IDs
            # are known.  Preserve the original array here without silently
            # collapsing a one-element vector to a scalar.
            self.bodies[name]["surface_measure"] = np.asarray(surface_measure, dtype=np.float64)

    def get_surface_ids(self, name, all_particles=False):
        body = self.bodies[name]
        pts = body["points"]
        poffset = body["poffset"]
        if all_particles:
            return np.arange(pts.shape[0], dtype=np.int32) + poffset
        surface_ids = body.get("boundary_id", body.get("boundary_ids", None))
        if surface_ids is None:
            if self.boundary_extractor is None:
                from src.mpm.generator.BoundaryExtractor import BoundaryExtractor

                self.boundary_extractor = BoundaryExtractor()
            self.boundary_extractor.load_points(pts)
            surface_ids = self.boundary_extractor.method_geometric_boundary()
        # surface_ids = np.arange(0, pts.shape[0], 1)
        return surface_ids + poffset

    def add_cube(self, start, end, spacing, ppc=1, init_v=None, name=None, grid_size=None, xmin=None, xmax=None):
        if not isinstance(ppc, (int, np.integer)) or ppc <= 0:
            raise ValueError("ppc must be a positive integer")
        ratios = (np.asarray(end) - np.asarray(start)) / spacing
        counts = np.floor(ratios + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(ratios))).astype(int)
        nx, ny, nz = counts

        gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        voxel_base = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) * spacing + start

        if ppc == 1:
            local_pos = np.array([[0.5, 0.5, 0.5]])
            local_indices = np.zeros((1, 3), dtype=int)
        else:
            grid_1d = (np.arange(ppc) + 0.5) / ppc
            lx, ly, lz = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
            local_pos = np.stack([lx, ly, lz], axis=-1).reshape(-1, 3)
            ix, iy, iz = np.meshgrid(np.arange(ppc), np.arange(ppc), np.arange(ppc), indexing="ij")
            local_indices = np.stack([ix, iy, iz], axis=-1).reshape(-1, 3)

        particles = (voxel_base[:, None, :] + local_pos[None, :, :] * spacing).reshape(-1, 3)

        voxel_vol = spacing**3
        mp_vol = voxel_vol / len(local_pos)

        name = self._get_name(name)
        self._pack_meta(particles, init_v, mp_vol, name, grid_size, xmin, xmax)

        voxel_indices = np.repeat(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1), len(local_pos), axis=0)
        subparticle_indices = np.tile(local_indices, (voxel_base.shape[0], 1))
        maxima = np.array([nx - 1, ny - 1, nz - 1])
        boundary_mask = np.any(
            ((voxel_indices == 0) & (subparticle_indices == 0))
            | ((voxel_indices == maxima) & (subparticle_indices == ppc - 1)),
            axis=1,
        )
        boundary_ids = np.flatnonzero(boundary_mask)
        self.bodies[name].update({"boundary_ids": boundary_ids})

    def add_sphere(
        self,
        center,
        radius,
        n_particles=None,
        spacing=None,
        init_v=None,
        name=None,
        grid_size=None,
        xmin=None,
        xmax=None,
        seed=None,
    ):
        if n_particles is not None:
            sampler = qmc.Halton(d=3, scramble=True, seed=seed)
            uvw = sampler.random(n_particles)
            phi = np.arccos(1 - 2 * uvw[:, 0])
            theta = 2 * np.pi * uvw[:, 1]
            r = radius * (uvw[:, 2] ** (1 / 3))
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points = np.stack([x, y, z], axis=1) + center
            vol = 4 / 3 * np.pi * radius**3 / n_particles
        elif spacing is not None:
            N_r = spacing[0]
            points_list = []
            for i in range(N_r):
                r_layer = radius * (i + 0.5) / N_r
                N_theta = max(1, int(2 * np.pi * r_layer / (radius / N_r)))
                N_phi = max(1, int(np.pi * r_layer / (radius / N_r)))
                for j in range(N_theta):
                    theta = 2 * np.pi * j / N_theta
                    for k in range(N_phi):
                        phi = np.pi * (k + 0.5) / N_phi
                        x = r_layer * np.sin(phi) * np.cos(theta)
                        y = r_layer * np.sin(phi) * np.sin(theta)
                        z = r_layer * np.cos(phi)
                        points_list.append([x, y, z])
            points = np.array(points_list) + center
            vol = 4 / 3 * np.pi * radius**3 / len(points)
        else:
            raise ValueError("Either n_particles or spacing must be provided.")
        name = self._get_name(name)
        self._pack_meta(points, init_v, vol, name, grid_size, xmin, xmax)
        distances = np.linalg.norm(points - center, axis=1)
        boundary_ids = np.where(distances >= radius * 0.99)[0]
        self.bodies[name].update({"boundary_ids": boundary_ids})

    def add_hemisphere(
        self,
        center,
        radius,
        n_particles,
        side="lower",
        center_is_mass_center=True,
        init_v=None,
        name=None,
        grid_size=None,
        xmin=None,
        xmax=None,
        seed=None,
    ):
        points = sample_solid_hemisphere_halton(
            center,
            radius,
            n_particles,
            side=side,
            center_is_mass_center=center_is_mass_center,
            seed=seed,
        )
        volume = (2.0 / 3.0) * np.pi * radius**3 / int(n_particles)
        name = self._get_name(name)
        self._pack_meta(points, init_v, volume, name, grid_size, xmin, xmax)
        geometric_center_z = (
            float(center[2]) + (-3.0 / 8.0 if side == "lower" else 3.0 / 8.0) * radius
            if center_is_mass_center
            else float(center[2])
        )
        relative = points - np.asarray([center[0], center[1], geometric_center_z], dtype=np.float64)
        curved = np.linalg.norm(relative, axis=1) >= 0.99 * radius
        flat = np.abs(points[:, 2] - geometric_center_z) <= 0.01 * radius
        self.bodies[name].update({"boundary_ids": np.flatnonzero(curved | flat).astype(np.int32)})

    def add_cylinder(
        self,
        center,
        radius,
        height,
        n_particles=None,
        spacing=None,
        init_v=None,
        name=None,
        grid_size=None,
        xmin=None,
        xmax=None,
    ):
        if n_particles is not None:
            sampler = qmc.Halton(d=3, scramble=True)
            uvw = sampler.random(n_particles)
            theta = 2 * np.pi * uvw[:, 0]
            r = radius * np.sqrt(uvw[:, 1])
            z = height * (uvw[:, 2] - 0.5)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.stack([x, y, z], axis=1) + center
            vol = np.pi * radius**2 * height / n_particles
        elif spacing is not None:
            N_r, N_theta, N_z = spacing
            points_list = []
            for i in range(N_r):
                r_layer = radius * (i + 0.5) / N_r
                N_theta_layer = max(1, int(2 * np.pi * r_layer / (radius / N_r)))
                for j in range(N_theta_layer):
                    theta = 2 * np.pi * j / N_theta_layer
                    for k in range(N_z):
                        z = height * (k + 0.5) / N_z - height / 2
                        x = r_layer * np.cos(theta)
                        y = r_layer * np.sin(theta)
                        points_list.append([x, y, z])
            points = np.array(points_list) + center
            vol = np.pi * radius**2 * height / len(points)
        else:
            raise ValueError("Either n_particles or spacing must be provided.")
        name = self._get_name(name)
        self._pack_meta(points, init_v, vol, name, grid_size, xmin, xmax)

        r = np.linalg.norm(points[:, :2], axis=1)
        boundary_ids = np.where(
            (r >= radius * 0.99) | (points[:, 2] >= height / 2 * 0.99) | (points[:, 2] <= -height / 2 * 0.99)
        )[0]
        self.bodies[name].update({"boundary_ids": boundary_ids})

    def add_from_file(self, filename, vol=1.0, init_v=None, name=None, grid_size=None, xmin=None, xmax=None):
        data = np.loadtxt(filename, delimiter=",")
        name = self._get_name(name)
        self._pack_meta(data, init_v, vol, name, grid_size, xmin, xmax)

    def add_rectangle(self, start, end, spacing, ppc=1, init_v=None, name=None, grid_size=None, xmin=None, xmax=None):
        ratios = (np.asarray(end) - np.asarray(start)) / spacing
        counts = np.floor(ratios + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(ratios))).astype(int)
        nx, ny = counts

        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        cell_base = np.stack([gx, gy], axis=-1).reshape(-1, 2) * spacing + start

        if ppc == 1:
            local_pos = np.array([[0.5, 0.5]])
        else:
            grid_1d = (np.arange(ppc) + 0.5) / ppc
            lx, ly = np.meshgrid(grid_1d, grid_1d, indexing="ij")
            local_pos = np.stack([lx, ly], axis=-1).reshape(-1, 2)

        particles = (cell_base[:, None, :] + local_pos[None, :, :] * spacing).reshape(-1, 2)

        cell_area = spacing**2
        mp_area = cell_area / len(local_pos)

        name = self._get_name(name)
        self._pack_meta(particles, init_v, mp_area, name, grid_size, xmin, xmax)

        cx, cy = gx.flatten(), gy.flatten()
        boundary_mask = (cx == 0) | (cx == nx - 1) | (cy == 0) | (cy == ny - 1)
        boundary_ids = []
        for i, mask in enumerate(boundary_mask):
            if mask:
                start_idx = i * len(local_pos)
                end_idx = (i + 1) * len(local_pos)
                boundary_ids.extend(range(start_idx, end_idx))
        boundary_ids = np.array(boundary_ids)
        self.bodies[name].update({"boundary_ids": boundary_ids})

    def add_circle(
        self,
        center,
        radius,
        n_particles=None,
        spacing=None,
        init_v=None,
        name=None,
        grid_size=None,
        xmin=None,
        xmax=None,
    ):
        if n_particles is not None:
            sampler = qmc.Halton(d=2, scramble=True)
            uv = sampler.random(n_particles)
            theta = 2 * np.pi * uv[:, 0]
            r = radius * np.sqrt(uv[:, 1])
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.stack([x, y], axis=1) + center
        elif spacing is not None:
            N_r, N_theta = spacing
            points_list = []
            dr = radius / N_r
            for i in range(N_r):
                r_layer = dr * (i + 0.5)
                N_theta = max(1, int(2 * np.pi * r_layer / dr))
                for j in range(N_theta):
                    theta = 2 * np.pi * j / N_theta
                    x = r_layer * np.cos(theta)
                    y = r_layer * np.sin(theta)
                    points_list.append([x, y])
            points = np.array(points_list) + center
            n_particles = len(points)
        else:
            raise ValueError("Either n_particles or spacing must be provided.")
        area = np.pi * radius**2 / n_particles
        name = self._get_name(name)
        self._pack_meta(points, init_v, area, name, grid_size, xmin, xmax)

        distances = np.linalg.norm(points - center, axis=1)
        boundary_ids = np.where(distances >= radius * 0.99)[0]
        self.bodies[name].update({"boundary_ids": boundary_ids})

    def add_semi_circle(
        self,
        center,
        radius,
        n_particles=None,
        spacing=None,
        init_v=None,
        name=None,
        grid_size=None,
        xmin=None,
        xmax=None,
    ):
        if n_particles is not None:
            sampler = qmc.Halton(d=2, scramble=True)
            uv = sampler.random(n_particles * 2)
            theta = 2 * np.pi * uv[:, 0]
            r = radius * np.sqrt(uv[:, 1])
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.stack([x, y], axis=1) + center
            points = points[points[:, 1] <= center[1]]
            n_particles = len(points)
        elif spacing is not None:
            N_r, N_theta = spacing
            points_list = []
            dr = radius / N_r
            for i in range(N_r):
                r_layer = dr * (i + 0.5)
                N_theta_layer = max(1, int(np.pi * r_layer / dr))
                for j in range(N_theta_layer):
                    theta = np.pi + np.pi * j / N_theta_layer
                    x = r_layer * np.cos(theta)
                    y = r_layer * np.sin(theta)
                    points_list.append([x, y])
            points = np.array(points_list) + center
            n_particles = len(points)
        else:
            raise ValueError("Either n_particles or spacing must be provided.")

        area = (np.pi * radius**2 / 2) / n_particles  # 半圆面积
        name = self._get_name(name)
        self._pack_meta(points, init_v, area, name, grid_size, xmin, xmax)

        distances = np.linalg.norm(points - center, axis=1)
        boundary_ids = np.where(distances >= radius * 0.99)[0]
        self.bodies[name].update({"boundary_ids": boundary_ids})

    def add_ring(
        self,
        center,
        r_in,
        r_out,
        n_particles=None,
        spacing=None,
        init_v=None,
        name=None,
        grid_size=None,
        xmin=None,
        xmax=None,
    ):
        structured_boundary_ids = None
        if n_particles is not None:
            sampler = qmc.Halton(d=2, scramble=True)
            uv = sampler.random(n_particles)
            theta = 2 * np.pi * uv[:, 0]
            r = np.sqrt((r_out**2 - r_in**2) * uv[:, 1] + r_in**2)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.stack([x, y], axis=1) + center
            area = np.pi * (r_out**2 - r_in**2) / n_particles
        elif spacing is not None:
            points = []
            N_r, N_theta = map(int, spacing)
            if N_r <= 0 or N_theta <= 0:
                raise ValueError("ring spacing counts must be positive")
            for i in range(N_r):
                r_layer = np.sqrt(r_in**2 + (r_out**2 - r_in**2) * (i + 0.5) / N_r)
                for j in range(N_theta):
                    theta = 2 * np.pi * j / N_theta
                    x = r_layer * np.cos(theta)
                    y = r_layer * np.sin(theta)
                    points.append([x, y])
            points = np.array(points) + center
            structured_boundary_ids = np.arange((N_r - 1) * N_theta, N_r * N_theta, dtype=np.int32)
            area = np.pi * (r_out**2 - r_in**2) / len(points)
        else:
            raise ValueError("Either n_particles or spacing must be provided.")
        name = self._get_name(name)
        self._pack_meta(points, init_v, area, name, grid_size, xmin, xmax)

        distances = np.linalg.norm(points - center, axis=1)
        boundary_ids = (
            np.where(distances >= r_out * 0.99)[0] if structured_boundary_ids is None else structured_boundary_ids
        )
        self.bodies[name].update({"boundary_ids": boundary_ids})
