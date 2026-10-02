## @package ParticleToSdf
# Converting particles to sdf with method:
# "Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels" by JinHun Yu and Greg Turk


import taichi as ti
import numpy as np
from src.vdb_grid import *


@ti.data_oriented
class ParticleToSdf:

    # Implementation of anisotropic sampling of particles

    def __init__(self, sdf_voxel_dim, particle_voxel_dim, level_configs, max_num_particles):
        assert level_configs is not None, "Level configuration must not be None."
        self.sdf = VdbGrid(sdf_voxel_dim, level_configs)
        self.max_num_particles = max_num_particles
        self.aux = VdbGrid(sdf_voxel_dim, level_configs, ti.f32)
        self.particle_aux = VdbGrid(particle_voxel_dim, level_configs, ti.i32)

        # Sampling Mass and density
        self.mass = ti.field(dtype=ti.f32, shape=max_num_particles)
        self.density = ti.field(dtype=ti.f32, shape=max_num_particles)

        # Anisotropic kernel
        self.G = ti.Matrix.field(3, 3, ti.f32, shape=max_num_particles)
        self.x_bar = ti.Vector.field(3, ti.f32, shape=max_num_particles)

        self.kr = 4
        self.ks = 1400
        self.kn = 0.5
        self.Neps = 25
        self.akLambda = 0.9

    def clear(self):
        self.sdf.clear()
        self.aux.clear()
        self.particle_aux.clear()

    @ti.func
    def isotropic_weight(self, x_i: ti.template(), x_j: ti.template(), r_i: ti.template()):
        norm_dx = (x_i - x_j).norm()
        res = 0.0
        if norm_dx < r_i:
            norm_dx /= r_i
            res = 1.0 - norm_dx * norm_dx * norm_dx

        return res



    @ti.kernel
    def fill_particle_id_grid(self, particle_pos: ti.template(), num_particles: ti.template()):
        ti.loop_config(block_dim=512)

        for id in range(num_particles):
            self.G[id] = ti.Matrix.identity(dt=ti.f32, n=3)
            self.particle_aux.set_value_packed(particle_pos[id], id + 1)

    @ti.kernel
    def compute_anisotropic_kernel_with_grid(self, particle_pos: ti.template(), smoothing_radius: ti.f32):
        smoothing_voxel_radius = ti.ceil(smoothing_radius * self.particle_aux.data_wrapper.inv_voxel_dim, ti.i32) - 1
        print(f"Smoothing radius {smoothing_voxel_radius}")
        for i, j, k in self.particle_aux.data_wrapper.leaf_value:
            id = int(self.particle_aux.read_value_world(i, j, k))

            if id > 0:
                x_i = particle_pos[id - 1]

                sum_weight = 0.0
                x_wi = ti.Vector.zero(n=3, dt=ti.f32)
                for dx, dy, dz in ti.ndrange(
                        (-2 * smoothing_voxel_radius[0], 2 * smoothing_voxel_radius[0] + 1),
                        (-2 * smoothing_voxel_radius[1], 2 * smoothing_voxel_radius[1] + 1),
                        (-2 * smoothing_voxel_radius[2], 2 * smoothing_voxel_radius[2] + 1)
                ):
                    nx = i + dx
                    ny = j + dy
                    nz = k + dz

                    new_id = int(self.particle_aux.read_value_world(nx, ny, nz))

                    if new_id > 0:
                        x_j = particle_pos[new_id - 1]
                        w_ij = self.isotropic_weight(x_i, x_j, 2 * smoothing_radius)
                        sum_weight += w_ij
                        x_wi += w_ij * x_j

                if sum_weight == 0.0:
                    self.G[id - 1] = ti.Matrix.zero(ti.f32, 3, 3)
                else:
                    # Only consider non-isolated particles
                    x_wi /= sum_weight
                    self.x_bar[id - 1] = (1 - self.akLambda) * x_i + self.akLambda * x_wi

                    C_i = ti.Matrix.zero(ti.f32, 3, 3)
                    num_neighbor_particles = 0
                    for dx, dy, dz in ti.ndrange(
                            (-2 * smoothing_voxel_radius[0], 2 * smoothing_voxel_radius[0] + 1),
                            (-2 * smoothing_voxel_radius[1], 2 * smoothing_voxel_radius[1] + 1),
                            (-2 * smoothing_voxel_radius[2], 2 * smoothing_voxel_radius[2] + 1)
                    ):
                        nx = i + dx
                        ny = j + dy
                        nz = k + dz

                        new_id = int(self.particle_aux.read_value_world(nx, ny, nz))

                        if new_id > 0:
                            x_j = particle_pos[new_id - 1]
                            w_ij = self.isotropic_weight(x_i, x_j, 2 * smoothing_radius)
                            if w_ij > 0.0:
                                num_neighbor_particles += 1
                                dp = x_j - x_wi
                                C_i += w_ij * dp.outer_product(dp)

                    C_i /= sum_weight

                    R, Sigma, RT = ti.svd(C_i)

                    # Then, we see if we need to modify C_i
                    # condition is  sigma_1 >= kr * sigma_d
                    if num_neighbor_particles > self.Neps:
                        Sigma[1, 1] = ti.max(Sigma[1, 1], Sigma[0, 0] / self.kr)
                        Sigma[2, 2] = ti.max(Sigma[2, 2], Sigma[0, 0] / self.kr)
                        Sigma *= self.ks
                    else:
                        Sigma[0, 0] = self.kn
                        Sigma[1, 1] = self.kn
                        Sigma[2, 2] = self.kn

                    self.G[id - 1] = (1 / smoothing_radius) * R @ Sigma.inverse() @ RT.transpose()


    @ti.kernel
    def sampling_kernel(self, particle_pos: ti.template(), smoothing_radius: ti.f32):
        smoothing_voxel_radius = ti.ceil(smoothing_radius * self.particle_aux.data_wrapper.inv_voxel_dim, ti.i32) - 1
        for i, j, k in self.particle_aux.data_wrapper.leaf_value:
            id = int(self.particle_aux.read_value_world(i, j, k))
            if id > 0:
                pos = particle_pos[id - 1]

                for di, dj, dk in ti.ndrange(
                        (-2 * smoothing_voxel_radius[0], 2 * smoothing_voxel_radius[0] + 1),
                        (-2 * smoothing_voxel_radius[1], 2 * smoothing_voxel_radius[1] + 1),
                        (-2 * smoothing_voxel_radius[2], 2 * smoothing_voxel_radius[2] + 1)
                ):
                    other_id = int(self.particle_aux.read_value_world(i + di, j + dj, k + dk))
                    if other_id > 0:
                        other_pos = self.x_bar[other_id - 1]
                        dist = (self.G[other_id - 1] @ (other_pos - pos)).norm()
                        weight = cubic_spline_kernel_3d(1, dist) * self.G[other_id - 1].determinant()
                        self.density[id - 1] += weight
                # print(f"Sampled Density: {self.density[id - 1]};")

    @ti.kernel
    def mark_surface_vertex(self):
        for i, j, k in self.particle_aux.data_wrapper.leaf_value:
            vertex_pos = self.particle_aux.transform.voxel_to_coord(i, j, k)
            sdf_i, sdf_j, sdf_k = self.sdf.transform.coord_to_voxel_packed(vertex_pos)
            for dx, dy, dz in ti.static(ti.ndrange((-1, 3), (-1, 3), (-1, 3))):
                self.sdf.add_value_world(sdf_i + dx, sdf_j + dy, sdf_k + dz, 1)

    @ti.kernel
    def compute_sdf_fixed_volume(self, smoothing_radius: ti.f32, volume: ti.f32):
        smoothing_voxel_radius = ti.ceil(smoothing_radius * self.particle_aux.data_wrapper.inv_voxel_dim, ti.i32) - 1
        for i, j, k in self.sdf.data_wrapper.leaf_value:
            sdf_value = 0.0

            vertex_pos = self.sdf.transform.voxel_to_coord(i, j, k)
            particle_i, particle_j, particle_k = self.particle_aux.transform.coord_to_voxel_packed(vertex_pos)
            for dx, dy, dz in ti.ndrange(
                    (-2 * smoothing_voxel_radius[0], 2 * smoothing_voxel_radius[0] + 1),
                    (-2 * smoothing_voxel_radius[1], 2 * smoothing_voxel_radius[1] + 1),
                    (-2 * smoothing_voxel_radius[2], 2 * smoothing_voxel_radius[2] + 1)
            ):
                nx = particle_i + dx
                ny = particle_j + dy
                nz = particle_k + dz
                id = int(self.particle_aux.read_value_world(nx, ny, nz))

                if id > 0:
                    Gr = self.G[id - 1] @ (vertex_pos - self.x_bar[id - 1])
                    sdf_value -= volume * self.G[id - 1].determinant() * cubic_spline_kernel_3d(1, Gr.norm()) #/ self.density[id - 1]

            self.sdf.set_value_world(i, j, k, sdf_value)


    ## @brief Implementation of anisotropic kernel sampling of particles
    def particle_to_sdf_anisotropic(self, particle_pos: ti.template(), num_particles: ti.template(),
                                    particle_radius: ti.f32, smoothing_radius=0.03):
        particle_volume = ti.static(4 * np.pi / 3) * particle_radius * particle_radius * particle_radius
        # Step 1: Fill particle grid
        self.fill_particle_id_grid(particle_pos, num_particles)
        # Step 2: Compute anisotropic kernel
        self.compute_anisotropic_kernel_with_grid(particle_pos, smoothing_radius)
        # Step 3: Mark Surface Vertices
        self.mark_surface_vertex()

        # Step 4: Compute sdf with fixed volume
        self.sampling_kernel(particle_pos, smoothing_radius)
        self.compute_sdf_fixed_volume(smoothing_radius, particle_volume)






    @ti.kernel
    def rasterize_particles(self, particle_pos: ti.template(), num_particles: ti.template(),
                            particle_radius: ti.template()):
        particle_scale = 1
        particle_radius_voxel = 1 + ti.ceil(particle_radius * particle_scale / self.sdf.transform.voxel_dim, ti.i32)

        for id in range(num_particles):
            pos = particle_pos[id]
            pos_voxel_coord = self.sdf.transform.coord_to_voxel_packed(pos)
            for i, j, k in ti.ndrange(
                    (-particle_radius_voxel[0], particle_radius_voxel[0] + 1),
                    (-particle_radius_voxel[1], particle_radius_voxel[1] + 1),
                    (-particle_radius_voxel[2], particle_radius_voxel[2] + 1)
            ):
                adjacent_voxel_coord = pos_voxel_coord + ti.Vector([i, j, k])
                center = self.sdf.transform.voxel_to_coord_packed(adjacent_voxel_coord + ti.Vector([0.5, 0.5, 0.5]))
                distance = (center - pos).norm()
                if distance - particle_radius * particle_scale < 0 or self.sdf.read_value_world(adjacent_voxel_coord[0], adjacent_voxel_coord[1], adjacent_voxel_coord[2]) >= 0.0:
                    self.aux.max_value_world(adjacent_voxel_coord[0], adjacent_voxel_coord[1], adjacent_voxel_coord[2],
                                             distance - particle_radius * particle_scale)



    @ti.kernel
    def gaussian_kernel(self, kernel_width: ti.template()):

        for i, j, k in self.sdf.data_wrapper.leaf_value:
            filtered_value = 0.0
            sample_count = 0
            for di, dj, dk in ti.static(ti.ndrange(
                    (-kernel_width, kernel_width + 1),
                    (-kernel_width, kernel_width + 1),
                    (-kernel_width, kernel_width + 1)
            )):
                ni = i + di
                nj = j + dj
                nk = k + dk
                filtered_value += self.sdf.read_value_world(ni, nj, nk) * ti.static(self.gaussian_filter(di, dj, dk))
            self.aux.set_value_world(i, j, k, filtered_value)

    @ti.func
    def can_dilate(self, i, j, k):
        return self.sdf.read_value_world(i, j, k) >= 0.0

    @ti.kernel
    def dilate_kernel(self):

        for i, j, k in self.sdf.data_wrapper.leaf_value:
            value = self.sdf.read_value_world(i, j, k)
            if value < 0.0:
                if self.can_dilate(i + 1, j, k):
                    self.aux.max_value_world(i + 1, j, k, 0.0)
                if self.can_dilate(i - 1, j, k):
                    self.aux.max_value_world(i - 1, j, k, 0.0)
                if self.can_dilate(i, j + 1, k):
                    self.aux.max_value_world(i, j + 1, k, 0.0)
                if self.can_dilate(i, j - 1, k):
                    self.aux.max_value_world(i, j - 1, k, 0.0)
                if self.can_dilate(i, j, k + 1):
                    self.aux.max_value_world(i, j, k + 1, 0.0)
                if self.can_dilate(i, j, k - 1):
                    self.aux.max_value_world(i, j, k - 1, 0.0)

    @ti.func
    def can_erode(self, i, j, k):
        return self.sdf.read_value_world(i, j, k) >= 0.0
    @ti.kernel
    def erode_kernel(self):

        for i, j, k in self.sdf.data_wrapper.leaf_value:
            value = self.sdf.read_value_world(i, j, k)
            if value < 0.0:
                is_erode = self.can_erode(i + 1, j, k)
                is_erode |= self.can_erode(i - 1, j, k)
                is_erode |= self.can_erode(i, j + 1, k)
                is_erode |= self.can_erode(i, j - 1, k)
                is_erode |= self.can_erode(i, j, k + 1)
                is_erode |= self.can_erode(i, j, k - 1)

                if not is_erode:
                    self.aux.set_value_world(i, j, k, value)
                else:
                    self.aux.set_value_world(i, j, k, 0.0)
