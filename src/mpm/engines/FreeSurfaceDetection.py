import taichi as ti

import src.utils.GlobalVariable as GlobalVariable
from src.utils.constants import PI, ZEROMAT3X3, ZEROVEC3F
from src.utils.MatrixFunction import eigenvalue
from src.utils.ShapeFunctions import g_guassian, guassian
from src.utils.ScalarFunction import linearize3d
from src.utils.VectorFunction import normalize, squared


@ti.kernel
def assign_particle_free_surface(
    start_index: int, end_index: int, particle: ti.template(), material_id: ti.template(), material: ti.template()
):
    density_ratio_tolerance = 0.76
    for i in range(start_index, end_index):
        np = material_id[i]
        if int(particle[np].active) == 1:
            is_free_surface = 0
            if particle[np].mass_density / material.density <= density_ratio_tolerance:
                is_free_surface = 1
            particle[np].free_surface = ti.u8(is_free_surface)
            particle[np].coupling = ti.u8(is_free_surface)


@ti.kernel
def find_boundary_direction_by_geometry(
    igrid_size: float,
    cnum: ti.types.vector(3, int),
    particle_num: int,
    particle: ti.template(),
    particle_id: ti.template(),
    particle_count: ti.template(),
):
    for np in range(particle_num):
        normal = ZEROVEC3F
        if particle[np].free_surface == ti.u8(1) and int(particle[np].active) == 1:
            p_coord = particle[np].x
            grid_idx = ti.floor(p_coord * igrid_size, int)
            smoothing_length = 1.33 * particle[np].rad
            body_id1 = particle[np].bodyID
            renormalization_matrix_inv = ZEROMAT3X3
            temporary_vec = ZEROVEC3F

            x_begin = ti.max(grid_idx[0] - 1, 0)
            x_end = ti.min(grid_idx[0] + 2, cnum[0])
            y_begin = ti.max(grid_idx[1] - 1, 0)
            y_end = ti.min(grid_idx[1] + 2, cnum[1])
            z_begin = ti.max(grid_idx[2] - 1, 0)
            z_end = ti.min(grid_idx[2] + 2, cnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cell_id = linearize3d(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cell_id], particle_count[cell_id + 1]):
                            neighbor_id = particle_id[hash_index]
                            if np == neighbor_id:
                                continue
                            body_id2 = particle[neighbor_id].bodyID
                            if body_id1 == body_id2:
                                rel_coord = particle[neighbor_id].x - p_coord
                                kernel_gradient = g_guassian(smoothing_length, -rel_coord)
                                renormalization_matrix_inv += (
                                    particle[neighbor_id].m
                                    / particle[neighbor_id].mass_density
                                    * kernel_gradient.outer_product(rel_coord)
                                )
                                temporary_vec += (
                                    particle[neighbor_id].m / particle[neighbor_id].mass_density * kernel_gradient
                                )
            normal = normalize(-renormalization_matrix_inv.inverse() @ temporary_vec)
        particle[np].normal = normal


@ti.kernel
def find_free_surface_by_geometry(
    igrid_size: float,
    cnum: ti.types.vector(3, int),
    particle_num: int,
    particle: ti.template(),
    particle_id: ti.template(),
    particle_count: ti.template(),
):
    for np in range(particle_num):
        if particle[np].free_surface == ti.u8(1) and int(particle[np].active) == 1:
            p_coord = particle[np].x
            grid_idx = ti.floor(p_coord * igrid_size, int)
            smoothing_length = 1.33 * particle[np].rad

            x_begin = ti.max(grid_idx[0] - 1, 0)
            x_end = ti.min(grid_idx[0] + 2, cnum[0])
            y_begin = ti.max(grid_idx[1] - 1, 0)
            y_end = ti.min(grid_idx[1] + 2, cnum[1])
            z_begin = ti.max(grid_idx[2] - 1, 0)
            z_end = ti.min(grid_idx[2] + 2, cnum[2])

            # filter internal particle by particle mass_density
            normal = particle[np].normal
            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cell_id = linearize3d(neigh_i, neigh_j, neigh_k, cnum)
                        t_coord = p_coord + smoothing_length * normal
                        for hash_index in range(particle_count[cell_id], particle_count[cell_id + 1]):
                            neighbor_id = particle_id[hash_index]
                            if np == neighbor_id:
                                continue
                            n_coord = particle[neighbor_id].x
                            rel_coord_np = n_coord - p_coord
                            distance_np = rel_coord_np.norm()
                            rel_coord_nt = n_coord - t_coord
                            distance_nt = rel_coord_nt.norm()

                            if distance_np < ti.sqrt(2) * smoothing_length:
                                if ti.acos(normal.dot(rel_coord_np) / distance_np) < 0.25 * PI:
                                    particle[np].free_surface = ti.u8(0)
                                    particle[np].coupling = ti.u8(0)
                                    break
                            else:
                                if distance_nt < smoothing_length:
                                    particle[np].free_surface = ti.u8(0)
                                    particle[np].coupling = ti.u8(0)
                                    break


@ti.kernel
def find_free_surface_by_geometry_eigen(
    icell_size: float,
    radius: float,
    cnum: ti.types.vector(3, int),
    start_point: ti.types.vector(3, float),
    particle_num: int,
    particle: ti.template(),
    particle_id: ti.template(),
    hash_table: ti.template(),
):
    for np in range(particle_num):
        if int(particle[np].free_surface) == 1:
            p_coord = particle[np].x - start_point
            grid_idx = ti.floor(p_coord * icell_size, int)
            smoothing_length = 1.33 * radius
            renormalization_matrix_inv = ZEROMAT3X3
            temporary_vec = ZEROVEC3F

            x_begin = ti.max(grid_idx[0] - 2, 0)
            x_end = ti.min(grid_idx[0] + 3, cnum[0])
            y_begin = ti.max(grid_idx[1] - 2, 0)
            y_end = ti.min(grid_idx[1] + 3, cnum[1])
            z_begin = ti.max(grid_idx[2] - 2, 0)
            z_end = ti.min(grid_idx[2] + 3, cnum[2])

            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cell_id = linearize3d(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(
                            hash_table[cell_id].current, hash_table[cell_id].current + hash_table[cell_id].count
                        ):
                            neighbor_id = particle_id[hash_index]
                            if np == neighbor_id:
                                continue
                            rel_coord = particle[neighbor_id].x - p_coord - start_point
                            kernel_gradient = g_guassian(smoothing_length, -rel_coord)
                            renormalization_matrix_inv += particle[neighbor_id].vol * kernel_gradient.outer_product(
                                rel_coord
                            )
                            temporary_vec += particle[neighbor_id].vol * kernel_gradient

            vector = eigenvalue(renormalization_matrix_inv)
            lambda_ = vector[0]
            normal = normalize(-renormalization_matrix_inv.inverse() @ temporary_vec)
            if lambda_ <= 0.76:
                if lambda_ > 0.2:
                    for neigh_i in range(x_begin, x_end):
                        for neigh_j in range(y_begin, y_end):
                            for neigh_k in range(z_begin, z_end):
                                cell_id = linearize3d(neigh_i, neigh_j, neigh_k, cnum)
                                t_coord = p_coord + smoothing_length * normal
                                for hash_index in range(
                                    hash_table[cell_id].current, hash_table[cell_id].current + hash_table[cell_id].count
                                ):
                                    neighbor_id = particle_id[hash_index]
                                    if np == neighbor_id:
                                        continue
                                    n_coord = particle[neighbor_id].x - start_point
                                    rel_coord_np = n_coord - p_coord
                                    distance_np = rel_coord_np.norm()
                                    rel_coord_nt = n_coord - t_coord
                                    distance_nt = rel_coord_nt.norm()

                                    if distance_np < ti.sqrt(2) * smoothing_length:
                                        if ti.acos(normal.dot(rel_coord_np) / distance_np) < 0.25 * PI:
                                            particle[np].free_surface = ti.u8(0)
                                            particle[np].coupling = ti.u8(0)
                                            break
                                    else:
                                        if distance_nt < smoothing_length:
                                            particle[np].free_surface = ti.u8(0)
                                            particle[np].coupling = ti.u8(0)
                                            break
            else:
                particle[np].free_surface = ti.u8(0)
                particle[np].coupling = ti.u8(0)
            particle[np].normal = normal
            particle[np].lambda_ = lambda_


# ========================================================= #
#                     Surface Tension                       #
# ========================================================= #
@ti.kernel
def kernel_calculate_surface_tension(
    kappa: float,
    igrid_size: float,
    cnum: ti.types.vector(3, int),
    particle_num: int,
    particle: ti.template(),
    particle_id: ti.template(),
    particle_current: ti.template(),
    particle_count: ti.template(),
):
    # reference: J.P. Morris. Simulating surface tension with smoothed particle hydrodynamics. International Journal for Numerical Methods in Fluids. 2000 (33) 333-353.
    for np in range(particle_num):
        if particle[np].free_surface == ti.u8(1) and int(particle[np].active) == 1:
            p_coord = particle[np].x
            grid_idx = ti.floor(p_coord * igrid_size, int)
            smoothing_length = 1.33 * particle[np].rad
            epsilon = 0.01 / smoothing_length

            x_begin = ti.max(grid_idx[0] - 1, 0)
            x_end = ti.min(grid_idx[0] + 2, cnum[0])
            y_begin = ti.max(grid_idx[1] - 1, 0)
            y_end = ti.min(grid_idx[1] + 2, cnum[1])
            z_begin = ti.max(grid_idx[2] - 1, 0)
            z_end = ti.min(grid_idx[2] + 2, cnum[2])

            normala = particle[np].normal
            Na = 1.0 if squared(normala) > epsilon * epsilon else 0.0
            curvature_star, eta = 0.0, 0.0
            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cell_id = linearize3d(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(
                            particle_count[cell_id] - particle_current[cell_id], particle_count[cell_id]
                        ):
                            neighbor_id = particle_id[hash_index]
                            if np == neighbor_id:
                                continue
                            rel_coord = particle[neighbor_id].x - p_coord
                            normalb = particle[neighbor_id].normal
                            Nb = 1.0 if squared(normalb) > epsilon * epsilon else 0.0
                            curvature_star += (
                                min(Na, Nb)
                                * particle[neighbor_id].vol
                                * (normalb - normala)
                                * g_guassian(smoothing_length, -rel_coord)
                            )
                            eta += min(Na, Nb) * particle[neighbor_id].vol * guassian(smoothing_length, -rel_coord)
            curvature = curvature_star / eta
            particle[np].external_force -= curvature * normala * kappa


# ---- SemiImplicit TwoPhaseSingleLayer compatibility kernels ----
@ti.kernel
def assign_particle_free_surface_poisson(particle_num: int, particle: ti.template(), material: ti.template()):
    density_ratio_tolerance = 0.76 * 0.6
    for np in range(particle_num):
        if int(particle[np].active) == 1:
            is_free_surface = 0
            material_id = int(particle[np].materialID)
            if particle[np].mass_density / material[material_id].solid_density <= density_ratio_tolerance:
                is_free_surface = 1
            particle[np].free_surface = ti.u8(is_free_surface)


@ti.kernel
def assign_particle_free_surface_by_volume_multilayer(
    particle_num: int,
    particle: ti.template(),
    cell_volumefrac: ti.template(),
    grid_size: ti.template(),
    cnum: ti.template(),
):
    volume_fraction_tolerance = 0.3
    for np in range(particle_num):
        if int(particle[np].active) == 1:
            is_free_surface = 0
            ic = ti.floor((particle[np].x / grid_size), int)
            if ti.static(GlobalVariable.DIMENSION == 2):
                for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                    neigh = ic + ti.Vector([dx, dy])
                    if 0 <= neigh[0] < cnum[0] and 0 <= neigh[1] < cnum[1]:
                        icell = neigh[0] + neigh[1] * cnum[0]
                        if cell_volumefrac[icell] <= volume_fraction_tolerance:
                            is_free_surface = 1
            else:
                for dx, dy, dz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    neigh = ic + ti.Vector([dx, dy, dz])
                    if 0 <= neigh[0] < cnum[0] and 0 <= neigh[1] < cnum[1] and 0 <= neigh[2] < cnum[2]:
                        icell = neigh[0] + cnum[0] * (neigh[1] + neigh[2] * cnum[1])
                        if cell_volumefrac[icell] <= volume_fraction_tolerance:
                            is_free_surface = 1
            particle[np].free_surface = ti.u8(is_free_surface)


@ti.kernel
def record_free_surface_particle_2d(
    domain: ti.types.vector(2, float),
    grid_size: ti.types.vector(2, float),
    particle_num: int,
    particle: ti.template(),
    free_paticle_num: ti.types.ndarray(),
    free_particle_list: ti.template(),
):
    free_paticle_num[0] = ti.u8(0)
    free_particle_list.fill(-1)
    for np in range(particle_num):
        if particle[np].free_surface == ti.u8(1):
            p_coord = particle[np].x
            if (
                p_coord[0] < grid_size[0] * 1.5
                or p_coord[0] > domain[0] - grid_size[0] * 1.5
                or p_coord[1] < grid_size[1] * 1.5
                or p_coord[1] > domain[1] - grid_size[1] * 1.5
            ):
                particle[np].free_surface = ti.u8(0)
            else:
                index = ti.atomic_add(free_paticle_num[0], 1)
                free_particle_list[index] = np
