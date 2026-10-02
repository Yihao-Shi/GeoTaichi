import taichi as ti

from src.utils.constants import PI, ZEROMAT3x3, ZEROVEC3f
from src.utils.MatrixFunction import get_eigenvalue
from src.utils.ShapeFunctions import GGuassian, Guassian
from src.utils.ScalarFunction import linearize3D
from src.utils.VectorFunction import Normalize, Squared


@ti.kernel
def assign_particle_free_surface(start_index: int, end_index: int, particle: ti.template(), materialID: ti.template(), material: ti.template()):
    density_ratio_tolerance = 0.76
    for i in range(start_index, end_index):
        np = materialID[i]
        if int(particle[np].active) == 1:
            is_free_surface = 0
            if particle[np].mass_density / material.density <= density_ratio_tolerance:
                is_free_surface = 1
            particle[np].free_surface = ti.u8(is_free_surface)
            particle[np].coupling = ti.u8(is_free_surface)


@ti.kernel
def find_boundary_direction_by_geometry(igrid_size: float, cnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), particleID: ti.template(), particle_count: ti.template()):
    for np in range(particleNum):
        normal = ZEROVEC3f
        if particle[np].free_surface == ti.u8(1) and int(particle[np].active) == 1:
            p_coord = particle[np].x
            grid_idx = ti.floor(p_coord * igrid_size, int)
            smoothing_length = 1.33 * particle[np].rad
            bodyID1 = particle[np].bodyID
            renormalization_matrix_inv = ZEROMAT3x3
            temporary_vec = ZEROVEC3f

            x_begin = ti.max(grid_idx[0] - 1, 0)
            x_end = ti.min(grid_idx[0] + 2, cnum[0])
            y_begin = ti.max(grid_idx[1] - 1, 0)
            y_end = ti.min(grid_idx[1] + 2, cnum[1])
            z_begin = ti.max(grid_idx[2] - 1, 0)
            z_end = ti.min(grid_idx[2] + 2, cnum[2])
            
            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                            neighborID = particleID[hash_index]
                            if np == neighborID: continue
                            bodyID2 = particle[neighborID].bodyID
                            if bodyID1 == bodyID2:
                                rel_coord = particle[neighborID].x - p_coord
                                kernel_gradient = GGuassian(smoothing_length, -rel_coord)
                                renormalization_matrix_inv += particle[neighborID].m / particle[neighborID].mass_density * kernel_gradient.outer_product(rel_coord)
                                temporary_vec += particle[neighborID].m / particle[neighborID].mass_density * kernel_gradient
            normal = Normalize(-renormalization_matrix_inv.inverse() @ temporary_vec)
        particle[np].normal = normal


@ti.kernel
def find_free_surface_by_geometry(igrid_size: float, cnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), particleID: ti.template(), particle_count: ti.template()):
    for np in range(particleNum):
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
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        t_coord = p_coord + smoothing_length * normal
                        for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                            neighborID = particleID[hash_index]
                            if np == neighborID: continue
                            n_coord = particle[neighborID].x 
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
def find_free_surface_by_geometry_eigen(icell_size: float, radius: float, cnum: ti.types.vector(3, int), start_point: ti.types.vector(3, float), particleNum: int, particle: ti.template(), particleID: ti.template(), hash_table: ti.template()):
    for np in range(particleNum):
        if int(particle[np].free_surface) == 1:
            p_coord = particle[np].x - start_point
            grid_idx = ti.floor(p_coord * icell_size, int)
            smoothing_length = 1.33 * radius
            renormalization_matrix_inv = ZEROMAT3x3
            temporary_vec = ZEROVEC3f

            x_begin = ti.max(grid_idx[0] - 2, 0)
            x_end = ti.min(grid_idx[0] + 3, cnum[0])
            y_begin = ti.max(grid_idx[1] - 2, 0)
            y_end = ti.min(grid_idx[1] + 3, cnum[1])
            z_begin = ti.max(grid_idx[2] - 2, 0)
            z_end = ti.min(grid_idx[2] + 3, cnum[2])
            
            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(hash_table[cellID].current, hash_table[cellID].current + hash_table[cellID].count):
                            neighborID = particleID[hash_index]
                            if np == neighborID: continue
                            rel_coord = particle[neighborID].x - p_coord - start_point
                            kernel_gradient = GGuassian(smoothing_length, -rel_coord)
                            renormalization_matrix_inv += particle[neighborID].vol * kernel_gradient.outer_product(rel_coord)
                            temporary_vec += particle[neighborID].vol * kernel_gradient
            
            vector = get_eigenvalue(renormalization_matrix_inv)
            lambda_ = vector[0]
            normal = Normalize(-renormalization_matrix_inv.inverse() @ temporary_vec)
            if lambda_ <= 0.76:
                if lambda_ > 0.2:
                    for neigh_i in range(x_begin, x_end):
                        for neigh_j in range(y_begin, y_end):
                            for neigh_k in range(z_begin, z_end):
                                cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                                t_coord = p_coord + smoothing_length * normal
                                for hash_index in range(hash_table[cellID].current, hash_table[cellID].current + hash_table[cellID].count):
                                    neighborID = particleID[hash_index]
                                    if np == neighborID: continue
                                    n_coord = particle[neighborID].x - start_point
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
def kernel_calculate_surface_tension(kappa: float, igrid_size: float, cnum: ti.types.vector(3, int), particleNum: int, particle: ti.template(), particleID: ti.template(), particle_current: ti.template(), particle_count: ti.template()):
    # reference: J.P. Morris. Simulating surface tension with smoothed particle hydrodynamics. International Journal for Numerical Methods in Fluids. 2000 (33) 333-353.
    for np in range(particleNum):
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
            Na = 1. if Squared(normala) > epsilon * epsilon else 0.
            curvature_star, eta = 0., 0.
            for neigh_i in range(x_begin, x_end):
                for neigh_j in range(y_begin, y_end):
                    for neigh_k in range(z_begin, z_end):
                        cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                        for hash_index in range(particle_count[cellID] - particle_current[cellID], particle_count[cellID]):
                            neighborID = particleID[hash_index]
                            if np == neighborID: continue
                            rel_coord = particle[neighborID].x - p_coord
                            normalb = particle[neighborID].normal
                            Nb = 1. if Squared(normalb) > epsilon * epsilon else 0.
                            curvature_star += min(Na, Nb) * particle[neighborID].vol * (normalb - normala) * GGuassian(smoothing_length, -rel_coord)
                            eta += min(Na, Nb) * particle[neighborID].vol * Guassian(smoothing_length, -rel_coord)
            curvature = curvature_star / eta
            particle[np].external_force -= curvature * normala * kappa

