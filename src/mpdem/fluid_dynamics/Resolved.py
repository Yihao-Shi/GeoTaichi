import taichi as ti

from src.utils.BitFunction import Zero2One
from src.utils.constants import ZEROVEC3f, DBL_EPSILON
from src.utils.TypeDefination import vec3f, vec3i
from src.utils.ScalarFunction import linearize3D, vectorize_id
from src.utils.VectorFunction import SquaredLength, inner_multiply, Squared
from src.utils.Quaternion import SetToRotate


@ti.kernel
def calculate_sdf(self, gridSum: int, gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), iserach_size: float, cnum: ti.types.vector(3, int), 
                    particle_count: ti.template(), particleID: ti.template(), particle: ti.template()):
    self.sdf.fill(0)
    self.sdfID.fill(-1)
    for ng in range(gridSum):
        I = vectorize_id(ng, gnum)
        position = I * grid_size
        grid_idx = ti.floor(position * iserach_size - 0.5, int)
        
        x_begin = ti.max(grid_idx[0], 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1], 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2], 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        neighbor = particleID[hash_index]
                        radius = particle[neighbor].rad
                        sdf = SquaredLength(position, particle[neighbor].x)
                        if sdf < radius * radius and ti.abs(self.sdf[I]) < 10 * DBL_EPSILON:
                            self.sdf[I] = ti.sqrt(sdf) - radius
                            self.sdfID[I] = neighbor

@ti.kernel
def calculate_sdf_level_set(self, gridSum: int, gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), iserach_size: float, cnum: ti.types.vector(3, int), 
                            particle_count: ti.template(), particleID: ti.template(), rigid: ti.template(), box: ti.template(), vertice: ti.template(), grid: ti.template()):
    self.sdf.fill(0)
    self.sdfID.fill(-1)
    for ng in range(gridSum):
        I = vectorize_id(ng, gnum)
        position = I * grid_size
        grid_idx = ti.floor(position * iserach_size - 0.5, int)
        
        x_begin = ti.max(grid_idx[0], 0)
        x_end = ti.min(grid_idx[0] + 2, cnum[0])
        y_begin = ti.max(grid_idx[1], 0)
        y_end = ti.min(grid_idx[1] + 2, cnum[1])
        z_begin = ti.max(grid_idx[2], 0)
        z_end = ti.min(grid_idx[2] + 2, cnum[2])

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                for neigh_k in range(z_begin, z_end):
                    cellID = linearize3D(neigh_i, neigh_j, neigh_k, cnum)
                    for hash_index in range(particle_count[cellID], particle_count[cellID + 1]):
                        neighbor = particleID[hash_index]
                        mass_center = rigid[neighbor]._get_position()
                        rotate_matrix = SetToRotate(rigid[neighbor].q)
                        local_node = rotate_matrix.transpose() @ (position - mass_center)
                        if not box[neighbor]._in_box(local_node): continue
                        sdf = box[neighbor].distance(local_node, grid)
                        if sdf < 0.:
                            self.sdf[I] = sdf
                            self.sdfID[I] = neighbor

@ti.func
def compute_signs(self, I):
    sign_sum = -1
    if self.sdf[I] > 0:
        sign_sum = 1
    return sign_sum

@ti.kernel
def calcualte_sphereical_volume_fraction(self, grid_size: ti.types.vector(3, float), particle: ti.template()):
    for I in ti.grouped(self.fraction):
        sign_sum = 0
        sign_sum += self.compute_signs(I + vec3i(0, 0, 0))
        sign_sum += self.compute_signs(I + vec3i(1, 0, 0))
        sign_sum += self.compute_signs(I + vec3i(0, 1, 0))
        sign_sum += self.compute_signs(I + vec3i(1, 1, 0))
        sign_sum += self.compute_signs(I + vec3i(0, 0, 1))
        sign_sum += self.compute_signs(I + vec3i(1, 0, 1))
        sign_sum += self.compute_signs(I + vec3i(0, 1, 1))
        sign_sum += self.compute_signs(I + vec3i(1, 1, 1))

        all_out = 0
        if sign_sum == 8: all_out = 1
        elif sign_sum == -8: all_out = -1
        
        if all_out == 1:
            self.fraction[I] = 0.
        elif all_out == -1:
            self.fraction[I] = 1.
        else:
            self.fraction[I] = self.calculate_cell_fraction1(I, grid_size, particle)

@ti.kernel
def calcualte_level_set_volume_fraction(self, grid_size: ti.types.vector(3, float), box: ti.template(), rigid: ti.template(), grid: ti.template()):
    for I in ti.grouped(self.fraction):
        sign_sum = 0
        sign_sum += self.compute_signs(I + vec3i(0, 0, 0))
        sign_sum += self.compute_signs(I + vec3i(1, 0, 0))
        sign_sum += self.compute_signs(I + vec3i(0, 1, 0))
        sign_sum += self.compute_signs(I + vec3i(1, 1, 0))
        sign_sum += self.compute_signs(I + vec3i(0, 0, 1))
        sign_sum += self.compute_signs(I + vec3i(1, 0, 1))
        sign_sum += self.compute_signs(I + vec3i(0, 1, 1))
        sign_sum += self.compute_signs(I + vec3i(1, 1, 1))

        all_out = 0
        if sign_sum == 8: all_out = 1
        elif sign_sum == -8: all_out = -1
        
        if all_out == 1:
            self.fraction[I] = 0.
        elif all_out == -1:
            self.fraction[I] = 1.
        else:
            self.fraction[I] = self.calculate_cell_fraction1(I, grid_size, box, grid, rigid)

@ti.func
def sum_weights(self, I, center_dist, diag_length):
    weight = 0.125
    if self.sdf[I] > 0.:
        weight = -center_dist / diag_length
    return weight

@ti.func
def calculate_cell_fraction1(self, I, grid_size, particle):
    # Reference: Parallel Open Source CFD-DEM for Resolved Particle-fluid Interaction
    pID = self.sdfID[I]
    center = (I + 0.5) * grid_size
    center_dist = ti.min(0, ti.sqrt(SquaredLength(center, particle[center].x)) - particle[pID].rad)
    diag_length = 0.5 * ti.sqrt(Squared(grid_size))

    frac = 0.
    frac += self.sum_weights(I + vec3i(0, 0, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 0, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(0, 1, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 1, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(0, 0, 1), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 0, 1), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(0, 1, 1), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 1, 1), center_dist, diag_length)
    return frac

@ti.func
def calculate_cell_fraction1_level_set(self, I, grid_size, box, grid, rigid):
    # Reference: Parallel Open Source CFD-DEM for Resolved Particle-fluid Interaction
    pID = self.sdfID[I]
    mass_center = rigid[pID]._get_position()
    rotate_matrix = SetToRotate(rigid[pID].q)
    center = (I + 0.5) * grid_size
    local_node = rotate_matrix.transpose() @ (center - mass_center)
    center_dist = ti.min(0, box[pID].distance(local_node, grid))
    diag_length = 0.5 * ti.sqrt(Squared(grid_size))

    frac = 0.
    frac += self.sum_weights(I + vec3i(0, 0, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 0, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(0, 1, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 1, 0), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(0, 0, 1), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 0, 1), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(0, 1, 1), center_dist, diag_length)
    frac += self.sum_weights(I + vec3i(1, 1, 1), center_dist, diag_length)
    return frac

@ti.func
def calculate_apex(self, I_curr, grid_size):
    p, pause = ZEROVEC3f, False
    I_diag = Zero2One(I_curr)
    phiA = self.sdf[I_curr]
    phiB = self.sdf[I_diag]
    if phiA * phiB <= 0:
        A = I_curr * grid_size
        B = I_diag * grid_size
        p = A - ti.abs(phiA) / (DBL_EPSILON + ti.abs(phiA) + ti.abs(phiB)) * (A - B)
        pause = True
    return p, pause

@ti.func
def calculate_apexs(self, I, grid_size):
    apex, pause = self.calculate_apex(I, grid_size)
    for k in range(self.dimension):
        apex, pause = self.calculate_apex(I + ti.Vector.unit(self.dimension, k), grid_size)
        if pause:
            break
    return apex

@ti.func
def calculate_line_fraction(self, phi1, phi2):
    frac = 0.
    if phi1 > 0 and phi2 > 0:
        frac = 0.
    if phi1 <= 0 and phi2 <= 0:
        frac = 1.
    if (phi1 > 0) // phi2 < 0:
        frac = -phi2 / (phi1 - phi2)
    else:
        frac = -phi1 / (phi2 - phi1)
    return frac

@ti.func
def calculate_face_fraction(self, index1, index2, index3, index4, grid_size):
    vertice1 = index1 * grid_size
    vertice2 = index2 * grid_size
    vertice3 = index3 * grid_size
    vertice4 = index4 * grid_size

    apex, pause = self.calculate_apex(index1, grid_size)
    if not pause:
        apex, pause = self.calculate_apex(index2, grid_size)

    area = 0.
    # bottom left vertice
    phiO = self.sdf[index1]
    phiA = self.sdf[index2]
    O = vertice1
    A = vertice2
    area += ti.abs(0.5 * (A - O).cross(apex - O).norm()) * self.calculate_line_fraction(phiO, phiA)
    # bottom right vertice
    phiO = self.sdf[index2]
    phiA = self.sdf[index3]
    O = vertice2
    A = vertice3
    area += ti.abs(0.5 * (A - O).cross(apex - O).norm()) * self.calculate_line_fraction(phiO, phiA)
    # top left vertice
    phiO = self.sdf[index3]
    phiA = self.sdf[index4]
    O = vertice3
    A = vertice4
    area += ti.abs(0.5 * (A - O).cross(apex - O).norm()) * self.calculate_line_fraction(phiO, phiA)
    # top right vertice
    phiO = self.sdf[index4]
    phiA = self.sdf[index1]
    O = vertice4
    A = vertice1
    area += ti.abs(0.5 * (A - O).cross(apex - O).norm()) * self.calculate_line_fraction(phiO, phiA)

@ti.func
def calculate_cell_fraction2_level_set(self, I, grid_size, box, grid, rigid):
    # Reference: Effective Geometric Algorithms for Immersed Boundary Method Using Signed Distance Field
    apex = self.calculate_apexs(I, grid_size)
    
    volume = 0. 
    # front face
    index1 = I + vec3i(0, 0, 0)
    index2 = I + vec3i(1, 0, 0)
    index3 = I + vec3i(0, 0, 1)
    index4 = I + vec3i(1, 0, 1)
    fcenter = (I + vec3f(0.5, 0, 0.5)) * grid_size
    fnorm = vec3f(0, 1, 0)
    eps_f = self.calculate_face_fraction(index1, index2, index3, index4)
    volume += 1./3. * eps_f * ti.abs((apex - fcenter).dot(fnorm))
    # back face
    index1 = I + vec3i(0, 1, 0)
    index2 = I + vec3i(1, 1, 0)
    index3 = I + vec3i(0, 1, 1)
    index4 = I + vec3i(1, 1, 1)
    fcenter = (I + vec3f(0.5, 1, 0.5)) * grid_size
    fnorm = vec3f(0, -1, 0)
    eps_f = self.calculate_face_fraction(index1, index2, index3, index4)
    volume += 1./3. * eps_f * ti.abs((apex - fcenter).dot(fnorm))
    # left face
    index1 = I + vec3i(0, 0, 0)
    index2 = I + vec3i(0, 1, 0)
    index3 = I + vec3i(0, 1, 1)
    index4 = I + vec3i(0, 0, 1)
    fcenter = (I + vec3f(0., 0.5, 0.5)) * grid_size
    fnorm = vec3f(1, 0, 0)
    eps_f = self.calculate_face_fraction(index1, index2, index3, index4)
    volume += 1./3. * eps_f * ti.abs((apex - fcenter).dot(fnorm))
    # right face
    index1 = I + vec3i(1, 0, 0)
    index2 = I + vec3i(1, 1, 0)
    index3 = I + vec3i(1, 1, 1)
    index4 = I + vec3i(1, 0, 1)
    fcenter = (I + vec3f(1., 0.5, 0.5)) * grid_size
    fnorm = vec3f(-1, 0, 0)
    eps_f = self.calculate_face_fraction(index1, index2, index3, index4)
    volume += 1./3. * eps_f * ti.abs((apex - fcenter).dot(fnorm))
    # bottom face
    index1 = I + vec3i(0, 0, 0)
    index2 = I + vec3i(1, 0, 0)
    index3 = I + vec3i(1, 1, 0)
    index4 = I + vec3i(0, 1, 0)
    fcenter = (I + vec3f(0.5, 0.5, 0.)) * grid_size
    fnorm = vec3f(0, 0, 1)
    eps_f = self.calculate_face_fraction(index1, index2, index3, index4)
    volume += 1./3. * eps_f * ti.abs((apex - fcenter).dot(fnorm))
    # top face
    index1 = I + vec3i(0, 0, 1)
    index2 = I + vec3i(1, 0, 1)
    index3 = I + vec3i(1, 1, 1)
    index4 = I + vec3i(0, 1, 1)
    fcenter = (I + vec3f(0.5, 0.5, 1.)) * grid_size
    fnorm = vec3f(0, 0, -1)
    eps_f = self.calculate_face_fraction(index1, index2, index3, index4)
    volume += 1./3. * eps_f * ti.abs((apex - fcenter).dot(fnorm))
    return volume / inner_multiply(grid_size)

@ti.func
def compute_weights(self, cellID, verticeID, box, grid, rigid):
    pass

@ti.func
def calculate_cell_fraction3_level_set(self, I, grid_size, box, grid, rigid):
    # Reference: Signed distance field enhanced fully resolved CFD-DEM for simulation of granular flows involving multiphase fluids and irregularly shaped particles
    frac = 0.
    frac += self.compute_weights(I, I + vec3i(0, 0, 0), box, grid, rigid)
    frac += self.compute_weights(I, I + vec3i(1, 0, 0), box, grid, rigid)
    frac += self.compute_weights(I, I + vec3i(0, 1, 0), box, grid, rigid)
    frac += self.compute_weights(I, I + vec3i(1, 1, 0), box, grid, rigid)
    frac += self.compute_weights(I, I + vec3i(0, 0, 1), box, grid, rigid)
    frac += self.compute_weights(I, I + vec3i(1, 0, 1), box, grid, rigid)
    frac += self.compute_weights(I, I + vec3i(0, 1, 1), box, grid, rigid)
    frac += self.compute_weights(I, I + vec3i(1, 1, 1), box, grid, rigid)
    return frac
    