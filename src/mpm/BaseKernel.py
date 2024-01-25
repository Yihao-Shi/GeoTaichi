import taichi as ti

from src.utils.constants import MThreshold, Threshold, ZEROVEC3f
from src.utils.VectorFunction import SquareLen, Zero2OneVector, linear_id


@ti.kernel
def update_particle_storage_(particleNum: int, particle: ti.template(), stateVars: ti.template()) -> int:
    remaining_particle = 0
    ti.loop_config(serialize=True)
    for np in range(particleNum):
        if int(particle[np].active) == 1:
            particle[remaining_particle] = particle[np]
            stateVars[remaining_particle] = stateVars[np]
            remaining_particle += 1
    return remaining_particle

@ti.kernel
def find_max_radius_(particleNum: int, particle: ti.template()) -> float:
    rmax = 0.
    for np in range(particleNum):
        ti.atomic_max(rmax, particle[np].rad)
    return rmax

@ti.kernel
def find_min_z_position_(particleNum: int, particle: ti.template()) -> float:
    min_zpos = MThreshold
    for np in range(particleNum):
        zpos = particle[np].x[2]
        ti.atomic_min(min_zpos, zpos)
    return min_zpos

@ti.kernel
def find_max_velocity_(particleNum: int, particle: ti.template()) -> float:
    max_vel = 0.
    for np in range(particleNum):
        vel = particle[np].v.norm()
        ti.atomic_max(max_vel, vel)
    return max_vel


@ti.kernel
def find_particle_max_radius_(particleNum: int, particle: ti.template()) -> float:
    max_radius = 0.
    for np in range(particleNum):
        radius = particle[np].rad
        ti.atomic_max(max_radius, radius)
    return max_radius


@ti.kernel
def find_particle_min_radius_(particleNum: int, particle: ti.template()) -> float:
    min_radius = MThreshold
    for np in range(particleNum):
        radius = particle[np].rad
        ti.atomic_min(min_radius, radius)
    return min_radius


@ti.kernel
def find_particle_min_mass_(particleNum: int, particle: ti.template()) -> float:
    min_mass = MThreshold
    for np in range(particleNum):
        radius = particle[np].m
        ti.atomic_min(min_mass, radius)
    return min_mass


@ti.kernel
def modify_particle_bodyID_in_region(value: int, particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].bodyID = ti.u8(value)


@ti.kernel
def modify_particle_materialID_in_region(value: int, particleNum: int, particle: ti.template(), material: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].materialID = ti.u8(value)
            particle[np].m = particle[np].vol * material[int(particle[np].materialID)].density


@ti.kernel
def modify_particle_position_in_region(factor: int, value: ti.types.vector(3, float), particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].x = factor * particle[np].x + value


@ti.kernel
def modify_particle_velocity_in_region(factor: int, value: ti.types.vector(3, float), particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].v = factor * particle[np].v + value


@ti.kernel
def modify_particle_traction_in_region(factor: int, value: ti.types.vector(3, float), particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].traction = factor * particle[np].traction + value


@ti.kernel
def modify_particle_stress_in_region(factor: int, value: ti.types.vector(6, float), particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].stress = factor * particle[np].stress + value


@ti.kernel
def modify_particle_fix_v_in_region(value: ti.types.vector(3, ti.u8), particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].fix_v = value
            particle[np].unfix_v = Zero2OneVector(value)


@ti.kernel
def modify_particle_bodyID(value: int, particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == ti.u8(bodyID):
            particle[np].bodyID = ti.u8(value)


@ti.kernel
def modify_particle_materialID(value: int, particleNum: int, particle: ti.template(), material: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == ti.u8(bodyID):
            particle[np].materialID = ti.u8(value)
            particle[np].m = particle[np].vol * material[int(particle[np].materialID)].density


@ti.kernel
def modify_particle_position(factor: int, value: ti.types.vector(3, float), particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == ti.u8(bodyID):
            particle[np].x = factor * particle[np].x + value


@ti.kernel
def modify_particle_velocity(factor: int, value: ti.types.vector(3, float), particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == ti.u8(bodyID):
            particle[np].v = factor * particle[np].v + value


@ti.kernel
def modify_particle_traction(factor: int, value: ti.types.vector(3, float), particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == ti.u8(bodyID):
            particle[np].traction = factor * particle[np].traction + value


@ti.kernel
def modify_particle_stress(factor: int, value: ti.types.vector(6, float), particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == ti.u8(bodyID):
            particle[np].stress = factor * particle[np].stress + value


@ti.kernel
def modify_particle_fix_v(value: ti.types.vector(3, ti.u8), particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == ti.u8(bodyID):
            particle[np].fix_v = value
            particle[np].unfix_v = Zero2OneVector(value)


@ti.kernel
def kernel_initialize_particle_fbar(particle_fbar: ti.template()):
    for np in particle_fbar:
        particle_fbar[np].initialize()

    
@ti.kernel
def kernel_delete_particles(particleNum: int, particle: ti.template(), bodyID: int):
    for np in range(particleNum):
        if particle[np].bodyID == bodyID:
            particle[np].active = ti.u8(0)


@ti.kernel
def kernel_delete_particles_in_region(particleNum: int, particle: ti.template(), is_in_region: ti.template()):
    for np in range(particleNum):
        if is_in_region(particle[np].x):
            particle[np].active = ti.u8(0)


@ti.kernel
def check_in_domain(domain: ti.types.vector(3, float), particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        if int(particle[np].active) == 1 and not is_in_domain(domain, particle[np].x):
            particle[np].active = ti.u8(0)

@ti.func
def is_in_domain(domain, position):
    in_domain = 1
    if in_domain == 1 and position[0] < Threshold: in_domain = 0
    elif in_domain == 1 and position[1] < Threshold: in_domain = 0
    elif in_domain == 1 and position[2] < Threshold: in_domain = 0
    elif in_domain == 1 and position[0] > domain[0]: in_domain = 0
    elif in_domain == 1 and position[1] > domain[1]: in_domain = 0
    elif in_domain == 1 and position[2] > domain[2]: in_domain = 0
    return in_domain


@ti.kernel
def reset_verlet_disp_(particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        particle[np].verletDisp = ZEROVEC3f


@ti.kernel
def validate_particle_displacement_(limit: float, particleNum: int, particle: ti.template()) -> int:
    flag = 0
    for np in range(particleNum):
        if flag == 0 and SquareLen(particle[np].verletDisp) > limit:
            flag = 1
    return flag

# ========================================================= #
#                Assign Particle to Cell                    #
# ========================================================= #
@ti.kernel
def calculate_particles_position_(particleNum: int, igrid_size: float, particle: ti.template(), particle_count: ti.template(), cnum: ti.types.vector(3, int)):
    # TODO: using morton code
    particle_count.fill(0)
    for np in range(particleNum):  
        grid_idx = ti.floor(particle[np].x * igrid_size , int)
        cellID = linear_id(grid_idx, cnum)
        ti.atomic_add(particle_count[cellID], 1)
    
@ti.kernel
def insert_particle_to_cell_(igrid_size: float, particleNum: int, particle: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template(), cnum: ti.types.vector(3, int)):
    particle_current.fill(0)
    for np in range(particleNum):
        grid_idx = ti.floor(particle[np].x * igrid_size , int)
        cellID = linear_id(grid_idx, cnum)
        grain_location = particle_count[cellID] - ti.atomic_add(particle_current[cellID], 1) - 1
        particleID[grain_location] = np


@ti.kernel
def calculate_particles_position_v2_(particleNum: int, igrid_size: float, particle: ti.template(), particle_count: ti.template(), cnum: ti.types.vector(3, int)):
    # TODO: using morton code
    particle_count.fill(0)
    for np in range(particleNum):  
        if particle[np].free_surface == ti.u8(1):
            grid_idx = ti.floor(particle[np].x * igrid_size , int)
            cellID = linear_id(grid_idx, cnum)
            ti.atomic_add(particle_count[cellID], 1)
    
@ti.kernel
def insert_particle_to_cell_v2_(igrid_size: float, particleNum: int, particle: ti.template(), particle_count: ti.template(), particle_current: ti.template(), particleID: ti.template(), cnum: ti.types.vector(3, int)):
    particle_current.fill(0)
    for np in range(particleNum):
        if particle[np].free_surface == ti.u8(1):
            grid_idx = ti.floor(particle[np].x * igrid_size , int)
            cellID = linear_id(grid_idx, cnum)
            grain_location = particle_count[cellID] - ti.atomic_add(particle_current[cellID], 1) - 1
            particleID[grain_location] = np




