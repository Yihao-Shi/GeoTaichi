import taichi as ti

from src.utils.constants import PI, Threshold
from src.utils.TypeDefination import vec3i
from src.utils.ShapeFunctions import Guassian
from src.utils.ScalarFunction import linearize3D, vectorize_id


@ti.kernel
def compute_ambient_fluid_variable(sphereNum: int, dependent_domain: int, cnum: ti.types.vector(3, int), gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), igrid_size: ti.types.vector(3, float), gravity: ti.types.vector(3, float), 
                                   node: ti.template(), particle: ti.template(), sphere: ti.template(), cell_fraction: ti.template(), fluid_volume: ti.template(), fluid_velocity: ti.template(), fluid_volume_fraction: ti.template(), stress_divergence: ti.template()):
    fluid_volume.fill(0)
    fluid_volume_fraction.fill(0)
    fluid_velocity.fill(0)
    stress_divergence.fill(0)
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        particle_radius = particle[np].rad
        cell_volume = grid_size[0] * grid_size[1] * grid_size[2]

        base_bound = ti.max(ti.cast((position - dependent_domain * particle_radius) * igrid_size, int), 0)
        upper_bound = ti.min(ti.cast((position + dependent_domain * particle_radius) * igrid_size, int), cnum)
        for k, j, i in ti.ndrange((base_bound[2], upper_bound[2]), (base_bound[1], upper_bound[1]), (base_bound[0], upper_bound[0])):
            node_coords = (vec3i(i, j, k) + 0.5) * grid_size
            relative_coord = position - node_coords
            weights = Guassian(particle_radius, relative_coord, dependent_domain)
            fluid_volume_fraction[nsphere] += weights * cell_fraction[linearize3D(i, j, k, cnum)] * cell_volume
            fluid_volume[nsphere] += weights * cell_volume

        upper_bound1 = ti.min(ti.cast((position + dependent_domain * particle_radius) * igrid_size, int) + 1, cnum)
        for k, j, i in ti.ndrange((base_bound[2], upper_bound1[2]), (base_bound[1], upper_bound1[1]), (base_bound[0], upper_bound1[0])):
            node_coords = vec3i(i, j, k) * grid_size
            relative_coord = position - node_coords
            nodeID = linearize3D(i, j, k, gnum)
            weights = Guassian(particle_radius, relative_coord, dependent_domain)
            stress_divergence[nsphere] += weights * (node[nodeID, 0].force - node[nodeID, 0].m * gravity)
            fluid_velocity[nsphere] += weights * node[nodeID, 0].momentum * cell_volume

    for nsphere in range(sphereNum):
        fluid_volume_fraction[nsphere] = fluid_volume_fraction[nsphere] / fluid_volume[nsphere]
        fluid_velocity[nsphere] = fluid_velocity[nsphere] / fluid_volume[nsphere]
        stress_divergence[nsphere] = stress_divergence[nsphere] / fluid_volume[nsphere]


@ti.kernel
def compute_spherical_particle_fluid_force(sphereNum: int, influence_domain: int, cnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), igrid_size: ti.types.vector(3, float), gravity: ti.types.vector(3, float), 
                                           material: ti.template(), particle: ti.template(), sphere: ti.template(), cell_drag_force: ti.template(), fluid_volume: ti.template(), fluid_velocity: ti.template(), fluid_volume_fraction: ti.template(), stress_divergence: ti.template(), drag_coefficient_model: ti.template()):
    fluid_volume.fill(0)
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        particle_radius = particle[np].rad
        cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
        fluid_volume_mapping(nsphere, influence_domain, cnum, grid_size, igrid_size, position, particle_radius, cell_volume, fluid_volume)

    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        fluid_density, fluid_viscosity = material.density, material.viscosity
        position = particle[np].x
        particle_radius = particle[np].rad
        particle_volume = 4./3. * PI * particle_radius * particle_radius * particle_radius
        normalized_cell_volume = (grid_size[0] * grid_size[1] * grid_size[2]) / fluid_volume[nsphere]
        epsilon_f = fluid_volume_fraction[nsphere]
        velocity_f = fluid_velocity[nsphere]
        force_f = stress_divergence[nsphere]
    
        relative_velocity = velocity_f - particle[np].v
        drag_force = drag_coefficient_model.drag_law(epsilon_f, fluid_density, fluid_viscosity, particle_radius, relative_velocity)
        interaction_force = drag_force + particle_volume * force_f #- particle_volume * fluid_density * gravity 
        particle[np].contact_force += interaction_force
        drag_force_mapping(influence_domain, cnum, grid_size, igrid_size, position, particle_radius, normalized_cell_volume, cell_drag_force, -drag_force)


@ti.kernel
def compute_spherical_particle_void_fraction(sphereNum: int, cnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), igrid_size: ti.types.vector(3, float), 
                                           particle: ti.template(), sphere: ti.template(), cell_fraction: ti.template(), fluid_volume: ti.template()):
    fluid_volume.fill(0)
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        particle_radius = particle[np].rad
        cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
        fluid_volume_mapping(nsphere, 3, cnum, grid_size, igrid_size, position, particle_radius, cell_volume, fluid_volume)

    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        position = particle[np].x
        particle_radius = particle[np].rad
        particle_volume = 4./3. * PI * particle_radius * particle_radius * particle_radius
        normalized_cell_volume = (grid_size[0] * grid_size[1] * grid_size[2]) / fluid_volume[nsphere]
        void_fraction_mapping(cnum, grid_size, igrid_size, position, particle_radius, particle_volume, normalized_cell_volume, cell_fraction)

@ti.func
def fluid_volume_mapping(nsphere, support_size, cnum, grid_size, igrid_size, position, radius, cell_volume, fluid_volume):
    base_bound = ti.max(ti.cast((position - support_size * radius) * igrid_size, int), 0)
    upper_bound = ti.min(ti.cast((position + support_size * radius) * igrid_size, int), cnum)
    for k, j, i in ti.ndrange((base_bound[2], upper_bound[2]), (base_bound[1], upper_bound[1]), (base_bound[0], upper_bound[0])):
        node_coords = (vec3i(i, j, k) + 0.5) * grid_size
        relative_coord = position - node_coords
        weights = Guassian(radius, relative_coord, support_size)
        fluid_volume[nsphere] += weights * cell_volume

@ti.func
def void_fraction_mapping(cnum, grid_size, igrid_size, position, radius, volume, cell_volume, cell_fraction):
    base_bound = ti.max(ti.cast((position - 3 * radius) * igrid_size, int), 0)
    upper_bound = ti.min(ti.cast((position + 3 * radius) * igrid_size, int), cnum)
    for k, j, i in ti.ndrange((base_bound[2], upper_bound[2]), (base_bound[1], upper_bound[1]), (base_bound[0], upper_bound[0])):
        node_coords = (vec3i(i, j, k) + 0.5) * grid_size
        relative_coord = position - node_coords
        cell_fraction[linearize3D(i, j, k, cnum)] += Guassian(radius, relative_coord) * volume * cell_volume


@ti.func
def drag_force_mapping(influence_domain, cnum, grid_size, igrid_size, position, radius, cell_volume, cell_drag_force, drag_force):
    base_bound = ti.max(ti.cast((position - influence_domain * radius) * igrid_size, int), 0)
    upper_bound = ti.min(ti.cast((position + influence_domain * radius) * igrid_size, int), cnum)
    for k, j, i in ti.ndrange((base_bound[2], upper_bound[2]), (base_bound[1], upper_bound[1]), (base_bound[0], upper_bound[0])):
        node_coords = (vec3i(i, j, k) + 0.5) * grid_size
        relative_coord = position - node_coords
        cell_drag_force[linearize3D(i, j, k, cnum)] += Guassian(radius, relative_coord, influence_domain) * drag_force * cell_volume


@ti.kernel
def finalize_coupling(cellSum: int, cnum: ti.types.vector(3, int), gnum: ti.types.vector(3, int), grid_size: ti.types.vector(3, float), 
                      node: ti.template(), cell_fraction: ti.template(),  cell_drag_force: ti.template()):
    for nc in range(cellSum):
        if cell_fraction[nc] > Threshold:
            cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
            cell_fraction[nc] = 1. - cell_fraction[nc] / cell_volume
            averaged_drag_force = 0.125 * cell_drag_force[nc]

            ig, jg, kg = vectorize_id(nc, cnum)
            node[linearize3D(ig, jg, kg, gnum), 0].force += averaged_drag_force
            node[linearize3D(ig+1, jg, kg, gnum), 0].force += averaged_drag_force
            node[linearize3D(ig, jg+1, kg, gnum), 0].force += averaged_drag_force
            node[linearize3D(ig+1, jg+1, kg, gnum), 0].force += averaged_drag_force
            node[linearize3D(ig, jg, kg+1, gnum), 0].force += averaged_drag_force
            node[linearize3D(ig+1, jg, kg+1, gnum), 0].force += averaged_drag_force
            node[linearize3D(ig, jg+1, kg+1, gnum), 0].force += averaged_drag_force
            node[linearize3D(ig+1, jg+1, kg+1, gnum), 0].force += averaged_drag_force
