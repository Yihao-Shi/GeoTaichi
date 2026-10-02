import taichi as ti

from src.utils.constants import Threshold, ZEROVEC3f
from src.utils.Quaternion import SetToRotate
from src.utils.ScalarFunction import PairingMapping, linearize3D
from src.utils.TypeDefination import vec3f


@ti.func
def _valid_height(height, no_data):
    return ti.abs(height - no_data) > Threshold


@ti.func
def _heightfield_point(xind, yind, cell_size, height):
    return vec3f(float(xind) * cell_size, float(yind) * cell_size, height)


@ti.func
def _set_heightfield_triangle(point1, point2, point3):
    active = ti.u8(1)
    normal = ZEROVEC3f
    cross = (point2 - point1).cross(point3 - point1)
    norm = cross.norm()
    if norm > Threshold:
        normal = cross / norm
    else:
        active = ti.u8(0)
    return active, point1, normal


@ti.func
def _digital_elevation_triangle(position, cell_size, icell_size, cnum, height_dim, no_data, heightfield):
    active = ti.u8(0)
    point1 = ZEROVEC3f
    point2 = ZEROVEC3f
    point3 = ZEROVEC3f
    normal = ZEROVEC3f

    xcoord = position[0] * icell_size
    ycoord = position[1] * icell_size
    xind = ti.floor(xcoord, int)
    yind = ti.floor(ycoord, int)

    if xind >= 0 and xind < cnum[0] and yind >= 0 and yind < cnum[1]:
        ind00 = linearize3D(xind, yind, 0, height_dim)
        ind10 = linearize3D(xind + 1, yind, 0, height_dim)
        ind01 = linearize3D(xind, yind + 1, 0, height_dim)
        ind11 = linearize3D(xind + 1, yind + 1, 0, height_dim)

        height00 = heightfield[ind00]
        height10 = heightfield[ind10]
        height01 = heightfield[ind01]
        height11 = heightfield[ind11]

        valid00 = _valid_height(height00, no_data)
        valid10 = _valid_height(height10, no_data)
        valid01 = _valid_height(height01, no_data)
        valid11 = _valid_height(height11, no_data)

        point00 = _heightfield_point(xind, yind, cell_size, height00)
        point10 = _heightfield_point(xind + 1, yind, cell_size, height10)
        point01 = _heightfield_point(xind, yind + 1, cell_size, height01)
        point11 = _heightfield_point(xind + 1, yind + 1, cell_size, height11)

        if valid00 and valid10 and valid01 and valid11:
            local_x = xcoord - float(xind)
            local_y = ycoord - float(yind)
            if local_x >= local_y:
                active, point1, normal = _set_heightfield_triangle(point00, point10, point11)
            else:
                active, point1, normal = _set_heightfield_triangle(point11, point01, point00)
        elif (not valid00) and valid10 and valid01 and valid11:
            active, point1, normal = _set_heightfield_triangle(point10, point11, point01)
        elif valid00 and (not valid10) and valid01 and valid11:
            active, point1, normal = _set_heightfield_triangle(point00, point11, point01)
        elif valid00 and valid10 and (not valid01) and valid11:
            active, point1, normal = _set_heightfield_triangle(point00, point10, point11)
        elif valid00 and valid10 and valid01 and (not valid11):
            active, point1, normal = _set_heightfield_triangle(point00, point10, point01)
    return active, point1, normal


@ti.kernel
def kernel_particle_digital_elevation_heightfield_force_assemble_(
    particleNum: int,
    dt: ti.template(),
    max_material_num: int,
    surfaceProps: ti.template(),
    particle: ti.template(),
    terrain_material: int,
    cell_size: float,
    icell_size: float,
    cnum: ti.types.vector(2, int),
    height_dim: ti.types.vector(2, int),
    no_data: float,
    heightfield: ti.template(),
    cplist: ti.template(),
    model_type: ti.template(),
):
    for np in range(particleNum):
        if int(particle[np].active) == 0:
            cplist[np]._no_contact()
            continue

        position = particle[np]._get_position()
        active, point_on_plane, normal = _digital_elevation_triangle(
            position, cell_size, icell_size, cnum, height_dim, no_data, heightfield
        )
        if active == 0:
            cplist[np]._no_contact()
            continue

        radius = particle[np]._get_radius()
        distance = (position - point_on_plane).dot(normal)
        gapn = distance - radius

        materialID = PairingMapping(particle[np].materialID, terrain_material, max_material_num)
        if gapn < surfaceProps[materialID].ncut:
            velocity = particle[np]._get_velocity()
            angular_velocity = particle[np]._get_angular_velocity()
            mass = particle[np]._get_mass()
            projection = position - distance * normal
            contact_position = projection - 0.5 * gapn * normal
            relative_velocity = velocity + angular_velocity.cross(contact_position - position)
            tangential_overlap = cplist[np].oldTangOverlap

            if ti.static(model_type == 2):
                rad_eff = radius
                rolling_overlap = cplist[np].oldRollAngle
                twisting_overlap = cplist[np].oldTwistAngle
                wr_rel = normal.cross(angular_velocity)
                (
                    normal_force,
                    tangential_force,
                    momentum,
                    tangential_overlap_temp,
                    rolling_overlap_temp,
                    twisting_overlap_temp,
                ) = surfaceProps[materialID]._force_assemble(
                    mass,
                    rad_eff,
                    gapn,
                    1.0,
                    radius,
                    normal,
                    relative_velocity,
                    angular_velocity,
                    wr_rel,
                    tangential_overlap,
                    rolling_overlap,
                    twisting_overlap,
                    dt,
                )
                total_force = normal_force + tangential_force
                resultant_momentum = total_force.cross(position - contact_position) + momentum
                cplist[np]._set_contact(tangential_overlap_temp, rolling_overlap_temp, twisting_overlap_temp)
                particle[np]._update_contact_interaction(total_force, resultant_momentum)
            else:
                rad_eff = radius + 0.5 * gapn
                normal_force, tangential_force, momentum, tangential_overlap_temp = surfaceProps[
                    materialID
                ]._force_assemble(
                    mass,
                    rad_eff,
                    gapn,
                    1.0,
                    radius,
                    normal,
                    relative_velocity,
                    angular_velocity,
                    tangential_overlap,
                    dt,
                )
                total_force = normal_force + tangential_force
                resultant_momentum = total_force.cross(position - contact_position) + momentum
                cplist[np]._set_contact(normal_force, tangential_force, tangential_overlap_temp)
                particle[np]._update_contact_interaction(total_force, resultant_momentum)
        else:
            cplist[np]._no_contact()


@ti.kernel
def kernel_LSparticle_digital_elevation_heightfield_force_assemble_(
    surfaceNum: int,
    dt: ti.template(),
    max_material_num: int,
    surfaceProps: ti.template(),
    rigid: ti.template(),
    vertice: ti.template(),
    surface: ti.template(),
    box: ti.template(),
    terrain_material: int,
    cell_size: float,
    icell_size: float,
    cnum: ti.types.vector(2, int),
    height_dim: ti.types.vector(2, int),
    no_data: float,
    heightfield: ti.template(),
    cplist: ti.template(),
):
    for global_node in range(surfaceNum):
        end1 = surface[global_node]
        if int(rigid[end1].active) == 0:
            cplist[global_node]._no_contact()
            continue

        local_node = rigid[end1].global_node_to_local(global_node)
        mass_center = rigid[end1]._get_position()
        rotate_matrix = SetToRotate(rigid[end1].q)
        surface_node = mass_center + rotate_matrix @ (box[end1].scale * vertice[local_node].x)
        active, point_on_plane, normal = _digital_elevation_triangle(
            surface_node, cell_size, icell_size, cnum, height_dim, no_data, heightfield
        )
        if active == 0:
            cplist[global_node]._no_contact()
            continue

        gapn = (surface_node - point_on_plane).dot(normal)
        materialID = PairingMapping(rigid[end1].materialID, terrain_material, max_material_num)
        if gapn < surfaceProps[materialID].ncut:
            coeff = vertice[local_node].parameter
            contact_position = surface_node - 0.5 * gapn * normal
            velocity = rigid[end1]._get_velocity()
            angular_velocity = rigid[end1]._get_angular_velocity()
            relative_velocity = velocity + angular_velocity.cross(contact_position - mass_center)
            tangential_overlap = cplist[global_node].oldTangOverlap
            mass = rigid[end1]._get_mass()
            rad_eff = rigid[end1]._get_contact_radius(contact_position)
            normal_force, tangential_force, momentum, tangential_overlap_temp = surfaceProps[
                materialID
            ]._force_assemble(
                mass, rad_eff, gapn, coeff, rad_eff, normal, relative_velocity, angular_velocity, tangential_overlap, dt
            )

            total_force = normal_force + tangential_force
            resultant_momentum = total_force.cross(mass_center - contact_position) + momentum
            cplist[global_node]._set_contact(normal_force, tangential_force, tangential_overlap_temp)
            rigid[end1]._update_contact_interaction(total_force, resultant_momentum)
            if int(rigid[end1].is_soft) == 1 and cplist[global_node]._is_active():
                vertice[local_node]._update_contact_interaction(total_force, ZEROVEC3f)
        else:
            cplist[global_node]._no_contact()
