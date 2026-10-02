import taichi as ti


@ti.func
def get_sphere_aabb(point_cloud, radius):
    box_min = point_cloud - radius * ti.Vector.one(float, point_cloud.m)
    box_max = point_cloud + radius * ti.Vector.one(float, point_cloud.m)
    return box_min, box_max


@ti.func
def get_aabb(point_cloud):
    box_min = ti.Vector.zero(float, point_cloud.m)
    box_max = ti.Vector.zero(float, point_cloud.m)
    for i in ti.static(range(point_cloud.n)):
        for d in ti.static(range(point_cloud.m)):
            box_min[d] = ti.min(box_min[d], point_cloud[i, d])
            box_max[d] = ti.max(box_max[d], point_cloud[i, d])
    return box_min, box_max


@ti.func
def get_obb(point_cloud):
    center = ti.Vector.zero(float, point_cloud.m)
    for i in ti.static(range(point_cloud.n)):
        global_coord = ti.Vector([point_cloud[i, d] for d in ti.static(range(point_cloud.m))])
        center += global_coord
    center /= point_cloud.n

    consistent_matrix = ti.Matrix.zero(float, point_cloud.m, point_cloud.m)
    for i in ti.static(range(point_cloud.n)):
        global_coord = ti.Vector([point_cloud[i, d] for d in ti.static(range(point_cloud.m))])
        center_coord = global_coord - center
        consistent_matrix += center_coord.outer_product(center_coord)
    consistent_matrix /= point_cloud.n

    _, eigen_vector = ti.sym_eig(consistent_matrix)
    local_coords = ti.Matrix.zero(float, point_cloud.n, point_cloud.m)
    for i in ti.static(range(point_cloud.n)):
        global_coord = ti.Vector([point_cloud[i, d] for d in ti.static(range(point_cloud.m))])
        local_coord = eigen_vector.transpose() @ (global_coord - center)
        for d in ti.static(range(point_cloud.m)):
            local_coords[i, d] = local_coord[d]

    box_min, box_max = get_aabb(local_coords)
    return box_min, box_max, eigen_vector
