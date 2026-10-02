"""Taichi kernels for DEM clump-template geometry."""

import taichi as ti

from src.utils.constants import PI, ZEROMAT3x3, ZEROVEC3f
from src.utils.TypeDefination import vec4f, vec3f


@ti.kernel
def _kernel_get_max_radius_(nspheres: int, rad_pebble: ti.types.ndarray()) -> float:
    max_radius = 0.0
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if max_radius < rad_pebble[pebble]:
            max_radius = rad_pebble[pebble]
    return max_radius


@ti.kernel
def _kernel_get_min_radius_(nspheres: int, rad_pebble: ti.types.ndarray()) -> float:
    min_radius = 1e15
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if min_radius > rad_pebble[pebble]:
            min_radius = rad_pebble[pebble]
    return min_radius


@ti.kernel
def _kernel_bounding_box_left_limit_(
    nspheres: int, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()
) -> ti.types.vector(3, float):
    xmin = ZEROVEC3f + 1e15
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if x_pebble[pebble, 0] - rad_pebble[pebble] < xmin[0]:
            xmin[0] = x_pebble[pebble, 0] - rad_pebble[pebble]
        if x_pebble[pebble, 1] - rad_pebble[pebble] < xmin[1]:
            xmin[1] = x_pebble[pebble, 1] - rad_pebble[pebble]
        if x_pebble[pebble, 2] - rad_pebble[pebble] < xmin[2]:
            xmin[2] = x_pebble[pebble, 2] - rad_pebble[pebble]
    return xmin


@ti.kernel
def _kernel_bounding_box_right_limit_(
    nspheres: int, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()
) -> ti.types.vector(3, float):
    xmax = ZEROVEC3f - 1e15
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        if x_pebble[pebble, 0] + rad_pebble[pebble] > xmax[0]:
            xmax[0] = x_pebble[pebble, 0] + rad_pebble[pebble]
        if x_pebble[pebble, 1] + rad_pebble[pebble] > xmax[1]:
            xmax[1] = x_pebble[pebble, 1] + rad_pebble[pebble]
        if x_pebble[pebble, 2] + rad_pebble[pebble] > xmax[2]:
            xmax[2] = x_pebble[pebble, 2] + rad_pebble[pebble]
    return xmax


@ti.kernel
def _kernel_bounding_sphere_1(
    nspheres: int, visit: ti.template(), x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()
) -> ti.types.vector(4, float):
    x_bound, r_bound = ZEROVEC3f, 1e15
    ti.loop_config(serialize=True)
    for _ in range(200):
        for pebble in range(nspheres):
            visit[pebble] = 0

        isphere, nvisit = -1, 0
        while isphere < 0 or visit[isphere] == 1 or isphere >= nspheres:
            isphere = int(ti.random() * nspheres)

        nvisit += 1
        visit[isphere] = 1

        x_bound_temp = x_pebble[isphere]
        rbound_temp = rad_pebble[isphere]

        while nvisit < nspheres:
            while isphere < 0 or visit[isphere] == 1 or isphere >= nspheres:
                isphere = int(ti.random() * nspheres)

            nvisit += 1
            visit[isphere] = 1

            d = x_pebble[isphere] - x_bound_temp
            dist = d.norm()

            if dist + rad_pebble[isphere] > rbound_temp:
                fact = (dist + rad_pebble[isphere] - rbound_temp) / (2.0 * dist)
                d *= fact
                x_bound_temp += d
                rbound_temp += d.norm()

        if rbound_temp < r_bound:
            r_bound = rbound_temp
            x_bound = x_bound_temp
    return vec4f([x_bound, r_bound])


@ti.kernel
def _kernel_bounding_sphere_2(
    nspheres: int, x_pebble: ti.types.ndarray(), rad_pebble: ti.types.ndarray()
) -> ti.types.vector(3, float):
    volume = 0.0
    pvol = vec3f(0, 0, 0)
    for pebble in range(nspheres):
        radius = rad_pebble[pebble]
        position = vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2])
        volume += 4.0 / 3.0 * PI * radius * radius * radius
        pvol += 4.0 / 3.0 * PI * radius * radius * radius * position
    return pvol / volume


@ti.kernel
def _kernel_bounding_radius_(
    nspheres: int,
    bounding_center: ti.types.vector(3, float),
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
) -> float:
    bounding_radius = 0.0
    for pebble in range(nspheres):
        ti.atomic_max(
            bounding_radius,
            rad_pebble[pebble]
            + (vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2]) - bounding_center).norm(),
        )
    return bounding_radius


@ti.kernel
def _kernel_check_bounding_sphere_(
    nspheres: int, x_bound: ti.types.vector(3, float), r_bound: float, x_pebble: ti.types.ndarray()
):
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        temp = x_bound - vec3f(x_pebble[pebble, 0], x_pebble[pebble, 1], x_pebble[pebble, 2])
        if temp.norm() > r_bound:
            print(
                f"Bounding sphere calculation for template failed, pebble{pebble} is out of range. Try number should increase."
            )


@ti.kernel
def _kernel_center_of_mass_1(
    ntry: int,
    nspheres: int,
    xcm: ti.template(),
    xmin: ti.types.vector(3, float),
    xmax: ti.types.vector(3, float),
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
) -> float:
    nsuccess, x_try = 0, ZEROVEC3f
    ti.loop_config(serialize=True)
    for _ in range(ntry):
        x_try[0] = xmin[0] + (xmax[0] - xmin[0]) * ti.random()
        x_try[1] = xmin[1] + (xmax[1] - xmin[1]) * ti.random()
        x_try[2] = xmin[2] + (xmax[2] - xmin[2]) * ti.random()

        alreadyChecked = False
        for pebble in range(nspheres):
            dist_j_sqr = (
                (x_try[0] - x_pebble[pebble, 0]) * (x_try[0] - x_pebble[pebble, 0])
                + (x_try[1] - x_pebble[pebble, 1]) * (x_try[1] - x_pebble[pebble, 1])
                + (x_try[2] - x_pebble[pebble, 2]) * (x_try[2] - x_pebble[pebble, 2])
            )
            if alreadyChecked:
                break
            if dist_j_sqr < rad_pebble[pebble] * rad_pebble[pebble]:
                xcm[0] = (xcm[0] * nsuccess + x_try[0]) / (nsuccess + 1)
                xcm[1] = (xcm[1] * nsuccess + x_try[1]) / (nsuccess + 1)
                xcm[2] = (xcm[2] * nsuccess + x_try[2]) / (nsuccess + 1)
                nsuccess += 1
                alreadyChecked = True

    # transform into a system with center of mass=0/0/0
    for pebble in range(nspheres):
        for d in ti.static(range(3)):
            x_pebble[pebble, d] -= xcm[d]
    return nsuccess


@ti.kernel
def _kernel_center_of_mass_2(
    grid_size: float,
    nspheres: int,
    xmin: ti.types.vector(3, float),
    xmax: ti.types.vector(3, float),
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
) -> ti.types.vector(4, float):
    gnum = ti.ceil((xmax - xmin) / grid_size, int)
    gridSum = ti.cast(gnum[0] * gnum[1] * gnum[2], int)
    volume = 0.0
    volume_center = vec3f(0, 0, 0)
    for ng in range(gridSum):
        i = (ng % (gnum[0] * gnum[1])) % gnum[0]
        j = (ng % (gnum[0] * gnum[1])) // gnum[0]
        k = ng // (gnum[0] * gnum[1])
        grid_center = (vec3f(i, j, k) + 0.5) * grid_size + xmin

        is_overlap = 0
        for npebble in range(nspheres):
            if (
                grid_center - vec3f(x_pebble[npebble, 0], x_pebble[npebble, 1], x_pebble[npebble, 2])
            ).norm() < rad_pebble[npebble]:
                is_overlap = 1
                break

        if is_overlap == 1:
            volume += grid_size * grid_size * grid_size
            volume_center += grid_size * grid_size * grid_size * grid_center
    volume_center /= volume

    for pebble in range(nspheres):
        for d in ti.static(range(3)):
            x_pebble[pebble, d] -= volume_center[d]
    return vec4f(volume_center[0], volume_center[1], volume_center[2], volume)


@ti.kernel
def _kernel_inertia_moment_1(
    ntry: int,
    nspheres: int,
    xcm: ti.types.vector(3, float),
    xmin: ti.types.vector(3, float),
    xmax: ti.types.vector(3, float),
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
) -> ti.types.matrix(3, 3, float):
    x_try, moi_vol = ZEROVEC3f, ZEROMAT3x3
    ti.loop_config(serialize=True)
    for _ in range(ntry):
        x_try[0] = xmin[0] + (xmax[0] - xmin[0]) * ti.random()
        x_try[1] = xmin[1] + (xmax[1] - xmin[1]) * ti.random()
        x_try[2] = xmin[2] + (xmax[2] - xmin[2]) * ti.random()

        alreadyChecked = False
        for pebble in range(nspheres):
            if alreadyChecked:
                break
            dist_j_sqr = (
                (x_try[0] - x_pebble[pebble, 0]) * (x_try[0] - x_pebble[pebble, 0])
                + (x_try[1] - x_pebble[pebble, 1]) * (x_try[1] - x_pebble[pebble, 1])
                + (x_try[2] - x_pebble[pebble, 2]) * (x_try[2] - x_pebble[pebble, 2])
            )

            if dist_j_sqr < rad_pebble[pebble] * rad_pebble[pebble]:
                moi_vol[0, 0] += (x_try[1] - xcm[1]) * (x_try[1] - xcm[1]) + (x_try[2] - xcm[2]) * (x_try[2] - xcm[2])
                moi_vol[0, 1] -= (x_try[0] - xcm[0]) * (x_try[1] - xcm[1])
                moi_vol[0, 2] -= (x_try[0] - xcm[0]) * (x_try[2] - xcm[2])
                moi_vol[1, 0] -= (x_try[1] - xcm[1]) * (x_try[0] - xcm[0])
                moi_vol[1, 1] += (x_try[0] - xcm[0]) * (x_try[0] - xcm[0]) + (x_try[2] - xcm[2]) * (x_try[2] - xcm[2])
                moi_vol[1, 2] -= (x_try[1] - xcm[1]) * (x_try[2] - xcm[2])
                moi_vol[2, 0] -= (x_try[2] - xcm[2]) * (x_try[0] - xcm[0])
                moi_vol[2, 1] -= (x_try[2] - xcm[2]) * (x_try[1] - xcm[1])
                moi_vol[2, 2] += (x_try[0] - xcm[0]) * (x_try[0] - xcm[0]) + (x_try[1] - xcm[1]) * (x_try[1] - xcm[1])
                alreadyChecked = True
    return moi_vol


@ti.kernel
def _kernel_inertia_moment_2(
    grid_size: float,
    nspheres: int,
    xcm: ti.types.vector(3, float),
    xmin: ti.types.vector(3, float),
    xmax: ti.types.vector(3, float),
    x_pebble: ti.types.ndarray(),
    rad_pebble: ti.types.ndarray(),
) -> ti.types.matrix(3, 3, float):
    gnum = ti.ceil((xmax - xmin) / grid_size, int)
    gridSum = ti.cast(gnum[0] * gnum[1] * gnum[2], int)
    moi_vol = ZEROMAT3x3
    for ng in range(gridSum):
        i = (ng % (gnum[0] * gnum[1])) % gnum[0]
        j = (ng % (gnum[0] * gnum[1])) // gnum[0]
        k = ng // (gnum[0] * gnum[1])
        grid_center = (vec3f(i, j, k) + 0.5) * grid_size + xmin

        is_overlap = 0
        for npebble in range(nspheres):
            if (
                grid_center - vec3f(x_pebble[npebble, 0], x_pebble[npebble, 1], x_pebble[npebble, 2])
            ).norm() < rad_pebble[npebble]:
                is_overlap = 1
                break

        if is_overlap == 1:
            moi_vol[0, 0] += grid_center[1] * grid_center[1] + grid_center[2] * grid_center[2]
            moi_vol[1, 1] += grid_center[0] * grid_center[0] + grid_center[2] * grid_center[2]
            moi_vol[2, 2] += grid_center[1] * grid_center[1] + grid_center[0] * grid_center[0]
            moi_vol[0, 1] -= grid_center[0] * grid_center[1]
            moi_vol[1, 0] -= grid_center[0] * grid_center[1]
            moi_vol[0, 2] -= grid_center[0] * grid_center[2]
            moi_vol[2, 0] -= grid_center[0] * grid_center[2]
            moi_vol[1, 2] -= grid_center[1] * grid_center[2]
            moi_vol[2, 1] -= grid_center[1] * grid_center[2]
    return grid_size * grid_size * grid_size * moi_vol


@ti.kernel
def _kernel_jacobi_(moi: ti.types.matrix(3, 3, float), evectors: ti.template()) -> ti.types.vector(3, float):
    error = 1
    evalues, b, z = ZEROVEC3f, ZEROVEC3f, ZEROVEC3f
    matrix = ZEROMAT3x3

    for i, j in ti.static(ti.ndrange(3, 3)):
        matrix[i, j] = moi[i, j]
        moi[i, j] = 32

    for i in ti.static(range(3)):
        b[i] = matrix[i, i]
        evalues[i] = matrix[i, i]

    ti.loop_config(serialize=True)
    for iter in range(50):
        sm = 0.0
        for i in range(2):
            for j in range(i + 1, 3):
                sm += ti.abs(matrix[i, j])
        if sm == 0.0:
            error = 0
            break

        tresh = 0.0
        if iter < 4:
            tresh = 0.2 * sm / (3 * 3)

        for i in range(2):
            for j in range(i + 1, 3):
                g = 100.0 * ti.abs(matrix[i, j])
                if (
                    iter > 4
                    and abs(evalues[i]) + g == ti.abs(evalues[i])
                    and ti.abs(evalues[j]) + g == ti.abs(evalues[j])
                ):
                    matrix[i, j] = 0.0
                elif ti.abs(matrix[i, j]) > tresh:
                    h = evalues[j] - evalues[i]
                    t = 0.0
                    if ti.abs(h) + g == ti.abs(h):
                        t = matrix[i, j] / h
                    else:
                        theta = 0.5 * h / matrix[i, j]
                        t = 1.0 / (ti.abs(theta) + ti.sqrt(1.0 + theta * theta))
                        if theta < 0.0:
                            t = -t
                    c = 1.0 / ti.sqrt(1.0 + t * t)
                    s = t * c
                    tau = s / (1.0 + c)
                    h = t * matrix[i, j]
                    z[i] -= h
                    z[j] += h
                    evalues[i] -= h
                    evalues[j] += h
                    matrix[i, j] = 0.0
                    for k in range(i):
                        u = matrix[k, i]
                        v = matrix[k, j]
                        matrix[k, i] = u - s * (v + u * tau)
                        matrix[k, j] = v + s * (u - v * tau)
                    for k in range(i + 1, j):
                        u = matrix[i, k]
                        v = matrix[k, j]
                        matrix[i, k] = u - s * (v + u * tau)
                        matrix[k, j] = v + s * (u - v * tau)
                    for k in range(j + 1, 3):
                        u = matrix[i, k]
                        v = matrix[j, k]
                        matrix[i, k] = u - s * (v + u * tau)
                        matrix[j, k] = v + s * (u - v * tau)
                    for k in range(3):
                        u = evectors[k, i]
                        v = evectors[k, j]
                        evectors[k, i] = u - s * (v + u * tau)
                        evectors[k, j] = v + s * (u - v * tau)

        for i in range(3):
            b[i] += z[i]
            evalues[i] = b[i]
            z[i] = 0.0

    if error == 1:
        print("Insufficient Jacobi rotations for rigid body")
    return evalues


@ti.kernel
def _kernel_pebble_cartesian_coosys_to_local_(
    nspheres: int,
    x_pebble: ti.types.ndarray(),
    ex_space: ti.types.vector(3, float),
    ey_space: ti.types.vector(3, float),
    ez_space: ti.types.vector(3, float),
):
    ti.loop_config(serialize=True)
    for pebble in range(nspheres):
        x_copy = ZEROVEC3f
        for d in ti.static(range(3)):
            x_copy[d] = x_pebble[pebble, d]

        x_pebble[pebble, 0] = x_copy[0] * ex_space[0] + x_copy[1] * ex_space[1] + x_copy[2] * ex_space[2]
        x_pebble[pebble, 1] = x_copy[0] * ey_space[0] + x_copy[1] * ey_space[1] + x_copy[2] * ey_space[2]
        x_pebble[pebble, 2] = x_copy[0] * ez_space[0] + x_copy[1] * ez_space[1] + x_copy[2] * ez_space[2]
