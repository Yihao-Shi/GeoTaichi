import numpy as np
from scipy.stats import lognorm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import time
import taichi as ti

@ti.kernel
def check_overlap(grid: ti.types.ndarray(), radius_array:ti.types.ndarray(), positions: ti.types.ndarray(), radius: ti.types.ndarray(), overlap: ti.template()):
    for i, j in ti.ndrange(grid.shape[0], positions.shape[0]):
        dist = (grid[i,0]-positions[j,0])**2 + (grid[i,1]-positions[j,1])**2 + (grid[i,2]-positions[j,2])**2
        if dist < (radius_array[i] + radius[j])**2:
            overlap[i] = 0
    
def compute_lattice_dims(N, lx, ly, lz):
    avg_vol = (lx * ly * lz) / N
    cell_size = avg_vol ** (1/3)
    nx = max(1, int(np.floor(lx / cell_size)))
    ny = max(1, int(np.floor(ly / cell_size)))
    nz = max(1, int(np.floor(lz / cell_size)))
    while nx * ny * nz < N:
        if (lx / nx) >= (ly / ny) and (lx / nx) >= (lz / nz):
            nx += 1
        elif (ly / ny) >= (lz / nz):
            ny += 1
        else:
            nz += 1
    return nx, ny, nz

def place_large_particles_lattice(radius_array, x_range, y_range, z_range, epsilon_ratio=1e-3):
    N = len(radius_array)
    lx, ly, lz = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]

    # 根据颗粒数计算 lattice 尺寸
    nx, ny, nz = compute_lattice_dims(N, lx, ly, lz)

    # spacing 根据 box 尺寸决定，保证均匀填充
    spacing_x = lx / nx
    spacing_y = ly / ny
    spacing_z = lz / nz
    spacing = min(spacing_x, spacing_y, spacing_z)

    xs = np.linspace(x_range[0] + spacing/2, x_range[1] - spacing/2, nx)
    ys = np.linspace(y_range[0] + spacing/2, y_range[1] - spacing/2, ny)
    zs = np.linspace(z_range[0] + spacing/2, z_range[1] - spacing/2, nz)
    grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)

    if len(grid) < N:
        raise ValueError("空间不足，放不下所有大颗粒，请扩大domain或减小半径")

    np.random.shuffle(grid)
    selected = grid[:N]

    # 微小扰动
    perturb = (np.random.uniform(-1,1,(N,3)) * epsilon_ratio * radius_array[:, None])
    positions = selected + perturb

    return positions, radius_array, spacing

def place_particles_lattice_once(radius_array, x_range, y_range, z_range, epsilon_ratio=1e-3):
    N = len(radius_array)
    lx, ly, lz = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]

    # 根据颗粒数计算 lattice 尺寸
    nx, ny, nz = compute_lattice_dims(N, lx, ly, lz)

    # spacing 根据 box 尺寸决定，保证均匀填充
    spacing_x = lx / nx
    spacing_y = ly / ny
    spacing_z = lz / nz
    spacing = min(spacing_x, spacing_y, spacing_z)

    xs = np.linspace(x_range[0] + spacing/2, x_range[1] - spacing/2, nx)
    ys = np.linspace(y_range[0] + spacing/2, y_range[1] - spacing/2, ny)
    zs = np.linspace(z_range[0] + spacing/2, z_range[1] - spacing/2, nz)
    grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)

    if len(grid) < N:
        raise ValueError("空间不足，放不下所有大颗粒，请扩大domain或减小半径")

    np.random.shuffle(grid)
    selected = grid[:N]

    # 微小扰动
    perturb = (np.random.uniform(-1,1,(N,3)) * epsilon_ratio * radius_array[:, None])
    positions = selected + perturb

    return positions, radius_array

def place_particles_lattice_twice(radius_array, x_range, y_range, z_range, ratio, positions, radius, epsilon_ratio=1e-3):
    N_expand = int(len(radius_array) / ratio * 1.1) # 预留10%冗余
    N = len(radius_array)
    lx, ly, lz = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]

    # 根据颗粒数计算 lattice 尺寸
    nx, ny, nz = compute_lattice_dims(N_expand, lx, ly, lz)

    # spacing 根据 box 尺寸决定，保证均匀填充
    spacing_x = lx / nx
    spacing_y = ly / ny
    spacing_z = lz / nz
    spacing = min(spacing_x, spacing_y, spacing_z)

    xs = np.linspace(x_range[0] + spacing/2, x_range[1] - spacing/2, nx)
    ys = np.linspace(y_range[0] + spacing/2, y_range[1] - spacing/2, ny)
    zs = np.linspace(z_range[0] + spacing/2, z_range[1] - spacing/2, nz)
    grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)

    overlap = ti.field(ti.f32, shape=grid.shape[0])
    overlap.fill(1)
    check_overlap(grid, radius_array, positions, radius, overlap)
    grid_valid = grid[overlap.to_numpy()==1]

    if len(grid_valid) < N:
        raise ValueError("空间不足，放不下所有小颗粒，请扩大domain或减小半径")

    np.random.shuffle(grid_valid)
    selected = grid_valid[:N]

    # 微小扰动
    perturb = (np.random.uniform(-1,1,(N,3)) * epsilon_ratio * radius_array[:, None])
    positions = selected + perturb

    return positions, radius_array
# def place_small_particles_lattice(radius_array, x_range, y_range, z_range, large_positions, large_radii, large_spacing, epsilon_ratio=1e-3):
#     N = len(radius_array)
#     lx, ly, lz = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]

#     small_spacing = large_spacing / 14.5  # 小颗粒 lattice 间距细一点

#     nx = max(1, int(np.floor(lx / small_spacing)))
#     ny = max(1, int(np.floor(ly / small_spacing)))
#     nz = max(1, int(np.floor(lz / small_spacing)))

#     xs = np.linspace(x_range[0], x_range[1], nx)
#     ys = np.linspace(y_range[0], y_range[1], ny)
#     zs = np.linspace(z_range[0], z_range[1], nz)
#     grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)

#     tree = cKDTree(large_positions)

#     candidate_positions = []
#     candidate_radii = []
#     idx_radius = 0

#     for pos_small in grid:
#         if idx_radius >= N:
#             break
#         r_small = radius_array[idx_radius]
#         perturb = np.random.uniform(-1,1,3) * epsilon_ratio * r_small
#         pos_perturbed = pos_small + perturb

#         dist, idx = tree.query(pos_perturbed)
#         if dist > large_radii[idx] + r_small:
#             candidate_positions.append(pos_perturbed)
#             candidate_radii.append(r_small)
#             idx_radius += 1

#     if idx_radius < N:
#         print(f"警告：只成功放置了{idx_radius}个小颗粒，目标{N}")

#     return np.array(candidate_positions), np.array(candidate_radii)


def expectation_of_x_3_lognorm_x(mu, sigma):
    return np.exp(3 * mu + 4.5 * sigma ** 2)

def disribute_particle_number(N, w_1, m_1, w_2, m_2):
    N_1 = int(N * w_1 * m_2 / (m_2 * w_1 + m_1 * w_2))
    N_2 = N - N_1
    return N_1, N_2

def generate_lognormal_particle_sizes(N, mean_linear, std_linear, dmin, dmax):
    oversample = int(N * 1.5)
    diameters = lognorm.rvs(s=std_linear, scale=mean_linear, size=oversample)
    valid = diameters[(diameters >= dmin) & (diameters <= dmax)]
    while len(valid) < N:
        extra = lognorm.rvs(s=std_linear, scale=mean_linear, size=(N - len(valid)) * 2)
        valid = np.concatenate([valid, extra[(extra >= dmin) & (extra <= dmax)]])
    return np.sort(valid[:N])


def write_sphere_text(N, x_range_1, y_range_1, z_range_1, x_range_2, y_range_2, z_range_2, group_1, group_2, vis=False, tol=1e-3, max_iter=10, group=3):
    print('# Writing spheres into SpherePacking ......')
    print(f"Total requested spheres: {N}")

    mu_1 = np.log(group_1['D50'])
    mu_2 = np.log(group_2['D50'])
    m_1 = expectation_of_x_3_lognorm_x(mu_1, group_1['sigma'])
    m_2 = expectation_of_x_3_lognorm_x(mu_2, group_2['sigma'])
    
    # 初始分配
    num_group_1, num_group_2 = disribute_particle_number(N, group_1['omega'], m_1, group_2['omega'], m_2)

    target = group_1['omega']
    N1_low, N1_high = int(0.9 * num_group_1), int(1.2 * num_group_1)
    best_N1, best_ratio = None, None

    for it in range(max_iter):
        N1 = (N1_low + N1_high) // 2
        N2 = N - N1

        radius_1 = generate_lognormal_particle_sizes(
            N1, group_1['D50']*1e6, group_1['sigma']*1e6,
            group_1['dmin']*1e6, group_1['dmax']*1e6
        ) / 2. * 1e-6
        radius_2 = generate_lognormal_particle_sizes(
            N2, group_2['D50']*1e6, group_2['sigma']*1e6,
            group_2['dmin']*1e6, group_2['dmax']*1e6
        ) / 2. * 1e-6

        ratio = np.sum(radius_1**3) / (np.sum(radius_1**3) + np.sum(radius_2**3))
        print(f"迭代 {it+1}: N1={N1}, N2={N2}, 实际质量比={ratio:.4f}")

        best_N1, best_ratio = N1, ratio

        if abs(ratio - target) < tol:
            break

        if ratio > target:
            # 大颗粒太多，减少 N1
            N1_high = N1
        else:
            # 大颗粒太少，增加 N1
            N1_low = N1

    print(f"最终结果: 大颗粒数量 {best_N1}, 小颗粒数量 {N-best_N1}")
    print(f"最终实际质量比: {ratio:.4f}")

    pos1, rad1 = place_particles_lattice_once(radius_1, x_range_1, y_range_1, z_range_1)
    ratio_finished = np.sum(rad1**3) / (np.sum(radius_1**3) + np.sum(radius_2**3))

    print("大颗粒放置完成")

    group_num = int(len(radius_2) / group)
    # 小颗粒分几组放置
    for i in range(group-1):
        pos2, rad2 = place_particles_lattice_twice(radius_2[i*group_num:(i+1)*group_num], x_range_2, y_range_2, z_range_2, 1-ratio_finished, pos1, rad1)
        pos1 = np.vstack([pos1, pos2])
        rad1 = np.hstack([rad1, rad2])
        ratio_finished = np.sum(rad1**3) / (np.sum(radius_1**3) + np.sum(radius_2**3))
        print(f"第 {i+1} 组小颗粒放置完成")

    pos2, rad2 = place_particles_lattice_twice(radius_2[(group-1)*group_num:], x_range_2, y_range_2, z_range_2, 1-ratio_finished, pos1, rad1)
    print("小颗粒全部放置完成")
    positions = np.vstack([pos2, pos1])
    radii = np.hstack([rad2, rad1])
    idx = np.argsort(radii)
    radii_sorted = radii[idx]
    positions_sorted = positions[idx]

    radii = radii_sorted
    positions = positions_sorted


    np.savetxt('SpherePacking.txt', np.column_stack((positions, radii)),
               header="# PositionX PositionY PositionZ Radius", comments='')

    if vis:
        plt.hist(radius_1*2, bins=50, alpha=0.6, label='Large particles (diameter)')
        plt.xlabel('Diameter (µm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.hist(radius_2*2, bins=50, alpha=0.6, label='Small particles (diameter)')
        plt.xlabel('Diameter (µm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    t0 = time.time()
    N = int(1.4e7)  # 根据机器性能调整总数，示例用10万颗粒
    group_1 = {'D50': 2*8.5e-6,
               'sigma': 0.45e-6,
               'dmin': 2*0.5e-6,
               'dmax': 2*35e-6,
               'omega': 0.7}

    group_2 = {'D50': 2*0.5e-6,
               'sigma': 0.45e-6,
               'dmin': 2*0.25e-6,
               'dmax': 2*15e-6,
               'omega': 0.3}

    x_range_1 = (9e-6, 351e-6)
    y_range_1 = (9e-6, 351e-6)
    z_range_1 = (9e-6, 351e-6)

    x_range_2 = (6e-7, 3594e-7)
    y_range_2 = (6e-7, 3594e-7)
    z_range_2 = (6e-7, 3594e-7)

    write_sphere_text(N, x_range_1, y_range_1, z_range_1, x_range_2, y_range_2, z_range_2, group_1, group_2, vis=True, group=10)
    print(f"总用时: {time.time() - t0:.2f} 秒")


