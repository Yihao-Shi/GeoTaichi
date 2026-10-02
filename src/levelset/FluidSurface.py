import taichi as ti


@ti.data_oriented
class Fluid_Surface:
    def __init__(self, dim=2, diff_n_grid = 100, diff_dx = 1, consider_tension = True):
        self.dim = dim
        self.diff_n_grid = diff_n_grid              # 表面网格数
        self.diff_dx = diff_dx                      # 网格单元大小
        self.diff_inv_dx = 1 / self.diff_dx         # 倒数
        self.consider_tension = consider_tension    # 是否考虑张力

        # surface level set
        self.sign_distance_field = ti.field(ti.f32, shape=(self.diff_n_grid,) * dim)
        self.edge = ti.Struct.field({"begin_point": ti.types.vector(2, ti.f32),
                                     "end_point": ti.types.vector(2, ti.f32)}, shape=self.diff_n_grid ** 2)
        self.edge_num = ti.field(int, shape=())

        # surface tension
        self.SDF = ti.Struct.field({"gradient": ti.types.vector(2, ti.f32),
                                     "laplacian": float}, shape=(self.diff_n_grid,) * dim)      # SDF梯度和曲率
        self.surface_particle_num = ti.field(ti.i32, shape=())                                  # 表面粒子数
        self.surface_particles = ti.Struct.field({"position": ti.types.vector(dim, ti.f32)}, shape=(1000))


        self.radius = self.diff_dx * 0.8                                            # 粒子半径 = 网格间距
        self.mc_result = ti.field(ti.i32, shape=(self.diff_n_grid,) * dim)          # 液表编号

    @ti.kernel
    def calculate_gradient(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j = I
            u, v = .0, .0
            # 根据不同边界条件，给出网格点的SDF的梯度
            if i == 0:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i, j]) * 0.5 * self.diff_inv_dx
            elif i == self.diff_n_grid - 1:
                u = (self.sign_distance_field[i, j] - self.sign_distance_field[i - 1, j]) * 0.5 * self.diff_inv_dx
            else:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i - 1, j]) * 0.5 * self.diff_inv_dx

            if j == 0:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j]) * 0.5 * self.diff_inv_dx
            elif j == self.diff_n_grid - 1:
                v = (self.sign_distance_field[i, j] - self.sign_distance_field[i, j - 1]) * 0.5 * self.diff_inv_dx
            else:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j - 1]) * 0.5 * self.diff_inv_dx
            self.SDF[I].gradient = ti.Vector([u, v]).normalized()

    @ti.kernel
    def calculate_laplacian(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j = I
            u, v = .0, .0
            # 处理 i 方向上的拉普拉斯算子
            if i == 0:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i, j]) * self.diff_inv_dx **2
            elif i == self.diff_n_grid - 1:
                u = (-self.sign_distance_field[i, j] + self.sign_distance_field[i - 1, j]) * self.diff_inv_dx **2
            else:
                u = (self.sign_distance_field[i + 1, j] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[i - 1, j]) * self.diff_inv_dx **2
            # 处理 j 方向上的拉普拉斯算子
            if j == 0:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j]) * self.diff_inv_dx **2
            elif j == self.diff_n_grid - 1:
                v = (-self.sign_distance_field[i, j] + self.sign_distance_field[i, j - 1]) * self.diff_inv_dx **2
            else:
                v = (self.sign_distance_field[i, j + 1] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[i, j - 1]) * self.diff_inv_dx **2
            # 计算拉普拉斯算子
            self.SDF[I].laplacian = u + v

    @ti.kernel
    def init_surface_particles(self):                                       # 每帧开始，将构建的表面粒子删除
        for i in range(self.surface_particle_num[None]):
            self.surface_particles[i].position = ti.Vector([0.0, 0.0])
        self.surface_particle_num[None] = 0

    @ti.kernel
    def create_particle(self):                                          # 根据edge容器生成表面节点坐标，并赋予给容器
        self.surface_particle_num[None] = 0

        for n in range(self.edge_num[None]):
            ab = self.edge[n].end_point - self.edge[n].begin_point      # 终点—起点坐标
            for i in ti.static(range(4)):
                pos = self.edge[n].begin_point + ((i / 3) * ab)         # 离散出更多点的坐标
                index = ti.atomic_add(self.surface_particle_num[None], 1)   # 计算所有表面粒子数量
                self.surface_particles[index].position = pos                # 储存表面粒子坐标

    @ti.func
    def linear_interpolation(self, pos: ti.template()):          # 线性插值函数：对周围四个网格角点的SDF梯度和拉普拉斯算子进行加权
        base = (pos * self.diff_inv_dx).cast(int)
        fx = pos * self.diff_inv_dx - base.cast(float)
        w = [(1 - fx) * self.diff_dx, fx * self.diff_dx]
        result_g = ti.Vector([0.0, 0.0])
        result_l = 0.0

        for i, j in ti.static(ti.ndrange(2, 2)):                # 线性插值：网格SDF梯度和算子 ——> 表面粒子
            weight = w[i][0] * w[j][1] * self.diff_inv_dx ** 2
            offset = ti.Vector([i, j])
            result_g += self.SDF[base + offset].gradient * weight
            result_l += self.SDF[base + offset].laplacian * weight
        return result_g, result_l

    def build_surface(self, simulator):
        self.init_surface_particles()                   # 初始化表面粒子
        self.calculate_gradient()                       # 计算SDF梯度
        self.calculate_laplacian()                      # 计算SDF曲率
        self.create_particle()                          # 根据显式液面线段生成表面粒子坐标