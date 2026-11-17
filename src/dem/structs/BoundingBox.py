import taichi as ti
from src.utils.constants import THRESHOLD, ZEROMAT2X2
from src.utils.ScalarFunction import bi_interpolate, linearize3d
from src.utils.TypeDefinition import vec3f, vec3i, vec2f


@ti.dataclass
class BoundingBox:
    xmin: vec3f
    xmax: vec3f
    shape_min: vec3f
    shape_max: vec3f
    shape_radius: float
    startGrid: int
    gnum: vec3i
    grid_space: float
    scale: float
    reference_surface_area: float
    extent: int

    @ti.func
    def _restart(self, gnum, xmin, xmax, start_grid, grid_space, scale, extent):
        self.gnum = int(gnum)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.shape_min = float(xmin)
        self.shape_max = float(xmax)
        self.shape_radius = 0.5 * (self.shape_max - self.shape_min).norm()
        self.startGrid = float(start_grid)
        self.grid_space = float(grid_space)
        self.scale = float(scale)
        self.reference_surface_area = 0.0
        self.extent = float(extent)

    @ti.func
    def _set_bounding_box(self, xmin, xmax):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.shape_min = float(xmin)
        self.shape_max = float(xmax)
        self.shape_radius = 0.5 * (self.shape_max - self.shape_min).norm()

    @ti.func
    def _get_center(self):
        return 0.5 * (self.xmin + self.xmax)

    @ti.func
    def _get_shape_center(self):
        return 0.5 * (self.shape_min + self.shape_max)

    @ti.func
    def _scale(self, centor_of_mass):
        self.xmin = float(self.scale * (self.xmin - centor_of_mass) + centor_of_mass)
        self.xmax = float(self.scale * (self.xmax - centor_of_mass) + centor_of_mass)
        self.shape_min = float(self.scale * (self.shape_min - centor_of_mass) + centor_of_mass)
        self.shape_max = float(self.scale * (self.shape_max - centor_of_mass) + centor_of_mass)
        self.shape_radius = float(self.scale * self.shape_radius)
        self.grid_space = float(self.scale * self.grid_space)

    @ti.func
    def _set_shape_box(self, shape_min, shape_max):
        self.shape_min = shape_min
        self.shape_max = shape_max

    @ti.func
    def _set_shape_radius(self, shape_radius):
        self.shape_radius = ti.max(float(shape_radius), 0.0)

    @ti.func
    def _add_grid(self, start_grid, grid_space, gnum, scale, extent):
        self.startGrid = int(start_grid)
        self.grid_space = float(grid_space)
        self.gnum = int(gnum)
        self.scale = float(scale)
        self.extent = int(extent)

    @ti.func
    def _set_reference_surface_area(self, reference_surface_area):
        self.reference_surface_area = float(reference_surface_area)

    @ti.func
    def _translate(self, offset):
        self.xmin += offset
        self.xmax += offset

    @ti.func
    def _get_dim(self):
        return self.xmax - self.xmin

    @ti.func
    def _get_shape_dim(self):
        return self.shape_max - self.shape_min

    @ti.func
    def _in_box(self, point):
        return (
            (point[0] >= self.xmin[0])
            and (point[0] <= self.xmax[0])
            and (point[1] >= self.xmin[1])
            and (point[1] <= self.xmax[1])
            and (point[2] >= self.xmin[2])
            and (point[2] <= self.xmax[2])
        )

    @ti.func
    def closet_corner(self, point):
        ret_indices = vec3i(0, 0, 0)
        for index in ti.static(range(3)):
            ret_indices[index] = ti.min(
                self.gnum[index] - 2, ti.max(0, int((point[index] - self.xmin[index]) / self.grid_space))
            )
        return ret_indices

    @ti.func
    def distance(self, point, grid):
        space = self.grid_space
        indices = self.closet_corner(point)
        x_ind, y_ind, z_ind = indices[0], indices[1], indices[2]

        ind000 = linearize3d(x_ind, y_ind, z_ind, self.gnum) + self.startGrid
        ind001 = linearize3d(x_ind, y_ind, z_ind + 1, self.gnum) + self.startGrid
        ind010 = linearize3d(x_ind, y_ind + 1, z_ind, self.gnum) + self.startGrid
        ind011 = linearize3d(x_ind, y_ind + 1, z_ind + 1, self.gnum) + self.startGrid
        ind100 = linearize3d(x_ind + 1, y_ind, z_ind, self.gnum) + self.startGrid
        ind101 = linearize3d(x_ind + 1, y_ind, z_ind + 1, self.gnum) + self.startGrid
        ind110 = linearize3d(x_ind + 1, y_ind + 1, z_ind, self.gnum) + self.startGrid
        ind111 = linearize3d(x_ind + 1, y_ind + 1, z_ind + 1, self.gnum) + self.startGrid

        temp1 = self.xmin + vec3i(x_ind, y_ind, z_ind) * space
        temp2 = self.xmin + vec3i(x_ind, y_ind + 1, z_ind) * space
        temp3 = self.xmin + vec3i(x_ind, y_ind, z_ind + 1) * space
        yz_coord = vec2f(point[1], point[2])
        y_extr = vec2f(temp1[1], temp2[1])
        z_extr = vec2f(temp1[2], temp3[2])

        known_valx0, known_valx1 = ZEROMAT2X2, ZEROMAT2X2
        known_valx0[0, 0] = grid[ind000].distance_field * self.scale
        known_valx0[0, 1] = grid[ind001].distance_field * self.scale
        known_valx0[1, 0] = grid[ind010].distance_field * self.scale
        known_valx0[1, 1] = grid[ind011].distance_field * self.scale

        known_valx1[0, 0] = grid[ind100].distance_field * self.scale
        known_valx1[0, 1] = grid[ind101].distance_field * self.scale
        known_valx1[1, 0] = grid[ind110].distance_field * self.scale
        known_valx1[1, 1] = grid[ind111].distance_field * self.scale

        f0yz = bi_interpolate(yz_coord, y_extr, z_extr, known_valx0)
        f1yz = bi_interpolate(yz_coord, y_extr, z_extr, known_valx1)
        return (point[0] - temp1[0]) / space * (f1yz - f0yz) + f0yz

    @ti.func
    def calculate_gradient(self, point, grid):
        indices = self.closet_corner(point)
        x_ind, y_ind, z_ind = indices[0], indices[1], indices[2]
        spacing = self.grid_space

        x_red = (point[0] - (self.xmin[0] + x_ind * spacing)) / spacing
        y_red = (point[1] - (self.xmin[1] + y_ind * spacing)) / spacing
        z_red = (point[2] - (self.xmin[2] + z_ind * spacing)) / spacing

        normal = vec3f(0, 0, 0)
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                for k in ti.static(range(2)):
                    Ind = linearize3d(x_ind + i, y_ind + j, z_ind + k, self.gnum) + self.startGrid
                    ls_val = grid[Ind].distance_field * self.scale
                    normal[0] += (
                        ls_val * (2 * i - 1) * ((1 - j) * (1 - y_red) + j * y_red) * ((1 - k) * (1 - z_red) + k * z_red)
                    )
                    normal[1] += (
                        ls_val * (2 * j - 1) * ((1 - i) * (1 - x_red) + i * x_red) * ((1 - k) * (1 - z_red) + k * z_red)
                    )
                    normal[2] += (
                        ls_val * (2 * k - 1) * ((1 - i) * (1 - x_red) + i * x_red) * ((1 - j) * (1 - y_red) + j * y_red)
                    )
        # The sums are derivatives with respect to reduced cell coordinates.
        return normal / ti.max(spacing, THRESHOLD)

    @ti.func
    def calculate_normal(self, point, grid):
        return self.calculate_gradient(point, grid).normalized()
