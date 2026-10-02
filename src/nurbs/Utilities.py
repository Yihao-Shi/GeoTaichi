import numpy as np

TOL = 2.2204460492503131e-14



def make_list(input):
    if isinstance(input, (list, tuple, np.ndarray)):
        return list(input)
    else:
        return [input]
    
def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(matrix[i, j], end=' ')
        print('\n')


def print_vector(vector):
    for i in range(vector.shape[0]):
        print(vector[i])


def normalized(arr):
    arr_norm = np.linalg.norm(arr)
    if arr_norm != 0:
        return arr / np.linalg.norm(arr)
    else: return arr


def assemble_point_weight(point, weight):
    return np.vstack(((point.T * weight), weight)).T


def split_point_weight(wpoint):
    return (wpoint[:,0:-1].T / wpoint[:,-1]).T, wpoint[:,-1]


def check_closed_boundary(xi, knot):
    if xi < knot[0]: xi += knot[-1] - knot[0]
    elif xi > knot[-1]: xi -= knot[-1] - knot[0]
    return xi


def check_open_boundary(xi, knot):
    if xi < knot[0]: xi = knot[0]
    elif xi > knot[-1]: xi = knot[-1]
    return xi


def normalize(knot_vector: np.ndarray):
    if not 0.0 <= knot_vector.all() <= 1.0:
        return knot_vector
    else:
        max_knot = np.max(knot_vector)
        min_knot = np.min(knot_vector)

        normalized = knot_vector - min_knot * np.ones(knot_vector.shape)
        normalized *= 1 / (max_knot - min_knot)
        return normalized


def check_knot_vector(degree, knot_vector):
    knot_vector = normalize(np.asarray(knot_vector))
    if not np.allclose(np.zeros(degree + 1), knot_vector[:degree + 1]):
        raise TypeError("Knot vector error")
    if not np.allclose(np.ones(degree + 1), knot_vector[-1 * (degree + 1)]):
        raise TypeError("Knot vector error")
    previous_knot = knot_vector[0]
    for knot in knot_vector:
        if knot < previous_knot:
            raise TypeError("Knot vector must be non-decreasing")
        previous_knot = knot
    return knot_vector


def generate_uniform(degree, num_ctrlpts):
    length_knot_vector = num_ctrlpts + degree + 1
    num_middle_knots = length_knot_vector - 2 * degree
    middle_knot_vector = np.linspace(0, 1, num_middle_knots)
    knot_vector = np.concatenate((np.zeros(degree), middle_knot_vector, np.ones(degree)))
    return knot_vector


def evaluate_bounding_box(ctrlpts):
    dimension = len(ctrlpts[0])
    bbmin = [float('inf') for _ in range(0, dimension)]
    bbmax = [float('-inf') for _ in range(0, dimension)]
    for cpt in ctrlpts:
        for i, arr in enumerate(zip(cpt, bbmin)):
            if arr[0] < arr[1]:
                bbmin[i] = arr[0]
        for i, arr in enumerate(zip(cpt, bbmax)):
            if arr[0] > arr[1]:
                bbmax[i] = arr[0]
    return np.asarray(bbmin), np.asarray(bbmax)


def check_same_side(point1, point2, point):
    vec1 = point2 - point
    vec2 = point1 - point
    return np.dot(vec1, vec2) >= 0


def solve_using_cod(A, b, rcond=1e-12):
    Q, R = np.linalg.qr(A, mode='reduced')
    U, S, VT = np.linalg.svd(R, full_matrices=False)
    tol = rcond * S.max()
    S_inv = np.diag(np.where(S > tol, 1.0 / S, 0.0))
    R_inv = VT.T @ S_inv @ U.T
    rhs = Q.T @ b
    x = R_inv @ rhs
    return x


def project_point_to_plane(points):
    points = np.asarray(points)
    relative_positions = points - np.mean(points, axis=0) 
    cov_matrix = np.cov(relative_positions, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues /= points.shape[0]
    x_coords = np.dot(relative_positions, eigenvectors[:,0]) 
    y_coords = np.dot(relative_positions, eigenvectors[:,1]) 
    return np.column_stack((x_coords, y_coords)) 


def point_to_edges(point, vertice1, vertice2, norm="L2"):
    pa = point - vertice1
    ba = vertice2 - vertice1
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    if norm == "L1":
        return np.sum(np.abs(pa - ba * h))
    elif norm == "L2":
        return np.linalg.norm(pa - ba * h)
    else:
        raise ValueError("please choose L1 or L2 norm")
    

def point_to_point(point1, point2, axis=None, norm="L2"):
    if norm == "L1":
        return np.sum(np.abs(point1 - point2), axis=axis)
    elif norm == "L2":
        return np.linalg.norm(point1 - point2, axis=axis)
    else:
        raise ValueError("please choose L1 or L2 norm")


def get_determinant_3x2(matrix):
    U, sigma, V = np.linalg.svd(matrix)
    return sigma[0] * sigma[1]


def project_evd(matrix):
    eigenvalues, Q = np.linalg.eigh(matrix)
    eigenvalues_proj = np.maximum(eigenvalues, 0)
    A_proj = Q @ np.diag(eigenvalues_proj) @ Q.T
    return A_proj


def quadratic_root(a, b, c):
    root = np.array([np.inf, np.inf])
    if b * b - 4. * a * c > 0.:
        root[0] = 0.5 * (-b + np.sqrt(b * b - 4. * a * c)) / a
        root[1] = 0.5 * (-b - np.sqrt(b * b - 4. * a * c)) / a
    return root


def quadrilateral_to_triangle_connectivity(connectivity):
    window_size = 3
    shape = connectivity.shape[:-1] + (connectivity.shape[-1] - window_size + 1, window_size)
    strides = connectivity.strides + (connectivity.strides[-1],)
    return np.lib.stride_tricks.as_strided(connectivity, shape=shape, strides=strides)


def get_ctrlpt_dofs(arr, dim=3):
    return np.repeat(arr, dim) * dim + np.tile(np.arange(dim), len(arr))


def insert_midpoints(points):
    points = np.asarray(points)
    midpoints = 0.5 * (points[:-1] + points[1:])
    new_points = np.empty((len(points) + len(midpoints), 2))
    new_points[0::2] = points
    new_points[1::2] = midpoints
    return new_points


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + 0.5 * (x[-1]*y[0] - x[0]*y[-1])

def tsp_greedy(points, initial_index=0):
    n = len(points)
    if n == 0:
        return points
    visited = np.zeros(n, dtype=bool)
    path = [initial_index]
    visited[initial_index] = True
    current_index = initial_index
    for _ in range(n - 1):
        remaining_indices = np.where(~visited)[0]
        distances = np.linalg.norm(points[remaining_indices] - points[current_index], axis=1)
        next_index = remaining_indices[np.argmin(distances)]
        path.append(next_index)
        visited[next_index] = True
        current_index = next_index
    return np.array(path)

def sort_points_counterclockwise(points, initial_index=0, method=0):
    if method == 0:
        order = tsp_greedy(points, initial_index)
        sorted_points = points[order]
        area = polygon_area(sorted_points)
        if area < 0:
            # 保持起始点不变，反转其余点的顺序
            sorted_points = np.concatenate(([sorted_points[0]], sorted_points[1:][::-1]))
        return sorted_points
    elif method == 1:
        # 找到重心
        centroid = np.mean(points, axis=0)
        
        # 计算每个点相对于重心的角度
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]