import numpy as np
from itertools import product

from src.utils.linalg import flip2d_linear


class QuadrilateralMesh:
    def __init__(self, nx, ny, dx, dy, ghost_cell):
        self.ghost_cell = ghost_cell
        self.node_connectivity = None
        self.nodal_coords = None
        self.face_map = {}
        self.grid_num = np.array([nx, ny])
        self.grid_size = np.array([dx, dy])
        self.set_nodal_coords(nx, ny, dx, dy)
        self.set_node_connectivity(nx, ny)
        self.generate_structured_grid(nx, ny, dx, dy)

    def set_nodal_coords(self, nx, ny, dx=1.0, dy=1.0):
        X = np.linspace(-self.ghost_cell * dx, (nx - self.ghost_cell) * dx, int(nx + 1))
        Y = np.linspace(-self.ghost_cell * dy, (ny - self.ghost_cell) * dy, int(ny + 1))
        self.nodal_coords = flip2d_linear(np.array(list(product(X, Y))), size_u=X.shape[0], size_v=Y.shape[0])

    def set_node_connectivity(self, nx, ny):
        total_cell_number = nx * ny
        self.node_connectivity = np.zeros((total_cell_number, 4), dtype=int)

        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

        n0 = i * (ny + 1) + j
        n1 = (i + 1) * (ny + 1) + j
        n2 = (i + 1) * (ny + 1) + (j + 1)
        n3 = i * (ny + 1) + (j + 1)

        self.node_connectivity = np.stack([n0, n1, n2, n3], axis=-1).reshape(-1, 4)

    def generate_structured_grid(self, nx, ny, dx=1.0, dy=1.0):
        # x-direction faces (vertical)
        i = np.arange(nx + 1)
        j = np.arange(ny)
        ii, jj = np.meshgrid(i, j, indexing='ij')

        x_face_centers = np.stack([ii.ravel() * dx, (jj.ravel() + 0.5) * dy], axis=1)

        face2cell_x = []
        for id_i in i:
            for id_j in j:
                cells = []
                if id_i < nx:
                    cells.append((id_i, id_j))
                if id_i > 0:
                    cells.append((id_i - 1, id_j))
                face2cell_x.append(cells)

        tags_x = np.where((ii == 0) | (ii == nx), "boundary", "internal").ravel()

        self.face_map['x'] = {
            "face_centers": x_face_centers,
            "face2cell": np.array(face2cell_x, dtype=object),
            "tags": tags_x
        }

        # y-direction faces (horizontal)
        i = np.arange(nx)
        j = np.arange(ny + 1)
        ii, jj = np.meshgrid(i, j, indexing='ij')

        y_face_centers = np.stack([(ii.ravel() + 0.5) * dx, jj.ravel() * dy], axis=1)

        face2cell_y = []
        for id_i in i:
            for id_j in j:
                cells = []
                if id_j < ny:
                    cells.append((id_i, id_j))
                if id_j > 0:
                    cells.append((id_i, id_j - 1))
                face2cell_y.append(cells)

        tags_y = np.where((jj == 0) | (jj == ny), "boundary", "internal").ravel()

        self.face_map['y'] = {
            "face_centers": y_face_centers,
            "face2cell": np.array(face2cell_y, dtype=object),
            "tags": tags_y
        }

    def write(self, filename='Element.txt'):
        print('#', "Writing cell(s) into 'Element.txt' ......")
        with open(filename, 'w') as f:
            f.write("# QuadMesh Export\n")
            f.write("\n[node_connectivity]\n")
            for row in self.node_connectivity:
                f.write(" ".join(map(str, map(int, row))) + "\n")

            for direction in ['x', 'y']:
                if direction not in self.face_map:
                    continue
                face_data = self.face_map[direction]
                f.write(f"\n[face_{direction}]\n")

                f.write("face_centers\n")
                for center in face_data['face_centers']:
                    f.write(" ".join(f"{x:.6f}" for x in center) + "\n")

                f.write("face2cell\n")
                for cells in face_data['face2cell']:
                    f.write(" ".join(f"{c[0]} {c[1]}" for c in cells) + "\n")

                f.write("tags\n")
                for tag in face_data['tags']:
                    f.write(tag + "\n")

    def read(self, filename='Element.txt'):
        print('#', "Reading 'Element.txt' into cell(s) ......")
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        i = 0
        while i < len(lines):
            line = lines[i]
            if line == "[node_connectivity]":
                nodes = []
                i += 1
                while i < len(lines) and not lines[i].startswith("["):
                    nodes.append(list(map(int, lines[i].split())))
                    i += 1
                self.node_connectivity = np.array(nodes)
                continue

            elif line.startswith("[face_"):
                direction = line[-1]
                face_centers = []
                face2cell = []
                tags = []

                i += 1
                while i < len(lines) and not lines[i].startswith("["):
                    if lines[i] == "face_centers":
                        i += 1
                        while i < len(lines) and lines[i] not in ["face2cell", "tags"]:
                            face_centers.append(list(map(float, lines[i].split())))
                            i += 1
                    if i < len(lines) and lines[i] == "face2cell":
                        i += 1
                        while i < len(lines) and lines[i] != "tags":
                            tokens = list(map(int, lines[i].split()))
                            cells = [tuple(tokens[j:j+2]) for j in range(0, len(tokens), 2)]
                            face2cell.append(cells)
                            i += 1
                    if i < len(lines) and lines[i] == "tags":
                        i += 1
                        while i < len(lines) and not lines[i].startswith("["):
                            tags.append(lines[i])
                            i += 1

                self.face_map[direction] = {
                    "face_centers": np.array(face_centers),
                    "face2cell": np.array(face2cell, dtype=object),
                    "tags": np.array(tags)
                }
                continue

            i += 1
