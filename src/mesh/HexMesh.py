import numpy as np
from itertools import product

from src.utils.linalg import flip3d_linear


class HexahedronMesh:
    def __init__(self, nx, ny, nz, dx, dy, dz, ghost_cell):
        self.ghost_cell = ghost_cell
        self.node_connectivity = None
        self.face_map = {}
        self.grid_num = np.array([nx, ny, nz])
        self.grid_size = np.array([dx, dy, dz])
        self.set_nodal_coords(nx, ny, nz, dx, dy, dz)
        self.set_node_connectivity(nx, ny, nz)
        self.generate_structured_grid(nx, ny, nz, dx, dy, dz)

    def set_nodal_coords(self, nx, ny, nz, dx, dy, dz):
        X = np.linspace(-self.ghost_cell * dx, (nx - self.ghost_cell) * dx, int(nx + 1))
        Y = np.linspace(-self.ghost_cell * dy, (ny - self.ghost_cell) * dy, int(ny + 1))
        Z = np.linspace(-self.ghost_cell * dz, (nz - self.ghost_cell) * dz, int(nz + 1))
        self.nodal_coords = flip3d_linear(np.array(list(product(X, Y, Z))), size_u=X.shape[0], size_v=Y.shape[0], size_w=Z.shape[0])

    def set_node_connectivity(self, nx, ny, nz):
        total_cell_number = nx * ny * nz
        self.node_connectivity = np.zeros((total_cell_number, 8))
        self.node_connectivity[:, 0] = np.arange(0, total_cell_number, 1)
        self.node_connectivity[:, 1] = np.arange(0, total_cell_number, 1) + 1
        self.node_connectivity[:, 2] = np.arange(0, total_cell_number, 1) + nx + 1
        self.node_connectivity[:, 3] = np.arange(0, total_cell_number, 1) + nx
        self.node_connectivity[:, 4] = np.arange(0, total_cell_number, 1) + nx * ny
        self.node_connectivity[:, 5] = np.arange(0, total_cell_number, 1) + nx * ny + 1
        self.node_connectivity[:, 6] = np.arange(0, total_cell_number, 1) + nx * ny + nx + 1
        self.node_connectivity[:, 7] = np.arange(0, total_cell_number, 1) + nx * ny + nx

    def generate_structured_grid(self, nx, ny, nz, dx, dy, dz):
        # === X direction faces ===
        i = np.arange(nx + 1)
        j = np.arange(ny)
        k = np.arange(nz)
        ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')

        x_centers = np.stack([
            ii.ravel() * dx,
            (jj.ravel() + 0.5) * dy,
            (kk.ravel() + 0.5) * dz
        ], axis=1)

        x_tags = np.where((ii == 0) | (ii == nx), "boundary", "internal").ravel()

        face2cell_x = []
        for xi, yj, zk in zip(ii.ravel(), jj.ravel(), kk.ravel()):
            cells = []
            if xi < nx:
                cells.append((xi, yj, zk))
            if xi > 0:
                cells.append((xi - 1, yj, zk))
            face2cell_x.append(cells)

        self.face_map['x'] = {
            "face_centers": x_centers,
            "face2cell": np.array(face2cell_x, dtype=object),
            "tags": x_tags
        }

        # === Y direction faces ===
        i = np.arange(nx)
        j = np.arange(ny + 1)
        k = np.arange(nz)
        ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')

        y_centers = np.stack([
            (ii.ravel() + 0.5) * dx,
            jj.ravel() * dy,
            (kk.ravel() + 0.5) * dz
        ], axis=1)

        y_tags = np.where((jj == 0) | (jj == ny), "boundary", "internal").ravel()

        face2cell_y = []
        for xi, yj, zk in zip(ii.ravel(), jj.ravel(), kk.ravel()):
            cells = []
            if yj < ny:
                cells.append((xi, yj, zk))
            if yj > 0:
                cells.append((xi, yj - 1, zk))
            face2cell_y.append(cells)

        self.face_map['y'] = {
            "face_centers": y_centers,
            "face2cell": np.array(face2cell_y, dtype=object),
            "tags": y_tags
        }

        # === Z direction faces ===
        i = np.arange(nx)
        j = np.arange(ny)
        k = np.arange(nz + 1)
        ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')

        z_centers = np.stack([
            (ii.ravel() + 0.5) * dx,
            (jj.ravel() + 0.5) * dy,
            kk.ravel() * dz
        ], axis=1)

        z_tags = np.where((kk == 0) | (kk == nz), "boundary", "internal").ravel()

        face2cell_z = []
        for xi, yj, zk in zip(ii.ravel(), jj.ravel(), kk.ravel()):
            cells = []
            if zk < nz:
                cells.append((xi, yj, zk))
            if zk > 0:
                cells.append((xi, yj, zk - 1))
            face2cell_z.append(cells)

        self.face_map['z'] = {
            "face_centers": z_centers,
            "face2cell": np.array(face2cell_z, dtype=object),
            "tags": z_tags
        }

    def write(self, filename='Element.txt'):
        print('#', "Writing cell(s) into 'Element.txt' ......")
        with open(filename, 'w') as f:
            f.write("# HexahedronMesh Export\n")
            f.write("\n[node_connectivity]\n")
            for row in self.node_connectivity:
                f.write(" ".join(map(str, map(int, row))) + "\n")

            for direction in ['x', 'y', 'z']:
                if direction not in self.face_map:
                    continue
                face_data = self.face_map[direction]
                f.write(f"\n[face_{direction}]\n")

                f.write("face_centers\n")
                for center in face_data['face_centers']:
                    f.write(" ".join(f"{x:.6f}" for x in center) + "\n")

                f.write("face2cell\n")
                for cells in face_data['face2cell']:
                    f.write(" ".join(f"{c[0]} {c[1]} {c[2]}" for c in cells) + "\n")

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
                direction = line[-2]  # 'x', 'y', or 'z'
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
                            cells = [tuple(tokens[j:j+3]) for j in range(0, len(tokens), 3)]
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