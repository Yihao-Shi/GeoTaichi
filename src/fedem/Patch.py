"""Device-resident deforming triangular FEM contact surface."""

import numpy as np
import taichi as ti


def _modifier_value(mapping, name, default=None):
    if mapping is None:
        return default
    normalized = {str(key).replace("_", "").lower(): value for key, value in mapping.items()}
    return normalized.get(name.replace("_", "").lower(), default)


@ti.data_oriented
class FEMSurfacePatch:
    def __init__(self, node_count, faces, face_body, modifier=None):
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        face_body = np.ascontiguousarray(face_body, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("FEDEM surface facets must have shape (n, 3)")
        if face_body.shape != (faces.shape[0],):
            raise ValueError("one FEM body id is required for every surface facet")
        self.node_count = int(node_count)
        self.face_count = int(faces.shape[0])
        node_body = np.full(self.node_count, -1, dtype=np.int32)
        for face, body_id in zip(faces, face_body):
            for node in face:
                previous = int(node_body[node])
                if previous >= 0 and previous != int(body_id):
                    raise ValueError("a FEDEM surface node cannot belong to different FEM bodies")
                node_body[node] = int(body_id)
        surface_vertices = np.unique(faces).astype(np.int32)
        self.surface_vertex_count = int(surface_vertices.size)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=max(self.face_count, 1))
        self.face_body = ti.field(dtype=ti.i32, shape=max(self.face_count, 1))
        self.node_body = ti.field(dtype=ti.i32, shape=self.node_count)
        self.surface_vertices = ti.field(dtype=ti.i32, shape=max(self.surface_vertex_count, 1))
        self.nodes = None
        self.offset_nodes = None
        self.search_positions = ti.Vector.field(3, dtype=float, shape=max(self.surface_vertex_count, 1))
        self.normals = ti.Vector.field(3, dtype=float, shape=max(self.face_count, 1))
        self.mean_normals = ti.Vector.field(3, dtype=float, shape=self.node_count)
        self.node_area = ti.field(dtype=float, shape=self.node_count)
        self.maximum_displacement = ti.field(dtype=float, shape=())
        self.minimum_bounding_radius = ti.field(dtype=float, shape=())
        self.maximum_bounding_radius = ti.field(dtype=float, shape=())
        self.degenerate_face = ti.field(dtype=ti.i32, shape=())
        padded_faces = np.zeros((max(self.face_count, 1), 3), dtype=np.int32)
        padded_body = np.zeros(max(self.face_count, 1), dtype=np.int32)
        padded_faces[: self.face_count] = faces
        padded_body[: self.face_count] = face_body
        self.faces.from_numpy(padded_faces)
        self.face_body.from_numpy(padded_body)
        self.node_body.from_numpy(node_body)
        padded_vertices = np.zeros(max(self.surface_vertex_count, 1), dtype=np.int32)
        padded_vertices[: self.surface_vertex_count] = surface_vertices
        self.surface_vertices.from_numpy(padded_vertices)

        orientation = str(_modifier_value(modifier, "Orientation", "None"))
        normalized = orientation.replace("_", "").replace("-", "").lower()
        self.orientation = {
            "none": 0,
            "parallel": 1,
            "centrioetal": 2,
            "centripetal": 2,
            "inverse": 3,
        }.get(normalized, -1)
        if self.orientation < 0:
            raise ValueError("FEDEM patch Orientation must be Parallel, Centripetal/Centrioetal, Inverse, or None")
        direction = _modifier_value(modifier, "Direction", (0.0, 0.0, 1.0))
        center = _modifier_value(modifier, "Center", (0.0, 0.0, 0.0))
        self.direction = tuple(float(value) for value in np.asarray(direction).reshape(3))
        self.center = tuple(float(value) for value in np.asarray(center).reshape(3))
        self.offset = float(_modifier_value(modifier, "Value", 0.0))
        if abs(self.offset) > 0.0:
            self.offset_nodes = ti.Vector.field(3, dtype=float, shape=self.node_count)
            self.nodes = self.offset_nodes
            self.update_contact_nodes = self._update_offset_nodes
        else:
            self.update_contact_nodes = self._bind_authoritative_nodes

    def _bind_authoritative_nodes(self, source):
        self.nodes = source

    def _update_offset_nodes(self, source):
        self._copy_offset_nodes(source)

    @ti.kernel
    def _copy_offset_nodes(self, source: ti.template()):
        for node in range(self.node_count):
            self.offset_nodes[node] = source[node]

    @ti.kernel
    def _measure_displacement(self):
        self.maximum_displacement[None] = 0.0
        for local in range(self.surface_vertex_count):
            node = self.surface_vertices[local]
            ti.atomic_max(
                self.maximum_displacement[None],
                (self.nodes[node] - self.search_positions[local]).norm(),
            )

    @ti.kernel
    def _compute_normals(self):
        for face in range(self.face_count):
            ids = self.faces[face]
            cross = (self.nodes[ids[1]] - self.nodes[ids[0]]).cross(self.nodes[ids[2]] - self.nodes[ids[0]])
            length = cross.norm()
            self.normals[face] = cross / length if length > 1.0e-30 else ti.Vector.zero(float, 3)

    @ti.kernel
    def _orient_normals(
        self,
        orientation: ti.i32,
        direction: ti.types.vector(3, float),
        center: ti.types.vector(3, float),
    ):
        for face in range(self.face_count):
            normal = self.normals[face]
            if orientation == 1 and normal.dot(direction) < 0.0:
                normal = -normal
            elif orientation == 2:
                ids = self.faces[face]
                centroid = (self.nodes[ids[0]] + self.nodes[ids[1]] + self.nodes[ids[2]]) / 3.0
                if normal.dot(centroid - center) > 0.0:
                    normal = -normal
                    temporary = self.faces[face][1]
                    self.faces[face][1] = self.faces[face][2]
                    self.faces[face][2] = temporary
            elif orientation == 3:
                normal = -normal
            self.normals[face] = normal

    @ti.kernel
    def _compute_mean_normals(self):
        for node in range(self.node_count):
            self.mean_normals[node] = ti.Vector.zero(float, 3)
        for face in range(self.face_count):
            ids = self.faces[face]
            a, b, c = self.nodes[ids[0]], self.nodes[ids[1]], self.nodes[ids[2]]
            for local in ti.static(range(3)):
                first = b - a
                second = c - a
                if local == 1:
                    first, second = c - b, a - b
                elif local == 2:
                    first, second = a - c, b - c
                denominator = first.norm() * second.norm()
                angle = 0.0
                if denominator > 1.0e-30:
                    cosine = ti.max(-1.0, ti.min(1.0, first.dot(second) / denominator))
                    angle = ti.acos(cosine)
                ti.atomic_add(self.mean_normals[ids[local]], angle * self.normals[face])
        for node in range(self.node_count):
            length = self.mean_normals[node].norm()
            if length > 1.0e-30:
                self.mean_normals[node] /= length

    @ti.kernel
    def _compute_node_area(self):
        for node in range(self.node_count):
            self.node_area[node] = 0.0
        for face in range(self.face_count):
            ids = self.faces[face]
            area = 0.5 * (self.nodes[ids[1]] - self.nodes[ids[0]]).cross(self.nodes[ids[2]] - self.nodes[ids[0]]).norm()
            for local in ti.static(range(3)):
                ti.atomic_add(self.node_area[ids[local]], area / 3.0)

    @ti.kernel
    def _offset_nodes(self, value: float):
        for node in range(self.node_count):
            self.nodes[node] += value * self.mean_normals[node]

    @ti.kernel
    def _reduce_bounding_radii(self):
        self.minimum_bounding_radius[None] = 1.0e30
        self.maximum_bounding_radius[None] = 0.0
        self.degenerate_face[None] = 0
        for face in range(self.face_count):
            ids = self.faces[face]
            edge_1 = self.nodes[ids[1]] - self.nodes[ids[0]]
            edge_2 = self.nodes[ids[2]] - self.nodes[ids[0]]
            edge_3 = self.nodes[ids[1]] - self.nodes[ids[2]]
            area = 0.5 * edge_1.cross(edge_2).norm()
            if area <= 1.0e-30:
                self.degenerate_face[None] = 1
            else:
                radius = 0.25 * ti.sqrt(edge_1.norm_sqr() * edge_2.norm_sqr() * edge_3.norm_sqr()) / area
                ti.atomic_min(self.minimum_bounding_radius[None], radius)
                ti.atomic_max(self.maximum_bounding_radius[None], radius)

    @ti.kernel
    def commit_search_positions(self):
        for local in range(self.surface_vertex_count):
            node = self.surface_vertices[local]
            self.search_positions[local] = self.nodes[node]
        self.maximum_displacement[None] = 0.0

    def update(
        self,
        fem_positions,
        update_normals=True,
        update_area=True,
        measure_displacement=True,
    ):
        self.update_contact_nodes(fem_positions)
        needs_normals = bool(update_normals) or abs(self.offset) > 0.0
        if needs_normals:
            self._compute_normals()
            self._orient_normals(self.orientation, self.direction, self.center)
        if abs(self.offset) > 0.0:
            self._compute_mean_normals()
            self._offset_nodes(self.offset)
            self._compute_normals()
            self._orient_normals(self.orientation, self.direction, self.center)
        if update_area:
            self._compute_node_area()
        if measure_displacement:
            self._measure_displacement()

    def bounding_radii(self):
        self._reduce_bounding_radii()
        if int(self.degenerate_face[None]) != 0:
            raise ValueError("FEDEM contact surface contains a degenerate triangle")
        return (
            float(self.minimum_bounding_radius[None]),
            float(self.maximum_bounding_radius[None]),
        )


__all__ = ["FEMSurfacePatch"]
