"""Closed contact solids registered to the visual bucket pin and truck bed."""

import itertools
from pathlib import Path

import bmesh
import bpy


def write_obj(name, directory, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    surface = bmesh.new()
    surface.from_mesh(mesh)
    bmesh.ops.triangulate(surface, faces=list(surface.faces))
    bmesh.ops.recalc_face_normals(surface, faces=list(surface.faces))
    assert all(edge.is_manifold and edge.is_contiguous for edge in surface.edges), name
    assert all(face.calc_area() > 1e-10 for face in surface.faces), name
    assert surface.calc_volume(signed=True) > 0, name
    surface.to_mesh(mesh)
    surface.free()
    bpy.ops.object.select_all(action="DESELECT")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    path = Path(directory) / f"{name}.obj"
    bpy.ops.wm.obj_export(
        filepath=str(path), export_selected_objects=True, forward_axis="Y", up_axis="Z", export_materials=False
    )
    exported = {tuple(map(float, line.split()[1:])) for line in path.read_text().splitlines() if line.startswith("v ")}
    assert exported == {tuple(round(val, 6) for val in v.co) for v in mesh.vertices}, name
    return obj


def bucket_mesh(directory):
    # Sections run from rear rim to the tooth envelope, in bucket-pin (y, z).
    outer = (
        (0.39, 0.35),
        (0.20, 0.15),
        (0.088, -0.13),
        (0.12, -0.35),
        (0.25, -0.55),
        (0.43, -0.72),
        (0.65, -0.789),
        (0.88, -0.772),
        (1.13, -0.65),
        (1.31, -0.48),
        (1.49, 0.065),
    )
    inner = (
        (0.44, 0.36),
        (0.26, 0.14),
        (0.15, -0.13),
        (0.18, -0.33),
        (0.30, -0.51),
        (0.46, -0.65),
        (0.66, -0.714),
        (0.87, -0.70),
        (1.08, -0.59),
        (1.25, -0.44),
        (1.43, 0.065),
    )
    n = len(outer)
    vertices = [
        (x, y, z) for x, section in ((-0.66, outer), (0.56, outer), (-0.59, inner), (0.49, inner)) for y, z in section
    ]
    faces = [tuple(range(start, start + n)) for start in (0, n, 2 * n, 3 * n)]
    for a, b in ((0, n), (2 * n, 3 * n)):
        faces.extend((a + i, a + i + 1, b + i + 1, b + i) for i in range(n - 1))
    for i in (0, n - 1):
        faces.append((i, n + i, 3 * n + i, 2 * n + i))
    faces.extend(((0, n - 1, 3 * n - 1, 2 * n), (n, 2 * n - 1, 4 * n - 1, 3 * n)))
    return write_obj("bucket_contact", directory, vertices, faces)


def bed_mesh(directory, truck_location):
    # Extract the boundary of the occupied cells, leaving the top of the bed open.
    axes = ((-1.43, -1.25, 1.25, 1.43), (-1.95, -1.66, 3.77, 4.06), (1.45, 1.62, 2.86))
    cells = set(itertools.product(range(3), range(3), range(2))) - {(1, 1, 1)}
    indices, vertices, faces = {}, [], []
    for cell in sorted(cells):
        for axis, step in itertools.product(range(3), (-1, 1)):
            neighbor = list(cell)
            neighbor[axis] += step
            if tuple(neighbor) in cells:
                continue
            face = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = list(cell)
                point[axis] += int(step > 0)
                point[(axis + 1) % 3] += u
                point[(axis + 2) % 3] += v
                key = tuple(point)
                if key not in indices:
                    indices[key] = len(vertices)
                    vertices.append(tuple(values[i] + offset for values, i, offset in zip(axes, point, truck_location)))
                face.append(indices[key])
            faces.append(tuple(face if step > 0 else reversed(face)))
    return write_obj("truck_bed_contact", directory, vertices, faces)
