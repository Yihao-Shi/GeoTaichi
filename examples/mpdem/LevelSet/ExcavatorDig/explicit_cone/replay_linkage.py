"""Fit the visible stick and bucket cylinder around the prescribed bucket sweep."""

from math import pi

import bmesh
import bpy
from mathutils import Vector


def fit_linkage():
    objects = bpy.data.objects
    stick, bucket = objects["stick"], objects["bucket"]
    shaft = objects["stick_mesh"]
    pin = bucket.location.copy()
    axis = -pin.normalized()
    across = Vector((1.0, 0.0, 0.0))
    normal = axis.cross(across)

    # The bucket sweeps 1.50 m above the pin during the carry phase.
    stem_end = pin + 1.57 * axis
    local = shaft.matrix_basis.inverted()
    mesh = bmesh.new()
    mesh.from_mesh(shaft.data)
    cut = bmesh.ops.bisect_plane(
        mesh,
        geom=list(mesh.verts) + list(mesh.edges) + list(mesh.faces),
        dist=1e-5,
        plane_co=local @ stem_end,
        plane_no=local.to_3x3() @ axis,
        clear_inner=True,
    )
    bmesh.ops.holes_fill(
        mesh, edges=[edge for edge in cut["geom_cut"] if isinstance(edge, bmesh.types.BMEdge) and edge.is_boundary]
    )
    mesh.to_mesh(shaft.data)
    mesh.free()
    shaft.data.update()

    paint = bpy.data.materials.new("Linkage yellow")
    shader = paint.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.65, 0.32, 0.037, 1.0)
    shader.inputs["Roughness"].default_value = 0.52
    steel = bpy.data.materials.new("Hinge steel")
    shader = steel.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.18, 0.19, 0.2, 1.0)
    shader.inputs["Metallic"].default_value = 0.7

    forks = []
    for sign in (-1, 1):
        vertices = []
        # Widen the two cheeks beyond the bucket side plates before the swept range.
        for distance, offset, half_width, half_height in (
            (1.83, 0.12, 0.13, 0.22),
            (1.58, 0.74, 0.085, 0.18),
            (0.08, 0.74, 0.085, 0.15),
            (-0.125, 0.74, 0.075, 0.115),
        ):
            center = pin + distance * axis + (-0.05 + sign * offset) * across
            vertices.extend(
                center + i * half_width * across + j * half_height * normal
                for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            )
        faces = [(0, 1, 2, 3), (15, 14, 13, 12)]
        for base in (0, 4, 8):
            for i in range(4):
                j = (i + 1) % 4
                faces.append((base + i, base + 4 + i, base + 4 + j, base + j))
        data = bpy.data.meshes.new("Stick fork")
        data.from_pydata(vertices, [], faces)
        fork = bpy.data.objects.new("Stick fork", data)
        stick.users_collection[0].objects.link(fork)
        fork.parent = stick
        fork.data.materials.append(paint)
        bevel = fork.modifiers.new("Rounded plate edges", "BEVEL")
        bevel.width, bevel.segments = 0.035, 3
        forks.append(fork)

    def add_pin(parent, center, length, radius, name):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=length)
        obj = bpy.context.object
        obj.name, obj.parent, obj.location = name, parent, center
        obj.rotation_euler.y = pi / 2
        obj.data.materials.append(steel)

    add_pin(stick, pin - 0.05 * across, 1.69, 0.094, "Bucket hinge pin")
    barrel, rod = objects["bucket_ram_barrel"], objects["bucket_ram_rod"]
    # Offset both cylinder mounts along the common hinge axis.
    add_pin(stick, barrel.location + 0.43 * across, 0.97, 0.092, "Cylinder base pin")
    add_pin(bucket, rod.location + 0.43 * across, 0.97, 0.072, "Cylinder rod pin")
    barrel.location.x += 0.86
    rod.location.x += 0.86
    return shaft, *forks
