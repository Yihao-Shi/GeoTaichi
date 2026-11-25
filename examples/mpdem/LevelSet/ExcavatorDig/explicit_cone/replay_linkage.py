"""Pose a single stick, pin coupler and bucket four-bar for replay."""

from math import acos, cos, pi, sin, tau

import bmesh
import bpy
from mathutils import Matrix, Vector


def fit_linkage(trench=False):
    objects = bpy.data.objects
    boom, stick, bucket = (objects[name] for name in ("boom", "stick", "bucket"))
    shaft = objects["stick_mesh"]
    stick_pin, boom_end = bucket.location.copy(), stick.location.copy()
    up = -stick_pin.normalized()
    across = Vector((1, 0, 0))
    normal = across.cross(up)
    hinge = Vector((0, -0.009, 0.66))
    center = -0.05 * across

    # Keep the original textured box section and trim its skirt at the new pin.
    local = shaft.matrix_basis.inverted()
    mesh = bmesh.new()
    mesh.from_mesh(shaft.data)
    cut = bmesh.ops.bisect_plane(
        mesh,
        geom=list(mesh.verts) + list(mesh.edges) + list(mesh.faces),
        dist=1e-5,
        plane_co=local @ (stick_pin - 0.11 * up),
        plane_no=local.to_3x3() @ up,
        clear_inner=True,
    )
    bmesh.ops.holes_fill(
        mesh, edges=[e for e in cut["geom_cut"] if isinstance(e, bmesh.types.BMEdge) and e.is_boundary]
    )
    mesh.to_mesh(shaft.data)
    mesh.free()
    shaft.data.update()

    def material(name, color, metallic, roughness):
        mat = bpy.data.materials.new(name)
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Metallic"].default_value = metallic
        shader.inputs["Roughness"].default_value = roughness
        return mat

    paint = material("Equipment ochre enamel", (0.49, 0.248, 0.033), 0.13, 0.43)
    steel = material("Oxidized coupler steel", (0.082, 0.075, 0.064), 0.62, 0.53)
    pin_steel = material("Machined pin steel", (0.22, 0.21, 0.19), 0.85, 0.34)
    chrome = objects["bucket_ram_rod_fittings"].data.materials[0]

    def pin(name, parent, location, radius, width, mat=pin_steel):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=width)
        obj = bpy.context.object
        obj.name, obj.parent, obj.location = name, parent, location
        obj.rotation_euler.y = pi / 2
        obj.data.materials.append(mat)
        bevel = obj.modifiers.new("Machined edge", "BEVEL")
        bevel.width, bevel.segments = 0.008, 2
        return obj

    def plate(name, parent, a, b, radius, width, mat):
        length = (b - a).length
        outline = [
            (x + radius * cos(i * tau / 24), radius * sin(i * tau / 24))
            for x, start in ((length, -6), (0, 6))
            for i in range(start, start + 13)
        ]
        n = len(outline)
        vertices = [(x, y, z) for z in (-width / 2, width / 2) for x, y in outline]
        faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
        faces.extend((i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n))
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new(name, mesh)
        stick.users_collection[0].objects.link(obj)
        obj.parent, obj.location = parent, a
        if parent is not None:
            tangent = (b - a).normalized()
            obj.rotation_euler = Matrix((tangent, across.cross(tangent), across)).transposed().to_euler()
        obj.data.materials.append(mat)
        bevel = obj.modifiers.new("Rounded plate edges", "BEVEL")
        bevel.width, bevel.segments = 0.012, 3
        return obj

    drive = 0.541 * up + 0.331 * normal
    anchor = 2.60 * up + 0.40 * normal
    lug = hinge + Vector((0, -0.346, -0.219))
    coupler = [
        plate("Pin coupler cheek", bucket, center + side * across, hinge + center + side * across, 0.112, 0.065, steel)
        for side in (-0.29, 0.29)
    ]
    if trench:
        # Leave the center open where the stick passes between the coupler lugs.
        coupler.extend(
            plate(
                "Coupler back plate",
                bucket,
                center + Vector((side, -0.108, 0.17)),
                lug + center + side * across,
                0.112,
                0.065,
                steel,
            )
            for side in (-0.29, 0.29)
        )
    else:
        coupler.append(
            plate("Coupler back plate", bucket, center + Vector((0, -0.108, 0.17)), lug + center, 0.112, 0.62, steel)
        )
    pin("Bucket adapter pin", bucket, center, 0.091, 0.65)
    coupler_pin = pin("Stick tip pin", bucket, hinge + center, 0.101, 0.65)
    coupler.append(pin("Bucket link pin", bucket, lug + center, 0.079, 0.79))
    stick_drive = stick_pin + drive + center
    pin("Rocker pivot", stick, stick_drive, 0.081, 0.41)
    plate("Welded rocker mount", stick, stick_pin + 0.59 * up + center, stick_drive, 0.103, 0.31, paint)
    pin("Cylinder head pin", stick, stick_pin + anchor + center, 0.094, 0.34)
    rocker = [plate("Bucket rocker", None, Vector(), Vector((0.621, 0, 0)), 0.093, 0.052, paint) for _ in range(2)]
    link = [plate("Bucket power link", None, Vector(), Vector((0.830, 0, 0)), 0.098, 0.056, steel) for _ in range(2)]
    apex_pin = pin("Rocker cross pin", None, Vector(), 0.08, 0.79)
    for name in ("bucket_ram_barrel", "bucket_ram_rod"):
        for obj in objects[name].children:
            obj.hide_render = True

    def tube(name, radius, length, mat):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=length)
        obj = bpy.context.object
        obj.name = name
        obj.data.materials.append(mat)
        bevel = obj.modifiers.new("Turned edge", "BEVEL")
        bevel.width, bevel.segments = 0.012, 3
        for face in obj.data.polygons:
            face.use_smooth = len(face.vertices) == 4
        return obj

    eye = tube("Bucket cylinder clevis", 0.066, 0.19, steel)
    barrel = tube("Bucket cylinder barrel", 0.105, 1.29, paint)
    gland = tube("Bucket cylinder gland", 0.115, 0.085, steel)
    rod = tube("Bucket cylinder piston", 0.064, 1.33, chrome)
    for joint in (boom, stick):
        joint.animation_data_clear()

    def pose(world):
        axis = world.to_3x3() @ across
        base = boom.matrix_world.translation
        tip = world @ hinge
        reach = tip - base
        distance = reach.length
        l1, l2 = boom_end.length, stick_pin.length
        cosine = (l1 * l1 + distance * distance - l2 * l2) / (2 * l1 * distance)
        assert -1 < cosine < 1, ("Stick reach", distance)
        direction = reach / distance
        angle = acos(cosine)
        elbow = base + l1 * (cos(angle) * direction - sin(angle) * axis.cross(direction))
        for joint, a, b, rest in ((boom, base, elbow, boom_end), (stick, elbow, tip, stick_pin)):
            yaw = across.rotation_difference(axis)
            rotation = (yaw @ rest).rotation_difference(b - a) @ yaw
            transform = rotation.to_matrix().to_4x4()
            transform.translation = a
            joint.matrix_world = transform
            bpy.context.view_layer.update()
        bucket.matrix_world = world
        bpy.context.view_layer.update()
        mount = stick.matrix_world @ (stick_pin + anchor + center)
        d = stick.matrix_world @ stick_drive
        b = world @ (lug + center)
        span = b - d
        length = span.length
        projected = (0.621**2 - 0.830**2 + length**2) / (2 * length)
        assert abs(projected) < 0.621, ("Bucket four-bar reach", length)
        direction = span / length
        c = d + projected * direction - (0.621**2 - projected**2) ** 0.5 * axis.cross(direction)

        def align(parts, start, end, offset):
            tangent = (end - start).normalized()
            transform = Matrix((tangent, axis.cross(tangent), axis)).transposed().to_4x4()
            for obj, side in zip(parts, (-offset, offset)):
                transform.translation = start + side * axis
                obj.matrix_world = transform

        align(rocker, d, c, 0.18)
        align(link, b, c, 0.36)
        apex_pin.location = c
        apex_pin.rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
        length = (c - mount).length
        assert 1.49 < length < 2.78, ("Bucket cylinder stroke", length)
        direction = (c - mount) / length
        rotation = direction.to_track_quat("Z", "Y").to_euler()
        for obj, location in (
            (eye, mount + 0.095 * direction),
            (barrel, mount + 0.825 * direction),
            (gland, mount + 1.47 * direction),
            (rod, c - 0.665 * direction),
        ):
            obj.location, obj.rotation_euler = location, rotation

    moving = (shaft, coupler_pin, *rocker, *link, apex_pin, eye, barrel, gland, rod)
    clearance_pairs = [(shaft, part) for part in (*coupler[:-1], *rocker, *link, barrel, rod)]
    clearance_pairs.extend((cheek, part) for cheek in coupler[:2] for part in (*rocker, *link))
    return pose, moving, coupler, clearance_pairs
