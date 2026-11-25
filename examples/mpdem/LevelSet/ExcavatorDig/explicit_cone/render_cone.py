"""Render the solver's world-space point snapshots in the excavator scene."""

import argparse
import csv
import json
from math import floor
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_linkage import fit_linkage


def fit_bucket(bucket, shell):
    contact = bpy.data.objects["bucket_contact"]
    to_pin = bucket.matrix_world.inverted() @ contact.matrix_world
    profile = [to_pin @ vertex.co for vertex in contact.data.vertices]
    # shortcut: these two scenes share the same bowl profile. Use a named lip edge for other assets.
    front = max(point.y for point in profile)
    rim = sorted((point for point in profile if abs(point.y - front) < 1e-5), key=lambda point: point.x)
    assert len(rim) == 2 and rim[1].x > rim[0].x and abs(rim[1].z - rim[0].z) < 1e-5
    edge = rim[0]
    lower = max(
        (point for point in profile if abs(point.x - edge.x) < 1e-5 and point.y < front - 1e-5),
        key=lambda point: point.y,
    )
    forward = (edge - lower).normalized()
    outward = forward.cross(Vector((1, 0, 0)))

    material = bpy.data.materials.new("Bucket steel")
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.12, 0.104, 0.084, 1)
    shader.inputs["Metallic"].default_value = 0.4
    shader.inputs["Roughness"].default_value = 0.58
    shell.data = contact.data.copy()
    shell.data.transform(shell.matrix_world.inverted() @ contact.matrix_world)
    shell.data.materials.clear()
    shell.data.materials.append(material)
    assert all(
        (shell.matrix_world @ visual.co - contact.matrix_world @ source.co).length < 1e-5
        for visual, source in zip(shell.data.vertices, contact.data.vertices, strict=True)
    )
    for face in shell.data.polygons:
        face.use_smooth = True
    shell.data.set_sharp_from_angle(angle=0.5)

    # Seat the adapters in the outside of the lip and leave 4 cm of tooth ahead of it.
    vertices, faces = [], []
    sections = ((-0.10, 0.094, -0.008, 0.043), (-0.025, 0.080, -0.008, 0.043), (0.04, 0.042, 0.003, 0.017))
    for position in (0.13, 0.315, 0.5, 0.685, 0.87):
        seat = rim[0].lerp(rim[1], position)
        start = len(vertices)
        for length, width, low, high in sections:
            for side, offset in ((-1, low), (1, low), (1, high), (-1, high)):
                vertices.append(seat + length * forward + offset * outward + Vector((side * width / 2, 0, 0)))
        faces.extend(((start + 3, start + 2, start + 1, start), (start + 8, start + 9, start + 10, start + 11)))
        for ring in (start, start + 4):
            for j in range(4):
                k = (j + 1) % 4
                faces.append((ring + j, ring + k, ring + 4 + k, ring + 4 + j))
    mesh = bpy.data.meshes.new("Short bucket teeth")
    mesh.from_pydata(vertices, [], faces)
    tooth_steel = material.copy()
    tooth_steel.name = "Bucket tooth steel"
    shader = tooth_steel.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.17, 0.149, 0.117, 1)
    mesh.materials.append(tooth_steel)
    teeth = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(teeth)
    teeth.parent = bucket
    bevel = teeth.modifiers.new("Tooth edge bevel", "BEVEL")
    bevel.width, bevel.segments = 0.003, 2
    return teeth


def fit_truck_bed():
    """Skin the actual contact bed so its walls occlude the same cargo as the VTUs."""
    bed = bpy.data.objects["bed_mesh"]
    contact = bpy.data.objects["truck_bed_contact"]
    material = bpy.data.materials.new("Truck bed enamel")
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.51, 0.27, 0.047, 1)
    shader.inputs["Roughness"].default_value = 0.55
    shader.inputs["Metallic"].default_value = 0.2
    lining = material.copy()
    lining.name = "Scuffed bed liner"
    shader = lining.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.16, 0.127, 0.091, 1)
    shader.inputs["Roughness"].default_value = 0.68
    shader.inputs["Metallic"].default_value = 0.48

    bed.data = contact.data.copy()
    bed.data.transform(bed.matrix_world.inverted() @ contact.matrix_world)
    bed.data.materials.clear()
    bed.data.materials.append(material)
    bed.data.materials.append(lining)
    xyz = [bed.matrix_world @ v.co for v in bed.data.vertices]
    xmin, ymin, zmin = (min(p[i] for p in xyz) for i in range(3))
    xmax, ymax, zmax = (max(p[i] for p in xyz) for i in range(3))
    for face in bed.data.polygons:
        center = bed.matrix_world @ face.center
        if (
            xmin + 0.01 < center.x < xmax - 0.01
            and ymin + 0.01 < center.y < ymax - 0.01
            and zmin + 0.01 < center.z < zmax - 0.01
        ):
            face.material_index = 1
    bevel = bed.modifiers.new("Welded edges", "BEVEL")
    bevel.width, bevel.segments = 0.014 / bed.scale.x, 2

    # Form the outer stiffeners without changing the welded liner or the load coordinates.
    for x in (xmin, xmax):
        for fraction in (0.06, 0.27, 0.49, 0.71, 0.94):
            bpy.ops.mesh.primitive_cube_add(size=1)
            rib = bpy.context.object
            rib.name = "Truck bed side stiffener"
            rib.dimensions = (0.075, 0.084, zmax - zmin - 0.15)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            rib.parent = bed.parent
            rib.matrix_parent_inverse = bed.parent.matrix_world.inverted()
            rib.location = (x, ymin + fraction * (ymax - ymin), (zmin + zmax) / 2)
            rib.data.materials.append(material)
            bevel = rib.modifiers.new("Pressed edges", "BEVEL")
            bevel.width, bevel.segments = 0.016, 2


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("trajectory", type=Path)
parser.add_argument("--view", choices=("right-front", "left-front"), default="right-front")
parser.add_argument("--textures", type=Path, help="Poly Haven Quarry 02 and Gravel Ground 01 texture directory")
parser.add_argument("--frames", type=int, nargs="+", help="Render these snapshot frame numbers")
parser.add_argument("--check-linkage", action="store_true", help="Check each replay pose without rendering")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
run, output, trajectory = args.run, args.output, args.trajectory
config = json.loads((run / "run-config.json").read_text())
dt = config["dt_s"]
steps_per_frame = round(config["snapshot_interval_s"] / dt)
count = config["particles"]
with trajectory.open(newline="") as stream:
    motion = [
        (
            Vector([float(row[f"pin_{axis}_m"]) for axis in "xyz"]),
            Quaternion([float(row[f"q{axis}"]) for axis in "wxyz"]).normalized(),
        )
        for row in csv.DictReader(stream)
    ]

scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 1920, 1440
scene.render.resolution_percentage = 100
scene.render.use_persistent_data = True
scene.cycles.samples = 16
scene.cycles.adaptive_threshold = 0.08
if sys.platform == "darwin":
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.refresh_devices()
    for device in preferences.devices:
        device.use = device.type == "METAL"
    if any(device.use for device in preferences.devices):
        scene.cycles.device = "GPU"
bpy.data.collections["Preview soil"].hide_render = True
camera = scene.camera
target = Vector((2.1, -3.1, 2.4))
camera.location = (13 if args.view == "right-front" else 2 * target.x - 13, -19, 17)
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.ortho_scale = 24.5
for obj in list(scene.camera.children):
    if obj.type == "FONT":
        bpy.data.objects.remove(obj, do_unlink=True)


# Use photographic site light and gravel without adding geometry to the simulation.
for obj in scene.objects:
    if obj.type == "LIGHT":
        obj.hide_render = True
world = scene.world.node_tree
background = world.nodes.get("Background")
background.inputs["Strength"].default_value = 0.65
sun_data = bpy.data.lights.new("Afternoon sun", "SUN")
sun_data.energy, sun_data.angle = 2.2, 0.075
sun = bpy.data.objects.new(sun_data.name, sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = Vector((0.52, 0.28, -1)).to_track_quat("-Z", "Y").to_euler()

if args.textures:
    environment = world.nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(args.textures / "quarry_02_1k.hdr"))
    world.links.new(environment.outputs["Color"], background.inputs["Color"])
    gravel = bpy.data.materials["Gravel yard"].node_tree
    shader = gravel.nodes.get("Principled BSDF")
    coordinates = gravel.nodes.new("ShaderNodeTexCoord")
    mapping = gravel.nodes.new("ShaderNodeVectorMath")
    mapping.operation = "SCALE"
    mapping.inputs["Scale"].default_value = 0.44
    gravel.links.new(coordinates.outputs["Object"], mapping.inputs[0])
    for name, channel in (("diff", "Base Color"), ("rough", "Roughness"), ("nor_gl", "Normal")):
        texture = gravel.nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(args.textures / f"gravel_ground_01_{name}_1k.jpg"))
        gravel.links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        color = texture.outputs["Color"]
        if name != "diff":
            texture.image.colorspace_settings.name = "Non-Color"
        if name == "nor_gl":
            normal = gravel.nodes.new("ShaderNodeNormalMap")
            normal.inputs["Strength"].default_value = 0.48
            gravel.links.new(color, normal.inputs["Color"])
            color = normal.outputs["Normal"]
        if name == "diff":
            # Break up repetition over the yard with a second scale and a broad dirt pattern.
            rotation = gravel.nodes.new("ShaderNodeVectorRotate")
            rotation.inputs["Angle"].default_value = 1.12
            gravel.links.new(mapping.outputs["Vector"], rotation.inputs["Vector"])
            second = gravel.nodes.new("ShaderNodeTexImage")
            second.image = texture.image
            gravel.links.new(rotation.outputs["Vector"], second.inputs["Vector"])
            noise = gravel.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 0.33
            gravel.links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
            mix = gravel.nodes.new("ShaderNodeMixRGB")
            gravel.links.new(noise.outputs["Fac"], mix.inputs["Fac"])
            gravel.links.new(color, mix.inputs[1])
            gravel.links.new(second.outputs["Color"], mix.inputs[2])
            dirt = gravel.nodes.new("ShaderNodeMixRGB")
            dirt.blend_type = "MULTIPLY"
            dirt.inputs[2].default_value = (0.63, 0.57, 0.47, 1)
            gravel.links.new(noise.outputs["Fac"], dirt.inputs["Fac"])
            gravel.links.new(mix.outputs[0], dirt.inputs[1])
            color = dirt.outputs[0]
        gravel.links.new(color, shader.inputs[channel])

files = sorted(path for path in (run / "snapshots").glob("*.npy") if path.stem.isdecimal())
assert files, run / "snapshots"
initial = np.load(files[0], allow_pickle=False)
assert initial.shape == (count, 3)
rng = np.random.default_rng(94)
mesh = bpy.data.meshes.new("Sand grains")
mesh.vertices.add(count)
for name, low, high in (("grain_radius", 0.021, 0.030), ("sand_tone", 0.0, 1.0)):
    attribute = mesh.attributes.new(name, "FLOAT", "POINT")
    attribute.data.foreach_set("value", rng.uniform(low, high, count).astype(np.float32))
soil = bpy.data.objects.new("Sand grains following MPM points", mesh)
scene.collection.objects.link(soil)

sand = bpy.data.materials.new("Dry matte sand")
sand.use_nodes = True
shader = sand.node_tree.nodes.get("Principled BSDF")
shader.inputs["Roughness"].default_value = 0.94
shader.inputs["Specular IOR Level"].default_value = 0.12
tone = sand.node_tree.nodes.new("ShaderNodeAttribute")
tone.attribute_name = "sand_tone"
ramp = sand.node_tree.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.31, 0.235, 0.14, 1.0)
ramp.color_ramp.elements[1].color = (0.46, 0.37, 0.24, 1.0)
sand.node_tree.links.new(tone.outputs["Fac"], ramp.inputs["Fac"])
sand.node_tree.links.new(ramp.outputs["Color"], shader.inputs["Base Color"])

tree = bpy.data.node_groups.new("Render dry sand", "GeometryNodeTree")
tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
source = tree.nodes.new("NodeGroupInput")
target = tree.nodes.new("NodeGroupOutput")
points = tree.nodes.new("GeometryNodeMeshToPoints")
radius = tree.nodes.new("GeometryNodeInputNamedAttribute")
radius.data_type = "FLOAT"
radius.inputs["Name"].default_value = "grain_radius"
tree.links.new(radius.outputs["Attribute"], points.inputs["Radius"])
material = tree.nodes.new("GeometryNodeSetMaterial")
material.inputs["Material"].default_value = sand
tree.links.new(source.outputs["Geometry"], points.inputs["Mesh"])
tree.links.new(points.outputs["Points"], material.inputs["Geometry"])
tree.links.new(material.outputs["Geometry"], target.inputs["Geometry"])
soil.modifiers.new("Display sand", "NODES").node_group = tree

bucket = bpy.data.objects["bucket"]
bucket.animation_data_clear()
pose_linkage, linkage, coupler, clearance_pairs = fit_linkage(trench="trench" in config)
ram_pairs = []
for name in ("boom_ram_L", "boom_ram_R", "stick_ram"):
    pair = []
    for part, suffix in (("barrel", "mesh"), ("rod", "fittings")):
        pivot = bpy.data.objects[f"{name}_{part}"]
        pivot.animation_data_clear()
        child = bpy.data.objects[f"{name}_{part}_{suffix}"]
        local_center = sum((Vector(corner) for corner in child.bound_box), Vector()) / 8
        pair.append((pivot, child.matrix_local @ local_center))
    ram_pairs.append(pair)


def bvh(obj, graph):
    obj = obj.evaluated_get(graph)
    xyz = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    return BVHTree.FromPolygons(xyz, [poly.vertices[:] for poly in obj.data.polygons])


bucket_mesh = bpy.data.objects["bucket_mesh"]
bpy.context.view_layer.update()
bucket_teeth = fit_bucket(bucket, bucket_mesh)
fit_truck_bed()
bpy.context.view_layer.update()
boom_mesh = bpy.data.objects["boom_mesh"]
truck_meshes = [
    obj for obj in bpy.data.objects["dump-truck"].children_recursive if obj.type == "MESH" and not obj.hide_render
]
graph = bpy.context.evaluated_depsgraph_get()
truck = {obj: bvh(obj, graph) for obj in truck_meshes} if args.check_linkage else {}
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
output.mkdir(parents=True, exist_ok=True)
for path in files:
    step = int(path.stem)
    index = round(step / steps_per_frame)
    if args.frames and index not in args.frames:
        continue
    image = output / f"{index:04d}.png"
    if image.exists() and not args.check_linkage:
        continue
    t = step * dt
    if not args.check_linkage:
        positions = np.load(path, allow_pickle=False)
        assert positions.shape == initial.shape and np.isfinite(positions).all(), path
        mesh.vertices.foreach_set("co", positions.ravel())
        mesh.update()
    release_t = config.get("release_time_s", 0.0)
    cycle_t = 0.0 if release_t is None else max(t - release_t, 0.0)
    frame = min(24 * cycle_t / config["slowdown"], len(motion) - 1)
    base = min(floor(frame), len(motion) - 2)
    alpha = frame - base
    scene.frame_set(base + 1, subframe=alpha)
    pin, q = motion[base]
    next_pin, next_q = motion[base + 1]
    world = q.slerp(next_q, alpha).to_matrix().to_4x4()
    world.translation = pin.lerp(next_pin, alpha)
    pose_linkage(world)
    bpy.context.view_layer.update()
    for (barrel, barrel_axis), (rod, rod_axis) in ram_pairs:
        elbow = barrel.matrix_world.translation
        tip = rod.matrix_world.translation
        axis = tip - elbow
        for pivot, local_axis, location, direction in ((barrel, barrel_axis, elbow, axis), (rod, rod_axis, tip, -axis)):
            transform = local_axis.rotation_difference(direction).to_matrix().to_4x4()
            transform.translation = location
            pivot.matrix_world = transform
    if args.check_linkage:
        bpy.context.view_layer.update()
        bucket_parts = (bucket_mesh, bucket_teeth)
        parts = {obj: bvh(obj, graph) for obj in (*linkage, *coupler, *bucket_parts, boom_mesh)}
        bucket_pairs = tuple((shell, part) for shell in bucket_parts for part in linkage)
        for a, b in (*clearance_pairs, *bucket_pairs):
            assert not parts[a].overlap(parts[b]), (f"{t:.2f} s", a.name, b.name)
        for a, tree in parts.items():
            for b, truck_tree in truck.items():
                assert not tree.overlap(truck_tree), (f"{t:.2f} s", a.name, b.name)
        continue
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)

if args.check_linkage:
    print(
        f"Checked linkage, bucket and truck clearance at {len(files) if not args.frames else len(args.frames)} replay poses"
    )
