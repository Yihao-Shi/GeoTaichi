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
            noise.inputs["Scale"].default_value = .33
            gravel.links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
            mix = gravel.nodes.new("ShaderNodeMixRGB")
            gravel.links.new(noise.outputs["Fac"], mix.inputs["Fac"])
            gravel.links.new(color, mix.inputs[1])
            gravel.links.new(second.outputs["Color"], mix.inputs[2])
            dirt = gravel.nodes.new("ShaderNodeMixRGB")
            dirt.blend_type = "MULTIPLY"
            dirt.inputs[2].default_value = (.63, .57, .47, 1)
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
pose_linkage, linkage, coupler = fit_linkage()
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
bed = bpy.data.objects["bed_mesh"]
graph = bpy.context.evaluated_depsgraph_get()
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
        shell = bvh(bucket_mesh, graph)
        truck = bvh(bed, graph)
        assert not truck.overlap(shell), (f"{t:.2f} s", bed.name, bucket_mesh.name)
        for part in (*linkage, *coupler):
            part_tree = bvh(part, graph)
            assert part in coupler or not shell.overlap(part_tree), (f"{t:.2f} s", part.name, bucket_mesh.name)
            assert not truck.overlap(part_tree), (f"{t:.2f} s", part.name, bed.name)
        continue
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)

if args.check_linkage:
    print(f"Checked bucket clearance at {len(files)} replay poses")
