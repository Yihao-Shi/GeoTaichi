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
scene.cycles.samples = 12
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


def label(text, x, y, size):
    curve = bpy.data.curves.new(text, "FONT")
    curve.body, curve.size = text, size
    curve.materials.append(bpy.data.materials["Title ink"])
    obj = bpy.data.objects.new(text, curve)
    scene.collection.objects.link(obj)
    obj.parent, obj.location = scene.camera, (x, y, -2)
    return curve


label("EXCAVATOR  /  TRUCK LOADING", -9.3, 6.68, 0.47)
clock = label("", -9.28, 6.17, 0.26)

files = sorted(path for path in (run / "snapshots").glob("*.npy") if path.stem.isdecimal())
assert files, run / "snapshots"
initial = np.load(files[0], allow_pickle=False)
assert initial.shape == (count, 3)
grains_per_point = 1
rng = np.random.default_rng(94)
offsets = rng.normal(size=(count, grains_per_point, 3))
offsets *= 0.047 / np.linalg.norm(offsets, axis=2, keepdims=True)
offsets = offsets.astype(np.float32)
mesh = bpy.data.meshes.new("Sand grains")
mesh.vertices.add(count * grains_per_point)
for name, low, high in (("grain_radius", 0.022, 0.032), ("sand_tone", 0.0, 1.0)):
    attribute = mesh.attributes.new(name, "FLOAT", "POINT")
    attribute.data.foreach_set("value", rng.uniform(low, high, len(mesh.vertices)).astype(np.float32))
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
ramp.color_ramp.elements[0].color = (0.24, 0.154, 0.076, 1.0)
ramp.color_ramp.elements[1].color = (0.64, 0.48, 0.29, 1.0)
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
linkage = fit_linkage()
ram_pairs = []
for name in ("boom_ram_L", "boom_ram_R", "stick_ram", "bucket_ram"):
    pair = []
    for part, suffix in (("barrel", "mesh"), ("rod", "fittings")):
        pivot = bpy.data.objects[f"{name}_{part}"]
        pivot.animation_data_clear()
        child = bpy.data.objects[f"{name}_{part}_{suffix}"]
        local_center = sum((Vector(corner) for corner in child.bound_box), Vector()) / 8
        pair.append((pivot, child.matrix_local @ local_center))
    ram_pairs.append(pair)


def bvh(obj):
    xyz = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    return BVHTree.FromPolygons(xyz, [poly.vertices[:] for poly in obj.data.polygons])


bucket_mesh = bpy.data.objects["bucket_mesh"]
bucket_ram = [obj for pivot, _ in ram_pairs[-1] for obj in pivot.children]
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
output.mkdir(parents=True, exist_ok=True)
for path in files:
    step = int(path.stem)
    image = output / f"{round(step / steps_per_frame):04d}.png"
    if image.exists() and not args.check_linkage:
        continue
    t = step * dt
    if not args.check_linkage:
        positions = np.load(path, allow_pickle=False)
        assert positions.shape == initial.shape and np.isfinite(positions).all(), path
        mesh.vertices.foreach_set("co", (positions[:, None, :] + offsets).ravel())
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
    bucket.matrix_world = world
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
        shell = bvh(bucket_mesh)
        for part in (*linkage, *bucket_ram):
            assert not shell.overlap(bvh(part)), (f"{t:.2f} s", part.name)
        continue
    clock.body = f"t = {t:0.2f} s   |   {count:,} material points"
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)

if args.check_linkage:
    print(f"Checked bucket clearance at {len(files)} replay poses")
