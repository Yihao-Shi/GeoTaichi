"""Render full MPM snapshots with a translucent camera-facing half of the bed.

Run in Blender with -- RUN OUTPUT. Blend the section and full PNG sequences
at 78% and 22%, respectively, to retain dense front soil at fixed opacity.
"""

import argparse
import csv
import json
from math import pi
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pile_archive import frame_paths, load_frame

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--frame", type=int, action="append", help="Render only these replay frame indices")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
config = json.loads((args.run / "run-config.json").read_text())
count = config["particles"]
dt = config["dt_s"]
soil_height = config["soil_size_m"][2]
with (args.run / "progress.csv").open(newline="") as stream:
    rows = {round(float(row["time_s"]) / dt): row for row in csv.DictReader(stream)}
rows[0] = dict(advance_m=0.0, turns=0.0, pile_force_n=0.0)
snapshots = frame_paths(args.run / "snapshots")
assert snapshots and {int(path.stem) for path in snapshots} <= rows.keys()

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 8
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 3
scene.render.use_persistent_data = True
if sys.platform == "darwin":
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.refresh_devices()
    for device in preferences.devices:
        device.use = device.type == "METAL"
    if any(device.use for device in preferences.devices):
        scene.cycles.device = "GPU"
scene.render.resolution_x, scene.render.resolution_y = 1520, 1640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "AgX"
scene.world = bpy.data.worlds.new("Daylight")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.38


def material(name, color, roughness, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


bpy.ops.wm.obj_import(filepath=str(Path(__file__).parent / "inputs/screw_pile.obj"), forward_axis="Y", up_axis="Z")
pile = bpy.context.selected_objects[0]
assert len(pile.data.polygons) == 14012 and np.isclose(max(v.co.z for v in pile.data.vertices), 13.0)
pile.data.materials.clear()
pile.data.materials.append(material("Blued steel", (0.13, 0.28, 0.42), 0.32, 0.65))
pile.rotation_mode = "QUATERNION"
x, y, z, w = config["tilt"]
tilt = Quaternion((w, x, y, z)).normalized()
axis = tilt @ Vector((0, 0, 1))
start = Vector(config["start_tip_m"])

rng = np.random.default_rng(94)
radii = rng.uniform(0.064, 0.088, count).astype(np.float32)
variation = rng.uniform(-0.095, 0.095, count).astype(np.float32)
mesh = bpy.data.meshes.new("MPM snapshots")
mesh.vertices.add(count)
mesh.attributes.new("grain_radius", "FLOAT", "POINT").data.foreach_set("value", radii)
mesh.attributes.new("sand_tone", "FLOAT", "POINT")
mesh.attributes.new("keep_in_section", "BOOLEAN", "POINT")
soil = bpy.data.objects.new("One grain per MPM point", mesh)
scene.collection.objects.link(soil)
sand = material("Dry sand", (0.45, 0.29, 0.12), 0.94)
shader = sand.node_tree.nodes["Principled BSDF"]
shader.inputs["Specular IOR Level"].default_value = 0.13
tone = sand.node_tree.nodes.new("ShaderNodeAttribute")
tone.attribute_name = "sand_tone"
ramp = sand.node_tree.nodes.new("ShaderNodeValToRGB")
colors = ((0.225, 0.115, 0.046), (0.46, 0.285, 0.122), (0.72, 0.548, 0.32))
for element, color in zip(
    (ramp.color_ramp.elements[0], ramp.color_ramp.elements.new(0.5), ramp.color_ramp.elements[-1]), colors
):
    element.color = (*color, 1.0)
sand.node_tree.links.new(tone.outputs["Fac"], ramp.inputs["Fac"])
sand.node_tree.links.new(ramp.outputs["Color"], shader.inputs["Base Color"])

tree = bpy.data.node_groups.new("Replay grains", "GeometryNodeTree")
for direction in ("INPUT", "OUTPUT"):
    tree.interface.new_socket(name="Geometry", in_out=direction, socket_type="NodeSocketGeometry")
source = tree.nodes.new("NodeGroupInput")
points = tree.nodes.new("GeometryNodeMeshToPoints")
for name, kind, socket in (("grain_radius", "FLOAT", "Radius"), ("keep_in_section", "BOOLEAN", "Selection")):
    attribute = tree.nodes.new("GeometryNodeInputNamedAttribute")
    attribute.data_type = kind
    attribute.inputs["Name"].default_value = name
    tree.links.new(attribute.outputs["Attribute"], points.inputs[socket])
assign = tree.nodes.new("GeometryNodeSetMaterial")
assign.inputs["Material"].default_value = sand
tree.links.new(source.outputs["Geometry"], points.inputs["Mesh"])
tree.links.new(points.outputs["Points"], assign.inputs["Geometry"])
tree.links.new(assign.outputs["Geometry"], tree.nodes.new("NodeGroupOutput").inputs["Geometry"])
soil.modifiers.new("Render grains", "NODES").node_group = tree

bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.12))
backdrop = material("Backdrop", (0.22, 0.24, 0.25), 0.94)
backdrop_shader = backdrop.node_tree.nodes["Principled BSDF"]
backdrop_shader.inputs["Emission Color"].default_value = (0.42, 0.46, 0.49, 1.0)
backdrop_shader.inputs["Emission Strength"].default_value = 1.0
bpy.context.object.data.materials.append(backdrop)
bpy.ops.object.camera_add(location=(12, -40, 29))
camera = bpy.context.object
camera.rotation_euler = (Vector((0.35, 0.0, 14.2)) - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera
camera.data.type, camera.data.ortho_scale = "ORTHO", 35.5
for name, loc, energy, size in (("Key", (-15, -18, 32), 14000, 14.0), ("Fill", (8, 12, 24), 7500, 12.0)):
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.object
    light.name = name
    light.rotation_euler = (Vector((0.0, 0.0, 12.0)) - light.location).to_track_quat("-Z", "Y").to_euler()
    light.data.energy, light.data.size = energy, size


def ink(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    mat.node_tree.links.new(emission.outputs["Emission"], nodes["Material Output"].inputs["Surface"])
    return mat


def overlay(obj, x, y):
    scene.collection.objects.link(obj)
    obj.parent, obj.location = camera, (x, y, -2.0)
    obj.visible_shadow = obj.visible_diffuse = obj.visible_glossy = False


text_mat = ink("Slate ink", (0.022, 0.042, 0.059))


def label(text, x, y, size):
    curve = bpy.data.curves.new(text, "FONT")
    curve.body, curve.size = text, size
    curve.materials.append(text_mat)
    overlay(bpy.data.objects.new(text, curve), x, y)
    return curve


label("Screw pile installation", -15.0, 15.5, 1.10)
label(f"{count:,} material points", -15.0, 14.35, 0.64)
readout = label("", -15.0, 13.27, 0.6)
label("Depth below initial surface (m)", -15.0, -14.25, 0.61)
for i in range(48):
    shade = 1 - (i + 0.5) / 48
    band = min(int(shade * 2), 1)
    color = tuple(np.interp(shade * 2 - band, (0, 1), (a, b)) for a, b in zip(colors[band], colors[band + 1]))
    swatch = bpy.data.meshes.new("Depth scale")
    swatch.from_pydata([(0, 0, 0), (0.2, 0, 0), (0.2, 0.33, 0), (0, 0.33, 0)], [], [(0, 1, 2, 3)])
    swatch.materials.append(ink("Depth scale", color))
    overlay(bpy.data.objects.new("Depth scale", swatch), -15 + i * 0.2, -15)
for text, xpos in (("0", -15.0), (f"{soil_height / 2:g}", -10.35), (f"{soil_height:g}", -6.0)):
    label(text, xpos, -15.78, 0.52)
label("Front soil  /  22% opacity", 5.0, -15.05, 0.61)

for layer in ("section", "full"):
    (args.output / layer).mkdir(parents=True, exist_ok=True)
for frame in range(len(snapshots)) if args.frame is None else args.frame:
    path = snapshots[frame]
    step = int(path.stem)
    row = rows[step]
    points = load_frame(path, count)
    assert points.shape == (count, 3) and np.isfinite(points).all(), path
    mesh.vertices.foreach_set("co", np.ascontiguousarray(points, dtype=np.float32).ravel())
    mesh.attributes["sand_tone"].data.foreach_set(
        "value", np.clip(points[:, 2] / soil_height + variation, 0, 1).astype(np.float32)
    )
    pile.rotation_quaternion = tilt @ Quaternion((0, 0, 1), -2 * pi * float(row["turns"]))
    pile.location = start - float(row["advance_m"]) * axis
    readout.body = f'{step * dt:5.1f} s   |   Advance {float(row["advance_m"]):.2f} m   |   Force {float(row["pile_force_n"]) / 1000:.1f} kN'
    # Blending depth-tested images keeps the front opacity independent of grain count.
    for layer, selection in (("section", points[:, 1] >= 0), ("full", np.ones(count, dtype=bool))):
        image = args.output / layer / f"{frame:04d}.png"
        if image.is_file():
            continue
        mesh.attributes["keep_in_section"].data.foreach_set("value", selection)
        mesh.update()
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
    print(f"Rendered frame {frame:04d} with {count:,} MPM points", flush=True)
