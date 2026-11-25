"""Author the trench cut on the existing excavator and truck rig in Blender."""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
sys.path.insert(0, str(HERE.parent / "explicit_cone"))
import build_scene

build_scene.ROOT = INPUTS
trench = json.loads((INPUTS / "trench.json").read_text())
minimum = Vector(trench["minimum_m"])
size = Vector(trench["size_m"])
maximum = minimum + size
height = trench["wall_height_m"]
thickness = 0.25

bpy.context.scene.frame_set(1)
for name in ("upper_body", "boom", "stick", "bucket"):
    obj = bpy.data.objects[name]
    obj.animation_data_clear()
    obj.rotation_quaternion = (1, 0, 0, 0)
for obj in bpy.context.scene.objects:
    if obj.type == "EMPTY" and "_ram" in obj.name:
        obj.animation_data_clear()

mound = bpy.data.objects["Preview soil mound"]
bpy.ops.mesh.primitive_cube_add(size=1, location=minimum + size / 2)
preview = bpy.context.object
mound.data = preview.data
mound.location, mound.rotation_euler, mound.scale = preview.location, (0, 0, 0), size
bpy.data.objects.remove(preview, do_unlink=True)

wall_material = build_scene.material("Trench concrete", (0.29, 0.28, 0.25))
walls = []
for axis in range(2):
    for side, edge in ((-1, minimum[axis]), (1, maximum[axis])):
        center = minimum + size / 2
        center[axis] = edge + side * thickness / 2
        center.z = height / 2
        dimensions = Vector((size.x, size.y, height))
        dimensions[axis] = thickness
        if axis == 1:
            dimensions.x += 2 * thickness
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        wall = bpy.context.object
        wall.name = "Trench wall"
        wall.scale = dimensions
        wall.data.materials.append(wall_material)
        walls.append(wall)
bpy.context.view_layer.update()


# A single slow lift keeps the bucket moving through the former intermediate stop.
def ease(value):
    return value**3 * (10 + value * (-15 + 6 * value))


lift_end = 100
delay = lift_end - 52
build_scene.smooth = ease
build_scene.LAST_FRAME += delay
bpy.context.scene.frame_end = build_scene.LAST_FRAME
build_scene.KEYS = (
    (1, "Reach", 0, 6.40, 2.40, -127),
    (10, "Reach", 0, 8.48, 2.30, -127),
    (16, "Cut", 0, 8.48, 1.59, -101),
    (25, "Curl", 0, 7.75, 0.56, -97),
    (34, "Lift", 0, 6.15, 0.60, -61),
    *((frame + delay, *key) for frame, *key in build_scene.KEYS[6:]),
)
bucket_contact = bpy.data.objects["bucket_contact"]
bed_contact = bpy.data.objects["truck_bed_contact"]
build_scene.animate(bpy.data.objects["excavator"], bucket_contact, bed_contact)
bucket = bpy.data.objects["bucket"]
wall_trees = [build_scene.object_bvh(wall, wall.matrix_world) for wall in walls]
for frame in range(1, build_scene.LAST_FRAME + 1):
    bpy.context.scene.frame_set(frame)
    for obj, transform in (
        (bucket_contact, bucket.matrix_world),
        (bpy.data.objects["bucket_mesh"], bpy.data.objects["bucket_mesh"].matrix_world),
    ):
        proxy = build_scene.object_bvh(obj, transform)
        assert not any(wall.overlap(proxy) for wall in wall_trees), (frame, obj.name)
bpy.context.scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(INPUTS / "excavator-trench.blend"))
