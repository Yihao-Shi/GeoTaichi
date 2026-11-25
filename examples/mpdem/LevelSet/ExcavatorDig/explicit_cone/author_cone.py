import json
from math import radians, tan
from pathlib import Path
import sys

import bpy

directory = Path(sys.argv[-1]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene

build_scene.ROOT = directory

cone = json.loads((directory / "cone.json").read_text())
radius, center_y = cone["radius_m"], cone["center_y_m"]
height = radius * tan(radians(cone["slope_deg"]))
mound = bpy.data.objects["Preview soil mound"]
bpy.ops.mesh.primitive_cone_add(
    vertices=128, radius1=radius, radius2=0, depth=height, location=(0, center_y, height / 2)
)
preview = bpy.context.object
mound.data = preview.data
mound.location, mound.rotation_euler, mound.scale = preview.location, (0, 0, 0), (1, 1, 1)
bpy.data.objects.remove(preview, do_unlink=True)


# Use one quintic lift so the bucket eases into the swing at zero acceleration.
def ease(value):
    return value**3 * (10 + value * (-15 + 6 * value))


lift_end = 100
delay = lift_end - 52
build_scene.smooth = ease
build_scene.LAST_FRAME += delay
bpy.context.scene.frame_end = build_scene.LAST_FRAME
build_scene.KEYS = (
    (1, "Reach", 0, 6.30, 3.85, -127),
    (10, "Reach", 0, 8.35, 3.85, -127),
    (16, "Cut", 0, 8.35, 2.49, -101),
    (25, "Curl", 0, 7.70, 1.40, -97),
    (34, "Lift", 0, 6.65, 1.38, -61),
    *((frame + delay, *key) for frame, *key in build_scene.KEYS[6:]),
)

for name in ("upper_body", "boom", "stick", "bucket"):
    joint = bpy.data.objects[name]
    joint.animation_data_clear()
    joint.rotation_quaternion = (1, 0, 0, 0)
for obj in bpy.context.scene.objects:
    if obj.type == "EMPTY" and "_ram" in obj.name:
        obj.animation_data_clear()
bpy.context.view_layer.update()
bucket = bpy.data.objects["bucket_contact"]
bed = bpy.data.objects["truck_bed_contact"]
bpy.context.scene.frame_set(1)
build_scene.animate(bpy.data.objects["excavator"], bucket, bed)
bpy.context.scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(directory / "excavator-cone.blend"))
