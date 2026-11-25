"""Author the rig animation, render Blender stills, and assemble the GIF with ffmpeg."""

import csv
from math import acos, atan2, cos, degrees, hypot, pi, radians, sin
from pathlib import Path
from random import Random
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import bpy
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_geometry import bed_mesh, bucket_mesh, write_obj

ROOT = Path(__file__).resolve().parent
FPS, LAST_FRAME = 24, 145
TRUCK_POSITION = (5.45, 0.35, 0.0)
PILE_Y = -6.25
PILE_RADIUS_X, PILE_RADIUS_Y, WALL_HEIGHT, CAP_HEIGHT = 3.25, 2.50, 4.0, 0.8
TOOTH = Vector((-0.05, 1.49, 0.065))
# Frame, phase, yaw, tooth reach at zero yaw, tooth height, absolute bucket pitch.
KEYS = (
    (1, "Reach", 0, 5.60, 5.22, -127),
    (10, "Reach", 0, 7.40, 5.18, -127),
    (16, "Cut", 0, 7.45, 4.95, -101),
    (25, "Curl", 0, 6.55, 2.72, -97),
    (34, "Curl", 0, 5.60, 2.74, -61),
    (41, "Lift", 0, 4.95, 3.30, -12),
    (52, "Swing", 0, 3.65, 5.40, 27),
    (67, "Swing", 42, 4.43, 5.40, 27),
    (82, "Swing", 90, 3.95, 4.68, 27),
    (91, "Dump", 90, 4.70, 4.88, 27),
    (106, "Dump", 90, 6.12, 3.25, -105),
    (118, "Retract", 90, 6.12, 3.25, -105),
    (133, "Retract", 90, 4.21, 4.90, -59),
    (145, "Retract", 90, 3.83, 4.96, -32),
)
STILLS = {25: "01_cut.png", 37: "02_lift.png", 73: "03_swing.png", 106: "04_dump.png", 145: "05_loaded_truck.png"}


def smooth(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def rotate_x(vector, angle):
    return Quaternion((1, 0, 0), angle) @ vector


def pin_target(key):
    _, _, yaw, reach, height, pitch = key
    tooth_offset = rotate_x(TOOTH, radians(pitch))
    return Vector((radians(yaw), reach + tooth_offset.y, height - tooth_offset.z, radians(pitch)))


def sample_pose(frame):
    for start, end in zip(KEYS, KEYS[1:]):
        if frame <= end[0]:
            blend = smooth((frame - start[0]) / (end[0] - start[0]))
            return start[1] if frame < end[0] else end[1], pin_target(start).lerp(pin_target(end), blend)
    raise ValueError(frame)


def joint_angles(boom, stick, bucket, reach, height, pitch):
    # Solve for the bucket pin in the boom plane, where reach points along -Y.
    a = (-stick.location.y, stick.location.z)
    b = (-bucket.location.y, bucket.location.z)
    reach += boom.location.y
    height -= boom.location.z + boom.parent.location.z
    l1, l2 = hypot(*a), hypot(*b)
    cosine = (reach**2 + height**2 - l1**2 - l2**2) / (2 * l1 * l2)
    assert -1 < cosine < 1, (reach, height, cosine)
    elbow = -acos(cosine)
    first = atan2(height, reach) - atan2(l2 * sin(elbow), l1 + l2 * cos(elbow))
    boom_angle = atan2(a[1], a[0]) - first
    stick_angle = atan2(b[1], b[0]) - first - elbow - boom_angle
    return boom_angle, stick_angle, pitch - boom_angle - stick_angle


def material(name, color, roughness=0.75):
    mat = bpy.data.materials.new(name)
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    return mat


def add_light(location, target, power, size):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.rotation_euler = (Vector(target) - light.location).to_track_quat("-Z", "Y").to_euler()
    light.data.energy, light.data.shape, light.data.size = power, "DISK", size


def title(cam, x, y, size, text):
    data = bpy.data.curves.new(text, "FONT")
    data.body, data.size = text, size
    obj = bpy.data.objects.new(text, data)
    bpy.context.collection.objects.link(obj)
    obj.parent, obj.location = cam, (x, y, -2)
    return obj


def make_mound(soil, contact_dir):
    # The soil-only rollers hold the vertical base. Only the gently sloped cap is free.
    nx, ny = 16, 12
    verts = []
    faces = []
    for j in range(ny + 1):
        v = 2 * j / ny - 1
        for i in range(nx + 1):
            u = 2 * i / nx - 1
            height = WALL_HEIGHT + CAP_HEIGHT * cos(pi * u / 2) * cos(pi * v / 2)
            verts.append((PILE_RADIUS_X * u, PILE_Y + PILE_RADIUS_Y * v, height))
    for j in range(ny):
        for i in range(nx):
            k = j * (nx + 1) + i
            faces.append((k, k + 1, k + nx + 2, k + nx + 1))
    perimeter = list(range(nx + 1))
    perimeter.extend(j * (nx + 1) + nx for j in range(1, ny + 1))
    perimeter.extend(ny * (nx + 1) + i for i in range(nx - 1, -1, -1))
    perimeter.extend(j * (nx + 1) for j in range(ny - 1, 0, -1))
    base = len(verts)
    verts.extend((*verts[index][:2], 0.0) for index in perimeter)
    for i, k in enumerate(perimeter):
        j = (i + 1) % len(perimeter)
        faces.append((k, base + i, base + j, perimeter[j]))
    faces.append(tuple(reversed(range(base, len(verts)))))
    mound = write_obj("soil_initial_volume", contact_dir, verts, faces)
    mound.name = "Preview soil mound"
    mound.data.materials.append(soil)
    for face in mound.data.polygons:
        face.use_smooth = face.normal.z > 0
    mound.shape_key_add(name="Undisturbed")
    cut = mound.shape_key_add(name="Bucket cut")
    # shortcut: prescribed scoop scar and clods for the preview, replace with solver output for a physical run.
    for vertex in cut.data:
        x, y, height = vertex.co
        notch = smooth((0.84 - abs(x + 0.05)) / 0.40) * smooth((0.70 - abs(y + 6.38)) / 0.40)
        vertex.co.z = height - notch * max(0, height - 3.95) * 0.85
    for frame, value in ((16, 0), (25, 0.70), (33, 1)):
        cut.value = value
        cut.keyframe_insert(data_path="value", frame=frame)


def make_load(truck, bucket, soil_materials):
    rng = Random(27)
    for i in range(96):
        size = rng.uniform(0.035, 0.115)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=size)
        clod = bpy.context.object
        clod.name = f"Soil clod {i + 1:02d}"
        clod.data.materials.append(soil_materials[i % len(soil_materials)])
        for v in clod.data.vertices:
            v.co *= rng.uniform(0.78, 1.28)
        local = Vector((rng.uniform(-0.52, 0.42), rng.uniform(0.58, 1.08), rng.uniform(-0.44, -0.19)))
        released_at = rng.randint(98, 108)
        arrive_at = released_at + rng.randint(12, 15)
        x, y = rng.uniform(-0.87, 0.87), rng.uniform(-1.03, 1.26)
        bed = bpy.data.objects["bed_mesh"]
        inv = bed.matrix_world.inverted()
        above = truck.matrix_world @ Vector((x, y, 8))
        hit, point, _, _ = bed.ray_cast(inv @ above, Vector((0, 0, -1)))
        assert hit, (x, y)
        end = bed.matrix_world @ point
        assert 1.59 < end.z < 1.64, end
        end.z += size * 0.67 + 0.33 * (1 - min(1.0, hypot(x / 1.03, y / 1.55)))
        source = None
        for frame in range(25, arrive_at + 1):
            bpy.context.scene.frame_set(frame)
            if frame <= released_at:
                clod.location = bucket.matrix_world @ local
                source = clod.location.copy()
            else:
                t = (frame - released_at) / (arrive_at - released_at)
                clod.location = source.lerp(end, t)
                duration = (arrive_at - released_at) / FPS
                clod.location.z += 0.5 * 9.81 * duration**2 * t * (1 - t)
            clod.keyframe_insert(data_path="location", frame=frame)
        for frame, factor in ((25, 0.001), (36, 1.0)):
            clod.scale = (factor,) * 3
            clod.keyframe_insert(data_path="scale", frame=frame)


def object_bvh(obj, transform):
    vertices = [transform @ v.co for v in obj.data.vertices]
    return BVHTree.FromPolygons(vertices, [f.vertices[:] for f in obj.data.polygons])


def animate(excavator, bucket_contact, bed_contact):
    yaw, boom, stick, bucket = [bpy.data.objects[name] for name in ("upper_body", "boom", "stick", "bucket")]
    cylinders = [
        tuple(bpy.data.objects[f"{name}_{end}"] for end in ("barrel", "rod"))
        for name in ("boom_ram_L", "boom_ram_R", "stick_ram", "bucket_ram")
    ]
    rest = [rod.matrix_world.translation - barrel.matrix_world.translation for barrel, rod in cylinders]
    visual_bucket = bpy.data.objects["bucket_mesh"]
    bed_tree = object_bvh(bed_contact, bed_contact.matrix_world)
    mound = bpy.data.objects["Preview soil mound"]
    soil_tree = object_bvh(mound, mound.matrix_world)
    links = [obj for joint in (boom, stick) for obj in joint.children if obj.type == "MESH"]
    rows = []
    dump_end = max(frame for frame, phase, *_ in KEYS if phase == "Dump")
    pitch_axis = (1, 0, 0)
    for frame in range(1, LAST_FRAME + 1):
        phase, (azimuth, reach, height, pitch) = sample_pose(frame)
        angles = joint_angles(boom, stick, bucket, reach, height, pitch)
        for joint, axis, angle in zip(
            (yaw, boom, stick, bucket), ((0, 0, 1), pitch_axis, pitch_axis, pitch_axis), (azimuth, *angles)
        ):
            joint.rotation_quaternion = Quaternion(axis, angle)
            joint.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        bpy.context.view_layer.update()
        for (barrel, rod), direction in zip(cylinders, rest):
            axis = rod.matrix_world.translation - barrel.matrix_world.translation
            world_rotation = direction.rotation_difference(axis)
            for obj in (barrel, rod):
                obj.rotation_quaternion = obj.parent.matrix_world.to_quaternion().inverted() @ world_rotation
                obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        pin, tooth = bucket.matrix_world.translation, bucket.matrix_world @ TOOTH
        expected = Vector((reach * sin(azimuth), -reach * cos(azimuth), height))
        assert (pin - expected).length < 2e-5
        min_z = min((visual_bucket.matrix_world @ v.co).z for v in visual_bucket.data.vertices)
        proxy_min_z = min((bucket.matrix_world @ v.co).z for v in bucket_contact.data.vertices)
        assert min(min_z, proxy_min_z) > 0.0, (frame, min_z, proxy_min_z)
        if frame <= 16:
            for obj in (*links, visual_bucket):
                assert not soil_tree.overlap(object_bvh(obj, obj.matrix_world)), (frame, obj.name)
        if frame >= 52:
            assert not bed_tree.overlap(object_bvh(bucket_contact, bucket.matrix_world)), frame
        if frame == dump_end:
            assert abs(tooth.x - TRUCK_POSITION[0]) < 1.0 and abs(tooth.y - TRUCK_POSITION[1]) < 0.65
        rotation = bucket.matrix_world.to_quaternion()
        rows.append(
            (
                frame,
                (frame - 1) / FPS,
                phase,
                degrees(azimuth),
                *map(degrees, angles),
                degrees(pitch),
                *pin,
                *rotation,
                *tooth,
                min_z,
                proxy_min_z,
            )
        )
    assert excavator.location.length == 0
    with (ROOT / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frame",
                "time_s",
                "phase",
                "yaw_deg",
                "boom_deg",
                "stick_deg",
                "bucket_deg",
                "pitch_deg",
                "pin_x_m",
                "pin_y_m",
                "pin_z_m",
                "qw",
                "qx",
                "qy",
                "qz",
                "tooth_x_m",
                "tooth_y_m",
                "tooth_z_m",
                "visual_bucket_min_z_m",
                "contact_bucket_min_z_m",
            )
        )
        writer.writerows(tuple(round(x, 6) if isinstance(x, float) else x for x in row) for row in rows)
    return bucket


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for vehicle in ("excavator", "dump-truck"):
        bpy.ops.import_scene.gltf(filepath=str(ROOT / "assets" / f"{vehicle}.glb"))
    truck = bpy.data.objects["dump-truck"]
    truck.location = TRUCK_POSITION
    contact_dir = ROOT / "contact"
    contact_dir.mkdir(exist_ok=True)
    bucket_contact = bucket_mesh(contact_dir)
    bed_contact = bed_mesh(contact_dir, truck.location)
    for obj in (bucket_contact, bed_contact):
        obj.hide_render = True
        obj.hide_set(True)
    bucket_contact.parent = bpy.data.objects["bucket"]
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.fps, scene.frame_end = FPS, LAST_FRAME
    scene.render.engine = "CYCLES"
    scene.cycles.samples, scene.cycles.use_denoising = 16, True
    scene.render.resolution_x, scene.render.resolution_y = 880, 660
    scene.world = bpy.data.worlds.new("Daylight")
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.55, 0.66, 0.83, 1)
    background.inputs["Strength"].default_value = 0.46
    scene.view_settings.view_transform = "AgX"

    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.015))
    bpy.context.object.data.materials.append(material("Gravel yard", (0.27, 0.30, 0.29)))
    soil = material("Ochre soil", (0.32, 0.14, 0.058))
    soil_nodes = soil.node_tree
    noise = soil_nodes.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 27
    bump = soil_nodes.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value, bump.inputs["Distance"].default_value = 0.32, 0.13
    soil_nodes.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    soil_nodes.links.new(bump.outputs["Normal"], soil_nodes.nodes.get("Principled BSDF").inputs["Normal"])
    before_soil = set(bpy.context.scene.objects)
    make_mound(soil, contact_dir)
    soils = [soil] + [
        material("Clod shade", color) for color in ((0.24, 0.090, 0.028), (0.37, 0.165, 0.064), (0.26, 0.108, 0.049))
    ]

    bpy.context.view_layer.update()
    bucket = animate(bpy.data.objects["excavator"], bucket_contact, bed_contact)
    make_load(truck, bucket, soils)
    preview_soil = bpy.data.collections.new("Preview soil")
    scene.collection.children.link(preview_soil)
    for obj in set(scene.objects) - before_soil:
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        preview_soil.objects.link(obj)
    bpy.ops.object.camera_add(location=(-10.5, -16.5, 20))
    cam = bpy.context.object
    cam.rotation_euler = (Vector((2.5, -1.85, 1.8)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type, cam.data.ortho_scale = "ORTHO", 18.0
    scene.camera = cam
    add_light((-8, -7, 13), (1, 0, 0), 2400, 9)
    add_light((9, 4, 10), (2, -2, 1.5), 1900, 8)
    add_light((0, 11, 9), (1, 0, 2), 1100, 7)
    ink = bpy.data.materials.new("Title ink")
    emission = ink.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs[0].default_value = (0.017, 0.026, 0.032, 1)
    ink.node_tree.links.new(emission.outputs[0], ink.node_tree.nodes.get("Material Output").inputs[0])
    title(cam, -7.55, 5.28, 0.49, "EXCAVATOR LOADING CYCLE").data.materials.append(ink)
    title(cam, -7.53, 4.49, 0.18, "KINEMATIC PREVIEW").data.materials.append(ink)
    phase_starts = [(key[0], key[1]) for i, key in enumerate(KEYS) if i == 0 or key[1] != KEYS[i - 1][1]]
    for (start, phase), (end, _) in zip(phase_starts, phase_starts[1:] + [(LAST_FRAME + 1, "")]):
        badge = title(cam, -7.53, 4.80, 0.27, phase.upper())
        badge.data.materials.append(ink)
        for frame, hidden in ((0, True), (start, False), (end, True)):
            badge.hide_render = hidden
            badge.keyframe_insert(data_path="hide_render", frame=frame)

    scene.frame_set(1)
    bpy.ops.file.pack_all()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "excavator.blend"))
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if args == ["--geometry"]:
        return
    if args == ["--preview"]:
        frames = ROOT / "stills"
        frames.mkdir(exist_ok=True)
        for frame, name in STILLS.items():
            scene.frame_set(frame)
            scene.render.filepath = str(frames / name)
            bpy.ops.render.render(write_still=True)
        return
    frames = ROOT / "stills"
    frames.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="excavator_frames_") as temp:
        for i, frame in enumerate(range(1, LAST_FRAME + 1, 3)):
            scene.frame_set(frame)
            output = Path(temp) / f"{i:04d}.png"
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
            if frame in STILLS:
                shutil.copyfile(output, frames / STILLS[frame])
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "8",
                "-i",
                str(Path(temp) / "%04d.png"),
                "-filter_complex",
                "[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                str(ROOT / "excavator_cycle.gif"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
