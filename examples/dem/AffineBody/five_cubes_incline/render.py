"""Render saved triangle VTUs; never prescribe or interpolate cube motion."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
COLORS = [(0.86, 0.23, 0.10), (0.93, 0.53, 0.08), (0.13, 0.55, 0.63), (0.15, 0.31, 0.58), (0.43, 0.25, 0.60)]


def render(data_path, preview):
    import bpy
    from mathutils import Vector

    data = np.load(data_path)
    vertices, faces, owners, times = (data[k] for k in ("vertices", "faces", "owners", "times"))
    folder = data_path.parent
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y = 960, 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.5
    scene.world = bpy.data.worlds.new("Studio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.85, 0.92, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

    def material(name, color):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Roughness"].default_value = 0.4
        return mat

    def mesh(name, points, triangles, mat):
        geometry = bpy.data.meshes.new(name)
        geometry.from_pydata(points.tolist(), [], triangles.tolist())
        geometry.materials.append(mat)
        obj = bpy.data.objects.new(name, geometry)
        scene.collection.objects.link(obj)
        return obj

    theta = np.pi / 6
    up, normal = np.array([np.cos(theta), 0, np.sin(theta)]), np.array([-np.sin(theta), 0, np.cos(theta)])

    def ramp_point(s, y, h=0):
        return np.array([0, y, 0.1]) + s * up + h * normal

    ramp = np.array(
        [ramp_point(s, y, h) for h in (-0.08, 0) for s, y in [(0, 0.05), (4.05, 0.05), (4.05, 2.65), (0, 2.65)]]
    )
    ramp_faces = np.array(
        [
            (4, 5, 6),
            (4, 6, 7),
            (0, 2, 1),
            (0, 3, 2),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ]
    )
    mesh("Incline display surface", ramp, ramp_faces, material("Ramp", (0.45, 0.52, 0.59)))
    line_mat = material("Lane markers", (0.7, 0.76, 0.8))
    for y in (0.6, 1.1, 1.6, 2.1):
        for s in np.arange(0.1, 3.9, 0.22):
            quad = np.array(
                [
                    ramp_point(a, b, 0.001)
                    for a, b in [(s, y - 0.005), (s + 0.12, y - 0.005), (s + 0.12, y + 0.005), (s, y + 0.005)]
                ]
            )
            mesh("Paint", quad, np.array([(0, 1, 2), (0, 2, 3)]), line_mat)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.025))
    bpy.context.object.data.materials.append(material("Floor", (0.88, 0.90, 0.93)))
    body_materials = [material(f"Cube {i+1}", color) for i, color in enumerate(COLORS)]
    objects = []
    for i in range(5):
        ids = np.flatnonzero(owners == i)
        mapping = np.full(len(owners), -1, dtype=int)
        mapping[ids] = np.arange(len(ids))
        triangles = faces[np.all(owners[faces] == i, axis=1)]
        objects.append((mesh(f"Cube {i+1}", vertices[0, ids], mapping[triangles], body_materials[i]), ids))
    target = Vector(ramp_point(2, 1.35, 0.05))
    bpy.ops.object.camera_add(location=target + Vector((-5.5, -7.5, 6.5)))
    camera = bpy.context.object
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type, camera.data.ortho_scale = "ORTHO", 6.7
    scene.camera = camera
    for location, power, size in [((-1, -3, 8), 1200, 5), ((5, 3, 7), 900, 4)]:
        bpy.ops.object.light_add(type="AREA", location=location)
        bpy.context.object.data.energy = power
        bpy.context.object.data.shape = "DISK"
        bpy.context.object.data.size = size
        bpy.context.object.rotation_euler = (target - bpy.context.object.location).to_track_quat("-Z", "Y").to_euler()

    def label(text, x, y, size, mat):
        curve = bpy.data.curves.new(text, "FONT")
        curve.body, curve.size = text, size
        text_material = bpy.data.materials.new(text)
        text_material.use_nodes = True
        nodes = text_material.node_tree.nodes
        nodes.clear()
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = mat.diffuse_color
        target = nodes.new("ShaderNodeOutputMaterial")
        text_material.node_tree.links.new(emission.outputs[0], target.inputs["Surface"])
        curve.materials.append(text_material)
        obj = bpy.data.objects.new(text, curve)
        scene.collection.objects.link(obj)
        obj.rotation_euler = camera.rotation_euler
        obj.location = camera.location + camera.rotation_euler.to_matrix() @ Vector((x, y, -5))
        return curve

    ink = material("Ink", (0.035, 0.055, 0.085))
    label("FIVE CUBES / ONE INCLINE", -2.72, 1.88, 0.20, ink)
    label("30° slope  •  ABD + IPC  •  converged friction", -2.72, 1.65, 0.10, ink)
    clock = label("", 1.6, 1.88, 0.13, ink)
    for i, mu in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
        x = -2.70 + 1.1 * i
        label(f"{i+1}   μ = {mu:.2f}", x, -2.20, 0.14, body_materials[i])
        label("SLIDING" if i < 3 else "STICKING", x, -2.40, 0.095, ink)
    indices = [0, len(times) - 1] if preview else range(len(times))
    for frame in indices:
        for obj, ids in objects:
            obj.data.vertices.foreach_set("co", vertices[frame, ids].ravel())
            obj.data.update()
        clock.body = f"t = {times[frame]:.2f} s"
        scene.render.filepath = str(folder / f"frame_{frame:06d}.png")
        bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "images/abd_incline.gif")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--blender", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    if args.blender:
        render(args.source, args.preview)
        return
    sys.path.insert(0, str(ROOT))
    from blender.core.vtu import read_vtu_geometry

    source = args.source.resolve()
    summary = json.loads((source / "validation_summary.json").read_text())
    assert summary["passed"], "Refusing to render a failed trajectory"
    saved = sorted((source / "vtks").glob("GraphicAffineBody*.vtu"))
    assert saved, "No saved triangle meshes"
    paths = saved[::2]
    if paths[-1] != saved[-1]:
        paths.append(saved[-1])
    frames = [read_vtu_geometry(path) for path in paths]
    assert len(frames) >= 2
    first = frames[0]
    faces = np.asarray(first.faces)
    # This example emits five contiguous eight-vertex cube meshes.
    assert len(first.points) == 40 and faces.shape == (60, 3)
    owners = np.repeat(np.arange(5), 8)
    assert all(np.array_equal(frame.faces, faces) for frame in frames)
    points = np.array([frame.points for frame in frames])
    assert np.isfinite(points).all() and set(owners) == set(range(5))
    times = []
    for path in paths:
        # The matching recorder metadata supplies actual time, including short final steps.
        with np.load(source / "particles" / f"AffineBody{path.stem[-6:]}.npz") as metadata:
            times.append(float(metadata["t_current"]))
    times = np.asarray(times)
    assert np.all(np.diff(times) > 0)
    folder = source / "render"
    folder.mkdir(exist_ok=True)
    data_path = folder / "meshes.npz"
    np.savez(data_path, vertices=points, faces=faces, owners=owners, times=times)
    command = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        str(data_path),
        "--blender",
    ]
    if args.preview:
        command.append("--preview")
    subprocess.run(command, check=True)
    if not args.preview:
        sys.path.insert(0, str(ROOT))
        from tools.blender_cfdem_gif import export

        export({"output": str(args.output.resolve()), "fps": 20, "frames": times.tolist()}, folder)


if __name__ == "__main__":
    main()
