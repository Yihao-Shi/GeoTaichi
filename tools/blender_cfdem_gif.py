#!/usr/bin/env python3
"""Render the three CFDEM gallery cases with Blender Cycles and FFmpeg.

Run with a Python environment containing numpy, scipy and scikit-image:
    python tools/blender_cfdem_gif.py --case all
    python tools/blender_cfdem_gif.py --case dam --preview

Saved particle volumes supply the moving water surface; saved LSDEM meshes/
rotations supply the grains. Sphere positions come from the same saved snapshot.
No simulation is rerun and no fluid time steps are synthesized.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "dkt": (
        "IBMDraftingKissingTumbling/OutputData/remote_20261019/matching_and_sphere_queue_20261019__ibmdraftingkissingtumbling",
        "dkt.gif",
        "Drafting / kissing / tumbling",
        [0.01, 0.01, 0.04],
    ),
    "sphere": (
        "IBMResolvedSphereSettling/OutputData/remote_20261019/sphere_grid16_cpu_retry_20261019__resolved_sphere_grid16_cpu",
        "sphere_oil.gif",
        "Sphere settling in oil",
        [0.1, 0.1, 0.16],
    ),
    "dam": (
        "IBMLevelSetDamBreak3D/OutputData/dam3d_highres_sdf_vf/output",
        "ibm_break.gif",
        "Dam break / buoyant grains",
        [0.48, 0.192, 0.288],
    ),
}


def close_enclosed_voids(sdf, spacing):
    """Remove particle-reconstruction voids without moving the outer surface."""
    from scipy.ndimage import binary_fill_holes

    fluid = sdf < 0
    enclosed = binary_fill_holes(fluid) & ~fluid
    if np.any(enclosed):
        sdf = sdf.copy()
        sdf[enclosed] = -0.05 * spacing
    return sdf


def close_surface_holes(vertices, faces):
    """Restore missing triangle/quad facets; retain closed air pockets and self contacts."""
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    counts = np.bincount(mesh.edges_unique_inverse)
    changed = np.any(counts == 1)
    if changed:
        trimesh.repair.fill_holes(mesh)
        if not np.array_equal(mesh.vertices, vertices) or not np.array_equal(mesh.faces[: len(faces)], faces):
            raise ValueError("Surface repair must preserve saved vertices and existing faces")
        counts = np.bincount(mesh.edges_unique_inverse)
    balance = np.bincount(mesh.edges_unique_inverse, weights=np.sign(mesh.edges[:, 1] - mesh.edges[:, 0]))
    if np.any(counts == 1) or np.any(balance != 0):
        raise ValueError("Fluid surface has an unsupported open boundary")
    return (np.asarray(mesh.vertices), np.asarray(mesh.faces)) if changed else (vertices, faces)


def water_surface(grid, particles, depth=None, close_particle_voids=False, occupancy_threshold=0.5):
    from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
    from skimage.measure import marching_cubes

    if not 0.0 < occupancy_threshold < 1.0:
        raise ValueError("occupancy threshold must lie in (0, 1)")
    dims = grid["dims"]
    dimension = len(dims)
    planes = grid.get("solid_planes")
    if planes is not None:
        planes = np.asarray(planes, dtype=float)
        if (
            planes.ndim != 2
            or planes.shape[1] != dimension + 1
            or not np.isfinite(planes).all()
            or np.any(np.linalg.norm(planes[:, :dimension], axis=1) == 0)
        ):
            raise ValueError("Solid planes must have finite normals and offsets")
    coords = grid["coords"].reshape(*dims, dimension)
    origin = coords[(0,) * dimension]
    spacing = np.array([coords[tuple(int(i == d) for i in range(dimension))][d] - origin[d] for d in range(dimension)])
    active = particles["active"].astype(bool)
    positions = particles["position"][active]
    volumes = particles["volume"][active]
    if not len(positions) or not np.isfinite(positions).all() or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Saved fluid positions and volumes must be finite and volumes positive")
    solid = np.asarray(grid["cell_type"])[..., 0] == 2
    solid_sdf = None
    if "cell_solid_sdf" in grid:
        solid_sdf = np.asarray(grid["cell_solid_sdf"]).reshape(solid.shape)
    elif solid.any():
        solid_sdf = distance_transform_edt(~solid, sampling=spacing) - distance_transform_edt(solid, sampling=spacing)
    # Use particle-scale display cells; this never changes the simulation grid.
    refinement = int(np.clip(np.rint(min(spacing) / np.quantile(volumes, 0.1) ** (1 / dimension)), 1, 2))
    cell_shape = (dims - 1) * refinement
    spacing = spacing / refinement
    if planes is not None:
        solid_sdf = np.full(tuple(cell_shape), np.inf)
        for plane in planes:
            distance = np.full(tuple(cell_shape), plane[-1])
            for axis in range(dimension):
                shape = [1] * dimension
                shape[axis] = cell_shape[axis]
                distance -= plane[axis] * (origin[axis] + (np.arange(cell_shape[axis]) + 0.5) * spacing[axis]).reshape(
                    shape
                )
            solid_sdf = np.minimum(solid_sdf, distance)
        valid = solid_sdf >= 0
    elif solid_sdf is not None:
        if not np.isfinite(solid_sdf).all():
            raise ValueError("Solid boundary distances must be finite")
        solid_sdf = zoom(solid_sdf, refinement, order=1, mode="nearest", grid_mode=True)
        valid = solid_sdf >= 0
    else:
        valid = np.ones(tuple(cell_shape), dtype=bool)
    # Each cloud is a bulk-volume quadrature; overlapping phases must not be summed.
    sigma = 0.65 * refinement
    support = gaussian_filter(valid.astype(float), sigma)

    def smooth_support(sample_positions, sample_volumes):
        position = (sample_positions - origin) / spacing - 0.5
        base = np.floor(position).astype(np.int64)
        fraction = position - base
        volume = sample_volumes / np.prod(spacing)
        supported_weight = np.zeros(len(position))
        for offset in np.ndindex(*([2] * dimension)):
            indices = base + offset
            inside = np.all((indices >= 0) & (indices < cell_shape), axis=1)
            inside[inside] &= valid[tuple(indices[inside].T)]
            weights = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
            supported_weight[inside] += weights[inside]
        occupancy = np.zeros(tuple(cell_shape))
        for offset in np.ndindex(*([2] * dimension)):
            indices = base + offset
            inside = np.all((indices >= 0) & (indices < cell_shape), axis=1)
            inside[inside] &= valid[tuple(indices[inside].T)]
            inside &= supported_weight > 0
            weights = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
            np.add.at(occupancy, tuple(indices[inside].T), (volume * weights)[inside] / supported_weight[inside])
        return gaussian_filter(occupancy, sigma) / np.maximum(support, 1e-12)

    # shortcut: sub-cell droplets are filtered; refine the reconstruction grid to resolve them.
    smooth = smooth_support(positions, volumes)
    if "wet_soil_position" in particles:
        wet_position, wet_volume = particles["wet_soil_position"], particles["wet_soil_volume"]
        if not np.isfinite(wet_position).all() or not np.isfinite(wet_volume).all() or np.any(wet_volume <= 0):
            raise ValueError("Wet soil support must have finite positions and positive volumes")
        smooth = np.maximum(smooth, smooth_support(wet_position, wet_volume))
    sdf = ((occupancy_threshold - smooth) * min(spacing)).astype(np.float32)
    if close_particle_voids:
        sdf = close_enclosed_voids(sdf, min(spacing))
    if not np.isfinite(sdf).all() or not sdf.min() < 0:
        raise ValueError("Fluid SDF must be finite and contain fluid")
    # Cubic interpolation at half-cell spacing rounds grid-scale facets without extra blur.
    fine_shape = 2 * np.array(sdf.shape) - 1
    sdf = zoom(sdf, fine_shape / np.array(sdf.shape), order=3, mode="nearest")
    surface_spacing = spacing / 2
    if close_particle_voids:
        # Cubic upsampling can reintroduce tiny sign islands through overshoot.
        sdf = close_enclosed_voids(sdf, min(surface_spacing))
    if solid_sdf is not None and planes is None:
        # Intersect with physical walls after smoothing, without moving particle positions.
        sdf = np.maximum(sdf, -zoom(solid_sdf, fine_shape / np.array(solid_sdf.shape), order=1, mode="nearest"))
    # Extend edge support, then cap at the physical domain instead of leaving open meshes.
    sdf = np.pad(sdf, 2, mode="edge")
    surface_origin = origin + spacing / 2 - 2 * surface_spacing
    for axis in range(dimension):
        coordinate = surface_origin[axis] + np.arange(sdf.shape[axis]) * surface_spacing[axis]
        wall_distance = np.maximum(
            origin[axis] - coordinate, coordinate - (origin[axis] + (dims[axis] - 1) * spacing[axis] * refinement)
        )
        shape = [1] * dimension
        shape[axis] = len(coordinate)
        sdf = np.maximum(sdf, wall_distance.reshape(shape))
    if planes is not None:
        for plane in planes:
            distance = np.full(sdf.shape, -plane[-1])
            for axis in range(dimension):
                shape = [1] * dimension
                shape[axis] = sdf.shape[axis]
                distance += plane[axis] * (
                    surface_origin[axis] + np.arange(sdf.shape[axis]) * surface_spacing[axis]
                ).reshape(shape)
            sdf = np.maximum(sdf, distance)
    if not sdf.min() < 0 < sdf.max():
        raise ValueError("Boundary-clipped fluid surface must bracket zero")
    if dimension == 2:
        if depth is None or depth <= 0:
            raise ValueError("2-D fluid needs a positive display extrusion depth")
        # A thin extrusion displays a 2-D solution; it does not add simulated flow.
        sdf = np.pad(np.repeat(sdf[..., None], 3, axis=2), ((0, 0), (0, 0), (1, 1)), constant_values=min(spacing) / 2)
        vertices, faces, _, _ = marching_cubes(
            sdf, 0, spacing=(*surface_spacing, depth / 3), gradient_direction="descent", allow_degenerate=False
        )
        vertices += [*surface_origin, -depth / 6]
        vertices[:, 2] = np.clip(vertices[:, 2], 0, depth)
        return close_surface_holes(vertices[:, [0, 2, 1]], faces[:, ::-1])
    vertices, faces, _, _ = marching_cubes(
        sdf, 0, spacing=surface_spacing, gradient_direction="descent", allow_degenerate=False
    )
    # SDF values live at cell centers, not the grid vertices in coords.
    return close_surface_holes(vertices + surface_origin, faces)


def grain_surface(surface, rigid):
    from scipy.spatial.transform import Rotation

    master = surface["master"]
    local = np.arange(len(master)) - rigid["startNode"][master] + rigid["localNode"][master]
    local_vertices = surface["vertices"][local] * rigid["scale"][master, None]
    rotated = Rotation.from_quat(rigid["quanternion"][master]).apply(local_vertices)
    return rotated + rigid["mass_center"][master], surface["connectivity"]


def prepare(case, folder, preview, source=None):
    relative, filename, title, domain = CASES[case]
    source = Path(source).resolve() if source else ROOT / "examples/cfdem/FullyResolved" / relative
    output = ROOT / "images" / filename
    folder.mkdir(parents=True, exist_ok=True)
    scale = 3.6 / max(domain)
    frames = []
    config = json.loads((source / "configuration.json").read_text()) if case != "dam" else {}
    grids = sorted((source / "grids").glob("MPMGrid[0-9]*.npz"))
    if not grids:
        raise FileNotFoundError(f"No saved grids in {source}")
    for index, path in enumerate(grids[:1] if preview else grids):
        step = path.stem[-6:]
        with (
            np.load(path) as grid,
            np.load(source / f"particles/LSDEMRigid{step}.npz") as rigid,
            np.load(source / f"particles/MPMParticle{step}.npz") as particles,
        ):
            time = float(grid["t_current"])
            if any(abs(time - float(data["t_current"])) > 1e-8 for data in (rigid, particles)):
                raise ValueError(f"Mismatched fluid/solid times at {step}")
            vertices, faces = water_surface(grid, particles)
            vertices = np.clip(vertices, 0, domain)
            data = {
                "water": vertices * scale,
                "water_faces": faces,
            }
            frame = {"time": time, "mesh": f"mesh_{index:06d}.npz"}
            if case == "dam":
                with np.load(source / f"particles/LSDEMSurface{step}.npz") as surface:
                    if abs(time - float(surface["t_current"])) > 1e-8:
                        raise ValueError(f"Mismatched surface time at {step}")
                    grains, grain_faces = grain_surface(surface, rigid)
                data.update(grains=grains * scale, grain_faces=grain_faces)
            else:
                frame["centers"] = (rigid["mass_center"] * scale).tolist()
            np.savez(folder / frame["mesh"], **data)
            frames.append(frame)
    manifest = {
        "case": case,
        "source": str(source),
        "output": str(output),
        "title": title,
        "domain": (np.array(domain) * scale).tolist(),
        "frames": frames,
        "radius": config["diameter"] * scale / 2 if case != "dam" else None,
        "fps": 5,
        "fluid_surface_method": "particle_position_and_volume",
        "width": 720 if case == "dkt" else 960,
        "height": 1080 if case == "dkt" else 800 if case == "sphere" else 720,
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path, manifest


def render(path, samples, device="auto", resume=False):
    import bpy
    from mathutils import Vector

    config = json.loads(path.read_text())
    folder, case = path.parent, config["case"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 17
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 8
    devices = bpy.context.preferences.addons["cycles"].preferences
    devices.get_devices()
    if device != "CPU" and any(d.type == "METAL" for d in devices.devices):
        devices.compute_device_type = "METAL"
        devices.get_devices()
        for device in devices.devices:
            device.use = device.type == "METAL"
        if any(d.use for d in devices.devices):
            scene.cycles.device = "GPU"
    scene.render.resolution_x = config["width"]
    scene.render.resolution_y = config["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.world = bpy.data.worlds.new("Studio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.72, 0.78, 0.85, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.5

    def material(name, color, roughness=0.5, metal=0):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metal
        return mat

    water = material(
        "Oil" if case == "sphere" else "Water",
        config.get("water_color", (0.95, 0.86, 0.62) if case == "sphere" else (0.48, 0.86, 0.95)),
        0.18,
    )
    shader = water.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Transmission Weight"].default_value = config.get("water_transmission", 1)
    shader.inputs["IOR"].default_value = 1.47 if case == "sphere" else 1.333
    # A transparent component keeps the immersed grains legible through the tank.
    transparent = water.node_tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs[0].default_value = (0.98, 0.91, 0.72, 1) if case == "sphere" else (0.69, 0.9, 0.97, 1)
    mix = water.node_tree.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = config.get("water_visibility", 0.28)
    water.node_tree.links.new(transparent.outputs[0], mix.inputs[1])
    water.node_tree.links.new(shader.outputs[0], mix.inputs[2])
    water.node_tree.links.new(mix.outputs[0], water.node_tree.nodes.get("Material Output").inputs["Surface"])
    stone = material("Sandy porous grains", (0.45, 0.24, 0.095), 0.78)
    noise = stone.node_tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 38
    bump = stone.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    bump.inputs["Distance"].default_value = 0.012
    stone.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    stone.node_tree.links.new(bump.outputs["Normal"], stone.node_tree.nodes.get("Principled BSDF").inputs["Normal"])
    copper = material("Leading sphere", (0.65, 0.19, 0.065), 0.24, 0.35)
    blue = material("Trailing sphere", (0.035, 0.23, 0.40), 0.24, 0.35)
    soft = material("Deformable FEM grains", (0.65, 0.22, 0.065), 0.4)
    rail = material("Tank outline", (0.26, 0.34, 0.39), 0.32, 0.2)
    wall_material = material("See-through boundaries", (0.40, 0.48, 0.53), 0.35)
    wall_nodes = wall_material.node_tree
    wall_transparent = wall_nodes.nodes.new("ShaderNodeBsdfTransparent")
    wall_mix = wall_nodes.nodes.new("ShaderNodeMixShader")
    wall_mix.inputs[0].default_value = 0.16
    wall_nodes.links.new(wall_transparent.outputs[0], wall_mix.inputs[1])
    wall_nodes.links.new(wall_nodes.nodes["Principled BSDF"].outputs[0], wall_mix.inputs[2])
    wall_nodes.links.new(wall_mix.outputs[0], wall_nodes.nodes["Material Output"].inputs["Surface"])
    floor = material("Studio floor", (0.84, 0.86, 0.87), 0.7)
    label_mat = material("Labels", (0.07, 0.11, 0.16))
    emission = label_mat.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.045, 0.065, 0.085, 1)
    label_mat.node_tree.links.new(
        emission.outputs[0], label_mat.node_tree.nodes.get("Material Output").inputs["Surface"]
    )

    def mesh(name, vertices, faces, mat):
        data = bpy.data.meshes.new(name)
        data.from_pydata(vertices.tolist(), [], faces.tolist())
        data.update()
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        data.materials.append(mat)
        for polygon in data.polygons:
            polygon.use_smooth = True
        if mat == water:
            # Weight shading normals by area/angle without moving the saved surface.
            normals = obj.modifiers.new("Water surface normals", "WEIGHTED_NORMAL")
            normals.mode = "FACE_AREA_WITH_ANGLE"
            normals.keep_sharp = True
        return obj

    def particle_spheres(name, positions, radii, mat):
        obj = mesh(name, positions, np.empty((0, 3), dtype=int), mat)
        attribute = obj.data.attributes.new("particle_radius", "FLOAT", "POINT")
        attribute.data.foreach_set("value", radii)
        nodes = bpy.data.node_groups.new("Particle spheres", "GeometryNodeTree")
        nodes.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        nodes.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        source = nodes.nodes.new("NodeGroupInput")
        output = nodes.nodes.new("NodeGroupOutput")
        sphere = nodes.nodes.new("GeometryNodeMeshIcoSphere")
        sphere.inputs["Radius"].default_value = 1
        sphere.inputs["Subdivisions"].default_value = 2
        surface = nodes.nodes.new("GeometryNodeSetMaterial")
        surface.inputs["Material"].default_value = mat
        radius = nodes.nodes.new("GeometryNodeInputNamedAttribute")
        radius.data_type = "FLOAT"
        radius.inputs["Name"].default_value = "particle_radius"
        instances = nodes.nodes.new("GeometryNodeInstanceOnPoints")
        nodes.links.new(source.outputs["Geometry"], instances.inputs["Points"])
        nodes.links.new(sphere.outputs["Mesh"], surface.inputs["Geometry"])
        nodes.links.new(surface.outputs["Geometry"], instances.inputs["Instance"])
        nodes.links.new(radius.outputs["Attribute"], instances.inputs["Scale"])
        nodes.links.new(instances.outputs["Instances"], output.inputs["Geometry"])
        obj.modifiers.new("Particle instances", "NODES").node_group = nodes
        return obj

    def boundary_lines(segments):
        data = bpy.data.curves.new("Boundary edges", "CURVE")
        data.dimensions, data.bevel_depth, data.bevel_resolution = "3D", 0.004, 2
        data.materials.append(rail)
        for a, b in segments:
            spline = data.splines.new("POLY")
            spline.points.add(1)
            spline.points[0].co, spline.points[1].co = (*a, 1), (*b, 1)
        obj = bpy.data.objects.new("Boundary edges", data)
        scene.collection.objects.link(obj)
        return obj

    with np.load(folder / config["frames"][0]["mesh"]) as data:
        if "boundary" in data:
            mesh("Physical boundary", data["boundary"], data["boundary_faces"], rail)

    size = np.array(config.get("display_domain", config["domain"]))
    center = Vector(size / 2)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(size[0] / 2, size[1] / 2, -0.065))
    platform = bpy.context.object
    platform.dimensions = (size[0] + 0.12, size[1] + 0.12, 0.1)
    platform.data.materials.append(floor)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.12))
    bpy.context.object.data.materials.append(floor)
    # Thin edges describe the display tank without extra refracting walls.
    corners = np.array([[x, y, z] for x in [0, size[0]] for y in [0, size[1]] for z in [0, size[2]]])
    curves = bpy.data.curves.new("Tank edges", "CURVE")
    curves.dimensions = "3D"
    curves.bevel_depth = 0.007 if case != "dkt" else 0.0035
    curves.bevel_resolution = 3
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            if np.count_nonzero(a != b) == 1:
                spline = curves.splines.new("POLY")
                spline.points.add(1)
                spline.points[0].co = (*a, 1)
                spline.points[1].co = (*b, 1)
    edges = bpy.data.objects.new("Tank", curves)
    scene.collection.objects.link(edges)
    curves.materials.append(rail)
    edges.hide_render = not config.get("show_tank", True)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    direction = Vector(
        config.get(
            "camera_direction",
            (0.16, -1, 0.1) if case == "dkt" else (0.9, -1.5, 0.8) if case == "dam" else (0.8, -1.8, 0.65),
        )
    ).normalized()
    camera.location = center + direction * 10
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.sensor_fit = "HORIZONTAL"
    scene.camera = camera
    inverse = camera.rotation_euler.to_matrix().transposed()
    projected = np.array([inverse @ (Vector(c) - center) for c in corners])
    aspect = config["width"] / config["height"]
    camera.data.ortho_scale = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]) * aspect) * 1.24
    view_width, view_height = camera.data.ortho_scale, camera.data.ortho_scale / aspect

    def label(name, text, x, y, height):
        data = bpy.data.curves.new(name, "FONT")
        data.body, data.size = text, height
        data.materials.append(label_mat)
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.rotation_euler = camera.rotation_euler
        obj.location = camera.location + camera.rotation_euler.to_matrix() @ Vector((x, y, -5))
        return data

    label("Title", config["title"], -view_width * 0.46, view_height * 0.445, view_width * 0.031)
    time_label = label("Time", "", -view_width * 0.46, view_height * 0.4, view_width * 0.026)
    if config.get("legend"):
        label("Legend", config["legend"], -view_width * 0.46, view_height * 0.355, view_width * 0.021)
    for name, location, energy, diameter in [
        ("Key", (1, -3, 7), 650, 5),
        ("Fill", (-4, -1, 4), 400, 4),
        ("Rim", (2, 4, 5), 800, 3),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size = energy, "DISK", diameter
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()

    spheres = []
    if config.get("radius") is not None:
        for i, location in enumerate(config["frames"][0]["centers"]):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=config["radius"], location=location)
            obj = bpy.context.object
            obj.data.materials.append(copper if i == 0 else blue)
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            spheres.append(obj)
    animated = []
    for index, frame in enumerate(config["frames"]):
        if resume and (folder / f"frame_{index:06d}.png").is_file():
            continue
        for obj in animated:
            data = obj.data
            groups = [modifier.node_group for modifier in obj.modifiers if modifier.type == "NODES"]
            bpy.data.objects.remove(obj, do_unlink=True)
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            else:
                bpy.data.curves.remove(data)
            for group in groups:
                bpy.data.node_groups.remove(group)
        animated = []
        with np.load(folder / frame["mesh"]) as data:
            if "water" in data:
                obj = mesh("Fluid", data["water"], data["water_faces"], water)
                animated.append(obj)
            if "grains" in data:
                animated.append(
                    mesh(
                        "Solids",
                        data["grains"],
                        data["grain_faces"],
                        copper if config.get("solid_material") == "metal" else stone,
                    )
                )
            if "solid_positions" in data:
                animated.append(
                    particle_spheres("Solid particles", data["solid_positions"], data["solid_radii"], stone)
                )
            if "water_positions" in data and len(data["water_positions"]):
                animated.append(particle_spheres("Fluid points", data["water_positions"], data["water_radii"], water))
            for key, mat in (("soft", soft), ("soil", stone), ("rigid", blue), ("walls", wall_material)):
                if key in data:
                    obj = mesh(key, data[key], data[f"{key}_faces"], mat)
                    if key == "walls" or (key == "rigid" and config.get("rigid_flat_shading")):
                        for polygon in obj.data.polygons:
                            polygon.use_smooth = False
                    animated.append(obj)
            if "wall_segments" in data:
                animated.append(boundary_lines(data["wall_segments"]))
        if spheres:
            for obj, location in zip(spheres, frame["centers"]):
                obj.location = location
        time_label.body = f"t = {frame['time']:.3f} s"
        scene.render.filepath = str(folder / f"frame_{index:06d}.png")
        print(f"RENDER {case}: {index + 1}/{len(config['frames'])}", flush=True)
        bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--source", type=Path, help="Saved result directory for a single case")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--device", choices=["auto", "CPU"], default="auto")
    parser.add_argument("--preview", action="store_true", help="Render only the first PNG; do not replace GIFs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep already rendered PNGs in --work-dir")
    parser.add_argument("--render", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.source and args.case == "all":
        parser.error("--source requires a single --case")
    if args.render:
        render(args.render, args.samples, args.device, args.resume)
        return
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="cfdem_blender_"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    print(f"Render assets / PNGs / backups: {work}", flush=True)
    for case in CASES if args.case == "all" else [args.case]:
        path, config = prepare(case, work / case, args.preview, args.source)
        if args.prepare_only:
            continue
        command = [
            args.blender,
            "-b",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--render",
            str(path),
            "--samples",
            str(args.samples),
            "--device",
            args.device,
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)
        if not args.preview:
            export(config, path.parent)


def export(config, folder):
    from PIL import Image

    output = Path(config["output"])
    if not all((folder / f"frame_{i:06d}.png").is_file() for i in range(len(config["frames"]))):
        raise ValueError("Render is incomplete; refusing to replace the GIF")
    for i in range(len(config["frames"])):
        with Image.open(folder / f"frame_{i:06d}.png") as frame:
            if np.asarray(frame.convert("RGB").crop((0, 0, 8, 8))).mean() < 32:
                raise ValueError(f"Black studio background in frame {i}; rerender before exporting")
    # Same palettegen/paletteuse pipeline as tools/vtu2gif.py.
    candidate = folder / output.name
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-framerate",
            str(config["fps"]),
            "-i",
            str(folder / "frame_%06d.png"),
            "-vf",
            "split[a][b];[a]palettegen=stats_mode=diff[p];" "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
            "-frames:v",
            str(len(config["frames"])),
            "-loop",
            "0",
            str(candidate),
        ],
        check=True,
    )
    backup = folder / f"previous_{output.name}"
    if output.exists() and not backup.exists():
        shutil.copy2(output, backup)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".gif", delete=False) as handle:
        staged = Path(handle.name)
    shutil.copy2(candidate, staged)
    staged.replace(output)
    print(f"Saved {output} ({len(config['frames'])} frames)", flush=True)


if __name__ == "__main__":
    main()
