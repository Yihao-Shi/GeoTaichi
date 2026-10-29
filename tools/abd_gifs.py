"""Render recorded ABD triangle meshes; no checkpoints or simulated intermediate states.

python tools/abd_gifs.py /path/to/manifest.json
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyvista as pv
from PIL import Image

if __package__:
    from .blender_cfdem_gif import export
else:
    from blender_cfdem_gif import export


def read_mesh(path):
    path = Path(path)
    if path.suffix != ".vtu" or path.name.startswith("._"):
        raise ValueError(f"Expected a native VTU mesh: {path}")
    mesh = pv.read(path)
    if not mesh.n_points or not mesh.n_cells or not np.isfinite(mesh.points).all():
        raise ValueError(f"Empty or non-finite mesh: {path}")
    if not np.all(mesh.celltypes == pv.CellType.TRIANGLE):
        raise ValueError(f"Expected triangle surfaces: {path}")
    surface = mesh.extract_surface(algorithm="dataset_surface")
    original = np.asarray(surface.point_data["vtkOriginalPointIds"])
    assert np.array_equal(surface.points, mesh.points[original])
    return surface


def container_panels(bounds):
    """Five rectangles share exact corners; the container is open at the top."""
    lower, upper = np.asarray(bounds, dtype=float)
    assert lower.shape == upper.shape == (3,) and np.all(upper > lower)
    vertices = np.asarray(
        [[x, y, z] for z in (lower[2], upper[2]) for y in (lower[1], upper[1]) for x in (lower[0], upper[0])]
    )
    indices = dict(bottom=[0, 1, 3, 2], left=[0, 2, 6, 4], right=[1, 3, 7, 5], front=[0, 1, 5, 4], back=[2, 3, 7, 6])
    return {name: pv.PolyData(vertices[ids], faces=[4, 0, 1, 2, 3]) for name, ids in indices.items()}


def render(path):
    path = Path(path)
    config = json.loads(path.read_text())
    folder = path.parent
    native = config["native_frames"]
    times = np.asarray([row["time"] for row in native])
    assert len(native) >= 2 and np.isfinite(times).all() and np.all(np.diff(times) > 0)
    assert not (
        config["case"] == "cradle" and config.get("integration") == [1.0, 0.5, 1.0] and not config["complete"]
    ), "BackwardEuler comparison preview GIFs excluded"
    first = read_mesh(native[0]["path"])
    cells = first.faces.copy()
    ids = np.asarray(first.cell_data["bodyID"], dtype=int)
    assert np.array_equal(np.unique(ids), np.arange(config["body_count"]))
    # Stream full-resolution frames: a thousand-bunny sequence cannot stay in RAM.
    bounds = []
    for row in native:
        mesh = read_mesh(row["path"])
        assert np.array_equal(mesh.faces, cells) and np.array_equal(mesh.cell_data["bodyID"], ids)
        bounds.append(mesh.bounds)
    del mesh
    bounds = np.asarray(bounds)
    lower, upper = bounds[:, ::2].min(axis=0), bounds[:, 1::2].max(axis=0)
    palette = np.asarray(
        [[67, 133, 167], [231, 173, 78], [73, 161, 145], [178, 110, 101], [117, 130, 174], [162, 172, 116]],
        dtype=np.uint8,
    )
    colours = palette[ids % len(palette)]
    if config["case"] == "cubes":
        colours = np.tile(np.asarray([82, 146, 180], dtype=np.uint8), (len(ids), 1))
        colours[ids % 2 == 1] = [110, 166, 193]
        colours[ids == config["body_count"] - 1] = [207, 81, 65]

    def prepare(mesh):
        mesh.cell_data["display_rgb"] = colours
        mesh.set_active_scalars("display_rgb", preference="cell")
        if config["case"] != "cubes":
            mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
        return mesh

    prepare(first)
    view = pv.Plotter(off_screen=True, window_size=(900, 660))
    view.set_background("white")
    view.enable_anti_aliasing("ssaa")
    actor = view.add_mesh(
        first,
        scalars="display_rgb",
        rgb=True,
        preference="cell",
        show_scalar_bar=False,
        smooth_shading=config["case"] != "cubes",
        ambient=0.25,
        diffuse=0.75,
        specular=0.25,
        specular_power=25,
    )
    right_wall = None
    centre = (lower + upper) / 2
    if config["case"] in ("bunny", "cubes"):
        if config["case"] == "bunny":
            box = np.asarray(config["container_bounds"], dtype=float)
            lower, upper = np.minimum(lower, box[0]), np.maximum(upper, box[1])
            # The physical bottom is an infinite plane, including the released pile's runout.
            ground = pv.Plane(
                center=(*((lower + upper)[:2] / 2), box[0, 2] - 1e-5),
                i_size=upper[0] - lower[0] + 0.3,
                j_size=upper[1] - lower[1] + 0.3,
            )
            view.add_mesh(ground, color="#f0f0ed", ambient=0.4, diffuse=0.6)
            for name, surface in container_panels(box).items():
                wall = view.add_mesh(
                    surface,
                    color="#e8e9e6" if name == "bottom" else "#aab9c1",
                    opacity=1.0 if name == "bottom" else 0.16,
                    show_edges=name != "bottom",
                    edge_color="#7f919d",
                    line_width=1,
                )
                if name == "right":
                    right_wall = wall
            centre = (lower + upper) / 2
        else:
            floor = pv.Plane(
                center=(centre[0], centre[1], 0.05),
                i_size=upper[0] - lower[0] + 0.30,
                j_size=upper[1] - lower[1] + 0.28,
            )
            view.add_mesh(floor, color="#e8e9e6", ambient=0.4, diffuse=0.6)
        direction = np.array([1.4, -2.2, 1.3])
    else:
        pivots = np.column_stack((0.55 + np.arange(5) * 0.1022, np.full(5, 0.3), np.full(5, 0.75)))
        beam = pv.Line(pivots[0] + [-0.04, 0, 0], pivots[-1] + [0.04, 0, 0]).tube(radius=0.004)
        view.add_mesh(beam, color="#59616b")
        upper[2] = 0.77
        centre = (lower + upper) / 2
        direction = np.array([0.10, -2.5, 0.55])
    view.camera_position = (centre + direction * max(upper - lower) * 2, centre, (0, 0, 1))
    view.enable_parallel_projection()
    view.reset_camera()
    view.camera.zoom(1.10)
    camera = dict(
        position=list(view.camera.position),
        focal_point=list(view.camera.focal_point),
        parallel_scale=view.camera.parallel_scale,
    )
    raw = folder / "raw_frames"
    raw.mkdir(exist_ok=True)
    coverage = []
    for index, row in enumerate(native):
        mesh = prepare(first if index == 0 else read_mesh(row["path"]))
        actor.mapper.dataset = mesh
        if right_wall is not None:
            release = config.get("release_time")
            right_wall.visibility = release is None or row["time"] < release - 1e-10
        if config["case"] == "cradle":
            point_ids = np.asarray(mesh.point_data["bodyID"], dtype=int)
            centres = np.asarray([mesh.points[point_ids == i].mean(axis=0) for i in range(5)])
            ends = np.stack((pivots, centres), axis=1).reshape(-1, 3)
            lines = np.column_stack((np.full(5, 2), 2 * np.arange(5), 2 * np.arange(5) + 1)).ravel()
            rods = pv.PolyData(ends, lines=lines).tube(radius=0.0014)
            view.add_mesh(rods, name="rods", color="#67717e")
        view.add_text(config["title"], position="upper_left", font_size=16, color="#263844", name="title")
        label = f"t = {row['time']:.3f} s"
        if not config["complete"]:
            label += "  |  preview"
        view.add_text(label, position="lower_left", font_size=12, color="#34495b", name="time")
        image = view.screenshot(raw / f"native_{index:06d}.png", return_img=True)
        fraction = float(np.mean(np.min(image[:, :, :3], axis=2) < 220))
        assert fraction > 0.015, f"Blank render at {row['time']}"
        coverage.append(fraction)
    view.close()
    # Hold saved states on a uniform playback clock; never interpolate physical poses.
    frame_count = max(90, int(np.ceil((times[-1] - times[0]) * 18)) + 1)
    clock = np.linspace(times[0], times[-1], frame_count)
    indices = np.maximum(np.searchsorted(times, clock, side="right") - 1, 0)
    assert indices[0] == 0 and indices[-1] == len(native) - 1
    config["frames"] = []
    for index, source in enumerate(indices):
        target = folder / f"frame_{index:06d}.png"
        if target.exists():
            target.unlink()
        shutil.copyfile(raw / f"native_{source:06d}.png", target)
        config["frames"].append(dict(time=float(times[source]), source_index=int(source)))
    config["fps"] = 18
    export(config, folder)
    output = Path(config["output"])
    with Image.open(output) as gif:
        assert gif.n_frames > 1 and gif.size == (900, 660)
        decoded_frames = gif.n_frames
    for row in native:
        row["sha256"] = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
    proof = dict(
        config,
        camera=camera,
        minimum_object_coverage=min(coverage),
        decoded_gif_frames=decoded_frames,
        pyvista=pv.__version__,
        vtk=pv.vtk_version_info,
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        checkpoint_files_read=0,
        native_geometry_preserved=True,
        source_time_range=[float(times[0]), float(times[-1])],
    )
    (folder / "render_proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), source_frames=len(native), times=proof["source_time_range"])), flush=True)


if __name__ == "__main__":
    render(sys.argv[1])
