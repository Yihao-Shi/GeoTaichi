"""Export screw-pile geometry from explicit replay snapshots and prescribed poses."""

import csv
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import meshio
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from pile_archive import frame_paths, load_frame

run, output = map(Path, sys.argv[1:])
config = json.loads((run / "run-config.json").read_text())
mesh = trimesh.load_mesh(Path(__file__).parent / "inputs/screw_pile.obj", merge_tex=True, merge_norm=True)
tilt = Rotation.from_quat(config["tilt"])
axis = tilt.apply([0.0, 0.0, 1.0])
start = np.asarray(config["start_tip_m"])
with (run / "progress.csv").open(newline="") as stream:
    rows = {round(float(row["time_s"]) / config["dt_s"]): row for row in csv.DictReader(stream)}
rows[0] = {"advance_m": 0.0, "turns": 0.0}
paths = frame_paths(run / "snapshots")
assert paths and {int(path.stem) for path in paths} <= rows.keys()
output.mkdir()
root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
collection = ET.SubElement(root, "Collection")
for path in paths:
    step = int(path.stem)
    row = rows[step]
    points = load_frame(path, config["particles"])
    assert points.shape == (config["particles"], 3) and np.isfinite(points).all(), path
    spin = Rotation.from_rotvec([0, 0, -2 * np.pi * float(row["turns"])])
    vertices = (tilt * spin).apply(mesh.vertices) + start - float(row["advance_m"]) * axis
    for part, (name, xyz, cells) in enumerate(
        (
            ("soil", points, ("vertex", np.arange(len(points)).reshape(-1, 1))),
            ("screw_pile", vertices, ("triangle", mesh.faces)),
        )
    ):
        filename = f"{name}_{path.stem}.vtu"
        meshio.write_points_cells(output / filename, xyz, [cells])
        ET.SubElement(collection, "DataSet", timestep=f"{step * config['dt_s']:g}", part=str(part), file=filename)
ET.indent(root)
ET.ElementTree(root).write(output / "screw_pile.pvd", encoding="utf-8", xml_declaration=True)
print(f"Exported {len(paths)} geometry frames")
