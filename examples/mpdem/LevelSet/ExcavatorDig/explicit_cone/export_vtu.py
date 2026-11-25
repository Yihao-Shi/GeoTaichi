"""Export and verify world-space soil and rigid-body VTUs from saved checkpoints."""

import argparse
from pathlib import Path
import json
from xml.etree import ElementTree as ET

import meshio
import numpy as np
from scipy.spatial.transform import Rotation
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--rigid-name", choices=("bucket_truck", "screw_pile"), default="bucket_truck")
args = parser.parse_args()
run, output = args.run, args.output
output.mkdir()
config = json.loads((run / "run-config.json").read_text())
origin = np.asarray(config["origin_m"])
root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
collection = ET.SubElement(root, "Collection")
reader = vtkXMLUnstructuredGridReader()
times = []
for particle_file in sorted((run / "particles").glob("MPMParticle*.npz")):
    index = particle_file.stem.removeprefix("MPMParticle")
    with (
        np.load(particle_file, allow_pickle=False) as soil,
        np.load(run / "particles" / f"LSDEMRigid{index}.npz", allow_pickle=False) as rigid,
        np.load(run / "particles" / f"LSDEMSurface{index}.npz", allow_pickle=False) as surface,
    ):
        time = float(soil["t_current"])
        assert time == rigid["t_current"] == surface["t_current"]
        times.append(time)
        xyz = soil["position"] + origin
        n = config["particles"]
        assert xyz.shape == (n, 3) and np.all(soil["volume"] > 0)
        stress = soil["stress"]
        data = {key: soil[key] for key in ("particleID", "materialID", "velocity", "mass", "volume")}
        if "radius" in soil:
            data["radius"] = soil["radius"]
        for axis, values in zip(("xx", "yy", "zz", "xy", "yz", "xz"), stress.T):
            data[f"sigma_{axis}_Pa"] = values
        data["mean_compression_Pa"] = -stress[:, :3].mean(axis=1)
        master = surface["master"]
        local_ids = np.arange(len(master)) - rigid["startNode"][master] + rigid["localNode"][master]
        rotations = Rotation.from_quat(rigid["quanternion"][master])
        vertices = (
            rigid["mass_center"][master]
            + origin
            + rotations.apply(rigid["scale"][master, None] * surface["vertices"][local_ids])
        )
        triangles = surface["connectivity"]
        assert np.all(master[triangles] == master[triangles[:, :1]])
        rigid_data = {"bodyID": master, "body_contact_force_N": rigid["contact_force"][master]}
        for part, (name, points, cells, point_data) in enumerate(
            (
                ("soil", xyz, ("vertex", np.arange(n).reshape(-1, 1)), data),
                (args.rigid_name, vertices, ("triangle", triangles), rigid_data),
            )
        ):
            filename = f"{name}_{index}.vtu"
            path = output / filename
            assert np.isfinite(points).all() and all(np.isfinite(v).all() for v in point_data.values())
            meshio.write_points_cells(path, points, [cells], point_data=point_data)
            reader.SetFileName(str(path))
            reader.Update()
            vtk = reader.GetOutput()
            assert vtk.GetNumberOfPoints() == len(points) and vtk.GetNumberOfCells() == len(cells[1])
            assert np.array_equal(vtk_to_numpy(vtk.GetPoints().GetData()), points)
            for key, value in point_data.items():
                assert np.array_equal(vtk_to_numpy(vtk.GetPointData().GetArray(key)), value), (path, key)
            ET.SubElement(collection, "DataSet", timestep=f"{time:g}", part=str(part), file=filename)
assert times and np.all(np.diff(times) > 0), times
ET.indent(root)
ET.ElementTree(root).write(
    output / ("excavator.pvd" if args.rigid_name == "bucket_truck" else "screw_pile.pvd"),
    encoding="utf-8",
    xml_declaration=True,
)
print(f"VTK readback verified {len(times)} native frames and all exported fields.")
