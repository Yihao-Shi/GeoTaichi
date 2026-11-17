"""Export completed footing snapshots for independent ParaView rendering."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.mpm.elements.AdaptiveGridOutput import build_adaptive_leaf_mesh
from third_party.pyevtk.hl import pointsToVTK, unstructuredGridToVTK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    run = json.loads((output / "run.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    if run["status"] != "completed" or not evaluation["passed"] or run["level"] != 2:
        raise ValueError("Rendering requires a completed, evaluated L2 run")
    render = output / "render"
    vtk = render / "vtk"
    vtk.mkdir(parents=True, exist_ok=True)
    collections = {
        name: ET.Element("VTKFile", type="Collection", version="0.1") for name in ("soil", "footing", "grid")
    }
    datasets = {name: ET.SubElement(root, "Collection") for name, root in collections.items()}
    frames = []
    for index, path in enumerate(sorted((output / "particles").glob("MPMParticle*.npz"))):
        with np.load(path, allow_pickle=True) as data:
            time = float(data["t_current"])
            strain = data["state_vars"].item()["epdstrain"]
            assert np.isfinite(strain).all() and np.all(data["volume"] > 0)
            assert len(strain) == len(data["position"]) == int(data["body_num"])
            for name, body in (("soil", 0), ("footing", 1)):
                mask = (data["bodyID"] == body) & (data["active"] == 1)
                pos = data["position"][mask]
                file = pointsToVTK(
                    str(vtk / f"{name}{index:06d}"),
                    np.ascontiguousarray(pos[:, 0]),
                    np.ascontiguousarray(pos[:, 1]),
                    np.zeros(len(pos)),
                    data={
                        "epdstrain": np.ascontiguousarray(strain[mask]),
                        "particle_width": np.ascontiguousarray(np.sqrt(data["volume"][mask])),
                    },
                )
                ET.SubElement(datasets[name], "DataSet", timestep=str(time), file=str(Path(file).relative_to(render)))
            mesh = build_adaptive_leaf_mesh(
                data["refined_cell"], data["coarse_cnum"], [run["fine_h"]] * 2, run["level"]
            )
            coords, cells = mesh["coords"], mesh["connectivity"]
            assert cells.min() >= 0 and cells.max() < len(coords)
            file = unstructuredGridToVTK(
                str(vtk / f"grid{index:06d}"),
                np.ascontiguousarray(coords[:, 0]),
                np.ascontiguousarray(coords[:, 1]),
                np.full(len(coords), -0.02),
                connectivity=cells.ravel(),
                offsets=np.arange(4, 4 * len(cells) + 1, 4, dtype=np.int64),
                cell_types=np.full(len(cells), 9, dtype=np.uint8),
                cellData={"grid_level": mesh["grid_level"]},
            )
            ET.SubElement(datasets["grid"], "DataSet", timestep=str(time), file=str(Path(file).relative_to(render)))
            frames.append({"time_s": time, "maximum_epdstrain": float(strain.max()), "leaf_cells": len(cells)})
    assert len(frames) == evaluation["frames"] and np.all(np.diff([f["time_s"] for f in frames]) > 0)
    for name, collection in collections.items():
        ET.ElementTree(collection).write(render / f"{name}.pvd", encoding="utf-8", xml_declaration=True)
    (render / "inputs.json").write_text(json.dumps({"source": str(output), "frames": frames}, indent=2) + "\n")
    print(f"Prepared {len(frames)} saved states in {render}")


if __name__ == "__main__":
    main()
