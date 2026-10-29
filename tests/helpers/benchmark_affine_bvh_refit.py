"""Compare rebuilt and refitted ABD BVH queries on the original bunny surface."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.bodies >= 2 and args.repeats >= 2 and not args.output.exists()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    import taichi as ti
    from src.dem.neighbor.AffineBodyNeighbor import AffineBodyBoundingVolumeHierarchy
    from src.utils.FieldIO import field_to_numpy_prefix

    ti.init(arch=ti.cuda, default_fp=ti.f64, offline_cache=False)
    assert ti.lang.impl.current_cfg().arch == ti.cuda
    mesh = trimesh.load(root / "assets/mesh/LSDEM/bunny_impact_watertight.ply", force="mesh")
    nv, nf, ne = len(mesh.vertices), len(mesh.faces), len(mesh.edges_unique)
    side = int(np.ceil(args.bodies ** (1 / 3)))
    centers = np.array([[i % side, (i // side) % side, i // side**2] for i in range(args.bodies)]) * 0.263516012
    centers[1] = centers[0] + [0.06, 0, 0]
    vertices = (mesh.vertices[None, :, :] - mesh.center_mass + centers[:, None, :]).reshape(-1, 3)
    faces = (mesh.faces[None, :, :] + nv * np.arange(args.bodies)[:, None, None]).reshape(-1, 3)
    edges = (mesh.edges_unique[None, :, :] + nv * np.arange(args.bodies)[:, None, None]).reshape(-1, 2)
    positions = ti.Vector.field(3, ti.f64, shape=len(vertices))
    displacements = ti.Vector.field(3, ti.f64, shape=len(vertices))
    face_field = ti.Vector.field(3, ti.i32, shape=len(faces))
    edge_field = ti.Vector.field(2, ti.i32, shape=len(edges))
    node_body = ti.field(ti.i32, shape=len(vertices))
    face_body = ti.field(ti.i32, shape=len(faces))
    edge_body = ti.field(ti.i32, shape=len(edges))
    positions.from_numpy(vertices)
    displacements.fill(0)
    face_field.from_numpy(faces.astype(np.int32))
    edge_field.from_numpy(edges.astype(np.int32))
    node_body.from_numpy(np.repeat(np.arange(args.bodies, dtype=np.int32), nv))
    face_body.from_numpy(np.repeat(np.arange(args.bodies, dtype=np.int32), nf))
    edge_body.from_numpy(np.repeat(np.arange(args.bodies, dtype=np.int32), ne))
    config = SimpleNamespace(
        search="BVH",
        max_point_triangle_pairs=max(80000, args.bodies * 1000),
        max_edge_edge_pairs=max(400000, args.bodies * 4000),
    )
    broad = AffineBodyBoundingVolumeHierarchy(config, len(vertices), len(faces), len(edges), 0.006)
    build = broad.lbvh.build

    def update():
        broad.update(positions, displacements, face_field, edge_field, node_body, face_body, edge_body, 0.006)

    def fingerprint():
        pt = np.column_stack(
            (
                field_to_numpy_prefix(broad.candidate_vertex, broad.last_vertex_face_candidate_pairs),
                field_to_numpy_prefix(broad.candidate_face, broad.last_vertex_face_candidate_pairs),
            )
        )
        ee = np.column_stack(
            (
                field_to_numpy_prefix(broad.candidate_edge0, broad.last_edge_edge_candidate_pairs),
                field_to_numpy_prefix(broad.candidate_edge1, broad.last_edge_edge_candidate_pairs),
            )
        )
        pt, ee = pt[np.lexsort(pt.T[::-1])], ee[np.lexsort(ee.T[::-1])]
        return dict(pt=len(pt), ee=len(ee), sha256=hashlib.sha256(pt.tobytes() + ee.tobytes()).hexdigest())

    update()
    expected = fingerprint()
    assert expected["pt"] > 0 and expected["ee"] > 0
    samples = []
    for repeat in range(args.repeats):
        positions.from_numpy(vertices + [0, 0, -0.001 * (repeat + 1)])
        for mode in ("build", "refit"):
            broad.lbvh.build = build if mode == "build" else broad.lbvh.refit
            ti.sync()
            began = time.perf_counter()
            update()
            ti.sync()
            seconds = time.perf_counter() - began
            actual = fingerprint()
            assert actual == expected, (mode, repeat, actual, expected)
            samples.append(dict(mode=mode, repeat=repeat, seconds=seconds))
    report = dict(
        backend="cuda",
        precision="float64",
        bodies=args.bodies,
        vertices_per_body=nv,
        candidate_identity_verified=True,
        candidates=expected,
        samples=samples,
        scope="Broad-phase timing only; first complete build warms all kernels",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
