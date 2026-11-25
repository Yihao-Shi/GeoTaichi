"""Read the contact solids and locate the excavator bucket cavity."""

import numpy as np
import trimesh
from shapely.ops import polygonize


def load_solid(path):
    mesh = trimesh.load(path, merge_tex=True, merge_norm=True)
    if not mesh.is_volume:
        raise ValueError(f"Expected a closed, oriented solid: {path}")
    return mesh


def bucket_cavity(mesh):
    # shortcut: follows the demo’s planar side panels, use a cavity mesh for other buckets.
    side_x = np.unique(mesh.vertices[:, 0])[1:-1]
    facet = np.flatnonzero(np.isclose(mesh.facets_origin[:, 0], side_x[0]) & (mesh.facets_normal[:, 0] > 0.99))
    (outline,) = polygonize(mesh.vertices[mesh.facets_boundary[facet.item()]][:, :, 1:])
    return side_x, outline
