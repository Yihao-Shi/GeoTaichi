"""The spherical DKT benchmark needs surface accuracy below its overlap gate."""

import numpy as np
import trimesh

from examples.cfdem.FullyResolved.IBMDraftingKissingTumbling.drafting_kissing_tumbling import SPHERE_MESH


def test_dkt_surface_resolves_the_prescribed_sphere():
    mesh = trimesh.load(SPHERE_MESH, force="mesh")
    assert mesh.is_watertight and mesh.is_convex
    mesh.apply_translation(-mesh.center_mass)
    equivalent_radius = (3.0 * mesh.volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    face_radius = np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles_center)
    vertex_radius = np.linalg.norm(mesh.vertices, axis=1)
    # Keep both inward faceting and outward vertices within 0.5% of the
    # volume-equivalent sphere, leaving margin below the 1.5% overlap gate.
    assert np.min(face_radius) / equivalent_radius >= 0.995
    assert np.max(vertex_radius) / equivalent_radius <= 1.005
