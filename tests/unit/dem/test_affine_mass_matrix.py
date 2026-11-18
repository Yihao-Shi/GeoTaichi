from src.dem.generator.AffineBodyTemplate import AffineBodyTemplate
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.utils.errors import GeometryError
import numpy as np
import pytest
import trimesh
from types import SimpleNamespace

from src.dem.engines.AffineBodyState import AffineBodyState


def mass_matrix(mesh, density=1.0):
    material = np.asarray(mesh.vertices) + 0.25
    basis = np.column_stack((1.0 - material.sum(axis=1), material))
    return AffineBodyState._mass_matrix(mesh.vertices, mesh.faces, basis, density, abs(mesh.volume))


@pytest.mark.parametrize("rotation", [np.eye(3), trimesh.transformations.rotation_matrix(0.7, [1, 2, 3])[:3, :3]])
def test_solid_affine_mass_matches_translation_and_full_rotational_inertia(rotation):
    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    mesh.apply_transform(transform)
    density = 2.5
    matrix = mass_matrix(mesh, density)
    controls = np.vstack((np.zeros(3), np.eye(3))) - 0.25
    np.testing.assert_allclose(matrix.sum(), density * mesh.volume, rtol=1e-14)
    for omega in ([0, 0, 1], [1, 2, -3]):
        velocity = np.cross(omega, controls)
        actual = np.einsum("ad,ab,bd->", velocity, matrix, velocity)
        expected = np.asarray(omega) @ (density * mesh.moment_inertia) @ omega
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


def test_solid_affine_mass_is_independent_of_surface_subdivision_and_winding():
    mesh = trimesh.creation.box()
    reference = mass_matrix(mesh)
    np.testing.assert_allclose(mass_matrix(mesh.subdivide()), reference, atol=1e-15)
    mesh.invert()
    np.testing.assert_allclose(mass_matrix(mesh), reference, atol=1e-15)


def test_solid_affine_mass_rejects_invalid_density():
    with pytest.raises(ValueError, match="density"):
        mass_matrix(trimesh.creation.box(), density=np.nan)


def test_solid_template_rejects_open_surface_and_repairs_reversed_winding():
    mesh = trimesh.creation.box()
    template = AffineBodyTemplate()
    open_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[:-1], process=False)
    with pytest.raises(RuntimeError, match="closed"):
        template._extract_mesh(SimpleNamespace(mesh=open_mesh, _reset=False))
    mesh.invert()
    normalized, unused_rotation, unused_center = template._extract_mesh(SimpleNamespace(mesh=mesh, _reset=False))
    assert normalized.volume == pytest.approx(1.0)
    np.testing.assert_allclose(normalized.center_mass, 0.0, atol=1e-14)


def test_affine_and_levelset_templates_share_surface_measure_without_mutating_input():
    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    mesh.apply_translation([2.0, -3.0, 4.0])
    original_vertices = np.array(mesh.vertices)
    affine = AffineBodyTemplate()
    affine.surface_template({"Name": "box", "Object": SimpleNamespace(mesh=mesh, _reset=True)})

    levelset = GeneralShapeTemplate()
    levelset.objects = SimpleNamespace(mesh=trimesh.Trimesh(affine.vertices, affine.faces, process=False))
    levelset.calculate_surface_parameter()

    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    assert affine.volume == pytest.approx(6.0)
    assert levelset.surface_area == pytest.approx(22.0)
    np.testing.assert_allclose(affine.surface_weights.sum(), 22.0)
    np.testing.assert_allclose(levelset.parameter, affine.surface_weights / 22.0)


@pytest.mark.parametrize(
    "options,key", [({"VisualizeMode": "invalid"}, "VisualizeMode"), ({"RayPath": "invalid"}, "RayPath")]
)
def test_levelset_template_reports_invalid_configuration(options, key):
    with pytest.raises(ValueError, match=key):
        GeneralShapeTemplate().levelset_template({"Name": "invalid", "Object": None, **options})


def test_levelset_template_rejects_degenerate_surface_measure():
    template = GeneralShapeTemplate()
    template.objects = SimpleNamespace(
        mesh=trimesh.Trimesh([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0, 1, 2]], process=False)
    )
    with pytest.raises(GeometryError, match="non-positive"):
        template.calculate_surface_parameter()
