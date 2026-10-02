from pathlib import Path

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem.boundaries import NeumannBoundary
from src.fem.elements import create_element
from src.fem.generator import FEMGenerateManager
from src.physics_model.consititutive_model.finite_strain import (
    ClothARAP,
)
from src.fem.MaterialManager import FEMMaterialManager


def create_material(model, **parameters):
    return FEMMaterialManager().material_handle(model, **parameters)
from src.fem.engines.FEMSolver import FEMSolver
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.fem.cloth import create_cloth_element
from src.fem import FEM


pytestmark = [pytest.mark.unit, pytest.mark.fem, pytest.mark.geometry, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    yield
    ti.reset()


@pytest.mark.parametrize("element_type,cell_count,quadrature_count", [("TET4", 6, 1), ("HEX8", 1, 8)])
def test_box_reference_map_and_volume(element_type, cell_count, quadrature_count):
    mesh = FEMGenerateManager().create_box(size=(2.0, 3.0, 4.0), divisions=(1, 1, 1), element_type=element_type)
    element = create_element(mesh)

    assert mesh.number_of_cells == cell_count
    assert element.quadrature_count == quadrature_count
    np.testing.assert_allclose(np.sum(element.reference_weights), 24.0, rtol=2.0e-15)
    expected = np.repeat(np.eye(3)[None, None, :, :], mesh.number_of_cells * quadrature_count, axis=0).reshape(
        mesh.number_of_cells, quadrature_count, 3, 3
    )
    np.testing.assert_allclose(element.deformation_gradients(mesh.points), expected, atol=2.0e-15)


def test_tet4_device_edge_matrix_gradient_matches_reference_operator():
    mesh = FEMGenerateManager().create_box(
        size=(2.0, 3.0, 4.0),
        divisions=(1, 1, 1),
        element_type="TET4",
    )
    element = create_element(mesh)
    material = create_material(
        "StVK",
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1.0,
    )
    assembler = ClassicalAssembler(mesh, element, material)
    assert not hasattr(assembler, "cell_hessian")
    deformation = np.array(
        [[1.1, 0.2, -0.1], [0.05, 0.9, 0.15], [0.0, -0.08, 1.2]]
    )
    current = mesh.points @ deformation.T + (0.4, -0.3, 0.7)

    device = assembler.deformation_gradients(current)
    reference = element.deformation_gradients(current)

    assert assembler.is_tetrahedron
    np.testing.assert_allclose(device, reference, atol=2.0e-15)
    np.testing.assert_allclose(
        device,
        np.broadcast_to(deformation, device.shape),
        atol=2.0e-15,
    )


def test_fem_rest_shape_precomputes_initial_deformation_gradient():
    fem = FEM(log=False)
    fem.set_configuration(
        dimension=3, solver_type="Explicit", backend="taichi"
    )
    current_mesh = fem.create_mesh(
        "box", size=(1.0, 1.0, 1.0), element_type="HEX8"
    )
    current_shape = current_mesh.points.copy()
    rest_shape = current_shape.copy()
    rest_shape[:, 0] *= 0.5
    mesh = fem.add_mesh(current_mesh, rest_shape=rest_shape)
    fem.add_material(
        "StVK",
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=2.0,
    )
    fem.set_solver(step=0)

    engine = fem.build()

    expected = np.diag((2.0, 1.0, 1.0))
    np.testing.assert_allclose(mesh.rest_shape, rest_shape)
    np.testing.assert_allclose(engine.positions, current_shape)
    np.testing.assert_allclose(engine.displacement, 0.0)
    np.testing.assert_allclose(
        engine.initial_deformation_gradients,
        np.broadcast_to(
            expected, engine.initial_deformation_gradients.shape
        ),
        atol=2.0e-15,
    )
    assert np.sum(engine.element.reference_weights) == pytest.approx(0.5)
    assert np.sum(engine.mass) == pytest.approx(1.0)


def test_fem_rest_shape_defaults_to_current_shape():
    mesh = FEMGenerateManager().create_box(
        size=(1.0, 2.0, 3.0), element_type="TET4"
    )
    material = create_material(
        "StVK", young_modulus=1.0e4, poisson_ratio=0.3
    )
    solver = FEMSolver(mesh, material)

    np.testing.assert_allclose(mesh.rest_shape, mesh.points)
    np.testing.assert_allclose(
        solver.initial_deformation_gradients,
        np.broadcast_to(
            np.eye(3), solver.initial_deformation_gradients.shape
        ),
        atol=2.0e-15,
    )


def test_fem_rest_shape_drives_classical_and_cloth_membranes():
    mesh = FEMGenerateManager().create_rectangle(
        size=(2.0, 1.0), divisions=(1, 1)
    )
    rest_shape = mesh.points.copy()
    rest_shape[:, 0] *= 0.5
    mesh.set_rest_shape(
        rest_shape, update_material_coordinates=True
    )

    classical = create_element(
        mesh, thickness=0.1, formulation="classical"
    )
    cloth = create_cloth_element(
        mesh, ClothARAP(10.0, thickness=0.1)
    )

    classical_stretches = np.linalg.svd(
        classical.deformation_gradients(mesh.points)[:, 0],
        compute_uv=False,
    )
    np.testing.assert_allclose(
        classical_stretches,
        np.broadcast_to((2.0, 1.0), classical_stretches.shape),
        atol=2.0e-15,
    )
    cloth_stretches = np.linalg.svd(
        cloth.compute_deformation_gradient(mesh.points),
        compute_uv=False,
    )
    np.testing.assert_allclose(
        cloth_stretches,
        np.broadcast_to((2.0, 1.0), cloth_stretches.shape),
        atol=2.0e-15,
    )


def test_triangle_membrane_uses_constant_edge_matrix_gradient():
    mesh = FEMGenerateManager().create_rectangle(size=(2.0, 1.0), divisions=(2, 1))
    mesh.material_points[:, :2] *= (0.7, 1.4)
    element = create_element(mesh, thickness=0.25)
    affine = np.asarray(((1.2, 0.1, 0.0), (-0.2, 0.9, 0.0), (0.3, -0.1, 1.0)))
    current = mesh.points @ affine.T

    deformation_gradient = element.deformation_gradients(current)
    reference_gradient = element.deformation_gradients(mesh.points)
    expected = np.empty_like(deformation_gradient)
    natural_gradient = np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    for element_id, cell in enumerate(mesh.cells):
        reference_frame = element._triangle_frame(mesh.points[cell])
        current_frame = element._triangle_frame(current[cell])
        reference_jacobian = (mesh.points[cell] @ reference_frame[:2].T).T @ natural_gradient
        current_jacobian = (current[cell] @ current_frame[:2].T).T @ natural_gradient
        expected[element_id, 0] = current_jacobian @ np.linalg.inv(reference_jacobian)

    assert deformation_gradient.shape[-2:] == (2, 2)
    assert element.formulation == "classical_membrane"
    np.testing.assert_allclose(
        reference_gradient,
        np.broadcast_to(np.eye(2), reference_gradient.shape),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(deformation_gradient, expected, atol=2.0e-15)
    np.testing.assert_allclose(np.sum(element.reference_weights), 0.5, atol=2.0e-15)


def test_axisymmetric_tri3_revolves_measure_and_uses_3d_no_swirl_map():
    mesh = FEMGenerateManager().create_rectangle(
        size=(1.0, 0.2),
        divisions=(1, 1),
        origin=(0.5, 0.0, 0.0),
    )
    element = create_element(mesh, axisymmetric=True, axis_offset=0.0)

    expected_volume = np.pi * (1.5**2 - 0.5**2) * 0.2
    assert element.formulation == "axisymmetric_solid"
    assert element.constitutive_dimension == 3
    assert np.sum(element.reference_weights) == pytest.approx(
        expected_volume, rel=2.0e-15
    )
    np.testing.assert_allclose(
        element.deformation_gradients(mesh.points),
        np.broadcast_to(
            np.eye(3),
            (mesh.number_of_cells, 1, 3, 3),
        ),
        atol=2.0e-15,
    )

    current = mesh.points.copy()
    current[:, 0] *= 1.2
    current[:, 1] *= 0.8
    expected = np.diag((1.2, 0.8, 1.2))
    np.testing.assert_allclose(
        element.deformation_gradients(current),
        np.broadcast_to(expected, (mesh.number_of_cells, 1, 3, 3)),
        atol=2.0e-15,
    )

    material = create_material(
        "StVK",
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1.0,
    )
    assembler = ClassicalAssembler(
        mesh,
        element,
        material,
        project_pd=True,
        assemble_type="HashTriplet",
        linear_solver="PCG",
    )
    energy, force, stiffness, _ = assembler.assemble(
        current, need_stiffness=True
    )
    matrix = stiffness.toarray()
    assert energy > 0.0
    assert np.all(np.isfinite(force))
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=2.0e-11)

    exact_assembler = ClassicalAssembler(
        mesh,
        element,
        material,
        project_pd=False,
        assemble_type="HashTriplet",
        linear_solver="PCG",
    )
    exact_energy, exact_force, exact_stiffness, _ = exact_assembler.assemble(
        current, need_stiffness=True
    )
    direction = np.zeros_like(current)
    direction[:, :2] = np.array(
        [[0.13, -0.09], [-0.04, 0.11], [0.07, 0.05], [-0.08, -0.03]]
    )
    epsilon = 1.0e-6
    plus_energy, plus_force, _, _ = exact_assembler.assemble(
        current + epsilon * direction, need_stiffness=False
    )
    minus_energy, minus_force, _, _ = exact_assembler.assemble(
        current - epsilon * direction, need_stiffness=False
    )
    finite_energy_derivative = (plus_energy - minus_energy) / (2.0 * epsilon)
    analytic_energy_derivative = np.sum(exact_force * direction)
    assert analytic_energy_derivative == pytest.approx(
        finite_energy_derivative, rel=2.0e-7, abs=2.0e-7
    )
    finite_force_derivative = (plus_force - minus_force) / (2.0 * epsilon)
    analytic_force_derivative = (
        exact_stiffness.toarray() @ direction.reshape(-1)
    ).reshape(direction.shape)
    np.testing.assert_allclose(
        analytic_force_derivative,
        finite_force_derivative,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert exact_energy == pytest.approx(energy)

    collapse_direction = ti.Vector.field(3, ti.f64, shape=mesh.number_of_nodes)
    direction_values = np.zeros_like(current)
    direction_values[:, 0] = -2.0 * current[:, 0]
    collapse_direction.from_numpy(direction_values)
    assert assembler.maximum_material_step_device(
        assembler.positions, collapse_direction, safety=0.9
    ) == pytest.approx((1.0 - np.sqrt(0.1)) / 2.0, abs=2.0e-12)

    traction = NeumannBoundary().add_traction(
        (0.0, 1.0, 0.0),
        selector=lambda centroid: np.isclose(centroid[:, 1], 0.2),
    )
    boundary_force = traction.force(
        mesh, axisymmetric=True, axis_offset=0.0
    )
    np.testing.assert_allclose(
        np.sum(boundary_force, axis=0),
        (0.0, np.pi * (1.5**2 - 0.5**2), 0.0),
        rtol=2.0e-15,
        atol=2.0e-15,
    )

    solver = FEMSolver(
        mesh,
        material,
        dimension=2,
        axisymmetric=True,
        axis_offset=0.0,
    )
    constrained = solver.state.constrained.to_numpy().reshape(-1, 3)
    assert np.all(constrained[:, 2] == 1)


def test_classical_tri3_local_jacobian_removes_rigid_body_rotation():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 0.6), divisions=(2, 1))
    angle = 0.63
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (0.8 * np.sin(angle), 0.8 * np.cos(angle), -0.6),
            (0.6 * np.sin(angle), 0.6 * np.cos(angle), 0.8),
        )
    )
    current = mesh.points @ rotation.T + (0.3, -0.2, 0.7)
    material = create_material(
        "StVK",
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        thickness=0.02,
    )
    solver = FEMSolver(mesh, material)

    deformation_gradient = solver.element.deformation_gradients(current)
    assembler = ClassicalAssembler(mesh, solver.element, material)
    energy, force, _, _ = assembler.assemble(current)

    np.testing.assert_allclose(
        deformation_gradient,
        np.broadcast_to(np.eye(2), deformation_gradient.shape),
        atol=2.0e-15,
    )
    assert energy == pytest.approx(0.0, abs=1.0e-24)
    np.testing.assert_allclose(force, 0.0, atol=1.0e-12)


def test_named_sets_and_boundary_traction_integrate_total_force():
    mesh = FEMGenerateManager().create_box(size=(2.0, 3.0, 4.0), element_type="HEX8")
    boundary = NeumannBoundary().add_traction(
        (5.0, 0.0, 0.0), selector=lambda centroid: np.isclose(centroid[:, 0], 2.0)
    )

    assert mesh.select_nodes(node_set="xmin").size == 4
    force = boundary.force(mesh)
    np.testing.assert_allclose(np.sum(force, axis=0), (60.0, 0.0, 0.0), atol=1.0e-13)


def test_transient_pressure_reuses_reference_boundary_geometry(monkeypatch):
    mesh = FEMGenerateManager().create_box(
        size=(2.0, 3.0, 4.0), element_type="HEX8"
    )
    boundary = NeumannBoundary().add_pressure(
        lambda time, coordinates: 2.0 * time,
        selector=lambda centroid: np.isclose(centroid[:, 0], 2.0),
    )
    original = boundary._facet_geometry
    geometry_calls = 0

    def counted_geometry(mesh_value, facets, owners):
        nonlocal geometry_calls
        geometry_calls += 1
        return original(mesh_value, facets, owners)

    monkeypatch.setattr(boundary, "_facet_geometry", counted_geometry)
    first = boundary.force(mesh, time=0.5)
    second = boundary.force(mesh, time=1.0)

    assert geometry_calls == 1
    np.testing.assert_allclose(np.sum(first, axis=0), (12.0, 0.0, 0.0))
    np.testing.assert_allclose(np.sum(second, axis=0), (24.0, 0.0, 0.0))


def test_circle_and_cylinder_parameters_create_valid_meshes():
    generator = FEMGenerateManager()
    circle = generator.create_circle(radius=2.0, radial_divisions=2, circumferential_divisions=12)
    cylinder = generator.create_cylinder(
        radius=1.0,
        height=2.0,
        radial_divisions=1,
        circumferential_divisions=8,
        height_divisions=1,
    )

    assert circle.node_sets["rim"].size == 12
    assert cylinder.node_sets["top"].size == 9
    assert create_element(circle).minimum_jacobian_ratio(circle.points) > 0.999999
    assert create_element(cylinder).minimum_jacobian_ratio(cylinder.points) > 0.999999


def test_obj_quad_is_imported_as_two_tri3_cells():
    filename = Path(__file__).resolve().parents[2] / "data" / "fem_quad.obj"
    mesh = FEMGenerateManager().read(filename)

    assert mesh.cell_type == "triangle"
    assert mesh.number_of_nodes == 4
    assert mesh.number_of_cells == 2


def test_membrane_step_limit_stops_before_zero_area_state():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    element = create_element(mesh)
    direction = np.zeros_like(mesh.points)
    direction[:, 0] = -2.0 * mesh.points[:, 0]

    maximum_step = element.maximum_admissible_step(mesh.points, direction)

    assert 0.4 < maximum_step < 0.5
    assert element.minimum_jacobian_ratio(mesh.points + maximum_step * direction) > 0.0


def test_cloth_reference_operators_use_material_coordinates():
    mesh = FEMGenerateManager().create_rectangle(size=(2.0, 1.0), divisions=(1, 1))
    mesh.material_points[:, 0] *= 0.5
    element = create_cloth_element(mesh, ClothARAP(10.0, thickness=0.2))
    deformation_gradient = element.compute_deformation_gradient(mesh.points)
    deformation_metric = element.compute_metric_tensor(mesh.points)

    right_cauchy_green = np.einsum("eji,ejk->eik", deformation_gradient, deformation_gradient)
    # The convected metric A @ inv(A0) is generally non-symmetric, but is
    # similar to F.T @ F and therefore has the same two invariants.
    np.testing.assert_allclose(
        np.trace(deformation_metric, axis1=1, axis2=2), np.trace(right_cauchy_green, axis1=1, axis2=2)
    )
    np.testing.assert_allclose(np.linalg.det(deformation_metric), np.linalg.det(right_cauchy_green))
    np.testing.assert_allclose(np.linalg.svd(deformation_gradient[0], compute_uv=False), (2.0, 1.0), atol=2.0e-15)
    np.testing.assert_allclose(np.sum(element.integration_weight), 0.2, atol=2.0e-15)


def test_quadratic_bending_builds_one_psd_operator_for_two_triangles():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    material = ClothARAP(10.0, thickness=0.1, bending_stiffness=30.0)
    element = create_cloth_element(mesh, material)

    assert element.bending_connectivity.shape == (1, 4)
    assert np.min(np.linalg.eigvalsh(element.bending_stiffness[0])) > -1.0e-14
    element_positions = mesh.points[element.bending_connectivity[0]]
    np.testing.assert_allclose(
        element.bending_stiffness[0] @ element_positions, 0.0, atol=1.0e-14
    )


def test_obj_texture_coordinates_become_cloth_material_coordinates(tmp_path):
    filename = tmp_path / "textured_quad.obj"
    filename.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 2 0 0",
                "v 2 1 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "f 1/1 2/2 3/3 4/4",
            )
        ),
        encoding="utf8",
    )

    mesh = FEMGenerateManager().read(filename)

    assert mesh.number_of_nodes == 4
    np.testing.assert_allclose(np.ptp(mesh.points, axis=0), (2.0, 1.0, 0.0))
    np.testing.assert_allclose(np.ptp(mesh.material_points, axis=0), (1.0, 1.0, 0.0))


def test_obj_uv_seam_keeps_shared_physical_cloth_node(tmp_path):
    filename = tmp_path / "uv_seam.obj"
    filename.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "vt 0.25 0",
                "f 1/1 2/2 3/3",
                "f 1/5 3/3 4/4",
            )
        ),
        encoding="utf8",
    )

    mesh = FEMGenerateManager().read(filename)

    assert mesh.number_of_nodes == 4
    assert mesh.material_points.shape[0] == 5
    assert mesh.cells[0, 0] == mesh.cells[1, 0]
    assert mesh.material_cells[0, 0] != mesh.material_cells[1, 0]
