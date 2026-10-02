import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import DirichletBoundary, FEM, NeumannBoundary
from src.fem.cloth.ClothAssembler import ClothAssembler
from src.fem.engines.FEMSolver import FEMSolver
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.fem.generator import FEMGenerateManager
from src.physics_model.consititutive_model.finite_strain import (
    ClothARAP,
    ClothNeoHookean,
)
from src.fem.MaterialManager import FEMMaterialManager


def create_material(model, **parameters):
    return FEMMaterialManager().material_handle(model, **parameters)


def signed_dihedral_angle(points):
    x0, x1, x2, x3 = points
    normal0 = np.cross(x1 - x0, x2 - x0)
    normal1 = np.cross(x2 - x3, x1 - x3)
    cosine = np.dot(normal0, normal1) / (np.linalg.norm(normal0) * np.linalg.norm(normal1))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return -angle if np.dot(np.cross(normal1, normal0), x1 - x2) < 0.0 else angle


pytestmark = [pytest.mark.unit, pytest.mark.fem, pytest.mark.assembly, pytest.mark.cpu, pytest.mark.serial]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


@pytest.mark.parametrize(
    "material",
    [
        ClothARAP(2.0e4, compression_stiffness=3.0e4, thickness=0.02),
        ClothNeoHookean(2.0e4, 0.25, thickness=0.02),
    ],
)
def test_cloth_assembler_matches_energy_and_force_differences(material):
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 0.5), divisions=(1, 1))
    mesh.material_points[:, 0] = 0.7 * mesh.points[:, 0] + 0.15 * mesh.points[:, 1]
    mesh.material_points[:, 1] = 0.8 * mesh.points[:, 1]
    positions = mesh.points.copy()
    positions[:, 0] *= 1.12
    positions[:, 1] *= 0.91
    positions[:, 2] = 0.08 * positions[:, 0] * positions[:, 1]
    assembler = ClothAssembler(mesh, material, project_pd=False)
    reference_solver = FEMSolver(mesh, material)

    assert reference_solver.element.formulation == "surface_cloth"
    assert reference_solver.element.deformation_gradients(positions).shape[-2:] == (3, 2)

    device_energy, device_force, device_stiffness, _ = assembler.assemble_system(positions, need_stiffness=True)
    assert np.isfinite(device_energy)
    np.testing.assert_allclose(device_stiffness.toarray(), device_stiffness.toarray().T, atol=2.0e-8)
    np.testing.assert_allclose(
        assembler.deformation_gradient.to_numpy(),
        reference_solver.element.deformation_gradients(positions)[:, 0],
        atol=2.0e-14,
    )
    direction = np.random.default_rng(81).normal(size=positions.shape)
    epsilon = 2.0e-7
    plus_energy = assembler.assemble_system(positions + epsilon * direction)[0]
    minus_energy = assembler.assemble_system(positions - epsilon * direction)[0]
    assert np.sum(device_force * direction) == pytest.approx(
        (plus_energy - minus_energy) / (2.0 * epsilon),
        rel=3.0e-5,
        abs=3.0e-5,
    )
    plus_force = assembler.assemble_system(positions + epsilon * direction)[1]
    minus_force = assembler.assemble_system(positions - epsilon * direction)[1]
    finite_difference = (plus_force - minus_force) / (2.0 * epsilon)
    tangent_product = (device_stiffness @ direction.reshape(-1)).reshape(positions.shape)
    np.testing.assert_allclose(tangent_product, finite_difference, rtol=3.0e-5, atol=3.0e-5)


def test_taichi_quadratic_bending_force_matches_energy_difference():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    material = ClothARAP(1.0e3, thickness=0.1, bending_stiffness=2.0e5)
    assembler = ClothAssembler(mesh, material, project_pd=False)
    positions = mesh.points.copy()
    positions[-1, 2] = 0.1
    energy, force, _, _ = assembler.assemble_system(positions)
    direction = np.zeros_like(positions)
    direction[-1, 2] = 1.0
    epsilon = 1.0e-6
    plus = assembler.assemble_system(positions + epsilon * direction)[0]
    minus = assembler.assemble_system(positions - epsilon * direction)[0]

    assert energy > 0.0
    assert np.sum(force * direction) == pytest.approx((plus - minus) / (2.0 * epsilon), rel=2.0e-7)


def test_taichi_dihedral_bending_has_analytic_force_and_hessian():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    material = ClothARAP(
        1.0e3,
        thickness=0.1,
        bending_stiffness=2.0e5,
        bending_model="Dihedral",
    )
    assembler = ClothAssembler(mesh, material, project_pd=False)
    positions = mesh.points.copy()
    positions[-1, 2] = 0.12
    direction = np.random.default_rng(812).normal(size=positions.shape)
    energy, force, stiffness, _ = assembler.assemble_system(positions, need_stiffness=True)
    hinge = assembler.element.bending_connectivity[0]
    angle = signed_dihedral_angle(positions[hinge])
    expected_bending_energy = (
        material.quadratic_bending_modulus
        * assembler.element.bending_edge_length[0]
        / assembler.element.bending_height[0]
        * (angle - assembler.element.bending_rest_angle[0]) ** 2
    )
    membrane_energy = np.sum(assembler.element_energy.to_numpy())
    epsilon = 2.0e-7
    plus_energy, plus_force, _, _ = assembler.assemble_system(positions + epsilon * direction)
    minus_energy, minus_force, _, _ = assembler.assemble_system(positions - epsilon * direction)

    assert assembler.bending_model == "Dihedral"
    assert energy > 0.0
    assert energy - membrane_energy == pytest.approx(expected_bending_energy, rel=2.0e-12, abs=2.0e-12)
    assert np.sum(force * direction) == pytest.approx(
        (plus_energy - minus_energy) / (2.0 * epsilon),
        rel=3.0e-5,
        abs=3.0e-6,
    )
    np.testing.assert_allclose(
        (stiffness @ direction.reshape(-1)).reshape(positions.shape),
        (plus_force - minus_force) / (2.0 * epsilon),
        rtol=3.0e-4,
        atol=3.0e-4,
    )


@pytest.mark.parametrize("assemble_type", ["COO", "HashTriplet"])
def test_dihedral_bending_project_pd_supports_sparse_backends(assemble_type):
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    material = ClothARAP(
        1.0e3,
        thickness=0.1,
        bending_stiffness=2.0e5,
        bending_model="Dihedral",
    )
    assembler = ClothAssembler(
        mesh,
        material,
        project_pd=True,
        assemble_type=assemble_type,
        linear_solver="PCG",
    )
    assert not hasattr(assembler, "element_stiffness")
    assert not hasattr(assembler, "bending_hessian")
    positions = mesh.points.copy()
    positions[-1, 2] = 0.12

    _, _, stiffness, _ = assembler.assemble_system(positions, need_stiffness=True)
    dense = stiffness.toarray()

    np.testing.assert_allclose(dense, dense.T, atol=2.0e-8)
    assert np.linalg.eigvalsh(dense).min() >= -2.0e-8


def test_cloth_can_project_membrane_without_approximating_bending():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    material = ClothARAP(1.0e3, thickness=0.1, bending_stiffness=2.0e5, bending_model="Dihedral")
    assembler = ClothAssembler(
        mesh,
        material,
        project_pd=True,
        project_bending_pd=False,
    )
    reference = ClothAssembler(mesh, material, project_pd=False)
    positions = mesh.points.copy()
    positions[-1, 2] = 0.12
    assembler.assemble_system(positions, need_stiffness=True)
    reference.assemble_system(positions, need_stiffness=True)

    assert assembler.project_pd
    assert not assembler.project_bending_pd
    np.testing.assert_allclose(assembler.bending_tangent.to_numpy(), reference.bending_tangent.to_numpy())


def test_cloth_reuses_sparse_matrix_between_newton_assemblies():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    assembler = ClothAssembler(mesh, ClothARAP(1.0e3, thickness=0.1))

    first = assembler.assemble_device(need_stiffness=True)[1]
    second = assembler.assemble_device(need_stiffness=True)[1]

    assert second is first


def test_stitch_spring_and_sdf_energies_have_device_analytic_tangents():
    mesh = FEMGenerateManager().create_rectangle(size=(1.0, 1.0), divisions=(1, 1))
    material = ClothARAP(1.0e3, thickness=0.1)
    energies = [
        {
            "type": "GarmentStitch",
            "stitches": [[0, 1, 3]],
            "ratios": [0.35],
            "stiffness": 8.0e3,
        },
        {
            "type": "Spring",
            "nodes": [2],
            "targets": mesh.points[[2]] + np.array([[0.0, 0.0, 0.04]]),
            "stiffness": 5.0e3,
        },
        {
            "type": "SDF",
            "nodes": [3],
            "targets": [[1.0, 1.0, 0.0]],
            "normals": [[0.0, 0.0, 1.0]],
            "stiffness": 3.0e3,
            "dhat": 0.08,
        },
    ]
    assembler = ClothAssembler(
        mesh,
        material,
        bending_model="None",
        cloth_energies=energies,
        project_pd=False,
    )
    positions = mesh.points.copy()
    positions[:, 0] *= 1.02
    positions[2, 2] = 0.01
    positions[3, 2] = 0.03
    direction = np.random.default_rng(813).normal(size=positions.shape)
    energy, force, stiffness, _ = assembler.assemble_system(positions, need_stiffness=True)
    node_area = assembler.energy_data.node_area
    stitch_difference = positions[0] - 0.65 * positions[1] - 0.35 * positions[3]
    spring_difference = positions[2] - energies[1]["targets"][0]
    sdf_distance = positions[3, 2]
    sdf_gap = sdf_distance / energies[2]["dhat"] - 1.0
    expected_optional_energy = (
        0.5 * 8.0e3 * node_area[0] * stitch_difference.dot(stitch_difference)
        + 0.5 * 5.0e3 * spring_difference.dot(spring_difference)
        - 3.0e3 * node_area[3] * 0.08 / 6.0 * sdf_gap**3
    )
    membrane_energy = np.sum(assembler.element_energy.to_numpy())
    epsilon = 2.0e-7
    plus_energy, plus_force, _, _ = assembler.assemble_system(positions + epsilon * direction)
    minus_energy, minus_force, _, _ = assembler.assemble_system(positions - epsilon * direction)

    assert energy > 0.0
    assert energy - membrane_energy == pytest.approx(expected_optional_energy, rel=2.0e-12, abs=2.0e-12)
    assert assembler.stitch_count == assembler.spring_count == assembler.sdf_count == 1
    assert np.sum(force * direction) == pytest.approx(
        (plus_energy - minus_energy) / (2.0 * epsilon),
        rel=4.0e-5,
        abs=4.0e-5,
    )
    np.testing.assert_allclose(
        (stiffness @ direction.reshape(-1)).reshape(positions.shape),
        (plus_force - minus_force) / (2.0 * epsilon),
        rtol=5.0e-5,
        atol=5.0e-5,
    )


def test_fem_facade_configures_dihedral_and_optional_cloth_energies():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 1.0), divisions=(1, 1))
    fem.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e3,
        thickness=0.1,
        bending_stiffness=2.0e5,
    )
    fem.set_bending_model("Dihedral")
    fem.add_stitch([[0, 1, 3]], stiffness=1.0e3, ratios=[0.5])
    fem.add_spring([2], stiffness=1.0e3, targets=mesh.points[[2]])
    fem.add_sdf(
        [3],
        stiffness=1.0e3,
        dhat=0.1,
        targets=mesh.points[[3]],
        normals=[[0.0, 0.0, 1.0]],
    )
    fem.set_solver(linear_solver="BiCGSTAB", project_pd=False)

    engine = fem.build()

    assert engine.cloth_assembler.bending_model == "Dihedral"
    assert engine.cloth_assembler.stitch_count == 1
    assert engine.cloth_assembler.spring_count == 1
    assert engine.cloth_assembler.sdf_count == 1


def test_cloth_material_automatically_uses_taichi_implicit_engine_and_line_search():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 0.5), divisions=(2, 1))
    fem.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e4,
        compression_stiffness=1.0e4,
        density=1.0,
        thickness=0.01,
    )
    fem.add_boundary_condition(
        dirichlet=DirichletBoundary().add("xmin", "all", 0.0),
        neumann=NeumannBoundary().add_nodal_force("xmax", (1.0, 0.0, 0.0), total=True),
    )
    fem.set_solver(
        quasi_static=True,
        dt=1.0,
        step=1,
        max_iterations=30,
        residual_tolerance=1.0e-8,
        line_search=True,
        project_pd=True,
    )

    engine = fem.build()
    result = fem.run(verbose=False)

    assert engine.backend == "taichi"
    assert type(engine).__name__ == "ClothImplicitFEM"
    assert result.converged
    assert np.mean(result.displacement[mesh.node_sets["xmax"], 0]) > 0.0
    assert all("line_search_step" in iteration for iteration in result.history[0]["iterations"])


def test_cloth_material_automatically_uses_taichi_explicit_engine():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_mesh(geometry="rectangle", size=(1.0, 0.5), divisions=(2, 1))
    fem.add_material(
        "ClothNeoHookean",
        young_modulus=1.0e4,
        poisson_ratio=0.25,
        density=1.0,
        thickness=0.01,
        bending_stiffness=10.0,
    )
    fem.add_boundary_condition(dirichlet=DirichletBoundary().add("xmin", "all", 0.0))
    fem.set_solver(dt=1.0e-5, step=1, gravity=(0.0, 0.0, -9.81))

    engine = fem.build()
    result = fem.run(verbose=False)

    assert engine.backend == "taichi"
    assert type(engine).__name__ == "ClothExplicitFEM"
    assert result.time == pytest.approx(1.0e-5)
    assert np.min(result.displacement[:, 2]) < 0.0


def test_cloth_rejects_volume_and_two_dimensional_solver_configuration():
    volume = FEM(log=False)
    volume.set_configuration(dimension=3, solver_type="Explicit")
    volume.add_mesh(geometry="box", element_type="TET4")
    volume.add_material("ClothARAP", stretch_stiffness=1.0e4)
    with pytest.raises(ValueError, match="TRI3"):
        volume.build()

    planar = FEM(log=False)
    planar.set_configuration(dimension=2, solver_type="Explicit")
    planar.add_mesh(geometry="rectangle")
    planar.add_material("ClothARAP", stretch_stiffness=1.0e4)
    with pytest.raises(ValueError, match="dimension=3"):
        planar.build()


@pytest.mark.parametrize("element_type", ["TRI3", "TET4", "HEX8"])
def test_taichi_classical_fem_matches_energy_gradient(element_type):
    generator = FEMGenerateManager()
    if element_type == "TRI3":
        mesh = generator.create_rectangle(size=(1.0, 0.5), divisions=(1, 1))
    else:
        mesh = generator.create_box(size=(1.0, 0.5, 0.4), divisions=(1, 1, 1), element_type=element_type)
    material = create_material(
        "StVK",
        young_modulus=2.0e4,
        poisson_ratio=0.25,
        thickness=0.02,
    )
    reference = FEMSolver(mesh, material)
    positions = mesh.points.copy()
    positions[:, 0] *= 1.03
    positions[:, 1] *= 0.98
    positions[:, 2] += 0.02 * positions[:, 0] * positions[:, 1]
    assembler = ClassicalAssembler(reference.mesh, reference.element, material, project_pd=False)

    device_energy, device_force, stiffness, _ = assembler.assemble(positions, need_stiffness=True)
    direction = np.random.default_rng(181).normal(size=positions.shape)
    epsilon = 2.0e-7
    plus_energy = assembler.assemble(positions + epsilon * direction)[0]
    minus_energy = assembler.assemble(positions - epsilon * direction)[0]
    assert np.sum(device_force * direction) == pytest.approx(
        (plus_energy - minus_energy) / (2.0 * epsilon),
        rel=4.0e-5,
        abs=4.0e-5,
    )
    plus_force = assembler.assemble(positions + epsilon * direction)[1]
    minus_force = assembler.assemble(positions - epsilon * direction)[1]
    np.testing.assert_allclose(
        (stiffness @ direction.reshape(-1)).reshape(positions.shape),
        (plus_force - minus_force) / (2.0 * epsilon),
        rtol=5.0e-5,
        atol=5.0e-5,
    )
    expected_dimension = 2 if element_type == "TRI3" else 3
    assert assembler.deformation_gradients(positions).shape[-2:] == (
        expected_dimension,
        expected_dimension,
    )


def test_standard_stvk_automatically_uses_taichi_backend_when_runtime_is_active():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit", backend="taichi")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 0.5), divisions=(2, 1))
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1.0, thickness=0.01)
    fem.add_boundary_condition(
        dirichlet=DirichletBoundary().add("xmin", "xy", 0.0).add("all", "z", 0.0),
        neumann=NeumannBoundary().add_nodal_force("xmax", (1.0, 0.0, 0.0), total=True),
    )
    fem.set_solver(quasi_static=True, dt=1.0, step=1, residual_tolerance=1.0e-7)

    engine = fem.build()
    result = fem.run(verbose=False)

    assert engine.backend == "taichi"
    assert type(engine).__name__ == "ClassicalImplicitFEM"
    assert result.converged
    assert np.mean(result.displacement[mesh.node_sets["xmax"], 0]) > 0.0


def test_standard_stvk_explicit_integrator_uses_taichi_assembly():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", backend="taichi")
    fem.add_mesh(geometry="rectangle", size=(1.0, 0.5), divisions=(2, 1))
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1.0, thickness=0.01)
    fem.add_boundary_condition(dirichlet=DirichletBoundary().add("xmin", "all", 0.0))
    fem.set_solver(dt=1.0e-5, step=1, gravity=(0.0, 0.0, -9.81))

    engine = fem.build()
    result = fem.run(verbose=False)

    assert engine.backend == "taichi"
    assert type(engine).__name__ == "ClassicalExplicitFEM"
    assert result.time == pytest.approx(1.0e-5)
    assert np.min(result.displacement[:, 2]) < 0.0
