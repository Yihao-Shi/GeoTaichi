"""Contracts for automatic markers and named test partitions."""

import json
import importlib.util
from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.required, pytest.mark.cpu]

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PARTITION_FILE = REPOSITORY_ROOT / "tests" / "testing" / "test_partitions.json"
PYPROJECT_FILE = REPOSITORY_ROOT / "pyproject.toml"


def _load_suite_configuration():
    """Load the root suite configuration without relying on ``conftest`` order.

    Pytest imports nested ``conftest.py`` files under the same conventional
    module name.  A plain ``import conftest`` can therefore resolve to the
    material-suite fixture module when the complete test tree is collected.
    Loading the root file under a private name makes this taxonomy oracle
    deterministic both on its own and as part of the full suite.
    """

    path = REPOSITORY_ROOT / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location(
        "_geotaichi_root_suite_configuration",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test suite configuration from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite_configuration = _load_suite_configuration()

EXPECTED_NAMED_MARKER_PARTITIONS = {
    "coupling": "coupling",
    "gallery": "gallery",
    "pinn": "pinn",
    "metal": "metal",
    "vulkan": "vulkan",
    "serial": "serial",
    "requires-network": "requires_network",
}

BACKEND_MARKERS = {"matrix_free", "coo", "hash_triplet"}


def _markers(path, node_name="", parameters=None):
    return set(
        suite_configuration._inferred_marker_names(
            path,
            node_name=node_name,
            parameters=parameters,
        )
    )


def _partition_data():
    return json.loads(PARTITION_FILE.read_text(encoding="utf-8"))


def _marker_atoms(expression):
    return {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
        if token not in {"and", "or", "not"}
    }


def test_partition_schema_paths_and_marker_registration_are_consistent():
    data = _partition_data()
    assert data["version"] == 2
    partitions = data["partitions"]
    pyproject = PYPROJECT_FILE.read_text(encoding="utf-8")

    for name, definition in partitions.items():
        assert definition.get("description"), name
        for relative_path in definition.get("paths", ()):
            assert (REPOSITORY_ROOT / relative_path).exists(), (
                name,
                relative_path,
            )
        for marker in _marker_atoms(definition.get("markers", "")):
            assert f'"{marker}:' in pyproject, (
                name,
                marker,
            )

    for partition_name, marker in EXPECTED_NAMED_MARKER_PARTITIONS.items():
        assert partitions[partition_name]["markers"] == marker

    assert partitions["all"]["paths"] == [
        "tests/unit",
        "tests/integration",
        "tests/verification",
        "tests/regression",
    ]
    assert "tests/benchmarks" not in partitions["all"]["paths"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "tests/integration/igampm/test_igampm_coupling_barrier.py",
            {"integration", "igampm", "iga", "mpm", "coupling", "contact", "ipc"},
        ),
        (
            "tests/integration/igampm/test_igampm_coupling_friction_2d.py",
            {"integration", "igampm", "iga", "mpm", "coupling", "contact", "ipc"},
        ),
        (
            "tests/unit/iga/contact/test_fully_implicit_friction_taichi_blocks.py",
            {"unit", "iga", "contact", "ipc"},
        ),
        (
            "tests/unit/iga/contact/test_igampm_lagged_friction_fixed_point.py",
            {"unit", "igampm", "iga", "mpm", "coupling", "contact", "ipc"},
        ),
    ],
)
def test_migrated_igampm_ipc_files_are_in_the_ipc_partition(path, expected):
    assert expected <= _markers(path)


def test_explicit_igampm_dem_contact_is_dem_but_not_ipc():
    markers = _markers(
        "tests/integration/igampm/test_igampm_explicit_dem_contact.py"
    )
    assert {"integration", "igampm", "iga", "mpm", "dem", "contact"} <= markers
    assert "ipc" not in markers


@pytest.mark.parametrize(
    "path",
    [
        "tests/unit/linear_solver/test_buildtriplet_block_jacobi.py",
        "tests/unit/linear_solver/test_buildtriplet_dim4.py",
        "tests/unit/linear_solver/test_taichi_block_pattern_cache.py",
        "tests/integration/igampm/test_igampm_hash_assembly.py",
        "tests/integration/igampm/test_igampm_ipc_hash_contact.py",
    ],
)
def test_block_hash_triplet_files_receive_assembly_backend_markers(path):
    assert {"assembly", "hash_triplet"} <= _markers(path)


def test_scalar_coordinate_sparse_hash_reduction_remains_coo_only():
    markers = _markers(
        "tests/unit/iga/assembly/test_coordinate_sparse_hash_reduction.py"
    )
    assert {"unit", "iga", "assembly", "coo"} <= markers
    assert "hash_triplet" not in markers


@pytest.mark.parametrize(
    ("assemble_type", "expected_backend"),
    [
        ("MatrixFree", "matrix_free"),
        ("COO", "coo"),
        ("HashTriplet", "hash_triplet"),
    ],
)
def test_parameterized_assembly_backend_marks_only_selected_value(
    assemble_type,
    expected_backend,
):
    markers = _markers(
        "tests/integration/mpm/test_sparse_grid_implicit_solid.py",
        node_name="test_sparse_grid_implicit_solid_3d_assemblies",
        parameters={"assemble_type": assemble_type},
    )
    assert "assembly" in markers
    assert markers & BACKEND_MARKERS == {expected_backend}


def test_affine_combined_backend_contract_selects_all_three_partitions():
    markers = _markers(
        "tests/integration/dem/test_affine_body.py",
        node_name="test_affine_body_backends",
    )
    assert {"assembly", *BACKEND_MARKERS} <= markers


def test_iga_backend_node_names_distinguish_coo_and_hash_triplet():
    path = "tests/unit/iga/solver/test_iga_backend.py"
    coo_markers = _markers(
        path,
        node_name="test_iga_backend_solves_with_coo_taichi_pcg",
    )
    hash_markers = _markers(
        path,
        node_name="test_iga_hash_bicgstab_stays_in_taichi_solver",
    )
    assert coo_markers & BACKEND_MARKERS == {"coo"}
    assert hash_markers & BACKEND_MARKERS == {"hash_triplet"}


def test_contact_aware_ipc_marks_only_assembly_nodes_as_hash_triplet():
    path = "tests/integration/igampm/test_igampm_contact_aware_solver.py"
    geometry_markers = _markers(
        path,
        node_name="test_conservative_point_nurbs_step_and_armijo_preserve_dmin",
    )
    assembly_markers = _markers(
        path,
        node_name=(
            "test_taichi_monolithic_assembly_merges_blocks_and_eliminates_dirichlet"
        ),
    )
    assert {"ipc", "contact"} <= geometry_markers
    assert "hash_triplet" not in geometry_markers
    assert {"ipc", "contact", "assembly", "hash_triplet"} <= assembly_markers


def test_parameter_ids_do_not_infer_backend_markers():
    markers = _markers(
        "tests/integration/mpm/test_sparse_grid_implicit_solid.py",
        node_name="test_unparameterized_contract[HashTriplet-COO-MatrixFree]",
    )
    assert markers & BACKEND_MARKERS == set()


def test_assemble_type_callspec_remains_the_parameterized_backend_oracle():
    markers = _markers(
        "tests/integration/mpm/test_sparse_grid_implicit_solid.py",
        node_name="test_unparameterized_contract[misleading-COO-label]",
        parameters={"assemble_type": "HashTriplet"},
    )
    assert markers & BACKEND_MARKERS == {"hash_triplet"}


def test_testing_meta_nodes_do_not_classify_themselves():
    markers = _markers(
        "tests/unit/testing/test_test_taxonomy.py",
        node_name=(
            "test_ipc_hash_triplet_matrix_free_coo_output[HashTriplet]"
        ),
        parameters={"assemble_type": "COO"},
    )
    assert markers == {"unit"}


def test_coo_uses_a_lexical_boundary_but_compound_mapping_is_preserved():
    coordinates = _markers(
        "tests/unit/mpm/test_thb_gallery_support.py",
        node_name="test_thb_boundary_selection_uses_refined_node_coordinates",
    )
    explicit_coo = _markers(
        "tests/unit/iga/solver/test_iga_backend.py",
        node_name="test_iga_backend_solves_with_coo_taichi_pcg",
    )
    coordinate_sparse = _markers(
        "tests/unit/iga/assembly/test_coordinate_sparse_hash_reduction.py"
    )

    assert "coo" not in coordinates
    assert "coo" in explicit_coo
    assert "coo" in coordinate_sparse


@pytest.mark.parametrize(
    ("path", "node_name", "expected_backends"),
    [
        (
            "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
            "test_fully_implicit_point_plane_production_jacobian",
            {"hash_triplet"},
        ),
        (
            "tests/integration/mpm/test_ipc_soft_soft_friction_assembly.py",
            "test_soft_particle_contact_compaction_has_stable_device_raw_slots",
            {"hash_triplet"},
        ),
        (
            "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
            "test_mixed_fully_implicit_pt_scattered_full_jacobian_matches_fd",
            {"hash_triplet"},
        ),
        (
            "tests/unit/dem/contact/test_ipc_affine_friction_assembly.py",
            "test_ipc_affine_affine_friction_production_assembly",
            {"coo"},
        ),
        (
            "tests/unit/dem/contact/test_ipc_affine_friction_fixed_point.py",
            "test_affine_direct_solve_restores_erased_inertia_floor",
            {"coo"},
        ),
        (
            "tests/unit/dem/contact/test_ipc_affine_fully_implicit_friction.py",
            "test_affine_fully_implicit_selects_nonsymmetric_linear_backends",
            {"coo", "hash_triplet"},
        ),
        (
            "tests/verification/ipc/test_igampm_fully_implicit_friction.py",
            "test_production_monolithic_residual_probe_preserves_gpu_matrices",
            {"hash_triplet"},
        ),
        (
            "tests/integration/mpm/test_solid_engine_kernel_dispatch.py",
            "test_solid_engine_grid_velocity_dispatch_smoke",
            {"matrix_free"},
        ),
        (
            "tests/unit/linear_solver/test_matrix_free_bicgstab_and_csr.py",
            "test_matrix_free_bicgstab_matches_dense_oracle",
            {"matrix_free"},
        ),
        (
            "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
            "test_coordinate_sparse_propagates_pcg_failure_state",
            {"coo"},
        ),
        (
            "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
            "test_hash_triplet_pcg_verifies_recursive_residual",
            {"hash_triplet"},
        ),
    ],
)
def test_mixed_files_use_exact_path_and_node_backend_mappings(
    path,
    node_name,
    expected_backends,
):
    markers = _markers(path, node_name=node_name)
    assert "assembly" in markers
    assert markers & BACKEND_MARKERS == expected_backends


def test_mixed_linear_solver_csr_node_is_not_matrix_free():
    markers = _markers(
        "tests/unit/linear_solver/test_matrix_free_bicgstab_and_csr.py",
        node_name="test_compressed_sparse_row_cpu_solve_matches_numpy",
    )
    assert markers & BACKEND_MARKERS == set()


def test_backend_node_name_does_not_match_on_a_different_path():
    markers = _markers(
        "tests/unit/geometry/test_unrelated_contract.py",
        node_name="test_fully_implicit_point_plane_production_jacobian",
    )
    assert markers & BACKEND_MARKERS == set()
