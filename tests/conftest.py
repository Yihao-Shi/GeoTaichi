"""Shared pytest taxonomy and isolated Taichi runtime for GeoTaichi."""

import gc
import os
from pathlib import Path
import re
import tempfile

import pytest

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(
        tempfile.gettempdir(),
        f"geotaichi-matplotlib-{os.getpid()}",
    ),
)

from tests.helpers.dimension_isolation import (  # noqa: E402
    configure_isolated_dimension_from_environment,
    pytest_pyfunc_call,
)


# Set the ordinary test process to the canonical 2D configuration, or apply a
# dimension-isolated child's override, before pytest imports test modules and
# their Taichi annotations.
configure_isolated_dimension_from_environment()


_GEOTAICHI_ENVIRONMENT_PREFIXES = ("GEOTAICHI_", "GT_")


_TOKEN_MARKERS = {
    "material": ("materials",),
    "constitutive": ("materials",),
    "contact": ("contact",),
    "friction": ("contact",),
    "barrier": ("contact",),
    "ipc": ("ipc", "contact"),
    "assembly": ("assembly",),
    "hash_triplet": ("assembly", "hash_triplet"),
    "hashtriplet": ("assembly", "hash_triplet"),
    "hash_assembly": ("assembly", "hash_triplet"),
    # Hash reduction is also used to reduce scalar COO entries.  It is not,
    # by itself, evidence that the block-structured BuildTriplet backend ran.
    "hash_reduction": ("assembly",),
    "triplet": ("assembly",),
    "matrix_free": ("assembly", "matrix_free"),
    "matrixfree": ("assembly", "matrix_free"),
    "coordinate_sparse": ("assembly", "coo"),
    "coo": ("assembly", "coo"),
    "linear_solver": ("linear_solver",),
    "precondition": ("linear_solver",),
    "topology": ("geometry",),
    "geometry": ("geometry",),
    "primitive": ("geometry",),
    "nurbs": ("geometry",),
    "coupling": ("coupling",),
    "field_io": ("io",),
    "output": ("io",),
    "recorder": ("io",),
    "vtu": ("io",),
    "gallery": ("gallery",),
}

_LAYER_DIRECTORIES = {
    "unit": "unit",
    "integration": "integration",
    "verification": "verification",
    "regression": "regression",
    "benchmarks": "benchmark",
}

_IPC_COMPATIBILITY_STEM_PREFIXES = (
    "test_igampm_coupling_barrier",
    "test_igampm_coupling_friction",
)

_IPC_COMPATIBILITY_STEMS = {
    "test_fully_implicit_friction_taichi_blocks",
    "test_igampm_contact_aware_solver",
    "test_igampm_lagged_friction_fixed_point",
}

_HASH_TRIPLET_PATHS = {
    "tests/integration/igampm/test_igampm_hash_assembly.py",
    "tests/integration/igampm/test_igampm_ipc_hash_contact.py",
    "tests/unit/linear_solver/test_buildtriplet_block_jacobi.py",
    "tests/unit/linear_solver/test_buildtriplet_dim4.py",
    "tests/unit/linear_solver/test_taichi_block_pattern_cache.py",
}

# Backend inference for mixed-purpose files is keyed by both path and base
# node name. Global node-name sets can classify an unrelated test with the
# same name, while a parameter ID such as ``[HashTriplet]`` is only a display
# label and is not evidence about the implementation under test.
_HASH_TRIPLET_NODE_PATHS = {
    # IGA-MPM contact-aware assembly.
    (
        "tests/integration/igampm/test_igampm_contact_aware_solver.py",
        "test_default_monolithic_solve_routes_to_taichi_cuda_backend",
    ),
    (
        "tests/integration/igampm/test_igampm_contact_aware_solver.py",
        "test_monolithic_dirichlet_elimination_keeps_symmetry",
    ),
    (
        "tests/integration/igampm/test_igampm_contact_aware_solver.py",
        "test_nonzero_dirichlet_values_are_newton_corrections_to_absolute_targets",
    ),
    (
        "tests/integration/igampm/test_igampm_contact_aware_solver.py",
        "test_taichi_monolithic_assembly_merges_blocks_and_eliminates_dirichlet",
    ),
    # Direct implicit MPM.
    (
        "tests/integration/mpm/test_direct_mpm_backend.py",
        "test_direct_implicit_mpm_cuda_hash_solve_stays_in_taichi_fields",
    ),
    (
        "tests/integration/mpm/test_direct_mpm_backend.py",
        "test_direct_implicit_mpm_non_cuda_keeps_scipy_fallback",
    ),
    (
        "tests/integration/mpm/test_direct_mpm_backend.py",
        "test_direct_ul_tl_fixed_stiffness_stencil_is_exact_and_reuses_mapping",
    ),
    # Fully implicit direct-MPM IPC.
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_fully_implicit_point_plane_production_jacobian",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_residual_only_probe_preserves_all_gpu_triplet_buffers",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_cuda_monolithic_soft_particle_matrix_matches_cpu_oracle",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_lagged_material_tangent_is_projected_before_global_scatter",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_cuda_lagged_monolithic_matrix_is_spd_and_uses_pcg",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_surface_measure_scales_barrier_and_friction_once",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_production_stribeck_jacobian_in_transition",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_complete_fully_implicit_residual_jacobian_by_finite_difference",
    ),
    (
        "tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py",
        "test_fully_implicit_point_plane_newton_smoke",
    ),
    # Direct soft-particle contact.
    (
        "tests/integration/mpm/test_ipc_soft_soft_friction_assembly.py",
        "test_ipc_soft_soft_friction_production_assembly",
    ),
    (
        "tests/integration/mpm/test_ipc_soft_soft_friction_assembly.py",
        "test_direct_mpm_particle_barrier_projection_is_lagged_only",
    ),
    (
        "tests/integration/mpm/test_ipc_soft_soft_friction_assembly.py",
        "test_soft_particle_contact_compaction_has_stable_device_raw_slots",
    ),
    (
        "tests/integration/mpm/test_ipc_soft_soft_friction_assembly.py",
        "test_ground_contact_compaction_is_surface_wall_deterministic",
    ),
    # SoftAffine mixed MPM-affine assembly.
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_residual_probe_does_not_reset_gpu_triplets",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_fully_implicit_triplet_capacity_covers_full_mixed_pt",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_lagged_triplet_capacity_covers_two_symmetric_stencils",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_triplet_safety_rejects_unsafe_values",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_matrix_assembly_reports_triplet_overflow",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_cuda_rejects_host_hash_reduction",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_lagged_requires_official_projected_spd_pcg",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_cuda_hot_loop_has_no_nonlinear_vector_numpy_roundtrip",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_merit_jacobian_removes_solver_diagonal_shift",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_device_newton_vectors_and_jp_stay_in_taichi_fields",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_production_bilateral_pp_kernels_and_frozen_cache",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_fi_barrier_retains_exact_negative_curvature",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_soft_fully_implicit_friction_full_jacobian_matches_fd",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_mixed_fully_implicit_pt_scattered_full_jacobian_matches_fd",
    ),
    (
        "tests/unit/mpdem/test_ipc_soft_affine_official_flow.py",
        "test_soft_affine_production_mixed_pt_uses_frozen_bilateral_stencil",
    ),
    # Affine-body fully implicit assembly.
    (
        "tests/unit/dem/contact/test_ipc_affine_fully_implicit_friction.py",
        "test_affine_cuda_nonlinear_hot_loop_has_no_full_vector_numpy_roundtrip",
    ),
    (
        "tests/unit/dem/contact/test_ipc_affine_fully_implicit_friction.py",
        "test_affine_fully_implicit_selects_nonsymmetric_linear_backends",
    ),
    (
        "tests/unit/dem/contact/test_ipc_affine_fully_implicit_friction.py",
        "test_affine_hash_capacity_bounds_raw_contact_scatter",
    ),
    (
        "tests/unit/dem/contact/test_ipc_affine_fully_implicit_friction.py",
        "test_affine_hash_overflow_is_reported_before_linear_solve",
    ),
    # Fully implicit IGA-MPM assembly.
    (
        "tests/verification/ipc/test_igampm_fully_implicit_friction.py",
        "test_exact_point_nurbs_fully_implicit_cuda_path_requires_full_bicgstab_state",
    ),
    (
        "tests/verification/ipc/test_igampm_fully_implicit_friction.py",
        "test_production_iga_mpm_friction_all_columns_match_force_fd",
    ),
    (
        "tests/verification/ipc/test_igampm_fully_implicit_friction.py",
        "test_production_monolithic_residual_probe_preserves_gpu_matrices",
    ),
    (
        "tests/verification/ipc/test_igampm_fully_implicit_friction.py",
        "test_production_3d_surface_fully_implicit_assembly_matches_all_columns_fd",
    ),
    # Mixed linear-solver contract module.
    (
        "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
        "test_hash_triplet_pcg_uses_same_true_residual_contract",
    ),
    (
        "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
        "test_hash_triplet_pcg_verifies_recursive_residual",
    ),
    (
        "tests/unit/iga/solver/test_iga_backend.py",
        "test_iga_hash_bicgstab_stays_in_taichi_solver",
    ),
}

_COO_NODE_PATHS = {
    (
        "tests/unit/dem/contact/test_ipc_affine_friction_assembly.py",
        "test_ipc_affine_affine_friction_production_assembly",
    ),
    (
        "tests/unit/dem/contact/test_ipc_affine_friction_fixed_point.py",
        "test_affine_direct_solve_restores_erased_inertia_floor",
    ),
    (
        "tests/unit/dem/contact/test_ipc_affine_fully_implicit_friction.py",
        "test_affine_merit_slope_removes_explicit_jacobian_shift",
    ),
    (
        "tests/unit/dem/contact/test_ipc_affine_fully_implicit_friction.py",
        "test_affine_fully_implicit_selects_nonsymmetric_linear_backends",
    ),
    (
        "tests/unit/iga/solver/test_iga_backend.py",
        "test_iga_backend_solves_with_coo_taichi_pcg",
    ),
    (
        "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
        "test_coordinate_sparse_propagates_pcg_failure_state",
    ),
}

_MATRIX_FREE_NODE_PATHS = {
    (
        "tests/integration/mpm/test_solid_engine_kernel_dispatch.py",
        "test_solid_engine_grid_velocity_dispatch_smoke",
    ),
    (
        "tests/unit/linear_solver/test_matrix_free_bicgstab_and_csr.py",
        "test_matrix_free_bicgstab_matches_dense_oracle",
    ),
    (
        "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
        "test_pcg_uses_true_residual_for_stiff_jacobi_system",
    ),
    (
        "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
        "test_pcg_verifies_recursive_residual_before_reporting_convergence",
    ),
    (
        "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
        "test_bicgstab_nonzero_initial_guess_uses_actual_shadow_residual",
    ),
    (
        "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
        "test_preconditioned_bicgstab_reports_true_last_state",
    ),
}

_MULTI_BACKEND_NODE_PATHS = {
    (
        "tests/integration/dem/test_affine_body.py",
        "test_affine_body_backends",
    ),
}

_MIXED_BACKEND_PATHS = {
    "tests/unit/linear_solver/test_matrix_free_bicgstab_and_csr.py",
    "tests/unit/linear_solver/test_matrix_free_krylov_contract.py",
}

_BACKEND_VALUE_MARKERS = {
    "matrixfree": ("assembly", "matrix_free"),
    "coo": ("assembly", "coo"),
    "hash": ("assembly", "hash_triplet"),
    "hashtriplet": ("assembly", "hash_triplet"),
}


def pytest_addoption(parser):
    group = parser.getgroup("geotaichi")
    group.addoption(
        "--taichi-arch",
        choices=("cpu", "gpu", "cuda", "metal", "vulkan"),
        default=os.environ.get("GEOTAICHI_TEST_ARCH", "cpu").lower(),
        help="backend used by tests that request the taichi_runtime fixture",
    )
    group.addoption(
        "--taichi-fp",
        choices=("f32", "f64"),
        default="f64",
        help="default floating-point type for the taichi_runtime fixture",
    )
    group.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="execute opt-in scalability benchmarks",
    )


@pytest.fixture
def taichi_runtime(request):
    """Create an isolated Taichi runtime for a new unit test.

    The fixture is opt-in because some integration tests own a complete
    GeoTaichi runtime lifecycle.  A test may override the command-line backend with
    ``@pytest.mark.parametrize("taichi_runtime", ["cuda"], indirect=True)``.
    """

    import taichi as ti

    repository_root = Path(str(request.config.rootpath)).resolve()
    try:
        caller_working_directory = Path.cwd()
    except FileNotFoundError:
        # A macOS Taichi backend may leave the process inside a temporary
        # compiler directory that disappears when the previous runtime is
        # reset.  Recover the shared test process before the next ti.init().
        os.chdir(repository_root)
        caller_working_directory = repository_root

    arch_name = getattr(request, "param", None)
    if isinstance(arch_name, dict):
        fp_name = arch_name.get("fp", request.config.getoption("--taichi-fp"))
        arch_name = arch_name.get("arch", request.config.getoption("--taichi-arch"))
    else:
        fp_name = request.config.getoption("--taichi-fp")
        arch_name = arch_name or request.config.getoption("--taichi-arch")

    ti.reset()
    try:
        ti.init(
            arch=getattr(ti, str(arch_name)),
            default_fp=getattr(ti, str(fp_name)),
            cpu_max_num_threads=1,
            offline_cache=False,
        )
    except Exception as error:
        ti.reset()
        if arch_name != "cpu":
            pytest.skip(f"Taichi {arch_name} backend is unavailable: {error}")
        raise

    try:
        yield ti
    finally:
        try:
            ti.sync()
        finally:
            ti.reset()
            gc.collect()
            try:
                os.chdir(caller_working_directory)
            except FileNotFoundError:
                os.chdir(repository_root)


def _module_scalar_state(module):
    """Return mutable scalar configuration without copying implementation data."""

    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_") and (value is None or isinstance(value, (bool, int, float, str)))
    }


def _snapshot_geotaichi_configuration():
    """Capture process-global switches changed by GeoTaichi solver setup."""

    import src.iga.config as iga_config
    import src.igampm.config as igampm_config
    import src.mpm.config as mpm_config
    import src.utils.GlobalVariable as global_variable

    modules = (
        global_variable,
        iga_config,
        mpm_config,
        igampm_config,
    )
    return {
        "modules": tuple((module, _module_scalar_state(module)) for module in modules),
        "environment": {
            name: value for name, value in os.environ.items() if name.startswith(_GEOTAICHI_ENVIRONMENT_PREFIXES)
        },
    }


def _restore_geotaichi_configuration(snapshot):
    """Restore a snapshot without invoking coupled dimension setters.

    ``src.igampm.config.set_dimension`` intentionally updates the standalone
    IGA and direct-MPM modules as a production convenience. Direct assignment
    is required here so a test that deliberately gives the three modules
    different values cannot leak any of them into the next test.
    """

    for module, state in snapshot["modules"]:
        for name, value in state.items():
            setattr(module, name, value)

    saved_environment = snapshot["environment"]
    current_names = tuple(name for name in os.environ if name.startswith(_GEOTAICHI_ENVIRONMENT_PREFIXES))
    for name in current_names:
        if name not in saved_environment:
            os.environ.pop(name, None)
    os.environ.update(saved_environment)


@pytest.fixture(scope="module", autouse=True)
def _isolated_taichi_test_module():
    """Keep compiled kernels and Taichi fields inside one test module.

    Some numerical modules intentionally amortize compilation through a
    module-scoped fixture. Resetting at the module boundary preserves that
    contract while preventing a kernel specialized for one module's static
    dimension or feature flags from being reused by the next module.
    """

    import taichi as ti

    configuration = _snapshot_geotaichi_configuration()
    ti.reset()
    try:
        yield
    finally:
        try:
            runtime = ti.lang.impl.get_runtime()
            if runtime.prog is not None:
                ti.sync()
        finally:
            ti.reset()
            _restore_geotaichi_configuration(configuration)
            gc.collect()


@pytest.fixture(autouse=True)
def _isolated_geotaichi_configuration():
    """Restore mutable solver configuration and test environment per test."""

    snapshot = _snapshot_geotaichi_configuration()
    try:
        yield
    finally:
        _restore_geotaichi_configuration(snapshot)


def _relative_test_path(item, rootpath):
    path = Path(str(item.path))
    try:
        return path.resolve().relative_to(Path(str(rootpath)).resolve())
    except ValueError:
        return path


def _lexical_tokens(value):
    return {token for token in re.split(r"[^a-z0-9]+", str(value).lower()) if token}


def _normalized_backend_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _inferred_marker_names(relative_path, node_name="", parameters=None):
    """Infer stable taxonomy markers without importing a test module.

    Directory and file names provide broad layer/domain markers.  Node names
    and the explicit ``assemble_type`` parameter distinguish tests that cover
    only one assembly backend.  The latter is important for parametrized
    MatrixFree/COO/HashTriplet integration tests: marking the whole module
    would make every backend partition select all three parameter values.
    """

    relative = Path(str(relative_path))
    parts = tuple(part.lower() for part in relative.parts)
    stem = relative.stem.lower()
    normalized_node_name = str(node_name).split("[", 1)[0].lower()
    relative_key = relative.as_posix().lower()
    searchable = "_".join((*parts, normalized_node_name))
    lexical_tokens = _lexical_tokens(searchable)
    markers = set()

    if len(parts) >= 2 and parts[0] == "tests" and parts[1] in _LAYER_DIRECTORIES:
        markers.add(_LAYER_DIRECTORIES[parts[1]])

    # Taxonomy tests describe other nodes in their own names and parameter
    # IDs. Inferring from those meta descriptions selects the taxonomy suite
    # itself into the physics/backend partitions that it audits.
    if parts[:3] == ("tests", "unit", "testing"):
        return frozenset(markers)

    # Stable directory names and lexical filename tokens provide the primary
    # solver/domain tags.  Lexical matching lets ``explicit_dem_contact`` mean
    # DEM without treating arbitrary substrings as solver names.
    if "ipc" in parts or "ipc" in lexical_tokens:
        markers.update(("ipc", "contact"))
    if "linear_solver" in parts:
        markers.add("linear_solver")

    for domain in ("dem", "mpm", "iga", "pinn", "fem"):
        if domain in parts or domain in lexical_tokens:
            markers.add(domain)
    if "mpdem" in parts or "mpdem" in lexical_tokens:
        markers.update(("mpdem", "mpm", "dem", "coupling"))
    if "fedem" in parts or "fedem" in lexical_tokens:
        markers.update(("fedem", "fem", "dem", "coupling"))
    if "fempm" in parts or "fempm" in lexical_tokens:
        markers.update(("fempm", "fem", "mpm", "coupling"))
    if "lsm" in parts or "lsm" in lexical_tokens or "levelset" in parts:
        markers.add("lsm")
    if "lsmpm" in parts or "lsmpm" in lexical_tokens:
        markers.update(("lsm", "mpm"))

    # Coupled solvers normally appear in filenames rather than directories.
    if "igampm" in searchable or "iga_mpm" in searchable:
        markers.update(("igampm", "iga", "mpm", "coupling"))

    for token, marker_names in _TOKEN_MARKERS.items():
        # ``coo`` is short enough that substring matching also matches words
        # such as ``coordinates``. The explicit ``coordinate_sparse`` token
        # above retains the intended compound-name classification.
        if token == "coo":
            matched = token in lexical_tokens
        else:
            matched = token in searchable
        # These modules contain matrix-free, COO, HashTriplet, and CSR tests,
        # so their filenames cannot classify every node as matrix-free.
        if relative_key in _MIXED_BACKEND_PATHS and token in {"matrix_free", "matrixfree"}:
            matched = False
        if matched:
            markers.update(marker_names)

    # These migrated IGA-MPM files exercise IPC even though their historical
    # filenames say only barrier/friction/contact-aware.
    if stem in _IPC_COMPATIBILITY_STEMS or stem.startswith(_IPC_COMPATIBILITY_STEM_PREFIXES):
        markers.update(("ipc", "contact"))

    # BuildTriplet is the block HashTriplet representation.  Keep this list
    # explicit so scalar CoordinateSparseMatrix hash reduction remains COO.
    if relative_key in _HASH_TRIPLET_PATHS or stem.startswith(_IPC_COMPATIBILITY_STEM_PREFIXES):
        markers.update(("assembly", "hash_triplet"))

    node_path = (relative_key, normalized_node_name)
    if node_path in _MULTI_BACKEND_NODE_PATHS:
        markers.update(("assembly", "matrix_free", "coo", "hash_triplet"))
    if node_path in _HASH_TRIPLET_NODE_PATHS:
        markers.update(("assembly", "hash_triplet"))
    if node_path in _COO_NODE_PATHS:
        markers.update(("assembly", "coo"))
    if node_path in _MATRIX_FREE_NODE_PATHS:
        markers.update(("assembly", "matrix_free"))

    parameters = parameters or {}
    for parameter_name in ("assemble_type", "assembly_type"):
        if parameter_name not in parameters:
            continue
        backend = _normalized_backend_name(parameters[parameter_name])
        markers.update(_BACKEND_VALUE_MARKERS.get(backend, ()))

    return frozenset(markers)


def _add_markers(item, marker_names):
    present = {marker.name for marker in item.iter_markers()}
    for marker_name in marker_names:
        if marker_name not in present:
            item.add_marker(getattr(pytest.mark, marker_name))
            present.add(marker_name)


def pytest_collection_modifyitems(config, items):
    """Attach layer/domain/solver markers from stable path conventions.

    Assembly-representation markers may come from an unambiguous path, node,
    or parameter value. Execution-device markers are intentionally never
    inferred from a filename: several legacy ``*_gpu.py`` regressions run on
    CPU unless an environment variable selects CUDA. New tests must mark their
    actual runtime explicitly.
    """

    run_benchmarks = config.getoption("--run-benchmarks")
    benchmark_skip = pytest.mark.skip(reason="pass --run-benchmarks to execute scalability benchmarks")

    for item in items:
        relative = _relative_test_path(item, config.rootpath)
        callspec = getattr(item, "callspec", None)
        parameters = getattr(callspec, "params", None)
        _add_markers(
            item,
            _inferred_marker_names(
                relative,
                node_name=item.name,
                parameters=parameters,
            ),
        )

        if item.get_closest_marker("benchmark") and not run_benchmarks:
            item.add_marker(benchmark_skip)
