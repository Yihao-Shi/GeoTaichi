import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_COMMON = (
    REPO_ROOT / "research" / "LSMPM" / "scripts" / "validation_common.py"
)


def load_validation_common():
    spec = importlib.util.spec_from_file_location(
        "lsmpm_validation_common", VALIDATION_COMMON
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_surface_levelset_alignment_policy_is_layered_by_physical_role():
    policy = load_validation_common()

    assert policy.SOFT_SURFACE_LEVELSET_CONTACT_GRID_CELLS == 1.0
    assert policy.SOFT_SURFACE_LEVELSET_TERMINAL_GRID_CELLS == 1.25
    assert policy.SOFT_SURFACE_LEVELSET_HISTORY_GRID_CELLS == 1.5
    assert policy.SOFT_SURFACE_LEVELSET_ASSEMBLY_GRID_CELLS == 1.5
    assert policy.SOFT_SURFACE_LEVELSET_RMS_GRID_CELLS == 1.0
    assert (
        policy.SOFT_SURFACE_LEVELSET_CONTACT_GRID_CELLS
        <= policy.SOFT_SURFACE_LEVELSET_TERMINAL_GRID_CELLS
        <= policy.SOFT_SURFACE_LEVELSET_HISTORY_GRID_CELLS
    )
    assert (
        policy.SOFT_SURFACE_LEVELSET_RMS_GRID_CELLS
        <= policy.SOFT_SURFACE_LEVELSET_HISTORY_GRID_CELLS
    )
