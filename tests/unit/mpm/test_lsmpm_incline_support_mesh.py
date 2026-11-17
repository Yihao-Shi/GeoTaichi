from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "research" / "LSMPM" / "scripts" / "run_v2_inclined_plane.py"
SPEC = importlib.util.spec_from_file_location("lsmpm_incline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_structured_support_mesh_is_watertight_and_spacing_bounded() -> None:
    size = np.asarray([1.10, 0.40, 0.04], dtype=np.float64)
    spacing = 0.0124
    mesh, intervals = MODULE.structured_box_surface_mesh(size, spacing)

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    np.testing.assert_allclose(mesh.volume, np.prod(size), rtol=0.0, atol=1.0e-12)
    assert mesh.vertices.shape[0] > 646
    assert np.all(size / intervals <= spacing * (1.0 + 1.0e-12))
    assert np.max(mesh.edges_unique_length) <= np.sqrt(2.0) * spacing * (
        1.0 + 1.0e-12
    )


def test_incline_accuracy_gate_depends_only_on_displacement() -> None:
    checks = {
        name: False for name in MODULE.INCLINE_INTEGRITY_CHECKS
    }
    checks.update(
        {
            "displacement_theory": True,
            "velocity_theory": False,
            "normal_force_balance": False,
            "tangential_force_balance": False,
            "release_quasistatic": False,
            "surface_levelset_alignment": False,
            "soft_volume_evolution": False,
        }
    )

    integrity, gating, diagnostic = MODULE.classify_incline_validation_checks(
        checks
    )

    assert gating == {"displacement_theory": True}
    assert all(value is False for value in integrity.values())
    assert diagnostic
    assert not any(diagnostic.values())
    assert all(gating.values())
