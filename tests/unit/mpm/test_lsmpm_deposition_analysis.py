import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "research" / "LSMPM" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "analyze_v4_deposition_matrix",
    SCRIPT_DIR / "analyze_v4_deposition_matrix.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_retry16_deposition_modulus_matrix_and_case_names():
    assert MODULE.PARTICLE_COUNT == 600
    assert MODULE.YOUNG == (1.0e4, 1.0e5, 5.0e5)
    assert MODULE.REPLICATE_KEY == (0.50, 1.0e5, "seed2")
    assert MODULE.case_name(0.25, 1.0e4) == "soft_f025_E1e4"
    assert MODULE.case_name(0.50, 1.0e5) == "soft_f050_E1e5"
    assert MODULE.case_name(0.75, 5.0e5) == "soft_f075_E5e5"


def test_retry16_deposition_runner_matches_analysis_moduli():
    source = (SCRIPT_DIR / "run_v4_deposition_matrix.sh").read_text(encoding="utf-8")
    assert "for young in 1.0e4 1.0e5 5.0e5" in source
    assert "pilot_f075_E1e4 0.75 1.0e4 0.12" in source
    assert "soft_f050_E1e5_seed2 0.50 1.0e5" in source
    assert "readonly BODY_COUNT=600" in source
    assert 'ADVECTION_SCHEME="${GT_SOFT_LEVELSET_ADVECTION_SCHEME:-WENO5}"' in source
    assert 'soft_soft_contact_potential_path") == "work_conjugate_history"' in source
    assert '--body-count "$BODY_COUNT"' in source
    assert "readonly SDF_DEFORMATION_PADDING_CELLS=4" in source
    assert "readonly SDF_ADVECTION_INTERVAL=5" in source
    assert "readonly MAX_TERMINAL_ALIGNMENT_GRID_CELLS=2.0" in source
    assert "PRESERVE_%s_FAILED_COMPLETE" in source
    for stale_name in ("E2e5", "E1e6", "E5e6"):
        assert stale_name not in source


def test_retry21_queues_the_complete_n600_deposition_protocol():
    source = (SCRIPT_DIR / "run_gpu_server_float64_queue_retry21.sh").read_text(encoding="utf-8")
    expected_stages = (
        "deposition_n600_capacity_preflight",
        "deposition_n600_capacity_formal",
        "deposition_n600_representative_preflight",
        "deposition_n600_representative_formal",
        "deposition_n600_matrix_preflight",
        "deposition_n600_matrix_formal",
        "deposition_n600_overlap_audit",
    )
    assert 'RUN_DEPOSITION="${GT_RETRY21_RUN_DEPOSITION:-1}"' in source
    assert "20260904_3080ti_f64_retry21_deposition_sdf_fix3_40k" in source
    assert "GT_SOFT_LEVELSET_ADVECTION_SCHEME:-WENO5" in source
    assert "GT_SOFT_LEVELSET_ADVECTION_CFL:-0.20" in source
    assert 'CONFINING_PRESSURE="${GT_V5_CONFINING_PRESSURE:-40000}"' in source
    assert '--confining-pressure "$CONFINING_PRESSURE"' in source
    assert all(stage in source for stage in expected_stages)
    assert "particle_count=600 audited_cases=%d" in source
    assert "triaxial_n500_capacity_preflight" in source
    assert "v5_validation_manifest" in source
    assert '"${TRIAXIAL_MANIFEST_ARGS[@]}"' in source
    assert '"$(stage_result bootstrap)" == PASS' in source


def test_n600_deposition_is_required_by_downstream_protocols():
    capacity_source = (SCRIPT_DIR / "build_v5_capacity_protocol.py").read_text(encoding="utf-8")
    manifest_source = (SCRIPT_DIR / "build_v5_ch4_validation_manifest.py").read_text(encoding="utf-8")
    assert 'particle_count_per_case", -1)) == 600' in capacity_source
    assert 'particle_count_per_case", -1)) == 600' in manifest_source
    assert '"deposition_particle_count": 600' in manifest_source
    assert '"confining_pressure": args.confining_pressure' in manifest_source


def test_triaxial_production_uses_requested_soft_fraction_matrix():
    source = (SCRIPT_DIR / "run_gpu_v5_compact_production_n8000.sh").read_text(encoding="utf-8")
    requested_rigid_fractions = (
        "run_case 1.000 1000",
        "run_case 0.800 0800",
        "run_case 0.500 0500",
        "run_case 0.200 0200",
        "run_case 0.000 0000",
    )
    assert all(case in source for case in requested_rigid_fractions)
    assert "--expected-count 5" in source
    assert 'fraction_tag" == "0000"' in source


def _source_phase(body_count, soft_fraction, phase_seed):
    soft_count = int(round(soft_fraction * body_count))
    ranking = np.random.default_rng(phase_seed + 7919).permutation(body_count)
    phase = np.zeros(body_count, dtype=np.uint8)
    phase[ranking[:soft_count]] = 1
    return phase


def test_deposition_packing_restores_explicit_source_body_order():
    source_centers = np.arange(18, dtype=np.float64).reshape(6, 3)
    source_phase = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    source_body_id = np.argsort(source_phase, kind="stable")
    packing = {
        "centers": source_centers[source_body_id],
        "phase": source_phase[source_body_id],
        "source_body_id": source_body_id,
    }

    canonical = MODULE.canonicalize_packing(packing, {"body_count": 6})

    np.testing.assert_array_equal(canonical["centers"], source_centers)
    np.testing.assert_array_equal(canonical["phase"], source_phase)
    np.testing.assert_array_equal(canonical["source_body_id"], np.arange(6))


def test_deposition_packing_reconstructs_legacy_stable_phase_partition():
    body_count = 8
    soft_fraction = 0.5
    phase_seed = 20260715
    source_centers = np.arange(24, dtype=np.float64).reshape(body_count, 3)
    source_phase = _source_phase(body_count, soft_fraction, phase_seed)
    body_order = np.argsort(source_phase, kind="stable")
    legacy_packing = {
        "centers": source_centers[body_order],
        "phase": source_phase[body_order],
    }
    config = {
        "body_count": body_count,
        "soft_fraction": soft_fraction,
        "phase_seed_effective": phase_seed,
    }

    canonical = MODULE.canonicalize_packing(legacy_packing, config)

    np.testing.assert_array_equal(canonical["centers"], source_centers)
    np.testing.assert_array_equal(canonical["phase"], source_phase)
