import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZER = (
    REPO_ROOT
    / "research"
    / "LSMPM"
    / "scripts"
    / "analyze_v2_alignment_convergence_retry16.py"
)
CASES = tuple(f"ellipsoid_{path}" for path in ("wall", "rs", "ss"))


def write_level(root: Path, spacing: float, normalized_rms: float) -> None:
    for case in CASES:
        case_dir = root / case
        case_dir.mkdir(parents=True)
        (case_dir / "config.json").write_text(
            json.dumps({"physical_grid_spacing": spacing}), encoding="utf-8"
        )
        (case_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "history_max_surface_sdf_rms_over_grid_spacing": normalized_rms,
                    "contact_max_surface_sdf_error_over_grid_spacing": 0.5,
                    "soft_surface_levelset_alignment": {
                        "area_weighted_rms_phi_over_grid_spacing": normalized_rms
                    },
                }
            ),
            encoding="utf-8",
        )
        (case_dir / "formal_verification.json").write_text(
            json.dumps({"production_eligible": True}), encoding="utf-8"
        )


def test_retry16_alignment_convergence_uses_absolute_area_weighted_rms(tmp_path):
    coarse = tmp_path / "coarse"
    fine = tmp_path / "fine"
    write_level(coarse, spacing=0.004, normalized_rms=0.6)
    write_level(fine, spacing=0.003, normalized_rms=0.7)
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"

    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--coarse",
            str(coarse),
            "--fine",
            str(fine),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert len(summary["rows"]) == 3
    assert all(row["absolute_rms_decreased"] for row in summary["rows"])
