from pathlib import Path

from geotaichi_mcp.core.resources import load_physics_validation_rubric
from geotaichi_mcp.knowledge.physics_validation import score_physics_validation
from geotaichi_mcp.knowledge.tools import geotaichi_review_model, geotaichi_score_physics


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _contract():
    return {
        "schema_version": 1,
        "title": "FEM soft particle against rigid LSDEM body",
        "module": "fedem",
        "dimension": 3,
        "coordinate_assumption": "3d",
        "units": {"length": "m", "mass": "kg", "time": "s"},
        "physics": {"processes": ["frictionless contact"], "governing_assumptions": ["finite strain"]},
        "validation": {
            "observable": "maximum normal force",
            "expectation": 10.0,
            "tolerance": {"relative": 0.05},
            "evidence": "Hertz reference",
            "invariants": [
                {
                    "kind": "contact_gap",
                    "expectation": [-5.0e-5, 1.0e-3],
                    "tolerance": 0.0,
                    "basis": "contact resolution",
                },
                {
                    "kind": "action_reaction",
                    "expectation": 0.0,
                    "tolerance": {"absolute": 1.0e-6},
                    "basis": "float64 reduction scale",
                },
            ],
        },
        "unresolved": [],
    }


def _passing_evidence(production=True):
    return {
        "schema_version": 1,
        "task_status": "completed",
        "finite_state": True,
        "capacity_overflow": False,
        "solver_converged": "not_applicable",
        "timestep_consistent": True,
        "command": "python reduced_soft_rigid.py",
        "backend": "cuda",
        "precision": "float64",
        "production_parameters": production,
        "checks": [
            {
                "name": "peak force",
                "kind": "contract_observable",
                "observed": 10.2,
                "expected": 10.0,
                "tolerance": {"relative": 0.05},
                "evidence": "output/contact-force.csv",
            },
            {
                "name": "gap bound",
                "kind": "contact_gap",
                "observed": -2.0e-5,
                "expected": [-5.0e-5, 1.0e-3],
                "tolerance": 0.0,
                "evidence": "output/contact-gap.csv",
            },
            {
                "name": "exchange residual",
                "kind": "action_reaction",
                "observed": 2.0e-7,
                "expected": 0.0,
                "tolerance": {"absolute": 1.0e-6},
                "evidence": "output/exchange-balance.json",
            },
        ],
    }


def test_physics_scorer_accepts_complete_solver_specific_evidence():
    result = score_physics_validation(
        _contract(),
        _passing_evidence(),
        load_physics_validation_rubric(REPOSITORY_ROOT),
    )

    assert result["decision"] == "accept"
    assert result["score"] == 100.0
    assert not result["missing_evidence"]
    assert not result["hard_failures"]
    assert all(check["status"] == "pass" for check in result["checks"])


def test_physics_scorer_distinguishes_missing_evidence_from_failure():
    result = score_physics_validation(
        _contract(),
        {},
        load_physics_validation_rubric(REPOSITORY_ROOT),
    )

    assert result["decision"] == "insufficient_evidence"
    assert "contract_observable" in result["missing_evidence"]
    assert any(item.startswith("physical_invariant:contact_gap") for item in result["missing_evidence"])
    assert not result["hard_failures"]


def test_physics_scorer_hard_failure_cannot_be_offset_by_other_checks():
    evidence = _passing_evidence()
    evidence["finite_state"] = False
    result = score_physics_validation(
        _contract(),
        evidence,
        load_physics_validation_rubric(REPOSITORY_ROOT),
    )

    assert result["decision"] == "reject"
    assert result["score"] <= 39.0
    assert result["hard_failures"] == ["execution.finite_state=False"]


def test_physics_scorer_rejects_known_task_failure_even_when_checks_are_missing():
    result = score_physics_validation(
        _contract(),
        {"task_status": "failed"},
        load_physics_validation_rubric(REPOSITORY_ROOT),
    )

    assert result["decision"] == "reject"
    assert result["hard_failures"] == ["execution.task_status='failed'"]
    assert "contract_observable" in result["missing_evidence"]


def test_physics_scorer_reports_reduced_validation_separately():
    result = score_physics_validation(
        _contract(),
        _passing_evidence(production=False),
        load_physics_validation_rubric(REPOSITORY_ROOT),
    )

    assert result["decision"] == "accept_reduced"
    assert result["score"] == 100.0
    assert "intended backend" in result["next_actions"][0]


def test_physics_scorer_rejects_tolerance_changes_outside_the_contract():
    evidence = _passing_evidence()
    evidence["checks"][1]["tolerance"] = 1.0
    result = score_physics_validation(
        _contract(),
        evidence,
        load_physics_validation_rubric(REPOSITORY_ROOT),
    )

    assert result["decision"] == "reject"
    assert result["checks"][1]["status"] == "invalid"
    assert result["checks"][1]["reason"] == "tolerance differs from model contract"
    assert "check.gap bound:tolerance_mismatch" in result["hard_failures"]


def test_mcp_scoring_and_review_tools_drive_the_next_agent_stage(tmp_path):
    import json

    contract_path = tmp_path / "contract.json"
    evidence_path = tmp_path / "evidence.json"
    script_path = tmp_path / "model.py"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    evidence_path.write_text(json.dumps(_passing_evidence()), encoding="utf-8")
    script_path.write_text(
        """\
from geotaichi import DEM, FEM, FEDEM, init

init(dim=3, arch="cpu", log=False)
dem = DEM(log=False)
fem = FEM(log=False)
coupling = FEDEM(dem, fem, log=False)
coupling.run()
""",
        encoding="utf-8",
    )

    score = geotaichi_score_physics(str(contract_path), str(evidence_path))
    review = geotaichi_review_model(str(script_path), str(contract_path), str(evidence_path))

    assert score["ok"] and score["data"]["decision"] == "accept"
    assert review["ok"]
    assert review["data"]["stage"] == "prepare_handoff"
    assert review["data"]["ready_for_handoff"]
    assert review["data"]["repair_policy"]["maximum_iterations"] == 3
