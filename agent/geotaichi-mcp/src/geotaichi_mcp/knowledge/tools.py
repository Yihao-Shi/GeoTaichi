"""Read-only MCP business functions for model knowledge and inspection."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..core.config import find_repo_root, resolve_user_path
from ..core.contracts import build_error, build_ok
from ..core.resources import load_capability_index, load_physics_validation_rubric, load_template
from .documentation import audit_helper_document, browse_capabilities, query_capabilities
from .inspection import inspect_model
from .physics_validation import score_physics_validation


def _load_json_file(path: str) -> Dict[str, Any]:
    repo = find_repo_root(required=False)
    resolved = resolve_user_path(path, repo, must_exist=True)
    with resolved.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("JSON input must contain an object: %s" % resolved)
    return document


def geotaichi_browse_capabilities(path: str = "") -> Dict[str, Any]:
    """Browse a known facade/method/key/example path in the generated GeoTaichi capability tree."""
    try:
        return browse_capabilities(path or None)
    except Exception as exc:
        return build_error("capability_browse_failed", str(exc))


def geotaichi_query_capabilities(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search facade methods, configuration keys, and examples, returning paths suitable for browse."""
    try:
        return query_capabilities(query, limit)
    except Exception as exc:
        return build_error("capability_query_failed", str(exc))


def geotaichi_get_model_template(module: str) -> Dict[str, Any]:
    """Return the canonical MPM, DEM, MPDEM, CFDEM, IGA, or IGAMPM lifecycle template."""
    try:
        content, path = load_template(module)
        return build_ok({"module": module.lower(), "source": str(path), "content": content})
    except (FileNotFoundError, ValueError) as exc:
        return build_error("template_not_found", str(exc))
    except Exception as exc:
        return build_error("template_load_failed", str(exc))


def geotaichi_inspect_model(script_path: str, contract_path: str = "") -> Dict[str, Any]:
    """Statically inspect a model's imports, facade lifecycle, indexed methods, keys, and optional contract."""
    try:
        repo = find_repo_root(required=False)
        script = resolve_user_path(script_path, repo, must_exist=True)
        contract = None
        if contract_path:
            contract_file = resolve_user_path(contract_path, repo, must_exist=True)
            with contract_file.open("r", encoding="utf-8") as stream:
                contract = json.load(stream)
        result = inspect_model(script, load_capability_index(repo), contract)
        if not result["valid"]:
            return build_error(
                "model_inspection_failed",
                "The model has static lifecycle or API errors.",
                result,
            )
        return build_ok(result)
    except FileNotFoundError as exc:
        return build_error("input_not_found", str(exc))
    except SyntaxError as exc:
        return build_error("python_syntax_error", str(exc), {"line": exc.lineno, "offset": exc.offset})
    except Exception as exc:
        return build_error("model_inspection_error", str(exc))


def geotaichi_score_physics(contract_path: str, evidence_path: str = "") -> Dict[str, Any]:
    """Score run evidence against the model contract and solver-specific physical validation rubric."""
    try:
        repo = find_repo_root(required=False)
        contract = _load_json_file(contract_path)
        evidence = _load_json_file(evidence_path) if evidence_path else {}
        return build_ok(score_physics_validation(contract, evidence, load_physics_validation_rubric(repo)))
    except FileNotFoundError as exc:
        return build_error("validation_input_not_found", str(exc))
    except (ValueError, json.JSONDecodeError) as exc:
        return build_error("invalid_validation_input", str(exc))
    except Exception as exc:
        return build_error("physics_scoring_failed", str(exc))


def geotaichi_review_model(script_path: str, contract_path: str, evidence_path: str = "") -> Dict[str, Any]:
    """Run the static and physical review stages and return the next bounded agent action."""
    try:
        repo = find_repo_root(required=False)
        script = resolve_user_path(script_path, repo, must_exist=True)
        contract = _load_json_file(contract_path)
        evidence = _load_json_file(evidence_path) if evidence_path else {}
        inspection = inspect_model(script, load_capability_index(repo), contract)
        physics = score_physics_validation(contract, evidence, load_physics_validation_rubric(repo))

        if not inspection["valid"]:
            stage = "repair_static_model"
            next_actions = ["Resolve static lifecycle/API errors before executing the model."]
        elif physics["blockers"]:
            stage = "resolve_model_contract"
            next_actions = ["Resolve model-contract placeholders and physical choices before execution."]
        elif physics["decision"] == "insufficient_evidence":
            stage = "collect_physics_evidence"
            next_actions = ["Run a reduced case and collect every missing physical validation item."]
        elif physics["decision"] == "reject":
            stage = "diagnose_physics_failure"
            next_actions = ["Classify and repair the first hard or failed physics check, then rerun the same rubric."]
        elif physics["decision"] == "accept_reduced":
            stage = "run_production_validation"
            next_actions = ["Repeat unchanged acceptance checks at production backend, resolution, and duration."]
        else:
            stage = "prepare_handoff"
            next_actions = ["Freeze artifacts, evidence ledger, effective runtime, and the validated handoff."]
        next_actions.extend(action for action in physics["next_actions"] if action not in next_actions)
        return build_ok(
            {
                "stage": stage,
                "ready_for_execution": inspection["valid"] and not physics["blockers"],
                "ready_for_handoff": stage == "prepare_handoff",
                "inspection": inspection,
                "physics_score": physics,
                "next_actions": next_actions,
                "repair_policy": {
                    "maximum_iterations": 3,
                    "preserve_physics": True,
                    "forbidden_shortcuts": [
                        "widen acceptance tolerance without a derivation",
                        "change material/contact law to make a run pass",
                        "hide NaN, capacity overflow, or non-convergence",
                    ],
                },
            }
        )
    except FileNotFoundError as exc:
        return build_error("review_input_not_found", str(exc))
    except SyntaxError as exc:
        return build_error("python_syntax_error", str(exc), {"line": exc.lineno, "offset": exc.offset})
    except (ValueError, json.JSONDecodeError) as exc:
        return build_error("invalid_review_input", str(exc))
    except Exception as exc:
        return build_error("model_review_failed", str(exc))


def geotaichi_audit_api_docs(categories: str = "", kind: str = "methods") -> Dict[str, Any]:
    """Audit indexed public methods or keys against docs/helper/geotaichi_api_reference.tex."""
    try:
        repo = find_repo_root()
        selected = [item.strip().lower() for item in categories.split(",") if item.strip()] or None
        docs_path = repo / "docs" / "helper" / "geotaichi_api_reference.tex"
        if not docs_path.is_file():
            return build_error("helper_doc_not_found", str(docs_path))
        return audit_helper_document(docs_path, selected, kind, repo)
    except Exception as exc:
        return build_error("api_audit_failed", str(exc))
