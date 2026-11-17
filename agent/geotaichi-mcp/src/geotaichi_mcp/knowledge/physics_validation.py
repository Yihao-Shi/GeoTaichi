"""Evidence-based physical validation scoring for model-agent decisions."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


PLACEHOLDER_FRAGMENTS = ("replace-me", "replace-with", "replace with")
EXECUTION_FIELDS = (
    ("task_status", "completed"),
    ("finite_state", True),
    ("capacity_overflow", False),
    ("solver_converged", (True, "not_applicable")),
    ("timestep_consistent", (True, "not_applicable")),
)
REPRODUCIBILITY_FIELDS = ("command", "backend", "precision", "production_parameters")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(fragment in lowered for fragment in PLACEHOLDER_FRAGMENTS)
    if isinstance(value, dict):
        return any(_has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_placeholder(item) for item in value)
    return False


def _tolerance_value(tolerance: Any, expected: float) -> Tuple[Optional[float], Optional[str]]:
    if _is_number(tolerance) and float(tolerance) >= 0.0:
        return float(tolerance), None
    if not isinstance(tolerance, dict):
        return None, "tolerance must be a non-negative number or an object with absolute/relative"
    absolute = tolerance.get("absolute")
    relative = tolerance.get("relative")
    bounds = []
    if _is_number(absolute) and float(absolute) >= 0.0:
        bounds.append(float(absolute))
    if _is_number(relative) and float(relative) >= 0.0:
        bounds.append(float(relative) * max(abs(expected), 1.0e-30))
    if not bounds:
        return None, "tolerance requires a non-negative absolute or relative value"
    return max(bounds), None


def _evaluate_numeric(observed: Any, expected: Any, tolerance: Any) -> Tuple[Optional[bool], Dict[str, Any]]:
    if not _is_number(observed):
        return None, {"reason": "observed must be a finite number"}
    observed_value = float(observed)
    if _is_number(expected):
        expected_value = float(expected)
        allowed, error = _tolerance_value(tolerance, expected_value)
        if error:
            return None, {"reason": error}
        residual = abs(observed_value - expected_value)
        return residual <= allowed, {"absolute_error": residual, "allowed_error": allowed}
    if isinstance(expected, (list, tuple)) and len(expected) == 2 and all(_is_number(item) for item in expected):
        lower, upper = float(expected[0]), float(expected[1])
        if lower > upper:
            return None, {"reason": "expected range lower bound exceeds upper bound"}
        if lower <= observed_value <= upper:
            distance = 0.0
        else:
            distance = min(abs(observed_value - lower), abs(observed_value - upper))
        return distance == 0.0, {"distance_to_range": distance, "expected_range": [lower, upper]}
    return None, {"reason": "expected must be a finite number or a two-value range"}


def evaluate_check(check: Any, known_kinds: set[str], index: int) -> Dict[str, Any]:
    """Evaluate one evidence check without trusting an unexplained pass flag."""
    if not isinstance(check, dict):
        return {
            "name": "check_%d" % index,
            "kind": "unknown",
            "status": "invalid",
            "required": True,
            "reason": "check must be an object",
        }
    name = str(check.get("name") or "check_%d" % index)
    kind = str(check.get("kind") or "unknown")
    required = check.get("required", True) is not False
    result: Dict[str, Any] = {"name": name, "kind": kind, "required": required}
    if kind not in known_kinds:
        result.update({"status": "invalid", "reason": "unknown physics check kind"})
        return result
    evidence = check.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        result.update({"status": "invalid", "reason": "a source, artifact, command, or derivation is required"})
        return result
    result["evidence"] = evidence.strip()

    if "observed" in check and "expected" in check:
        result.update(
            {
                "observed": check.get("observed"),
                "expected": check.get("expected"),
                "tolerance": check.get("tolerance"),
            }
        )
        passed, details = _evaluate_numeric(check.get("observed"), check.get("expected"), check.get("tolerance"))
        result.update(details)
        if passed is None:
            result["status"] = "invalid"
        else:
            result["status"] = "pass" if passed else "fail"
        return result

    status = check.get("status")
    if status in {"pass", "fail", "skip", "not_run"}:
        result["status"] = status
        return result
    if isinstance(check.get("passed"), bool):
        result["status"] = "pass" if check["passed"] else "fail"
        result["reported_result"] = True
        return result
    result.update({"status": "invalid", "reason": "provide numeric observed/expected/tolerance or an explicit status"})
    return result


def _bind_contract_criteria(
    contract: Dict[str, Any],
    raw_checks: List[Any],
    checks: List[Dict[str, Any]],
) -> List[str]:
    """Reject evidence criteria that differ from the pre-run model contract."""
    validation = contract.get("validation") if isinstance(contract.get("validation"), dict) else {}
    declarations = validation.get("invariants")
    declared_by_kind = {
        str(item.get("kind")): item for item in declarations or [] if isinstance(item, dict) and item.get("kind")
    }
    binding_errors = []
    for raw, check in zip(raw_checks, checks):
        if (
            not isinstance(raw, dict)
            or check["kind"] == "unknown"
            or check.get("reason") == "unknown physics check kind"
        ):
            continue
        declaration = None
        if check["kind"] == "contract_observable":
            declaration = {
                "expectation": validation.get("expectation"),
                "tolerance": validation.get("tolerance"),
            }
        else:
            declaration = declared_by_kind.get(check["kind"])
        if declaration is None:
            check.update({"status": "invalid", "reason": "criterion is not declared in model contract"})
            binding_errors.append("check.%s:undeclared_criterion" % check["name"])
            continue
        if raw.get("expected") != declaration.get("expectation"):
            check.update({"status": "invalid", "reason": "expected value differs from model contract"})
            binding_errors.append("check.%s:expected_mismatch" % check["name"])
            continue
        if raw.get("tolerance") != declaration.get("tolerance"):
            check.update({"status": "invalid", "reason": "tolerance differs from model contract"})
            binding_errors.append("check.%s:tolerance_mismatch" % check["name"])
    return binding_errors


def _execution_group(evidence: Dict[str, Any], weight: float) -> Dict[str, Any]:
    items = []
    for field, expected in EXECUTION_FIELDS:
        value = evidence.get(field)
        accepted = value in expected if isinstance(expected, tuple) else value == expected
        status = "pass" if accepted else ("missing" if value is None else "fail")
        items.append({"field": field, "value": value, "status": status})
    passed = sum(item["status"] == "pass" for item in items)
    return {"score": weight * passed / len(items), "weight": weight, "items": items}


def _observable_group(checks: List[Dict[str, Any]], weight: float) -> Dict[str, Any]:
    candidates = [check for check in checks if check["kind"] == "contract_observable"]
    passed = any(check["status"] == "pass" for check in candidates)
    return {
        "score": weight if passed else 0.0,
        "weight": weight,
        "status": "pass" if passed else ("fail" if candidates else "missing"),
        "checks": candidates,
    }


def _invariant_group(
    checks: List[Dict[str, Any]],
    required_sets: List[List[str]],
    weight: float,
) -> Dict[str, Any]:
    results = []
    for alternatives in required_sets:
        candidates = [check for check in checks if check["kind"] in alternatives]
        passed = any(check["status"] == "pass" for check in candidates)
        results.append(
            {
                "alternatives": alternatives,
                "status": "pass" if passed else ("fail" if candidates else "missing"),
                "checks": [check["name"] for check in candidates],
            }
        )
    passed_count = sum(result["status"] == "pass" for result in results)
    score = weight * passed_count / max(1, len(results))
    return {"score": score, "weight": weight, "requirements": results}


def _reproducibility_group(evidence: Dict[str, Any], weight: float) -> Dict[str, Any]:
    items = []
    for field in REPRODUCIBILITY_FIELDS:
        value = evidence.get(field)
        present = isinstance(value, bool) if field == "production_parameters" else value not in (None, "")
        items.append({"field": field, "status": "pass" if present else "missing", "value": value})
    score = weight * sum(item["status"] == "pass" for item in items) / len(items)
    return {"score": score, "weight": weight, "items": items}


def score_physics_validation(
    contract: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    rubric: Dict[str, Any],
) -> Dict[str, Any]:
    """Score physical evidence while keeping missing evidence distinct from failure."""
    evidence = evidence or {}
    known_kinds = set((rubric.get("check_kinds") or {}).keys())
    raw_checks = evidence.get("checks") or []
    if not isinstance(raw_checks, list):
        raw_checks = [raw_checks]
    checks = [evaluate_check(check, known_kinds, index) for index, check in enumerate(raw_checks, 1)]
    criterion_errors = _bind_contract_criteria(contract, raw_checks, checks)

    module = str(contract.get("module") or "").lower()
    module_policy = (rubric.get("module_policies") or {}).get(module, {})
    required_sets = module_policy.get("required_check_sets") or [[]]
    weights = {name: float(group["weight"]) for name, group in rubric["groups"].items()}
    groups = {
        "execution_integrity": _execution_group(evidence, weights["execution_integrity"]),
        "contract_observable": _observable_group(checks, weights["contract_observable"]),
        "physical_invariants": _invariant_group(checks, required_sets, weights["physical_invariants"]),
        "reproducibility": _reproducibility_group(evidence, weights["reproducibility"]),
    }
    raw_score = round(sum(group["score"] for group in groups.values()), 2)

    blockers = []
    if not module_policy:
        blockers.append("No physics validation policy exists for module %r." % module)
    if _has_placeholder(contract):
        blockers.append("The model contract still contains placeholder values.")
    validation = contract.get("validation") if isinstance(contract.get("validation"), dict) else {}
    if not isinstance(validation.get("invariants"), list) or not validation["invariants"]:
        blockers.append("The model contract does not declare machine-readable physical invariant criteria.")
    unresolved = contract.get("unresolved")
    if isinstance(unresolved, list) and unresolved:
        blockers.append("The model contract has unresolved physical choices.")

    hard_failures = []
    for item in groups["execution_integrity"]["items"]:
        if item["status"] == "fail":
            hard_failures.append("execution.%s=%r" % (item["field"], item["value"]))
    hard_failures.extend(
        "check.%s" % check["name"] for check in checks if check["required"] and check["status"] == "fail"
    )
    hard_failures.extend(criterion_errors)

    missing_evidence = []
    missing_evidence.extend(
        "execution.%s" % item["field"] for item in groups["execution_integrity"]["items"] if item["status"] == "missing"
    )
    if groups["contract_observable"]["status"] == "missing":
        missing_evidence.append("contract_observable")
    missing_evidence.extend(
        "physical_invariant:%s" % "|".join(requirement["alternatives"])
        for requirement in groups["physical_invariants"]["requirements"]
        if requirement["status"] == "missing"
    )
    missing_evidence.extend(
        "reproducibility.%s" % item["field"]
        for item in groups["reproducibility"]["items"]
        if item["status"] == "missing"
    )
    declarations = validation.get("invariants") or []
    for declaration in declarations:
        if not isinstance(declaration, dict) or declaration.get("required", True) is False:
            continue
        kind = str(declaration.get("kind") or "")
        if kind and not any(check["kind"] == kind for check in checks):
            missing_evidence.append("declared_invariant:%s" % kind)
    missing_evidence = list(dict.fromkeys(missing_evidence))

    threshold = float(rubric.get("decision_threshold", 80.0))
    score = min(raw_score, 39.0) if hard_failures else raw_score
    if hard_failures:
        decision = "reject"
    elif blockers or missing_evidence:
        decision = "insufficient_evidence"
    elif score < threshold:
        decision = "reject"
    elif evidence.get("production_parameters") is True:
        decision = "accept"
    else:
        decision = "accept_reduced"

    next_actions = []
    if blockers:
        next_actions.append("Resolve contract placeholders and physical choices before changing solver parameters.")
    if missing_evidence:
        next_actions.append("Collect the listed evidence with an artifact path, command, or derivation.")
    if hard_failures:
        next_actions.append("Diagnose the first hard failure; do not compensate by widening tolerances.")
    if decision == "accept_reduced":
        next_actions.append("Repeat the same checks at intended backend, resolution, and duration.")
    if decision == "accept":
        next_actions.append("Freeze the evidence ledger and prepare the validated handoff.")

    return {
        "schema_version": 1,
        "module": module,
        "score": round(score, 2),
        "raw_score": raw_score,
        "threshold": threshold,
        "decision": decision,
        "groups": groups,
        "checks": checks,
        "blockers": blockers,
        "hard_failures": hard_failures,
        "missing_evidence": missing_evidence,
        "next_actions": next_actions,
    }
