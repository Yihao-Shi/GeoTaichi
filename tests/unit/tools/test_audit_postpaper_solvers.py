import ast
import csv
from pathlib import Path

from tools.audit_postpaper_solvers import (
    FunctionCollector,
    GUARD_DECISION_COLUMNS,
    LEDGER_COLUMNS,
    STYLE_AUDIT_COLUMNS,
    caller_map,
    ledger_rows,
    mechanical_style_issues,
)


def collect(source: str, path: str = "src/fem/engines/ExplicitFEM.py"):
    collector = FunctionCollector("fem", Path(path), source)
    collector.visit(ast.parse(source))
    return collector.records


def test_nested_function_behavior_is_not_attributed_to_outer_scope():
    records = collect(
        """
def outer(value):
    if value:
        first()
    def inner():
        if value is None:
            second()
    return value
"""
    )
    outer, inner = records
    assert outer.function == "outer"
    assert outer.calls == ("first",)
    assert len(outer.guards) == 1
    assert inner.function == "outer.inner"
    assert inner.calls == ("second",)
    assert len(inner.guards) == 1


def test_hot_getattr_dispatch_is_nominated_for_one_time_binding():
    records = collect(
        """
class ExplicitFEM:
    def substep(self, scene):
        update = getattr(self, "update", None)
        if update is not None:
            update(scene)
"""
    )
    assert records[0].classification == "bind_or_specialize_candidate"
    assert records[0].frequency == "every_step_candidate"


def test_caller_map_prefers_exact_same_module_and_ledger_has_required_fields():
    records = collect(
        """
def helper(value):
    return value

def caller(value):
    return helper(value)
""",
        path="src/mpm/engines/Engine.py",
    )
    callers, resolution = caller_map(records)
    helper = records[0]
    assert callers[helper.qualified_name] == {records[1].qualified_name}
    assert resolution[helper.qualified_name] == {"module_exact"}
    rows = ledger_rows(records)
    assert tuple(rows[0]) == LEDGER_COLUMNS
    assert "no guard deletion authorized" in rows[0]["proof"]


def test_guard_decision_schema_is_stable_for_incremental_audits():
    assert GUARD_DECISION_COLUMNS == (
        "file",
        "class",
        "function",
        "guard_line",
        "final_decision",
        "configured_caller_proof",
        "behavior_invariant_test",
        "regression",
        "notes",
    )


def test_style_audit_distinguishes_property_setters_from_duplicate_methods():
    issues = mechanical_style_issues(
        """
class State:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
"""
    )
    assert issues["duplicate_method_definitions"] == []


def test_style_audit_detects_new_mechanical_hazards():
    issues = mechanical_style_issues(
        """
from module import *

def route(values=[]):
    try:
        pass
    except BaseException:
        return None
"""
    )
    assert issues["wildcard_imports"] == [2]
    assert issues["mutable_defaults"] == ["route:L4"]
    assert issues["broad_exception_without_reraise"] == [7]


def test_style_audit_checks_new_class_and_function_declarations():
    issues = mechanical_style_issues(
        """
class conforming_solver:
    def AssembleSystem(self):
        return None
"""
    )
    assert issues["nonconforming_declarations"] == [
        "class:conforming_solver:L2",
        "function:AssembleSystem:L3",
    ]


def test_generated_style_audit_has_no_unresolved_violation():
    repo = Path(__file__).resolve().parents[3]
    report = repo / "research/llm_assist/postpaper/solver_audit/style_audit.csv"
    with report.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert tuple(rows[0]) == STYLE_AUDIT_COLUMNS
    assert [row["file"] for row in rows if row["status"] == "violation"] == []


def test_every_generated_binding_candidate_has_an_explicit_decision():
    repo = Path(__file__).resolve().parents[3]
    report = repo / "research/llm_assist/postpaper/solver_audit"
    with (report / "function_audit.csv").open(encoding="utf-8", newline="") as stream:
        candidates = {
            (row["file"], row["class"], row["function"])
            for row in csv.DictReader(stream)
            if row["classification"] == "bind_or_specialize_candidate"
        }
    with (report / "guard_decisions.csv").open(encoding="utf-8", newline="") as stream:
        decisions = {
            (row["file"], row["class"], row["function"]) for row in csv.DictReader(stream) if row["final_decision"]
        }

    assert len(candidates) == 8
    assert candidates <= decisions
