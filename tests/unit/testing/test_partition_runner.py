"""Pure-Python contracts for the named pytest partition runner."""

import pytest

from tests.testing import run_partition


pytestmark = [pytest.mark.required, pytest.mark.cpu]


def test_extract_marker_arguments_supports_repeated_and_attached_forms():
    remaining, expressions = run_partition._extract_marker_arguments(
        [
            "-m",
            "slow",
            "-mnot gpu",
            "-m=serial or cpu",
            "-k",
            "friction",
            "-x",
        ]
    )

    assert remaining == ["-k", "friction", "-x"]
    assert expressions == ["slow", "not gpu", "serial or cpu"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["-m"],
        ["-m", "-x"],
        ["-m="],
        ["-m", ""],
    ],
)
def test_extract_marker_arguments_rejects_missing_or_empty_expression(
    arguments,
):
    with pytest.raises(ValueError, match="requires a marker expression"):
        run_partition._extract_marker_arguments(arguments)


def test_combined_marker_expression_parenthesizes_every_intersection_term():
    expression = run_partition._combined_marker_expression(
        "ipc",
        ["slow or gpu", "not serial"],
    )

    assert expression == "(ipc) and (slow or gpu) and (not serial)"


def test_combined_marker_expression_keeps_a_single_filter_readable():
    assert (
        run_partition._combined_marker_expression("unit", [])
        == "unit"
    )
    assert (
        run_partition._combined_marker_expression(None, ["slow or gpu"])
        == "slow or gpu"
    )
    assert run_partition._combined_marker_expression(None, []) is None


def _capture_main_command(monkeypatch, partitions, argv):
    captured = {}

    monkeypatch.setattr(run_partition, "_load_partitions", lambda: partitions)
    monkeypatch.setattr(
        run_partition,
        "_existing_paths",
        lambda definition: list(definition.get("paths", ())),
    )

    def fake_call(command, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        return 17

    monkeypatch.setattr(run_partition.subprocess, "call", fake_call)
    result = run_partition.main(argv)
    assert result == 17
    assert captured["cwd"] == run_partition.REPOSITORY_ROOT
    return captured["command"]


def test_main_intersects_partition_and_user_markers_once(monkeypatch):
    command = _capture_main_command(
        monkeypatch,
        {
            "ipc": {
                "paths": ["tests/unit"],
                "markers": "ipc",
            }
        },
        ["ipc", "--", "-m", "slow or gpu", "-k", "friction", "-x"],
    )

    assert command.count("-m") == 2
    # One ``-m`` belongs to ``python -m pytest``; only the second is a pytest
    # marker option.
    marker_index = command.index("-m", 2)
    assert command[marker_index + 1] == "(ipc) and (slow or gpu)"
    assert command[marker_index + 2 :] == ["-k", "friction", "-x"]


def test_main_all_partition_accepts_user_marker_without_extra_wrapper(
    monkeypatch,
):
    command = _capture_main_command(
        monkeypatch,
        {"all": {"paths": ["tests/unit"]}},
        ["all", "-mslow"],
    )

    marker_index = command.index("-m", 2)
    assert command[marker_index + 1] == "slow"


def test_main_without_user_marker_keeps_partition_marker(monkeypatch):
    command = _capture_main_command(
        monkeypatch,
        {
            "unit": {
                "paths": ["tests/unit"],
                "markers": "unit",
            }
        },
        ["unit", "-q"],
    )

    marker_index = command.index("-m", 2)
    assert command[marker_index + 1 :] == ["unit", "-q"]


def test_main_intersects_marker_from_partition_pytest_arguments(
    monkeypatch,
):
    command = _capture_main_command(
        monkeypatch,
        {
            "custom": {
                "paths": ["tests/unit"],
                "markers": "unit",
                "pytest_args": ["--run-benchmarks", "-m=not slow"],
            }
        },
        ["custom", "-m", "cpu or gpu"],
    )

    assert "--run-benchmarks" in command
    marker_index = command.index("-m", 2)
    assert command[marker_index + 1] == (
        "(unit) and (not slow) and (cpu or gpu)"
    )
