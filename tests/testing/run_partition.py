#!/usr/bin/env python3
"""Run a named GeoTaichi pytest partition."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARTITION_FILE = Path(__file__).with_name("test_partitions.json")


def _load_partitions():
    with PARTITION_FILE.open(encoding="utf-8") as stream:
        data = json.load(stream)
    partitions = data.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError(f"{PARTITION_FILE} must contain a 'partitions' object")
    return partitions


def _print_partitions(partitions):
    width = max(len(name) for name in partitions)
    for name, definition in partitions.items():
        description = definition.get("description", "")
        print(f"{name:<{width}}  {description}")


def _existing_paths(definition):
    selected = []
    for relative in definition.get("paths", ()):
        candidate = REPOSITORY_ROOT / relative
        if candidate.exists():
            selected.append(candidate.relative_to(REPOSITORY_ROOT).as_posix())
    for pattern in definition.get("globs", ()):
        for candidate in sorted(REPOSITORY_ROOT.glob(pattern)):
            if candidate.exists():
                selected.append(
                    candidate.relative_to(REPOSITORY_ROOT).as_posix()
                )
    # A file can match both ``*ipc*`` and ``*friction*``.  Preserve stable
    # order while preventing pytest from collecting it twice.
    return list(dict.fromkeys(selected))


def _extract_marker_arguments(arguments):
    """Remove pytest ``-m`` options and return their non-empty expressions."""

    remaining = []
    expressions = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-m":
            if index + 1 >= len(arguments) or arguments[index + 1].startswith(
                "-"
            ):
                raise ValueError("pytest option -m requires a marker expression")
            expression = arguments[index + 1].strip()
            index += 2
        elif argument.startswith("-m="):
            expression = argument[3:].strip()
            index += 1
        elif argument.startswith("-m") and len(argument) > 2:
            expression = argument[2:].strip()
            index += 1
        else:
            remaining.append(argument)
            index += 1
            continue

        if not expression:
            raise ValueError("pytest option -m requires a marker expression")
        expressions.append(expression)

    return remaining, expressions


def _combined_marker_expression(partition_expression, extra_expressions):
    """Intersect a partition marker with every caller-provided marker."""

    expressions = []
    if partition_expression:
        expressions.append(str(partition_expression).strip())
    expressions.extend(
        str(expression).strip()
        for expression in extra_expressions
        if str(expression).strip()
    )
    if not expressions:
        return None
    if len(expressions) == 1:
        return expressions[0]
    return " and ".join(f"({expression})" for expression in expressions)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one named pytest partition; unknown arguments are forwarded to pytest."
    )
    parser.add_argument("partition", nargs="?", help="partition name from test_partitions.json")
    parser.add_argument("--list", action="store_true", help="list available partitions and exit")
    options, pytest_args = parser.parse_known_args(argv)

    try:
        partitions = _load_partitions()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if options.list:
        _print_partitions(partitions)
        return 0
    if options.partition is None:
        parser.error("a partition is required (use --list to see available names)")
    if options.partition not in partitions:
        choices = ", ".join(partitions)
        parser.error(f"unknown partition {options.partition!r}; choose one of: {choices}")

    definition = partitions[options.partition]
    selected_paths = _existing_paths(definition)
    if (definition.get("paths") or definition.get("globs")) and not selected_paths:
        parser.error(
            f"partition {options.partition!r} has no matching tests; "
            "add its first test before running it"
        )

    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    try:
        partition_pytest_args, partition_marker_expressions = (
            _extract_marker_arguments(
                list(definition.get("pytest_args", ()))
            )
        )
        pytest_args, user_marker_expressions = _extract_marker_arguments(
            pytest_args
        )
    except ValueError as error:
        parser.error(str(error))

    command = [sys.executable, "-m", "pytest", *selected_paths]
    command.extend(partition_pytest_args)
    marker_expression = _combined_marker_expression(
        definition.get("markers"),
        [*partition_marker_expressions, *user_marker_expressions],
    )
    if marker_expression:
        command.extend(("-m", marker_expression))
    command.extend(pytest_args)

    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    return subprocess.call(command, cwd=REPOSITORY_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
