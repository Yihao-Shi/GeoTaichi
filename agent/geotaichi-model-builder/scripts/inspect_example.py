#!/usr/bin/env python3
"""Statically inspect a GeoTaichi example for lifecycle and API hazards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import build_error, build_ok, emit, find_repo_root, load_json


DEFAULT_INDEX = Path("agent/geotaichi-model-builder/references/capability-index.json")
REPO_ROOT = find_repo_root()
MCP_SOURCE = REPO_ROOT / "agent/geotaichi-mcp/src"
if str(MCP_SOURCE) not in sys.path:
    sys.path.insert(0, str(MCP_SOURCE))

from geotaichi_mcp.knowledge.inspection import inspect_model as inspect_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="GeoTaichi example/model script")
    parser.add_argument("--contract", type=Path, help="Optional model contract JSON")
    parser.add_argument("--index", type=Path, help="Capability index JSON")
    parser.add_argument("--repo-root", type=Path, help="GeoTaichi repository root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = (args.repo_root or find_repo_root()).resolve()
        index_path = args.index or (repo / DEFAULT_INDEX)
        if not index_path.is_absolute():
            index_path = repo / index_path
        contract = load_json(args.contract) if args.contract else None
        result = inspect_example(args.script, load_json(index_path), contract)
        payload = (
            build_ok(result)
            if result["valid"]
            else build_error(
                "example_inspection_failed",
                "The GeoTaichi example has static lifecycle/API errors.",
                result,
            )
        )
        emit(payload)
        return 0 if result["valid"] else 1
    except FileNotFoundError as exc:
        emit(build_error("input_not_found", str(exc)))
        return 1
    except SyntaxError as exc:
        emit(build_error("python_syntax_error", str(exc), {"line": exc.lineno, "offset": exc.offset}))
        return 1
    except Exception as exc:
        emit(build_error("example_inspection_error", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
