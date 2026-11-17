#!/usr/bin/env python3
"""Score GeoTaichi run evidence against a model contract and validation rubric."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import build_error, emit, find_repo_root, load_json


REPO_ROOT = find_repo_root()
MCP_SOURCE = REPO_ROOT / "agent/geotaichi-mcp/src"
if str(MCP_SOURCE) not in sys.path:
    sys.path.insert(0, str(MCP_SOURCE))

from geotaichi_mcp.core.resources import load_physics_validation_rubric
from geotaichi_mcp.knowledge.physics_validation import score_physics_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, help="Model contract JSON path")
    parser.add_argument("--evidence", type=Path, help="Validation evidence JSON path")
    parser.add_argument("--repo-root", type=Path, help="GeoTaichi repository root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = (args.repo_root or REPO_ROOT).resolve()
        contract = load_json(args.contract)
        evidence = load_json(args.evidence) if args.evidence else {}
        payload = score_physics_validation(contract, evidence, load_physics_validation_rubric(repo))
        emit({"ok": True, "data": payload})
        return 0 if payload["decision"] in {"accept", "accept_reduced"} else 2
    except FileNotFoundError as exc:
        emit(build_error("validation_input_not_found", str(exc)))
        return 1
    except ValueError as exc:
        emit(build_error("invalid_validation_input", str(exc)))
        return 1
    except Exception as exc:
        emit(build_error("physics_scoring_failed", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
