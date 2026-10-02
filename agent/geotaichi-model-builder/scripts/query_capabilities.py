#!/usr/bin/env python3
"""Browse or search the generated GeoTaichi capability index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import build_error, emit, find_repo_root, load_json


DEFAULT_INDEX = Path("agent/geotaichi-model-builder/references/capability-index.json")
REPO_ROOT = find_repo_root()
MCP_SOURCE = REPO_ROOT / "agent/geotaichi-mcp/src"
if str(MCP_SOURCE) not in sys.path:
    sys.path.insert(0, str(MCP_SOURCE))

from geotaichi_mcp.knowledge.documentation import browse, query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, help="GeoTaichi repository root")
    parser.add_argument("--index", type=Path, help="Capability index JSON path")
    subparsers = parser.add_subparsers(dest="action", required=True)
    browse_parser = subparsers.add_parser("browse", help="Browse a known capability path")
    browse_parser.add_argument("path", nargs="?", help="Category/section/item path")
    query_parser = subparsers.add_parser("query", help="Search capabilities by terms")
    query_parser.add_argument("terms", help="Natural-language or identifier query")
    query_parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = (args.repo_root or find_repo_root()).resolve()
        index_path = args.index or (repo / DEFAULT_INDEX)
        if not index_path.is_absolute():
            index_path = repo / index_path
        index = load_json(index_path)
        payload = browse(index, args.path) if args.action == "browse" else query(index, args.terms, args.limit)
        emit(payload)
        return 0 if payload.get("ok") else 1
    except FileNotFoundError as exc:
        emit(build_error("index_not_found", str(exc), {"action": "Run build_capability_index.py first"}))
        return 1
    except Exception as exc:
        emit(build_error("capability_query_failed", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
