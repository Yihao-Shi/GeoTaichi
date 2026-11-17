#!/usr/bin/env python3
"""Audit generated facade methods and configuration keys against helper API docs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from build_capability_index import build_index
from common import build_error, build_ok, emit, find_repo_root, normalized_tex


DEFAULT_DOC = Path("docs/helper/geotaichi_api_reference.tex")


def audit(repo: Path, docs_path: Path, categories: list[str], kind: str) -> dict[str, Any]:
    """Return structured coverage results for selected capability categories."""
    index = build_index(repo)
    docs = normalized_tex(docs_path.read_text(encoding="utf-8"))
    unknown = sorted(set(categories) - set(index["categories"]))
    if unknown:
        return build_error(
            "unknown_category",
            "One or more capability categories are unknown.",
            {"unknown": unknown, "available": sorted(index["categories"])},
        )

    results: dict[str, Any] = {}
    total_missing_methods = 0
    total_missing_keys = 0
    for name in categories:
        category = index["categories"][name]
        methods = category["public_methods"] if kind in {"methods", "all"} else []
        keys = category["configuration_keys"] if kind in {"keys", "all"} else []
        missing_methods = [method for method in methods if method["name"] not in docs]
        missing_keys = [key for key in keys if key["name"] not in docs]
        total_missing_methods += len(missing_methods)
        total_missing_keys += len(missing_keys)
        results[name] = {
            "method_count": len(methods),
            "missing_methods": missing_methods,
            "configuration_key_count": len(keys),
            "missing_configuration_keys": missing_keys,
        }

    complete = total_missing_methods == 0 and total_missing_keys == 0
    data = {
        "source": docs_path.relative_to(repo).as_posix(),
        "kind": kind,
        "categories": results,
        "summary": {
            "complete": complete,
            "missing_method_count": total_missing_methods,
            "missing_configuration_key_count": total_missing_keys,
        },
    }
    if complete:
        return build_ok(data)
    return build_error(
        "documentation_coverage_incomplete",
        "Current public methods or configuration keys are missing from the helper API reference.",
        data,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, help="GeoTaichi repository root")
    parser.add_argument("--docs", type=Path, help="API reference TeX path")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(
            (
                "mpm",
                "dem",
                "mpdem",
                "cfdem",
                "fem",
                "fedem",
                "fempm",
                "iga",
                "igampm",
            )
        ),
    )
    parser.add_argument("--kind", choices=("methods", "keys", "all"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = (args.repo_root or find_repo_root()).resolve()
        docs = args.docs or (repo / DEFAULT_DOC)
        if not docs.is_absolute():
            docs = repo / docs
        payload = audit(repo, docs, args.categories, args.kind)
        emit(payload)
        return 0 if payload.get("ok") else 1
    except Exception as exc:
        emit(build_error("api_audit_failed", str(exc)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
