#!/usr/bin/env python3
"""Shared helpers for GeoTaichi model-agent scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the GeoTaichi repository root from a path inside the repository."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "geotaichi").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("GeoTaichi repository root was not found")


def build_ok(data: Any) -> dict[str, Any]:
    """Return a stable success envelope."""
    return {"ok": True, "data": data}


def build_error(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable error envelope."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"ok": False, "error": error}


def emit(payload: dict[str, Any], *, pretty: bool = True) -> None:
    """Print a JSON payload using deterministic formatting."""
    print(
        json.dumps(
            payload,
            indent=2 if pretty else None,
            ensure_ascii=False,
            sort_keys=False,
        )
    )


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON from path."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, data: Any) -> None:
    """Write deterministic UTF-8 JSON, creating only the parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=False)
        stream.write("\n")


def normalized_tex(text: str) -> str:
    """Normalize escaped identifiers for source-to-LaTeX coverage checks."""
    return text.replace(r"\_", "_")
