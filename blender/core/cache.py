"""Immutable local cache for one terminal SolverJob output directory."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable


TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def cache_task_output(source: Path, cache_root: Path, scene_id: str, task_id: str) -> Path:
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("task_id must be a 12-character lowercase hexadecimal identifier")
    if not scene_id or any(character in scene_id for character in "/\\"):
        raise ValueError("scene_id must be a non-empty path-safe identifier")
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(str(source))
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / scene_id / task_id
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".%s.%s.tmp" % (task_id, uuid.uuid4().hex[:8]))
    shutil.copytree(source, temporary)
    os.replace(str(temporary), str(destination))
    return destination


def cache_task_artifacts(
    artifacts: Iterable[Dict[str, Any]],
    task_directory: Path,
    cache_root: Path,
    scene_id: str,
    task_id: str,
) -> Path:
    """Copy a bounded artifact listing into one immutable scene/task cache."""
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("task_id must be a 12-character lowercase hexadecimal identifier")
    if not scene_id or any(character in scene_id for character in "/\\"):
        raise ValueError("scene_id must be a non-empty path-safe identifier")
    task_directory = task_directory.expanduser().resolve()
    if not task_directory.is_dir():
        raise NotADirectoryError(str(task_directory))
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / scene_id / task_id
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".%s.%s.tmp" % (task_id, uuid.uuid4().hex[:8]))
    temporary.mkdir()
    try:
        for record in artifacts:
            relative = Path(str(record.get("relative_path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("artifact relative_path must stay inside the task directory")
            source = Path(str(record.get("path", ""))).expanduser().resolve()
            try:
                resolved_relative = source.relative_to(task_directory)
            except ValueError as exc:
                raise ValueError("artifact source must stay inside the task directory") from exc
            if resolved_relative != relative:
                raise ValueError("artifact path does not match its relative_path")
            if not source.is_file():
                raise FileNotFoundError(str(source))
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        os.replace(str(temporary), str(destination))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = ["cache_task_artifacts", "cache_task_output"]
