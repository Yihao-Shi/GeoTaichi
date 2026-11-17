"""Runtime configuration and repository discovery for GeoTaichi MCP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def _looks_like_repo(path: Path) -> bool:
    return (path / "geotaichi").is_dir() and (path / "src").is_dir()


def find_repo_root(start: Optional[PathLike] = None, required: bool = True) -> Optional[Path]:
    """Locate a GeoTaichi checkout without importing the numerical package."""
    configured = os.environ.get("GEOTAICHI_REPO_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    if start is not None:
        candidates.append(Path(start).expanduser())
    candidates.extend((Path.cwd(), Path(__file__).resolve().parent.parent))

    visited = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            candidate = candidate.parent
        for current in (candidate, *candidate.parents):
            if current in visited:
                continue
            visited.add(current)
            if _looks_like_repo(current):
                return current
    if required:
        raise RuntimeError(
            "GeoTaichi repository root was not found; set GEOTAICHI_REPO_ROOT to a checkout containing geotaichi/ and src/."
        )
    return None


def default_workspace(repo_root: Optional[Path] = None) -> Path:
    """Return the persistent task workspace, honoring an explicit override."""
    configured = os.environ.get("GEOTAICHI_MCP_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    repo = repo_root or find_repo_root(required=False)
    base = repo if repo is not None else Path.cwd()
    return (base / ".geotaichi-mcp" / "tasks").resolve()


def resolve_user_path(value: PathLike, repo_root: Optional[Path], *, must_exist: bool = False) -> Path:
    """Resolve a user path relative to the repository when it is not absolute."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (repo_root or Path.cwd()) / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(str(path))
    return path
