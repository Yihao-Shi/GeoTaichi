"""Locate the canonical agent resources in a checkout or installed wheel."""

from __future__ import annotations

import json
import os
import sysconfig
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import find_repo_root


TEMPLATE_FILES = {
    "mpm": "mpm_model_template.py",
    "dem": "dem_model_template.py",
    "mpdem": "mpdem_model_template.py",
    "cfdem": "coupled_model_template.py",
    "fedem": "fedem_model_template.py",
    "fempm": "fempm_model_template.py",
    "iga": "iga_model_template.py",
    "igampm": "igampm_model_template.py",
    "coupled": "coupled_model_template.py",
}

CONTRACT_FILES = {
    "model-contract": "model-contract.json",
    "solver-job": "solver-job.json",
    "scene-manifest": "scene-manifest.json",
}


def _installed_resource_root() -> Path:
    return Path(sysconfig.get_path("data")) / "share" / "geotaichi-mcp"


def resource_roots(repo_root: Optional[Path] = None) -> List[Path]:
    """Return resource locations in precedence order."""
    roots: List[Path] = []
    configured = os.environ.get("GEOTAICHI_MCP_RESOURCE_ROOT")
    if configured:
        roots.append(Path(configured).expanduser().resolve())
    repo = repo_root or find_repo_root(required=False)
    if repo is not None:
        roots.extend(
            [
                repo / "agent" / "geotaichi-model-builder" / "references",
                repo / "agent" / "geotaichi-model-builder" / "assets",
            ]
        )
    roots.append(_installed_resource_root())
    unique: List[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def locate_resource(name: str, repo_root: Optional[Path] = None) -> Path:
    """Locate a named canonical resource or raise a diagnostic error."""
    searched = []
    for root in resource_roots(repo_root):
        candidate = root / name
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Resource %r was not found; searched: %s" % (name, ", ".join(searched)))


def load_capability_index(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    configured = os.environ.get("GEOTAICHI_MCP_CAPABILITY_INDEX")
    path = (
        Path(configured).expanduser().resolve() if configured else locate_resource("capability-index.json", repo_root)
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_physics_validation_rubric(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load the canonical evidence-scoring rules owned by the model-builder knowledge base."""
    path = locate_resource("physics-validation-rubric.json", repo_root)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_template(module: str, repo_root: Optional[Path] = None) -> Tuple[str, Path]:
    normalized = module.strip().lower()
    if normalized not in TEMPLATE_FILES:
        raise ValueError("Unknown model module %r; choose one of: %s" % (module, ", ".join(sorted(TEMPLATE_FILES))))
    path = locate_resource(TEMPLATE_FILES[normalized], repo_root)
    return path.read_text(encoding="utf-8"), path


def load_contract_template(name: str, repo_root: Optional[Path] = None) -> Tuple[str, Path]:
    """Load one canonical JSON contract template by stable resource name."""
    normalized = name.strip().lower()
    if normalized not in CONTRACT_FILES:
        raise ValueError("Unknown contract template %r; choose one of: %s" % (name, ", ".join(sorted(CONTRACT_FILES))))
    path = locate_resource(CONTRACT_FILES[normalized], repo_root)
    return path.read_text(encoding="utf-8"), path
