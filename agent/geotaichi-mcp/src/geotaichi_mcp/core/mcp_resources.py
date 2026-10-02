"""LLM-sized MCP resources backed by canonical model-builder assets."""

from __future__ import annotations

import json
from typing import Any, Dict

from .config import find_repo_root
from .resources import load_capability_index, load_contract_template, load_physics_validation_rubric, locate_resource


WORKFLOW_FILES = {
    "mpm": "workflow-mpm.md",
    "dem": "workflow-dem.md",
    "fem": "workflow-fem.md",
    "iga": "workflow-iga.md",
    "coupling": "workflow-coupling.md",
}


def resource_index() -> str:
    """Return the routing index agents should read before other resources."""
    return """# GeoTaichi MCP resource index

Read only the resource relevant to the current task:

- `geotaichi://contracts/model-contract`: physical model contract template.
- `geotaichi://contracts/solver-job`: trusted execution envelope shared by Blender, CLI, and MCP.
- `geotaichi://contracts/scene-manifest`: Blender-neutral scene interchange template.
- `geotaichi://capabilities/summary`: top-level capability navigation; use the browse/query tools for details.
- `geotaichi://validation/rubric`: physical evidence check kinds, solver policies, weights, and acceptance threshold.
- `geotaichi://workflow/{name}`: one of `mpm`, `dem`, `fem`, `iga`, or `coupling`.

Inspect a model before execution. SolverJob execution is trusted code and requires an explicit confirmation argument.
"""


def capability_summary() -> Dict[str, Any]:
    """Return navigation metadata without placing the full generated index in context."""
    index = load_capability_index(find_repo_root(required=False))
    return {
        "schema_version": index.get("schema_version"),
        "description": index.get("description"),
        "navigation": index.get("navigation"),
        "quick_ref": index.get("quick_ref"),
        "categories": sorted((index.get("categories") or {}).keys()),
        "detail_tools": ["geotaichi_browse_capabilities", "geotaichi_query_capabilities"],
    }


def contract_resource(name: str) -> str:
    content, _ = load_contract_template(name, find_repo_root(required=False))
    return content


def workflow_resource(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in WORKFLOW_FILES:
        raise ValueError("Unknown workflow %r; choose one of: %s" % (name, ", ".join(sorted(WORKFLOW_FILES))))
    path = locate_resource(WORKFLOW_FILES[normalized], find_repo_root(required=False))
    return path.read_text(encoding="utf-8")


def register_resources(mcp: Any) -> None:
    """Register stable static resources and one workflow URI template."""

    @mcp.resource(
        "geotaichi://index",
        name="GeoTaichi resource index",
        description="Route to the smallest relevant GeoTaichi knowledge resource.",
        mime_type="text/markdown",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    )
    def geotaichi_resource_index() -> str:
        return resource_index()

    @mcp.resource(
        "geotaichi://capabilities/summary",
        name="GeoTaichi capability summary",
        description="Top-level generated capability navigation; use browse/query tools for details.",
        mime_type="application/json",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    )
    def geotaichi_capability_summary() -> str:
        return json.dumps(capability_summary(), ensure_ascii=False, indent=2)

    @mcp.resource(
        "geotaichi://validation/rubric",
        name="GeoTaichi physical validation rubric",
        description="Canonical solver-specific evidence scoring policy used by the model-agent review loop.",
        mime_type="application/json",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    )
    def geotaichi_validation_rubric() -> str:
        rubric = load_physics_validation_rubric(find_repo_root(required=False))
        return json.dumps(rubric, ensure_ascii=False, indent=2)

    def register_contract(resource_name: str, title: str) -> None:
        def read_contract() -> str:
            return contract_resource(resource_name)

        read_contract.__name__ = "read_%s" % resource_name.replace("-", "_")
        mcp.resource(
            "geotaichi://contracts/%s" % resource_name,
            name=title,
            description="Canonical JSON template maintained by the GeoTaichi model-builder.",
            mime_type="application/json",
            annotations={"readOnlyHint": True, "idempotentHint": True},
        )(read_contract)

    register_contract("model-contract", "Physical model contract")
    register_contract("solver-job", "Trusted SolverJob execution envelope")
    register_contract("scene-manifest", "Blender-neutral SceneManifest")

    @mcp.resource(
        "geotaichi://workflow/{name}",
        name="GeoTaichi workflow guide",
        description="One facade-specific workflow: mpm, dem, fem, iga, or coupling.",
        mime_type="text/markdown",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    )
    def geotaichi_workflow(name: str) -> str:
        return workflow_resource(name)
