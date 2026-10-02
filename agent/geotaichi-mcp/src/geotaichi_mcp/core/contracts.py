"""Stable response envelopes shared by GeoTaichi MCP tools."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


MAX_RESPONSE_CHARS = 8000


def _truncate_strings(value: Any, budget: int) -> Any:
    if isinstance(value, str):
        if len(value) <= budget:
            return value
        cut = value[:budget].rsplit("\n", 1)[0]
        return cut + "\n... (truncated; request a narrower result or continue from the returned offset)"
    if isinstance(value, dict):
        return {key: _truncate_strings(item, budget) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_strings(item, budget) for item in value]
    return value


def _enforce_size(envelope: Dict[str, Any], max_chars: int = MAX_RESPONSE_CHARS) -> Dict[str, Any]:
    serialized = json.dumps(envelope, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return envelope

    envelope["data"] = _truncate_strings(envelope.get("data"), max_chars // 2)
    error = envelope.get("error")
    if isinstance(error, dict) and error.get("details") is not None:
        error["details"] = _truncate_strings(error["details"], max_chars // 2)
    serialized = json.dumps(envelope, ensure_ascii=False, default=str)
    if len(serialized) > max_chars:
        summary = {
            "_truncated": True,
            "_original_size": len(serialized),
            "_message": "Response is too large; use a more specific path, query, or log offset.",
        }
        if envelope.get("data") is not None:
            envelope["data"] = summary
        if isinstance(error, dict) and error.get("details") is not None:
            error["details"] = summary
    return envelope


def build_ok(data: Any) -> Dict[str, Any]:
    """Build a successful business-level tool response."""
    return _enforce_size({"ok": True, "data": data})


def build_error(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    data: Any = None,
) -> Dict[str, Any]:
    """Build a failed business-level tool response."""
    error: Dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    envelope: Dict[str, Any] = {"ok": False, "error": error}
    if data is not None:
        envelope["data"] = data
    return _enforce_size(envelope)


def build_docs_data(action: str, entries: list, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build the common inner payload for capability documentation tools."""
    return {
        "source": "geotaichi_capabilities",
        "action": action,
        "entries": entries,
        "summary": summary,
    }
