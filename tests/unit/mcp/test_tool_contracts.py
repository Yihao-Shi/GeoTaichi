import inspect
from argparse import Namespace
from pathlib import Path

import pytest

from geotaichi_mcp.core.contracts import MAX_RESPONSE_CHARS, build_error, build_ok
from geotaichi_mcp.execution.task_manager import TaskManager
from geotaichi_mcp.server import _http_auth
from geotaichi_mcp.tools import (
    configure_task_manager,
    geotaichi_check_task_status,
    geotaichi_execute_code,
    geotaichi_get_model_template,
    geotaichi_submit_solver_job,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_response_envelopes_are_coherent_and_bounded():
    success = build_ok({"value": 1})
    error = build_error("bad_input", "bad value", {"field": "x"})
    large = build_ok({"output": "x" * (MAX_RESPONSE_CHARS * 3)})
    large_error = build_error("large", "large details", {"output": "x" * (MAX_RESPONSE_CHARS * 3)})

    assert success == {"ok": True, "data": {"value": 1}}
    assert error["ok"] is False
    assert error["error"]["code"] == "bad_input"
    assert len(large["data"]["output"]) < MAX_RESPONSE_CHARS
    assert len(large_error["error"]["details"]["output"]) < MAX_RESPONSE_CHARS


def test_business_tools_do_not_require_fastmcp(tmp_path):
    configure_task_manager(TaskManager(workspace=tmp_path / "tasks"))

    code = geotaichi_execute_code("print(6 * 7)")
    template = geotaichi_get_model_template("iga")
    missing = geotaichi_check_task_status("000000000000")

    assert code["ok"] and code["data"]["stdout"].strip() == "42"
    assert template["ok"] and "from geotaichi import IGA" in template["data"]["content"]
    assert not missing["ok"] and missing["error"]["code"] == "task_not_found"


def test_execute_code_contract_exposes_optional_live_task_id():
    signature = inspect.signature(geotaichi_execute_code)

    assert list(signature.parameters) == ["code", "timeout", "working_directory", "task_id"]
    assert signature.parameters["task_id"].default == ""


def test_single_root_build_manifest_packages_the_vendored_vtk_writer():
    manifest = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"third_party.pyevtk"' in manifest
    assert '"third_party.pyevtk.*"' in manifest


def test_trusted_solver_job_requires_explicit_confirmation():
    result = geotaichi_submit_solver_job("unused.json", confirm_trusted_execution=False)

    assert not result["ok"]
    assert result["error"]["code"] == "trusted_execution_not_confirmed"


def test_network_transport_requires_bearer_token(monkeypatch):
    monkeypatch.delenv("GEOTAICHI_TEST_AUTH_TOKEN", raising=False)
    args = Namespace(
        transport="http",
        auth_token_env="GEOTAICHI_TEST_AUTH_TOKEN",
        allow_unauthenticated_http=False,
    )

    with pytest.raises(SystemExit, match="must contain a bearer token"):
        _http_auth(args)

    args.allow_unauthenticated_http = True
    assert _http_auth(args) is None
