import asyncio
import os
import sys
from pathlib import Path

import pytest


fastmcp = pytest.importorskip("fastmcp")
from fastmcp import Client

from geotaichi_mcp.execution.task_manager import TaskManager
from geotaichi_mcp.server import create_server
from geotaichi_mcp.tools import configure_task_manager


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.integration
def test_fastmcp_in_process_transport_registers_and_calls_tools(tmp_path):
    async def exercise():
        configure_task_manager(TaskManager(workspace=tmp_path / "tasks", repo_root=REPOSITORY_ROOT))
        async with Client(create_server()) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            resource_templates = await client.list_resource_templates()
            result = await client.call_tool("geotaichi_execute_code", {"code": "print('transport-ok')"})
        return tools, resources, resource_templates, result

    tools, resources, resource_templates, result = asyncio.run(exercise())

    assert len(tools) == 15
    assert {tool.name for tool in tools} >= {
        "geotaichi_execute_code",
        "geotaichi_execute_task",
        "geotaichi_review_model",
        "geotaichi_score_physics",
        "geotaichi_submit_solver_job",
        "geotaichi_validate_solver_job",
    }
    assert {str(resource.uri) for resource in resources} >= {
        "geotaichi://index",
        "geotaichi://contracts/solver-job",
        "geotaichi://contracts/scene-manifest",
        "geotaichi://validation/rubric",
    }
    assert {str(resource.uriTemplate) for resource in resource_templates} == {
        "geotaichi://workflow/{name}"
    }
    execute_tool = next(tool for tool in tools if tool.name == "geotaichi_execute_code")
    assert "task_id" in execute_tool.inputSchema["properties"]
    assert result.is_error is False
    assert result.structured_content["ok"] is True
    assert result.structured_content["data"]["stdout"].strip() == "transport-ok"


@pytest.mark.integration
def test_stdio_transport_mutates_a_running_task_namespace(tmp_path):
    script = tmp_path / "stdio_live.py"
    script.write_text(
        """\
import time
from geotaichi_mcp.execution.live import checkpoint

counter = 2
stop = False
print("stdio-live-ready", flush=True)
while not stop:
    checkpoint()
    time.sleep(0.005)
print("stdio-live-final=%s" % counter, flush=True)
""",
        encoding="utf-8",
    )
    mcp_python_path = os.pathsep.join(
        path
        for path in (
            str(REPOSITORY_ROOT / "agent" / "geotaichi-mcp" / "src"),
            str(REPOSITORY_ROOT),
            os.environ.get("PYTHONPATH", ""),
        )
        if path
    )
    config = {
        "mcpServers": {
            "geotaichi": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "geotaichi_mcp",
                    "--repo-root",
                    str(REPOSITORY_ROOT),
                    "--workspace",
                    str(tmp_path / "tasks"),
                ],
                "env": {"PYTHONPATH": mcp_python_path},
            }
        }
    }

    async def call_data(client, name, arguments):
        result = await client.call_tool(name, arguments)
        assert result.is_error is False
        return result.structured_content

    async def exercise():
        async with Client(config, timeout=15) as client:
            submitted = await call_data(
                client,
                "geotaichi_execute_task",
                {"entry_script": str(script), "description": "stdio live transport integration"},
            )
            task_id = submitted["data"]["task_id"]
            for _ in range(500):
                status = await call_data(
                    client,
                    "geotaichi_check_task_status",
                    {"task_id": task_id, "max_output_chars": 3000},
                )
                if "stdio-live-ready" in status["data"]["stdout"]["text"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("task did not reach live checkpoint")

            mutation = await call_data(
                client,
                "geotaichi_execute_code",
                {
                    "task_id": task_id,
                    "code": "counter += 40\n_geotaichi_mcp_result = counter",
                    "timeout": 2,
                },
            )
            stopped = await call_data(
                client,
                "geotaichi_execute_code",
                {"task_id": task_id, "code": "stop = True", "timeout": 2},
            )
            for _ in range(500):
                terminal = await call_data(
                    client,
                    "geotaichi_check_task_status",
                    {"task_id": task_id, "max_output_chars": 3000},
                )
                if terminal["data"]["task"]["status"] == "completed":
                    return mutation, stopped, terminal
                await asyncio.sleep(0.01)
            raise AssertionError("live stdio task did not complete")

    mutation, stopped, terminal = asyncio.run(exercise())

    assert mutation["ok"] and mutation["data"]["result"] == 42
    assert mutation["data"]["execution_mode"] == "live"
    assert stopped["ok"]
    assert "stdio-live-final=42" in terminal["data"]["stdout"]["text"]
