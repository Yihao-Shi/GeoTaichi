from pathlib import Path

from geotaichi_mcp.core.mcp_resources import register_resources
from geotaichi_mcp.core.resources import load_capability_index, load_contract_template, load_template
from geotaichi_mcp.knowledge.documentation import browse, query
from geotaichi_mcp.knowledge.inspection import inspect_model
from geotaichi_mcp.tools import register_tools


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_capability_browse_and_query_cover_all_facades():
    index = load_capability_index(REPOSITORY_ROOT)

    root = browse(index, None)
    names = {entry["name"] for entry in root["data"]["entries"]}
    assert root["ok"]
    assert names == {
        "mpm",
        "dem",
        "mpdem",
        "cfdem",
        "fem",
        "fedem",
        "fempm",
        "iga",
        "igampm",
    }

    result = query(index, "soft levelset advection", 10)
    assert result["ok"]
    assert result["data"]["entries"]
    assert all("path" in entry for entry in result["data"]["entries"])


def test_natural_language_query_routes_soft_particle_levelset_coupling():
    index = load_capability_index(REPOSITORY_ROOT)

    for terms in (
        "deformable soft particles colliding with rigid level-set bodies",
        "soft particle and level set dem coupling",
        "软颗粒和level set dem耦合",
    ):
        result = query(index, terms, 10)
        entries = result["data"]["entries"]

        assert result["ok"]
        assert entries, terms
        assert entries[0]["path"] == "fedem", (terms, entries[:3])
        assert entries[0]["match_reason"] == "physics_route"
        assert "matched_terms" in entries[0]


def test_natural_language_query_distinguishes_fluid_solid_coupling():
    result = query(load_capability_index(REPOSITORY_ROOT), "fluid solid coupling with contact", 10)

    assert result["ok"]
    assert result["data"]["entries"][0]["path"] == "cfdem"


def test_natural_language_query_uses_physics_routes_and_negative_constraints():
    index = load_capability_index(REPOSITORY_ROOT)
    cases = {
        "deformable material points only, with no FEM surface and no rigid particles": "mpm",
        "non-spherical rigid particles represented by signed distance fields": "dem",
        "an MPM soft continuum interacting with rigid DEM bodies without fluid drag": "mpdem",
        "drag feedback between a carrier fluid and discrete grains": "cfdem",
        "TET4 soft body only，不使用 DEM or MPM": "fem",
        "finite-element soft particle，不是 MPM material points，碰撞 level-set DEM body": "fedem",
        "MPM points contact FEM，不包含 rigid DEM body": "fempm",
        "a spline solid discretized by control points without material points": "iga",
        "material points 接触 NURBS surface，not a FEM mesh": "igampm",
    }

    for terms, expected in cases.items():
        result = query(index, terms, 10)
        entries = result["data"]["entries"]

        assert entries, terms
        assert entries[0]["path"] == expected, (terms, entries[:3])
        assert entries[0]["match_reason"] == "physics_route"


def test_exact_api_query_keeps_record_level_results():
    index = load_capability_index(REPOSITORY_ROOT)
    result = query(index, "set_solver", 10)

    assert result["ok"]
    assert any(entry["kind"] == "method" for entry in result["data"]["entries"])


def test_templates_use_canonical_agent_assets():
    for module, filename in {
        "mpm": "mpm_model_template.py",
        "dem": "dem_model_template.py",
        "mpdem": "mpdem_model_template.py",
        "iga": "iga_model_template.py",
        "igampm": "igampm_model_template.py",
    }.items():
        content, path = load_template(module, REPOSITORY_ROOT)
        assert path.name == filename
        assert "def build_and_run" in content
        assert "publish_live_namespace" in content
        assert path.is_relative_to(REPOSITORY_ROOT / "agent" / "geotaichi-model-builder" / "assets")


def test_model_inspector_is_shared_with_agent_cli(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
        """\
from geotaichi import MPM, init

init(dim=2, arch="cpu", log=False)
model = MPM(log=False)
model.set_configuration(domain=[1.0, 1.0])
model.set_solver({"Timestep": 1.0e-4, "SimulationTime": 1.0e-3, "SaveInterval": 1.0e-3})
model.run()
""",
        encoding="utf-8",
    )

    result = inspect_model(script, load_capability_index(REPOSITORY_ROOT))

    assert result["valid"], result
    assert result["facts"]["facades"]["model"]["category"] == "mpm"


def test_registers_partitioned_public_tools():
    class FakeMCP:
        def __init__(self):
            self.names = []
            self.annotations = {}

        def tool(self, **kwargs):
            def decorator(function):
                self.names.append(function.__name__)
                self.annotations[function.__name__] = kwargs.get("annotations")
                return function

            return decorator

    server = FakeMCP()
    register_tools(server)

    assert server.names == [
        "geotaichi_browse_capabilities",
        "geotaichi_query_capabilities",
        "geotaichi_get_model_template",
        "geotaichi_inspect_model",
        "geotaichi_score_physics",
        "geotaichi_review_model",
        "geotaichi_audit_api_docs",
        "geotaichi_validate_solver_job",
        "geotaichi_check_task_status",
        "geotaichi_list_tasks",
        "geotaichi_list_task_artifacts",
        "geotaichi_interrupt_task",
        "geotaichi_submit_solver_job",
        "geotaichi_execute_task",
        "geotaichi_execute_code",
    ]
    assert server.annotations["geotaichi_validate_solver_job"]["readOnlyHint"]
    assert server.annotations["geotaichi_interrupt_task"]["destructiveHint"]

    safe = FakeMCP()
    register_tools(safe, profile="safe")
    assert len(safe.names) == 12
    assert "geotaichi_submit_solver_job" not in safe.names
    assert "geotaichi_execute_code" not in safe.names


def test_contract_templates_and_mcp_resources_are_canonical():
    class FakeMCP:
        def __init__(self):
            self.resources = {}

        def resource(self, uri, **kwargs):
            def decorator(function):
                self.resources[uri] = (function, kwargs)
                return function

            return decorator

    for name in ("model-contract", "solver-job", "scene-manifest"):
        content, path = load_contract_template(name, REPOSITORY_ROOT)
        assert path.is_file()
        assert '"schema_version": 1' in content

    server = FakeMCP()
    register_resources(server)
    assert "geotaichi://index" in server.resources
    assert "geotaichi://contracts/solver-job" in server.resources
    assert "geotaichi://validation/rubric" in server.resources
    assert "geotaichi://workflow/{name}" in server.resources
    reader, metadata = server.resources["geotaichi://contracts/solver-job"]
    assert '"execution"' in reader()
    assert metadata["annotations"]["readOnlyHint"]
