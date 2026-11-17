"""The retry uses a solid mesh and retains the original failure diagnostics."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[3]
CASES = (
    "bunny_pile_wall_release",
    "thousand_cubes_heavy_ball_impact",
)


def load_case(name):
    path = ROOT / f"examples/dem/AffineBody/{name}/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bunny_input_is_closed_and_preserves_sparse_geometry():
    mesh = trimesh.load(load_case(CASES[0]).MESH, force="mesh")
    original = trimesh.load(ROOT / "assets/bunny_sparse.obj", force="mesh")
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert np.isfinite(mesh.vertices).all() and mesh.volume > 0.0
    assert mesh.vertices.shape == original.vertices.shape == (2503, 3)
    np.testing.assert_allclose(mesh.vertices, original.vertices, rtol=0.0, atol=1e-8)
    assert abs(mesh.volume / original.volume - 1.0) < 0.01


@pytest.mark.parametrize("name", CASES)
def test_failed_run_records_diagnostics_and_reraises(tmp_path, monkeypatch, name):
    failure = RuntimeError("Affine lagged IPC CCD produced no strictly feasible step")
    diagnostics = {"last_failure": {"step": 167, "time": 0.06, "message": str(failure)}}

    walls = []
    allocations = []
    solvers = []
    parameters = []

    class DEM:
        def set_affine_body_parameters(self, **kwargs):
            parameters.append(kwargs)

        def set_solver(self, value, **kwargs):
            solvers.append(value)

        def add_wall(self, value):
            walls.append(value)

        def memory_allocate(self, value, **kwargs):
            allocations.append(value)

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

        def run(self, **kwargs):
            raise failure

        def diagnostics_snapshot(self):
            return diagnostics

    monkeypatch.setitem(
        sys.modules,
        "geotaichi",
        SimpleNamespace(
            init=lambda **kwargs: None,
            DEM=lambda **kwargs: DEM(),
            polyhedron=lambda **kwargs: None,
        ),
    )
    monkeypatch.setattr(sys, "argv", [name, "--output-dir", str(tmp_path)])
    with pytest.raises(RuntimeError) as raised:
        load_case(name).main()
    assert raised.value is failure
    assert solvers[0]["Timestep"] == 0.001 and solvers[0]["enable_step_retry"]
    assert solvers[0]["step_retry_max_retries"] == 2
    assert solvers[0]["step_retry_reduction"] == 0.5
    assert solvers[0]["step_retry_minimum_timestep"] == 0.00025
    assert parameters[0]["young_modulus"] == 2e12
    assert parameters[0]["newton_tolerance"] == (1e-5 if name == CASES[0] else 0.05)
    assert parameters[0]["linear_tolerance"] == (1e-12 if name == CASES[0] else 1e-10)
    assert json.loads((tmp_path / "solver_failure.json").read_text()) == diagnostics
    assert not (tmp_path / "validation_summary.json").exists()
    if name == CASES[0]:
        assert len(walls) == 5
        case = load_case(name)
        centre = case.container_bounds(1000).mean(axis=0)
        assert all(np.dot(centre - wall["WallCenter"], wall["OuterNormal"]) > 0 for wall in walls)
        assert {tuple(wall["OuterNormal"]) for wall in walls} == {
            (0, 0, 1),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
        }
        assert allocations[0]["max_affine_body_number"] == 1000
        assert parameters[0]["barrier_stiffness"] == 8e12
        assert parameters[0]["linear_max_iteration"] == 12000
        assert allocations[0]["max_point_triangle_pairs"] == 16000000
        assert allocations[0]["max_edge_edge_pairs"] == 32000000
    else:
        assert allocations[0]["max_affine_body_number"] == 1001
        assert allocations[0]["surface_node_number"] == 1000 * 8 + 2562


def test_thousand_bunnies_fit_complete_container_without_initial_overlap(monkeypatch):
    case = load_case(CASES[0])
    monkeypatch.setattr(sys, "argv", ["bunny"])
    assert case.arguments().count == 1000
    points = np.asarray(case.centers(1000))
    bounds = case.container_bounds(1000)
    assert points.shape == (1000, 3) and len(np.unique(points, axis=0)) == 1000
    assert all(len(np.unique(points[:, axis])) == 10 for axis in range(3))
    radius = case.BUNNY_RADIUS
    assert np.min(points - radius - bounds[0]) >= 0.006 - 1e-12
    assert np.min(bounds[1] - points - radius) >= 0.006 - 1e-12
    from scipy.spatial import cKDTree

    assert cKDTree(points).query(points, k=2)[0][:, 1].min() >= 2 * radius + 0.012 - 1e-12
    monkeypatch.setattr(sys, "argv", ["bunny", "--steps", "2", "--smoke"])
    assert case.arguments().steps == 2


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("complete_history", [True, False])
def test_adaptive_history_requires_the_complete_accepted_time(tmp_path, monkeypatch, name, complete_history):
    increments = [0.0005, 0.0005, 0.001 if complete_history else 0.00075]
    history = [
        {"time": clock, "step_retry": {"accepted_timestep": dt, "retry_count": int(i == 0)}}
        for i, (clock, dt) in enumerate(zip((0.0005, 0.001, 0.002), increments))
    ]
    count = 4 if name == CASES[0] else 5
    engine = SimpleNamespace(
        history=history,
        state=SimpleNamespace(world_vertices=lambda: [np.array([[0.5, 0.5, 0.2]]) for _ in range(count)]),
        maximum_hash_triplets=1,
        operator=SimpleNamespace(max_hash_triplets=1),
    )

    class DEM:
        sims = SimpleNamespace(current_time=0.002, current_step=3, affine_max_newton_iteration=1000)
        enginer = engine

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    monkeypatch.setitem(
        sys.modules,
        "geotaichi",
        SimpleNamespace(init=lambda **kwargs: None, DEM=lambda **kwargs: DEM(), polyhedron=lambda **kwargs: None),
    )
    count_option = "--count" if name == CASES[0] else "--cube-count"
    monkeypatch.setattr(
        sys, "argv", [name, "--output-dir", str(tmp_path), count_option, "4", "--steps", "2", "--smoke"]
    )
    if complete_history:
        load_case(name).main()
    else:
        with pytest.raises(RuntimeError, match="validation failed"):
            load_case(name).main()
    summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert not summary["fixed_timestep_verified"]
    assert summary["timestep_policy_verified"] is complete_history
    assert summary["passed"] is complete_history


def test_exactly_thousand_cubes_and_resolved_sphere(monkeypatch):
    case = load_case(CASES[1])
    monkeypatch.setattr(sys, "argv", ["cubes", "--cube-count", "1000"])
    assert case.arguments().cube_count == 1000
    assert case.arguments().dt == 0.001 and case.arguments().steps == 250
    assert case.arguments().save_every * case.arguments().dt == 0.01
    points = np.asarray(case.cube_centers(1000))
    assert points.shape == (1000, 3)
    assert all(len(np.unique(points[:, axis])) == 10 for axis in range(3))
    mesh = trimesh.load(case.BALL_MESH, force="mesh")
    old = trimesh.load(ROOT / "assets/mesh/AffineBody/lowpoly_sphere.obj", force="mesh")
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert len(mesh.vertices) == 2562 and len(mesh.faces) == 5120
    radius = np.linalg.norm(mesh.vertices - mesh.center_mass, axis=1)
    np.testing.assert_allclose(radius, np.linalg.norm(old.vertices - old.center_mass, axis=1).max(), rtol=0, atol=1e-8)
