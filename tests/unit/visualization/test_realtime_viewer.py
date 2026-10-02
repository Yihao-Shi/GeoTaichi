from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyglet
import pytest

import third_party.pyrender as vendored_pyrender
from third_party.pyrender.trackball import Trackball
from third_party.pyrender.viewer import Viewer as PyRenderViewer
from src.visualization.adapters import (
    IGARenderSource,
    LevelSetDEMRenderSource,
    dem_particle_source,
    dem_render_sources,
    dem_triangle_wall_source,
    mpm_particle_source,
)
from src.visualization.realtime import (
    CallbackSceneModel,
    CameraOptions,
    PlaybackController,
    RealtimeViewer,
    RealtimeViewerOptions,
    _camera_pose,
    _particle_poses,
)
from src.visualization.sources import (
    ParticleRenderSource,
    TriangleMeshRenderSource,
)
from src.visualization.solver_adapters import _camera, _domain_bounds


class ArrayField:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to_numpy(self):
        return self.value.copy()


def test_vendored_pyrender_and_runtime_assets_live_under_third_party():
    project_root = Path(__file__).resolve().parents[3]
    package_root = Path(vendored_pyrender.__file__).resolve().parent

    assert package_root == project_root / "third_party" / "pyrender"
    assert (package_root / "LICENSE").is_file()
    assert (package_root / "fonts" / "OpenSans-Regular.ttf").is_file()
    assert (package_root / "shaders" / "mesh.vert").is_file()
    assert not (project_root / "src" / "visualization" / "_vendor").exists()


def particle_source():
    return ParticleRenderSource(
        positions=ArrayField(
            [
                [0.2, 0.3, 0.4],
                [0.3, 0.3, 0.4],
                [0.6, 0.7, 0.4],
            ]
        ),
        radii=lambda: np.asarray([0.05, 0.08, 0.11]),
        active=ArrayField([1, 0, 1]),
        count=3,
    )


def callback_model(*sources, reset=True):
    return CallbackSceneModel(
        name="callback",
        render_sources=sources or (particle_source(),),
        time_step=0.025,
        step_callback=lambda: None,
        reset_callback=(lambda: None) if reset else None,
    )


def test_playback_controller_separates_running_and_single_step_budgets():
    controller = PlaybackController(paused=True, steps_per_frame=5)

    assert controller.consume_step_budget() == 0
    controller.request_single_step()
    assert controller.paused is True
    assert controller.consume_step_budget() == 1
    assert controller.consume_step_budget() == 0

    controller.toggle()
    assert controller.consume_step_budget() == 5
    controller.set_steps_per_frame(3)
    assert controller.consume_step_budget() == 3


def test_callback_model_owns_time_bookkeeping_and_optional_reset():
    calls = {"step": 0, "reset": 0}
    model = CallbackSceneModel(
        name="callback",
        render_sources=(particle_source(),),
        time_step=0.025,
        step_callback=lambda: calls.__setitem__("step", calls["step"] + 1),
        reset_callback=lambda: calls.__setitem__(
            "reset", calls["reset"] + 1
        ),
        statistics_callback=lambda: {"solver": "fake"},
    )

    model.step()
    model.step()

    assert calls == {"step": 2, "reset": 0}
    assert model.step_index == 2
    assert model.simulation_time == pytest.approx(0.05)
    assert model.statistics() == {"solver": "fake"}
    assert model.reset() is True
    assert calls["reset"] == 1

    no_reset = callback_model(reset=False)
    assert no_reset.can_reset is False
    assert no_reset.reset() is False


def test_particle_source_filters_active_particles_and_keeps_radii_on_host():
    source = particle_source()

    snapshot = source.snapshot()

    np.testing.assert_allclose(
        snapshot.positions,
        [[0.2, 0.3, 0.4], [0.6, 0.7, 0.4]],
    )
    np.testing.assert_allclose(snapshot.radii, [0.05, 0.11])
    assert set(vars(source)) == {
        "positions",
        "radii",
        "count",
        "active",
        "color",
        "name",
    }


def test_variable_particle_radii_become_instance_scales():
    model = callback_model(particle_source())
    viewer = RealtimeViewer(
        model,
        RealtimeViewerOptions(particle_scale=1.5),
    )

    scene = viewer._build_scene()
    poses = viewer._bindings[0].primitive.poses

    assert viewer.backend_name == "PyRender + Pyglet + PyOpenGL"
    assert len(scene.meshes) == 2
    np.testing.assert_allclose(poses[:, :3, 3], [[0.2, 0.3, 0.4], [0.6, 0.7, 0.4]])
    np.testing.assert_allclose(poses[:, (0, 1, 2), (0, 1, 2)], [[0.075] * 3, [0.165] * 3])


def test_particle_binding_updates_positions_and_host_radii_without_rebuild():
    positions = ArrayField([[0.1, 0.2, 0.3]])
    radii = np.asarray([0.05])
    source = ParticleRenderSource(positions=positions, radii=lambda: radii)
    viewer = RealtimeViewer(callback_model(source))
    viewer._build_scene()
    original_mesh = viewer._bindings[0].mesh

    positions.value[0] = [0.4, 0.5, 0.6]
    radii[0] = 0.12
    viewer._synchronize_scene()

    binding = viewer._bindings[0]
    assert binding.mesh is original_mesh
    np.testing.assert_allclose(binding.primitive.poses[0, :3, 3], positions.value[0])
    np.testing.assert_allclose(
        np.diag(binding.primitive.poses[0])[:3],
        [0.12, 0.12, 0.12],
    )


def test_triangle_source_builds_pyrender_mesh():
    triangle = TriangleMeshRenderSource(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        faces=np.asarray([[0, 1, 2]]),
    )
    viewer = RealtimeViewer(callback_model(triangle))

    scene = viewer._build_scene()

    assert viewer._bindings[0].kind == "triangles"
    assert viewer._bindings[0].item_count == 1
    assert len(scene.meshes) == 2


def test_dynamic_triangle_source_replaces_mesh_with_current_vertices():
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    triangle = TriangleMeshRenderSource(
        vertices=lambda: vertices,
        faces=np.asarray([[0, 1, 2]]),
    )
    viewer = RealtimeViewer(callback_model(triangle))
    viewer._build_scene()
    original_mesh = viewer._bindings[0].mesh

    vertices[2] = [0.0, 2.0, 0.0]
    viewer._synchronize_scene()

    binding = viewer._bindings[0]
    assert binding.mesh is not original_mesh
    np.testing.assert_allclose(
        binding.primitive.positions,
        vertices,
    )


def test_dem_adapter_reads_existing_position_radius_and_active_members():
    scene = SimpleNamespace(
        particle=SimpleNamespace(
            x=ArrayField([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            rad=ArrayField([0.1, 0.2]),
            active=ArrayField([1, 0]),
        ),
        particleNum=np.asarray([2]),
    )

    snapshot = dem_particle_source(scene).snapshot()

    np.testing.assert_allclose(snapshot.positions, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(snapshot.radii, [0.1])


def test_mpm_adapter_uses_cpu_psize_without_creating_radius_field():
    particle = SimpleNamespace(
        x=ArrayField([[0.0, 0.0], [1.0, 0.0]]),
        active=ArrayField([1, 1]),
    )
    scene = SimpleNamespace(
        particle=particle,
        particleNum=np.asarray([2]),
        psize=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
    )

    source = mpm_particle_source(scene)
    snapshot = source.snapshot()

    np.testing.assert_allclose(snapshot.radii, [np.hypot(0.1, 0.2), 0.5])
    assert not hasattr(particle, "rad")


def test_dem_triangle_wall_adapter_preserves_each_facet():
    scene = SimpleNamespace(
        wall=SimpleNamespace(
            vertice1=ArrayField([[0.0, 0.0, 0.0]]),
            vertice2=ArrayField([[1.0, 0.0, 0.0]]),
            vertice3=ArrayField([[0.0, 1.0, 0.0]]),
        ),
        wallNum=np.asarray([1]),
    )

    snapshot = dem_triangle_wall_source(scene).snapshot()

    assert snapshot.vertices.shape == (3, 3)
    np.testing.assert_array_equal(snapshot.faces, [[0, 1, 2]])


def test_levelset_adapter_uses_canonical_scene_surface_and_connectivity():
    scene = SimpleNamespace(
        visualzie_surface_node=object(),
        surfaceNum=np.asarray([3]),
        connectivity=np.asarray([[0, 1, 2]]),
        visualize_surface=lambda sims: (
            {},
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ),
    )

    snapshot = LevelSetDEMRenderSource(object(), scene).snapshot()

    assert snapshot.vertices.shape == (3, 3)
    np.testing.assert_array_equal(snapshot.faces, [[0, 1, 2]])


def test_dem_source_selection_uses_levelset_surface_and_triangle_walls():
    sims = SimpleNamespace(
        scheme="LSDEM",
        wall_type=1,
    )
    scene = SimpleNamespace(
        particle=object(),
        wall=object(),
    )

    sources = dem_render_sources(sims, scene)

    assert len(sources) == 2
    assert isinstance(sources[0], LevelSetDEMRenderSource)
    assert isinstance(sources[1], TriangleMeshRenderSource)


def test_existing_solver_visualize_methods_delegate_to_generic_gui(monkeypatch):
    from src.dem.DEMBase import Solver as DEMSolver
    from src.mpm.MPMBase import Solver as MPMSolver
    from src.visualization import solver_adapters

    calls = []

    def dem_gui(solver, scene):
        calls.append(("dem", solver, scene))
        return "dem viewer"

    def mpm_gui(solver, scene, neighbor):
        calls.append(("mpm", solver, scene, neighbor))
        return "mpm viewer"

    monkeypatch.setattr(solver_adapters, "run_dem_gui", dem_gui)
    monkeypatch.setattr(solver_adapters, "run_mpm_gui", mpm_gui)
    dem_solver = object.__new__(DEMSolver)
    mpm_solver = object.__new__(MPMSolver)
    scene = object()
    neighbor = object()

    assert dem_solver.Visualize(scene) == "dem viewer"
    assert mpm_solver.Visualize(scene, neighbor) == "mpm viewer"
    assert calls == [
        ("dem", dem_solver, scene),
        ("mpm", mpm_solver, scene, neighbor),
    ]


def test_iga_run_delegates_to_generic_gui(monkeypatch):
    from src.iga.mainIGA import IGA
    from src.visualization import solver_adapters

    engine = object()
    iga = object.__new__(IGA)
    iga.engine = engine
    captured = {}

    def iga_gui(candidate, **kwargs):
        captured["engine"] = candidate
        captured.update(kwargs)
        return "iga viewer"

    monkeypatch.setattr(solver_adapters, "run_iga_gui", iga_gui)

    result = iga.run(
        visualize=True,
        visualize_resolution=9,
        verbose=False,
        postprocessing=("callback",),
    )

    assert result == "iga viewer"
    assert captured == {
        "engine": engine,
        "resolution": 9,
        "verbose": False,
        "postprocessing": ("callback",),
    }


def test_iga_adapter_samples_a_bilinear_surface():
    primitive = SimpleNamespace(
        dimension=2,
        degree=(1, 1),
        knot_vector_u=np.asarray([0.0, 0.0, 1.0, 1.0]),
        knot_vector_v=np.asarray([0.0, 0.0, 1.0, 1.0]),
        weights=np.ones(4),
    )
    patch = SimpleNamespace(
        control_points=ArrayField(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        prefix_total_num_ctrlpts=np.asarray([0, 4]),
        total_num_ctrlpts=np.asarray([0, 4]),
        primitive=SimpleNamespace(
            body={"surface": {"primitive": primitive}}
        ),
    )

    snapshot = IGARenderSource(patch, resolution=3).snapshot()

    assert snapshot.vertices.shape == (9, 3)
    assert snapshot.faces.shape == (8, 3)
    np.testing.assert_allclose(
        snapshot.vertices.min(axis=0), [0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        snapshot.vertices.max(axis=0), [1.0, 1.0, 0.0]
    )


def test_camera_pose_uses_pyrender_negative_z_view_convention():
    options = CameraOptions(
        position=(1.0, 0.5, 0.5),
        look_at=(0.5, 0.5, 0.5),
        up=(0.0, 0.0, 1.0),
    )

    pose = _camera_pose(options)

    np.testing.assert_allclose(pose[:3, 3], options.position)
    np.testing.assert_allclose(-pose[:3, 2], (-1.0, 0.0, 0.0))
    np.testing.assert_allclose(pose[:3, 1], options.up)


def test_two_dimensional_mpm_domain_gets_a_host_only_render_thickness():
    sims = SimpleNamespace(
        domain=np.asarray([4.0, 2.0]),
        look_from=(0.0, 0.0, 0.0),
        look_at=(0.0, 1.0, 0.0),
        camera_up=(0.0, 1.0, 0.0),
        view_angle=45.0,
        move_velocity=0.0,
    )

    lower, upper = _domain_bounds(sims)
    camera = _camera(sims)

    np.testing.assert_allclose(lower[:2], [0.0, 0.0])
    np.testing.assert_allclose(upper[:2], sims.domain)
    assert lower[2] < 0.0 < upper[2]
    np.testing.assert_allclose(camera.look_at, [2.0, 1.0, 0.0])
    assert camera.position[2] > upper[2]
    np.testing.assert_allclose(camera.up, [0.0, 1.0, 0.0])


def test_particle_poses_reject_nonfinite_geometry():
    with pytest.raises(FloatingPointError):
        _particle_poses(
            np.asarray([[np.nan, 0.0, 0.0]]),
            np.asarray([0.1]),
        )


def test_simulation_shortcuts_do_not_replace_render_shortcuts():
    keys = RealtimeViewer(callback_model())._registered_keys()

    assert set(keys) == {" ", ".", "b", "[", "]"}
    assert not set(keys) & set("acfhilmnoqrswz")


@pytest.mark.parametrize(
    ("button", "modifiers", "expected_state"),
    [
        (pyglet.window.mouse.LEFT, 0, Trackball.STATE_ROTATE),
        (pyglet.window.mouse.LEFT, pyglet.window.key.MOD_ALT, Trackball.STATE_PAN),
        (pyglet.window.mouse.LEFT, pyglet.window.key.MOD_SHIFT, Trackball.STATE_PAN),
        (pyglet.window.mouse.LEFT, pyglet.window.key.MOD_CTRL, Trackball.STATE_ZOOM),
        (pyglet.window.mouse.MIDDLE, 0, Trackball.STATE_PAN),
        (pyglet.window.mouse.RIGHT, 0, Trackball.STATE_ZOOM),
    ],
)
def test_mouse_buttons_match_trackball_controls(
    button, modifiers, expected_state
):
    class TrackballSpy:
        def __init__(self):
            self.state = None
            self.point = None

        def set_state(self, state):
            self.state = state

        def down(self, point):
            self.point = point

    viewer = type(
        "ViewerSpy",
        (),
        {
            "_trackball": TrackballSpy(),
            "viewer_flags": {"mouse_pressed": False},
        },
    )()

    PyRenderViewer.on_mouse_press(viewer, 12, 34, button, modifiers)

    assert viewer._trackball.state == expected_state
    np.testing.assert_array_equal(viewer._trackball.point, (12, 34))
    assert viewer.viewer_flags["mouse_pressed"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolution": (0, 720)},
        {"steps_per_frame": 0},
        {"steps_per_frame": 65, "maximum_steps_per_frame": 64},
        {"particle_scale": 0.0},
    ],
)
def test_viewer_options_reject_invalid_runtime_controls(kwargs):
    with pytest.raises(ValueError):
        RealtimeViewerOptions(**kwargs)
