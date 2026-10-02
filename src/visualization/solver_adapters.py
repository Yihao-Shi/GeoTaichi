"""Entry points that attach the real-time viewer to existing solvers."""

from __future__ import annotations

import time

import numpy as np
import taichi as ti

from src.utils.constants import Threshold

from .adapters import IGARenderSource, dem_render_sources, mpm_particle_source
from .realtime import (
    CallbackSceneModel,
    CameraOptions,
    RealtimeViewer,
    RealtimeViewerOptions,
)


def _resolution(value):
    if isinstance(value, (int, float)):
        size = int(value)
        return size, size
    return int(value[0]), int(value[1])


def _vector3(value, *, fill=0.0):
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 2:
        vector = np.append(vector, float(fill))
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError("viewer vectors must contain two or three finite values")
    return vector


def _domain_bounds(sims):
    domain = np.asarray(sims.domain, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(domain)) or np.any(domain <= 0.0):
        raise ValueError("simulation domain values must be finite and positive")
    if domain.size == 2:
        depth = max(float(np.max(np.abs(domain))) * 0.02, 1.0e-3)
        return (
            np.asarray([0.0, 0.0, -0.5 * depth]),
            np.asarray([domain[0], domain[1], 0.5 * depth]),
        )
    if domain.size == 3:
        return np.zeros(3), domain
    raise ValueError("simulation domain must contain two or three values")


def _camera(sims):
    position = _vector3(sims.look_from)
    look_at = _vector3(sims.look_at)
    up = _vector3(sims.camera_up)
    lower, upper = _domain_bounds(sims)
    view = look_at - position
    invalid_view = np.linalg.norm(view) <= np.finfo(float).eps
    invalid_up = (
        not invalid_view
        and np.linalg.norm(np.cross(view, up)) <= np.finfo(float).eps
    )
    if invalid_view or invalid_up:
        center = 0.5 * (lower + upper)
        span = max(float(np.max(upper - lower)), 1.0e-3)
        if np.asarray(sims.domain).size == 2:
            look_at = center
            position = center + np.asarray([0.0, 0.0, 1.5 * span])
            up = np.asarray([0.0, 1.0, 0.0])
        else:
            look_at = center
            position = center + np.asarray([1.2, -1.2, 0.8]) * span
            up = np.asarray([0.0, 0.0, 1.0])
    return CameraOptions(
        position=tuple(position),
        look_at=tuple(look_at),
        up=tuple(up),
        fov=float(sims.view_angle),
        movement_speed=max(float(sims.move_velocity), 1.0e-8),
    )


def _options(sims, title, *, steps_per_frame=1):
    return RealtimeViewerOptions(
        title=title,
        resolution=_resolution(sims.window_size),
        steps_per_frame=steps_per_frame,
        maximum_steps_per_frame=max(64, steps_per_frame),
        background_color=tuple(float(value) for value in sims.background_color),
        point_light_position=tuple(_vector3(sims.point_light)),
        camera=_camera(sims),
    )


def run_dem_gui(solver, scene):
    """Run the existing DEM solver one normal core step per viewer step."""

    sims = solver.sims
    if sims.scheme in (
        "LSDEM",
        "LSMPM",
        "PolySuperEllipsoid",
        "PolySuperQuadrics",
    ):
        # Real-time rendering must not depend on whether surface file output
        # was selected.  This is the existing visualization-position field,
        # allocated once at the scene's configured surface capacity.
        scene.activate_surface_node_visualization(sims, force=True)
    ti.sync()
    solver.engine.pre_calculation(sims, scene, solver.contact.neighbor)
    if sims.current_time < Threshold:
        solver.save_file(scene)
        solver.last_save_time = float(sims.current_time)
    solver.compile(scene)

    def advance():
        if sims.current_time > sims.time:
            return
        solver.core(scene)
        new_body = solver.generator.regenerate(scene)
        if (
            sims.current_time - solver.last_save_time - 0.1 * sims.delta
            > sims.save_interval
            or new_body
        ):
            solver.save_file(scene)
            if new_body:
                solver.engine.update_verlet_table(
                    sims, scene, solver.contact.neighbor
                )
                sims.set_max_bounding_sphere_radius(
                    scene.find_bounding_sphere_max_radius(sims)
                )
        sims.current_time += sims.delta
        sims.current_step += 1

    lower, upper = _domain_bounds(sims)
    model = CallbackSceneModel(
        name=f"DEM | {sims.scheme}",
        render_sources=dem_render_sources(sims, scene),
        time_step=max(float(sims.delta), np.finfo(float).eps),
        step_callback=advance,
        domain_min=tuple(lower),
        domain_max=tuple(upper),
        statistics_callback=lambda: {
            "particles": int(scene.particleNum[0]),
            "time": f"{sims.current_time:.6g}",
        },
    )
    start = time.perf_counter()
    RealtimeViewer(model, _options(sims, "GeoTaichi | DEM")).run()
    solver.physical_seconds = time.perf_counter() - start
    if (
        abs(sims.current_time - solver.last_save_time)
        > 0.9 * sims.save_interval
    ):
        solver.save_file(scene)


def run_mpm_gui(solver, scene, neighbor):
    """Run the existing MPM solver and visualize its material points."""

    sims = solver.sims
    ti.sync()
    solver.engine.pre_calculation(sims, scene, neighbor)
    if sims.current_time < Threshold:
        solver.save_file(scene)
        solver.last_save_time = float(sims.current_time)
    solver.compile(scene, neighbor)

    def advance():
        if sims.current_time > sims.time:
            return
        solver.core(scene, neighbor)
        new_body = solver.generator.regenerate(scene)
        if (
            sims.current_time - solver.last_save_time - 0.1 * sims.delta
            > sims.save_interval
            or new_body
        ):
            solver.save_file(scene)
        sims.current_time += sims.delta
        sims.current_step += 1

    lower, upper = _domain_bounds(sims)
    model = CallbackSceneModel(
        name="MPM material points",
        render_sources=(mpm_particle_source(scene, color=sims.particle_color),),
        time_step=max(float(sims.delta), np.finfo(float).eps),
        step_callback=advance,
        domain_min=tuple(lower),
        domain_max=tuple(upper),
        statistics_callback=lambda: {
            "particles": int(scene.particleNum[0]),
            "time": f"{sims.current_time:.6g}",
        },
    )
    RealtimeViewer(model, _options(sims, "GeoTaichi | MPM")).run()
    if (
        abs(sims.current_time - solver.last_save_time)
        > 0.9 * sims.save_interval
    ):
        solver.save_file(scene)


def run_iga_gui(engine, *, resolution=16, verbose=True, postprocessing=()):
    """Run an existing IGA engine through its normal substep method."""

    engine.initial_simulation()
    for callback in postprocessing:
        callback()
    completed_steps = 0

    def advance():
        nonlocal completed_steps
        if completed_steps >= engine.total_step:
            return
        for _ in range(engine.output_interval):
            engine.substep(verbose)
        engine.record()
        for callback in postprocessing:
            callback()
        completed_steps += 1

    initial = np.asarray(engine.patch.control_points.to_numpy(), dtype=np.float64)
    if initial.shape[1] == 2:
        initial = np.column_stack((initial, np.zeros(initial.shape[0])))
    lower = initial.min(axis=0)
    upper = initial.max(axis=0)
    padding = max(np.linalg.norm(upper - lower) * 0.05, 1.0e-3)
    lower -= padding
    upper += padding
    center = 0.5 * (lower + upper)
    camera = CameraOptions(
        position=tuple(center + np.asarray([1.4, -1.4, 1.0]) * np.max(upper - lower)),
        look_at=tuple(center),
        up=(0.0, 0.0, 1.0),
    )
    model = CallbackSceneModel(
        name="Isogeometric analysis",
        render_sources=(IGARenderSource(engine.patch, resolution=resolution),),
        time_step=max(
            float(engine.dt) * int(engine.output_interval),
            np.finfo(float).eps,
        ),
        step_callback=advance,
        domain_min=tuple(lower),
        domain_max=tuple(upper),
        statistics_callback=lambda: {
            "output step": completed_steps,
            "control points": initial.shape[0],
        },
    )
    options = RealtimeViewerOptions(
        title="GeoTaichi | IGA",
        camera=camera,
    )
    RealtimeViewer(model, options).run()
