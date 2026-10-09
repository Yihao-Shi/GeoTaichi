"""Time contact screening on the left face of a saved wavy-plate checkpoint.

Run from the repository root with a caller-selected JSON output path. Timings
exclude compilation and synchronize Taichi. This measures contact kernels,
not complete solver steps; checkpoint velocities provide CCD test directions.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import src.igampm.config as config
from src.igampm.contact.ContactSurface import CouplingContactSurface
from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin
from src.physics_model.contact_model.ipc.NurbsContact import get_distance_to_surface_fixed_dim


@ti.data_oriented
class ContactBenchmark(ImplicitEngineMixin):
    def __init__(self, controls, velocities, counts, points, directions, threshold, clearance):
        knots_u = np.r_[np.zeros(3), np.linspace(0, 1, counts[0] - 2), np.ones(3)]
        knots_v = np.r_[0.0, np.linspace(0, 1, counts[1]), 1.0]
        primitive = SimpleNamespace(
            num_ctrlpts_u=counts[0],
            num_ctrlpts_v=counts[1],
            control_points=controls,
            weights=np.ones(len(controls)),
            gather_boundary_ctrlpts=lambda: ([len(controls)], np.arange(len(controls)), [(knots_u, knots_v)], [(3, 1)]),
        )
        owner = SimpleNamespace(
            patch=SimpleNamespace(
                primitive=SimpleNamespace(body={"face": {"primitive": primitive}}), prefix_total_num_ctrlpts=[0]
            )
        )
        self.contact_surface = CouplingContactSurface(owner)
        self.contact_surface.update_surface_bounds()
        self.contact_surface.update_span_bounds()
        control_direction = ti.field(ti.f64, shape=velocities.size)
        control_direction.from_numpy(velocities.ravel())
        self.contact_surface.update_control_point_direction(control_direction)
        self.points = ti.Vector.field(3, ti.f64, shape=len(points))
        self.directions = ti.Vector.field(3, ti.f64, shape=len(points))
        self.points.from_numpy(points)
        self.directions.from_numpy(directions)
        self.distances = ti.field(ti.f64, shape=(2, len(points)))
        self.unsafe = ti.field(ti.i32, shape=(2, len(points)))
        self.threshold = float(threshold)
        self.clearance = float(clearance)
        self.knot_counts = len(knots_u), len(knots_v)

    @ti.kernel
    def query(self, prune: ti.template()):
        for i in self.points:
            point = self.points[i]
            distance = self.contact_surface.distance_lower_bound(0, point)
            if ti.static(prune):
                if distance <= self.threshold:
                    distance = ti.max(
                        distance, self.contact_surface.span_distance_lower_bound(0, point, self.threshold)
                    )
            if distance <= self.threshold:
                _, _, distance, _ = get_distance_to_surface_fixed_dim(
                    0,
                    0,
                    0,
                    self.knot_counts[0],
                    self.knot_counts[1],
                    self.contact_surface.knot_vector_u,
                    self.contact_surface.knot_vector_v,
                    self.contact_surface.control_points_hat,
                    self.contact_surface.weights,
                    point,
                    self.contact_surface.basis[0],
                    self.contact_surface,
                    0,
                    ti.Vector([0.5, 0.5]),
                )
            self.distances[int(prune), i] = distance

    @ti.kernel
    def motion_screen(self, prune: ti.template()):
        for i in self.points:
            limit = 0.9 * (self.distances[0, i] - self.clearance)
            bound = 0.0
            if ti.static(prune):
                bound = self.contact_surface.relative_motion_upper_bound(0, self.directions[i])
                if bound > limit:
                    bound = self._point_nurbs_motion_bound(self.directions[i], 0, self.contact_surface.total_ctrlpts)
            else:
                bound = self._point_nurbs_motion_bound(self.directions[i], 0, self.contact_surface.total_ctrlpts)
            self.unsafe[int(prune), i] = bound > limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.particles <= 0 or args.repeats <= 0:
        parser.error("particles and repeats must be positive")
    parameters = json.loads((args.checkpoint.parent / "parameters.json").read_text())
    if parameters.get("dimension") != 3 or parameters.get("method") != "igampm":
        parser.error("expected a 3D IGA-MPM wavy-plate checkpoint")
    with np.load(args.checkpoint) as data:
        shape = (parameters["height_segments"] + 1, -1, 2, 3)
        controls = data["elastic_position"].reshape(shape)[:, :, 0]
        counts = controls.shape[1], controls.shape[0]
        controls = controls.reshape(-1, 3)
        velocities = data["elastic_velocity"].reshape(shape)[:, :, 0].reshape(-1, 3) * parameters["dt"]
        points, directions = data["particle_position"], data["particle_velocity"] * parameters["dt"]
    slope = 2 * np.pi * parameters["cycles"] * parameters["amplitude"] / parameters["width"]
    threshold = parameters["dmin"] + 0.25 * parameters["spacing"] / parameters["ppc"] / np.sqrt(1 + slope**2)
    offsets = np.maximum(np.maximum(controls.min(axis=0) - points, points - controls.max(axis=0)), 0)
    eligible = np.flatnonzero(np.linalg.norm(offsets, axis=1) <= threshold)
    chosen = eligible[np.linspace(0, len(eligible) - 1, min(args.particles, len(eligible)), dtype=int)]
    if not len(chosen):
        parser.error("checkpoint has no candidates inside the whole-face bound")
    config.set_dimension(3)
    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    if ti.lang.impl.current_cfg().arch != getattr(ti, args.arch):
        raise RuntimeError("requested backend is unavailable")
    benchmark = ContactBenchmark(
        controls, velocities, counts, points[chosen], directions[chosen], threshold, parameters["dmin"]
    )
    report = dict(
        checkpoint=str(args.checkpoint),
        arch=args.arch,
        precision="float64",
        particles=len(chosen),
        eligible_pairs=len(eligible),
        controls=len(controls),
        repeats=args.repeats,
        warmup=2,
        scope="single-face contact kernels; CCD directions are saved velocities times dt",
    )
    for name in ("query", "motion_screen"):
        kernel = getattr(benchmark, name)
        for _ in range(2):
            for prune in (False, True):
                kernel(prune)
        ti.sync()
        durations = [[], []]
        for repeat in range(args.repeats):
            for prune in (False, True) if repeat % 2 == 0 else (True, False):
                start = time.perf_counter()
                kernel(prune)
                ti.sync()
                durations[int(prune)].append(time.perf_counter() - start)
        medians = [float(np.median(values)) for values in durations]
        report[name] = dict(baseline_seconds=medians[0], optimized_seconds=medians[1], speedup=medians[0] / medians[1])
    reference, accelerated = benchmark.distances.to_numpy()
    active = reference < threshold
    np.testing.assert_array_equal(accelerated < threshold, active)
    np.testing.assert_allclose(accelerated[active], reference[active], rtol=1e-10, atol=1e-12)
    assert np.all(accelerated <= reference + 1e-10)
    np.testing.assert_array_equal(*benchmark.unsafe.to_numpy())
    report.update(
        active_pairs=int(active.sum()), equivalent_active_distances=True, equivalent_motion_classification=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
