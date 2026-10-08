"""Time fixed-slot IGA solid assembly against the raw-matrix numerical oracle.

Run in separate source snapshots to compare kernel changes. Samples synchronize
Taichi and exclude compilation. This measures assembly, not a coupled timestep.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--elements", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.elements < 1 or args.repeats < 1:
        parser.error("elements and repeats must be positive")
    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64, offline_cache=False)
    import src.iga.config as config

    config.set_dimension(3)
    from src.iga import Cube, ImplicitIGA, Primitives
    from src.linear_solver.BuildTriplet import BuildTriplet

    body = Cube()
    body.set_parameters(start_point=[0.0, 0.0, 0.0], size=[1.0, 0.5, 0.2])
    body.generate_knot_u(degree=3, num_ctrlpts=args.elements + 3)
    body.generate_knot_v(degree=1, num_ctrlpts=2)
    body.generate_knot_w(degree=1, num_ctrlpts=2)
    body.generate_ctrlpts()
    body.generate_weights()
    primitives = Primitives()
    primitives.append(body, "solid")
    primitives.finialize()
    engine = ImplicitIGA(
        primitives=primitives,
        degree=[3, 1, 1],
        young_modulus=1e4,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, -9.8],
        dt=1e-3,
        step=0,
        assemble_type="Hash",
        path=str(args.output.parent),
    )
    engine.precompute()
    engine.grid_disp.from_numpy(np.linspace(-0.001, 0.001, engine.degree_of_freedom))
    coordinates, slots = engine.fixed_block_coordinates(upper_triangle=True)
    scatter = ti.field(ti.i32, shape=slots.shape)
    scatter.from_numpy(slots)
    fixed = BuildTriplet(
        dim=3,
        max_pairs_num=1,
        max_nonzeros=len(coordinates),
        max_active_nodes=engine.degree_of_freedom // 3,
        symmetric=False,
        matrix_symmetric=True,
    )
    fixed.install_fixed_pattern(coordinates)

    def assemble():
        fixed.reset_system()
        engine.assemble_body_matrix(project_spd=False, need_force=False, matrix=fixed, fixed_slots=scatter)
        fixed.finalize_taichi_assembly()
        ti.sync()

    assemble()
    assemble()
    engine.reset_linear_system()
    engine.assemble_body_matrix(project_spd=False, need_force=False)
    engine.hash_matrix.finalize_taichi_assembly()
    expected = engine.hash_matrix.to_scipy().tocsr()
    error = fixed.to_scipy().tocsr() - expected
    max_error = float(np.max(np.abs(error.data), initial=0.0))
    assert max_error <= 1e-9 + 2e-12 * np.max(np.abs(expected.data)), max_error
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        assemble()
        samples.append(time.perf_counter() - started)
    report = dict(
        arch=args.arch,
        precision="float64",
        degree=[3, 1, 1],
        elements=args.elements,
        dofs=engine.degree_of_freedom,
        fixed_seconds=float(np.median(samples)),
        samples_seconds=samples,
        matrix_max_error=max_error,
        compilation_seconds=ti.lang.impl.get_runtime().prog.get_total_compilation_time(),
        scope=__doc__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
