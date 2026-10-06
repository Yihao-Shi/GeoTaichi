import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from geotaichi import IGA, init
from src.iga import DirichletBoundary, Primitives, Rectangle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--default-fp", default="float64")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output-interval", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--residual", type=float, default=1.0e-4)
    parser.add_argument(
        "--output-dir",
        default=str(CASE_DIR / "OutputData" / "elastic_beam2d"),
    )
    arguments = parser.parse_args()
    if arguments.steps <= 0 or arguments.output_interval <= 0:
        raise ValueError("--steps and --output-interval must be positive")
    if arguments.dt <= 0.0 or arguments.residual <= 0.0:
        raise ValueError("--dt and --residual must be positive")

    init(
        dim=2,
        arch=arguments.arch,
        default_fp=arguments.default_fp,
        offline_cache=False,
        log=True,
    )

    iga = IGA(log=True)
    iga.set_configuration(dimension=2, solver_type="Implicit")

    degree_u, degree_v = 2, 2
    beam = Rectangle()
    beam.set_parameters(size=[1.0, 0.2])
    # The original 3 x 3 lattice produced only one sampled VTK quad.  This
    # denser lattice preserves the beam geometry while making bending and the
    # stress gradient legible in a Gallery animation.
    beam.generate_knot_u(degree=degree_u, num_ctrlpts=17)
    beam.generate_knot_v(degree=degree_v, num_ctrlpts=7)
    beam.generate_ctrlpts()
    beam.generate_weights()
    beam.activate_boundary()
    beam.gather_boundary_ctrlpts()

    primitives = Primitives()
    primitives.append(beam, "beam")
    primitives.finialize()

    fixed = np.where(np.isclose(beam.control_points[:, 0], 0.0))[0]
    dirichlet = DirichletBoundary()
    dirichlet.append([list(2 * fixed), list(2 * fixed + 1)], [0.0] * (2 * len(fixed)))

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=[degree_u, degree_v])
    iga.add_material(young_modulus=1.0e5, poisson_ratio=0.3, density=1000.0)
    iga.set_solver(
        dt=arguments.dt,
        step=arguments.steps,
        interval=arguments.output_interval,
        residual=arguments.residual,
        gravity=[0.0, -9.8],
        path=arguments.output_dir,
    )

    iga.run(verbose=False)


if __name__ == "__main__":
    main()
