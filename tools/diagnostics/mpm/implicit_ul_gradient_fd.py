"""Finite-difference diagnostic for the implicit UL-MPM residual.

The legacy script initialized a GPU runtime and a random simulation at import
time.  This migrated form is import-safe, deterministic, backend-selectable,
and treats the finite-difference comparison as a real pass/fail criterion.
"""

from __future__ import annotations

import argparse

import numpy as np


def run_gradient_check(
    *,
    arch: str = "cpu",
    seed: int = 0,
    step: float = 1.0e-5,
    tolerance: float = 1.0e-4,
) -> float:
    import taichi as ti

    import src.mpm.config as config
    from src.mpm.engines.direct import ImplicitULMPM
    from src.mpm.generator.Body import Body

    ti.reset()
    ti.init(
        arch=getattr(ti, arch),
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        kernel_profiler=False,
        offline_cache=False,
    )
    config.set_dimension(2)

    body = Body()
    body.add_rectangle2d(
        [1.0, 1.0],
        [2.0, 2.0],
        0.1,
        2,
        init_v=[2.0, 1.0],
    )
    mpm = ImplicitULMPM(
        domain=[5.0, 5.0],
        dx=0.1,
        dt=1.0e-2,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e10,
        poisson_ratio=0.3,
        density=1000.0,
        residual=1.0e-4,
        gravity=[0.0, -1000.0],
        interval=10,
        dirichlet=None,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=None,
    )

    @ti.kernel
    def initialize_deformation_gradient():
        for particle in ti.grouped(mpm.particle.F0):
            for row, column in ti.static(ti.ndrange(2, 2)):
                mpm.particle[particle].F0[row, column] = (
                    200.0 if row == column else -10.0
                )

    @ti.kernel
    def copy_field(destination: ti.template(), source: ti.template()):
        for index in destination:
            destination[index] = source[index]

    initialize_deformation_gradient()
    mpm.initial_simulation()
    mpm.grid_reset()
    mpm.compute_shapefn()
    mpm.mass_vel_acc_p2g()
    mpm.find_active_node()
    mpm.prefix_sum_executor.run(mpm.node2dof)
    mpm.active_dof = mpm.set_active_dof()
    mpm.compute_nodal_vel_acc()
    mpm.compute_mass_list(mpm.integration)

    rng = np.random.default_rng(seed)
    displacement = rng.uniform(
        -0.5, 0.5, size=mpm.grid_disp.shape[0]
    ).astype(np.float64)
    mpm.grid_disp.from_numpy(displacement)

    mpm.assemble_inertia_force(
        mpm.active_dof,
        mpm.damping,
        mpm.gravity,
        mpm.integration,
        mpm.grid_disp,
    )
    mpm.assemble_material_force(mpm.active_dof, mpm.grid_disp)

    finite_difference = np.zeros(mpm.active_dof, dtype=np.float64)
    copy_field(mpm.grid_disp_temp, mpm.grid_disp)
    for index in range(mpm.active_dof):
        mpm.grid_disp_temp[index] += step
        energy_plus = float(mpm.total_energy(mpm.grid_disp_temp))
        mpm.grid_disp_temp[index] -= 2.0 * step
        energy_minus = float(mpm.total_energy(mpm.grid_disp_temp))
        mpm.grid_disp_temp[index] += step
        finite_difference[index] = -(
            energy_plus - energy_minus
        ) / (2.0 * step)

    assembled = mpm.rhs.to_numpy()[: mpm.active_dof]
    scale = np.maximum(np.abs(finite_difference), 1.0e-10)
    max_error = float(
        np.max(np.abs(finite_difference - assembled) / scale)
    )
    ti.reset()
    if not np.isfinite(max_error) or max_error > tolerance:
        raise AssertionError(
            "UL-MPM residual/energy-gradient mismatch: "
            f"max relative error={max_error:.6e}, tolerance={tolerance:.6e}"
        )
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=float, default=1.0e-5)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    arguments = parser.parse_args()
    error = run_gradient_check(
        arch=arguments.arch,
        seed=arguments.seed,
        step=arguments.step,
        tolerance=arguments.tolerance,
    )
    print(f"UL-MPM maximum relative gradient error: {error:.6e}")


if __name__ == "__main__":
    main()
